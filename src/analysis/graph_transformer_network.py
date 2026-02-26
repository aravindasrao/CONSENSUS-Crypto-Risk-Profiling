# src/analysis/graph_transformer_network.py
"""
Graph Transformer Network for detecting complex relational patterns.
This is the core of the GNN Clustering and Advanced Analytics.
Requires: pip install torch torch-geometric
"""
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple, Union
import networkx as nx
from datetime import datetime, timedelta
import math
from collections import defaultdict
import platform
import warnings
import sys
import json
from pathlib import Path
from tqdm import tqdm
from scipy.stats import entropy

# Try to import torch_geometric, but handle if not available
try:
    from torch_geometric.nn import TransformerConv
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    warnings.warn("torch_geometric not available, some features will be disabled")
    # Create dummy classes to prevent errors
    class TransformerConv(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.linear = nn.Linear(kwargs.get('in_channels', 128), kwargs.get('out_channels', 128))
        def forward(self, x, edge_index, edge_attr=None):
            # Fallback to simple linear layer
            return self.linear(x)
    class Data:
        def __init__(self, x=None, edge_index=None, edge_attr=None):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.num_nodes = x.shape[0] if x is not None else 0
            self.num_edges = edge_index.shape[1] if edge_index is not None else 0


from src.core.database import DatabaseEngine
from config.config import config
from src.utils.transaction_logger import log_network_anomaly, log_suspicious_address

logger = logging.getLogger(__name__)

# Define a simple GCN model architecture
class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = TransformerConv(num_node_features, 16)
        self.conv2 = TransformerConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphTransformerNetwork(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GraphTransformerNetwork, self).__init__()
        # The number of heads is hardcoded to 4, a common choice for transformers.
        # The hidden_channels is the dimension per head.
        self.conv1 = TransformerConv(num_node_features, hidden_channels, heads=4, dropout=0.1)
        # The input to the second layer is the concatenated output of all heads.
        self.conv2 = TransformerConv(hidden_channels * 4, num_classes, heads=1, concat=False, dropout=0.1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class TornadoCashGraphAnalyzer:
    """
    Analyzes transaction graphs using a Graph Transformer Network to detect complex patterns.
    """
    def __init__(self, database: DatabaseEngine):
        self.database = database
        self.model = None
        self.device = self._get_safe_device()
        self.node_to_idx = {}
        logger.info("Tornado Cash Graph Analyzer initialized.")

    def _get_safe_device(self) -> str:
        """Get safe device for PyTorch operations."""
        if platform.system() == 'Darwin' and platform.machine() == 'arm64':
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def _build_full_graph(self, features_df: pd.DataFrame, test_mode: bool = False) -> Optional[Data]:
        """
        Builds a PyTorch Geometric Data object from transactions and features
        using a scalable, batched approach for edge fetching.
        """
        logger.info("Building graph with scalable batched edge fetching...")

        # 1. Define nodes and features from the input DataFrame
        if 'address' not in features_df.columns:
            logger.error("`address` column not found in features_df.")
            return None
        
        # Set index to address for easy lookup
        features_df = features_df.set_index('address')

        # Create a mapping from address string to integer index
        idx_to_addr = features_df.index.tolist()
        self.node_to_idx = {addr: i for i, addr in enumerate(idx_to_addr)}
        
        # Prepare node features (must align with node_to_idx)
        # FIX: Select only numeric columns to avoid the object dtype error
        numeric_features_df = features_df.select_dtypes(include=np.number)
        # --- FIX: Force conversion to numeric and fill NaNs to prevent TypeError ---
        sanitized_df = numeric_features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        numpy_array = sanitized_df.astype(np.float32).values
        node_features = torch.tensor(numpy_array, dtype=torch.float32).to(self.device)

        # 2. Fetch edges in batches to handle large datasets
        logger.info("Fetching edges in batches...")
        source_nodes_list = []
        target_nodes_list = []
        batch_size = 500000  # Process 500k transactions at a time
        
        total_txs_query = "SELECT COUNT(*) as count FROM transactions WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL"
        total_txs = self.database.fetch_one(total_txs_query)['count']
        
        # In test mode, we might want to limit the number of transactions scanned
        limit = 200000 if test_mode else total_txs

        for offset in tqdm(range(0, limit, batch_size), desc="Fetching Edges for GraphTransformer"):
            edges_df_batch = self.database.fetch_df(f"""
                SELECT from_addr, to_addr FROM transactions
                WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL
                LIMIT {batch_size} OFFSET {offset}
            """)

            if edges_df_batch.empty:
                break

            # Filter edges where both source and target are in our node set
            valid_edges = edges_df_batch[
                edges_df_batch['from_addr'].isin(self.node_to_idx) & 
                edges_df_batch['to_addr'].isin(self.node_to_idx)
            ]

            if not valid_edges.empty:
                source_nodes_list.extend([self.node_to_idx[addr] for addr in valid_edges['from_addr']])
                target_nodes_list.extend([self.node_to_idx[addr] for addr in valid_edges['to_addr']])

        if not source_nodes_list:
            logger.warning("No valid edges found for the graph after scanning transactions.")
            # We can still return a graph with nodes but no edges
            return Data(x=node_features, edge_index=torch.empty((2, 0), dtype=torch.long).to(self.device))

        # 3. Create the final edge_index tensor
        edge_index = torch.tensor([source_nodes_list, target_nodes_list], dtype=torch.long).to(self.device)
        
        graph_data = Data(x=node_features, edge_index=edge_index)
        logger.info(f"Graph built successfully: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges.")
        
        return graph_data
        
    def analyze_and_cluster(self, features_df: pd.DataFrame, test_mode: bool = False) -> Dict[str, Any]:
        """
        Main method to run GNN analysis and detect patterns.
        
        Args:
            features_df: DataFrame with 111 features from the Foundation Layer.
            test_mode: Flag to limit data for testing.
            
        Returns:
            Dictionary with GNN analysis results.
        """
        logger.info("Starting GNN-based clustering and pattern detection...")
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            logger.error("torch_geometric is not installed. GNN analysis is disabled.")
            return {'status': 'disabled', 'clusters': [], 'error': 'torch_geometric not found'}

        # 1. Build the graph from transactions and features
        graph_data = self._build_full_graph(features_df, test_mode=test_mode)
        if graph_data is None or graph_data.num_nodes == 0:
            logger.warning("Could not build a graph for GNN analysis.")
            return {'status': 'no_graph', 'clusters': []}

        # 2. Initialize and "train" the model
        # Get model dimensions from the centralized config
        embedding_dim = config.analysis['gnn_models']['graph_transformer']['embedding_dim']
        output_dim = config.analysis['gnn_models']['graph_transformer']['output_dim']
        num_node_features = graph_data.x.shape[1]
        self.model = GraphTransformerNetwork(
            num_node_features=num_node_features,
            hidden_channels=embedding_dim,
            num_classes=output_dim).to(self.device)
        self.model.eval() # Put model in evaluation mode
        
        # 3. Get node embeddings and predict clusters
        with torch.no_grad():
            log_logits = self.model(graph_data)
            predicted_clusters = log_logits.argmax(dim=1).cpu().numpy()

        # 4. Map results back to addresses
        clusters = []
        for i, address in enumerate(self.node_to_idx.keys()):
            clusters.append({'address': address, 'cluster_id': int(predicted_clusters[i])})
        
        logger.info(f"GNN clustering complete. Found {len(set(predicted_clusters))} patterns.")

        # 5. Process suspicion scores
        gnn_results = self._process_suspicion_scores(outputs={'node_embeddings': log_logits}, graph_data=graph_data)
        gnn_results['clusters'] = clusters
        
        self._store_gnn_results(gnn_results)
        
        return gnn_results
    
    def _process_suspicion_scores(self, outputs: Dict[str, Any], graph_data: Data) -> Dict[str, Any]:
        """
        Analyzes the GNN outputs to identify and log suspicious activity.
        
        Args:
            outputs: The raw output from the GNN model (e.g., node embeddings, attention weights).
            graph_data: The PyTorch Geometric Data object used for the analysis.
            
        Returns:
            A dictionary containing the analysis results.
        """
        # A simple method to derive a suspicion score from the model's output
        # For a full implementation, this would involve a dedicated head
        # Here, we'll use a simple heuristic based on entropy
        node_probabilities = F.softmax(outputs['node_embeddings'], dim=1).cpu().numpy()
        suspicion_scores = entropy(node_probabilities, axis=1) / math.log(node_probabilities.shape[1])
        
        # Identify the most suspicious nodes
        sorted_indices = np.argsort(suspicion_scores)[::-1]
        suspicious_nodes = []
        
        for idx in sorted_indices[:20]: # Top 20 most suspicious
            address = list(self.node_to_idx.keys())[idx]
            score = float(suspicion_scores[idx])
            
            # Log suspicious address with detailed metadata
            cluster_id = self.database.fetch_one("SELECT cluster_id FROM addresses WHERE address = ?", (address,))
            
            log_suspicious_address(
                address=address,
                reason="High GNN-based suspicion score",
                cluster_id=cluster_id['cluster_id'] if cluster_id else -1,
                metadata={'gnn_suspicion_score': score, 'top_gnn_rank': idx + 1}
            )
            
            suspicious_nodes.append({
                'address': address,
                'score': score,
                'cluster_id': cluster_id['cluster_id'] if cluster_id else -1
            })
            
        return {
            'status': 'completed',
            'suspicious_nodes': suspicious_nodes,
            'gnn_suspicion_scores': suspicion_scores.tolist()
        }

    def _store_gnn_results(self, gnn_results: Dict[str, Any]):
        """
        Stores the GNN analysis results in the database.
        
        Args:
            gnn_results: The dictionary containing GNN analysis results.
        """
        try:
            for node in gnn_results.get('suspicious_nodes', []):
                self.database.store_advanced_analysis_results(
                    address=node['address'],
                    analysis_type='gnn_pattern_detection',
                    results=node,
                    confidence_score=node['score'],
                    severity='HIGH' if node['score'] > 0.8 else 'MEDIUM'
                )

                # Store a standardized risk component for the UnifiedRiskScorer
                self.database.store_component_risk(
                    address=node['address'],
                    component_type='gnn_risk',
                    risk_score=node['score'],
                    confidence=0.9, # GNNs are highly confident
                    evidence={'reason': f"High suspicion score ({node['score']:.2f}) from Graph Transformer Network"},
                    source_analysis='graph_transformer_network'
                )
                
            logger.info(f"Stored GNN analysis results for {len(gnn_results.get('suspicious_nodes', []))} addresses.")

            # +++ NEW: Store cluster assignments for the consensus engine +++
            cluster_assignments = gnn_results.get('clusters', [])
            if not cluster_assignments:
                logger.warning("No GNN clusters found to store for consensus.")
                return

            records_to_insert = []
            for assignment in cluster_assignments:
                records_to_insert.append((
                    assignment['address'],
                    f"GNN_{assignment['cluster_id']}",  # Prefix to avoid collision
                    'graph_transformer_embedding',
                    0.7  # Default confidence for this advanced method
                ))

            with self.database.transaction():
                for record in records_to_insert:
                    # Use ON CONFLICT to make the operation idempotent.
                    self.database.execute("""
                        INSERT INTO cluster_assignments (address, cluster_id, cluster_type, confidence)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(address, cluster_type) DO UPDATE SET
                            cluster_id = excluded.cluster_id,
                            confidence = excluded.confidence
                    """, record)
            
            logger.info(f"Stored {len(records_to_insert)} GNN cluster assignments for consensus.")

        except Exception as e:
            logger.error(f"Failed to store GNN results: {e}")


def run_gnn_analysis(database: DatabaseEngine, test_mode: bool = False):
    """
    Function to orchestrate the GNN analysis, to be called by the pipeline.
    
    Args:
        database: The DatabaseEngine instance.
        test_mode: A flag for running in test mode.
        
    Returns:
        The analysis results from the GNN.
    """
    logger.info("Running GNN Analysis...")
    
    # 1. Fetch pre-calculated features from the Foundation Layer
    limit_clause = f"LIMIT {500 if test_mode else 20000}" # Limit for performance
    
    # Query the wide 'addresses' table directly. This is much more efficient.
    # We don't need to list all 111 columns; `SELECT *` is fine as the GNN analyzer
    # will handle the feature matrix creation from the DataFrame.
    features_df_wide = database.fetch_df(f"SELECT * FROM addresses {limit_clause}")
    
    if features_df_wide.empty:
        logger.error("No features found in the addresses table. Please run the Foundation Layer first.")
        return {'status': 'failed', 'error': 'No features for GNN'}

    # The data is already in the correct wide format.
    features_df_wide = features_df_wide.fillna(0)
    
    # 2. Instantiate and run the GNN analyzer
    gnn_analyzer = TornadoCashGraphAnalyzer(database)
    
    gnn_results = gnn_analyzer.analyze_and_cluster(features_df_wide, test_mode=test_mode)
    
    return gnn_results
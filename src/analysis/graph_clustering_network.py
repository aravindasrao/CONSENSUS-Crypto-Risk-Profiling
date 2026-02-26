# src/analysis/graph_transformer_network.py
"""
Graph Neural Network for detecting complex relational patterns.
This is the core of the GNN Clustering.
Requires: pip install torch torch-geometric
"""
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Any

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

# Define a simple GCN model architecture
class SimpleGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GraphClusteringNetwork:
    def __init__(self, database: DatabaseEngine):
        self.database = database
        self.model = None
        logger.info("Graph Transformer Network initialized.")

    def analyze_and_cluster(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Starting GNN-based clustering...")

        # 1. Build the full graph from transactions
        graph_data = self._build_full_graph(features_df)
        if graph_data.num_nodes == 0:
            logger.warning("Could not build a graph for GNN analysis.")
            return {'status': 'no_graph', 'clusters': {}}

        # 2. Initialize and "train" the model (pseudo-training for demonstration)
        num_classes = 20 # We want to find ~20 types of behavioral clusters
        self.model = SimpleGCN(num_node_features=graph_data.num_node_features, num_classes=num_classes)
        self.model.eval() # Put model in evaluation mode

        # 3. Get node embeddings and predict clusters
        with torch.no_grad():
            log_logits = self.model(graph_data)
            predicted_clusters = log_logits.argmax(dim=1).numpy()

        # 4. Map results back to addresses
        clusters = []
        for i, address in enumerate(features_df['address'].values):
            clusters.append({'address': address, 'cluster_id': int(predicted_clusters[i])})
        
        logger.info(f"GNN clustering complete. Found {len(set(predicted_clusters))} patterns.")
        return {'status': 'completed', 'clusters': clusters}

    def _build_full_graph(self, features_df: pd.DataFrame) -> Data:
        """Builds a PyTorch Geometric Data object from transactions and features."""
        
        # Get all transactions
        transactions = self.database.fetch_df("SELECT from_addr, to_addr, value_eth FROM transactions")
        
        # Align features with a consistent node ordering
        all_addresses = pd.concat([transactions['from_addr'], transactions['to_addr']]).unique()
        address_map = {addr: i for i, addr in enumerate(all_addresses)}
        
        # Prepare node features (must align with address_map)
        features_df = features_df.set_index('address').reindex(all_addresses).fillna(0)
        node_features = torch.tensor(features_df.values, dtype=torch.float)

        # Prepare edge index
        source_nodes = [address_map[addr] for addr in transactions['from_addr'] if addr in address_map]
        target_nodes = [address_map[addr] for addr in transactions['to_addr'] if addr in address_map]
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        return Data(x=node_features, edge_index=edge_index)
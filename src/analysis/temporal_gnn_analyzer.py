# src/analysis/temporal_gnn_analyzer.py
"""
Temporal Graph Convolutional Network (T-GCN) Analyzer.
Models the evolution of the transaction graph over time.
"""
import logging
import platform
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from collections import defaultdict
from tqdm import tqdm

try:
    import torch
    from torch_geometric.nn import GCNConv
    from torch_geometric.data import Data
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from src.core.database import DatabaseEngine
from config.config import config

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric availability at module level
if not PYG_AVAILABLE:
    logger.error("PyTorch or PyTorch Geometric is not installed. Temporal GNN analysis is disabled.")
    # Define dummy classes/functions to avoid import errors elsewhere
    class GCNConv: pass
    class Data: pass
    def run_temporal_gnn_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True) -> Dict[str, Any]:
        return {"status": "skipped", "reason": "PyTorch Geometric not installed"}
else:
    class T_GCN(torch.nn.Module):
        """Temporal Graph Convolutional Network."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.gcn = GCNConv(in_channels, out_channels)
            self.gru = torch.nn.GRU(out_channels, out_channels)
    
        def forward(self, x, edge_index, h):
            x = self.gcn(x, edge_index).relu()
            x, h = self.gru(x.unsqueeze(0), h)
            return x.squeeze(0), h
    
    class TemporalGNNAnalyzer:
        """Orchestrates Temporal GNN analysis."""
        def __init__(self, database: DatabaseEngine, device: torch.device):
            self.database = database
            self.device = device
            self.model = None
            self.node_to_idx = {}
            self.idx_to_node = []
    
        def analyze(self, features_df: pd.DataFrame, num_snapshots: int, test_mode: bool = False) -> Dict[str, Any]:
            """Runs the full analysis pipeline."""
            if not PYG_AVAILABLE:
                logger.error("torch_geometric is not installed. Temporal GNN analysis is disabled.")
                return {'status': 'disabled', 'error': 'torch_geometric not found'}
    
            logger.info("Creating graph snapshots for Temporal GNN...")
            snapshots = self._create_snapshots(features_df, num_snapshots, test_mode=test_mode)
            if not snapshots:
                return {'status': 'no_graph', 'error': 'Could not create temporal snapshots'}
    
            logger.info("Processing snapshots with T-GCN model...")
            final_embeddings = self._process_snapshots(snapshots)
    
            logger.info("Calculating risk scores from temporal embeddings...")
            risk_scores = self._calculate_risk_scores(final_embeddings)
    
            logger.info("Storing Temporal GNN results...")
            self._store_results(risk_scores)
            
            return {'status': 'completed', 'addresses_scored': len(risk_scores)}
    
        def _create_snapshots(self, features_df: pd.DataFrame, num_snapshots: int, test_mode: bool = False) -> List[Data]:
            """Creates a series of graph snapshots over time using a scalable, batched approach."""
            logger.info("Creating graph snapshots with scalable batched edge fetching...")

            # 1. Define nodes and features from the input DataFrame
            if 'address' not in features_df.columns:
                logger.error("`address` column not found in features_df.")
                return []
            
            features_df_aligned = features_df.set_index('address')
            self.idx_to_node = features_df_aligned.index.tolist()
            self.node_to_idx = {addr: i for i, addr in enumerate(self.idx_to_node)}
            
            # FIX: Ensure only numeric features are used for the tensor
            numeric_features_df = features_df_aligned.select_dtypes(include=np.number).drop(columns=['is_contract', 'is_tornado'], errors='ignore')
            # --- FIX: Force conversion to numeric and fill NaNs to prevent TypeError ---
            sanitized_df = numeric_features_df.apply(pd.to_numeric, errors='coerce').fillna(0)
            numpy_array = sanitized_df.astype(np.float32).values
            node_features = torch.tensor(numpy_array, dtype=torch.float32)

            # 2. Determine time range for snapshots
            time_range = self.database.fetch_one("SELECT MIN(timestamp) as min_ts, MAX(timestamp) as max_ts FROM transactions")
            if not time_range or time_range['min_ts'] is None:
                logger.warning("No transactions found to create snapshots.")
                return []
            
            min_ts, max_ts = time_range['min_ts'], time_range['max_ts']
            # Add a small epsilon to include the max_ts value in a bin
            bins = np.linspace(min_ts, max_ts + 1e-9, num_snapshots + 1)

            # 3. Fetch edges in batches and assign to snapshots
            snapshot_edges = defaultdict(list)
            batch_size = 500000
            
            total_txs_query = "SELECT COUNT(*) as count FROM transactions WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL"
            total_txs = self.database.fetch_one(total_txs_query)['count']
            limit = 200000 if test_mode else total_txs

            for offset in tqdm(range(0, limit, batch_size), desc="Fetching Edges for Snapshots"):
                edges_df_batch = self.database.fetch_df(f"""
                    SELECT from_addr, to_addr, timestamp FROM transactions
                    WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL
                    LIMIT {batch_size} OFFSET {offset}
                """)

                if edges_df_batch.empty:
                    break

                edges_df_batch['snapshot_id'] = pd.cut(edges_df_batch['timestamp'], bins=bins, labels=False, right=False)

                valid_edges = edges_df_batch[
                    edges_df_batch['from_addr'].isin(self.node_to_idx) & 
                    edges_df_batch['to_addr'].isin(self.node_to_idx)
                ]

                for snapshot_id, group in valid_edges.groupby('snapshot_id'):
                    src_indices = [self.node_to_idx[addr] for addr in group['from_addr']]
                    dst_indices = [self.node_to_idx[addr] for addr in group['to_addr']]
                    snapshot_edges[snapshot_id].extend(zip(src_indices, dst_indices))

            # 4. Build PyG Data objects for each snapshot
            snapshots = []
            for i in range(num_snapshots):
                edge_list = snapshot_edges.get(i, [])
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.empty((2, 0), dtype=torch.long)
                snapshots.append(Data(x=node_features, edge_index=edge_index).to(self.device))
            
            logger.info(f"Created {len(snapshots)} snapshots.")
            return snapshots
    
        def _process_snapshots(self, snapshots: List[Data]) -> np.ndarray:
            """Process the sequence of snapshots through the T-GCN."""
            if not snapshots: return np.array([])
            num_nodes = snapshots[0].num_nodes
            num_features = snapshots[0].num_node_features

            embedding_dim = config.analysis['gnn_models']['temporal_gnn']['embedding_dim']
            self.model = T_GCN(num_features, embedding_dim).to(self.device)
            self.model.eval() # Use in eval mode as we are not training here

            h = torch.zeros((1, num_nodes, embedding_dim), device=self.device) # Initial hidden state
            
            with torch.no_grad():
                for snapshot in snapshots:
                    x, h = self.model(snapshot.x, snapshot.edge_index, h)
            
            return x.cpu().numpy()
    
        def _calculate_risk_scores(self, final_embeddings: np.ndarray) -> Dict[str, float]:
            """Calculates anomaly scores based on final embeddings."""
            if final_embeddings.size == 0: return {}
            
            # Risk is proportional to the magnitude of the final embedding vector
            scores = np.linalg.norm(final_embeddings, axis=1)
            max_score = scores.max()
            normalized_scores = scores / max_score if max_score > 0 else np.zeros_like(scores)
    
            return {self.idx_to_node[i]: float(normalized_scores[i]) for i in range(len(normalized_scores))}
    
        def _store_results(self, risk_scores: Dict[str, float]):
            """Stores the results in the risk_components table."""
            for address, score in risk_scores.items():
                self.database.store_component_risk(
                    address=address,
                    component_type='temporal_gnn_risk',
                    risk_score=score,
                    confidence=0.7,
                    evidence={'method': 'Risk from final T-GCN hidden state embedding'},
                    source_analysis='temporal_gnn_analyzer'
                )
    
    def run_temporal_gnn_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True):
        """Main function to run Temporal GNN analysis."""
        # --- Enhanced GPU/CPU/MPS Device Setup ---
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("🚀 Using GPU (CUDA) for Temporal GNN analysis.")
            elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("🚀 Using GPU (MPS) on Apple Silicon for Temporal GNN analysis.")
            else:
                device = torch.device('cpu')
                logger.info("🐌 GPU not available. Using CPU for Temporal GNN analysis. This may be slow.")
        else:
            device = torch.device('cpu')
            logger.info("🐌 Using CPU for Temporal GNN analysis.")
    
        # Mitigate potential multiprocessing issues on CPU, especially on macOS
        if str(device) == 'cpu':
            torch.set_num_threads(1)
    
        logger.info("⏳ Running Temporal GNN Analysis...")
        num_snapshots = 10 if test_mode else 50
        
        # --- FIX: Query the wide 'addresses' table directly ---
        # This avoids the slow pivot operation and reads the data in the correct format.
        limit = 500 if test_mode else 20000
        limit_clause = f"LIMIT {limit}"

        features_df_wide = database.fetch_df(f"""
            SELECT *
            FROM addresses
            {limit_clause}
        """)
        if features_df_wide.empty:
            logger.error("No features found for Temporal GNN analysis.")
            return {'status': 'failed', 'error': 'No features available'}
    
        analyzer = TemporalGNNAnalyzer(database, device=device)
        return analyzer.analyze(features_df_wide, num_snapshots, test_mode=test_mode)
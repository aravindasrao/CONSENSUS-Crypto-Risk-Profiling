# src/analysis/hgn_analyzer.py
"""
Heterogeneous Graph Network (HGN) Analyzer using HGTConv.
Distinguishes between EOA and Contract nodes.
"""
import logging
import platform
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from tqdm import tqdm

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import HGTConv, Linear
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from src.core.database import DatabaseEngine
from config.config import config

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric availability at module level
if not PYG_AVAILABLE:
    logger.error("PyTorch or PyTorch Geometric is not installed. HGN analysis is disabled.")
    # Define dummy classes/functions to avoid import errors elsewhere
    class HGTConv: pass
    class Linear: pass
    class HeteroData: pass
    def run_hgn_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True) -> Dict[str, Any]:
        return {"status": "skipped", "reason": "PyTorch Geometric not installed"}
else:
    class HGT(torch.nn.Module):
        """Heterogeneous Graph Transformer model."""
        def __init__(self, hidden_channels, out_channels, num_heads, num_layers, node_types, metadata):
            super().__init__()
            self.lin_dict = torch.nn.ModuleDict()
            for node_type in node_types:
                self.lin_dict[node_type] = Linear(-1, hidden_channels)

            self.convs = torch.nn.ModuleList()
            for _ in range(num_layers):
                conv = HGTConv(hidden_channels, hidden_channels, metadata, num_heads)
                self.convs.append(conv)

            self.out_lin = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                node_type: self.lin_dict[node_type](x).relu()
                for node_type, x in x_dict.items()
            }
            for conv in self.convs:
                x_dict = conv(x_dict, edge_index_dict)
            
            # We are interested in the risk of EOAs
            return self.out_lin(x_dict['eoa'])

    class HGNAnalyzer:
        """Orchestrates HGN analysis."""
        def __init__(self, database: DatabaseEngine, device: torch.device):
            self.database = database
            self.device = device
            self.model = None
            self.eoa_map = {} # Mapping from original EOA address to its index

        def analyze(self, features_df: pd.DataFrame, test_mode: bool = False) -> Dict[str, Any]:
            """Runs the full analysis pipeline."""
            if not PYG_AVAILABLE:
                logger.error("torch_geometric is not installed. HGN analysis is disabled.")
                return {'status': 'disabled', 'error': 'torch_geometric not found'}

            logger.info("Building heterogeneous graph for HGN analysis...")
            data, metadata = self._build_hetero_graph(features_df, test_mode=test_mode)
            if data is None:
                return {'status': 'no_graph', 'error': 'Failed to build heterogeneous graph'}

            logger.info("Training HGN model...")
            self._train_model(data, metadata)

            logger.info("Generating risk scores from HGN...")
            risk_scores = self._generate_risk_scores(data)

            logger.info("Storing HGN results...")
            self._store_results(risk_scores)
            
            return {'status': 'completed', 'addresses_scored': len(risk_scores)}

        def _build_hetero_graph(self, features_df: pd.DataFrame, test_mode: bool = False) -> Optional[tuple]:
            """Builds a HeteroData object using a scalable, batched approach."""
            logger.info("Building heterogeneous graph with scalable batched edge fetching...")

            # 1. Define nodes and features from the input DataFrame
            if 'address' not in features_df.columns or 'is_contract' not in features_df.columns:
                logger.error("`address` or `is_contract` column not found in features_df.")
                return None, None

            # Split nodes into types based on the provided features_df
            eoas_df = features_df[features_df['is_contract'] == False]
            contracts_df = features_df[features_df['is_contract'] == True]
            
            # Set index for easy reindexing
            features_df = features_df.set_index('address')

            self.eoa_map = {addr: i for i, addr in enumerate(eoas_df['address'])}
            contract_map = {addr: i for i, addr in enumerate(contracts_df['address'])}
            
            # Create a combined map for quick lookup of any address type
            addr_map = {addr: is_c for addr, is_c in zip(features_df.index, features_df['is_contract'])}
            
            # Align features
            numeric_features_df = features_df.select_dtypes(include=np.number)
            
            # --- FIX: Force conversion to numeric, reindex, and fill NaNs to prevent TypeError ---
            eoa_sanitized_df = numeric_features_df.reindex(eoas_df['address']).apply(pd.to_numeric, errors='coerce').fillna(0)
            contract_sanitized_df = numeric_features_df.reindex(contracts_df['address']).apply(pd.to_numeric, errors='coerce').fillna(0)
            
            eoa_numpy = eoa_sanitized_df.astype(np.float32).values
            contract_numpy = contract_sanitized_df.astype(np.float32).values

            eoa_features = torch.tensor(eoa_numpy, dtype=torch.float32)
            contract_features = torch.tensor(contract_numpy, dtype=torch.float32)

            data = HeteroData()
            data['eoa'].x = eoa_features
            data['contract'].x = contract_features

            # 2. Fetch edges in batches
            logger.info("Fetching edges in batches for HGN...")
            edge_types = [('eoa', 'sends_to', 'eoa'), ('eoa', 'calls', 'contract'),
                          ('contract', 'sends_to', 'eoa'), ('contract', 'interacts_with', 'contract')]
            
            edges = {et: [] for et in edge_types}
            batch_size = 500000
            
            total_txs_query = "SELECT COUNT(*) as count FROM transactions WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL"
            total_txs = self.database.fetch_one(total_txs_query)['count']
            limit = 200000 if test_mode else total_txs

            for offset in tqdm(range(0, limit, batch_size), desc="Fetching Edges for HGN"):
                edges_df_batch = self.database.fetch_df(f"""
                    SELECT from_addr, to_addr FROM transactions
                    WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL
                    LIMIT {batch_size} OFFSET {offset}
                """)

                if edges_df_batch.empty:
                    break

                for _, row in edges_df_batch.iterrows():
                    src, dst = row['from_addr'], row['to_addr']
                    
                    if src not in addr_map or dst not in addr_map:
                        continue

                    src_is_contract, dst_is_contract = addr_map[src], addr_map[dst]

                    if src_is_contract and dst_is_contract:
                        edges[('contract', 'interacts_with', 'contract')].append([contract_map[src], contract_map[dst]])
                    elif src_is_contract and not dst_is_contract:
                        edges[('contract', 'sends_to', 'eoa')].append([contract_map[src], self.eoa_map[dst]])
                    elif not src_is_contract and dst_is_contract:
                        edges[('eoa', 'calls', 'contract')].append([self.eoa_map[src], contract_map[dst]])
                    elif not src_is_contract and not dst_is_contract:
                        edges[('eoa', 'sends_to', 'eoa')].append([self.eoa_map[src], self.eoa_map[dst]])

            # 3. Create final edge_index tensors
            for edge_type, edge_list in edges.items():
                if edge_list:
                    data[edge_type].edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                else:
                    data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

            data.to(self.device)
            logger.info(f"Heterogeneous graph built successfully.")
            return data, data.metadata()

        def _train_model(self, data: HeteroData, metadata: tuple):
            """Trains the HGT model using a simple regression task."""
            # For this example, we'll try to predict a simple target: the total outgoing volume
            # --- FIX: Corrected column name from 'total_volume_out_eth' to 'outgoing_volume_eth' ---
            target = self.database.fetch_df("SELECT address, outgoing_volume_eth FROM addresses").set_index('address')
            target_aligned = torch.tensor(target.reindex(self.eoa_map.keys()).fillna(0).values, dtype=torch.float).to(self.device)

            embedding_dim = config.analysis['gnn_models']['hgn']['embedding_dim']
            self.model = HGT(hidden_channels=embedding_dim, out_channels=1, num_heads=2, num_layers=2, node_types=data.node_types, metadata=metadata).to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

            for epoch in range(50):
                self.model.train()
                optimizer.zero_grad()
                out = self.model(data.x_dict, data.edge_index_dict)
                loss = F.mse_loss(out, target_aligned)
                loss.backward()
                optimizer.step()
                if (epoch + 1) % 10 == 0:
                    logger.debug(f'HGN Training Epoch: {epoch+1:02d}, Loss: {loss:.4f}')
                    
        def _generate_risk_scores(self, data: HeteroData) -> Dict[str, float]:
            """Generates risk scores based on the model's output."""
            self.model.eval()
            with torch.no_grad():
                scores_tensor = self.model(data.x_dict, data.edge_index_dict).cpu().squeeze().numpy()
            
            max_score = scores_tensor.max()
            normalized_scores = scores_tensor / max_score if max_score > 0 else np.zeros_like(scores_tensor)

            return {addr: float(normalized_scores[i]) for addr, i in self.eoa_map.items()}

        def _store_results(self, risk_scores: Dict[str, float]):
            """Stores the results in the risk_components table."""
            for address, score in risk_scores.items():
                self.database.store_component_risk(
                    address=address,
                    component_type='hgn_risk',
                    risk_score=score,
                    confidence=0.75,
                    evidence={'method': 'Risk score from Heterogeneous Graph Transformer'},
                    source_analysis='hgn_analyzer'
                )

    def run_hgn_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True):
        """Main function to run HGN analysis."""
        # --- Enhanced GPU/CPU/MPS Device Setup ---
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("🚀 Using GPU (CUDA) for HGN analysis.")
            elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("🚀 Using GPU (MPS) on Apple Silicon for HGN analysis.")
            else:
                device = torch.device('cpu')
                logger.info("🐌 GPU not available. Using CPU for HGN analysis. This may be slow.")
        else:
            device = torch.device('cpu')
            logger.info("🐌 Using CPU for HGN analysis.")

        # Mitigate potential multiprocessing issues on CPU, especially on macOS
        if str(device) == 'cpu':
            torch.set_num_threads(1)

        logger.info("🕸️ Running Heterogeneous Graph Network Analysis...")
        limit = 1000 if test_mode else 20000 # Increased limit for better graph structure
        
        # --- FIX: Query the wide 'addresses' table directly ---
        limit_clause = f"LIMIT {limit}" if limit else ""
        features_df_wide = database.fetch_df(f"SELECT * FROM addresses {limit_clause}")

        if features_df_wide.empty:
            logger.error("No features found in addresses table for HGN analysis.")
            return {'status': 'failed', 'error': 'No features available'}
        
        # The data is already in the correct wide format.
        features_df_wide = features_df_wide.fillna(0)

        analyzer = HGNAnalyzer(database, device=device)
        return analyzer.analyze(features_df_wide, test_mode=test_mode)
# src/analysis/graphsage_analyzer.py
"""
GraphSAGE Analyzer for scalable, inductive node representation learning.
"""

import logging
import platform
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm

# Import PyTorch and PyG
try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
    from sklearn.preprocessing import StandardScaler
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False

from src.core.database import DatabaseEngine
from config.config import config

logger = logging.getLogger(__name__)

# Check for PyTorch Geometric availability at module level
if not PYG_AVAILABLE:
    logger.error("PyTorch or PyTorch Geometric is not installed. GraphSAGE analysis is disabled.")
    # Define dummy classes/functions to avoid import errors elsewhere
    class SAGEConv: pass
    class Data: pass
    def run_graphsage_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True) -> Dict[str, Any]:
        return {"status": "skipped", "reason": "PyTorch Geometric not installed"}

else:
    class GraphSAGE(torch.nn.Module):
        """GraphSAGE model for node classification."""
        def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)

        def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return F.log_softmax(x, dim=1)

    def prepare_graph_data(database: DatabaseEngine, test_mode: bool = False) -> Optional[Tuple[Data, List[str], Dict[str, int], Any]]:
        """Prepares graph data from the database for PyTorch Geometric."""
        logger.info("Preparing graph data for GraphSAGE...")

        # --- REFACTOR: Use all available features from the wide 'addresses' table ---
        # This provides a richer feature set to the model without hardcoding columns.
        limit = 20000 if test_mode else 100000 # Limit for performance
        limit_clause = f"LIMIT {limit}"

        node_features = database.fetch_df(f"""
            SELECT *
            FROM addresses
            WHERE composite_risk_score IS NOT NULL
        """)
        
        if node_features.empty:
            logger.warning("No features found for GraphSAGE data preparation.")
            return None

        # Select only numeric features for the model, excluding identifiers and labels
        numeric_features_df = node_features.select_dtypes(include=np.number)
        feature_names = [col for col in numeric_features_df.columns if col not in ['address', 'cluster_id', 'composite_risk_score']]

        node_features = node_features.set_index('address').fillna(0) # Keep for address mapping

        # Create a mapping from address string to integer index
        idx_to_addr = node_features.index.tolist()
        addr_to_idx = {addr: i for i, addr in enumerate(idx_to_addr)}
        
        # --- BATCHED EDGE FETCHING FOR SCALABILITY ---
        logger.info("Fetching edges in batches to handle large datasets...")
        source_nodes_list = []
        target_nodes_list = []
        batch_size = 500000  # Process 500k transactions at a time
        
        total_txs_query = "SELECT COUNT(*) as count FROM transactions WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL"
        total_txs = database.fetch_one(total_txs_query)['count']
        
        limit = 20000 if test_mode else total_txs

        for offset in tqdm(range(0, limit, batch_size), desc="Fetching Edges"):
            edges_df_batch = database.fetch_df(f"""
                SELECT from_addr, to_addr FROM transactions
                WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL
                LIMIT {batch_size} OFFSET {offset}
            """)

            if edges_df_batch.empty:
                break

            # Filter edges where both source and target are in our node set
            valid_edges = edges_df_batch[
                edges_df_batch['from_addr'].isin(addr_to_idx) & 
                edges_df_batch['to_addr'].isin(addr_to_idx)
            ]

            if not valid_edges.empty:
                source_nodes_list.extend([addr_to_idx[addr] for addr in valid_edges['from_addr']])
                target_nodes_list.extend([addr_to_idx[addr] for addr in valid_edges['to_addr']])

        if not source_nodes_list:
            logger.warning("No valid edges found for the graph.")
            return None

        # Create edge index tensor
        source_nodes = torch.tensor(source_nodes_list, dtype=torch.long)
        target_nodes = torch.tensor(target_nodes_list, dtype=torch.long)
        edge_index = torch.stack([source_nodes, target_nodes], dim=0)

        # Create node feature tensor and labels
        # Use 'composite_risk_score' as the basis for labels
        if 'composite_risk_score' in node_features.columns:
            y = torch.tensor((node_features['composite_risk_score'] > 0.5).astype(int).values, dtype=torch.long)
            # --- FIX: Force conversion to numeric and fill NaNs to prevent TypeError ---
            sanitized_df = node_features[feature_names].apply(pd.to_numeric, errors='coerce').fillna(0)
            numpy_array = sanitized_df.astype(np.float32).values
            x_unscaled = torch.tensor(numpy_array, dtype=torch.float32)
        else:
            # Fallback if risk score is not available
            logger.warning("`composite_risk_score` not found. Using dummy labels.")
            y = torch.zeros(len(node_features), dtype=torch.long)
            x_unscaled = torch.tensor(node_features[feature_names].values, dtype=torch.float)

        # Scale features
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_unscaled.numpy())
        x = torch.tensor(x_scaled, dtype=torch.float)

        graph_data = Data(x=x, edge_index=edge_index, y=y)
        graph_data.num_classes = 2

        # Create train/val/test masks
        num_nodes = graph_data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(num_nodes * 0.7)
        val_size = int(num_nodes * 0.15)
        
        graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph_data.train_mask[indices[:train_size]] = True
        graph_data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph_data.val_mask[indices[train_size:train_size+val_size]] = True
        graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        graph_data.test_mask[indices[train_size+val_size:]] = True

        logger.info(f"Graph data created: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges.")
        return graph_data, idx_to_addr, addr_to_idx, scaler, feature_names

    def run_graphsage_analysis(database: DatabaseEngine, test_mode: bool = False, use_gpu: bool = True) -> Dict[str, Any]:
        """Orchestrates the GraphSAGE analysis pipeline."""
        
        # --- Enhanced GPU/CPU/MPS Device Setup ---
        if use_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info("🚀 Using GPU (CUDA) for GraphSAGE analysis.")
            elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("🚀 Using GPU (MPS) on Apple Silicon for GraphSAGE analysis.")
            else:
                device = torch.device('cpu')
                logger.info("🐌 GPU not available. Using CPU for GraphSAGE analysis. This may be slow.")
        else:
            device = torch.device('cpu')
            logger.info("🐌 Using CPU for GraphSAGE analysis.")
        
        # Mitigate potential multiprocessing issues on CPU, especially on macOS
        if str(device) == 'cpu':
            torch.set_num_threads(1)
        # ---

        prepared_data = prepare_graph_data(database, test_mode)
        if prepared_data is None:
            return {"status": "failed", "reason": "Could not prepare graph data."}
        
        data, idx_to_addr, addr_to_idx, scaler, feature_names = prepared_data

        # Add a check for label distribution to prevent training on a single class
        if data.train_mask.sum() > 0 and len(data.y[data.train_mask].unique()) < 2:
            logger.warning("GraphSAGE training skipped: Only one class present in the training data.")
            return {
                "status": "skipped",
                "reason": "Training data has only one class.",
                "accuracy": 0.0,
                "suspicious_nodes_found": 0
            }

        # Move data to the selected device
        data = data.to(device)

        # Initialize model using config and move to the selected device
        embedding_dim = config.analysis['gnn_models']['graphsage']['embedding_dim']
        model = GraphSAGE(
            in_channels=data.num_node_features,
            hidden_channels=embedding_dim,
            out_channels=data.num_classes
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        # --- Training Loop ---
        logger.info("Starting GraphSAGE model training...")
        model.train()
        epochs = 50 if test_mode else 200
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        # --- Evaluation ---
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum()) if int(data.test_mask.sum()) > 0 else 0
        logger.info(f"GraphSAGE Test Accuracy: {acc:.4f}")

        # --- Store Model and Mappings ---
        from pathlib import Path
        import pickle
        import json

        # --- FIX: Define models_dir before it is used ---
        models_dir = Path(config.get_models_path())
        models_dir.mkdir(exist_ok=True)

        # --- NEW: Save model metadata for the online engine ---
        model_meta = {
            'in_channels': model.conv1.in_channels,
            'hidden_channels': model.conv1.out_channels,
            'out_channels': model.conv2.out_channels,
            'feature_columns': feature_names
        }
        meta_path = models_dir / "graphsage_model_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(model_meta, f)

        model_path = models_dir / "graphsage_model.pth"
        torch.save(model.state_dict(), model_path)

        scaler_path = models_dir / "graphsage_scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        addr_map_path = models_dir / "graphsage_addr_map.json"
        with open(addr_map_path, 'w') as f:
            json.dump(addr_to_idx, f)

        # --- NEW: Save final embeddings for all nodes for similarity search ---
        with torch.no_grad():
            # Generate final embeddings for all nodes
            final_embeddings = model.conv1(data.x, data.edge_index).relu()
            final_embeddings = F.dropout(final_embeddings, p=0.5, training=False)
            final_embeddings = model.conv2(final_embeddings, data.edge_index)
        
        embeddings_path = models_dir / "graphsage_embeddings.pt"
        torch.save({'embeddings': final_embeddings.cpu(), 'addr_to_idx': addr_to_idx}, embeddings_path)

        logger.info(f"Saved GraphSAGE model and artifacts to {models_dir}")

        # --- Store Results ---
        # Identify nodes predicted as risky (class 1)
        suspicious_nodes_indices = (pred == 1).nonzero(as_tuple=True)[0]
        
        for idx in suspicious_nodes_indices:
            address = idx_to_addr[idx.item()]
            # The model's output is log_softmax, so we can get a confidence-like score
            # REUSE `out` tensor instead of re-running the model
            confidence = torch.exp(out[idx])[1].item()
            
            database.store_component_risk(
                address=address,
                component_type='graphsage_risk',
                risk_score=confidence, # Use prediction confidence as risk score
                confidence=0.8, # Base confidence for the model
                evidence={'method': 'Supervised GraphSAGE classification'},
                source_analysis='graphsage_analyzer'
            )
        
        # +++ NEW: Store all predictions as cluster assignments for consensus engine +++
        all_predictions = pred.cpu().numpy()
        records_to_insert = []
        for i, prediction in enumerate(all_predictions):
            records_to_insert.append((
                idx_to_addr[i],
                f"SAGE_{prediction}",  # e.g., SAGE_0, SAGE_1
                'graphsage_classification',
                torch.exp(out[i])[prediction].item()  # Use prediction confidence
            ))

        with database.transaction():
            for record in records_to_insert:
                # Use ON CONFLICT to make the operation idempotent.
                database.execute("""
                    INSERT INTO cluster_assignments (address, cluster_id, cluster_type, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(address, cluster_type) DO UPDATE SET
                        cluster_id = excluded.cluster_id,
                        confidence = excluded.confidence
                """, record)
        
        logger.info(f"Stored {len(records_to_insert)} GraphSAGE cluster assignments.")
        logger.info(f"Stored risk scores for {suspicious_nodes_indices.size(0)} addresses predicted as risky.")

        return {
            "status": "completed",
            "device_used": str(device),
            "accuracy": acc,
            "suspicious_nodes_found": suspicious_nodes_indices.size(0)
        }
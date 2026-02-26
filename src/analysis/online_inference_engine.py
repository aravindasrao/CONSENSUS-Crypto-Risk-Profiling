# src/analysis/online_inference_engine.py
"""
Online Inference Engine for Incremental Forensic Analysis.

This module handles the analysis of new, incoming transaction data without
rerunning the entire pipeline. It uses pre-trained models to generate
embeddings for new addresses and associate them with existing clusters.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import pickle
import platform
import time

# PyTorch imports
try:
    import torch
    import torch.nn.functional as F
    from src.analysis.graphsage_analyzer import GraphSAGE # Assuming GraphSAGE class is accessible
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from src.core.database import DatabaseEngine
from src.core.csv_data_manager import CSVDataManager
from src.analysis.foundation_layer import FoundationLayer
from config.config import config

logger = logging.getLogger(__name__)

class OnlineInferenceEngine:
    """
    Performs incremental analysis on new data using pre-trained models.
    """

    def __init__(self, database: DatabaseEngine):
        self.database = database
        self.models_dir = Path(config.get_models_path())
        self.model = None
        self.scaler = None
        self.addr_to_idx = {}
        self.idx_to_addr = []
        self.cluster_centroids = {}
        self.feature_columns = []
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif platform.system() == 'Darwin' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        logger.info("Online Inference Engine initialized.")

    def run_incremental_analysis(self, data_dir: str):
        """
        Main entry point for running the incremental update.
        This method is now refactored to process new addresses in batches for scalability.

        Args:
            data_dir: The directory containing new CSV transaction files.
        """
        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch is not available. Cannot run online inference.")
            return

        logger.info(f"Starting incremental analysis on data from: {data_dir}")

        # 1. Load the pre-trained models and necessary artifacts
        if not self._load_trained_model():
            return

        # 2. Ingest any new data from CSVs
        self._ingest_new_data(data_dir)

        # 3. Process new addresses in batches
        batch_size = 10000
        while True:
            logger.info(f"Processing new address batch...")
            
            # Fetch a batch of addresses that haven't been processed by this engine yet
            new_addresses_df = self.database.fetch_df(f"""
                SELECT a.address FROM addresses a
                LEFT JOIN cluster_assignments ca ON a.address = ca.address AND ca.cluster_type = 'online_inference'
                WHERE ca.address IS NULL
                LIMIT {batch_size}
            """)
            
            if new_addresses_df.empty:
                logger.info("No more new addresses to process. Incremental analysis complete.")
                break
            
            new_addresses = new_addresses_df['address'].tolist()

            # 4. Update/create features for the current batch of new addresses
            features_df = self._update_features_for_new_addresses(new_addresses)
            if features_df.empty:
                logger.error("Feature generation failed for a batch of new addresses. Aborting incremental run.")
                break

            # 5. Generate inductive embeddings for the new addresses
            new_embeddings = self._generate_inductive_embeddings(features_df)

            # 6. Probabilistic cluster association
            cluster_associations = self._associate_with_clusters(new_embeddings, features_df)

            # 7. Update database with new cluster assignments
            self._update_database(cluster_associations)

    def _load_trained_model(self) -> bool:
        """Loads the pre-trained GraphSAGE model, scaler, and address map."""
        logger.info(f"Loading trained models from {self.models_dir}...")

        # --- NEW: Check for all required files at the beginning for a clear failure message ---
        required_files = {
            "meta": self.models_dir / "graphsage_model_meta.json",
            "model": self.models_dir / "graphsage_model.pth",
            "scaler": self.models_dir / "graphsage_scaler.pkl",
            "address_map": self.models_dir / "graphsage_addr_map.json",
            "embeddings": self.models_dir / "graphsage_embeddings.pt"
        }

        for name, path in required_files.items():
            if not path.exists():
                logger.error(f"Missing required model artifact: '{name}' at {path}. Please run a full GPU analysis first to generate models.")
                return False

        try:
            with open(required_files["meta"], 'r') as f:
                model_meta = json.load(f)

            self.feature_columns = model_meta['feature_columns']
            self.model = GraphSAGE(
                in_channels=model_meta['in_channels'],
                hidden_channels=model_meta['hidden_channels'],
                out_channels=model_meta['out_channels']
            ).to(self.device)
            self.model.load_state_dict(torch.load(required_files["model"], map_location=self.device))
            self.model.eval() # Set to evaluation mode
            logger.info("GraphSAGE model loaded.")

            with open(required_files["scaler"], 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info("Feature scaler loaded.")

            with open(required_files["address_map"], 'r') as f:
                self.addr_to_idx = json.load(f)
            self.idx_to_addr = {i: addr for addr, i in self.addr_to_idx.items()}
            logger.info(f"Address map loaded with {len(self.addr_to_idx)} entries.")
            logger.info(f"Model configured to use {len(self.feature_columns)} features.")

            # Load existing embeddings and pre-calculate cluster centroids
            if not self._get_cluster_centroids():
                return False
            
            return True
        except Exception as e:
            logger.error(f"An error occurred while loading models: {e}")
            return False

    def _get_cluster_centroids(self) -> bool:
        """Loads embeddings and calculates the centroid for each existing cluster."""
        logger.info("Calculating centroids for existing clusters...")
        try:
            # 1. Load the saved embeddings tensor
            embeddings_path = self.models_dir / "graphsage_embeddings.pt"
            saved_data = torch.load(embeddings_path, map_location=self.device)
            all_embeddings = saved_data['embeddings']
            addr_to_idx = saved_data['addr_to_idx']
            idx_to_addr = {i: addr for addr, i in addr_to_idx.items()}

            # 2. Fetch cluster assignments for all known addresses
            assignments_df = self.database.fetch_df("SELECT address, cluster_id FROM cluster_assignments")
            if assignments_df.empty:
                logger.warning("No existing cluster assignments found. Cannot create centroids.")
                return False
            
            # 3. Group addresses by cluster ID
            clusters = assignments_df.groupby('cluster_id')['address'].apply(list).to_dict()

            # 4. Calculate centroid for each cluster
            for cluster_id, members in clusters.items():
                member_indices = [addr_to_idx[addr] for addr in members if addr in addr_to_idx]
                if member_indices:
                    cluster_embeddings = all_embeddings[member_indices]
                    centroid = torch.mean(cluster_embeddings, dim=0)
                    self.cluster_centroids[cluster_id] = centroid
            
            logger.info(f"Calculated centroids for {len(self.cluster_centroids)} clusters.")
            return True

        except FileNotFoundError:
            logger.error(f"Embeddings file not found at {embeddings_path}. Cannot calculate centroids.")
            return False
        except Exception as e:
            logger.error(f"Failed to calculate cluster centroids: {e}")
            return False

    def _ingest_new_data(self, data_dir: str):
        """Uses CSVDataManager to load only new data."""
        logger.info("Ingesting new data...")
        csv_manager = CSVDataManager(self.database)
        result = csv_manager.ensure_data_loaded(data_dir=data_dir, force_reload=False)

        if not result['success']:
            logger.error("Failed to ingest new data.")
        else:
            logger.info("New data ingestion complete.")

    def _update_features_for_new_addresses(self, new_addresses: List[str]) -> pd.DataFrame:
        """Runs the FoundationLayer for only the new addresses."""
        logger.info(f"Updating features for {len(new_addresses)} new or updated addresses.")
        foundation = FoundationLayer(self.database)
        features_df = foundation.extract_features(new_addresses)
        return features_df

    def _generate_inductive_embeddings(self, features_df: pd.DataFrame) -> np.ndarray:
        """Generates embeddings for new nodes using the pre-trained model."""
        logger.info(f"Generating inductive embeddings for {len(features_df)} new addresses...")
        
        if features_df.empty:
            return np.array([])

        # Ensure feature columns are in the correct order
        # Use the loaded feature columns to ensure consistency with the trained model
        feature_cols = self.feature_columns
        missing_cols = [col for col in feature_cols if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns for inference: {missing_cols}")

        X = features_df[feature_cols].values

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float).to(self.device)

        # Use the trained model as a feature extractor.
        # Since we don't have the new nodes in the original graph, we can't do message passing.
        # We apply the learned linear transformations from the GNN layers.
        with torch.no_grad():
            # This is a valid way to get an embedding using the learned weights
            # without needing the graph structure for the new nodes.
            embeddings = self.model.conv1.lin_l(X_tensor)
            embeddings = embeddings.relu()
            embeddings = self.model.conv2.lin_l(embeddings)

        return embeddings.cpu().numpy()

    def _associate_with_clusters(self, new_embeddings: np.ndarray, features_df: pd.DataFrame) -> List[Dict]:
        """Associates new nodes with existing clusters based on embedding similarity."""
        logger.info("Associating new addresses with existing clusters...")
        if new_embeddings.size == 0 or not self.cluster_centroids:
            return []

        new_embeddings_tensor = torch.tensor(new_embeddings, dtype=torch.float).to(self.device)
        associations = []
        
        # Prepare centroid tensor for efficient computation
        cluster_ids = list(self.cluster_centroids.keys())
        centroids_tensor = torch.stack(list(self.cluster_centroids.values()))

        # Calculate cosine similarity between all new embeddings and all centroids
        similarity_matrix = F.cosine_similarity(new_embeddings_tensor.unsqueeze(1), centroids_tensor.unsqueeze(0), dim=2)

        # Find the best match for each new embedding
        best_matches = torch.max(similarity_matrix, dim=1)
        best_scores, best_indices = best_matches.values, best_matches.indices

        for i, (score, index) in enumerate(zip(best_scores, best_indices)):
            associations.append({
                'address': features_df.iloc[i]['address'],
                'cluster_id': cluster_ids[index.item()],
                'confidence': score.item(),
                'evidence_type': 'online_inference'
            })
        return associations

    def _update_database(self, associations: List[Dict]):
        """Updates the database with new cluster assignments using a scalable bulk insert."""
        if not associations:
            logger.info("No new associations to update in the database.")
            return

        logger.info(f"Updating database with {len(associations)} new cluster associations.")
        
        updates_df = pd.DataFrame(associations)
        
        temp_table_name = f"temp_online_updates_{int(time.time() * 1000)}"
        try:
            self.database.connection.register(temp_table_name, updates_df)
            
            # Use DuckDB's efficient INSERT ... ON CONFLICT syntax
            self.database.execute(f"""
                INSERT INTO cluster_assignments (address, cluster_id, cluster_type, confidence)
                SELECT address, cluster_id, evidence_type, confidence FROM {temp_table_name}
                ON CONFLICT(address, cluster_type) DO UPDATE SET
                    cluster_id = excluded.cluster_id,
                    confidence = excluded.confidence,
                    created_at = CURRENT_TIMESTAMP
            """)
            
            logger.info(f"Database updated with {len(associations)} new associations.")
        except Exception as e:
            logger.error(f"Failed to bulk update online cluster associations: {e}", exc_info=True)
        finally:
            try:
                self.database.connection.unregister(temp_table_name)
            except Exception:
                pass
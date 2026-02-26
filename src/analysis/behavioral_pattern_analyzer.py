# src/analysis/behavioral_pattern_analyzer.py
"""
Analyzes deposit and withdrawal patterns to identify and cluster similar behaviors.
This is the core of the Behavioral Clustering
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import logging

from src.core.database import DatabaseEngine
from src.utils.comprehensive_contract_database import ComprehensiveContractDatabase

# Try to import hdbscan, but make it an optional dependency
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

logger = logging.getLogger(__name__)

class BehavioralPatternAnalyzer:
    def __init__(self, database: DatabaseEngine):
        self.database = database
        self.contract_db = ComprehensiveContractDatabase(database)
        self.pattern_features = None
        self.clusters = None
        logger.info("Behavioral Pattern Analyzer initialized (from Deposit/Withdrawal Patterns)")

    def analyze_and_cluster(self, contract_category: str = 'tornado_cash', min_transactions: int = 3, n_clusters: Optional[int] = None, method: str = 'hdbscan') -> Dict[str, Any]:
        """
        Analyzes deposit/withdrawal patterns and clusters them. This method is
        designed to be scalable by processing addresses in batches.
        Args:
            contract_category: The category of contracts to analyze (e.g., 'tornado_cash', 'other_mixers').
            min_transactions: Minimum number of deposit/withdrawal transactions
                              an address must have to be included in the analysis.
            n_clusters: Optional number of clusters to form. If None, the method
                        will attempt to determine an optimal number.
            method: The clustering algorithm to use ('kmeans', 'hdbscan'). Defaults to 'hdbscan'.
        Returns:
            A dictionary containing clustering results and status information.
        """
        logger.info(f"Starting scalable behavioral pattern analysis for contract category: '{contract_category}'...")
        
        # 1. Get all eligible addresses that meet the minimum transaction criteria
        mixer_contracts = self.contract_db.get_contracts_by_category(contract_category)
        if not mixer_contracts:
            logger.warning(f"No contracts found for category '{contract_category}'. Skipping behavioral analysis.")
            return {'status': 'no_contracts', 'clusters': {}}
        
        mixer_addresses_sql = ','.join([f"'{addr.lower()}'" for addr in mixer_contracts])

        address_query = f"""
        WITH user_activity AS (
            SELECT 
                CASE WHEN LOWER(to_addr) IN ({mixer_addresses_sql}) THEN from_addr ELSE to_addr END as user_address
            FROM transactions
            WHERE (LOWER(to_addr) IN ({mixer_addresses_sql}) OR LOWER(from_addr) IN ({mixer_addresses_sql}))
            GROUP BY user_address
            HAVING COUNT(*) >= ?
        )
        SELECT user_address FROM user_activity
        """
        eligible_addresses_df = self.database.fetch_df(address_query, (min_transactions,))

        if eligible_addresses_df.empty:
            logger.warning("No suitable deposit/withdrawal data found for pattern analysis.")
            return {'status': 'no_data', 'clusters': {}}

        eligible_addresses = eligible_addresses_df['user_address'].tolist()
        logger.info(f"Found {len(eligible_addresses)} addresses with sufficient activity for behavioral analysis.")

        # 2. Process addresses in batches to engineer features
        all_features_list = []
        batch_size = 5000  # Process 5000 addresses at a time
        for i in tqdm(range(0, len(eligible_addresses), batch_size), desc="Engineering Behavioral Features"):
            address_batch = eligible_addresses[i:i + batch_size]
            dep_with_data_batch = self._extract_batch_transaction_data(address_batch, mixer_addresses_sql)
            if not dep_with_data_batch.empty:
                batch_features_df = self._engineer_pattern_features(dep_with_data_batch)
                all_features_list.append(batch_features_df)

        if not all_features_list:
            logger.warning("No features were engineered. Aborting.")
            return {'status': 'no_features', 'clusters': {}}

        self.pattern_features = pd.concat(all_features_list, ignore_index=True)
        
        # 3. Perform clustering
        clustering_results = self._cluster_patterns(self.pattern_features, n_clusters=n_clusters, method=method)
        
        # Add cluster labels to the features DataFrame
        self.pattern_features['cluster_id'] = clustering_results['labels']

        # +++ NEW: Store results in the database for the consensus engine +++
        self._store_behavioral_clusters(self.pattern_features)

        logger.info(f"Behavioral clustering complete. Found {clustering_results['n_clusters']} distinct patterns.")
        return {
            'status': 'completed',
            'clusters': self.pattern_features[['address', 'cluster_id']].to_dict('records'),
            'metrics': clustering_results['metrics']
        }

    def _store_behavioral_clusters(self, features_df: pd.DataFrame):
        """Stores the behavioral cluster assignments in the database."""
        if 'cluster_id' not in features_df.columns:
            logger.warning("No cluster IDs found to store for behavioral patterns.")
            return

        records_to_insert = []
        for _, row in features_df.iterrows():
            records_to_insert.append((
                row['address'],
                f"BEH_{row['cluster_id']}",  # Prefix to avoid ID collision
                'behavioral_similarity',
                0.6  # Default confidence for this method
            ))
        
        with self.database.transaction():
            # Use ON CONFLICT to make the operation idempotent. If the job is rerun,
            # it will update the existing entry instead of creating a duplicate.
            for record in records_to_insert:
                self.database.execute("""
                    INSERT INTO cluster_assignments (address, cluster_id, cluster_type, confidence)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(address, cluster_type) DO UPDATE SET
                        cluster_id = excluded.cluster_id,
                        confidence = excluded.confidence
                """, record)
        logger.info(f"Stored {len(records_to_insert)} behavioral cluster assignments.")

    def _extract_batch_transaction_data(self, address_batch: List[str], mixer_addresses_sql: str) -> pd.DataFrame:
        """Extracts transaction data for a specific batch of addresses."""
        placeholders = ','.join(['?'] * len(address_batch))
        
        query = f"""
        SELECT
            t.timestamp,
            t.value_eth,
            CASE
                WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN t.from_addr
                ELSE t.to_addr
            END as user_address,
            CASE
                WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN 'deposit'
                ELSE 'withdrawal'
            END as tx_type
        FROM transactions t
        WHERE (LOWER(t.to_addr) IN ({mixer_addresses_sql}) OR LOWER(t.from_addr) IN ({mixer_addresses_sql}))
          AND (CASE WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN t.from_addr ELSE t.to_addr END) IN ({placeholders})
        """
        return self.database.fetch_df(query, tuple(address_batch))

    def _extract_transaction_data(self, min_transactions: int) -> pd.DataFrame:
        """Extracts deposit/withdrawal data related to known mixer contracts."""
        mixer_contracts = self.contract_db.get_contracts_by_category('tornado_cash')
        if not mixer_contracts:
            return pd.DataFrame()

        mixer_addresses_sql = ','.join([f"'{addr}'" for addr in mixer_contracts])
        
        # This query correctly identifies deposits (to mixer) and withdrawals (from mixer)
        # and filters for addresses with minimum activity.
        query = f"""
        WITH user_activity AS (
            SELECT 
                CASE WHEN to_addr IN ({mixer_addresses_sql}) THEN from_addr ELSE to_addr END as user_address,
                COUNT(*) as tx_count
            FROM transactions
            WHERE (to_addr IN ({mixer_addresses_sql}) OR from_addr IN ({mixer_addresses_sql}))
            GROUP BY user_address
            HAVING tx_count >= {min_transactions}
        )
        SELECT
            t.timestamp,
            t.value_eth,
            CASE
                WHEN t.to_addr IN ({mixer_addresses_sql}) THEN t.from_addr
                ELSE t.to_addr
            END as user_address,
            CASE
                WHEN t.to_addr IN ({mixer_addresses_sql}) THEN 'deposit'
                ELSE 'withdrawal'
            END as tx_type
        FROM transactions t
        WHERE (t.to_addr IN ({mixer_addresses_sql}) OR t.from_addr IN ({mixer_addresses_sql}))
          AND (t.from_addr IN (SELECT user_address FROM user_activity) OR t.to_addr IN (SELECT user_address FROM user_activity));
        """
        return self.database.fetch_df(query)

    def _engineer_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineers features that capture behavioral patterns for each address.
        Args:
            data: DataFrame containing transaction data with columns:
                  ['timestamp', 'value_eth', 'user_address', 'tx_type']
        Returns:
            DataFrame with engineered features for each address.
        """
        features_list = []
        for address, group in data.groupby('user_address'):
            deposits = group[group['tx_type'] == 'deposit']
            withdrawals = group[group['tx_type'] == 'withdrawal']

            if deposits.empty and withdrawals.empty:
                continue

            features = {'address': address}
            
            # Volume Features
            features['deposit_volume'] = deposits['value_eth'].sum()
            features['withdrawal_volume'] = withdrawals['value_eth'].sum()
            features['volume_ratio'] = features['deposit_volume'] / (features['withdrawal_volume'] + 1e-6)
            features['avg_deposit_amount'] = deposits['value_eth'].mean()
            features['avg_withdrawal_amount'] = withdrawals['value_eth'].mean()
            
            # Frequency Features
            features['deposit_count'] = len(deposits)
            features['withdrawal_count'] = len(withdrawals)
            
            # Temporal Features
            dep_timestamps = pd.to_datetime(deposits['timestamp'], unit='s')
            with_timestamps = pd.to_datetime(withdrawals['timestamp'], unit='s')
            
            features['avg_deposit_interval_hours'] = dep_timestamps.diff().mean().total_seconds() / 3600 if len(dep_timestamps) > 1 else 0
            features['avg_withdrawal_interval_hours'] = with_timestamps.diff().mean().total_seconds() / 3600 if len(with_timestamps) > 1 else 0
            
            # Interleaving Score (how mixed are deposits and withdrawals)
            all_tx = pd.concat([
                deposits.assign(type_code=0),
                withdrawals.assign(type_code=1)
            ]).sort_values('timestamp')
            features['interleaving_score'] = all_tx['type_code'].diff().abs().sum() / (len(all_tx) + 1e-6)

            features_list.append(features)

        features_df = pd.DataFrame(features_list).fillna(0)
        return features_df

    def _cluster_patterns(self, features_df: pd.DataFrame, n_clusters: Optional[int] = None, method: str = 'hdbscan') -> Dict[str, Any]:
        """Clusters the behavioral patterns using K-Means.
        Args:
            features_df: DataFrame containing engineered features for each address.
            n_clusters: Optional number of clusters to form. If None, the method
                        will attempt to determine an optimal number.
            method: The clustering algorithm to use ('kmeans', 'hdbscan').
        Returns:
            A dictionary containing clustering results and metrics.
        """
        feature_cols = [col for col in features_df.columns if col != 'address']
        X = features_df[feature_cols].values
        
        # FIX: Filter out features with zero variance before scaling
        valid_feature_mask = X.std(axis=0) != 0
        X_filtered = X[:, valid_feature_mask]
        
        if X_filtered.shape[1] == 0:
            logger.warning("All features have zero variance. Skipping clustering.")
            return {
                'method': 'none',
                'n_clusters': 0,
                'labels': np.zeros(X.shape[0]),
                'metrics': {'silhouette_score': 0}
            }
            
        # Use RobustScaler as it's less sensitive to outliers, which are common in this data
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        if method == 'hdbscan' and HDBSCAN_AVAILABLE:
            logger.info("Using HDBSCAN for density-based clustering.")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=max(5, int(len(X_scaled) * 0.01)), # min 5 or 1% of data
                                        min_samples=1,
                                        gen_min_span_tree=True,
                                        allow_single_cluster=True)
            labels = clusterer.fit_predict(X_scaled)
            # HDBSCAN labels outliers as -1. We'll give them a unique high cluster ID.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"HDBSCAN found {n_clusters} clusters and {np.sum(labels == -1)} outliers.")
            
            return {
                'method': 'hdbscan',
                'n_clusters': n_clusters,
                'labels': labels,
                'metrics': {'silhouette_score': silhouette_score(X_scaled, labels) if n_clusters > 1 else 0}
            }

        # Fallback to KMeans if HDBSCAN is not available or specified
        if n_clusters is None:
            max_k = min(25, len(features_df) - 1)
            if max_k > 1:
                # Simple silhouette score to find a reasonable k
                scores = {}
                for k in range(2, max_k):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X_scaled)
                    scores[k] = silhouette_score(X_scaled, kmeans.labels_)
                
                # Find the best score, but only if it's positive. Otherwise, no meaningful clusters.
                best_k = max(scores, key=scores.get)
                if scores[best_k] > 0:
                    n_clusters = best_k
                else:
                    n_clusters = 1 # No meaningful clusters found
            else:
                n_clusters = 1
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(X_scaled)

        return {
            'method': 'kmeans',
            'n_clusters': n_clusters,
            'labels': labels,
            'metrics': {'silhouette_score': silhouette_score(X_scaled, labels) if n_clusters > 1 else 0}
        }
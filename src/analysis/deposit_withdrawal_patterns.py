"""
Deposit and Withdrawal Pattern Analysis with Clustering

This module analyzes deposit and withdrawal patterns in blockchain transactions,
identifies behavioral patterns, and clusters similar patterns together.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')


from src.core.database import DatabaseEngine
from src.utils.comprehensive_contract_database import ComprehensiveContractDatabase

logger = logging.getLogger(__name__)


class DepositWithdrawalPatternAnalyzer:
    """
    Analyzes deposit and withdrawal patterns to identify clusters of similar behavior.
    
    Features analyzed:
    - Temporal patterns (timing, frequency, periodicity)
    - Volume patterns (amounts, distributions)
    - Sequence patterns (deposit-withdrawal relationships)
    - Behavioral patterns (mixing strategies)
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.contract_db = ComprehensiveContractDatabase(self.database)
        self.pattern_features = None
        self.clusters = None
        self.cluster_profiles = None
        logger.info("Deposit/Withdrawal Pattern Analyzer initialized")
    
    def analyze_patterns(self, 
                         min_transactions: int = 5,
                         clustering_method: str = 'kmeans',
                         n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Main analysis pipeline for deposit/withdrawal patterns.
        
        Args:
            min_transactions: Minimum transactions for an address to be included
            clustering_method: 'kmeans', 'dbscan', or 'hierarchical'
            n_clusters: Number of clusters (auto-determined if None)
        
        Returns:
            Analysis results with patterns and clusters
        """
        logger.info("Starting scalable deposit/withdrawal pattern analysis...")
        
        try:
            # 1. Get all eligible addresses
            eligible_addresses, mixer_addresses_sql = self._get_eligible_addresses(min_transactions)
            if not eligible_addresses:
                logger.warning("No addresses with sufficient activity found for pattern analysis.")
                return {'error': 'No data available for analysis', 'success': False}

            logger.info(f"Found {len(eligible_addresses)} addresses with sufficient activity for pattern analysis.")

            # 2. Process addresses in batches to engineer features
            all_raw_data_list = []
            all_features_list = []
            batch_size = 5000
            for i in tqdm(range(0, len(eligible_addresses), batch_size), desc="Engineering Deposit/Withdrawal Features"):
                address_batch = eligible_addresses[i:i + batch_size]
                batch_data = self._extract_batch_data(address_batch, mixer_addresses_sql)
                if not batch_data.empty:
                    all_raw_data_list.append(batch_data)
                    batch_features_df = self._engineer_pattern_features(batch_data)
                    all_features_list.append(batch_features_df)

            if not all_features_list:
                logger.warning("No features were engineered. Aborting.")
                return {'error': 'No features engineered', 'success': False}

            dep_with_data = pd.concat(all_raw_data_list, ignore_index=True)
            self.pattern_features = pd.concat(all_features_list, ignore_index=True)
            
            # 3. Perform clustering
            clustering_results = self._cluster_patterns(
                self.pattern_features,
                method=clustering_method,
                n_clusters=n_clusters
            )
            
            # 4. Profile clusters
            self.cluster_profiles = self._profile_clusters(
                self.pattern_features,
                clustering_results['labels']
            )
            
            # 5. Identify anomalous patterns
            anomalies = self._identify_anomalous_patterns(
                self.pattern_features,
                clustering_results['labels']
            )
            
            # 6. Generate insights
            insights = self._generate_pattern_insights(
                self.pattern_features,
                clustering_results,
                self.cluster_profiles
            )
            
            # 7. Update database with pattern clusters
            self._update_database_with_patterns(
                self.pattern_features,
                clustering_results['labels']
            )
            
            return {
                'success': True,
                'addresses_analyzed': len(dep_with_data['user_address'].unique()),
                'features_engineered': len(self.pattern_features.columns) - 1,
                'clustering_results': clustering_results,
                'cluster_profiles': self.cluster_profiles,
                'anomalous_patterns': anomalies,
                'insights': insights,
                'pattern_statistics': self._calculate_pattern_statistics(dep_with_data)
            }
            
        except Exception as e:
            logger.error(f"Pattern analysis failed: {e}")
            return {'error': str(e), 'success': False}

    def _get_eligible_addresses(self, min_transactions: int) -> Tuple[List[str], str]:
        """Gets a list of addresses eligible for analysis."""
        mixer_contracts = self.contract_db.get_contracts_by_category('tornado_cash')
        if not mixer_contracts:
            logger.warning("No Tornado Cash contracts found in the contract database.")
            return [], ""

        # DuckDB is case-insensitive by default, but using LOWER() is safer.
        mixer_addresses_sql = ','.join([f"'{addr.lower()}'" for addr in mixer_contracts])

        query = f"""
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
        eligible_addresses_df = self.database.fetch_df(query, (min_transactions,))
        
        if eligible_addresses_df.empty:
            return [], mixer_addresses_sql
            
        return eligible_addresses_df['user_address'].tolist(), mixer_addresses_sql

    def _extract_batch_data(self, address_batch: List[str], mixer_addresses_sql: str) -> pd.DataFrame:
        """Extracts transaction data for a specific batch of addresses."""
        placeholders = ','.join(['?'] * len(address_batch))
        query = f"""
        SELECT 
            t.from_addr, t.to_addr, t.value_eth, t.timestamp, t.block_number, t.gas, t.gas_price, t.method_name,
            CASE WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN t.from_addr ELSE t.to_addr END as user_address,
            CASE WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN 'deposit' ELSE 'withdrawal' END as tx_type
        FROM transactions t
        WHERE (LOWER(t.to_addr) IN ({mixer_addresses_sql}) OR LOWER(t.from_addr) IN ({mixer_addresses_sql}))
          AND (CASE WHEN LOWER(t.to_addr) IN ({mixer_addresses_sql}) THEN from_addr ELSE to_addr END) IN ({placeholders})
        ORDER BY t.timestamp
        """
        return self.database.fetch_df(query, tuple(address_batch))
    
    def _engineer_pattern_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features that capture deposit/withdrawal patterns.
        """
        features_list = []
        
        # Group by the new 'user_address' column which correctly identifies the user
        for address, group in data.groupby('user_address'):
            addr_deposits = group[group['tx_type'] == 'deposit']
            addr_withdrawals = group[group['tx_type'] == 'withdrawal']
            
            features = {'address': address}
            
            # 1. VOLUME FEATURES
            features['total_deposit_volume'] = addr_deposits['value_eth'].sum()
            features['total_withdrawal_volume'] = addr_withdrawals['value_eth'].sum()
            features['volume_ratio'] = (features['total_withdrawal_volume'] / 
                                       max(features['total_deposit_volume'], 0.001))
            features['avg_deposit_amount'] = addr_deposits['value_eth'].mean() if len(addr_deposits) > 0 else 0
            features['avg_withdrawal_amount'] = addr_withdrawals['value_eth'].mean() if len(addr_withdrawals) > 0 else 0
            features['deposit_amount_std'] = addr_deposits['value_eth'].std() if len(addr_deposits) > 1 else 0
            features['withdrawal_amount_std'] = addr_withdrawals['value_eth'].std() if len(addr_withdrawals) > 1 else 0
            
            # 2. FREQUENCY FEATURES
            features['deposit_count'] = len(addr_deposits)
            features['withdrawal_count'] = len(addr_withdrawals)
            features['deposit_withdrawal_ratio'] = (features['deposit_count'] / 
                                                   max(features['withdrawal_count'], 1))
            
            # 3. TEMPORAL FEATURES
            if len(addr_deposits) > 0:
                dep_timestamps = pd.to_datetime(addr_deposits['timestamp'], unit='s')
                features['deposit_timespan_days'] = (dep_timestamps.max() - dep_timestamps.min()).days
                features['avg_deposit_interval_hours'] = 0
                if len(dep_timestamps) > 1:
                    intervals = dep_timestamps.sort_values().diff().dropna()
                    features['avg_deposit_interval_hours'] = intervals.mean().total_seconds() / 3600
                features['deposit_time_std_hours'] = dep_timestamps.dt.hour.std() if len(dep_timestamps) > 1 else 0
            else:
                features['deposit_timespan_days'] = 0
                features['avg_deposit_interval_hours'] = 0
                features['deposit_time_std_hours'] = 0
            
            if len(addr_withdrawals) > 0:
                with_timestamps = pd.to_datetime(addr_withdrawals['timestamp'], unit='s')
                features['withdrawal_timespan_days'] = (with_timestamps.max() - with_timestamps.min()).days
                features['avg_withdrawal_interval_hours'] = 0
                if len(with_timestamps) > 1:
                    intervals = with_timestamps.sort_values().diff().dropna()
                    features['avg_withdrawal_interval_hours'] = intervals.mean().total_seconds() / 3600
                features['withdrawal_time_std_hours'] = with_timestamps.dt.hour.std() if len(with_timestamps) > 1 else 0
            else:
                features['withdrawal_timespan_days'] = 0
                features['avg_withdrawal_interval_hours'] = 0
                features['withdrawal_time_std_hours'] = 0
            
            # 4. SEQUENCE FEATURES
            features['has_both_types'] = 1 if (features['deposit_count'] > 0 and features['withdrawal_count'] > 0) else 0
            
            # Calculate deposit-withdrawal time gaps
            if features['has_both_types']:
                dep_times = pd.to_datetime(addr_deposits['timestamp'], unit='s')
                with_times = pd.to_datetime(addr_withdrawals['timestamp'], unit='s')
                
                # Find minimum time between any deposit and withdrawal
                min_gap_hours = float('inf')
                for dep_time in dep_times:
                    gaps = abs((with_times - dep_time).total_seconds() / 3600)
                    if len(gaps) > 0:
                        min_gap_hours = min(min_gap_hours, gaps.min())
                
                features['min_deposit_withdrawal_gap_hours'] = min_gap_hours if min_gap_hours != float('inf') else 0
                
                # Check for interleaved patterns
                all_times = pd.concat([
                    pd.Series(dep_times.values, index=['deposit']*len(dep_times)),
                    pd.Series(with_times.values, index=['withdrawal']*len(with_times))
                ]).sort_values()
                
                # Count transitions
                transitions = 0
                for i in range(1, len(all_times)):
                    if all_times.index[i] != all_times.index[i-1]:
                        transitions += 1
                features['transaction_transitions'] = transitions
                features['interleaving_score'] = transitions / max(len(all_times) - 1, 1)
            else:
                features['min_deposit_withdrawal_gap_hours'] = 0
                features['transaction_transitions'] = 0
                features['interleaving_score'] = 0
            
            # 5. BEHAVIORAL FEATURES
            features['uses_round_amounts'] = self._check_round_amounts(
                pd.concat([addr_deposits['value_eth'], addr_withdrawals['value_eth']])
            )
            features['time_consistency_score'] = self._calculate_time_consistency(
                pd.concat([addr_deposits['timestamp'], addr_withdrawals['timestamp']])
            )
            features['gas_price_variance'] = pd.concat([
                addr_deposits['gas_price'], 
                addr_withdrawals['gas_price']
            ]).std() if len(pd.concat([addr_deposits, addr_withdrawals])) > 1 else 0
            
            # 6. PATTERN COMPLEXITY
            features['pattern_complexity'] = self._calculate_pattern_complexity(features)
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _check_round_amounts(self, amounts: pd.Series) -> float:
        """Check if transactions use round amounts (0.1, 1, 10, 100 ETH)."""
        if len(amounts) == 0:
            return 0
        
        round_amounts = [0.1, 1.0, 10.0, 100.0]
        round_count = sum(1 for amt in amounts if any(abs(amt - r) < 0.001 for r in round_amounts))
        return round_count / len(amounts)
    
    def _calculate_time_consistency(self, timestamps: pd.Series) -> float:
        """Calculate how consistent the timing patterns are."""
        if len(timestamps) <= 2:
            return 1.0
        
        timestamps = pd.to_datetime(timestamps, unit='s').sort_values()
        intervals = timestamps.diff().dropna()
        
        if len(intervals) == 0:
            return 1.0
        
        # Calculate coefficient of variation
        mean_interval = intervals.mean()
        std_interval = intervals.std()
        
        if mean_interval.total_seconds() == 0:
            return 0
        
        cv = std_interval.total_seconds() / mean_interval.total_seconds()
        # Convert to score (lower CV = higher consistency)
        return max(0, 1 - cv)
    
    def _calculate_pattern_complexity(self, features: Dict) -> float:
        """Calculate overall pattern complexity score."""
        complexity = 0
        
        # More complex if using both deposits and withdrawals
        if features['has_both_types']:
            complexity += 2
        
        # More complex if high interleaving
        complexity += features['interleaving_score'] * 3
        
        # More complex if variable amounts
        complexity += min(features['deposit_amount_std'] / max(features['avg_deposit_amount'], 1), 1)
        complexity += min(features['withdrawal_amount_std'] / max(features['avg_withdrawal_amount'], 1), 1)
        
        # More complex if irregular timing
        complexity += (1 - features['time_consistency_score']) * 2
        
        return complexity
    
    def _cluster_patterns(self, 
                         features_df: pd.DataFrame,
                         method: str = 'kmeans',
                         n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """
        Cluster patterns using specified method.
        """
        # Prepare features for clustering
        feature_cols = [col for col in features_df.columns if col != 'address']
        X = features_df[feature_cols].fillna(0)

        # --- FIX: Filter out columns with zero variance before scaling ---
        non_constant_cols = X.columns[X.std() > 0]
        if len(non_constant_cols) == 0:
            logger.warning("All features have zero variance. Skipping clustering.")
            return {'method': 'none', 'n_clusters': 1, 'labels': np.zeros(len(X)), 'metrics': {}}
        
        X_filtered = X[non_constant_cols]

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_filtered)
        
        # Reduce dimensionality if needed
        if X_scaled.shape[1] > 10:
            pca = PCA(n_components=10)
            X_scaled = pca.fit_transform(X_scaled)
            explained_variance = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA reduced to 10 components, explaining {explained_variance:.2%} variance")
        
        # Determine optimal cluster count if not specified
        if n_clusters is None:
            n_clusters = self._find_optimal_clusters(X_scaled, method)
            logger.info(f"Auto-determined optimal clusters: {n_clusters}")
        
        # Perform clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'dbscan':
            eps = self._find_optimal_eps(X_scaled)
            clusterer = DBSCAN(eps=eps, min_samples=5)
        elif method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        labels = clusterer.fit_predict(X_scaled)
        
        # Calculate metrics
        metrics = {}
        if len(set(labels)) > 1:  # Need at least 2 clusters for metrics
            metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X_scaled, labels)
        else:
            metrics['silhouette_score'] = 0
            metrics['davies_bouldin_score'] = 0
        
        return {
            'method': method,
            'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),  # Exclude noise cluster
            'labels': labels,
            'metrics': metrics,
            'cluster_sizes': pd.Series(labels).value_counts().to_dict()
        }
    
    def _find_optimal_clusters(self, X: np.ndarray, method: str) -> int:
        """Find optimal number of clusters using elbow method."""
        if method != 'kmeans':
            return 5  # Default for non-kmeans methods
        
        max_clusters = min(10, len(X) // 10)
        scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                scores.append(score)
            else:
                scores.append(0)
        
        # Find elbow point
        if scores:
            optimal_k = scores.index(max(scores)) + 2
        else:
            optimal_k = 3
        
        return optimal_k
    
    def _find_optimal_eps(self, X: np.ndarray) -> float:
        """Find optimal eps parameter for DBSCAN."""
        from sklearn.neighbors import NearestNeighbors
        
        # Calculate k-distance graph
        k = min(5, len(X) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Sort distances
        distances = np.sort(distances[:, k-1], axis=0)
        
        # Find knee point (simple heuristic)
        knee_idx = int(len(distances) * 0.9)
        eps = distances[knee_idx]
        
        return eps
    
    def _profile_clusters(self, 
                         features_df: pd.DataFrame,
                         labels: np.ndarray) -> Dict[int, Dict]:
        """
        Create detailed profiles for each cluster.
        """
        features_df['cluster'] = labels
        profiles = {}
        
        for cluster_id in features_df['cluster'].unique():
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            
            profile = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'addresses': cluster_data['address'].tolist()[:10],  # Sample addresses
                
                # Volume characteristics
                'avg_deposit_volume': float(cluster_data['total_deposit_volume'].mean()),
                'avg_withdrawal_volume': float(cluster_data['total_withdrawal_volume'].mean()),
                'volume_ratio': float(cluster_data['volume_ratio'].mean()),
                
                # Frequency characteristics
                'avg_deposits': float(cluster_data['deposit_count'].mean()),
                'avg_withdrawals': float(cluster_data['withdrawal_count'].mean()),
                
                # Temporal characteristics
                'avg_deposit_interval': float(cluster_data['avg_deposit_interval_hours'].mean()),
                'avg_withdrawal_interval': float(cluster_data['avg_withdrawal_interval_hours'].mean()),
                
                # Behavioral characteristics
                'round_amount_usage': float(cluster_data['uses_round_amounts'].mean()),
                'time_consistency': float(cluster_data['time_consistency_score'].mean()),
                'interleaving_score': float(cluster_data['interleaving_score'].mean()),
                'pattern_complexity': float(cluster_data['pattern_complexity'].mean()),
                
                # Risk indicators
                'risk_score': self._calculate_cluster_risk(cluster_data),
                'pattern_type': self._classify_pattern_type(cluster_data)
            }
            
            profiles[cluster_id] = profile
        
        return profiles
    
    def _calculate_cluster_risk(self, cluster_data: pd.DataFrame) -> float:
        """Calculate risk score for a cluster."""
        risk = 0
        
        # High volume imbalance
        volume_imbalance = abs(1 - cluster_data['volume_ratio'].mean())
        risk += min(volume_imbalance * 0.3, 0.3)
        
        # Low time consistency
        risk += (1 - cluster_data['time_consistency_score'].mean()) * 0.2
        
        # High interleaving
        risk += cluster_data['interleaving_score'].mean() * 0.2
        
        # High complexity
        risk += min(cluster_data['pattern_complexity'].mean() / 10, 0.3)
        
        return min(risk, 1.0)
    
    def _classify_pattern_type(self, cluster_data: pd.DataFrame) -> str:
        """Classify the dominant pattern type in a cluster."""
        
        # Check various pattern indicators
        has_both = cluster_data['has_both_types'].mean() > 0.5
        deposit_dominant = cluster_data['deposit_count'].mean() > cluster_data['withdrawal_count'].mean() * 2
        withdrawal_dominant = cluster_data['withdrawal_count'].mean() > cluster_data['deposit_count'].mean() * 2
        high_interleaving = cluster_data['interleaving_score'].mean() > 0.5
        uses_round = cluster_data['uses_round_amounts'].mean() > 0.7
        consistent_timing = cluster_data['time_consistency_score'].mean() > 0.7
        
        if has_both and high_interleaving:
            return "active_mixer"
        elif deposit_dominant:
            return "deposit_heavy"
        elif withdrawal_dominant:
            return "withdrawal_heavy"
        elif consistent_timing and uses_round:
            return "systematic_user"
        elif not consistent_timing:
            return "irregular_user"
        else:
            return "balanced_user"
    
    def _identify_anomalous_patterns(self,
                                    features_df: pd.DataFrame,
                                    labels: np.ndarray) -> List[Dict]:
        """Identify anomalous patterns within clusters."""
        anomalies = []
        features_df['cluster'] = labels
        
        # For each cluster, find outliers
        for cluster_id in features_df['cluster'].unique():
            if cluster_id == -1:  # Noise points are already anomalies
                cluster_data = features_df[features_df['cluster'] == -1]
                for _, row in cluster_data.iterrows():
                    anomalies.append({
                        'address': row['address'],
                        'type': 'noise_point',
                        'cluster': -1,
                        'severity': 'high',
                        'reason': 'Does not fit any cluster pattern'
                    })
                continue
            
            cluster_data = features_df[features_df['cluster'] == cluster_id]
            
            # Calculate z-scores for numeric columns
            numeric_cols = cluster_data.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col not in ['cluster']]
            
            for col in numeric_cols:
                mean = cluster_data[col].mean()
                std = cluster_data[col].std()
                
                if std > 0:
                    z_scores = abs((cluster_data[col] - mean) / std)
                    outliers = cluster_data[z_scores > 3]
                    
                    for _, row in outliers.iterrows():
                        anomalies.append({
                            'address': row['address'],
                            'type': 'statistical_outlier',
                            'cluster': int(cluster_id),
                            'severity': 'medium',
                            'reason': f'Unusual {col}: {row[col]:.2f} (cluster mean: {mean:.2f})'
                        })
        
        # Remove duplicates
        seen = set()
        unique_anomalies = []
        for anomaly in anomalies:
            key = (anomaly['address'], anomaly['type'])
            if key not in seen:
                seen.add(key)
                unique_anomalies.append(anomaly)
        
        return unique_anomalies[:20]  # Return top 20 anomalies
    
    def _generate_pattern_insights(self,
                                  features_df: pd.DataFrame,
                                  clustering_results: Dict,
                                  cluster_profiles: Dict) -> Dict[str, Any]:
        """Generate actionable insights from pattern analysis."""
        insights = {
            'summary': {},
            'key_findings': [],
            'risk_indicators': [],
            'recommendations': []
        }
        
        # Summary statistics
        insights['summary'] = {
            'total_patterns_identified': clustering_results['n_clusters'],
            'clustering_quality': clustering_results['metrics'].get('silhouette_score', 0),
            'addresses_analyzed': len(features_df),
            'mixed_activity_addresses': int((features_df['has_both_types'] == 1).sum()),
            'deposit_only_addresses': int((features_df['withdrawal_count'] == 0).sum()),
            'withdrawal_only_addresses': int((features_df['deposit_count'] == 0).sum())
        }
        
        # Key findings based on clusters
        for cluster_id, profile in cluster_profiles.items():
            if profile['risk_score'] > 0.7:
                insights['risk_indicators'].append({
                    'cluster': cluster_id,
                    'risk_score': profile['risk_score'],
                    'pattern_type': profile['pattern_type'],
                    'size': profile['size'],
                    'reason': 'High-risk behavioral pattern detected'
                })
            
            # Pattern-specific findings
            if profile['pattern_type'] == 'active_mixer':
                insights['key_findings'].append(
                    f"Cluster {cluster_id}: Active mixing pattern with {profile['size']} addresses"
                )
            elif profile['interleaving_score'] > 0.7:
                insights['key_findings'].append(
                    f"Cluster {cluster_id}: High deposit-withdrawal interleaving pattern"
                )
            elif profile['round_amount_usage'] > 0.8:
                insights['key_findings'].append(
                    f"Cluster {cluster_id}: Systematic use of round amounts (privacy-conscious)"
                )
        
        # Recommendations
        if insights['summary']['clustering_quality'] < 0.3:
            insights['recommendations'].append(
                "Consider alternative clustering parameters for better pattern separation"
            )
        
        high_risk_clusters = len([p for p in cluster_profiles.values() if p['risk_score'] > 0.7])
        if high_risk_clusters > 0:
            insights['recommendations'].append(
                f"Priority investigation recommended for {high_risk_clusters} high-risk pattern clusters"
            )
        
        active_mixers = len([p for p in cluster_profiles.values() if p['pattern_type'] == 'active_mixer'])
        if active_mixers > 0:
            insights['recommendations'].append(
                f"Enhanced monitoring suggested for {active_mixers} active mixer clusters"
            )
        
        return insights
    
    def _calculate_pattern_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall statistics for deposit/withdrawal patterns."""
        deposits = data[data['tx_type'] == 'deposit']
        withdrawals = data[data['tx_type'] == 'withdrawal']
        
        return {
            'total_deposits': len(deposits),
            'total_withdrawals': len(withdrawals),
            'unique_depositors': deposits['from_addr'].nunique(),
            'unique_withdrawers': withdrawals['to_addr'].nunique(),
            'deposit_volume': float(deposits['value_eth'].sum()),
            'withdrawal_volume': float(withdrawals['value_eth'].sum()),
            'avg_deposit_amount': float(deposits['value_eth'].mean()) if len(deposits) > 0 else 0,
            'avg_withdrawal_amount': float(withdrawals['value_eth'].mean()) if len(withdrawals) > 0 else 0,
            'common_amounts': deposits['value_eth'].value_counts().head(5).to_dict() if len(deposits) > 0 else {}
        }
    
    def _update_database_with_patterns(self, 
                                      features_df: pd.DataFrame,
                                      labels: np.ndarray):
        """
        Update database with pattern cluster assignments and detailed features.
        """
        if features_df.empty:
            return

        logger.info("Updating database with detailed deposit/withdrawal patterns...")
        features_df['pattern_cluster_id'] = labels
        
        # Add risk score from cluster profiles
        risk_scores = features_df['pattern_cluster_id'].map(
            lambda cid: self.cluster_profiles.get(cid, {}).get('risk_score', 0.0) if cid != -1 else 1.0
        )
        features_df['pattern_risk_score'] = risk_scores

        # Add pattern type from cluster profiles
        pattern_types = features_df['pattern_cluster_id'].map(
            lambda cid: self.cluster_profiles.get(cid, {}).get('pattern_type', 'unknown') if cid != -1 else 'outlier'
        )
        features_df['pattern_type'] = pattern_types

        with self.database.transaction():
            # Prepare records for bulk insert/update
            records = features_df.to_dict('records')
            
            for record in records:
                # Ensure all feature keys exist for the query
                all_feature_keys = [
                    'address', 'pattern_cluster_id', 'pattern_risk_score', 'pattern_type', 'total_deposit_volume',
                    'total_withdrawal_volume', 'volume_ratio', 'avg_deposit_amount',
                    'avg_withdrawal_amount', 'deposit_count', 'withdrawal_count',
                    'deposit_withdrawal_ratio', 'avg_deposit_interval_hours',
                    'avg_withdrawal_interval_hours', 'min_deposit_withdrawal_gap_hours',
                    'interleaving_score', 'uses_round_amounts', 'time_consistency_score', 'pattern_complexity'
                ]
                # Prepare a tuple with all values, using defaults for any missing keys
                record_values = []
                for key in all_feature_keys:
                    value = record.get(key)
                    # Handle NaN values and convert numpy types to native Python types
                    if pd.isna(value):
                        value = 'unknown' if key == 'pattern_type' else 0
                    elif isinstance(value, (np.integer, np.int64)):
                        value = int(value)
                    elif isinstance(value, (np.floating, np.float64)):
                        value = float(value)
                    elif isinstance(value, np.bool_):
                        value = bool(value)
                    record_values.append(value)
                
                # --- FIX: Construct the SQL string safely before execution ---
                cols_str = ', '.join(all_feature_keys)
                placeholders_str = ', '.join(['?'] * len(all_feature_keys))
                updates_str = ', '.join([f"{key} = excluded.{key}" for key in all_feature_keys if key != 'address'])
                
                # Use INSERT OR REPLACE (or ON CONFLICT DO UPDATE for DuckDB) for idempotency
                self.database.execute(f"""
                    INSERT INTO deposit_withdrawal_patterns ({cols_str})
                    VALUES ({placeholders_str})
                    ON CONFLICT(address) DO UPDATE SET
                        {updates_str}
                """, tuple(record_values))

        # +++ NEW: Store risk scores in the central risk_components table +++
        # FIX: This loop is now inside the main transaction block, so we don't start a new one.
        risk_components_stored = 0
        for _, row in features_df.iterrows():
            if row.get('pattern_risk_score', 0.0) > 0.1:
                self.database.store_component_risk(
                    address=row['address'], component_type='DEPOSIT_WITHDRAWAL_ANOMALY', risk_score=row['pattern_risk_score'],
                    confidence=0.7, evidence={'pattern_type': row['pattern_type'], 'cluster_id': int(row['pattern_cluster_id'])},
                    source_analysis='deposit_withdrawal_pattern_analyzer'
                )
                risk_components_stored += 1
        if risk_components_stored > 0:
            logger.info(f"Stored {risk_components_stored} behavioral pattern risk components.")

        logger.info(f"Updated {len(features_df)} addresses with detailed pattern features.")
    
    def generate_pattern_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive report of deposit/withdrawal patterns.
        """
        if self.cluster_profiles is None:
            return "No pattern analysis results available. Run analyze_patterns() first."
        
        report = []
        report.append("# DEPOSIT/WITHDRAWAL PATTERN ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- Patterns Identified: {len(self.cluster_profiles)}")
        report.append(f"- Addresses Analyzed: {len(self.pattern_features)}")
        report.append("")
        
        # Cluster Details
        report.append("## Pattern Clusters")
        report.append("")
        
        for cluster_id, profile in sorted(self.cluster_profiles.items(), 
                                         key=lambda x: x[1]['risk_score'], 
                                         reverse=True):
            report.append(f"### Cluster {cluster_id}: {profile['pattern_type'].upper()}")
            report.append(f"- Size: {profile['size']} addresses")
            report.append(f"- Risk Score: {profile['risk_score']:.2f}")
            report.append(f"- Avg Deposits: {profile['avg_deposits']:.1f}")
            report.append(f"- Avg Withdrawals: {profile['avg_withdrawals']:.1f}")
            report.append(f"- Volume Ratio: {profile['volume_ratio']:.2f}")
            report.append(f"- Interleaving Score: {profile['interleaving_score']:.2f}")
            report.append(f"- Pattern Complexity: {profile['pattern_complexity']:.2f}")
            report.append("")
        
        # Risk Assessment
        report.append("## Risk Assessment")
        high_risk = [p for p in self.cluster_profiles.values() if p['risk_score'] > 0.7]
        medium_risk = [p for p in self.cluster_profiles.values() if 0.3 < p['risk_score'] <= 0.7]
        low_risk = [p for p in self.cluster_profiles.values() if p['risk_score'] <= 0.3]
        
        report.append(f"- High Risk Clusters: {len(high_risk)}")
        report.append(f"- Medium Risk Clusters: {len(medium_risk)}")
        report.append(f"- Low Risk Clusters: {len(low_risk)}")
        report.append("")
        
        # Pattern Type Distribution
        report.append("## Pattern Type Distribution")
        pattern_types = {}
        for profile in self.cluster_profiles.values():
            pattern_type = profile['pattern_type']
            pattern_types[pattern_type] = pattern_types.get(pattern_type, 0) + profile['size']
        
        for ptype, count in sorted(pattern_types.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {ptype}: {count} addresses")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("1. Prioritize investigation of high-risk pattern clusters")
        report.append("2. Monitor active mixer patterns for suspicious activity")
        report.append("3. Analyze temporal consistency for systematic users")
        report.append("4. Cross-reference pattern clusters with known entities")
        report.append("")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Pattern report saved to {output_path}")
        
        return report_text


def integrate_pattern_analysis(database: DatabaseEngine,
                              min_transactions: int = 5,
                              clustering_method: str = 'kmeans') -> Dict[str, Any]:
    """
    Integration function for use in the main analysis pipeline.
    """
    analyzer = DepositWithdrawalPatternAnalyzer(database)
    results = analyzer.analyze_patterns(
        min_transactions=min_transactions,
        clustering_method=clustering_method
    )
    
    if results.get('success'):
        # Generate report
        report = analyzer.generate_pattern_report()
        results['report'] = report
    
    return results
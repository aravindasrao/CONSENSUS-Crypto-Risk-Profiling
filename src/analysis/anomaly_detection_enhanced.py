# src/analysis/anomaly_detection_enhanced.py
"""
Fixed Enhanced anomaly detection with database storage and comprehensive analytics
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_anomaly_detection

logger = logging.getLogger(__name__)

class EnhancedAnomalyDetector:
    """
    Enhanced anomaly detection with database storage and analytics
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.anomalies_found = []
        
        logger.info("Enhanced anomaly detector initialized with storage")
    
    def detect_all_anomalies_with_storage(self, test_mode: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive anomaly detection with database storage
        """
        logger.info("Starting comprehensive anomaly detection with storage...")
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        start_time = datetime.now()
        
        results = {
            'session_id': session_id,
            'isolation_forest_anomalies': [],
            'svm_anomalies': [],
            'statistical_anomalies': [],
            'combined_anomalies': [],
            'total_anomalies_found': 0,
            'analysis_summary': {}
        }
        
        try:
            # Get addresses for analysis - FIXED: Use correct column names
            addresses = self._get_addresses_for_analysis(test_mode=test_mode)
            
            if len(addresses) == 0:
                logger.warning("No addresses found for anomaly detection")
                return results
            
            logger.info(f"Analyzing {len(addresses)} addresses for anomalies")
            
            # Extract features - FIXED: Use correct column names
            features_df = self._extract_features(addresses)
            
            if features_df.empty:
                logger.warning("No features extracted for anomaly detection")
                return results
            
            # Run different anomaly detection methods
            results['isolation_forest_anomalies'] = self._run_isolation_forest_enhanced(features_df, session_id)
            results['svm_anomalies'] = self._run_one_class_svm_enhanced(features_df, session_id)
            results['statistical_anomalies'] = self._run_statistical_anomaly_detection_enhanced(features_df, session_id)
            
            # Combine results
            results['combined_anomalies'] = self._combine_anomaly_results(
                results['isolation_forest_anomalies'],
                results['svm_anomalies'], 
                results['statistical_anomalies']
            )
            
            # Summary statistics
            results['total_anomalies_found'] = len(results['combined_anomalies'])
            results['analysis_summary'] = {
                'addresses_analyzed': len(addresses),
                'isolation_forest_anomalies': len(results['isolation_forest_anomalies']),
                'svm_anomalies': len(results['svm_anomalies']),
                'statistical_anomalies': len(results['statistical_anomalies']),
                'combined_anomalies': len(results['combined_anomalies']),
                'feature_count': len(self.feature_names)
            }
            
            # Store session info
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            session_data = {
                'session_id': session_id,
                'start_time': start_time,
                'end_time': end_time,
                'addresses_analyzed': len(addresses),
                'total_anomalies': results['total_anomalies_found'],
                'isolation_forest_count': len(results['isolation_forest_anomalies']),
                'svm_count': len(results['svm_anomalies']),
                'statistical_count': len(results['statistical_anomalies']),
                'combined_count': len(results['combined_anomalies']),
                'feature_count': len(self.feature_names),
                'processing_time_seconds': processing_time
            }
            
            self._store_session_data(session_data)
            
            logger.info(f"Enhanced anomaly detection completed: {results['total_anomalies_found']} anomalies found in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in enhanced anomaly detection: {e}")
            results['error'] = str(e)
        
        return results
    
    def _get_addresses_for_analysis(self, test_mode: bool = False, limit: int = None) -> List[str]:
        """Get addresses with sufficient transaction history for analysis - FIXED column names"""
        try:
            if limit is None:
                limit = 200 if test_mode else 200000 # Analyze up to 200k addresses
            addresses_df = self.database.fetch_df(f"""
                SELECT address
                FROM addresses
                WHERE total_transaction_count >= 10
                ORDER BY total_transaction_count DESC
                LIMIT ?
            """, (limit,))
            
            if not addresses_df.empty:
                return addresses_df['address'].tolist()
            else:
                logger.warning("No addresses found with sufficient transaction history")
                return []
                
        except Exception as e:
            logger.error(f"Error getting addresses for analysis: {e}")
            return []
    
    def _extract_features(self, addresses: List[str]) -> pd.DataFrame:
        """Extracts pre-calculated features from the wide 'addresses' table."""
        try:
            # This list should match the features defined in duckdb_schema.py
            feature_columns = [
                'tx_frequency_daily', 'tx_frequency_weekly', 'active_days_count', 'active_hours_count',
                'peak_activity_hour', 'weekend_activity_ratio', 'night_activity_ratio', 'burst_activity_count',
                'temporal_regularity_score', 'avg_time_gap_hours', 'time_gap_variance', 'max_inactive_period_days',
                'activity_span_days', 'temporal_entropy', 'circadian_rhythm_strength', 'temporal_clustering_score',
                'total_volume_eth', 'incoming_volume_eth', 'outgoing_volume_eth', 'net_flow_eth',
                'avg_transaction_value', 'median_transaction_value', 'max_single_transaction', 'min_single_transaction',
                'value_variance', 'value_std_deviation', 'coefficient_of_variation', 'large_tx_ratio',
                'small_tx_ratio', 'round_number_ratio', 'gini_coefficient', 'wealth_accumulation_rate',
                'value_distribution_skewness', 'economic_diversity_score',
                'degree_centrality', 'in_degree_centrality', 'out_degree_centrality', 'betweenness_centrality',
                'closeness_centrality', 'eigenvector_centrality', 'pagerank_score', 'clustering_coefficient',
                'unique_counterparties', 'network_reach_2hop', 'hub_score', 'authority_score',
                'local_efficiency', 'bridge_score', 'network_influence_score',
                'total_transaction_count', 'incoming_tx_count', 'outgoing_tx_count', 'tx_direction_ratio',
                'self_transaction_ratio', 'zero_value_tx_ratio', 'contract_interaction_ratio', 'unique_methods_count',
                'method_diversity_entropy', 'repeat_counterparty_ratio', 'avg_gas_per_tx', 'gas_usage_variance',
                'gas_price_variance', 'gas_optimization_score', 'automation_likelihood', 'interaction_complexity',
                'behavioral_consistency', 'pattern_entropy', 'counterparty_loyalty', 'operational_sophistication',
                'unusual_timing_score', 'high_frequency_burst_score', 'round_amount_preference',
                'gas_price_anomaly_score', 'volume_spike_indicator', 'mixing_behavior_score',
                'privacy_seeking_score', 'evasion_pattern_score', 'laundering_risk_indicator',
                'anonymity_behavior_score', 'suspicious_timing_pattern', 'composite_risk_score',
                'avg_gas_limit', 'avg_gas_used', 'gas_efficiency_ratio', 'gas_price_strategy',
                'contract_deployment_count', 'contract_call_frequency', 'advanced_function_usage',
                'operational_complexity', 'gas_optimization_trend', 'block_timing_consistency',
                'priority_fee_behavior', 'batch_transaction_score', 'mev_resistance_score',
                'technical_sophistication', 'infrastructure_usage',
                'market_timing_correlation', 'network_congestion_behavior', 'peak_hours_preference',
                'off_peak_activity', 'weekday_vs_weekend_ratio', 'market_volatility_response',
                'seasonal_activity_pattern', 'economic_event_sensitivity', 'bear_market_behavior',
                'bull_market_behavior', 'crisis_period_activity', 'fee_market_adaptation',
                'ecosystem_participation', 'institutional_timing', 'retail_behavior_score'
            ]
            self.feature_names = feature_columns

            columns_to_select = ", ".join([f'"{col}"' for col in feature_columns] + ['address'])
            placeholders = ','.join(['?'] * len(addresses))
            query = f"""
                SELECT {columns_to_select}
                FROM addresses
                WHERE address IN ({placeholders})
            """
            features_df = self.database.fetch_df(query, tuple(addresses))
            
            logger.info(f"Extracted {len(self.feature_names)} features for {len(features_df)} addresses")
            
            return features_df

        except Exception as e:
            logger.error(f"Error extracting features from wide table: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _run_isolation_forest_enhanced(self, features_df: pd.DataFrame, session_id: str) -> List[Dict[str, Any]]:
        """Run Isolation Forest with database storage"""
        try:
            logger.info("Running Isolation Forest anomaly detection...")
            
            # Prepare feature matrix
            feature_columns = [col for col in features_df.columns if col != 'address']
            X = features_df[feature_columns].fillna(0)
            
            # --- FIX: Sanitize data to handle infinity values ---
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Impute NaNs that might have been created from infinity
            X.fillna(X.median(), inplace=True)

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train Isolation Forest
            iso_forest = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Fit and predict
            anomaly_labels = iso_forest.fit_predict(X_scaled)
            # decision_function: negative for outliers. We invert it for easier interpretation.
            raw_scores = -iso_forest.decision_function(X_scaled)
            
            # Store model
            self.models['isolation_forest'] = iso_forest
            
            # Extract anomalies
            anomalies = []
            anomaly_records = []
            
            for i, (label, raw_score) in enumerate(zip(anomaly_labels, raw_scores)):
                if label == -1:  # Anomaly
                    address = str(features_df.iloc[i]['address'])
                    
                    # FIX: Normalize the score to a 0-1 range using a sigmoid-like function.
                    # A score of 0 (the threshold) will be 0.5. Higher scores approach 1.
                    normalized_score = 1 / (1 + np.exp(-raw_score * 5))
                    anomaly_score = float(np.clip(normalized_score, 0, 1))
                    confidence = float(np.clip(normalized_score, 0, 1)) # Confidence can be the score itself
                    
                    anomaly = {
                        'type': 'isolation_forest_anomaly', # FIX: Add type for better logging
                        'address': address,
                        'anomaly_score': anomaly_score,
                        'method': 'isolation_forest',
                        'confidence': confidence,
                        'rank': len(anomalies) + 1
                    }
                    
                    anomalies.append(anomaly)
                    
                    # Prepare for database storage
                    anomaly_records.append({
                        'address': address,
                        'method': 'isolation_forest',
                        'anomaly_score': anomaly_score,
                        'confidence': confidence,
                        'rank': anomaly['rank'],
                        'feature_contributing': 'multiple'
                    })
                    
                    log_anomaly_detection(anomaly)
            
            # Store anomaly detections in database
            if anomaly_records:
                for record in anomaly_records:
                    self.database.execute("""
                        INSERT INTO anomaly_detections 
                        (address, method, anomaly_score, confidence, rank, feature_contributing)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        record['address'],
                        record['method'],
                        record['anomaly_score'],
                        record['confidence'],
                        record['rank'],
                        record['feature_contributing']
                    ))
            
            # Store features for ALL addresses (not just anomalies) for analytics
            self._store_features_for_session(features_df, session_id, X_scaled, feature_columns)
            
            logger.info(f"Isolation Forest: {len(anomalies)} anomalies detected and stored")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest detection: {e}")
            return []
    
    def _run_one_class_svm_enhanced(self, features_df: pd.DataFrame, session_id: str) -> List[Dict[str, Any]]:
        """Run One-Class SVM with database storage"""
        try:
            logger.info("Running One-Class SVM anomaly detection...")
            
            # Prepare feature matrix
            feature_columns = [col for col in features_df.columns if col != 'address']
            X = features_df[feature_columns].fillna(0)
            
            # --- FIX: Sanitize data to handle infinity values ---
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Impute NaNs that might have been created from infinity
            X.fillna(X.median(), inplace=True)

            # Standardize features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train One-Class SVM
            svm_model = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1
            )
            
            # Fit and predict
            anomaly_labels = svm_model.fit_predict(X_scaled)
            # decision_function: negative for outliers. We invert it.
            raw_scores = -svm_model.decision_function(X_scaled)
            
            # Store model
            self.models['one_class_svm'] = svm_model
            
            # Extract anomalies
            anomalies = []
            anomaly_records = []
            
            for i, (label, raw_score) in enumerate(zip(anomaly_labels, raw_scores)):
                if label == -1:  # Anomaly
                    address = str(features_df.iloc[i]['address'])
                    
                    # FIX: Normalize the score to a 0-1 range using a sigmoid-like function
                    normalized_score = 1 / (1 + np.exp(-raw_score * 5))
                    anomaly_score = float(np.clip(normalized_score, 0, 1))
                    confidence = float(np.clip(normalized_score, 0, 1))
                    
                    anomaly = {
                        'type': 'one_class_svm_anomaly', # FIX: Add type for better logging
                        'address': address,
                        'anomaly_score': anomaly_score,
                        'method': 'one_class_svm',
                        'confidence': confidence,
                        'rank': len(anomalies) + 1
                    }
                    
                    anomalies.append(anomaly)
                    
                    # Prepare for database storage
                    anomaly_records.append({
                        'address': address,
                        'method': 'one_class_svm',
                        'anomaly_score': anomaly_score,
                        'confidence': confidence,
                        'rank': anomaly['rank'],
                        'feature_contributing': 'multiple'
                    })
                    
                    log_anomaly_detection(anomaly)
            
            # Store anomaly detections in database
            if anomaly_records:
                for record in anomaly_records:
                    self.database.execute("""
                        INSERT INTO anomaly_detections 
                        (address, method, anomaly_score, confidence, rank, feature_contributing)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        record['address'],
                        record['method'],
                        record['anomaly_score'],
                        record['confidence'],
                        record['rank'],
                        record['feature_contributing']
                    ))
            
            # Note: Features are already stored by Isolation Forest method
            # Only store if this is the first method being called
            if 'isolation_forest' not in self.models:
                self._store_features_for_session(features_df, session_id, X_scaled, feature_columns)
            
            logger.info(f"One-Class SVM: {len(anomalies)} anomalies detected and stored")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error in One-Class SVM detection: {e}")
            return []
    
    def _run_statistical_anomaly_detection_enhanced(self, features_df: pd.DataFrame, session_id: str) -> List[Dict[str, Any]]:
        """Run statistical anomaly detection with database storage"""
        try:
            logger.info("Running statistical anomaly detection...")
            
            anomalies = []
            anomaly_records = []
            
            # Get feature columns
            feature_columns = [col for col in features_df.columns if col != 'address']
            
            # --- FIX: Sanitize data to handle infinity values before statistical analysis ---
            features_df_clean = features_df.copy()
            features_df_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
            features_df_clean.fillna(features_df_clean.median(numeric_only=True), inplace=True)

            for feature in feature_columns:
                if feature in features_df_clean.columns:
                    values = features_df_clean[feature]
                    
                    if values.std() > 0:
                        z_scores = np.abs((values - values.mean()) / values.std())
                        
                        # Find outliers (Z-score > 3)
                        outlier_indices = np.where(z_scores > 3)[0]
                        
                        for idx in outlier_indices:
                            z_score_val = float(z_scores[idx])
                            
                            # FIX: Normalize the raw Z-score to a 0-1 risk score.
                            # A Z-score of 3 (our threshold) will be 0.3.
                            # A Z-score of 10 or more will be capped at 1.0.
                            normalized_score = min(1.0, z_score_val / 10.0)

                            anomaly = {
                                'type': 'statistical_anomaly',
                                'address': str(features_df_clean.iloc[idx]['address']),
                                'anomaly_score': normalized_score,
                                'method': 'statistical',
                                'feature': feature,
                                'confidence': min(z_score_val / 3.0, 1.0), # Confidence is how far past the threshold it is
                                'rank': int(idx + 1)
                            }
                            anomalies.append(anomaly)
                            
                            # Prepare for database storage
                            anomaly_records.append({
                                'address': anomaly['address'],
                                'method': 'statistical',
                                'anomaly_score': anomaly['anomaly_score'],
                                'confidence': anomaly['confidence'],
                                'rank': anomaly['rank'],
                                'feature_contributing': feature
                            })
                            
                            log_anomaly_detection(anomaly)
            
            # Remove duplicates and keep highest scoring
            unique_anomalies = {}
            unique_records = {}
            
            for i, anomaly in enumerate(anomalies):
                addr = anomaly['address']
                if addr not in unique_anomalies or anomaly['anomaly_score'] > unique_anomalies[addr]['anomaly_score']:
                    unique_anomalies[addr] = anomaly
                    if i < len(anomaly_records):
                        unique_records[addr] = anomaly_records[i]
            
            result_anomalies = list(unique_anomalies.values())
            result_records = list(unique_records.values())
            
            # Store anomaly detections in database
            if result_records:
                for record in result_records:
                    self.database.execute("""
                        INSERT INTO anomaly_detections 
                        (address, method, anomaly_score, confidence, rank, feature_contributing)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        record['address'],
                        record['method'],
                        record['anomaly_score'],
                        record['confidence'],
                        record['rank'],
                        record['feature_contributing']
                    ))
            
            # Note: Features are already stored by previous methods
            # Only store if this is the first method being called
            if not self.models:
                feature_columns = [col for col in features_df_clean.columns if col != 'address']
                X = features_df_clean[feature_columns].fillna(0)
                X_scaled = self.scaler.fit_transform(X)
                self._store_features_for_session(features_df_clean, session_id, X_scaled, feature_columns)
            
            logger.info(f"Statistical detection: {len(result_anomalies)} anomalies detected and stored")
            
            return result_anomalies
            
        except Exception as e:
            logger.error(f"Error in statistical anomaly detection: {e}")
            return []
    
    def _combine_anomaly_results(self, *anomaly_lists) -> List[Dict[str, Any]]:
        """Combine results from different anomaly detection methods"""
        try:
            combined_scores = {}
            
            # Collect all anomalies by address
            for anomaly_list in anomaly_lists:
                for anomaly in anomaly_list:
                    address = anomaly['address']
                    
                    if address not in combined_scores:
                        combined_scores[address] = {
                            'address': address,
                            'methods': [],
                            'total_score': 0.0,
                            'confidence_sum': 0.0,
                            'method_count': 0
                        }
                    
                    combined_scores[address]['methods'].append(anomaly['method'])
                    combined_scores[address]['total_score'] += anomaly.get('anomaly_score', 0)
                    combined_scores[address]['confidence_sum'] += anomaly.get('confidence', 0)
                    combined_scores[address]['method_count'] += 1
            
            # Create combined anomalies
            combined_anomalies = []
            
            for address, data in combined_scores.items():
                # Only include addresses detected by multiple methods or with high confidence
                if data['method_count'] >= 2 or (data['method_count'] >= 1 and data['confidence_sum'] >= 0.8):
                    combined_anomaly = {
                        'address': address,
                        'combined_score': data['total_score'],
                        'method_count': data['method_count'],
                        'methods': data['methods'],
                        'confidence': data['confidence_sum'] / data['method_count']
                    }
                    
                    combined_anomalies.append(combined_anomaly)
            
            # Sort by combined score
            combined_anomalies.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Log combined anomalies
            for anomaly in combined_anomalies:
                log_anomaly_detection({
                    'type': 'combined_anomaly',
                    'address': anomaly['address'],
                    'combined_score': anomaly['combined_score'],
                    'method_count': anomaly['method_count'],
                    'confidence': anomaly['confidence']
                })
            
            logger.info(f"Combined {len(combined_anomalies)} unique anomalous addresses")
            
            return combined_anomalies
            
        except Exception as e:
            logger.error(f"Error combining anomaly results: {e}")
            return []
    
    def _store_session_data(self, session_data: Dict[str, Any]):
        """Store anomaly detection session data"""
        try:
            self.database.execute("""
                INSERT INTO anomaly_sessions 
                (session_id, start_time, end_time, addresses_analyzed, total_anomalies,
                 isolation_forest_count, svm_count, statistical_count, combined_count,
                 feature_count, processing_time_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_data['session_id'],
                session_data['start_time'],
                session_data['end_time'],
                session_data['addresses_analyzed'],
                session_data['total_anomalies'],
                session_data['isolation_forest_count'],
                session_data['svm_count'],
                session_data['statistical_count'],
                session_data['combined_count'],
                session_data['feature_count'],
                session_data['processing_time_seconds']
            ))
            
            logger.info(f"Session data stored: {session_data['session_id']}")
            
        except Exception as e:
            logger.error(f"Error storing session data: {e}")
    
    def _store_features_for_session(self, features_df: pd.DataFrame, session_id: str, X_scaled: np.ndarray, feature_columns: List[str]):
        """Store feature data for the session in anomaly_features table"""
        try:
            logger.info(f"Storing {len(features_df)} address features for session {session_id}")
            
            feature_records = []
            
            for i, row in features_df.iterrows():
                address = str(row['address'])
                
                # Store each feature for this address
                for j, feature_name in enumerate(feature_columns):
                    original_value = float(row[feature_name]) if not pd.isna(row[feature_name]) else 0.0
                    normalized_value = float(X_scaled[i, j])
                    
                    # Calculate z-score for this feature
                    feature_values = features_df[feature_name].fillna(0)
                    if feature_values.std() > 0:
                        z_score = abs((original_value - feature_values.mean()) / feature_values.std())
                        is_outlier = z_score > 3.0
                    else:
                        z_score = 0.0
                        is_outlier = False
                    
                    feature_record = {
                        'session_id': session_id,
                        'address': address,
                        'feature_name': feature_name,
                        'feature_value': original_value,
                        'normalized_value': normalized_value,
                        'z_score': float(z_score),
                        'is_outlier':  bool(is_outlier) # FIXED: Convert numpy.bool to standard bool
                    }
                    
                    feature_records.append(feature_record)
            
            # Batch insert feature records
            for record in feature_records:
                self.database.execute("""
                    INSERT INTO anomaly_features 
                    (session_id, address, feature_name, feature_value, normalized_value, z_score, is_outlier)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    record['session_id'],
                    record['address'],
                    record['feature_name'],
                    record['feature_value'],
                    record['normalized_value'],
                    record['z_score'],
                    record['is_outlier']
                ))
            
            logger.info(f"Stored {len(feature_records)} feature records for session {session_id}")
            
        except Exception as e:
            logger.error(f"Error storing features for session: {e}")
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        try:
            summary = {
                'total_anomalies': len(self.anomalies_found),
                'model_performance': {},
                'feature_importance': {},
                'top_anomalies': []
            }
            
            # Model performance
            for model_name, model in self.models.items():
                if hasattr(model, 'score_samples'):
                    summary['model_performance'][model_name] = 'available'
                else:
                    summary['model_performance'][model_name] = 'basic'
            
            # Top anomalies
            if self.anomalies_found:
                sorted_anomalies = sorted(
                    self.anomalies_found, 
                    key=lambda x: x.get('confidence', 0), 
                    reverse=True
                )
                summary['top_anomalies'] = sorted_anomalies[:10]
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting anomaly summary: {e}")
            return {}


# Factory function
def create_enhanced_anomaly_detector(database: DatabaseEngine = None) -> EnhancedAnomalyDetector:
    """Create enhanced anomaly detector instance"""
    return EnhancedAnomalyDetector(database)
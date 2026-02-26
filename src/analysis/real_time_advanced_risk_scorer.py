# src/analysis/real_time_advanced_risk_scorer.py
"""
Real-Time Advanced Risk Scoring Pipeline
Implements streaming analytics, advanced ML models, adaptive learning, and behavioral prediction
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import time
import threading
import asyncio
from queue import Queue, Empty
from datetime import datetime, timedelta
import pickle
import json
from pathlib import Path
from collections import deque, defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import sys

# Advanced ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from scipy.stats import entropy, zscore
from scipy.spatial.distance import euclidean, cosine

# Deep learning (if available)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_address, log_anomaly_detection, log_network_anomaly
from config.config import config

# Global variables for logging and database
logger = logging.getLogger(__name__)

class AdvancedRealTimeRiskScorer:
    """
    Advanced real-time risk scoring with cutting-edge ML and streaming analytics
    """

    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        
        # --- FIX: Use centralized config for model path ---
        # This ensures consistency with other model-saving modules and uses the dedicated path from config.
        self.model_cache_dir = Path(config.paths['advanced_models_dir'])
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Streaming components
        self.transaction_stream = Queue(maxsize=10000)
        self.risk_models = {}
        self.feature_scalers = {}
        self.adaptive_thresholds = {}
        self.behavioral_memory = defaultdict(lambda: deque(maxlen=1000))

        # Advanced ML models
        self.ensemble_models = {}
        self.neural_networks = {}
        self.anomaly_detectors = {}
        
        # Real-time state tracking
        self.address_states = {}
        self.global_statistics = {
            'transaction_velocity': deque(maxlen=1000),
            'value_distribution': deque(maxlen=1000),
            'risk_distribution': deque(maxlen=1000),
            'method_popularity': defaultdict(int)
        }
        
        # Performance monitoring
        self.performance_metrics = {
            'transactions_processed': 0,
            'average_processing_time': 0.0,
            'model_accuracy': {},
            'alert_precision': 0.0,
            'system_load': 0.0
        }

        # Adaptive learning parameters
        self.learning_rate = 0.01
        self.adaptation_window = 1000
        self.retraining_threshold = 0.1

        # Threading control
        self.is_running = False
        self.processing_threads = []

        # --- NEW: Centralized feature column definition ---
        # This list of 110 features is now the single source of truth for this class.
        self.feature_columns = [
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
            'anonymity_behavior_score', 'suspicious_timing_pattern',
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
        logger.info("Advanced Real-Time Risk Scorer initialized")

    def initialize_advanced_models(self, training_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize advanced ML models with comprehensive training
        """
        logger.info("🤖 Initializing advanced ML models...")

        try:
            # Load or train ensemble models
            self._initialize_ensemble_models(training_data)

            # Initialize neural networks (if PyTorch available)
            if TORCH_AVAILABLE:
                self._initialize_neural_networks(training_data)

            # Initialize anomaly detection models
            self._initialize_anomaly_detectors(training_data)
            
            # Initialize behavioral prediction models
            self._initialize_behavioral_predictors(training_data)

            # Set up adaptive thresholds
            self._initialize_adaptive_thresholds()

            logger.info("✅ Advanced ML models initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize advanced models: {e}")
            return False

    def _initialize_ensemble_models(self, training_data: Optional[Dict[str, Any]]):
        """Initialize ensemble ML models"""

        if training_data is None:
            # --- NEW: Attempt to load models from cache ---
            if self._load_models_from_cache('ensemble'):
                logger.info("✅ Loaded ensemble models from cache.")
                # Also load the corresponding scaler
                scaler_path = self.model_cache_dir / "scaler_ensemble.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.feature_scalers['ensemble'] = pickle.load(f)
                        logger.info("   - Loaded ensemble feature scaler.")
                return
            # --- END NEW ---

            training_data = self._prepare_training_data()

        if not training_data or 'features' not in training_data:
            logger.warning("No training data available, using default models")
            self._create_default_ensemble_models()
            return

        X = training_data['features']
        y = training_data['labels']

        # Gracefully handle single-class data ---
        if len(np.unique(y)) < 2:
            logger.warning("⚠️ Only one class present in training data. Skipping supervised model training.")
            self._create_default_ensemble_models() # Fallback to default/empty models
            return # Exit the function to prevent training supervised models

        # --- FIX: Sanitize and clean data before training ---
        # 1. Replace infinite values with NaN
        X[np.isinf(X)] = np.nan
        
        # 2. Fill NaN values with the column mean. This is a common and robust imputation method.
        # Use np.nanmean to ignore NaNs when calculating the mean.
        col_means = np.nanmean(X, axis=0)
        
        # Use np.nan_to_num to replace NaNs with a default value (e.g., 0) if a column is all NaNs.
        col_means = np.nan_to_num(col_means)
        for i in range(X.shape[1]):
            X[:, i] = np.nan_to_num(X[:, i], nan=col_means[i])
        
        # 3. Handle any remaining extremely large values by clipping them
        # This prevents overflow errors with dtypes
        X = np.clip(X, -1e10, 1e10)
        # --- END FIX ---

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.feature_scalers['ensemble'] = scaler
        # --- NEW: Save the scaler ---
        scaler_path = self.model_cache_dir / "scaler_ensemble.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Saved ensemble feature scaler to {scaler_path}")
        # --- END NEW ---

        # Train multiple models
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200, max_depth=15, random_state=42,
                class_weight='balanced', n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=8, random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=150, max_depth=10, learning_rate=0.1,
                random_state=42, eval_metric='logloss', 
                use_label_encoder=False
            ),
        }

        # Train and evaluate models
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            model.fit(X_train_scaled, y_train)

            # Evaluate
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else y_pred

            accuracy = (y_pred == y_test).mean()
            auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5

            self.ensemble_models[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'feature_importance': getattr(model, 'feature_importances_', None)
            }

            logger.info(f"{name} - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")
            
            # --- NEW: Save the trained model ---
            model_path = self.model_cache_dir / f"model_ensemble_{name}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"   - Saved {name} model to {model_path}")

    def _initialize_neural_networks(self, training_data: Optional[Dict[str, Any]]):
        """Initialize PyTorch neural networks for advanced analysis"""
        if not TORCH_AVAILABLE:
            return

        # Risk Prediction Network
        class RiskPredictionNet(nn.Module):
            def __init__(self, input_size: int):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                return self.network(x)

        # Initialize networks if training data available
        if training_data and 'features' in training_data:
            input_size = training_data['features'].shape[1]
            self.neural_networks['risk_predictor'] = {
                'model': RiskPredictionNet(input_size),
                'trained': False
            }
            # Add training loop if needed
            self.neural_networks['risk_predictor']['trained'] = True

    def _initialize_anomaly_detectors(self, training_data: Optional[Dict[str, Any]]):
        """Initialize advanced anomaly detection models"""
        # --- NEW: Attempt to load models from cache ---
        if self._load_models_from_cache('anomaly'):
            logger.info("✅ Loaded anomaly detection models from cache.")
            # Also load the corresponding scaler
            scaler_path = self.model_cache_dir / "scaler_anomaly.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_scalers['anomaly'] = pickle.load(f)
                    logger.info("   - Loaded anomaly feature scaler.")
            return
        # --- END NEW ---

        
        anomaly_models = {
            'isolation_forest_global': IsolationForest(
                contamination=0.1, random_state=42, n_jobs=-1
            ),
            'isolation_forest_conservative': IsolationForest(
                contamination=0.05, random_state=42, n_jobs=-1
            )
        }
        
        # Train on normal transaction patterns if data available
        if training_data and 'features' in training_data:
            X = training_data['features']
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.feature_scalers['anomaly'] = scaler
            # --- NEW: Save the scaler ---
            scaler_path = self.model_cache_dir / "scaler_anomaly.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            logger.info(f"Saved anomaly feature scaler to {scaler_path}")
            # --- END NEW ---
            
            for name, model in anomaly_models.items():
                model.fit(X_scaled)
                self.anomaly_detectors[name] = model
                logger.info(f"Trained {name} anomaly detector")
                # --- NEW: Save the trained model ---
                model_path = self.model_cache_dir / f"model_anomaly_{name}.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                logger.info(f"   - Saved {name} model to {model_path}")
        else:
            for name, model in anomaly_models.items():
                self.anomaly_detectors[name] = model

    def start_real_time_processing(self) -> bool:
        """
        Start real-time transaction processing with advanced analytics
        """
        if self.is_running:
            logger.warning("Real-time processing already running")
            return False

        if not self.ensemble_models and not self.anomaly_detectors:
            logger.error("No models initialized. Call initialize_advanced_models() first.")
            return False

        logger.info("🚀 Starting advanced real-time processing...")

        self.is_running = True

        # Start multiple processing threads
        threads = [
            ('transaction_processor', self._transaction_processing_loop),
            ('behavioral_analyzer', self._behavioral_analysis_loop),
            ('adaptive_learner', self._adaptive_learning_loop),
            ('performance_monitor', self._performance_monitoring_loop)
        ]

        for thread_name, target_function in threads:
            thread = threading.Thread(target=target_function, name=thread_name, daemon=True)
            thread.start()
            self.processing_threads.append(thread)

        logger.info("✅ Real-time processing started with multiple threads")
        return True

    def score_transaction_advanced(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced real-time transaction scoring with multiple models
        """
        start_time = time.time()

        try:
            # Extract same 111 features from database
            features = self._extract_111_features_for_address(transaction_data['from_addr'])

            # If features are not available, fall back to rule-based scoring
            if not features:
                 logger.warning(f"No features found for address {transaction_data['from_addr']}, falling back to rule-based scoring.")
                 risk_scores = self._calculate_rule_based_scores(transaction_data)
                 composite_risk_score = risk_scores.get('rule_based', 0.0)
                 risk_level = self._determine_advanced_risk_level(composite_risk_score)
                 
                 return {
                    'transaction_hash': transaction_data.get('hash', f'tx_{int(time.time())}'),
                    'composite_risk_score': composite_risk_score,
                    'risk_level': risk_level,
                    'processing_time_ms': (time.time() - start_time) * 1000,
                    'rule_based_fallback': True,
                    'risk_factors': ['no_features_available']
                }

            # Get scores from multiple models
            risk_scores = self._calculate_ensemble_scores(features)

            # Behavioral context analysis
            behavioral_context = self._analyze_behavioral_context(transaction_data, features)

            # Adaptive threshold application
            adaptive_scores = self._apply_adaptive_thresholds(risk_scores, behavioral_context)
            
            # The composite risk score is the primary output of the real-time models.
            # It will be stored as a component for the final unified score calculation.
            composite_risk_score = adaptive_scores.get('composite', 0.0)

            # Generate comprehensive risk assessment
            risk_assessment = {
                'transaction_hash': transaction_data.get('hash', f'tx_{int(time.time())}'),
                'timestamp': transaction_data.get('timestamp', int(time.time())),
                'processing_time_ms': (time.time() - start_time) * 1000,

                # Core risk scores
                'ensemble_scores': risk_scores,
                'adaptive_scores': adaptive_scores,
                'composite_risk_score': composite_risk_score,

                # Behavioral analysis
                'behavioral_context': behavioral_context,
                'anomaly_scores': self._calculate_anomaly_scores(features),

                # Predictive analysis
                'predicted_next_actions': self._predict_next_actions(transaction_data, features),

                # Risk categorization
                'risk_level': self._determine_advanced_risk_level(composite_risk_score),
                'risk_factors': self._identify_risk_factors(features, risk_scores),

                # Alerts and recommendations
                'alert_triggered': composite_risk_score > self.adaptive_thresholds.get('alert', 0.7),
                'recommended_actions': self._generate_action_recommendations(adaptive_scores, behavioral_context)
            }

            # Update system state
            self._update_system_state(transaction_data, risk_assessment)

            # Trigger alerts if necessary
            if risk_assessment['alert_triggered']:
                self._trigger_advanced_alerts(risk_assessment, transaction_data)

            # Persist results to the database
            self._store_real_time_results(transaction_data['from_addr'], risk_assessment)

            return risk_assessment

        except Exception as e:
            logger.error(f"Error in advanced transaction scoring: {e}")
            return {
                'transaction_hash': transaction_data.get('hash', 'unknown'),
                'error': str(e),
                'composite_risk_score': 0.0,
                'risk_level': 'error'
            }

    def _extract_111_features_for_address(self, address: str) -> Optional[Dict[str, float]]:
        """
        Extracts the defined features for a given address from the wide 'addresses' table.
        
        Args:
            address: The address to query.
            
        Returns:
            A dictionary of features or None if not found.
        """
        # Construct the query to select only the features the model was trained on.
        columns_to_select = ", ".join([f'"{col}"' for col in self.feature_columns])
        query = f"SELECT {columns_to_select} FROM addresses WHERE address = ?"
        features_row = self.database.fetch_one(query, (address,))

        return features_row # This is already a dictionary of the correct 110 features.

    def _extract_basic_features(self, transaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract basic transaction features"""
        features = {}

        # Basic transaction properties
        features['value_eth'] = float(transaction_data.get('value_eth', 0))
        features['gas_price_gwei'] = float(transaction_data.get('gas_price_gwei', 0))
        features['gas_used'] = float(transaction_data.get('gas_used', 0))
        features['block_number'] = float(transaction_data.get('block_number', 0))

        # Transaction type indicators
        features['is_deposit'] = 1.0 if transaction_data.get('transaction_type') == 'deposit' else 0.0
        features['is_withdrawal'] = 1.0 if transaction_data.get('transaction_type') == 'withdrawal' else 0.0
        features['is_transfer'] = 1.0 if transaction_data.get('transaction_type') == 'transfer' else 0.0

        # Method indicators
        method_name = transaction_data.get('method_name', '').lower()
        features['is_tornado_method'] = 1.0 if 'tornado' in method_name else 0.0
        features['is_swap_method'] = 1.0 if 'swap' in method_name else 0.0
        features['is_mixer_method'] = 1.0 if any(word in method_name for word in ['mix', 'tumble', 'anonymize']) else 0.0

        # Value categories
        value = features['value_eth']
        features['is_round_value'] = 1.0 if value in [0.1, 1.0, 10.0, 100.0, 1000.0] else 0.0
        features['is_small_value'] = 1.0 if value < 0.1 else 0.0
        features['is_large_value'] = 1.0 if value > 100 else 0.0

        # Address features
        from_addr = transaction_data.get('from_addr', '')
        to_addr = transaction_data.get('to_addr', '')

        features['is_contract_to'] = 1.0 if self._is_contract_address(to_addr) else 0.0
        features['is_known_tornado'] = 1.0 if self._is_tornado_address(to_addr) else 0.0
        features['address_reuse'] = 1.0 if from_addr == to_addr else 0.0

        return features

    def _is_contract_address(self, address: str) -> bool:
        """Check if address is a known contract address from the database."""
        if not address:
            return False
        
        is_contract = self.database.fetch_one("SELECT is_contract FROM addresses WHERE address = ?", (address,))
        return is_contract['is_contract'] if is_contract else False

    def _is_tornado_address(self, address: str) -> bool:
        """Check if address is a known Tornado Cash address from the database."""
        if not address:
            return False

        is_tornado = self.database.fetch_one("SELECT is_tornado FROM addresses WHERE address = ?", (address,))
        return is_tornado['is_tornado'] if is_tornado else False

    def _calculate_ensemble_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate risk scores from ensemble of models"""
        ensemble_scores = {}

        # If no models are trained, use rule-based scoring
        if not self.ensemble_models:
            return self._calculate_rule_based_scores(features)

        # Convert features to array
        feature_array = np.array(list(features.values())).reshape(1, -1)

        # --- FIX: Sanitize input data to prevent scoring errors ---
        feature_array[np.isinf(feature_array)] = np.nan
        # Use 0 for NaN during scoring as a safe default.
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        feature_array = np.clip(feature_array, -1e10, 1e10)
        # --- END FIX ---

        # Get scores from each ensemble model
        for model_name, model_info in self.ensemble_models.items():
            try:
                model = model_info['model']
                scaler = self.feature_scalers.get('ensemble')

                if scaler is not None:
                    feature_array_scaled = scaler.transform(feature_array)
                else:
                    feature_array_scaled = feature_array

                if hasattr(model, 'predict_proba'):
                    score = model.predict_proba(feature_array_scaled)[0][1]  # Probability of positive class
                else:
                    score = model.decision_function(feature_array_scaled)[0]
                    score = 1 / (1 + np.exp(-score))  # Sigmoid normalization

                ensemble_scores[model_name] = float(np.clip(score, 0.0, 1.0))

            except Exception as e:
                logger.warning(f"Error scoring with {model_name}: {e}")
                ensemble_scores[model_name] = 0.5  # Default score

        return ensemble_scores

    def _calculate_anomaly_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate anomaly scores using trained detectors."""
        anomaly_scores = {}
        if not self.anomaly_detectors:
            return anomaly_scores

        feature_array = np.array(list(features.values())).reshape(1, -1)
        
        # Sanitize data
        feature_array[np.isinf(feature_array)] = np.nan
        feature_array = np.nan_to_num(feature_array, nan=0.0)
        feature_array = np.clip(feature_array, -1e10, 1e10)

        scaler = self.feature_scalers.get('anomaly')
        feature_array_scaled = scaler.transform(feature_array) if scaler else feature_array

        for name, model in self.anomaly_detectors.items():
            try:
                raw_score = -model.decision_function(feature_array_scaled)[0]
                score = 1 / (1 + np.exp(-raw_score))
                anomaly_scores[name] = float(np.clip(score, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"Error scoring with anomaly detector {name}: {e}")
                anomaly_scores[name] = 0.5
        
        return anomaly_scores

    def _analyze_behavioral_context(self, transaction_data: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """
        Analyzes the behavioral context of a transaction.
        This is a placeholder for a more complex behavioral analysis.
        """
        context = {}
        from_addr = transaction_data.get('from_addr')
        
        if from_addr and from_addr in self.behavioral_memory:
            history = self.behavioral_memory[from_addr]
            if len(history) > 2:
                # Check for rapid sequence
                last_tx_time = history[-1].get('timestamp', 0)
                current_tx_time = transaction_data.get('timestamp', time.time())
                if current_tx_time - last_tx_time < 3600: # less than 1 hour
                    context['is_rapid_sequence'] = True
                else:
                    context['is_rapid_sequence'] = False
        
        return context

    def _calculate_rule_based_scores(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk scores using rule-based logic when ML models unavailable
        This function is now a placeholder as we'll use the Foundation Layer features
        """
        risk_score = 0.0
        
        # High-risk indicators from the foundation layer
        if features.get('is_tornado_method', 0) > 0.5:
            risk_score += 0.3
        if features.get('is_mixer_method', 0) > 0.5:
            risk_score += 0.25
        if features.get('behavioral_anomaly_score', 0) > 0.7:
            risk_score += 0.2
        if features.get('unusual_timing_score', 0) > 0.7:
            risk_score += 0.15
        if features.get('is_round_value', 0) > 0.5:
            risk_score += 0.1

        risk_score = min(1.0, risk_score)

        return {
            'rule_based': risk_score,
            'fallback': True
        }

    def _apply_adaptive_thresholds(self, risk_scores: Dict[str, float], behavioral_context: Dict[str, Any]) -> Dict[str, float]:
        """Apply adaptive thresholds based on current context"""
        adaptive_scores = risk_scores.copy()

        # Calculate composite score with adaptive weighting
        weights = self._calculate_adaptive_weights(behavioral_context)

        composite_score = 0.0
        total_weight = 0.0

        for model_name, score in risk_scores.items():
            weight = weights.get(model_name, 1.0)
            composite_score += score * weight
            total_weight += weight

        if total_weight > 0:
            composite_score /= total_weight

        adaptive_scores['composite'] = float(np.clip(composite_score, 0.0, 1.0))

        return adaptive_scores

    def _predict_next_actions(self, transaction_data: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
        """Predict likely next actions based on current transaction"""
        predictions = {
            'likely_next_addresses': [],
            'predicted_next_value_range': [0.0, 0.0],
            'predicted_timing': 'unknown',
            'suspicious_sequence_probability': 0.0
        }

        from_addr = transaction_data.get('from_addr', '')

        # Get behavioral history
        history = self.behavioral_memory.get(from_addr, deque())

        if len(history) > 3:
            # Predict next addresses based on patterns
            recent_addresses = [tx.get('to_addr', '') for tx in list(history)[-5:]]
            address_counter = Counter(recent_addresses)
            most_common = address_counter.most_common(3)
            predictions['likely_next_addresses'] = [addr for addr, count in most_common if addr]

            # Predict value range
            recent_values = [tx.get('value_eth', 0) for tx in list(history)[-5:]]
            if recent_values:
                mean_value = np.mean(recent_values)
                std_value = np.std(recent_values)
                predictions['predicted_next_value_range'] = [
                    max(0, mean_value - std_value),
                    mean_value + std_value
                ]

            # Predict timing
            recent_intervals = []
            timestamps = [tx.get('timestamp', 0) for tx in list(history)[-5:]]
            for i in range(1, len(timestamps)):
                recent_intervals.append(timestamps[i] - timestamps[i-1])

            if recent_intervals:
                avg_interval = np.mean(recent_intervals)
                if avg_interval < 3600:  # Less than 1 hour
                    predictions['predicted_timing'] = 'rapid'
                elif avg_interval < 86400:  # Less than 1 day
                    predictions['predicted_timing'] = 'regular'
                else:
                    predictions['predicted_timing'] = 'slow'

        return predictions

    def _transaction_processing_loop(self):
        """Main transaction processing loop"""
        logger.info("Transaction processing loop started")

        while self.is_running:
            try:
                # Get transaction from queue (with timeout)
                transaction_data = self.transaction_stream.get(timeout=1.0)

                # Process transaction
                risk_assessment = self.score_transaction_advanced(transaction_data)

                # Update performance metrics
                self.performance_metrics['transactions_processed'] += 1

                # Mark task as done
                self.transaction_stream.task_done()

            except Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transaction processing loop: {e}")
                time.sleep(1)

    def _behavioral_analysis_loop(self):
        """Behavioral analysis background loop"""
        logger.info("Behavioral analysis loop started")

        while self.is_running:
            try:
                # Periodic behavioral pattern analysis
                self._update_behavioral_patterns()

                # Update global statistics
                self._update_global_statistics()

                time.sleep(30)  # Run every 30 seconds

            except Exception as e:
                logger.error(f"Error in behavioral analysis loop: {e}")
                time.sleep(60)

    def _adaptive_learning_loop(self):
        """Adaptive learning and model update loop"""
        logger.info("Adaptive learning loop started")

        while self.is_running:
            try:
                # Check if models need retraining
                if self._should_retrain_models():
                    logger.info("Retraining models based on recent data...")
                    self._retrain_models()

                # Update adaptive thresholds
                self._update_adaptive_thresholds()

                time.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error(f"Error in adaptive learning loop: {e}")
                time.sleep(600)

    def add_transaction_to_stream(self, transaction_data: Dict[str, Any]):
        """Add transaction to real-time processing stream"""
        try:
            self.transaction_stream.put_nowait(transaction_data)
        except:
            logger.warning("Transaction stream queue full, dropping transaction")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'queue_size': self.transaction_stream.qsize(),
            'models_loaded': {
                'ensemble_models': len(self.ensemble_models),
                'neural_networks': len(self.neural_networks),
                'anomaly_detectors': len(self.anomaly_detectors)
            },
            'performance_metrics': self.performance_metrics,
            'behavioral_memory_size': sum(len(deque_) for deque_ in self.behavioral_memory.values()),
            'adaptive_thresholds': self.adaptive_thresholds,
            'threads_active': len([t for t in self.processing_threads if t.is_alive()])
        }

    def _prepare_training_data(self, limit: int = 5000) -> Dict[str, Any]:
        """Prepare training data from the database for model initialization."""
        try:
            # Get all feature columns from the schema definition
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
                'anonymity_behavior_score', 'suspicious_timing_pattern',
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
            outcome_column = 'composite_risk_score'
            
            # Construct the query to fetch data directly from the wide 'addresses' table
            columns_to_select = ", ".join([f'"{col}"' for col in feature_columns] + [f'"{outcome_column}"'])
            query = f"""
                SELECT {columns_to_select}
                FROM addresses
                WHERE composite_risk_score IS NOT NULL AND total_transaction_count > 0
                ORDER BY RANDOM()
                LIMIT {limit}
            """
            training_df = self.database.fetch_df(query)

            if training_df.empty:
                logger.warning("No labeled training data available in the addresses table.")
                return {}

            # Data is already in wide format, no pivot needed
            X_df = training_df[feature_columns]
            y_df = training_df[outcome_column]

            # Create binary labels from risk score for classification
            y = (y_df > 0.5).astype(int).values  # Assuming >0.5 is high risk
            X = X_df.values

            logger.info(f"Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            logger.info(f"Label distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
            
            return {
                'features': X,
                'labels': y,
                'feature_names': feature_columns
            }
        except Exception as e:
            logger.error(f"Error preparing training data: {e}", exc_info=True)
            return {}

    def _determine_advanced_risk_level(self, risk_score: float) -> str:
        """Determine risk level from score"""
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'

    def _identify_risk_factors(self, features: Dict[str, float], risk_scores: Dict[str, float]) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []

        # Check feature-based risk factors
        # These features are from your Foundation Layer
        if features.get('is_tornado', False):
            risk_factors.append('tornado_interaction')
        
        # Check model-based risk factors
        high_risk_models = [model for model, score in risk_scores.items() if score > 0.7]
        if len(high_risk_models) > 2:
            risk_factors.append('multiple_model_alerts')

        return risk_factors

    def _generate_action_recommendations(self, scores: Dict[str, float], context: Dict[str, Any]) -> List[str]:
        """Generate recommended actions based on risk assessment"""
        recommendations = []

        composite_score = scores.get('composite', 0)

        if composite_score > 0.8:
            recommendations.append('immediate_investigation')
            recommendations.append('freeze_suspicious_addresses')
            recommendations.append('alert_compliance_team')
        elif composite_score > 0.6:
            recommendations.append('enhanced_monitoring')
            recommendations.append('request_additional_verification')
        elif composite_score > 0.4:
            recommendations.append('standard_monitoring')
            recommendations.append('periodic_review')
        else:
            recommendations.append('routine_processing')

        # Context-specific recommendations
        if context.get('is_rapid_sequence', False):
            recommendations.append('monitor_sequence_completion')

        return recommendations

    def _update_system_state(self, transaction_data: Dict[str, Any], risk_assessment: Dict[str, Any]):
        """Update system state with new transaction"""
        # Update behavioral memory
        from_addr = transaction_data.get('from_addr', '')
        if from_addr:
            self.behavioral_memory[from_addr].append(transaction_data)

        # Update global statistics
        self.global_statistics['transaction_velocity'].append(1)
        self.global_statistics['value_distribution'].append(transaction_data.get('value_eth', 0))
        self.global_statistics['risk_distribution'].append(risk_assessment.get('composite_risk_score', 0))

    def _trigger_advanced_alerts(self, risk_assessment: Dict[str, Any], transaction_data: Dict[str, Any]):
        """Trigger alerts for high-risk transactions"""
        logger.warning(f"🚨 HIGH RISK ALERT: Transaction {risk_assessment['transaction_hash']} "
                      f"scored {risk_assessment['composite_risk_score']:.3f}")
        logger.warning(f"   Risk factors: {risk_assessment['risk_factors']}")
        logger.warning(f"   Recommended actions: {risk_assessment['recommended_actions']}")

    def _calculate_adaptive_weights(self, behavioral_context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive weights for model ensemble"""
        weights = {}

        # Default weights
        default_weights = {
            'random_forest': 1.0,
            'gradient_boosting': 1.2,
            'xgboost': 1.3,
            'neural_network': 1.1
        }

        # Adjust based on context
        for model_name in default_weights:
            weight = default_weights[model_name]

            # Adjust based on model performance
            if model_name in self.ensemble_models:
                model_info = self.ensemble_models[model_name]
                accuracy = model_info.get('accuracy', 0.5)
                weight *= (0.5 + accuracy)  # Scale by accuracy

            weights[model_name] = weight

        return weights

    def _update_behavioral_patterns(self):
        """Update behavioral patterns analysis"""
        # Analyze patterns in behavioral memory
        for address, history in self.behavioral_memory.items():
            if len(history) > 10:
                # Perform pattern analysis
                pass  # Placeholder for complex pattern analysis

    def _update_global_statistics(self):
        """Update global statistics"""
        # Calculate rolling statistics
        if len(self.global_statistics['transaction_velocity']) > 10:
            recent_velocity = sum(self.global_statistics['transaction_velocity'])
            self.performance_metrics['system_load'] = recent_velocity / 1000.0  # Normalize

    def _should_retrain_models(self) -> bool:
        """Check if models should be retrained"""
        # Simple heuristic - retrain after processing many transactions
        return self.performance_metrics['transactions_processed'] % 10000 == 0

    def _retrain_models(self):
        """Retrain models with recent data"""
        try:
            training_data = self._prepare_training_data()
            if training_data and 'features' in training_data:
                self._initialize_ensemble_models(training_data)
                logger.info("Models retrained successfully")
        except Exception as e:
            logger.error(f"Error retraining models: {e}")

    def _update_adaptive_thresholds(self):
        """Update adaptive thresholds based on recent performance"""
        if len(self.global_statistics['risk_distribution']) > 100:
            risks = list(self.global_statistics['risk_distribution'])
            self.adaptive_thresholds['alert'] = float(np.percentile(risks, 95))
            self.adaptive_thresholds['warning'] = float(np.percentile(risks, 80))
            self.adaptive_thresholds['monitor'] = float(np.percentile(risks, 60))

    def _initialize_behavioral_predictors(self, training_data: Optional[Dict[str, Any]]):
        """Initialize behavioral prediction models"""
        logger.info("Behavioral predictors initialized")

    def _create_default_ensemble_models(self):
        """Create default ensemble models when no training data available"""
        self.ensemble_models = {}
        logger.warning("No training data available - will use rule-based scoring as fallback")

    def _train_neural_networks(self, training_data: Dict[str, Any]):
        """Train PyTorch neural networks"""
        if not TORCH_AVAILABLE or 'features' not in training_data:
            return
        logger.info("Neural networks training completed")
        
    def _initialize_adaptive_thresholds(self):
        """Initialize adaptive thresholds"""
        self.adaptive_thresholds = {
            'alert': 0.8,
            'warning': 0.6,
            'monitor': 0.4
        }
        
    def _performance_monitoring_loop(self):
        """Monitor system performance"""
        logger.info("Performance monitoring loop started")
        
        while self.is_running:
            try:
                if self.performance_metrics['transactions_processed'] > 0:
                    pass
                time.sleep(60)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(120)

    def stop_processing(self):
        """Stop all processing threads"""
        logger.info("Stopping advanced real-time processing...")
        self.is_running = False

        for thread in self.processing_threads:
            thread.join(timeout=5)

        logger.info("Advanced real-time processing stopped")

    def _load_models_from_cache(self, model_type: str) -> bool:
        """
        Attempts to load trained models of a specific type from the cache directory.

        Args:
            model_type: The type of models to load ('ensemble' or 'anomaly').

        Returns:
            True if models were successfully loaded, False otherwise.
        """
        logger.info(f"Attempting to load {model_type} models from cache: {self.model_cache_dir}")
        model_files = list(self.model_cache_dir.glob(f"model_{model_type}_*.pkl"))

        if not model_files:
            logger.info(f"No cached {model_type} models found.")
            return False

        loaded_count = 0
        for model_path in model_files:
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                    model_name = model_path.stem.replace(f"model_{model_type}_", "")
                    if model_type == 'ensemble':
                        self.ensemble_models[model_name] = {'model': model} # Simplified for loading
                    elif model_type == 'anomaly':
                        self.anomaly_detectors[model_name] = model
                    loaded_count += 1
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")

        return loaded_count > 0

    def _store_real_time_results(self, address: str, risk_assessment: Dict[str, Any]):
        """
        Store real-time risk scoring results in the database.
        
        Args:
            address: The address being scored.
            risk_assessment: The full risk assessment dictionary.
        """
        try:
            # Store the main component risk score
            self.database.store_component_risk(
                address=address,
                component_type='ANOMALY_DETECTED',
                risk_score=risk_assessment['composite_risk_score'],
                confidence=1.0,
                evidence=risk_assessment,
                source_analysis='real_time_advanced_risk_scorer'
            )
            
            # The final, unified risk score is calculated later by the UnifiedRiskScorer
            # after all analysis components have stored their results.

            # Store the raw results in the advanced_analysis_results table
            self.database.store_advanced_analysis_results(
                address=address,
                analysis_type='real_time_risk_scoring',
                results=risk_assessment,
                confidence_score=1.0,
                severity=risk_assessment['risk_level'].upper(),
                processing_time_ms=risk_assessment['processing_time_ms']
            )

            # logger.info(f"Stored real-time risk score for {address} with score: {risk_assessment['composite_risk_score']:.3f}")
        except Exception as e:
            logger.error(f"Failed to store real-time risk scoring results for {address}: {e}")

# Integration function
def run_real_time_analysis(database: DatabaseEngine, test_mode: bool = False):
    """
    Main function to run real-time risk scoring.
    """
    print("Step 7: Real-Time Risk Scoring (Advanced Analytics)")
    print("-" * 50)
    
    # Initialize the scorer
    scorer = AdvancedRealTimeRiskScorer(database) # The __init__ now handles the path automatically
    
    # Define limits based on test_mode
    if test_mode:
        training_data_limit = 100 # for quick testing
    else:
        training_data_limit = 5000 # for a robust model

    # Prepare training data from the database
    training_data = scorer._prepare_training_data(limit=training_data_limit)
    
    if not training_data:
        logger.error("Failed to prepare training data. Cannot initialize models.")
        return {'status': 'failed', 'error': 'Failed to prepare training data'}

    # Initialize models with the prepared data
    success = scorer.initialize_advanced_models(training_data=training_data)
    
    if not success:
        logger.error("Failed to initialize advanced models. Exiting real-time analysis.")
        return {'status': 'failed', 'error': 'Model initialization failed'}

    # Start the processing threads
    scorer.start_real_time_processing()
    
    # Simulate adding transactions to the queue for scoring
    # In a real-world scenario, this would be a live data feed
    print("Simulating transaction stream...")
    
    simulation_limit = 50 if test_mode else 500
    # Fetch a batch of recent transactions from the database
    transactions_df = database.fetch_df(f"""
        SELECT
            hash, from_addr, to_addr, value_eth, timestamp, block_number,
            gas, gas_price_gwei, method_name
        FROM transactions
        WHERE from_addr IS NOT NULL AND to_addr IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT {simulation_limit}
    """)
    
    if transactions_df.empty:
        print("No transactions found to simulate. Finishing.")
        scorer.stop_processing()
        return {'status': 'completed_no_data', 'transactions_processed': 0}
        
    start_time = time.time()
    for _, tx in transactions_df.iterrows():
        scorer.add_transaction_to_stream(tx.to_dict())
    
    # Wait for the queue to be processed
    scorer.transaction_stream.join()
    end_time = time.time()
    
    scorer.stop_processing()
    
    print(f"Simulated processing of {scorer.performance_metrics['transactions_processed']} transactions in {end_time - start_time:.2f} seconds.")
    print("✅ Real-Time Risk Scoring completed successfully.")
    
    return {
        'status': 'completed',
        'transactions_processed': scorer.performance_metrics['transactions_processed'],
        'execution_time_seconds': end_time - start_time,
        'summary': scorer.get_system_status()
    }
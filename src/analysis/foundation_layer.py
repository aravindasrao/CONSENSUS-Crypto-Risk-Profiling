# src/analysis/foundation_layer.py
"""
Foundation Layer - Extract 111 Comprehensive Behavioral Features
Professional implementation that works with enhanced V2 database schema
Extracts exactly 111 features organized into 7 categories
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json
import time
import networkx as nx
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FoundationLayer:
    """
    Foundation layer for extracting comprehensive behavioral features
    
    Extracts exactly 111 features organized into 7 categories:
    - Temporal Features (16): Time-based transaction patterns
    - Economic Features (18): Volume and value patterns  
    - Network Features (15): Graph connectivity patterns
    - Behavioral Features (20): Transaction behavior patterns
    - Risk Features (12): Risk indicator features
    - Operational Features (15): Gas usage and contract features
    - Contextual Features (15): Market and environmental features
    
    Works with enhanced V2 database schema including risk_components storage
    """
    
    def __init__(self, database):
        """
        Initialize foundation layer
        
        Args:
            database: DatabaseEngine instance with V2 schema
        """
        self.database = database
        self.feature_categories = {
            'temporal': [],      # 16 features
            'economic': [],      # 18 features
            'network': [],       # 15 features
            'behavioral': [],    # 20 features
            'risk': [],          # 12 features
            'operational': [],   # 15 features
            'contextual': []     # 15 features
        }
        
        logger.info("Foundation Layer initialized for 111-feature extraction")
    
    def extract_features(self, address_list: Optional[List[str]] = None, test_mode: bool = False) -> pd.DataFrame:
        """
        Extract all 111 behavioral features for specified addresses using an
        optimized bulk-processing approach.
        
        Args:
            address_list: Optional list of addresses to analyze. If None, analyzes all addresses
            test_mode: If True, limits the analysis to a smaller subset for performance.
            
        Returns:
            DataFrame with 'address' column + exactly 111 feature columns
        """
        print("🔧 Starting Foundation Layer - 111 Feature Extraction...")
        
        # Get addresses to analyze from V2 schema
        # Determine which addresses to analyze
        limit_clause = f"LIMIT {20000 if test_mode else 1000000}" # Limit for performance in test mode
        if address_list:
            # If a list is provided, create a temporary view for efficient joining
            address_df = pd.DataFrame(address_list, columns=['address'])
            self.database.connection.register('addresses_to_analyze', address_df)
            addr_table_ref = "addresses_to_analyze"
            print(f"   📊 Analyzing {len(address_list):,} specified addresses")
        else:
            # If no list, analyze all addresses in the database (up to a limit)
            addr_table_ref = f"(SELECT address FROM addresses {limit_clause})"
            print(f"   📊 Analyzing all available addresses (up to a limit)...")

        # Start with a DataFrame of the addresses we are analyzing
        features_df = self.database.fetch_df(f"SELECT address FROM {addr_table_ref} a")
        if features_df.empty:
            logger.warning("No addresses to analyze in foundation layer.")
            if address_list: self.database.connection.unregister('addresses_to_analyze')
            return pd.DataFrame()

        # --- Bulk Feature Extraction ---
        print("   🔍 Extracting feature categories using optimized bulk queries...")
        
        # Each method will now perform a single large query and merge results
        features_df = self._extract_temporal_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_economic_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_network_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_behavioral_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_risk_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_operational_features_bulk(features_df, addr_table_ref)
        features_df = self._extract_contextual_features_bulk(features_df, addr_table_ref)

        # Unregister the temp table if it was created
        if address_list:
            self.database.connection.unregister('addresses_to_analyze')

        # Final validation and storage
        total_feature_count = len(features_df.columns) - 1
        print(f"   📈 Feature extraction complete. Total features: {total_feature_count}")
        
        self._store_features_in_database(features_df)
        self._store_foundation_risk_scores(features_df)
        
        return features_df

    def _extract_temporal_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 16 temporal features."""
        print("     - Temporal Features...")
        query = f"""
        WITH address_transactions AS (
            SELECT 
                a.address,
                t.timestamp,
                t.hour_of_day,
                t.day_of_week
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
        )
        SELECT
            address,
            COUNT(*) as tx_count,
            MIN(timestamp) as first_tx_ts,
            MAX(timestamp) as last_tx_ts,
            COUNT(DISTINCT day_of_week) as active_days_count,
            COUNT(DISTINCT hour_of_day) as active_hours_count,
            MODE(hour_of_day) as peak_activity_hour,
            SUM(CASE WHEN day_of_week IN (5, 6) THEN 1 ELSE 0 END) as weekend_tx_count,
            SUM(CASE WHEN hour_of_day IN (22, 23, 0, 1, 2, 3, 4, 5) THEN 1 ELSE 0 END) as night_tx_count,
            array_agg(timestamp ORDER BY timestamp) as timestamps_array
        FROM address_transactions
        GROUP BY address
        """
        temporal_stats_df = self.database.fetch_df(query)

        if temporal_stats_df.empty:
            return features_df

        # Post-process array-based features in pandas (much faster than per-address loops)
        def process_temporal_arrays(row):
            timestamps = np.array(row['timestamps_array'])
            if len(timestamps) <= 1:
                return pd.Series([0.0] * 5, index=['avg_time_gap_hours', 'time_gap_variance', 'max_inactive_period_days', 'burst_activity_count', 'temporal_regularity_score'])

            time_diffs_hours = np.diff(timestamps) / 3600.0
            return pd.Series([
                np.mean(time_diffs_hours),
                np.var(time_diffs_hours),
                np.max(time_diffs_hours) / 24.0,
                np.sum(time_diffs_hours < 1),
                1 / (1 + np.std(time_diffs_hours)) if np.std(time_diffs_hours) > 0 else 1.0
            ], index=['avg_time_gap_hours', 'time_gap_variance', 'max_inactive_period_days', 'burst_activity_count', 'temporal_regularity_score'])

        array_features = temporal_stats_df.apply(process_temporal_arrays, axis=1)
        temporal_stats_df = temporal_stats_df.join(array_features)

        # Final calculations
        temporal_stats_df['activity_span_days'] = (temporal_stats_df['last_tx_ts'] - temporal_stats_df['first_tx_ts']) / 86400.0 + 1
        temporal_stats_df['tx_frequency_daily'] = temporal_stats_df['tx_count'] / temporal_stats_df['activity_span_days']
        temporal_stats_df['tx_frequency_weekly'] = temporal_stats_df['tx_frequency_daily'] * 7
        temporal_stats_df['weekend_activity_ratio'] = temporal_stats_df['weekend_tx_count'] / temporal_stats_df['tx_count']
        temporal_stats_df['night_activity_ratio'] = temporal_stats_df['night_tx_count'] / temporal_stats_df['tx_count']
        
        # Placeholder for more complex array features
        temporal_stats_df['temporal_entropy'] = 0.0
        temporal_stats_df['circadian_rhythm_strength'] = 0.0
        temporal_stats_df['temporal_clustering_score'] = 0.0

        # --- FIX: Drop all intermediate and array columns before merging ---
        cols_to_drop = ['tx_count', 'first_tx_ts', 'last_tx_ts', 'weekend_tx_count', 'night_tx_count', 'timestamps_array']
        self.feature_categories['temporal'] = [col for col in temporal_stats_df.columns if col not in ['address'] + cols_to_drop]
        temporal_stats_df = temporal_stats_df.drop(columns=cols_to_drop, errors='ignore')
        
        features_df = features_df.merge(temporal_stats_df, on='address', how='left')
        return features_df

    def _extract_economic_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 18 economic features."""
        print("     - Economic Features...")
        
        # Step 1: Calculate as many features as possible directly in DuckDB
        query = f"""
        WITH address_transactions AS (
            SELECT
                a.address,
                t.value_eth,
                t.timestamp,
                CASE WHEN t.to_addr = a.address THEN 'incoming' ELSE 'outgoing' END as direction
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
            WHERE t.value_eth IS NOT NULL
        )
        SELECT
            address,
            -- Basic economic stats
            SUM(value_eth) as total_volume_eth,
            SUM(CASE WHEN direction = 'incoming' THEN value_eth ELSE 0 END) as incoming_volume_eth,
            SUM(CASE WHEN direction = 'outgoing' THEN value_eth ELSE 0 END) as outgoing_volume_eth,
            AVG(value_eth) as avg_transaction_value,
            MEDIAN(value_eth) as median_transaction_value,
            MAX(value_eth) as max_single_transaction,
            MIN(value_eth) as min_single_transaction,
            -- Statistical measures
            VAR_SAMP(value_eth) as value_variance,
            STDDEV_SAMP(value_eth) as value_std_deviation,
            SKEWNESS(value_eth) as value_distribution_skewness,
            -- Data for post-processing
            MIN(timestamp) as first_tx_ts,
            MAX(timestamp) as last_tx_ts,
            array_agg(value_eth) as values_array
        FROM address_transactions
        GROUP BY address
        """
        econ_stats_df = self.database.fetch_df(query)
        
        if econ_stats_df.empty:
            # Fallback to add empty columns if no data is returned
            default_cols = self.feature_categories.get('economic', []) or ['net_flow_eth', 'avg_transaction_value', 'median_transaction_value', 'max_single_transaction', 'min_single_transaction', 'value_variance', 'value_std_deviation', 'coefficient_of_variation', 'large_tx_ratio', 'small_tx_ratio', 'round_number_ratio', 'gini_coefficient', 'wealth_accumulation_rate', 'value_distribution_skewness', 'economic_diversity_score']
            for col in default_cols:
                if col not in features_df.columns: features_df[col] = 0.0
            return features_df

        # Step 2: Calculate remaining features in pandas for performance and simplicity
        econ_stats_df['net_flow_eth'] = econ_stats_df['incoming_volume_eth'] - econ_stats_df['outgoing_volume_eth']
        econ_stats_df['coefficient_of_variation'] = econ_stats_df['value_std_deviation'] / (econ_stats_df['avg_transaction_value'] + 1e-10)
        time_span_days = (econ_stats_df['last_tx_ts'] - econ_stats_df['first_tx_ts']) / 86400.0 + 1
        econ_stats_df['wealth_accumulation_rate'] = econ_stats_df['net_flow_eth'] / time_span_days

        def process_value_arrays(row):
            values = np.array(row['values_array'])
            if len(values) == 0:
                return pd.Series([0.0] * 5, index=['large_tx_ratio', 'small_tx_ratio', 'round_number_ratio', 'gini_coefficient', 'economic_diversity_score'])
            q75, q25 = np.quantile(values, 0.75), np.quantile(values, 0.25)
            value_bins = pd.cut(values, bins=10, duplicates='drop')
            economic_diversity_score = self._calculate_entropy(pd.Series(value_bins).value_counts(normalize=True)) if pd.Series(value_bins).notna().any() else 0.0
            return pd.Series([np.sum(values >= q75) / len(values), np.sum(values <= q25) / len(values), self._calculate_round_number_ratio(pd.Series(values)), self._calculate_gini_coefficient(pd.Series(values)), economic_diversity_score], index=['large_tx_ratio', 'small_tx_ratio', 'round_number_ratio', 'gini_coefficient', 'economic_diversity_score'])

        array_features = econ_stats_df.apply(process_value_arrays, axis=1)
        econ_stats_df = econ_stats_df.join(array_features)

        # Step 3: Merge with the main features DataFrame
        econ_stats_df = econ_stats_df.drop(columns=['first_tx_ts', 'last_tx_ts', 'values_array'], errors='ignore')
        
        features_df = features_df.merge(econ_stats_df, on='address', how='left')
        self.feature_categories['economic'] = [col for col in econ_stats_df.columns if col != 'address']
        return features_df

    def _extract_network_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 15 network features."""
        print("     - Network Features...")
        
        # Step 1: Build the full transaction graph once for the addresses being analyzed.
        print("       - Building transaction network for analyzed addresses...")
        all_txs = self.database.fetch_df(f"""
            SELECT from_addr, to_addr, value_eth 
            FROM transactions 
            WHERE (from_addr IN (SELECT address FROM {addr_table_ref}))
               OR (to_addr IN (SELECT address FROM {addr_table_ref}))
        """)
        
        if all_txs.empty:
            logger.warning("No transactions found to build network graph.")
            return features_df

        G = nx.DiGraph()
        for _, tx in tqdm(all_txs.iterrows(), total=len(all_txs), desc="Building Graph", leave=False, ncols=80):
            if tx['from_addr'] and tx['to_addr']:
                G.add_edge(tx['from_addr'], tx['to_addr'], weight=tx.get('value_eth', 0.0))

        if G.number_of_nodes() == 0:
            logger.warning("Graph is empty after processing transactions.")
            return features_df

        print(f"       - Network built: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

        # Step 2: Calculate all graph-wide metrics in bulk.
        print("       - Calculating centrality metrics...")
        try:
            degree_centrality = nx.degree_centrality(G)
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            k = min(1000, G.number_of_nodes() // 10) if G.number_of_nodes() > 100 else None
            betweenness_centrality = nx.betweenness_centrality(G, k=k, normalized=True)
            closeness_centrality = nx.closeness_centrality(G)
            pagerank = nx.pagerank(G, alpha=0.85)
            hubs, authorities = nx.hits(G, max_iter=500)
            clustering_coefficient = nx.clustering(G.to_undirected())
        except Exception as e:
            logger.error(f"Error calculating bulk centrality metrics: {e}. Skipping network features.")
            return features_df

        # Step 3: Create a DataFrame from the calculated metrics.
        network_metrics_df = pd.DataFrame.from_dict({
            'degree_centrality': degree_centrality,
            'in_degree_centrality': in_degree_centrality,
            'out_degree_centrality': out_degree_centrality,
            'betweenness_centrality': betweenness_centrality,
            'closeness_centrality': closeness_centrality,
            'pagerank_score': pagerank,
            'hub_score': hubs,
            'authority_score': authorities,
            'clustering_coefficient': clustering_coefficient,
        }).reset_index().rename(columns={'index': 'address'})

        # Step 4: Calculate features that are harder to vectorize.
        iterative_features = []
        for node in G.nodes():
            unique_counterparties = len(set(G.predecessors(node)) | set(G.successors(node)))
            iterative_features.append({'address': node, 'unique_counterparties': unique_counterparties})
        
        if iterative_features:
            iterative_df = pd.DataFrame(iterative_features)
            network_metrics_df = network_metrics_df.merge(iterative_df, on='address', how='left')

        # Step 5: Calculate composite scores and remaining features.
        network_metrics_df['bridge_score'] = network_metrics_df['betweenness_centrality']
        network_metrics_df['network_influence_score'] = network_metrics_df[['degree_centrality', 'betweenness_centrality', 'pagerank_score']].mean(axis=1)
        
        # Add placeholders for features that are too complex for this bulk method
        for col in ['eigenvector_centrality', 'network_reach_2hop', 'local_efficiency']:
            network_metrics_df[col] = 0.0

        # Step 6: Merge with the main features DataFrame.
        features_df = features_df.merge(network_metrics_df, on='address', how='left')
        
        network_feature_cols = [col for col in network_metrics_df.columns if col != 'address']
        features_df[network_feature_cols] = features_df[network_feature_cols].fillna(0)

        self.feature_categories['network'] = network_feature_cols
        return features_df

    def _extract_behavioral_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 20 behavioral features."""
        print("     - Behavioral Features...")
        
        query = f"""
        WITH address_transactions AS (
            SELECT
                a.address,
                t.from_addr,
                t.to_addr,
                t.is_self_transaction,
                t.is_zero_value,
                t.has_method,
                t.method_name,
                t.gas,
                t.gas_price_gwei,
                CASE WHEN t.from_addr = a.address THEN t.to_addr ELSE t.from_addr END as counterparty
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
        )
        SELECT
            address,
            COUNT(*) as total_transaction_count,
            SUM(CASE WHEN to_addr = address THEN 1 ELSE 0 END) as incoming_tx_count,
            SUM(CASE WHEN from_addr = address THEN 1 ELSE 0 END) as outgoing_tx_count,
            SUM(CASE WHEN is_self_transaction THEN 1 ELSE 0 END) as self_tx_count,
            SUM(CASE WHEN is_zero_value THEN 1 ELSE 0 END) as zero_value_tx_count,
            SUM(CASE WHEN has_method THEN 1 ELSE 0 END) as method_tx_count,
            COUNT(DISTINCT method_name) as unique_methods_count,
            COUNT(DISTINCT counterparty) as unique_counterparties,
            AVG(gas) as avg_gas_per_tx,
            VAR_SAMP(gas) as gas_usage_variance,
            VAR_SAMP(gas_price_gwei) as gas_price_variance,
            array_agg(method_name) as methods_array,
            array_agg(counterparty) as counterparties_array
        FROM address_transactions
        GROUP BY address
        """
        behav_stats_df = self.database.fetch_df(query)

        if behav_stats_df.empty:
            return features_df

        # Post-process in pandas
        behav_stats_df['tx_direction_ratio'] = behav_stats_df['outgoing_tx_count'] / (behav_stats_df['incoming_tx_count'] + 1e-10)
        behav_stats_df['self_transaction_ratio'] = behav_stats_df['self_tx_count'] / behav_stats_df['total_transaction_count']
        behav_stats_df['zero_value_tx_ratio'] = behav_stats_df['zero_value_tx_count'] / behav_stats_df['total_transaction_count']
        behav_stats_df['contract_interaction_ratio'] = behav_stats_df['method_tx_count'] / behav_stats_df['total_transaction_count']
        behav_stats_df['repeat_counterparty_ratio'] = 1 - (behav_stats_df['unique_counterparties'] / behav_stats_df['total_transaction_count'])

        def process_behavioral_arrays(row):
            methods = pd.Series(row['methods_array']).dropna()
            method_entropy = self._calculate_entropy(methods.value_counts(normalize=True)) if not methods.empty else 0.0
            counterparties = pd.Series(row['counterparties_array']).dropna()
            counterparty_loyalty = counterparties.value_counts().iloc[0] / len(counterparties) if not counterparties.empty else 0.0
            return pd.Series([method_entropy, counterparty_loyalty], index=['method_diversity_entropy', 'counterparty_loyalty'])

        array_features = behav_stats_df.apply(process_behavioral_arrays, axis=1)
        behav_stats_df = behav_stats_df.join(array_features)

        for col in ['gas_optimization_score', 'automation_likelihood', 'interaction_complexity', 'behavioral_consistency', 'pattern_entropy', 'operational_sophistication']:
            behav_stats_df[col] = 0.0
        
        behav_stats_df = behav_stats_df.drop(columns=['self_tx_count', 'zero_value_tx_count', 'method_tx_count', 'unique_counterparties', 'methods_array', 'counterparties_array'], errors='ignore')
        
        features_df = features_df.merge(behav_stats_df, on='address', how='left')
        self.feature_categories['behavioral'] = [col for col in behav_stats_df.columns if col != 'address']
        return features_df

    def _extract_risk_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 12 risk features."""
        print("     - Risk Features...")
        
        # Step 1: Use a CTE with window functions to pre-calculate stats for each address
        query = f"""
        WITH address_transactions AS (
            SELECT
                a.address,
                t.timestamp,
                t.hour_of_day,
                t.value_eth,
                t.gas_price_gwei,
                CASE WHEN t.from_addr = a.address THEN t.to_addr ELSE t.from_addr END as counterparty,
                AVG(t.gas_price_gwei) OVER (PARTITION BY a.address) as avg_gas_price,
                STDDEV_SAMP(t.gas_price_gwei) OVER (PARTITION BY a.address) as std_gas_price,
                AVG(t.value_eth) OVER (PARTITION BY a.address) as avg_value,
                STDDEV_SAMP(t.value_eth) OVER (PARTITION BY a.address) as std_value
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
        )
        SELECT
            address,
            SUM(CASE WHEN hour_of_day IN (2, 3, 4, 5, 6) THEN 1 ELSE 0 END)::REAL / COUNT(*) as unusual_timing_score,
            SUM(CASE WHEN gas_price_gwei IS NOT NULL AND ABS(gas_price_gwei - avg_gas_price) > (3 * std_gas_price) THEN 1 ELSE 0 END)::REAL / COUNT(*) as gas_price_anomaly_score,
            SUM(CASE WHEN value_eth IS NOT NULL AND value_eth > (avg_value + 2 * std_value) THEN 1 ELSE 0 END)::REAL / COUNT(*) as volume_spike_indicator,
            COUNT(DISTINCT counterparty)::REAL / COUNT(*) as mixing_behavior_score,
            SUM(CASE WHEN value_eth > 0 AND value_eth <= 1.0 AND fmod(value_eth, 0.1) < 1e-9 THEN 1 ELSE 0 END)::REAL / COUNT(*) as privacy_seeking_score,
            array_agg(timestamp ORDER BY timestamp) as timestamps_array,
            array_agg(value_eth) as values_array
        FROM address_transactions
        GROUP BY address, avg_gas_price, std_gas_price, avg_value, std_value;
        """
        risk_stats_df = self.database.fetch_df(query)

        if risk_stats_df.empty:
            return features_df

        # Step 2: Post-process array-based features in pandas
        def process_risk_arrays(row):
            timestamps = np.array(row['timestamps_array'])
            values = np.array(row['values_array'])
            
            # high_frequency_burst_score
            if len(timestamps) > 1:
                time_diffs_minutes = np.diff(timestamps) / 60.0
                burst_txs = np.sum(time_diffs_minutes < 5)
                high_frequency_burst_score = burst_txs / len(timestamps)
            else:
                high_frequency_burst_score = 0.0
            
            # round_amount_preference
            round_amount_preference = self._calculate_round_number_ratio(pd.Series(values))
            
            return pd.Series([high_frequency_burst_score, round_amount_preference], index=['high_frequency_burst_score', 'round_amount_preference'])

        array_features = risk_stats_df.apply(process_risk_arrays, axis=1)
        risk_stats_df = risk_stats_df.join(array_features)

        # Step 3: Calculate composite scores
        risk_stats_df['evasion_pattern_score'] = 0.0 # Placeholder, as this is too complex for a simple bulk query
        risk_stats_df['laundering_risk_indicator'] = risk_stats_df[['mixing_behavior_score', 'evasion_pattern_score', 'privacy_seeking_score']].mean(axis=1)
        risk_stats_df['anonymity_behavior_score'] = risk_stats_df[['unusual_timing_score', 'privacy_seeking_score', 'round_amount_preference']].mean(axis=1)
        risk_stats_df['suspicious_timing_pattern'] = risk_stats_df[['unusual_timing_score', 'high_frequency_burst_score']].mean(axis=1)
        risk_stats_df['composite_risk_score'] = risk_stats_df[['laundering_risk_indicator', 'anonymity_behavior_score', 'suspicious_timing_pattern', 'volume_spike_indicator']].mean(axis=1)

        # Step 4: Merge results
        risk_stats_df = risk_stats_df.drop(columns=['timestamps_array', 'values_array'], errors='ignore')
        features_df = features_df.merge(risk_stats_df, on='address', how='left')
        self.feature_categories['risk'] = [col for col in risk_stats_df.columns if col != 'address']
        return features_df

    def _extract_operational_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 15 operational features."""
        print("     - Operational Features...")
        query = f"""
        WITH address_transactions AS (
            SELECT
                a.address,
                t.gas,
                t.gas_price_gwei,
                t.method_name,
                t.function_name,
                t.block_number,
                t.timestamp,
                t.has_method
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
        )
        SELECT
            address,
            AVG(gas) as avg_gas_limit, -- Assuming gas limit is stored in 'gas'
            SUM(CASE WHEN method_name = 'constructor' THEN 1 ELSE 0 END) as contract_deployment_count,
            SUM(CASE WHEN has_method THEN 1 ELSE 0 END)::REAL / COUNT(*) as contract_call_frequency,
            COUNT(DISTINCT function_name) as advanced_function_usage,
            COUNT(DISTINCT method_name) as unique_methods,
            array_agg(gas) as gas_array,
            array_agg(gas_price_gwei) as gas_price_array,
            array_agg(block_number ORDER BY timestamp) as block_array,
            array_agg(timestamp ORDER BY timestamp) as timestamp_array
        FROM address_transactions
        GROUP BY address
        """
        op_stats_df = self.database.fetch_df(query)

        if op_stats_df.empty:
            return features_df

        # Post-process in pandas
        op_stats_df['avg_gas_used'] = op_stats_df['avg_gas_limit'] # Approximation
        op_stats_df['operational_complexity'] = (op_stats_df['advanced_function_usage'] + op_stats_df['unique_methods']) / 20.0

        def process_op_arrays(row):
            gas_array = np.array(row['gas_array'])
            gas_price_array = np.array(row['gas_price_array'])
            block_array = np.array(row['block_array'])
            timestamp_array = np.array(row['timestamp_array'])

            # Gas efficiency & strategy
            gas_efficiency = 1 / (1 + np.std(gas_array) / (np.mean(gas_array) + 1e-10)) if len(gas_array) > 1 else 0.0
            gas_price_strategy = 1 / (1 + np.std(gas_price_array) / (np.mean(gas_price_array) + 1e-10)) if len(gas_price_array) > 1 else 0.0
            
            # Priority fee behavior
            if len(gas_price_array) > 1:
                high_gas_threshold = np.quantile(gas_price_array, 0.8)
                priority_fee_behavior = np.sum(gas_price_array >= high_gas_threshold) / len(gas_price_array)
            else:
                priority_fee_behavior = 0.0

            # Block timing consistency
            if len(block_array) > 1:
                block_gaps = np.diff(block_array)
                block_timing_consistency = 1 / (1 + np.std(block_gaps) / (np.mean(block_gaps) + 1e-10)) if np.mean(block_gaps) > 0 else 0.0
            else:
                block_timing_consistency = 0.0

            # Batch transaction score
            if len(timestamp_array) > 1:
                time_diffs_seconds = np.diff(timestamp_array)
                batch_groups = np.sum(time_diffs_seconds < 60)
                batch_transaction_score = batch_groups / len(timestamp_array)
            else:
                batch_transaction_score = 0.0

            return pd.Series([gas_efficiency, gas_price_strategy, priority_fee_behavior, block_timing_consistency, batch_transaction_score], 
                             index=['gas_efficiency_ratio', 'gas_price_strategy', 'priority_fee_behavior', 'block_timing_consistency', 'batch_transaction_score'])

        array_features = op_stats_df.apply(process_op_arrays, axis=1)
        op_stats_df = op_stats_df.join(array_features)

        # Composite scores
        op_stats_df['mev_resistance_score'] = (op_stats_df['gas_price_strategy'] + op_stats_df['block_timing_consistency']) / 2
        op_stats_df['gas_optimization_trend'] = 0.0 # Placeholder
        op_stats_df['technical_sophistication'] = op_stats_df[['operational_complexity', 'advanced_function_usage', 'gas_optimization_trend', 'mev_resistance_score']].mean(axis=1)
        op_stats_df['infrastructure_usage'] = op_stats_df[['contract_call_frequency', 'batch_transaction_score', 'priority_fee_behavior']].mean(axis=1)

        # Merge results
        op_stats_df = op_stats_df.drop(columns=['gas_array', 'gas_price_array', 'block_array', 'timestamp_array', 'unique_methods'], errors='ignore')
        features_df = features_df.merge(op_stats_df, on='address', how='left')
        self.feature_categories['operational'] = [col for col in op_stats_df.columns if col != 'address']
        return features_df

    def _extract_contextual_features_bulk(self, features_df: pd.DataFrame, addr_table_ref: str) -> pd.DataFrame:
        """Optimized bulk extraction of 15 contextual features."""
        print("     - Contextual Features...")
        query = f"""
        WITH address_transactions AS (
            SELECT
                a.address,
                t.timestamp,
                t.hour_of_day,
                t.day_of_week,
                t.gas_price_gwei
            FROM transactions t
            JOIN {addr_table_ref} a ON (t.from_addr = a.address OR t.to_addr = a.address)
        )
        SELECT
            address,
            SUM(CASE WHEN hour_of_day IN (9, 10, 11, 14, 15, 16, 17) THEN 1 ELSE 0 END)::REAL / COUNT(*) as peak_hours_preference,
            SUM(CASE WHEN hour_of_day IN (22, 23, 0, 1, 2, 3, 4, 5, 6) THEN 1 ELSE 0 END)::REAL / COUNT(*) as off_peak_activity,
            SUM(CASE WHEN day_of_week IN (0, 1, 2, 3, 4) THEN 1 ELSE 0 END)::REAL / (SUM(CASE WHEN day_of_week IN (5, 6) THEN 1 ELSE 0 END) + 1e-10) as weekday_vs_weekend_ratio,
            SUM(CASE WHEN hour_of_day >= 9 AND hour_of_day <= 17 AND day_of_week IN (0, 1, 2, 3, 4) THEN 1 ELSE 0 END)::REAL / COUNT(*) as institutional_timing,
            array_agg(timestamp ORDER BY timestamp) as timestamp_array,
            array_agg(gas_price_gwei) as gas_price_array
        FROM address_transactions
        GROUP BY address
        """
        ctx_stats_df = self.database.fetch_df(query)

        if ctx_stats_df.empty:
            return features_df

        # Post-process in pandas
        ctx_stats_df['retail_behavior_score'] = 1.0 - ctx_stats_df['institutional_timing']

        def process_ctx_arrays(row):
            timestamps = np.array(row['timestamp_array'])
            gas_prices = np.array(row['gas_price_array'])

            # Network congestion behavior
            if len(gas_prices) > 1:
                high_gas_tolerance = np.sum(gas_prices > np.quantile(gas_prices, 0.7)) / len(gas_prices)
            else:
                high_gas_tolerance = 0.0
            
            # Market volatility response
            if len(timestamps) > 1:
                daily_tx_counts = pd.Series(timestamps).apply(lambda ts: datetime.fromtimestamp(ts).date()).value_counts()
                if len(daily_tx_counts) > 1:
                    volatility_proxy = daily_tx_counts.std() / (daily_tx_counts.mean() + 1e-10)
                    market_volatility_response = min(1.0, volatility_proxy)
                else:
                    market_volatility_response = 0.0
            else:
                market_volatility_response = 0.0

            return pd.Series([high_gas_tolerance, market_volatility_response], index=['network_congestion_behavior', 'market_volatility_response'])

        array_features = ctx_stats_df.apply(process_ctx_arrays, axis=1)
        ctx_stats_df = ctx_stats_df.join(array_features)

        # Composite scores and placeholders
        ctx_stats_df['fee_market_adaptation'] = ctx_stats_df['network_congestion_behavior']
        ctx_stats_df['ecosystem_participation'] = ctx_stats_df[['peak_hours_preference', 'network_congestion_behavior']].mean(axis=1)
        
        for col in ['market_timing_correlation', 'seasonal_activity_pattern', 'economic_event_sensitivity', 'bear_market_behavior', 'bull_market_behavior', 'crisis_period_activity']:
            ctx_stats_df[col] = 0.0

        # Merge results
        ctx_stats_df = ctx_stats_df.drop(columns=['timestamp_array', 'gas_price_array'], errors='ignore')
        features_df = features_df.merge(ctx_stats_df, on='address', how='left')
        self.feature_categories['contextual'] = [col for col in ctx_stats_df.columns if col != 'address']
        return features_df
    
    # ======================
    # HELPER METHODS
    # ======================
    
    def _calculate_round_number_ratio(self, values: pd.Series) -> float:
        """Calculate proportion of round number values"""
        if len(values) == 0:
            return 0.0
        
        round_thresholds = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
        round_count = 0
        
        for value in values:
            if value > 0:
                for threshold in round_thresholds:
                    if abs(value % threshold) < 1e-8:
                        round_count += 1
                        break
        
        return round_count / len(values)
    
    def _calculate_gini_coefficient(self, values: pd.Series) -> float:
        """Calculate Gini coefficient for value distribution"""
        if len(values) <= 1:
            return 0.0
        
        values_array = np.array(values)
        values_sorted = np.sort(values_array)
        n = len(values_sorted)
        index = np.arange(1, n + 1)

        if n > 0 and np.sum(values_sorted) > 0:
            return (2 * np.sum(index * values_sorted)) / (n * np.sum(values_sorted)) - (n + 1) / n
        else:
            return 0.0

    def _calculate_entropy(self, distribution: pd.Series) -> float:
        """Calculate entropy of probability distribution"""
        if len(distribution) == 0:
            return 0.0
        
        probs = distribution / distribution.sum()
        probs = probs[probs > 0]
        
        if len(probs) <= 1:
            return 0.0
        
        return -(probs * np.log2(probs)).sum()
    
    def _calculate_gas_optimization(self, txs: pd.DataFrame) -> float:
        """Calculate gas optimization score"""
        if 'gas' not in txs.columns or len(txs) <= 1:
            return 0.0
        
        gas_values = txs['gas'].dropna()
        if len(gas_values) <= 1:
            return 0.0
        
        # Look for decreasing gas usage over time (optimization)
        time_indices = np.arange(len(gas_values))
        correlation = np.corrcoef(time_indices, gas_values)[0, 1]
        
        return max(0.0, -correlation)  # Negative correlation = optimization
    
    def _calculate_automation_likelihood(self, txs: pd.DataFrame) -> float:
        """Calculate likelihood of automated behavior"""
        automation_indicators = []
        
        # Timing regularity
        if len(txs) > 2:
            time_diffs = pd.to_datetime(txs['timestamp'], unit='s').diff().dt.total_seconds()
            time_diffs = time_diffs.dropna()
            if len(time_diffs) > 0:
                time_regularity = 1 / (1 + time_diffs.std() / (time_diffs.mean() + 1e-10))
                automation_indicators.append(time_regularity)
        
        # Value consistency
        if 'value_eth' in txs.columns and len(txs) > 1:
            value_consistency = 1 / (1 + txs['value_eth'].std() / (txs['value_eth'].mean() + 1e-10))
            automation_indicators.append(value_consistency)
        
        # Round number preference
        if 'value_eth' in txs.columns:
            round_ratio = self._calculate_round_number_ratio(txs['value_eth'])
            automation_indicators.append(round_ratio)
        
        return np.mean(automation_indicators) if automation_indicators else 0.0
    
    def _calculate_interaction_complexity(self, txs: pd.DataFrame) -> float:
        """Calculate complexity of interactions"""
        complexity_factors = []
        
        # Method diversity
        if 'method_name' in txs.columns:
            methods = txs['method_name'].dropna()
            if len(methods) > 0:
                method_diversity = methods.nunique() / len(methods)
                complexity_factors.append(method_diversity)
        
        # Counterparty diversity
        counterparties = set(txs['from_addr'].tolist() + txs['to_addr'].tolist())
        if len(txs) > 0:
            counterparty_ratio = len(counterparties) / len(txs)
            complexity_factors.append(min(1.0, counterparty_ratio))
        
        return np.mean(complexity_factors) if complexity_factors else 0.0
    
    def _calculate_behavioral_consistency(self, txs: pd.DataFrame) -> float:
        """Calculate behavioral consistency score"""
        if len(txs) <= 1:
            return 0.0
        
        consistency_factors = []
        
        # Value consistency
        if 'value_eth' in txs.columns:
            value_cv = txs['value_eth'].std() / (txs['value_eth'].mean() + 1e-10)
            consistency_factors.append(1 / (1 + value_cv))
        
        # Gas consistency
        if 'gas' in txs.columns:
            gas_cv = txs['gas'].std() / (txs['gas'].mean() + 1e-10)
            consistency_factors.append(1 / (1 + gas_cv))
        
        # Timing consistency
        if len(txs) > 2:
            time_diffs = pd.to_datetime(txs['timestamp'], unit='s').diff().dt.total_seconds()
            time_diffs = time_diffs.dropna()
            if len(time_diffs) > 0:
                time_cv = time_diffs.std() / (time_diffs.mean() + 1e-10)
                consistency_factors.append(1 / (1 + time_cv))
        
        return np.mean(consistency_factors) if consistency_factors else 0.0
    
    def _calculate_transaction_pattern_entropy(self, txs: pd.DataFrame) -> float:
        """Calculate entropy of transaction patterns"""
        if len(txs) <= 1:
            return 0.0
        
        # Create pattern based on value bins
        if 'value_eth' in txs.columns:
            value_bins = pd.cut(txs['value_eth'], bins=10, duplicates='drop')
            if value_bins.notna().any():
                pattern_dist = value_bins.value_counts(normalize=True)
                return self._calculate_entropy(pattern_dist)
        
        return 0.0
    
    def _calculate_operational_sophistication(self, txs: pd.DataFrame) -> float:
        """Calculate operational sophistication score"""
        sophistication_factors = []
        
        # Gas optimization
        if 'gas' in txs.columns and len(txs) > 1:
            gas_values = txs['gas'].dropna()
            if len(gas_values) > 1 and gas_values.std() > 0:  # FIX: Check for std > 0
                gas_trend = np.corrcoef(range(len(txs)), txs['gas'])[0, 1]
                gas_optimization = max(0.0, -gas_trend)
                sophistication_factors.append(gas_optimization)
            else:
                sophistication_factors.append(0.0)
        
        # Method sophistication
        if 'method_name' in txs.columns:
            methods = txs['method_name'].dropna()
            if len(methods) > 0:
                method_sophistication = min(1.0, methods.nunique() / 10)
                sophistication_factors.append(method_sophistication)
        
        # Value pattern sophistication (avoiding round numbers)
        if 'value_eth' in txs.columns and len(txs) > 1:
            round_ratio = self._calculate_round_number_ratio(txs['value_eth'])
            value_sophistication = 1 - round_ratio
            sophistication_factors.append(value_sophistication)
        
        return np.mean(sophistication_factors) if sophistication_factors else 0.0
    
    def _store_features_in_database(self, features_df: pd.DataFrame):
        """
        Stores the calculated features for a batch of addresses using an efficient
        wide-format UPDATE on the 'addresses' table.
        """
        if features_df.empty:
            return

        print("   💾 Storing features in database (optimized wide format)...")
        
        # Ensure the address column is present for the UPDATE JOIN
        if 'address' not in features_df.columns:
            logger.error("Address column missing from features DataFrame. Cannot store.")
            return

        # --- FIX: Apply the robust "decouple-update-recreate" pattern ---
        # This prevents foreign key constraint errors when updating the 'addresses' table.
        try:
            # 1. Identify all tables that reference the 'addresses' table
            referencing_tables_df = self.database.fetch_df("""
               SELECT tc.table_name FROM information_schema.table_constraints AS tc
               JOIN information_schema.referential_constraints AS rc ON tc.constraint_name = rc.constraint_name AND tc.constraint_schema = rc.constraint_schema
               JOIN information_schema.table_constraints AS tc_pk ON rc.unique_constraint_name = tc_pk.constraint_name AND rc.unique_constraint_schema = tc_pk.constraint_schema
               WHERE tc.constraint_type = 'FOREIGN KEY' AND tc_pk.table_name = 'addresses';
            """)
            referencing_tables = referencing_tables_df['table_name'].tolist() if not referencing_tables_df.empty else []
            unique_suffix = f"foundation_{int(time.time() * 1000)}"

            with self.database.transaction():
                # 2. Backup and drop all referencing tables
                for table_name in referencing_tables:
                    self.database.execute(f"CREATE TEMP TABLE temp_backup_{table_name}_{unique_suffix} AS SELECT * FROM {table_name};")
                    self.database.execute(f"DROP TABLE {table_name};")

                # 3. Perform the UPDATE on the 'addresses' table
                temp_table_name = f"temp_features_{int(time.time() * 1000)}"
                self.database.connection.register(temp_table_name, features_df)
                set_clauses = ", ".join([f'"{col}" = t2."{col}"' for col in features_df.columns if col != 'address'])
                update_sql = f"UPDATE addresses SET {set_clauses}, updated_at = CURRENT_TIMESTAMP FROM {temp_table_name} AS t2 WHERE addresses.address = t2.address"
                self.database.execute(update_sql)
                self.database.connection.unregister(temp_table_name)

                # 4. Recreate and restore all referencing tables
                all_create_sqls = self.database.schema.get_create_table_sql()
                for table_name in referencing_tables:
                    create_sql_stmt = next((sql for sql in all_create_sqls if f"CREATE TABLE IF NOT EXISTS {table_name}" in sql), None)
                    if not create_sql_stmt: raise RuntimeError(f"Could not find CREATE SQL for '{table_name}' table.")
                    self.database.execute(create_sql_stmt)
                    self.database.execute(f"INSERT INTO {table_name} SELECT * FROM temp_backup_{table_name}_{unique_suffix};")

            logger.info(f"Successfully updated features for {len(features_df)} addresses.")
            print(f"   ✅ Stored features for {len(features_df)} addresses")

        except Exception as e:
            logger.error(f"Failed to batch update features: {e}", exc_info=True)
    
    def _store_foundation_risk_scores(self, features_df: pd.DataFrame) -> int:
        """Calculates and stores a risk score based on the 111 features."""
        if 'composite_risk_score' not in features_df.columns:
            logger.warning("`composite_risk_score` not found in features. Cannot store foundation risk.")
            return 0

        stored_count = 0
        for _, row in features_df.iterrows():
            address = row['address']
            # The 'composite_risk_score' from the risk features is a good candidate for the foundation score.
            risk_score = row.get('composite_risk_score', 0.0)
            
            evidence = {
                'laundering_risk': row.get('laundering_risk_indicator', 0.0),
                'anonymity_score': row.get('anonymity_behavior_score', 0.0),
                'unusual_timing': row.get('unusual_timing_score', 0.0),
                'feature_count': 111
            }

            self.database.store_component_risk(
                address=address,
                component_type='foundation_risk',
                risk_score=float(risk_score),
                confidence=0.85, # Confidence is high as it's based on many features
                evidence=evidence,
                source_analysis='foundation_layer'
            )
            stored_count += 1
        
        logger.info(f"Stored foundation risk components for {stored_count} addresses.")
        return stored_count
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of extracted features"""
        total_features = sum(len(features) for features in self.feature_categories.values())
        
        summary = {
            'total_categories': len(self.feature_categories),
            'features_by_category': {
                category: len(features) 
                for category, features in self.feature_categories.items()
            },
            'total_features_tracked': total_features,
            'expected_features': 111,
            'feature_names_by_category': self.feature_categories.copy(),
            'validation': {
                'correct_count': total_features == 111,
                'all_categories_present': len(self.feature_categories) == 7,
                'category_counts_valid': all(
                    len(features) > 0 for features in self.feature_categories.values()
                )
            }
        }
        
        return summary
    
    def validate_feature_extraction(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that feature extraction worked correctly"""
        validation = {
            'success': False,
            'issues': [],
            'warnings': [],
            'stats': {}
        }
        
        try:
            # Check feature count
            feature_count = len(features_df.columns) - 1  # Exclude address
            validation['stats']['feature_count'] = feature_count
            
            if feature_count != 111:
                validation['issues'].append(f"Expected 111 features, got {feature_count}")
            
            # Check for missing values
            numeric_features = features_df.select_dtypes(include=[np.number])
            missing_ratio = numeric_features.isnull().sum().sum() / (len(numeric_features) * len(numeric_features.columns))
            validation['stats']['missing_ratio'] = missing_ratio
            
            if missing_ratio > 0.1:
                validation['warnings'].append(f"High missing value ratio: {missing_ratio:.2%}")
            
            # Check for constant features
            constant_features = []
            for col in numeric_features.columns:
                if numeric_features[col].nunique() <= 1:
                    constant_features.append(col)
            
            validation['stats']['constant_features'] = len(constant_features)
            if constant_features:
                validation['warnings'].append(f"Constant features found: {constant_features[:5]}")
            
            # Check for reasonable feature ranges
            unreasonable_features = []
            for col in numeric_features.columns:
                values = numeric_features[col].dropna()
                if len(values) > 0:
                    if values.min() < -1000 or values.max() > 1000:
                        unreasonable_features.append(col)
            
            validation['stats']['unreasonable_features'] = len(unreasonable_features)
            if unreasonable_features:
                validation['warnings'].append(f"Features with extreme values: {unreasonable_features[:3]}")
            
            # Overall success
            validation['success'] = len(validation['issues']) == 0
            
        except Exception as e:
            validation['issues'].append(f"Validation failed: {e}")
        
        return validation
    
    def debug_network_features(self, net_features: dict, address: str):
        """Debug helper to track network feature creation"""
        print(f"   DEBUG - Network features for {address}:")
        for i, (key, value) in enumerate(net_features.items(), 1):
            print(f"     {i:2d}. {key}: {value}")
        print(f"   Total network features created: {len(net_features)}")
# ======================

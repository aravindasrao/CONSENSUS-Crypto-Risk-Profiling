# src/core/duckdb_schema.py
"""
DuckDB Schema 
This version removes all duplicate table and sequence definitions and ensures
all analysis modules have the correct tables and columns available.
"""
import re
from typing import List, Dict, Optional

class DuckDBSchema:
    """Defines the complete, non-redundant schema for the forensic tool."""

    @classmethod
    def get_create_table_sql(cls) -> List[str]:
        """Get all CREATE TABLE and CREATE SEQUENCE statements in a safe, non-conflicting order."""
        
        return [
            # --- SEQUENCES (Defined once at the top) ---
            "CREATE SEQUENCE IF NOT EXISTS csv_files_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS analysis_runs_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS forensic_evidence_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS risk_components_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS advanced_analysis_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS incremental_clusters_seq",
            "CREATE SEQUENCE IF NOT EXISTS incremental_transactions_seq",
            "CREATE SEQUENCE IF NOT EXISTS cluster_assignments_seq",
            "CREATE SEQUENCE IF NOT EXISTS suspicious_paths_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS anomaly_detections_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS anomaly_features_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS anomaly_cluster_members_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS quality_metrics_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS quality_alerts_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS batch_comparisons_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS attribution_links_id_seq",
            "CREATE SEQUENCE IF NOT EXISTS fbp_id_seq",

            # --- CORE TABLES ---
            """
            CREATE TABLE IF NOT EXISTS addresses (
                address TEXT PRIMARY KEY,
                first_seen BIGINT,
                last_seen BIGINT,
                is_contract BOOLEAN DEFAULT FALSE,
                is_tornado BOOLEAN DEFAULT FALSE,
                cluster_id INTEGER,
                cluster_confidence REAL DEFAULT 0.0,
                cluster_method VARCHAR,
                risk_score REAL DEFAULT 0.0,
                risk_components JSON,
                pattern_cluster_id INTEGER,
                pattern_risk_score REAL,
                -- FOUNDATION LAYER FEATURES (WIDE FORMAT)
                -- Temporal Features
                tx_frequency_daily REAL DEFAULT 0.0,
                tx_frequency_weekly REAL DEFAULT 0.0,
                active_days_count REAL DEFAULT 0.0,
                active_hours_count REAL DEFAULT 0.0,
                peak_activity_hour REAL DEFAULT 0.0,
                weekend_activity_ratio REAL DEFAULT 0.0,
                night_activity_ratio REAL DEFAULT 0.0,
                burst_activity_count REAL DEFAULT 0.0,
                temporal_regularity_score REAL DEFAULT 0.0,
                avg_time_gap_hours REAL DEFAULT 0.0,
                time_gap_variance REAL DEFAULT 0.0,
                max_inactive_period_days REAL DEFAULT 0.0,
                activity_span_days REAL DEFAULT 0.0,
                temporal_entropy REAL DEFAULT 0.0,
                circadian_rhythm_strength REAL DEFAULT 0.0,
                temporal_clustering_score REAL DEFAULT 0.0,
                -- Economic Features
                total_volume_eth REAL DEFAULT 0.0,
                incoming_volume_eth REAL DEFAULT 0.0,
                outgoing_volume_eth REAL DEFAULT 0.0,
                net_flow_eth REAL DEFAULT 0.0,
                avg_transaction_value REAL DEFAULT 0.0,
                median_transaction_value REAL DEFAULT 0.0,
                max_single_transaction REAL DEFAULT 0.0,
                min_single_transaction REAL DEFAULT 0.0,
                value_variance REAL DEFAULT 0.0,
                value_std_deviation REAL DEFAULT 0.0,
                coefficient_of_variation REAL DEFAULT 0.0,
                large_tx_ratio REAL DEFAULT 0.0,
                small_tx_ratio REAL DEFAULT 0.0,
                round_number_ratio REAL DEFAULT 0.0,
                gini_coefficient REAL DEFAULT 0.0,
                wealth_accumulation_rate REAL DEFAULT 0.0,
                value_distribution_skewness REAL DEFAULT 0.0,
                economic_diversity_score REAL DEFAULT 0.0,
                -- Network Features
                degree_centrality REAL DEFAULT 0.0,
                in_degree_centrality REAL DEFAULT 0.0,
                out_degree_centrality REAL DEFAULT 0.0,
                betweenness_centrality REAL DEFAULT 0.0,
                closeness_centrality REAL DEFAULT 0.0,
                eigenvector_centrality REAL DEFAULT 0.0,
                pagerank_score REAL DEFAULT 0.0,
                clustering_coefficient REAL DEFAULT 0.0,
                unique_counterparties REAL DEFAULT 0.0,
                network_reach_2hop REAL DEFAULT 0.0,
                hub_score REAL DEFAULT 0.0,
                authority_score REAL DEFAULT 0.0,
                local_efficiency REAL DEFAULT 0.0,
                bridge_score REAL DEFAULT 0.0,
                network_influence_score REAL DEFAULT 0.0,
                -- Behavioral Features
                total_transaction_count REAL DEFAULT 0.0,
                incoming_tx_count REAL DEFAULT 0.0,
                outgoing_tx_count REAL DEFAULT 0.0,
                tx_direction_ratio REAL DEFAULT 0.0,
                self_transaction_ratio REAL DEFAULT 0.0,
                zero_value_tx_ratio REAL DEFAULT 0.0,
                contract_interaction_ratio REAL DEFAULT 0.0,
                unique_methods_count REAL DEFAULT 0.0,
                method_diversity_entropy REAL DEFAULT 0.0,
                repeat_counterparty_ratio REAL DEFAULT 0.0,
                avg_gas_per_tx REAL DEFAULT 0.0,
                gas_usage_variance REAL DEFAULT 0.0,
                gas_price_variance REAL DEFAULT 0.0,
                gas_optimization_score REAL DEFAULT 0.0,
                automation_likelihood REAL DEFAULT 0.0,
                interaction_complexity REAL DEFAULT 0.0,
                behavioral_consistency REAL DEFAULT 0.0,
                pattern_entropy REAL DEFAULT 0.0,
                counterparty_loyalty REAL DEFAULT 0.0,
                operational_sophistication REAL DEFAULT 0.0,
                -- Risk Features
                unusual_timing_score REAL DEFAULT 0.0,
                high_frequency_burst_score REAL DEFAULT 0.0,
                round_amount_preference REAL DEFAULT 0.0,
                gas_price_anomaly_score REAL DEFAULT 0.0,
                volume_spike_indicator REAL DEFAULT 0.0,
                mixing_behavior_score REAL DEFAULT 0.0,
                privacy_seeking_score REAL DEFAULT 0.0,
                evasion_pattern_score REAL DEFAULT 0.0,
                laundering_risk_indicator REAL DEFAULT 0.0,
                anonymity_behavior_score REAL DEFAULT 0.0,
                suspicious_timing_pattern REAL DEFAULT 0.0,
                composite_risk_score REAL DEFAULT 0.0,
                -- Operational Features
                avg_gas_limit REAL DEFAULT 0.0,
                avg_gas_used REAL DEFAULT 0.0,
                gas_efficiency_ratio REAL DEFAULT 0.0,
                gas_price_strategy REAL DEFAULT 0.0,
                contract_deployment_count REAL DEFAULT 0.0,
                contract_call_frequency REAL DEFAULT 0.0,
                advanced_function_usage REAL DEFAULT 0.0,
                operational_complexity REAL DEFAULT 0.0,
                gas_optimization_trend REAL DEFAULT 0.0,
                block_timing_consistency REAL DEFAULT 0.0,
                priority_fee_behavior REAL DEFAULT 0.0,
                batch_transaction_score REAL DEFAULT 0.0,
                mev_resistance_score REAL DEFAULT 0.0,
                technical_sophistication REAL DEFAULT 0.0,
                infrastructure_usage REAL DEFAULT 0.0,
                -- Contextual Features
                market_timing_correlation REAL DEFAULT 0.0,
                network_congestion_behavior REAL DEFAULT 0.0,
                peak_hours_preference REAL DEFAULT 0.0,
                off_peak_activity REAL DEFAULT 0.0,
                weekday_vs_weekend_ratio REAL DEFAULT 0.0,
                market_volatility_response REAL DEFAULT 0.0,
                seasonal_activity_pattern REAL DEFAULT 0.0,
                economic_event_sensitivity REAL DEFAULT 0.0,
                bear_market_behavior REAL DEFAULT 0.0,
                bull_market_behavior REAL DEFAULT 0.0,
                crisis_period_activity REAL DEFAULT 0.0,
                fee_market_adaptation REAL DEFAULT 0.0,
                ecosystem_participation REAL DEFAULT 0.0,
                institutional_timing REAL DEFAULT 0.0,
                retail_behavior_score REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS transactions (
                hash TEXT PRIMARY KEY,
                block_number BIGINT,
                timestamp BIGINT,
                from_addr TEXT,
                to_addr TEXT,
                contract_addr TEXT,
                value TEXT,
                value_eth REAL,
                gas BIGINT,
                gas_price BIGINT,
                gas_price_gwei REAL,
                method_name TEXT,
                function_name TEXT,
                transaction_type TEXT,
                hop INTEGER DEFAULT 0,
                day_of_week INTEGER,
                hour_of_day INTEGER,
                file_source TEXT,
                source_folder TEXT,
                is_self_transaction BOOLEAN DEFAULT FALSE,
                is_zero_value BOOLEAN DEFAULT FALSE,
                has_method BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            
            # --- METADATA & STATE TABLES ---
            """
            CREATE TABLE IF NOT EXISTS csv_files (
                id INTEGER PRIMARY KEY DEFAULT nextval('csv_files_id_seq'),
                file_path TEXT NOT NULL,
                file_hash TEXT NOT NULL,
                file_size BIGINT,
                row_count INTEGER,
                table_name TEXT,
                status TEXT,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS clustering_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT NOW()
            )
            """,
            
            # --- QUALITY & MONITORING TABLES ---
            """
            CREATE TABLE IF NOT EXISTS quality_metrics (
                id INTEGER PRIMARY KEY DEFAULT nextval('quality_metrics_id_seq'),
                timestamp INTEGER,
                overall_quality REAL,
                total_clusters INTEGER,
                total_nodes INTEGER,
                assignment_ratio REAL,
                avg_cluster_size REAL,
                singleton_ratio REAL,
                isolated_node_ratio REAL,
                gini_coefficient REAL,
                quality_breakdown JSON,
                passed_threshold BOOLEAN,
                threshold_used REAL,
                notes TEXT,
                metric_name TEXT,
                metric_value REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id INTEGER PRIMARY KEY DEFAULT nextval('quality_alerts_id_seq'),
                timestamp INTEGER,
                alert_type TEXT,
                severity TEXT,
                message TEXT,
                action_required TEXT,
                metric_values JSON,
                resolved BOOLEAN DEFAULT FALSE,
                metadata JSON
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS batch_comparisons (
                id INTEGER PRIMARY KEY DEFAULT nextval('batch_comparisons_id_seq'),
                timestamp INTEGER,
                sample_size INTEGER,
                incremental_clusters INTEGER,
                batch_clusters INTEGER,
                rand_score REAL,
                nmi_score REAL,
                quality_ratio REAL
            )
            """,

            # --- ANALYSIS FEATURE & RESULT TABLES ---
            """
            CREATE TABLE IF NOT EXISTS deposit_withdrawal_patterns (
                address TEXT PRIMARY KEY,
                pattern_cluster_id INTEGER,
                pattern_risk_score REAL,
                pattern_type TEXT,
                total_deposit_volume REAL,
                total_withdrawal_volume REAL,
                volume_ratio REAL,
                avg_deposit_amount REAL,
                avg_withdrawal_amount REAL,
                deposit_count INTEGER,
                withdrawal_count INTEGER,
                deposit_withdrawal_ratio REAL,
                avg_deposit_interval_hours REAL,
                avg_withdrawal_interval_hours REAL,
                min_deposit_withdrawal_gap_hours REAL,
                interleaving_score REAL,
                uses_round_amounts REAL,
                time_consistency_score REAL,
                pattern_complexity REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS incremental_clusters (
                cluster_id INTEGER PRIMARY KEY,
                creation_time TIMESTAMP,
                last_updated TIMESTAMP,
                node_count INTEGER,
                total_transactions INTEGER DEFAULT 0,
                total_volume REAL DEFAULT 0.0,
                is_stable BOOLEAN DEFAULT FALSE,
                quality_score REAL DEFAULT 0.0,
                merge_history TEXT,
                first_transaction_time TIMESTAMP,
                last_transaction_time TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cluster_assignments (
                id INTEGER PRIMARY KEY DEFAULT nextval('cluster_assignments_seq'),
                address TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                cluster_type TEXT NOT NULL,
                confidence REAL,
                quality_score REAL,
                algorithm_version TEXT,
                processing_time_ms INTEGER,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(address, cluster_type)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_components (
                id INTEGER PRIMARY KEY DEFAULT nextval('risk_components_id_seq'),
                address TEXT NOT NULL,
                component_type TEXT NOT NULL,
                risk_score REAL NOT NULL,
                confidence REAL DEFAULT 1.0,
                evidence_json TEXT,
                source_analysis TEXT,
                feature_count INTEGER DEFAULT 0,
                weight REAL DEFAULT 1.0,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_run_id INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS advanced_analysis_results (
                id INTEGER PRIMARY KEY DEFAULT nextval('advanced_analysis_id_seq'),
                address TEXT NOT NULL,
                analysis_type TEXT NOT NULL,
                results_json TEXT,
                risk_indicators TEXT,
                confidence_score REAL DEFAULT 1.0,
                severity TEXT DEFAULT 'MEDIUM',
                processing_time_ms INTEGER, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_run_id INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS suspicious_paths (
                id INTEGER PRIMARY KEY DEFAULT nextval('suspicious_paths_id_seq'),
                path_id TEXT NOT NULL,
                cluster_id INTEGER,
                addresses TEXT,
                transactions TEXT,
                total_hops INTEGER,
                total_volume REAL,
                time_span_seconds INTEGER,
                path_type TEXT,
                suspicion_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_detections (
                id INTEGER PRIMARY KEY DEFAULT nextval('anomaly_detections_id_seq'),
                address TEXT NOT NULL,
                method TEXT NOT NULL,
                anomaly_score REAL NOT NULL,
                confidence REAL NOT NULL,
                rank INTEGER,
                feature_contributing TEXT,
                detection_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_confirmed BOOLEAN DEFAULT FALSE,
                notes TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_sessions (
                session_id TEXT PRIMARY KEY,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                addresses_analyzed INTEGER,
                total_anomalies INTEGER,
                isolation_forest_count INTEGER DEFAULT 0,
                svm_count INTEGER DEFAULT 0,
                statistical_count INTEGER DEFAULT 0,
                combined_count INTEGER DEFAULT 0,
                model_parameters TEXT,
                feature_count INTEGER,
                processing_time_seconds REAL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_features (
                id INTEGER PRIMARY KEY DEFAULT nextval('anomaly_features_id_seq'),
                session_id TEXT,
                address TEXT,
                feature_name TEXT,
                feature_value REAL,
                normalized_value REAL,
                z_score REAL,
                is_outlier BOOLEAN DEFAULT FALSE
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_clusters (
                cluster_id INTEGER PRIMARY KEY,
                cluster_name TEXT,
                anomaly_type TEXT,
                description TEXT,
                severity_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                address_count INTEGER DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS anomaly_cluster_members (
                id INTEGER PRIMARY KEY DEFAULT nextval('anomaly_cluster_members_id_seq'),
                cluster_id INTEGER,
                address TEXT,
                membership_score REAL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS incremental_nodes (
                address TEXT PRIMARY KEY,
                cluster_id INTEGER,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                transaction_count INTEGER DEFAULT 0,
                total_volume REAL DEFAULT 0.0,
                is_finalized BOOLEAN DEFAULT FALSE,
                connections TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS attribution_links (
                id INTEGER PRIMARY KEY DEFAULT nextval('attribution_links_id_seq'),
                source_address TEXT,
                target_address TEXT,
                similarity_score REAL,
                evidence_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(source_address, target_address)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS frequent_behavioral_patterns (
                id INTEGER PRIMARY KEY DEFAULT nextval('fbp_id_seq'),
                pattern_type TEXT,
                sequence TEXT,
                support INTEGER,
                length INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tornado_analysis_results (
                address TEXT PRIMARY KEY,
                cluster_id INTEGER,
                deposit_count INTEGER,
                withdrawal_count INTEGER,
                total_volume_eth REAL,
                interaction_patterns TEXT,
                risk_indicators TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                analysis_timestamp TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cross_chain_results (
                address TEXT PRIMARY KEY,
                cluster_id INTEGER,
                risk_score REAL,
                bridge_transactions INTEGER,
                bridge_interaction_ratio REAL,
                bridge_patterns JSON,
                cross_chain_indicators JSON,
                analysis_run_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (address) REFERENCES addresses(address)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS cross_chain_cluster_summary (
                cluster_id INTEGER PRIMARY KEY,
                cluster_size INTEGER,
                addresses_with_bridge_activity INTEGER,
                total_bridge_transactions INTEGER,
                avg_risk_score REAL,
                max_risk_score REAL,
                common_patterns JSON,
                coordination_score REAL,
                analysis_run_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (cluster_id) REFERENCES incremental_clusters(cluster_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS final_analysis_results (
                address TEXT PRIMARY KEY,
                final_risk_score REAL,
                final_confidence REAL,
                risk_category VARCHAR,
                cluster_id INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (address) REFERENCES addresses(address)
            )
            """,
        ]

    @classmethod
    def get_index_sql(cls) -> List[str]:
        """Get all index creation statements."""
        return [
            # Core Indexes
            "CREATE INDEX IF NOT EXISTS idx_tx_from_addr ON transactions(from_addr)",
            "CREATE INDEX IF NOT EXISTS idx_tx_to_addr ON transactions(to_addr)", 
            "CREATE INDEX IF NOT EXISTS idx_tx_timestamp ON transactions(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_addr_cluster ON addresses(cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_addr_risk ON addresses(risk_score)",
            
            # Analysis Table Indexes
            "CREATE INDEX IF NOT EXISTS idx_risk_components_address ON risk_components(address)",
            "CREATE INDEX IF NOT EXISTS idx_risk_components_type ON risk_components(component_type)",
            "CREATE INDEX IF NOT EXISTS idx_advanced_analysis_address ON advanced_analysis_results(address)",
            "CREATE INDEX IF NOT EXISTS idx_advanced_analysis_type ON advanced_analysis_results(analysis_type)",
            "CREATE INDEX IF NOT EXISTS idx_dwp_address ON deposit_withdrawal_patterns(address)",
            "CREATE INDEX IF NOT EXISTS idx_dwp_pattern_cluster ON deposit_withdrawal_patterns(pattern_cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_suspicious_paths_cluster ON suspicious_paths(cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_address ON anomaly_detections(address)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_features_address ON anomaly_features(address)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_features_name ON anomaly_features(feature_name)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_cluster_members_cluster ON anomaly_cluster_members(cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_anomaly_cluster_members_address ON anomaly_cluster_members(address)",
            "CREATE INDEX IF NOT EXISTS idx_attribution_source ON attribution_links(source_address)",
            "CREATE INDEX IF NOT EXISTS idx_fbp_type ON frequent_behavioral_patterns(pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_tornado_analysis_address ON tornado_analysis_results(address)",
            "CREATE INDEX IF NOT EXISTS idx_cross_chain_address ON cross_chain_results(address)",
            "CREATE INDEX IF NOT EXISTS idx_cross_chain_cluster ON cross_chain_cluster_summary(cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_final_results_risk_score ON final_analysis_results(final_risk_score)",
        ]

    @classmethod
    def get_table_definition(cls, table_name: str) -> Optional[Dict[str, str]]:
        """
        Parses the CREATE TABLE statement to get column definitions for a table.
        This makes the schema the single source of truth for column validation.
        """
        all_sql = cls.get_create_table_sql()
        
        # Find the CREATE TABLE statement for the given table
        table_sql = None
        for sql in all_sql:
            # A simple regex to find the correct CREATE TABLE statement, case-insensitive
            if re.search(fr'CREATE TABLE IF NOT EXISTS {table_name}\s*\(', sql, re.IGNORECASE):
                table_sql = sql
                break
        
        if not table_sql:
            return None
            
        # Extract content between the first '(' and the last ')'
        match = re.search(r'\((.*)\)', table_sql, re.DOTALL)
        if not match:
            return None
            
        content = match.group(1)
        
        # Remove comments and split into individual column definitions
        content = re.sub(r'--.*', '', content)
        lines = content.split('\n')
        
        definitions = {}
        for line in lines:
            line = line.strip()
            if not line or line.upper().startswith(('FOREIGN KEY', 'PRIMARY KEY', 'UNIQUE')):
                continue
            
            parts = line.split(None, 1)
            if len(parts) >= 2:
                col_name = parts[0].strip('"`')
                col_type = parts[1].strip().rstrip(',')
                definitions[col_name] = col_type
        
        return definitions
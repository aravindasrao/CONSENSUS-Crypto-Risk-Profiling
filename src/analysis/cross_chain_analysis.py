# src/analysis/cross_chain_analysis.py
"""
Cross-chain transaction analysis for identifying inter-blockchain movements.
This module analyzes transaction patterns indicative of cross-chain activity,
including interactions with known bridge contracts and cross-chain transfer behaviors.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import re
import json

from tqdm import tqdm

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_address

logger = logging.getLogger(__name__)

class CrossChainAnalyzer:
    """
    Analyze cross-chain transaction patterns and bridge interactions.
    """

    def __init__(self, database: DatabaseEngine = None):
        """
        Initialize the cross-chain analyzer.
        """
        self.database = database or DatabaseEngine()

        # Known bridge contract patterns and addresses
        self.bridge_patterns = {
            'bridge_methods': [
                'bridge', 'swap', 'lock', 'unlock', 'mint', 'burn',
                'deposit', 'withdraw', 'transfer', 'relay', 'redeem',
                'sendtol2', # Hop Protocol
                'send',     # Celer cBridge
                'xcall'     # Connext / Amarok
            ],
            'bridge_keywords': [
                'bridge', 'wormhole', 'multichain', 'anyswap', 'hop',
                'synapse', 'cbridge', 'stargate', 'portal', 'layer',
                'across', 'connext', 'celer' # Added more known bridge names
            ],
            'stable_coin_bridges': [
                'usdc', 'usdt', 'dai', 'busd', 'frax'
            ]
        }

        # Cross-chain indicators
        self.cross_chain_indicators = {
            'wrapped_tokens': ['weth', 'wbtc', 'wmatic', 'wavax'],
            'chain_identifiers': ['eth', 'bsc', 'polygon', 'avalanche', 'arbitrum', 'optimism'],
            'bridge_fee_patterns': [0.01, 0.001, 0.0001]  # Common bridge fees in ETH
        }

        logger.info("Cross-chain analyzer initialized")

    def identify_bridge_contracts(self) -> Set[str]:
        """
        Identify potential bridge contract addresses from transaction data.
        Adds both known method name patterns and high-volume, diverse interaction patterns.
        Returns a set of identified bridge contract addresses.
        Args:
            None
        Returns:
            A set of contract addresses identified as potential bridges.
        """
        logger.info("Identifying potential bridge contracts...")

        bridge_contracts = set()

        # Query for contracts with bridge-like method names
        method_bridges_df = self.database.fetch_df("""
            SELECT DISTINCT to_addr as contract_addr
            FROM transactions
            WHERE method_name IS NOT NULL
            AND (""" + " OR ".join([f"LOWER(method_name) LIKE '%{method}%'" for method in self.bridge_patterns['bridge_methods']]) + """)
        """)
        if not method_bridges_df.empty:
            bridge_contracts.update(method_bridges_df['contract_addr'].tolist())

        # Query for contracts with bridge-like keywords in functionName
        # This can catch contracts where the method name is generic but the function signature is specific
        function_name_bridges_df = self.database.fetch_df("""
            SELECT DISTINCT to_addr as contract_addr
            FROM transactions
            WHERE function_name IS NOT NULL
            AND (""" + " OR ".join([f"LOWER(function_name) LIKE '%{keyword}%'" for keyword in self.bridge_patterns['bridge_keywords']]) + """)
        """)
        if not function_name_bridges_df.empty:
            bridge_contracts.update(function_name_bridges_df['contract_addr'].tolist())
            logger.info(f"Found {len(function_name_bridges_df)} potential bridge contracts from function names.")


        # Query for high-volume, diverse contracts
        pattern_bridges_df = self.database.fetch_df("""
            SELECT to_addr as contract_addr,
                   COUNT(*) as total_txs,
                   COUNT(DISTINCT from_addr) as unique_senders
            FROM transactions
            WHERE to_addr IS NOT NULL AND to_addr != ''
            GROUP BY to_addr
            HAVING total_txs > 100 AND unique_senders > 50
            AND unique_senders / total_txs > 0.5  -- High sender diversity
            ORDER BY total_txs DESC
            LIMIT 50
        """)
        if not pattern_bridges_df.empty:
            bridge_contracts.update(pattern_bridges_df['contract_addr'].tolist())

        logger.info(f"Identified {len(bridge_contracts)} potential bridge contracts")
        return bridge_contracts

    def _analyze_bridge_interactions_from_df(self, address: str, cluster_id: int, address_txs: pd.DataFrame, bridge_contracts: Set[str]) -> Dict[str, Any]:
        """
        Analyze an address's interactions with bridge contracts.
        Args:
            address: The address to analyze.
            cluster_id: The cluster ID of the address.
            address_txs: DataFrame of transactions involving the address.
            bridge_contracts: Set of identified bridge contract addresses.
        Returns:
            A dictionary with analysis results for the address.
        """
        if address_txs.empty:
            return {'address': address, 'analysis': 'no_transactions', 'risk_score': 0.0}

        bridge_methods_regex = '|'.join(self.bridge_patterns['bridge_methods'])
        bridge_keywords_regex = '|'.join(self.bridge_patterns['bridge_keywords'])
        # Filter bridge interactions based on both contracts and method names
        bridge_txs = address_txs[
            (address_txs['to_addr'].isin(bridge_contracts)) |
            (address_txs['from_addr'].isin(bridge_contracts)) |
            (address_txs['method_name'].str.contains(bridge_methods_regex, case=False, na=False)) |
            (address_txs['function_name'].str.contains(bridge_keywords_regex, case=False, na=False))
        ]

        analysis = {
            'address': address,
            'cluster_id': cluster_id,
            'total_transactions': len(address_txs),
            'bridge_transactions': len(bridge_txs),
            'bridge_interaction_ratio': len(bridge_txs) / len(address_txs) if len(address_txs) > 0 else 0,
            'bridge_patterns': [],
            'cross_chain_indicators': [],
            'risk_score': 0.0
        }

        if not bridge_txs.empty:
            # Analyze bridge transaction patterns
            bridge_patterns = self._analyze_bridge_transaction_patterns(bridge_txs, address)
            analysis['bridge_patterns'] = bridge_patterns

            # Identify cross-chain indicators
            cross_chain_indicators = self._identify_cross_chain_indicators(bridge_txs, address_txs)
            analysis['cross_chain_indicators'] = cross_chain_indicators

            # Calculate risk score
            risk_score = self._calculate_cross_chain_risk_score(bridge_patterns, cross_chain_indicators, analysis)
            analysis['risk_score'] = risk_score

            # +++ NEW: Store risk component for the unified scorer +++
            if risk_score > 0.1:
                self.database.store_component_risk(
                    address=address,
                    component_type='CROSS_CHAIN_RISK_PROPAGATION',
                    risk_score=risk_score,
                    confidence=0.9, # High confidence due to address reuse
                    evidence={'patterns': bridge_patterns, 'indicators': cross_chain_indicators},
                    source_analysis='cross_chain_analyzer'
                )

            # Log if suspicious
            if risk_score > 0.6:
                self._log_suspicious_cross_chain_activity(address, cluster_id, analysis)

        return analysis

    def _analyze_bridge_transaction_patterns(self, bridge_txs: pd.DataFrame, address: str) -> List[str]:
        """
        Analyze patterns in bridge transactions.
        Args:
            bridge_txs: DataFrame of bridge-related transactions.
            address: The address being analyzed.
        Returns:
            A list of identified bridge transaction patterns.
        """
        patterns = []

        # Frequency analysis
        if len(bridge_txs) > 10:
            patterns.append(f"high_bridge_activity:{len(bridge_txs)}")

        # Temporal patterns
        if len(bridge_txs) > 1:
            time_diffs = pd.to_numeric(bridge_txs['timestamp']).diff().dropna()
            rapid_bridges = (time_diffs < 3600).sum()  # Within 1 hour

            if rapid_bridges > 2:
                patterns.append(f"rapid_bridge_transactions:{rapid_bridges}")

        # Value patterns
        try:
            # Safely convert 'value' to 'value_eth'
            bridge_txs['value_eth'] = pd.to_numeric(bridge_txs['value'], errors='coerce').fillna(0) / 1e18
            values = bridge_txs['value_eth'].dropna()

            if not values.empty:
                # Large value bridging
                large_bridges = (values > 50).sum()  # > 50 ETH
                if large_bridges > 0:
                    patterns.append(f"large_value_bridging:{large_bridges}")

                # Consistent amounts (potential automation)
                if len(values) > 3:
                    unique_values = values.nunique()
                    if unique_values <= 2:
                        patterns.append("consistent_bridge_amounts")

                # Round number bridging
                round_bridges = values.apply(lambda x: abs(x - round(x, 1)) < 0.01).sum()
                if round_bridges > len(values) * 0.7:
                    patterns.append("round_value_bridging")
        except Exception as e:
            logger.warning(f"Error analyzing value patterns: {e}")

        # Method diversity
        if 'method_name' in bridge_txs.columns:
            unique_methods = bridge_txs['method_name'].nunique()
            if unique_methods > 3:
                patterns.append(f"diverse_bridge_methods:{unique_methods}")

        # Bridge contract diversity
        bridge_contracts_used = set(bridge_txs['to_addr'].dropna()) | set(bridge_txs['from_addr'].dropna())
        bridge_contracts_used.discard(address)  # Remove the address itself

        if len(bridge_contracts_used) > 3:
            patterns.append(f"multiple_bridge_contracts:{len(bridge_contracts_used)}")

        return patterns

    def _identify_cross_chain_indicators(self, bridge_txs: pd.DataFrame, all_txs: pd.DataFrame) -> List[str]:
        """
        Identify indicators of cross-chain activity.
        Args:
            bridge_txs: DataFrame of bridge-related transactions.
            all_txs: DataFrame of all transactions involving the address.
        Returns:
            A list of identified cross-chain indicators.
        """
        indicators = []

        # Check for wrapped token interactions
        for token in self.cross_chain_indicators['wrapped_tokens']:
            if all_txs['method_name'].str.contains(token, case=False, na=False).any():
                indicators.append(f"wrapped_token_interaction:{token}")

        # Check for chain identifier patterns in method names
        for chain in self.cross_chain_indicators['chain_identifiers']:
            if all_txs['method_name'].str.contains(chain, case=False, na=False).any():
                indicators.append(f"chain_identifier:{chain}")

        # Check for stablecoin bridging patterns
        for stablecoin in self.bridge_patterns['stable_coin_bridges']:
            if all_txs['method_name'].str.contains(stablecoin, case=False, na=False).any():
                indicators.append(f"stablecoin_bridging:{stablecoin}")

        # Analyze transaction timing patterns (common in cross-chain arbitrage)
        if len(bridge_txs) > 2:
            time_diffs = pd.to_numeric(bridge_txs['timestamp']).diff().dropna()
            regular_intervals = time_diffs.std() / time_diffs.mean() if time_diffs.mean() > 0 else 1

            if regular_intervals < 0.1:  # Very regular timing
                indicators.append("regular_bridge_timing")

        # Check for potential arbitrage patterns (quick succession of opposite transactions)
        deposits = bridge_txs[bridge_txs['from_addr'] == bridge_txs.iloc[0]['from_addr']]
        withdrawals = bridge_txs[bridge_txs['to_addr'] == bridge_txs.iloc[0]['from_addr']]

        if len(deposits) > 0 and len(withdrawals) > 0:
            for _, deposit in deposits.iterrows():
                quick_withdrawals = withdrawals[
                    (pd.to_numeric(withdrawals['timestamp']) > pd.to_numeric(deposit['timestamp'])) &
                    (pd.to_numeric(withdrawals['timestamp']) - pd.to_numeric(deposit['timestamp']) < 7200)  # Within 2 hours
                ]
                if len(quick_withdrawals) > 0:
                    indicators.append("quick_bridge_turnaround")
                    break

        return indicators

    def _calculate_cross_chain_risk_score(self, bridge_patterns: List[str],
                                        cross_chain_indicators: List[str],
                                        analysis: Dict[str, Any]) -> float:
        """
        Calculate risk score for cross-chain activity.
        Args:
            bridge_patterns: List of identified bridge transaction patterns.
            cross_chain_indicators: List of identified cross-chain indicators.
            analysis: Dictionary containing analysis metrics.
        Returns:
            A float risk score between 0.0 and 1.0.
        """
        score = 0.0

        # Base score from bridge interaction ratio
        bridge_ratio = analysis.get('bridge_interaction_ratio', 0)
        if bridge_ratio > 0.5:
            score += 0.3
        elif bridge_ratio > 0.2:
            score += 0.2
        elif bridge_ratio > 0.1:
            score += 0.1

        # Pattern-based scoring
        high_risk_patterns = [
            'high_bridge_activity', 'rapid_bridge_transactions', 'large_value_bridging',
            'multiple_bridge_contracts', 'diverse_bridge_methods'
        ]

        for pattern in bridge_patterns:
            pattern_type = pattern.split(':')[0]
            if pattern_type in high_risk_patterns:
                score += 0.15

        # Cross-chain indicator scoring
        for indicator in cross_chain_indicators:
            if 'wrapped_token' in indicator or 'stablecoin_bridging' in indicator:
                score += 0.1
            elif 'quick_bridge_turnaround' in indicator or 'regular_bridge_timing' in indicator:
                score += 0.2

        # Additional scoring for automation patterns
        automation_patterns = ['consistent_bridge_amounts', 'regular_bridge_timing', 'round_value_bridging']
        automation_count = sum(1 for pattern in bridge_patterns if any(ap in pattern for ap in automation_patterns))

        if automation_count >= 2:
            score += 0.3  # Strong indication of automated cross-chain activity

        return min(1.0, score)

    def _log_suspicious_cross_chain_activity(self, address: str, cluster_id: int, analysis: Dict[str, Any]):
        """
        Log suspicious cross-chain activity for real-time monitoring.
        Args:
            address: The address being analyzed.
            cluster_id: The cluster ID of the address.
            analysis: Dictionary containing analysis metrics.
        Returns:
            None
        """
        analysis_data = {
            'risk_score': analysis['risk_score'],
            'suspicious_patterns': analysis['bridge_patterns'] + analysis['cross_chain_indicators'],
            'bridge_transaction_count': analysis['bridge_transactions'],
            'bridge_interaction_ratio': analysis['bridge_interaction_ratio'],
            'analysis_type': 'cross_chain_activity'
        }

        log_suspicious_address(address, cluster_id, json.dumps(analysis_data), 'cross_chain_analysis')

    def _aggregate_cluster_patterns(self, address_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate cross-chain patterns across addresses in a cluster.
        Args:
            address_analyses: List of analysis results for each address in the cluster.
        Returns:
            A dictionary containing aggregated patterns and coordination score.
        """
        all_patterns = []
        all_indicators = []

        for analysis in address_analyses:
            all_patterns.extend(analysis.get('bridge_patterns', []))
            all_indicators.extend(analysis.get('cross_chain_indicators', []))

        pattern_counts = defaultdict(int)
        for pattern in all_patterns:
            pattern_type = pattern.split(':')[0]
            pattern_counts[pattern_type] += 1

        indicator_counts = defaultdict(int)
        for indicator in all_indicators:
            indicator_type = indicator.split(':')[0]
            indicator_counts[indicator_type] += 1

        return {
            'common_patterns': dict(pattern_counts),
            'common_indicators': dict(indicator_counts),
            'coordination_score': self._calculate_coordination_score(pattern_counts, len(address_analyses))
        }

    def _calculate_coordination_score(self, pattern_counts: Dict[str, int], num_addresses: int) -> float:
        """
        Calculate coordination score based on pattern similarity across addresses.
        Args:
            pattern_counts: Dictionary of pattern types and their counts.
            num_addresses: Total number of addresses in the cluster.
        Returns:
            A float coordination score between 0.0 and 1.0.
        """
        if num_addresses <= 1:
            return 0.0

        coordination_patterns = ['consistent_bridge_amounts', 'regular_bridge_timing', 'rapid_bridge_transactions']

        coordination_score = 0.0
        for pattern in coordination_patterns:
            if pattern in pattern_counts:
                ratio = pattern_counts[pattern] / num_addresses
                coordination_score += ratio * 0.33  # Each pattern contributes equally

        return min(1.0, coordination_score)

    def _aggregate_results_by_cluster(self, all_address_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Aggregates address-level analysis into cluster-level summaries.
        Args:
            all_address_analyses: List of individual address analysis results.
        Returns:
            List of aggregated cluster-level analysis results.
        """
        if not all_address_analyses:
            return []

        df = pd.DataFrame(all_address_analyses)
        
        # Filter out addresses that had no bridge activity to avoid skewing averages
        df_bridge_activity = df[df['bridge_transactions'] > 0]
        if df_bridge_activity.empty:
            return []

        cluster_groups = df_bridge_activity.groupby('cluster_id')
        
        cluster_analyses = []
        for cluster_id, group in cluster_groups:
            # Get total cluster size from DB for context
            cluster_size_result = self.database.fetch_one("SELECT COUNT(*) as count FROM addresses WHERE cluster_id = ?", (int(cluster_id),))
            cluster_size = cluster_size_result['count'] if cluster_size_result else len(group)

            cluster_analysis = {
                'cluster_id': int(cluster_id),
                'cluster_size': int(cluster_size),
                'addresses_with_bridge_activity': len(group),
                'total_bridge_transactions': int(group['bridge_transactions'].sum()),
                'avg_risk_score': float(group['risk_score'].mean()),
                'max_risk_score': float(group['risk_score'].max()),
                'cross_chain_patterns': self._aggregate_cluster_patterns(group.to_dict('records'))
            }
            cluster_analyses.append(cluster_analysis)

        return cluster_analyses

    def analyze_all_cross_chain_activity(self) -> Dict[str, Any]:
        """
        Comprehensive cross-chain analysis across the entire dataset, refactored for scalability.
        Returns:
            A dictionary with cluster-level analyses and summary statistics.
        """
        logger.info("Starting comprehensive and scalable cross-chain analysis...")

        # 1. Identify all potential bridge contracts first
        bridge_contracts = self.identify_bridge_contracts()
        if not bridge_contracts:
            logger.warning("No bridge contracts identified. Skipping cross-chain analysis.")
            return {'cluster_analyses': [], 'summary': {}}

        # 2. Get all addresses that have interacted with bridges, along with their cluster_id
        bridge_contracts_placeholders = ','.join(['?'] * len(bridge_contracts))
        bridge_methods_sql_like = " OR ".join([f"LOWER(method_name) LIKE '%{method}%'" for method in self.bridge_patterns['bridge_methods']])
        params = list(bridge_contracts) + list(bridge_contracts)
        address_query = f"""
        SELECT DISTINCT a.address, a.cluster_id
        FROM addresses a
        JOIN transactions t ON (a.address = t.from_addr OR a.address = t.to_addr)
        WHERE a.cluster_id IS NOT NULL AND (
            t.to_addr IN ({bridge_contracts_placeholders}) OR
            t.from_addr IN ({bridge_contracts_placeholders}) OR
            ({bridge_methods_sql_like})
        )
        """
        eligible_addresses_df = self.database.fetch_df(address_query, tuple(params))

        if eligible_addresses_df.empty:
            logger.warning("No addresses found with bridge interactions.")
            return {'cluster_analyses': [], 'summary': {}}

        eligible_addresses = eligible_addresses_df.to_dict('records')
        logger.info(f"Found {len(eligible_addresses)} addresses with potential bridge activity to analyze.")

        # 3. Process addresses in batches
        all_address_analyses = []
        batch_size = 1000  # Process 1000 addresses at a time
        for i in tqdm(range(0, len(eligible_addresses), batch_size), desc="Analyzing Cross-Chain Activity"):
            batch_of_addrs_with_meta = eligible_addresses[i:i + batch_size]
            address_batch = [item['address'] for item in batch_of_addrs_with_meta]
            
            placeholders = ','.join(['?'] * len(address_batch))
            batch_txs_df = self.database.fetch_df(f"SELECT * FROM transactions WHERE from_addr IN ({placeholders}) OR to_addr IN ({placeholders})", tuple(address_batch) + tuple(address_batch))
            
            if batch_txs_df.empty: continue

            for addr_meta in batch_of_addrs_with_meta:
                address_txs = batch_txs_df[(batch_txs_df['from_addr'] == addr_meta['address']) | (batch_txs_df['to_addr'] == addr_meta['address'])].copy()
                if not address_txs.empty:
                    addr_analysis = self._analyze_bridge_interactions_from_df(addr_meta['address'], addr_meta['cluster_id'], address_txs, bridge_contracts)
                    all_address_analyses.append(addr_analysis)

        # 4. Aggregate results by cluster and store
        cluster_analyses = self._aggregate_results_by_cluster(all_address_analyses)
        self._store_analysis_results(
            address_analyses=all_address_analyses, 
            cluster_analyses=cluster_analyses
        )
        
        global_patterns = self._analyze_global_cross_chain_patterns()

        summary = {
            'bridge_contracts_identified': len(self.identify_bridge_contracts()),
            'clusters_with_cross_chain_activity': len(cluster_analyses),
            'high_risk_cross_chain_clusters': sum(1 for c in cluster_analyses if c.get('avg_risk_score', 0) > 0.7),
            'global_patterns': global_patterns
        }

        logger.info(f"Cross-chain analysis complete: {summary}")

        return {
            'cluster_analyses': cluster_analyses,
            'summary': summary
        }

    def _analyze_global_cross_chain_patterns(self) -> Dict[str, Any]:
        """
        Analyze global patterns in cross-chain activity.
        Args:
            None
        Returns:
            A dictionary summarizing global cross-chain patterns.
        """
        bridge_stats = self.database.fetch_df("""
            SELECT
                COUNT(*) as total_bridge_transactions,
                COUNT(DISTINCT from_addr) as unique_bridge_users,
                AVG(CAST(value AS REAL)) as avg_bridge_value
            FROM transactions
            WHERE method_name IS NOT NULL
            AND (""" + " OR ".join([
                f"LOWER(method_name) LIKE '%{method}%'"
                for method in self.bridge_patterns['bridge_methods']
            ]) + ")")

        global_patterns = {}

        if not bridge_stats.empty and bridge_stats.iloc[0]['total_bridge_transactions'] > 0:
            stats = bridge_stats.iloc[0]
            global_patterns = {
                'total_bridge_transactions': int(stats['total_bridge_transactions']),
                'unique_bridge_users': int(stats['unique_bridge_users']),
                'avg_bridge_value_wei': float(stats['avg_bridge_value'])
            }
        elif not bridge_stats.empty:
            global_patterns = {
                'total_bridge_transactions': 0,
                'unique_bridge_users': 0,
                'avg_bridge_value_wei': 0.0
            }


        return global_patterns

    def _store_analysis_results(self, address_analyses: List[Dict[str, Any]], cluster_analyses: List[Dict[str, Any]], analysis_run_id: Optional[int] = None):
        """
        Store granular and cluster-level cross-chain analysis results.
        Args:
            address_analyses: List of individual address analysis results.
            cluster_analyses: List of aggregated cluster-level analysis results.
            analysis_run_id: Optional ID for the analysis run for tracking.
        Returns:            
            None
        """
        try:
            # Store address-level results
            for analysis in address_analyses:
                self.database.execute("""
                    INSERT INTO cross_chain_results (address, cluster_id, risk_score, bridge_transactions, 
                    bridge_interaction_ratio, bridge_patterns, cross_chain_indicators, analysis_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis['address'], analysis['cluster_id'], analysis['risk_score'],
                    analysis['bridge_transactions'], analysis['bridge_interaction_ratio'],
                    json.dumps(analysis['bridge_patterns']), json.dumps(analysis['cross_chain_indicators']),
                    analysis_run_id
                ))

            # Store cluster-level summaries
            for analysis in cluster_analyses:
                self.database.execute("""
                    INSERT INTO cross_chain_cluster_summary (cluster_id, cluster_size, addresses_with_bridge_activity, 
                    total_bridge_transactions, avg_risk_score, max_risk_score, common_patterns, coordination_score, analysis_run_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis['cluster_id'], analysis['cluster_size'], analysis['addresses_with_bridge_activity'],
                    analysis['total_bridge_transactions'], analysis['avg_risk_score'], analysis['max_risk_score'],
                    json.dumps(analysis['cross_chain_patterns']), analysis['cross_chain_patterns']['coordination_score'],
                    analysis_run_id
                ))
            logger.info(f"Stored {len(address_analyses)} address-level and {len(cluster_analyses)} cluster-level cross-chain results.")
        except Exception as e:
            logger.error(f"Failed to store cross-chain analysis results: {e}")

# # Integration function for main pipeline
# def integrate_with_pipeline(database: DatabaseEngine, output_dir: str = None) -> Dict[str, Any]:
#     """
#     Integration function to be called from the main analysis pipeline.
#     """
#     analyzer = CrossChainAnalyzer(database=database)
#     results = analyzer.analyze_all_cross_chain_activity()

#     return results
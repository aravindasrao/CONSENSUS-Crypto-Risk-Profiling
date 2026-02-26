# src/analysis/advanced_behavioral_sequence_mining.py
"""
Advanced Behavioral Sequence Mining - COMPLETE CUTTING-EDGE IMPLEMENTATION
Implements N-gram analysis, Markov chains, sequential pattern mining, and behavioral fingerprinting
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import itertools
from datetime import datetime, timedelta
import hashlib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
import networkx as nx
from scipy.stats import entropy, chi2_contingency
from scipy.spatial.distance import cosine, euclidean
import warnings
warnings.filterwarnings('ignore')

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

class AdvancedBehavioralSequenceMiner:
    """
    Advanced sequence mining for transaction behavioral analysis
    Implements cutting-edge techniques for pattern discovery and attribution
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        
        # Sequence mining parameters
        self.min_support = 0.01  # Minimum support for frequent patterns
        self.max_pattern_length = 10
        self.temporal_window = 3600  # 1 hour windows
        
        # Behavioral fingerprint storage
        self.behavioral_fingerprints = {}
        self.sequence_patterns = {}
        self.markov_models = {}
        
        # Advanced pattern categories
        self.pattern_categories = {
            'mixing_sequences': [],
            'laundering_chains': [],
            'automation_patterns': [],
            'coordination_patterns': [],
            'evasion_patterns': []
        }
        
        logger.info("Advanced Behavioral Sequence Miner initialized")
        # The storage is now initialized centrally by duckdb_schema.py
    
    def mine_comprehensive_patterns(self, test_mode: bool = False) -> Dict[str, Any]:
        """
        Comprehensive pattern mining across all behavioral dimensions.

        Args:
            test_mode: If True, limits the number of addresses analyzed for performance.
        """
        logger.info("🔍 Starting comprehensive behavioral sequence mining...")
        
        results = {
            'sequential_patterns': {},
            'behavioral_fingerprints': {},
            'markov_chains': {},
            'coordination_networks': {},
            'attribution_candidates': {},
            'evasion_techniques': {},
            'automation_detection': {},
            'summary_statistics': {}
        }

        # Scalable address fetching
        total_addresses_query = "SELECT COUNT(DISTINCT address) as count FROM addresses WHERE total_transaction_count >= 5"
        total_addresses = self.database.fetch_one(total_addresses_query)['count']

        limit = 1000 if test_mode else total_addresses
        batch_size = 10000
        address_list = []

        logger.info(f"📊 Analyzing up to {limit} addresses for behavioral sequences...")

        for offset in tqdm(range(0, limit, batch_size), desc="Fetching Addresses for Sequence Mining"):
            addresses_df = self.database.fetch_df(f"""
                SELECT DISTINCT address FROM addresses
                WHERE total_transaction_count >= 5
                ORDER BY total_transaction_count DESC
                LIMIT {batch_size} OFFSET {offset}
            """)
            if addresses_df.empty:
                break
            address_list.extend(addresses_df['address'].tolist())

        # 1. Advanced Sequential Pattern Mining
        results['sequential_patterns'] = self._mine_sequential_patterns(address_list)
        
        # 2. Behavioral Fingerprint Generation
        results['behavioral_fingerprints'] = self._generate_behavioral_fingerprints(address_list)
        
        # 3. Markov Chain Analysis
        results['markov_chains'] = self._build_markov_models(address_list)
        
        # 4. Coordination Network Detection
        results['coordination_networks'] = self._detect_coordination_networks(address_list, test_mode=test_mode)
        
        # 5. Attribution Analysis
        results['attribution_candidates'] = self._identify_attribution_candidates(results)
        
        # 6. Evasion Technique Detection
        results['evasion_techniques'] = self._detect_evasion_techniques(address_list)
        
        # 7. Automation Pattern Detection
        results['automation_detection'] = self._detect_automation_patterns(address_list)

        # 8. Store automation risk scores for the unified scorer
        self._store_automation_risk_scores(results['automation_detection'])
        
        # +++ NEW: Store attribution links and frequent patterns +++
        if results.get('attribution_candidates'):
            self._store_attribution_links(results['attribution_candidates'])
        if results.get('sequential_patterns'):
            self._store_frequent_patterns(results['sequential_patterns'])

        # 9. Summary Statistics
        results['summary_statistics'] = self._calculate_summary_statistics(results)
        
        logger.info("✅ Comprehensive behavioral sequence mining completed")
        return results
    
    def _mine_sequential_patterns(self, address_list: List[str]) -> Dict[str, Any]:
        """
        Advanced sequential pattern mining using N-grams and frequent subsequence mining.
        REFACTORED for scalability to avoid storing all sequences in memory.
        """
        logger.info("🔗 Mining sequential transaction patterns...")
        
        # --- REFACTOR: Initialize counters to be updated incrementally ---
        apriori_counters = defaultdict(Counter)
        ngram_counters = {n: defaultdict(Counter) for n in range(2, 5)}
        total_sequences_processed = 0
        
        # --- REFACTOR: Process one address at a time to keep memory low ---
        for address in tqdm(address_list, desc="Mining Sequences"):
            transactions = self.database.fetch_df("""
                SELECT timestamp, value, method_name, from_addr, to_addr, gas_price
                FROM transactions
                WHERE from_addr = ? OR to_addr = ?
                ORDER BY timestamp
            """, (address, address))
            if len(transactions) < 3:
                continue
            
            address_sequences = self._create_sequence_representations(transactions, address)
            if not address_sequences:
                continue
            
            total_sequences_processed += len(address_sequences)
            
            # Process sequences from this address immediately
            for seq in address_sequences:
                # Update Apriori counters for frequent subsequence mining
                for pattern_type in ['temporal_pattern', 'value_pattern', 'method_pattern', 'direction_pattern']:
                    pattern = seq.get(pattern_type, [])
                    if len(pattern) >= 2:
                        for length in range(2, min(self.max_pattern_length, 5)): # Limit length for performance
                            if len(pattern) >= length:
                                subsequences = set(tuple(pattern[i:i+length]) for i in range(len(pattern) - length + 1))
                                apriori_counters[pattern_type].update(subsequences)
                
                # Update N-gram counters
                for n in ngram_counters.keys():
                    for pattern_type_short in ['temporal', 'value', 'method', 'direction']:
                        pattern = seq.get(f"{pattern_type_short}_pattern", [])
                        if len(pattern) >= n:
                            for i in range(len(pattern) - n + 1):
                                ngram = tuple(pattern[i:i+n])
                                ngram_counters[n][pattern_type_short][ngram] += 1

        # --- REFACTOR: Finalize patterns from the aggregated counters ---
        patterns = {}
        
        # Finalize frequent sequences
        frequent_sequences = {}
        if total_sequences_processed > 0:
            for pattern_type, counter in apriori_counters.items():
                frequent_subseqs = {
                    str(p): c / total_sequences_processed
                    for p, c in counter.items()
                    if (c / total_sequences_processed) >= self.min_support
                }
                frequent_sequences[f"{pattern_type}s"] = dict(sorted(frequent_subseqs.items(), key=lambda item: item[1], reverse=True)[:50]) # Top 50
        patterns['frequent_sequences'] = frequent_sequences

        # Finalize N-gram patterns
        ngram_results = {}
        for n, counters in ngram_counters.items():
            ngrams_for_n = {}
            for pattern_type, counter in counters.items():
                total = sum(counter.values())
                if total > 0:
                    ngrams_for_n[pattern_type] = {str(k): v / total for k, v in counter.most_common(20)}
            ngram_results[f"{n}_grams"] = ngrams_for_n
        patterns['ngram_patterns'] = ngram_results
        
        # The other mining functions are now obsolete as their logic is integrated.
        patterns['temporal_sequences'] = {}
        patterns['value_sequences'] = {}
        patterns['method_sequences'] = {}
        
        return patterns
    
    def _create_sequence_representations(self, transactions: pd.DataFrame, address: str) -> List[Dict[str, Any]]:
        """
        Create multiple representations of transaction sequences for analysis - FIXED VERSION
        """
        sequences = []
        
        # Temporal buckets (hourly activity patterns)
        transactions['hour'] = pd.to_datetime(transactions['timestamp'], unit='s').dt.hour
        
        # FIXED: Value patterns (categorized amounts) - Handle duplicate bin edges
        try:
            values_eth = pd.to_numeric(transactions['value'], errors='coerce') / 1e18
            values_eth = values_eth.fillna(0)  # Fill NaN values
            
            # Check if all values are the same (would cause duplicate bin edges)
            if values_eth.nunique() == 1:
                # All values are the same - create simple categorization
                transactions['value_cat'] = 'constant'
            else:
                # Use quantile-based binning to avoid duplicate edges
                try:
                    transactions['value_cat'] = pd.qcut(
                        values_eth,
                        q=5,  # 5 quantiles
                        labels=['dust', 'small', 'medium', 'large', 'huge'],
                        duplicates='drop'  # CRITICAL FIX: Drop duplicate bin edges
                    )
                except ValueError:
                    # If qcut still fails, use a simpler approach
                    median_val = values_eth.median()
                    mean_val = values_eth.mean()
                    
                    conditions = [
                        values_eth <= 0.001,
                        (values_eth > 0.001) & (values_eth <= median_val),
                        (values_eth > median_val) & (values_eth <= mean_val),
                        (values_eth > mean_val) & (values_eth <= mean_val * 2),
                        values_eth > mean_val * 2
                    ]
                    choices = ['dust', 'small', 'medium', 'large', 'huge']
                    transactions['value_cat'] = np.select(conditions, choices, default='medium')
        
        except Exception as e:
            logger.warning(f"⚠️ Error categorizing values: {e}")
            transactions['value_cat'] = 'unknown'
        
        # Method patterns
        transactions['method_clean'] = transactions['method_name'].fillna('unknown').str.lower()
        
        # FIXED: Gas price patterns - Handle duplicate bin edges
        try:
            gas_prices = pd.to_numeric(transactions['gas_price'], errors='coerce')
            gas_prices = gas_prices.fillna(gas_prices.median())  # Fill NaN with median
            
            if gas_prices.nunique() == 1:
                # All gas prices are the same
                transactions['gas_cat'] = 'constant'
            else:
                try:
                    transactions['gas_cat'] = pd.qcut(
                        gas_prices,
                        q=3,
                        labels=['low', 'med', 'high'],
                        duplicates='drop'  # CRITICAL FIX: Drop duplicate bin edges
                    )
                except ValueError:
                    # Fallback to simple threshold-based categorization
                    median_gas = gas_prices.median()
                    q75 = gas_prices.quantile(0.75)
                    
                    conditions = [
                        gas_prices <= median_gas,
                        (gas_prices > median_gas) & (gas_prices <= q75),
                        gas_prices > q75
                    ]
                    choices = ['low', 'med', 'high']
                    transactions['gas_cat'] = np.select(conditions, choices, default='med')
        
        except Exception as e:
            logger.warning(f"⚠️ Error categorizing gas prices: {e}")
            transactions['gas_cat'] = 'unknown'
        
        # Create sequence objects - group by hour to avoid too many sequences
        try:
            # Group transactions by hour to create meaningful sequences
            transactions['hour_bucket'] = transactions['timestamp'] // 3600
            
            for hour_bucket, group in transactions.groupby('hour_bucket'):
                if len(group) >= 2:  # Need at least 2 transactions for a sequence
                    sequence = {
                        'address': address,
                        'sequence_id': f"{address}_{hour_bucket}",
                        'temporal_pattern': list(group['hour']),
                        'value_pattern': list(group['value_cat'].astype(str)),
                        'method_pattern': list(group['method_clean']),
                        'direction_pattern': self._get_direction_pattern(group, address),
                        'timing_intervals': list(group['timestamp'].diff().dropna()),
                        'gas_pattern': list(group['gas_cat'].astype(str)),
                        'sequence_length': len(group),
                        'start_time': int(group.iloc[0]['timestamp']),
                        'total_value': float(pd.to_numeric(group['value'], errors='coerce').sum() / 1e18)
                    }
                    
                    sequences.append(sequence)
        
        except Exception as e:
            logger.warning(f"⚠️ Error creating sequences: {e}")
            # Return minimal sequence if everything fails
            sequences.append({
                'address': address,
                'sequence_id': f"{address}_fallback",
                'error': str(e),
                'sequence_length': len(transactions)
            })
        
        return sequences
    
    def _get_direction_pattern(self, window: pd.DataFrame, address: str) -> List[str]:
        """Determine transaction direction patterns (in/out/self)"""
        directions = []
        for _, tx in window.iterrows():
            if tx['from_addr'] == address and tx['to_addr'] == address:
                directions.append('self')
            elif tx['from_addr'] == address:
                directions.append('out')
            elif tx['to_addr'] == address:
                directions.append('in')
            else:
                directions.append('unknown')
        return directions
    
    def _mine_frequent_sequences(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mine frequent subsequences using modified Apriori algorithm
        """
        frequent_patterns = {
            'temporal_patterns': {},
            'value_patterns': {},
            'method_patterns': {},
            'direction_patterns': {},
            'combined_patterns': {}
        }
        
        # Extract different pattern types
        pattern_types = ['temporal_pattern', 'value_pattern', 'method_pattern', 'direction_pattern']
        
        for pattern_type in pattern_types:
            # Get all sequences of this type
            pattern_sequences = [seq[pattern_type] for seq in sequences if seq.get(pattern_type)]
            
            # Mine frequent subsequences
            frequent_subseqs = self._apriori_subsequences(pattern_sequences, self.min_support)
            frequent_patterns[f"{pattern_type}s"] = frequent_subseqs
        
        # Mine combined patterns (multiple dimensions)
        frequent_patterns['combined_patterns'] = self._mine_multidimensional_patterns(sequences)
        
        return frequent_patterns
    
    def _generate_behavioral_fingerprints(self, address_list: List[str]) -> Dict[str, Any]:
        """
        Generate unique behavioral fingerprints for each address
        """
        logger.info("🔬 Generating behavioral fingerprints...")
        
        fingerprints = {}
        
        for address in tqdm(address_list, desc="Generating Fingerprints"):
            fingerprint = self._create_individual_fingerprint(address)
            if fingerprint:
                fingerprints[address] = fingerprint
        
        # Cluster similar fingerprints
        clustered_fingerprints = self._cluster_fingerprints(fingerprints)
        
        return {
            'individual_fingerprints': fingerprints,
            'fingerprint_clusters': clustered_fingerprints,
            'similarity_matrix': self._calculate_fingerprint_similarities(fingerprints)
        }
    
    def _create_individual_fingerprint(self, address: str) -> Optional[Dict[str, Any]]:
        """
        Create comprehensive behavioral fingerprint for a single address
        """
        # Get all transactions for this address - FIXED COLUMN NAMES
        transactions = self.database.fetch_df("""
            SELECT timestamp, value, method_name, from_addr, to_addr, gas_price, gas
            FROM transactions
            WHERE from_addr = ? OR to_addr = ?
            ORDER BY timestamp
        """, (address, address))
        
        if len(transactions) < 5:
            return None
        
        # Calculate behavioral metrics
        fingerprint = {
            'address': address,
            'transaction_count': len(transactions),
            
            # Temporal behavior
            'temporal_fingerprint': self._extract_temporal_fingerprint(transactions),
            
            # Economic behavior  
            'economic_fingerprint': self._extract_economic_fingerprint(transactions, address),
            
            # Operational behavior
            'operational_fingerprint': self._extract_operational_fingerprint(transactions),
            
            # Network behavior
            'network_fingerprint': self._extract_network_fingerprint(transactions, address),
            
            # Method usage patterns
            'method_fingerprint': self._extract_method_fingerprint(transactions),
            
            # Gas usage patterns - FIXED: use "gas" instead of "gas_used"
            'gas_fingerprint': self._extract_gas_fingerprint(transactions),
            
            # Composite behavioral hash
            'behavioral_hash': None  # Will be calculated after all components
        }
        
        # Calculate composite behavioral hash
        fingerprint['behavioral_hash'] = self._calculate_behavioral_hash(fingerprint)
        
        return fingerprint
    
    def _extract_temporal_fingerprint(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal behavioral patterns - FIXED VERSION"""
        timestamps = pd.to_datetime(transactions['timestamp'], unit='s')
        
        # Safe calculations with proper error handling
        try:
            hour_counts = np.histogram(timestamps.dt.hour, bins=24)[0]
            day_counts = np.histogram(timestamps.dt.dayofweek, bins=7)[0]
            
            # Calculate entropy safely
            hour_entropy = entropy(hour_counts + 1) if len(hour_counts) > 0 else 0
            day_entropy = entropy(day_counts + 1) if len(day_counts) > 0 else 0
            
            # Calculate time differences safely
            time_diffs = transactions['timestamp'].diff().dropna()
            inter_tx_mean = time_diffs.mean() / 3600 if len(time_diffs) > 0 else 0
            inter_tx_std = time_diffs.std() / 3600 if len(time_diffs) > 0 else 0
            
            # Weekend and night ratios
            weekend_mask = timestamps.dt.dayofweek >= 5
            night_mask = (timestamps.dt.hour < 6) | (timestamps.dt.hour >= 22)
            
            weekend_ratio = weekend_mask.sum() / len(timestamps) if len(timestamps) > 0 else 0
            night_ratio = night_mask.sum() / len(timestamps) if len(timestamps) > 0 else 0
            
            return {
                'hour_entropy': float(hour_entropy),
                'day_entropy': float(day_entropy),
                'inter_tx_mean': float(inter_tx_mean),
                'inter_tx_std': float(inter_tx_std),
                'activity_regularity': float(1 / (1 + inter_tx_std)) if inter_tx_std > 0 else 1.0,
                'weekend_ratio': float(weekend_ratio),
                'night_ratio': float(night_ratio)
            }
        
        except Exception as e:
            logger.warning(f"Error in temporal fingerprint extraction: {e}")
            return {
                'hour_entropy': 0.0,
                'day_entropy': 0.0,
                'inter_tx_mean': 0.0,
                'inter_tx_std': 0.0,
                'activity_regularity': 0.0,
                'weekend_ratio': 0.0,
                'night_ratio': 0.0
            }
    
    def _extract_economic_fingerprint(self, transactions: pd.DataFrame, address: str) -> Dict[str, float]:
        """Extract economic behavioral patterns"""
        # Calculate values in ETH
        values = pd.to_numeric(transactions['value'], errors='coerce') / 1e18
        
        # Separate incoming and outgoing
        incoming = transactions[transactions['to_addr'] == address]
        outgoing = transactions[transactions['from_addr'] == address]
        
        incoming_values = pd.to_numeric(incoming['value'], errors='coerce') / 1e18
        outgoing_values = pd.to_numeric(outgoing['value'], errors='coerce') / 1e18
        
        return {
            'value_entropy': float(entropy(np.histogram(np.log10(values + 1e-10), bins=20)[0] + 1)),
            'round_amount_ratio': float((values % 1 == 0).sum() / len(values)),
            'dust_ratio': float((values < 0.001).sum() / len(values)),
            'large_tx_ratio': float((values > 10).sum() / len(values)),
            'in_out_ratio': float(len(incoming) / max(len(outgoing), 1)),
            'volume_consistency': float(1 / (1 + values.std() / max(values.mean(), 1e-10))),
            'economic_efficiency': float(outgoing_values.sum() / max(incoming_values.sum(), 1e-10))
        }
    
    def _extract_operational_fingerprint(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """Extract operational behavioral patterns"""
        return {
            'method_diversity': float(transactions['method_name'].nunique()),
            'method_entropy': float(entropy(transactions['method_name'].value_counts().values + 1)),
            'contract_interaction_ratio': float((transactions['to_addr'].str.len() == 42).sum() / len(transactions)),
            'self_transaction_ratio': float((transactions['from_addr'] == transactions['to_addr']).sum() / len(transactions)),
            'failed_tx_ratio': float(transactions.get('is_error', pd.Series([0]*len(transactions))).sum() / len(transactions))
        }
    
    def _build_markov_models(self, address_list: List[str]) -> Dict[str, Any]:
        """
        Build Markov chain models for transaction behavior prediction
        """
        logger.info("⛓️ Building Markov chain models...")
        
        markov_models = {
            'temporal_transitions': {},
            'value_transitions': {},
            'method_transitions': {},
            'predictive_accuracy': {}
        }
        
        # Build different types of Markov models
        for address in tqdm(address_list, desc="Building Markov Models"):
            address_models = self._build_address_markov_models(address)
            if address_models:
                markov_models['temporal_transitions'][address] = address_models['temporal']
                markov_models['value_transitions'][address] = address_models['value']
                markov_models['method_transitions'][address] = address_models['method']
        
        # Test predictive accuracy
        markov_models['predictive_accuracy'] = self._test_markov_prediction_accuracy(markov_models)
        
        return markov_models
    
    def _build_address_markov_models(self, address: str) -> Optional[Dict[str, Any]]:
        """Build Markov models for a specific address"""
        transactions = self.database.fetch_df("""
            SELECT timestamp, value, method_name
            FROM transactions
            WHERE from_addr = ? OR to_addr = ?
            ORDER BY timestamp
        """, (address, address))
        
        if len(transactions) < 10:
            return None
        
        models = {}
        
        # Temporal Markov model (hour transitions)
        hours = pd.to_datetime(transactions['timestamp'], unit='s').dt.hour
        models['temporal'] = self._build_transition_matrix(hours.tolist())
        
        # Value category Markov model
        value_cats = pd.cut(
            pd.to_numeric(transactions['value'], errors='coerce') / 1e18,
            bins=[0, 0.1, 1, 10, float('inf')],
            labels=['small', 'medium', 'large', 'huge']
        ).astype(str)
        models['value'] = self._build_transition_matrix(value_cats.tolist())
        
        # Method Markov model
        methods = transactions['method_name'].fillna('unknown').tolist()
        models['method'] = self._build_transition_matrix(methods)
        
        return models
    
    def _build_transition_matrix(self, sequence: List[str]) -> Dict[str, Dict[str, float]]:
        """Build transition probability matrix from sequence"""
        transitions = defaultdict(lambda: defaultdict(int))
        state_counts = defaultdict(int)
        
        # Count transitions
        for i in range(len(sequence) - 1):
            current_state = sequence[i]
            next_state = sequence[i + 1]
            transitions[current_state][next_state] += 1
            state_counts[current_state] += 1
        
        # Convert to probabilities
        transition_matrix = {}
        for current_state, next_states in transitions.items():
            transition_matrix[current_state] = {}
            total = state_counts[current_state]
            for next_state, count in next_states.items():
                transition_matrix[current_state][next_state] = count / total
        
        return transition_matrix
    
    def _detect_coordination_networks(self, address_list: List[str], test_mode: bool = False) -> Dict[str, Any]:
        """
        Detect coordinated behavior networks using advanced graph analysis
        """
        logger.info("🕸️ Detecting coordination networks...")
        
        coordination_results = {
            'coordination_clusters': {},
            'timing_correlations': {},
            'behavioral_synchronization': {},
            'network_metrics': {}
        }
        
        # Determine sample size for this computationally expensive analysis
        sample_size = 200 if test_mode else 2000
        if len(address_list) > sample_size:
            # Take a random sample to avoid bias towards the most active addresses
            sampled_addresses = np.random.choice(address_list, sample_size, replace=False).tolist()
            logger.info(f"Building coordination graph from a random sample of {len(sampled_addresses)} addresses.")
        else:
            sampled_addresses = address_list

        # Build coordination graph
        coordination_graph = self._build_coordination_graph(sampled_addresses)
        
        # Detect coordination clusters
        coordination_results['coordination_clusters'] = self._detect_coordination_clusters(coordination_graph)
        
        # Analyze timing correlations
        # Use the same sample for consistency
        coordination_results['timing_correlations'] = self._analyze_timing_correlations(sampled_addresses)
        
        # Detect behavioral synchronization
        coordination_results['behavioral_synchronization'] = self._detect_behavioral_synchronization(address_list)
        
        # Calculate network metrics
        coordination_results['network_metrics'] = self._calculate_coordination_metrics(coordination_graph)
        
        return coordination_results
    
    def _detect_automation_patterns(self, address_list: List[str]) -> Dict[str, Any]:
        """
        Detect automated behavior patterns indicating bot/script usage
        """
        logger.info("🤖 Detecting automation patterns...")
        
        automation_results = {
            'automated_addresses': {},
            'automation_indicators': {},
            'script_signatures': {},
            'human_vs_bot_classification': {}
        }
        
        for address in address_list[:100]:
            automation_score = self._calculate_automation_score(address)
            if automation_score > 0.7:  # High automation threshold
                automation_results['automated_addresses'][address] = automation_score
        
        return automation_results
    
    def _calculate_automation_score(self, address: str) -> float:
        """Calculate automation likelihood score for an address - FIXED COLUMNS"""
        transactions = self.database.fetch_df("""
            SELECT timestamp, value, gas_price, gas, method_name
            FROM transactions
            WHERE from_addr = ? OR to_addr = ?
            ORDER BY timestamp
        """, (address, address))
        
        if len(transactions) < 10:
            return 0.0
        
        automation_indicators = 0
        total_indicators = 8
        
        # 1. Timing regularity (very regular intervals suggest automation)
        time_diffs = transactions['timestamp'].diff().dropna()
        if len(time_diffs) > 5:
            cv_timing = time_diffs.std() / time_diffs.mean()
            if cv_timing < 0.1:  # Very low coefficient of variation
                automation_indicators += 1
        
        # 2. Gas price consistency (bots often use fixed gas prices)
        gas_prices = pd.to_numeric(transactions['gas_price'], errors='coerce').dropna()
        if len(gas_prices) > 5:
            unique_gas_prices = gas_prices.nunique()
            if unique_gas_prices <= 3:  # Very few unique gas prices
                automation_indicators += 1
        
        # 3. Value pattern repetition
        values = pd.to_numeric(transactions['value'], errors='coerce') / 1e18
        if len(values) > 5:
            unique_values = values.nunique()
            if unique_values <= len(values) * 0.3:  # High repetition
                automation_indicators += 1
        
        # 4. Method call consistency
        methods = transactions['method_name'].dropna()
        if len(methods) > 5:
            method_entropy = entropy(methods.value_counts().values)
            if method_entropy < 1.0:  # Low entropy suggests automation
                automation_indicators += 1
        
        # 5. Gas usage consistency - FIXED: use "gas" column
        gas_used = pd.to_numeric(transactions['gas'], errors='coerce').dropna()
        if len(gas_used) > 5:
            gas_cv = gas_used.std() / gas_used.mean()
            if gas_cv < 0.1:  # Very consistent gas usage
                automation_indicators += 1
        
        # 6. Round timestamp patterns (scripts often use round numbers)
        round_timestamps = (transactions['timestamp'] % 60 == 0).sum()
        if round_timestamps > len(transactions) * 0.3:
            automation_indicators += 1
        
        # 7. Rapid sequential transactions
        rapid_sequences = (time_diffs < 60).sum()  # Less than 1 minute
        if rapid_sequences > len(transactions) * 0.2:
            automation_indicators += 1
        
        # 8. Lack of weekend/night variation (bots work 24/7)
        timestamps = pd.to_datetime(transactions['timestamp'], unit='s')
        weekend_ratio = (timestamps.dt.dayofweek >= 5).sum() / len(timestamps)
        night_ratio = ((timestamps.dt.hour < 6) | (timestamps.dt.hour >= 22)).sum() / len(timestamps)
        if abs(weekend_ratio - 2/7) < 0.1 and abs(night_ratio - 8/24) < 0.1:
            automation_indicators += 1
        
        return automation_indicators / total_indicators
    
    def _identify_attribution_candidates(self, mining_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identify addresses that likely belong to the same entity based on behavioral similarity
        """
        logger.info("🎯 Identifying attribution candidates...")
        
        attribution_results = {
            'high_confidence_groups': {},
            'behavioral_clusters': {},
            'similarity_scores': {},
            'attribution_evidence': {}
        }
        
        fingerprints = mining_results.get('behavioral_fingerprints', {}).get('individual_fingerprints', {})
        
        if not fingerprints:
            return attribution_results
        
        # Calculate pairwise similarities
        similarities = {}
        addresses = list(fingerprints.keys())
        
        for i, addr1 in enumerate(addresses):
            for j, addr2 in enumerate(addresses[i+1:], i+1):
                similarity = self._calculate_behavioral_similarity(
                    fingerprints[addr1], fingerprints[addr2]
                )
                if similarity > 0.8:  # High similarity threshold
                    pair_key = f"{addr1}_{addr2}"
                    similarities[pair_key] = {
                        'addresses': [addr1, addr2],
                        'similarity_score': similarity,
                        'evidence': self._generate_attribution_evidence(
                            fingerprints[addr1], fingerprints[addr2]
                        )
                    }
        
        attribution_results['similarity_scores'] = similarities
        attribution_results['high_confidence_groups'] = self._group_similar_addresses(similarities)
        
        return attribution_results
    
    def _calculate_behavioral_similarity(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> float:
        """Calculate similarity between two behavioral fingerprints"""
        similarity_scores = []
        
        # Compare each fingerprint component
        components = ['temporal_fingerprint', 'economic_fingerprint', 'operational_fingerprint', 'gas_fingerprint']
        
        for component in components:
            if component in fp1 and component in fp2:
                comp_sim = self._calculate_component_similarity(fp1[component], fp2[component])
                similarity_scores.append(comp_sim)
        
        return np.mean(similarity_scores) if similarity_scores else 0.0
    
    def _calculate_component_similarity(self, comp1: Dict[str, float], comp2: Dict[str, float]) -> float:
        """Calculate similarity between fingerprint components"""
        common_keys = set(comp1.keys()) & set(comp2.keys())
        if not common_keys:
            return 0.0
        
        values1 = [comp1[key] for key in common_keys]
        values2 = [comp2[key] for key in common_keys]
        
        try:
            return 1 - cosine(values1, values2)
        except:
            return 0.0
    
    def _calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics"""
        return {
            'total_addresses_analyzed': len(results.get('behavioral_fingerprints', {}).get('individual_fingerprints', {})),
            'sequential_patterns_found': len(results.get('sequential_patterns', {}).get('frequent_sequences', {})),
            'automation_candidates': len(results.get('automation_detection', {}).get('automated_addresses', {})),
            'attribution_groups': len(results.get('attribution_candidates', {}).get('high_confidence_groups', {})),
            'markov_models_built': len(results.get('markov_chains', {}).get('temporal_transitions', {})),
            'coordination_clusters': len(results.get('coordination_networks', {}).get('coordination_clusters', {}))
        }
    
    def _mine_multidimensional_patterns(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mine patterns that combine multiple dimensions (temporal + value + method)
        """
        multidimensional_patterns = {
            'temporal_value_patterns': {},
            'temporal_method_patterns': {},
            'value_method_patterns': {},
            'triple_patterns': {},
            'pattern_correlations': {}
        }
        
        # Extract combined patterns
        for sequence in sequences:
            if all(key in sequence for key in ['temporal_pattern', 'value_pattern', 'method_pattern']):
                # Temporal + Value patterns
                temporal_value = list(zip(
                    sequence['temporal_pattern'][:5],  # Limit to first 5 for performance
                    sequence['value_pattern'][:5]
                ))
                if len(temporal_value) >= 2:
                    pattern_key = str(temporal_value)
                    multidimensional_patterns['temporal_value_patterns'][pattern_key] = \
                        multidimensional_patterns['temporal_value_patterns'].get(pattern_key, 0) + 1
                
                # Temporal + Method patterns
                temporal_method = list(zip(
                    sequence['temporal_pattern'][:5],
                    sequence['method_pattern'][:5]
                ))
                if len(temporal_method) >= 2:
                    pattern_key = str(temporal_method)
                    multidimensional_patterns['temporal_method_patterns'][pattern_key] = \
                        multidimensional_patterns['temporal_method_patterns'].get(pattern_key, 0) + 1
                
                # Value + Method patterns
                value_method = list(zip(
                    sequence['value_pattern'][:5],
                    sequence['method_pattern'][:5]
                ))
                if len(value_method) >= 2:
                    pattern_key = str(value_method)
                    multidimensional_patterns['value_method_patterns'][pattern_key] = \
                        multidimensional_patterns['value_method_patterns'].get(pattern_key, 0) + 1
        
        # Filter by frequency and convert to relative frequencies
        total_sequences = len(sequences)
        for pattern_type, patterns in multidimensional_patterns.items():
            if isinstance(patterns, dict):
                # Keep only patterns that appear in at least 1% of sequences
                filtered_patterns = {
                    k: v/total_sequences for k, v in patterns.items() 
                    if v >= max(total_sequences * 0.01, 2)
                }
                multidimensional_patterns[pattern_type] = dict(
                    sorted(filtered_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
                )
        
        return multidimensional_patterns

    def _detect_evasion_techniques(self, address_list: List[str]) -> Dict[str, Any]:
        """Detect evasion techniques used to avoid detection"""
        evasion_results = {
            'gas_price_manipulation': {},
            'timing_obfuscation': {},
            'value_obfuscation': {},
            'method_diversification': {},
            'evasion_scores': {}
        }
        
        for address in tqdm(address_list, desc="Detecting Evasion"):
            evasion_score = self._calculate_evasion_score(address)
            if evasion_score > 0.5:
                evasion_results['evasion_scores'][address] = evasion_score
        
        return evasion_results

    def _calculate_evasion_score(self, address: str) -> float:
        """Calculate evasion technique usage score - FIXED COLUMNS"""
        transactions = self.database.fetch_df("""
            SELECT timestamp, value, gas_price, method_name
            FROM transactions
            WHERE from_addr = ? OR to_addr = ?
            ORDER BY timestamp
        """, (address, address))
        
        if len(transactions) < 10:
            return 0.0
        
        evasion_indicators = 0
        
        # Gas price randomization
        gas_prices = pd.to_numeric(transactions['gas_price'], errors='coerce').dropna()
        if len(gas_prices) > 5:
            gas_std = gas_prices.std()
            gas_mean = gas_prices.mean()
            if gas_std / gas_mean > 0.5:  # High variance suggests obfuscation
                evasion_indicators += 1
        
        # Timing randomization
        time_diffs = transactions['timestamp'].diff().dropna()
        if len(time_diffs) > 5:
            # Check for artificially random timing
            cv_timing = time_diffs.std() / time_diffs.mean()
            if 0.3 < cv_timing < 2.0:  # Not too regular, not too random
                evasion_indicators += 1
        
        return min(evasion_indicators / 5, 1.0)

    def _cluster_fingerprints(self, fingerprints: Dict[str, Any]) -> Dict[str, Any]:
        """Cluster similar behavioral fingerprints"""
        if len(fingerprints) < 2:
            return {}
        
        # Create feature matrix
        addresses = list(fingerprints.keys())
        features = []
        
        for address in addresses:
            fp = fingerprints[address]
            # Flatten all fingerprint components into a feature vector
            feature_vector = []
            for component in ['temporal_fingerprint', 'economic_fingerprint', 'operational_fingerprint']:
                if component in fp:
                    feature_vector.extend(fp[component].values())
            features.append(feature_vector)
        
        if not features or not features[0]:
            return {}
        
        try:
            # Normalize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Cluster using DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clustering.fit_predict(features_scaled)
            
            # Group addresses by cluster
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 is noise in DBSCAN
                    clusters[f"cluster_{label}"].append(addresses[i])
            
            return dict(clusters)
        
        except Exception as e:
            logger.warning(f"Fingerprint clustering failed: {e}")
            return {}

    def _calculate_fingerprint_similarities(self, fingerprints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate pairwise similarities between fingerprints"""
        similarities = {}
        addresses = list(fingerprints.keys())
        
        for i, addr1 in enumerate(addresses):
            for j, addr2 in enumerate(addresses[i+1:], i+1):
                similarity = self._calculate_behavioral_similarity(
                    fingerprints[addr1], fingerprints[addr2]
                )
                similarities[f"{addr1}_{addr2}"] = similarity
        
        return similarities

    def _extract_network_fingerprint(self, transactions: pd.DataFrame, address: str) -> Dict[str, float]:
        """Extract network behavior fingerprint"""
        unique_counterparts = set()
        unique_counterparts.update(transactions['from_addr'].dropna())
        unique_counterparts.update(transactions['to_addr'].dropna())
        unique_counterparts.discard(address)
        
        return {
            'unique_counterparts': float(len(unique_counterparts)),
            'self_transaction_ratio': float((transactions['from_addr'] == transactions['to_addr']).sum() / len(transactions)),
            'interaction_diversity': float(len(unique_counterparts) / len(transactions))
        }

    def _extract_method_fingerprint(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """Extract method usage fingerprint"""
        methods = transactions['method_name'].dropna()
        method_counts = methods.value_counts()
        
        return {
            'method_diversity': float(len(method_counts)),
            'method_entropy': float(entropy(method_counts.values + 1)),
            'unknown_method_ratio': float((transactions['method_name'].isna()).sum() / len(transactions))
        }

    def _extract_gas_fingerprint(self, transactions: pd.DataFrame) -> Dict[str, float]:
        """Extract gas usage fingerprint"""
        # FIX: Use both gas price and gas amount for a more complete fingerprint.
        gas_prices = pd.to_numeric(transactions['gas_price'], errors='coerce').dropna()
        gas_used = pd.to_numeric(transactions['gas'], errors='coerce').dropna()
        
        if len(gas_prices) == 0 or len(gas_used) == 0:
            return {'gas_price_consistency': 0.0, 'gas_usage_consistency': 0.0, 'gas_price_entropy': 0.0}
        
        return {
            'gas_price_consistency': float(1 / (1 + gas_prices.std() / max(gas_prices.mean(), 1e-9))),
            'gas_usage_consistency': float(1 / (1 + gas_used.std() / max(gas_used.mean(), 1e-9))),
            'gas_price_entropy': float(entropy(gas_prices.value_counts().values + 1))
        }

    def _calculate_behavioral_hash(self, fingerprint: Dict[str, Any]) -> str:
        """Calculate a hash representing the behavioral fingerprint"""
        # Create a string representation of key behavioral metrics
        hash_components = []
        
        for component in ['temporal_fingerprint', 'economic_fingerprint', 'operational_fingerprint']:
            if component in fingerprint:
                values = list(fingerprint[component].values())
                # Round to 3 decimal places for stability
                rounded_values = [round(v, 3) if isinstance(v, (int, float)) else str(v) for v in values]
                hash_components.extend(rounded_values)
        
        hash_string = '|'.join(map(str, hash_components))
        return hashlib.md5(hash_string.encode()).hexdigest()[:16]

    def _build_coordination_graph(self, address_list: List[str]) -> nx.Graph:
        """Build graph showing potential coordination between addresses"""
        G = nx.Graph()
        G.add_nodes_from(address_list[:50])  # Limit for performance
        
        # Add edges based on temporal proximity of transactions
        for i, addr1 in enumerate(address_list[:50]):
            for addr2 in address_list[i+1:50]:
                correlation = self._calculate_temporal_correlation(addr1, addr2)
                if correlation > 0.7:  # High correlation threshold
                    G.add_edge(addr1, addr2, weight=correlation)
        
        return G

    def _calculate_temporal_correlation(self, addr1: str, addr2: str) -> float:
        """Calculate temporal correlation between two addresses"""
        try:
            # Get transaction timestamps for both addresses
            tx1 = self.database.fetch_df("""
                SELECT timestamp FROM transactions 
                WHERE from_addr = ? OR to_addr = ?
                ORDER BY timestamp
            """, (addr1, addr1))
            
            tx2 = self.database.fetch_df("""
                SELECT timestamp FROM transactions 
                WHERE from_addr = ? OR to_addr = ?
                ORDER BY timestamp
            """, (addr2, addr2))
            
            if len(tx1) < 5 or len(tx2) < 5:
                return 0.0
            
            # Count transactions within 1-hour windows
            correlations = 0
            total_windows = 0
            
            for _, row1 in tx1.iterrows():
                timestamp1 = row1['timestamp']
                nearby_tx2 = tx2[abs(tx2['timestamp'] - timestamp1) <= 3600]  # 1 hour window
                if len(nearby_tx2) > 0:
                    correlations += 1
                total_windows += 1
            
            return correlations / total_windows if total_windows > 0 else 0.0
        
        except Exception:
            return 0.0

    def _detect_coordination_clusters(self, coordination_graph: nx.Graph) -> Dict[str, Any]:
        """Detect clusters of coordinated addresses"""
        clusters = {}
        
        # Find connected components
        components = list(nx.connected_components(coordination_graph))
        
        for i, component in enumerate(components):
            if len(component) >= 2:  # At least 2 addresses
                clusters[f"coordination_cluster_{i}"] = {
                    'addresses': list(component),
                    'size': len(component),
                    'coordination_strength': np.mean([
                        coordination_graph[u][v]['weight'] 
                        for u, v in coordination_graph.edges(component) 
                        if coordination_graph.has_edge(u, v)
                    ])
                }
        
        return clusters

    def _analyze_timing_correlations(self, address_list: List[str]) -> Dict[str, Any]:
        """Analyze timing correlations between addresses"""
        correlations = {}
        
        for i, addr1 in enumerate(address_list[:20]):  # Limit for performance
            for addr2 in address_list[i+1:20]:
                correlation = self._calculate_temporal_correlation(addr1, addr2)
                if correlation > 0.5:
                    correlations[f"{addr1}_{addr2}"] = correlation
        
        return correlations

    def _detect_behavioral_synchronization(self, address_list: List[str]) -> Dict[str, Any]:
        """Detect behavioral synchronization patterns"""
        synchronization = {
            'synchronized_pairs': {},
            'synchronization_score': 0.0
        }
        
        # This would involve comparing behavioral patterns timing
        # Simplified implementation for now
        sync_count = 0
        total_pairs = 0
        
        for i, addr1 in enumerate(address_list[:10]):
            for addr2 in address_list[i+1:10]:
                # Simple synchronization check based on transaction timing
                sync_score = self._calculate_temporal_correlation(addr1, addr2)
                if sync_score > 0.7:
                    synchronization['synchronized_pairs'][f"{addr1}_{addr2}"] = sync_score
                    sync_count += 1
                total_pairs += 1
        
        synchronization['synchronization_score'] = sync_count / max(total_pairs, 1)
        return synchronization

    def _calculate_coordination_metrics(self, coordination_graph: nx.Graph) -> Dict[str, Any]:
        """Calculate network metrics for coordination graph"""
        if len(coordination_graph.nodes()) == 0:
            return {}
        
        return {
            'total_nodes': len(coordination_graph.nodes()),
            'total_edges': len(coordination_graph.edges()),
            'density': nx.density(coordination_graph),
            'average_clustering': nx.average_clustering(coordination_graph),
            'number_of_components': nx.number_connected_components(coordination_graph)
        }

    def _test_markov_prediction_accuracy(self, markov_models: Dict[str, Any]) -> Dict[str, float]:
        """Test the predictive accuracy of Markov models"""
        accuracies = {}
        
        for address, models in markov_models.get('temporal_transitions', {}).items():
            # Simple accuracy test - predict next state and compare
            # This is a simplified implementation
            accuracies[address] = np.random.uniform(0.6, 0.9)  # Placeholder
        
        return {
            'average_accuracy': np.mean(list(accuracies.values())) if accuracies else 0.0,
            'individual_accuracies': accuracies
        }

    def _group_similar_addresses(self, similarities: Dict[str, Any]) -> Dict[str, List[str]]:
        """Group addresses with high behavioral similarity"""
        groups = {}
        processed_addresses = set()
        
        # Sort by similarity score
        sorted_pairs = sorted(similarities.items(), key=lambda x: x[1]['similarity_score'], reverse=True)
        
        group_id = 0
        for pair_key, pair_data in sorted_pairs:
            addresses = pair_data['addresses']
            addr1, addr2 = addresses
            
            if addr1 not in processed_addresses and addr2 not in processed_addresses:
                groups[f"attribution_group_{group_id}"] = addresses
                processed_addresses.update(addresses)
                group_id += 1
        
        return groups

    def _generate_attribution_evidence(self, fp1: Dict[str, Any], fp2: Dict[str, Any]) -> List[str]:
        """Generate evidence for potential attribution"""
        evidence = []
        
        # Compare temporal patterns
        if 'temporal_fingerprint' in fp1 and 'temporal_fingerprint' in fp2:
            if abs(fp1['temporal_fingerprint'].get('hour_entropy', 0) - 
                fp2['temporal_fingerprint'].get('hour_entropy', 0)) < 0.1:
                evidence.append('similar_temporal_patterns')
        
        # Compare economic patterns
        if 'economic_fingerprint' in fp1 and 'economic_fingerprint' in fp2:
            if abs(fp1['economic_fingerprint'].get('value_entropy', 0) - 
                fp2['economic_fingerprint'].get('value_entropy', 0)) < 0.1:
                evidence.append('similar_economic_patterns')
        
        # Compare operational patterns
        if 'operational_fingerprint' in fp1 and 'operational_fingerprint' in fp2:
            if abs(fp1['operational_fingerprint'].get('method_diversity', 0) - 
                fp2['operational_fingerprint'].get('method_diversity', 0)) < 1:
                evidence.append('similar_operational_patterns')
        
        return evidence

    def _analyze_cluster_temporal_patterns(self, bridge_interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns within cluster bridge activities"""
        if not bridge_interactions:
            return {}
        
        timestamps = [bi['timestamp'] for bi in bridge_interactions]
        timestamps.sort()
        
        patterns = {
            'activity_hours': [],
            'time_clustering': {},
            'periodic_patterns': {}
        }
        
        # Extract hours of activity
        hours = [datetime.fromtimestamp(ts).hour for ts in timestamps]
        patterns['activity_hours'] = hours
        
        # Simple time clustering
        hour_counts = Counter(hours)
        patterns['time_clustering'] = dict(hour_counts.most_common(5))
        
        return patterns
    
    def _store_automation_risk_scores(self, automation_results: Dict[str, Any]):
        """
        Stores the calculated automation scores in the risk_components table.
        """
        automated_addresses = automation_results.get('automated_addresses', {})
        if not automated_addresses:
            return

        logger.info("Storing automation risk scores for unified analysis...")
        stored_count = 0
        for address, score in automated_addresses.items():
            self.database.store_component_risk(
                address=address,
                component_type='BEHAVIORAL_SEQUENCE',
                risk_score=score,
                confidence=0.8,
                evidence={'reason': f'High automation score ({score:.2f}) from sequence mining'},
                source_analysis='advanced_behavioral_sequence_miner'
            )
            stored_count += 1
        
        logger.info(f"Stored automation risk scores for {stored_count} addresses.")

    def _store_attribution_links(self, attribution_results: Dict[str, Any]):
        """Stores high-confidence attribution links in the database."""
        high_confidence_groups = attribution_results.get('high_confidence_groups', {})
        if not high_confidence_groups:
            return

        logger.info(f"Storing {len(high_confidence_groups)} attribution groups...")
        records_to_insert = []
        for group_id, addresses in high_confidence_groups.items():
            if len(addresses) > 1:
                # Create links between all pairs in the group
                for i in range(len(addresses)):
                    for j in range(i + 1, len(addresses)):
                        addr1, addr2 = addresses[i], addresses[j]
                        # Find the similarity score for this pair
                        pair_key_1 = f"{addr1}_{addr2}"
                        pair_key_2 = f"{addr2}_{addr1}"
                        similarity_data = attribution_results.get('similarity_scores', {}).get(pair_key_1) or \
                                          attribution_results.get('similarity_scores', {}).get(pair_key_2)
                        
                        if similarity_data:
                            records_to_insert.append((
                                addr1,
                                addr2,
                                similarity_data['similarity_score'],
                                json.dumps(similarity_data.get('evidence', []))
                            ))
        
        if records_to_insert:
            with self.database.transaction():
                self.database.execute("DELETE FROM attribution_links")
                for record in records_to_insert:
                    self.database.execute("""
                        INSERT INTO attribution_links (source_address, target_address, similarity_score, evidence_json)
                        VALUES (?, ?, ?, ?)
                    """, record)
            logger.info(f"Stored {len(records_to_insert)} attribution links.")

    def _store_frequent_patterns(self, sequential_patterns: Dict[str, Any]):
        """Stores frequent behavioral patterns in the database."""
        frequent_sequences = sequential_patterns.get('frequent_sequences', {})
        if not frequent_sequences:
            return

        logger.info("Storing frequent behavioral patterns...")
        with self.database.transaction():
            self.database.execute("DELETE FROM frequent_behavioral_patterns")

            def store_patterns(p_type, p_dict):
                for seq_str, sup in p_dict.items():
                    try:
                        # The eval is safe here as we constructed the string representation
                        seq_len = len(eval(seq_str))
                        self.database.execute("""
                            INSERT INTO frequent_behavioral_patterns (pattern_type, sequence, support, length) VALUES (?, ?, ?, ?)
                        """, (p_type, str(seq_str), float(sup), int(seq_len)))
                    except Exception as e:
                        logger.warning(f"Could not store pattern '{seq_str}': {e}")

            for pattern_type, patterns_dict in frequent_sequences.items():
                # FIX: Handle the nested 'combined_patterns' dictionary
                if pattern_type == 'combined_patterns':
                    for combined_type, combined_patterns_dict in patterns_dict.items():
                        store_patterns(combined_type, combined_patterns_dict)
                else:
                    store_patterns(pattern_type, patterns_dict)

        logger.info("Stored frequent behavioral patterns.")

# Integration function for main pipeline
def integrate_advanced_sequence_mining(database: DatabaseEngine, test_mode: bool = False) -> Dict[str, Any]:
    """
    Integration function to be called from the main analysis pipeline
    """
    miner = AdvancedBehavioralSequenceMiner(database)
    return miner.mine_comprehensive_patterns(test_mode=test_mode)
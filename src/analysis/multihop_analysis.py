# src/analysis/multihop_analysis.py
"""
Multi-hop transaction path analysis for identifying complex routing patterns.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import deque, defaultdict
import networkx as nx
import json
from tqdm import tqdm
from dataclasses import dataclass

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_transaction

logger = logging.getLogger(__name__)

@dataclass
class TransactionPath:
    """Represents a multi-hop transaction path"""
    path_id: str
    addresses: List[str]
    transactions: List[str]
    total_hops: int
    total_volume: float
    time_span: int
    path_type: str
    suspicion_score: float

class MultiHopAnalyzer:
    """
    Analyze multi-hop transaction paths to identify complex money routing patterns.
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.transaction_paths = []
        self.suspicious_paths = []
        
        # Configuration
        self.max_hops = 10
        self.min_path_length = 3
        self.time_window_hours = 24
        
        logger.info("Multi-hop analyzer initialized")
    
    def find_transaction_paths(self, cluster_id: int, max_hops: int = None) -> List[TransactionPath]:
        """
        Finds multi-hop transaction paths for a given cluster by fetching its transactions
        using an efficient JOIN query.
        """
        # Fetch all unique transactions where at least one party is in the cluster
        cluster_txs = self.database.fetch_df("""
            SELECT DISTINCT t.hash, t.from_addr, t.to_addr, t.value, t.timestamp
            FROM transactions t
            JOIN addresses a ON t.from_addr = a.address OR t.to_addr = a.address
            WHERE a.cluster_id = ?
        """, (int(cluster_id),))
        
        return self.find_transaction_paths_from_df(cluster_id, cluster_txs, max_hops)

    def find_transaction_paths_from_df(self, cluster_id: int, cluster_txs: pd.DataFrame, max_hops: int = None) -> List[TransactionPath]:
        """
        Find multi-hop transaction paths within a cluster using an efficient DFS approach.
        REFACTORED: This method now uses a more efficient custom DFS implementation
        to avoid redundant path calculations inherent in calling `nx.all_simple_paths`
        in a loop.
        """
        # Convert cluster_id to Python int to prevent DuckDB type error
        cluster_id = int(cluster_id)
        max_hops = max_hops or self.max_hops
        
        if cluster_txs.empty:
            return []
        
        # Build a NetworkX graph for efficient traversal
        G = nx.DiGraph()
        for _, tx in cluster_txs.iterrows():
            # FIX: Add a check to ensure addresses are not None before adding to graph
            if pd.notna(tx['from_addr']) and pd.notna(tx['to_addr']):
                G.add_edge(
                    tx['from_addr'], 
                    tx['to_addr'], 
                    hash=tx['hash'], 
                    timestamp=tx['timestamp'], 
                    value_eth=float(tx.get('value', 0) or 0) / 1e18
                )
        
        paths = []
        
        # Start DFS only from source nodes (nodes with an in-degree of 0)
        source_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
        
        logger.debug(f"Cluster {cluster_id}: Found {len(source_nodes)} source nodes out of {G.number_of_nodes()} total nodes.")

        for start_node in source_nodes:
            # Use the efficient path generator
            for node_path in self._dfs_path_generator(G, start_node, max_hops):
                # The generator finds all paths; we filter for length here.
                if len(node_path) >= self.min_path_length:
                    path_obj = self._create_transaction_path_from_graph(node_path, G)
                    if path_obj:
                        paths.append(path_obj)
        
        logger.debug(f"Found {len(paths)} significant multi-hop paths in cluster {cluster_id}")
        return paths

    def _dfs_path_generator(self, G: nx.DiGraph, start_node: str, max_hops: int):
        """A generator that yields all simple paths from a start node using DFS."""
        # The stack will store tuples of (node, path_list)
        stack = [(start_node, [start_node])]
        
        while stack:
            current_node, path = stack.pop()
            
            # If it's a leaf node (a sink in the path), this path is complete.
            if G.out_degree(current_node) == 0:
                yield path
                continue # End of this path

            # If path is not too long, explore neighbors
            if len(path) <= max_hops:
                # We iterate in reverse to maintain a more "natural" DFS order
                # when popping from the stack.
                for neighbor in reversed(list(G.successors(current_node))):
                    if neighbor not in path: # Avoid cycles
                        new_path = path + [neighbor]
                        stack.append((neighbor, new_path))
    
    def _create_transaction_path_from_graph(self, node_path: List[str], G: nx.DiGraph) -> Optional[TransactionPath]:
        """
        Create a TransactionPath object from discovered path data.
        """
        # Extract details directly from the graph path
        transactions = []
        total_volume = 0.0
        timestamps = []
        
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i+1]
            edge_data = G.get_edge_data(u, v)
            if edge_data:
                transactions.append(edge_data['hash'])
                total_volume += edge_data['value_eth']
                timestamps.append(edge_data['timestamp'])
        
        time_span = max(timestamps) - min(timestamps) if timestamps else 0
        
        # Determine path type
        path_type = self._classify_path_type(node_path, None) # path_txs no longer needed
        
        # Calculate suspicion score
        suspicion_score = self._calculate_path_suspicion_score(
            node_path, total_volume, time_span, path_type
        )
        
        # FIXED: Prevent NoneType in addresses
        clean_addresses = [addr for addr in node_path if addr is not None]
        path_id = f"path_{hash(''.join(clean_addresses))}"
        
        return TransactionPath(
            path_id=path_id,
            addresses=clean_addresses,
            transactions=transactions,
            total_hops=len(clean_addresses) - 1,
            total_volume=total_volume,
            time_span=time_span,
            path_type=path_type,
            suspicion_score=suspicion_score,
        )
    
    def _classify_path_type(self, addresses: List[str], path_txs: Optional[pd.DataFrame]) -> str:
        """
        Classify the type of transaction path.
        """
        if len(addresses) < 3:
            return "simple"
        
        # Check for specific patterns
        unique_addresses = len(set(addresses))
        total_addresses = len(addresses)
        
        # Linear path (no repeated addresses)
        if unique_addresses == total_addresses:
            if len(addresses) > 7:
                return "long_chain"
            else:
                return "linear_path"
        
        # Check for cycles
        elif unique_addresses < total_addresses:
            return "cyclic_path"
        
        # Check for fan-out pattern
        address_counts = pd.Series(addresses).value_counts()
        if address_counts.max() > 2:
            return "fan_pattern"
        
        return "complex_path"
    
    def _calculate_path_suspicion_score(self, addresses: List[str], total_volume: float, 
                                        time_span: int, path_type: str) -> float:
        """
        REFACTORED: Calculate suspicion score with more contextual logic for value,
        distinguishing between high-value rapid movements and low-value structuring.
        """
        score = 0.0
        num_hops = len(addresses) - 1
        avg_tx_value = total_volume / max(num_hops, 1)
        
        # --- Path Type Scoring ---
        type_scores = {
            'long_chain': 0.2,
            'cyclic_path': 0.4,  # Cycles are highly suspicious
            'fan_pattern': 0.15,
            'complex_path': 0.25,
            'linear_path': 0.05,
            'simple': 0.0
        }
        score += type_scores.get(path_type, 0.0)
        
        # --- Contextual Value & Path Scoring ---

        # 1. High-Value, Rapid Movement (potential "smash and grab" or hack fund dispersal)
        if total_volume > 50 and time_span < 3600 * 6:  # >50 ETH in <6 hours
            # Scale score with volume up to 500 ETH
            score += 0.4 * (min(total_volume, 500) / 500)

        # 2. Structuring / Smurfing (many small transactions in a long chain)
        elif num_hops > 10 and avg_tx_value < 1.0 and total_volume > 10:
            score += 0.35

        # 3. General High Volume (less suspicious if not rapid)
        elif total_volume > 100:
            score += 0.2

        # --- Temporal Scoring ---
        # Rapid execution is more suspicious for longer chains
        if time_span < 3600 and num_hops > 3:  # < 1 hour for a >3 hop chain
            score += 0.2

        return min(1.0, score)
    
    def analyze_path_patterns(self, paths: List[TransactionPath]) -> Dict[str, Any]:
        """
        Analyze patterns across multiple transaction paths.
        """
        if not paths:
            return {'analysis': 'no_paths'}
        
        patterns = {
            'total_paths': len(paths),
            'path_types': {},
            'length_distribution': {},
            'volume_analysis': {},
            'temporal_analysis': {},
            'suspicious_paths': []
        }
        
        # Path type distribution
        type_counts = defaultdict(int)
        for path in paths:
            type_counts[path.path_type] += 1
        patterns['path_types'] = dict(type_counts)
        
        # Length distribution
        lengths = [path.total_hops for path in paths]
        patterns['length_distribution'] = {
            'min_hops': min(lengths),
            'max_hops': max(lengths),
            'avg_hops': np.mean(lengths),
            'median_hops': np.median(lengths)
        }
        
        # Volume analysis
        volumes = [path.total_volume for path in paths]
        patterns['volume_analysis'] = {
            'total_volume_eth': sum(volumes),
            'avg_volume_eth': np.mean(volumes),
            'max_volume_eth': max(volumes),
            'high_volume_paths': len([v for v in volumes if v > 10])
        }
        
        # Temporal analysis
        time_spans = [path.time_span for path in paths]
        patterns['temporal_analysis'] = {
            'avg_time_span_hours': np.mean(time_spans) / 3600,
            'rapid_paths': len([t for t in time_spans if t < 3600]),  # < 1 hour
            'extended_paths': len([t for t in time_spans if t > 86400])  # > 1 day
        }
        
        # Identify most suspicious paths
        suspicious_paths = sorted(paths, key=lambda p: p.suspicion_score, reverse=True)
        patterns['suspicious_paths'] = [
            {
                'path_id': path.path_id,
                'hops': path.total_hops,
                'volume_eth': path.total_volume,
                'suspicion_score': path.suspicion_score,
                'path_type': path.path_type,
                'addresses': path.addresses[:5]  # First 5 addresses
            }
            for path in suspicious_paths[:10]  # Top 10 suspicious paths
        ]
        
        return patterns
    
    def analyze_cluster_multihop(self, cluster_id: int) -> Dict[str, Any]:
        """
        Complete multi-hop analysis for a cluster.
        """
        logger.debug(f"Analyzing multi-hop patterns for cluster {cluster_id}")

        # FIXED: Ensure cluster_id is a standard Python int
        cluster_id = int(cluster_id)
        
        # Find transaction paths
        paths = self.find_transaction_paths(cluster_id)
        
        if not paths:
            return {
                'cluster_id': cluster_id,
                'analysis': 'no_multihop_paths',
                'total_paths': 0
            }
        
        # Analyze patterns
        pattern_analysis = self.analyze_path_patterns(paths)
        
        # Identify suspicious paths
        suspicious_paths = [p for p in paths if p.suspicion_score > 0.5]
        
        # Log suspicious paths
        self._log_suspicious_paths(cluster_id, suspicious_paths)
        
        analysis_result = {
            'cluster_id': cluster_id,
            'total_paths': len(paths),
            'suspicious_paths_count': len(suspicious_paths),
            'pattern_analysis': pattern_analysis,
            'risk_indicators': self._generate_risk_indicators(pattern_analysis)
        }
        
        return analysis_result
    
    def _log_suspicious_paths(self, cluster_id: int, suspicious_paths: List[TransactionPath]):
        """
        Log suspicious multi-hop paths for real-time monitoring.
        """
        for path in suspicious_paths[:5]:  # Log top 5 suspicious paths
            for tx_hash in path.transactions:
                # Get transaction details
                tx_data = self.database.fetch_df("""
                    SELECT * FROM transactions WHERE hash = ?
                """, (tx_hash,))
                
                if not tx_data.empty:
                    tx_row = tx_data.iloc[0].to_dict()
                    
                    reasons = [
                        'multihop_transaction',
                        f'path_length:{path.total_hops}',
                        f'path_type:{path.path_type}',
                        f'suspicion_score:{path.suspicion_score:.2f}'
                    ]
                    
                    log_suspicious_transaction(
                        tx_hash=tx_hash,
                        reason=reasons,
                        metadata={
                            'cluster_id': cluster_id,
                            'path_id': path.path_id,
                            'total_volume_eth': path.total_volume,
                            'time_span_seconds': path.time_span,
                            'addresses': list(path.addresses)[:5]  # First 5 addresses
                        }
                    )
    
    def _generate_risk_indicators(self, pattern_analysis: Dict[str, Any]) -> List[str]:
        """
        Generate risk indicators based on multi-hop pattern analysis.
        """
        indicators = []
        
        # Check for high number of long chains
        length_dist = pattern_analysis.get('length_distribution', {})
        if length_dist.get('max_hops', 0) > 8:
            indicators.append('very_long_transaction_chains')
        
        # Check for high volume multi-hop transactions
        volume_analysis = pattern_analysis.get('volume_analysis', {})
        if volume_analysis.get('high_volume_paths', 0) > 2:
            indicators.append('high_volume_multihop_transactions')
        
        # Check for rapid multi-hop execution
        temporal_analysis = pattern_analysis.get('temporal_analysis', {})
        if temporal_analysis.get('rapid_paths', 0) > 1:
            indicators.append('rapid_multihop_execution')
        
        # Check for complex path types
        path_types = pattern_analysis.get('path_types', {})
        if path_types.get('cyclic_path', 0) > 0:
            indicators.append('cyclic_transaction_paths')
        
        if path_types.get('long_chain', 0) > 0:
            indicators.append('long_chain_transactions')
        
        return indicators
    
    def analyze_all_multihop_patterns(self) -> Dict[str, Any]:
        """
        Analyze multi-hop patterns across all clusters using a scalable approach.
        """
        logger.info("Starting comprehensive multi-hop analysis...")

        # --- OPTIMIZATION: Process one cluster at a time to avoid loading all transactions ---
        logger.info("Fetching all cluster IDs to analyze...")
        clusters_df = self.database.fetch_df("SELECT DISTINCT cluster_id FROM addresses WHERE cluster_id IS NOT NULL ORDER BY cluster_id")
        
        if clusters_df.empty:
            logger.warning("No clusters found for multi-hop analysis.")
            return {'cluster_analyses': [], 'summary': {}}

        all_paths = []
        cluster_analyses = []
        total_paths = 0
        total_suspicious_paths = 0
        all_suspicious_paths_with_cluster = []
        
        logger.info(f"Analyzing paths for {len(clusters_df)} clusters...")
        for cluster_id in tqdm(clusters_df['cluster_id'], desc="Analyzing Multi-Hop Paths"):
            # Find transaction paths for this specific cluster
            paths = self.find_transaction_paths(int(cluster_id))
            if not paths:
                continue

            all_paths.extend(paths)
            
            pattern_analysis = self.analyze_path_patterns(paths)
            suspicious_paths = [p for p in paths if p.suspicion_score > 0.5]

            for path in suspicious_paths:
                all_suspicious_paths_with_cluster.append((cluster_id, path))
            
            analysis_result = {
                'cluster_id': cluster_id,
                'total_paths': len(paths),
                'suspicious_paths_count': len(suspicious_paths),
                'pattern_analysis': pattern_analysis,
                'risk_indicators': self._generate_risk_indicators(pattern_analysis)
            }
            cluster_analyses.append(analysis_result)

            total_paths += len(paths)
            total_suspicious_paths += len(suspicious_paths)

        # Store details of suspicious paths found for reporting
        self._store_suspicious_paths(all_suspicious_paths_with_cluster)

        # NEW: Store aggregated risk scores for each address
        self._store_address_risk_scores(all_paths)
        
        summary = {
            'total_clusters_analyzed': len(cluster_analyses),
            'total_multihop_paths': total_paths,
            'total_suspicious_paths': total_suspicious_paths,
            'clusters_with_multihop': len([a for a in cluster_analyses if a.get('total_paths', 0) > 0])
        }
        
        logger.info(f"Multi-hop analysis complete: {summary}")
        
        return {
            'cluster_analyses': cluster_analyses,
            'summary': summary
        }
    
    def _store_suspicious_paths(self, suspicious_paths_with_cluster: List[Tuple[int, TransactionPath]]):
        """
        Stores suspicious multi-hop path details in the database for later export.
        
        Args:
            suspicious_paths_with_cluster: A list of tuples, where each tuple contains
                                           (cluster_id, TransactionPath object).
        """
        if not suspicious_paths_with_cluster:
            return

        logger.info(f"Storing details for {len(suspicious_paths_with_cluster)} suspicious multi-hop paths...")
        records_to_insert = []
        for cluster_id, path in suspicious_paths_with_cluster:
            records_to_insert.append((
                path.path_id,
                int(cluster_id),
                json.dumps(path.addresses),
                json.dumps(path.transactions),
                int(path.total_hops),
                float(path.total_volume),
                int(path.time_span),
                path.path_type,
                float(path.suspicion_score)
            ))

        with self.database.transaction():
            # Clear old paths before inserting new ones to avoid duplicates from previous runs
            self.database.execute("DELETE FROM suspicious_paths")
            # Use a loop of execute calls since executemany is not available
            for record in records_to_insert:
                self.database.execute("""
                INSERT INTO suspicious_paths 
                (path_id, cluster_id, addresses, transactions, total_hops, total_volume, time_span_seconds, path_type, suspicion_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, record)
        logger.info("Successfully stored suspicious path details.")
    
    def _store_address_risk_scores(self, all_paths: List[TransactionPath]):
        """
        Aggregates path risk to the address level and stores it.
        """
        if not all_paths:
            return

        logger.info("Aggregating multi-hop path risk to individual addresses...")
        address_risk = defaultdict(float)

        # Find the maximum suspicion score for each address across all paths
        for path in all_paths:
            for address in path.addresses:
                if path.suspicion_score > address_risk[address]:
                    address_risk[address] = path.suspicion_score
        
        # Store the scores in the risk_components table
        stored_count = 0
        for address, score in address_risk.items():
            if score > 0.1: # Only store meaningful scores
                self.database.store_component_risk(
                    address=address,
                    component_type='multihop_risk',
                    risk_score=score,
                    confidence=0.85, # Confidence is high as this is based on direct flows
                    evidence={'reason': f'Part of a suspicious transaction path with score {score:.2f}'},
                    source_analysis='multihop_analyzer'
                )
                stored_count += 1
        
        logger.info(f"Stored multi-hop risk scores for {stored_count} addresses.")

# Integration function for main pipeline
def integrate_with_pipeline(database: DatabaseEngine, output_dir: str = None) -> Dict[str, Any]:
    """
    Integration function to be called from the main analysis pipeline.
    """
    analyzer = MultiHopAnalyzer(database=database)
    results = analyzer.analyze_all_multihop_patterns()
    
    return results
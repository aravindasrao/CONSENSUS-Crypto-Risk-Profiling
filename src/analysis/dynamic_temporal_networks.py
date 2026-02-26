# src/analysis/dynamic_temporal_networks.py
"""
Dynamic Temporal Network Analysis for Tornado Cash Transactions
Analyzes how transaction networks evolve over time to detect coordinated mixing behavior
"""
import logging
import pandas as pd
import numpy as np
import networkx as nx

from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from src.core.database import DatabaseEngine

logger = logging.getLogger(__name__)

class DynamicTemporalNetworkAnalyzer:
    """
    Analyzes how transaction networks evolve over time to detect coordinated mixing behavior
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.time_windows = []
        self.network_snapshots = {}
        self.evolution_metrics = {}
        
        logger.info("Dynamic Temporal Network Analyzer initialized")
    
    def analyze_network_evolution(self, 
                                time_window_hours: int = 24,
                                overlap_hours: int = 12) -> Dict[str, Any]:
        """
        Main method to analyze how transaction networks evolve over time
        
        INTEGRATION: Calls your existing database and extends cluster analysis
        """
        logger.info(f"🕐 Starting temporal network evolution analysis...")
        
        # Get temporal boundaries from your existing data
        time_range = self.database.fetch_df("""
            SELECT 
                MIN(timestamp) as min_time,
                MAX(timestamp) as max_time,
                COUNT(*) as total_transactions
            FROM transactions
        """)
        
        if time_range.empty:
            return {'error': 'No transaction data found'}
        
        min_time = int(time_range.iloc[0]['min_time'])
        max_time = int(time_range.iloc[0]['max_time'])
        
        # Create sliding time windows
        self.time_windows = self._create_time_windows(
            min_time, max_time, time_window_hours, overlap_hours
        )
        
        logger.info(f"Created {len(self.time_windows)} time windows for analysis")
        
        # Analyze each time window
        evolution_results = {
            'time_windows': len(self.time_windows),
            'window_analyses': [],
            'evolution_metrics': {},
            'suspicious_patterns': [],
            'coordination_detection': {}
        }

        logger.info("Analyzing each time window...")

        for i, (start_time, end_time) in tqdm(enumerate(self.time_windows), total=len(self.time_windows)):
            # logger.info(f"Analyzing window {i+1}/{len(self.time_windows)}")
            
            window_analysis = self._analyze_time_window(start_time, end_time, i)
            evolution_results['window_analyses'].append(window_analysis)
            
            # Store network snapshot for comparison
            self.network_snapshots[i] = window_analysis.get('network_graph', {})
        
        logger.info("Analyzing evolution patterns across time windows...")
        # Analyze evolution patterns across windows
        evolution_results['evolution_metrics'] = self._analyze_evolution_patterns()
        evolution_results['coordination_detection'] = self._detect_coordination_patterns()
        evolution_results['suspicious_patterns'] = self._identify_temporal_anomalies()

        # Store results in database
        self._store_temporal_results(evolution_results)
        
        logger.info("✅ Temporal network evolution analysis completed")
        return evolution_results
    
    def _create_time_windows(self, min_time: int, max_time: int, 
                           window_hours: int, overlap_hours: int) -> List[Tuple[int, int]]:
        """Create overlapping time windows for analysis"""
        """
        Args:
            min_time: Minimum timestamp in data
            max_time: Maximum timestamp in data
            window_hours: Size of each time window in hours
            overlap_hours: Overlap between consecutive windows in hours
        Returns:
            List of (start_time, end_time) tuples for each window
        """
        window_seconds = window_hours * 3600
        step_seconds = (window_hours - overlap_hours) * 3600
        
        windows = []
        current_start = min_time
        
        while current_start + window_seconds <= max_time:
            current_end = current_start + window_seconds
            windows.append((current_start, current_end))
            current_start += step_seconds
        
        return windows
    
    def _analyze_time_window(self, start_time: int, end_time: int, window_id: int) -> Dict[str, Any]:
        """
        Analyze transaction network for a specific time window
        
        INTEGRATION: Uses your existing database schema and cluster data
        """
        # --- OPTIMIZATION: Use a two-step query to avoid a large JOIN inside the loop ---
        # 1. Fetch only the transactions for the time window, which is fast.
        window_txs = self.database.fetch_df(f"""
            SELECT *
            FROM transactions
            WHERE timestamp >= {start_time} AND timestamp < {end_time}
        """)
        
        if window_txs.empty:
            return {
                'window_id': window_id,
                'start_time': start_time,
                'end_time': end_time,
                'transactions': 0,
                'network_graph': {}
            }

        # 2. Get unique addresses from this window and fetch their metadata.
        # This is more efficient than a large JOIN on the full transactions table.
        from_addrs = window_txs['from_addr'].dropna().unique()
        if len(from_addrs) > 0:
            placeholders = ','.join(['?'] * len(from_addrs))
            address_meta_df = self.database.fetch_df(f"""
                SELECT address, cluster_id, risk_score
                FROM addresses
                WHERE address IN ({placeholders})
            """, tuple(from_addrs))

            # 3. Merge the metadata back into the transactions DataFrame.
            window_txs = window_txs.merge(address_meta_df, left_on='from_addr', right_on='address', how='left')
            window_txs.drop(columns=['address'], inplace=True, errors='ignore')
        else:
            # If no from_addrs, add empty columns to maintain schema consistency.
            window_txs['cluster_id'] = None
            window_txs['risk_score'] = None
        # --- END OPTIMIZATION ---
        
        # Build network graph for this window
        G = self._build_temporal_graph(window_txs)
        
        # Analyze network properties
        network_metrics = self._calculate_network_metrics(G)
        
        # Detect suspicious patterns in this window
        suspicious_activities = self._detect_window_anomalies(window_txs, G)
        
        # INTEGRATION: Link with your existing cluster analysis
        cluster_temporal_analysis = self._analyze_cluster_temporal_behavior(window_txs)
        
        return {
            'window_id': window_id,
            'start_time': start_time,
            'end_time': end_time,
            'transactions': len(window_txs),
            'unique_addresses': len(set(window_txs['from_addr'].tolist() + window_txs['to_addr'].tolist())),
            'network_graph': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'components': nx.number_weakly_connected_components(G)
            },
            'network_metrics': network_metrics,
            'suspicious_activities': suspicious_activities,
            'cluster_analysis': cluster_temporal_analysis,
            'tornado_activity': self._analyze_tornado_activity_in_window(window_txs)
        }
    
    def _build_temporal_graph(self, transactions: pd.DataFrame) -> nx.DiGraph:
        """Build directed graph for temporal analysis"""
        G = nx.DiGraph()
        
        for _, tx in transactions.iterrows():
            from_addr = tx['from_addr']
            to_addr = tx['to_addr']
            
            if pd.notna(from_addr) and pd.notna(to_addr) and from_addr != to_addr:
                # Add edge with temporal information
                if G.has_edge(from_addr, to_addr):
                    G[from_addr][to_addr]['weight'] += float(tx.get('value_eth', 0))
                    G[from_addr][to_addr]['transaction_count'] += 1
                    G[from_addr][to_addr]['timestamps'].append(int(tx['timestamp']))
                else:
                    G.add_edge(from_addr, to_addr,
                             weight=float(tx.get('value_eth', 0)),
                             transaction_count=1,
                             timestamps=[int(tx['timestamp'])],
                             cluster_id=tx.get('cluster_id'),
                             risk_score=tx.get('risk_score', 0))
        
        return G
    
    def _calculate_network_metrics(self, G: nx.DiGraph) -> Dict[str, float]:
        """Calculate comprehensive network metrics"""
        if G.number_of_nodes() == 0:
            return {}
        
        metrics = {
            'avg_degree': np.mean([d for n, d in G.degree()]),
            'avg_in_degree': np.mean([d for n, d in G.in_degree()]),
            'avg_out_degree': np.mean([d for n, d in G.out_degree()]),
            'density': nx.density(G),
            'transitivity': nx.transitivity(G),
            'number_of_components': nx.number_weakly_connected_components(G)
        }
        
        # Calculate centrality measures for top nodes only (performance)
        if G.number_of_nodes() <= 1000:
            try:
                betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
                metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
                metrics['max_betweenness'] = max(betweenness.values()) if betweenness else 0
            except:
                metrics['avg_betweenness'] = 0
                metrics['max_betweenness'] = 0
        
        return metrics
    
    def _analyze_evolution_patterns(self) -> Dict[str, Any]:
        """
        Analyze how network structure evolves over time
        """
        if len(self.network_snapshots) < 2:
            return {'error': 'Insufficient time windows for evolution analysis'}
        
        evolution_metrics = {
            'network_growth': [],
            'density_changes': [],
            'component_evolution': [],
            'stability_score': 0,
            'growth_patterns': {}
        }
        
        # Track metrics across time windows
        for i in range(len(self.network_snapshots) - 1):
            current = self.network_snapshots[i]
            next_window = self.network_snapshots[i + 1]
            
            if current and next_window:
                # Network growth
                node_growth = next_window.get('nodes', 0) - current.get('nodes', 0)
                edge_growth = next_window.get('edges', 0) - current.get('edges', 0)
                
                evolution_metrics['network_growth'].append({
                    'window': i,
                    'node_growth': node_growth,
                    'edge_growth': edge_growth,
                    'growth_rate': edge_growth / max(current.get('edges', 1), 1)
                })
                
                # Density changes
                density_change = next_window.get('density', 0) - current.get('density', 0)
                evolution_metrics['density_changes'].append(density_change)
        
        # Calculate stability score
        if evolution_metrics['density_changes']:
            density_std = np.std(evolution_metrics['density_changes'])
            evolution_metrics['stability_score'] = 1 / (1 + density_std)
        
        return evolution_metrics
    
    def _detect_coordination_patterns(self) -> Dict[str, Any]:
        """
        Detect coordinated behavior across time windows
        
        INTEGRATION: Uses your existing risk scoring and cluster data
        """
        if len(self.network_snapshots) < 2:
            return {'error': 'Insufficient time windows for coordination analysis'}

        coordination_analysis = {
            'synchronized_activities': [],
            'burst_patterns': [],
            'coordination_score': 0,
            'suspicious_synchronization': []
        }

        # 1. Aggregate cluster activity over time
        cluster_activity_timeline = defaultdict(dict)
        for window_id, window_data in self.network_snapshots.items():
            if not window_data or 'cluster_analysis' not in window_data:
                continue
            for cluster_id, cluster_metrics in window_data['cluster_analysis'].items():
                cluster_activity_timeline[cluster_id][window_id] = cluster_metrics['transaction_count']

        # 2. Detect synchronized activation of clusters
        newly_active_counts = []
        for i in range(1, len(self.time_windows)):
            prev_window_id = i - 1
            current_window_id = i

            # Get active clusters in the current and previous windows
            current_active_clusters = set(cluster_activity_timeline.keys()) if current_window_id not in cluster_activity_timeline else set(cluster_id for cluster_id, activity in cluster_activity_timeline.items() if current_window_id in activity)
            prev_active_clusters = set(cluster_activity_timeline.keys()) if prev_window_id not in cluster_activity_timeline else set(cluster_id for cluster_id, activity in cluster_activity_timeline.items() if prev_window_id in activity)

            newly_activated = current_active_clusters - prev_active_clusters
            
            if len(newly_activated) > 1:
                coordination_analysis['synchronized_activities'].append({
                    'window_id': current_window_id,
                    'newly_active_cluster_count': len(newly_activated),
                    'newly_active_clusters': list(newly_activated)[:10] # Sample
                })
            newly_active_counts.append(len(newly_activated))

        # 3. Identify anomalous synchronization events (e.g., > 2 std deviations from mean)
        if newly_active_counts:
            mean_new_clusters = np.mean(newly_active_counts)
            std_new_clusters = np.std(newly_active_counts)
            threshold = mean_new_clusters + 2 * std_new_clusters

            for event in coordination_analysis['synchronized_activities']:
                if event['newly_active_cluster_count'] > threshold:
                    coordination_analysis['suspicious_synchronization'].append(event)

        # Calculate an overall coordination score
        if newly_active_counts:
            coordination_analysis['coordination_score'] = np.mean(newly_active_counts) / max(len(cluster_activity_timeline), 1)

        return coordination_analysis
    
    def _analyze_cluster_temporal_behavior(self, window_txs: pd.DataFrame) -> Dict[str, Any]:
        """
        INTEGRATION: Extend your existing cluster analysis with temporal dimension
        """
        cluster_analysis = {}
        
        # Group by cluster_id and analyze temporal patterns
        for cluster_id, cluster_txs in window_txs.groupby('cluster_id'):
            if pd.isna(cluster_id):
                continue
                
            cluster_analysis[int(cluster_id)] = {
                'transaction_count': len(cluster_txs),
                'unique_addresses': len(set(cluster_txs['from_addr'].tolist() + cluster_txs['to_addr'].tolist())),
                'avg_risk_score': cluster_txs['risk_score'].mean(),
                'temporal_span': cluster_txs['timestamp'].max() - cluster_txs['timestamp'].min(),
                'transaction_frequency': len(cluster_txs) / max((cluster_txs['timestamp'].max() - cluster_txs['timestamp'].min()) / 3600, 1)
            }
        
        return cluster_analysis
    
    def _analyze_tornado_activity_in_window(self, window_txs: pd.DataFrame) -> Dict[str, Any]:
        """
        INTEGRATION: Analyze Tornado Cash specific patterns in time window
        Links with your existing tornado_interactions.py
        """
        # Filter for potential Tornado Cash interactions
        tornado_patterns = window_txs[
            window_txs['method_name'].str.contains('deposit|withdraw|mix', case=False, na=False) |
            (window_txs['value_eth'].isin([0.1, 1.0, 10.0, 100.0]))  # Standard Tornado amounts
        ]
        
        return {
            'tornado_transactions': len(tornado_patterns),
            'standard_amounts_ratio': len(tornado_patterns) / max(len(window_txs), 1),
            'deposit_withdraw_ratio': self._calculate_deposit_withdraw_ratio(tornado_patterns)
        }
    
    def _calculate_deposit_withdraw_ratio(self, tornado_txs: pd.DataFrame) -> float:
        """Calculate ratio of deposits to withdrawals"""
        if tornado_txs.empty:
            return 0
        
        deposits = tornado_txs[tornado_txs['method_name'].str.contains('deposit', case=False, na=False)]
        withdraws = tornado_txs[tornado_txs['method_name'].str.contains('withdraw', case=False, na=False)]
        
        return len(deposits) / max(len(withdraws), 1)
    
    def _detect_window_anomalies(self, window_txs: pd.DataFrame, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """Detect anomalous patterns within a time window"""
        anomalies = []
        
        # High-degree nodes (potential mixing hubs)
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            high_degree_threshold = np.percentile(list(degrees.values()), 95)
            
            for node, degree in degrees.items():
                if degree > high_degree_threshold and degree > 10:
                    anomalies.append({
                        'type': 'high_degree_node',
                        'address': node,
                        'degree': degree,
                        'suspicion_score': min(degree / 100, 1.0)
                    })
        
        # Rapid transaction sequences
        window_txs['time_diff'] = window_txs['timestamp'].diff()
        rapid_txs = window_txs[window_txs['time_diff'] < 60]  # Less than 1 minute
        
        if len(rapid_txs) > 5:
            anomalies.append({
                'type': 'rapid_transaction_sequence',
                'count': len(rapid_txs),
                'suspicion_score': min(len(rapid_txs) / 20, 1.0)
            })
        
        return anomalies
    
    def _identify_temporal_anomalies(self) -> List[Dict[str, Any]]:
        """Identify suspicious patterns across time windows"""
        anomalies = []
        
        # Sudden activity spikes
        activity_counts = []
        for window_data in self.network_snapshots.values():
            activity_counts.append(window_data.get('edges', 0))
        
        if len(activity_counts) > 1:
            activity_mean = np.mean(activity_counts)
            activity_std = np.std(activity_counts)
            
            for i, (count, window_data) in enumerate(zip(activity_counts, self.network_snapshots.values())):
                if count > activity_mean + 2 * activity_std:
                    anomalies.append({
                        'type': 'activity_spike',
                        'window_id': i,
                        'activity_count': count,
                        # +++ NEW: Add timestamps for visualization +++
                        'start_time': window_data.get('start_time'),
                        'end_time': window_data.get('end_time'),
                        'z_score': (count - activity_mean) / max(activity_std, 1),
                        'suspicion_score': min((count - activity_mean) / (3 * activity_std), 1.0)
                    })
        
        return anomalies
    
    # INTEGRATION METHODS
    def integrate_with_existing_pipeline(self, cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        INTEGRATION: Method to connect with your existing analysis pipeline
        Called from run_analysis_enhanced.py
        """
        logger.info("🔗 Integrating temporal analysis with existing pipeline...")
        
        # Run temporal analysis
        temporal_results = self.analyze_network_evolution()
        
        # Enhance cluster results with temporal insights
        enhanced_results = cluster_results.copy()
        enhanced_results['temporal_network_analysis'] = temporal_results
        
        # Cross-reference temporal anomalies with existing clusters
        if 'cluster_discovery' in cluster_results:
            enhanced_results['temporal_cluster_correlation'] = self._correlate_with_clusters(
                temporal_results, cluster_results['cluster_discovery']
            )
        
        return enhanced_results
    
    def _correlate_with_clusters(self, temporal_results: Dict[str, Any], 
                               cluster_results: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate temporal anomalies with existing cluster analysis"""
        correlation = {
            'temporal_suspicious_clusters': [],
            'cluster_temporal_scores': {}
        }
        
        # This method would cross-reference your existing cluster IDs 
        # with temporal anomalies to identify clusters that show 
        # suspicious temporal behavior
        
        return correlation
    
    def _store_temporal_results(self, evolution_results: Dict[str, Any]):
        """Stores the temporal network analysis summary in the database."""
        suspicious_patterns = evolution_results.get('suspicious_patterns', [])
        if not suspicious_patterns:
            logger.info("No suspicious temporal patterns to store.")
            return
            
        # Summarize the findings for storage
        analysis_type = 'dynamic_temporal_network'
        address = 'system_wide_analysis'
        summary_to_store = {
            'time_windows_analyzed': evolution_results.get('time_windows', 0),
            'suspicious_patterns_found': len(suspicious_patterns),
            'top_suspicious_patterns': suspicious_patterns[:5] # Store top 5
        }

        logger.info(f"Storing {analysis_type} results...")

        # First, delete any existing record for this analysis type to prevent duplicates
        self.database.execute(
            "DELETE FROM advanced_analysis_results WHERE address = ? AND analysis_type = ?",
            (address, analysis_type)
        )

        # Use the highest suspicion score as the confidence score for the summary
        confidence = max(p.get('suspicion_score', 0.0) for p in suspicious_patterns) if suspicious_patterns else 0.0

        self.database.store_advanced_analysis_results(
            address=address,
            analysis_type=analysis_type,
            results=summary_to_store,
            confidence_score=confidence,
            severity='HIGH' if confidence > 0.7 else 'MEDIUM'
        )

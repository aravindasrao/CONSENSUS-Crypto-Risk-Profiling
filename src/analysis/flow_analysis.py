# src/analysis/flow_analysis.py
"""
Transaction flow analysis with DuckDB type conversion fixes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, deque
import networkx as nx
import json
from tqdm import tqdm

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_transaction
from src.utils.graph_analysis_utils import GraphTopologyAnalyzer

logger = logging.getLogger(__name__)

class FlowAnalyzer:
    """
    Analyzes transaction flows to identify suspicious money movement patterns.
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.flow_patterns = defaultdict(list)
        self.suspicious_flows = []
        self.topology_analyzer = GraphTopologyAnalyzer()
        
        logger.info("Flow analyzer initialized")
    
    def analyze_cluster_flows(self, cluster_id: int, cluster_txs: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze transaction flows within a specific cluster.
        This method now accepts a DataFrame of transactions to improve performance.
        """
        cluster_id = int(cluster_id)
        
        if cluster_txs.empty:
            return {'cluster_id': cluster_id, 'flow_analysis': 'no_transactions'}
        
        analysis = {
            'cluster_id': cluster_id,
            'total_transactions': len(cluster_txs),
            'flow_patterns': [],
            'suspicious_flows': [],
            'flow_graph_metrics': {},
            'temporal_patterns': {}
        }
        
        # Build flow graph
        flow_graph = self._build_flow_graph(cluster_txs)
        analysis['flow_graph_metrics'] = self.topology_analyzer.analyze_graph_topology(flow_graph)
        
        # Detect flow patterns
        patterns = self._detect_flow_patterns(cluster_txs, flow_graph)
        analysis['flow_patterns'] = patterns
        
        # Temporal flow analysis
        temporal_analysis = self._analyze_temporal_flows(cluster_txs)
        analysis['temporal_patterns'] = temporal_analysis
        
        # Identify suspicious flows
        suspicious = self._identify_suspicious_flows(cluster_txs, patterns, cluster_id)
        analysis['suspicious_flows'] = suspicious

        # FIXED: Store cluster-level analysis results
        if suspicious:
            self._store_flow_analysis(analysis)
        
        return analysis
    
    def _build_flow_graph(self, transactions: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed graph representing transaction flows.
        """
        G = nx.DiGraph()
        
        # Filter out rows with None or invalid addresses before iteration
        transactions_clean = transactions.dropna(subset=['from_addr', 'to_addr']).copy()
        
        # Filter out 'None' as a string if it exists
        transactions_clean = transactions_clean[transactions_clean['from_addr'] != 'None']
        transactions_clean = transactions_clean[transactions_clean['to_addr'] != 'None']
        
        for _, tx in transactions_clean.iterrows():
            from_addr = tx['from_addr']
            to_addr = tx['to_addr']
            
            try:
                # FIXED: Handle different value types and convert safely
                value_raw = tx['value']
                if pd.isna(value_raw) or value_raw == '' or value_raw == 'None':
                    value = 0.0
                else:
                    value = float(str(value_raw)) / 1e18  # Convert to ETH
            except (ValueError, TypeError):
                value = 0.0
            
            # Add edge with transaction data
            if G.has_edge(from_addr, to_addr):
                # Update existing edge
                G[from_addr][to_addr]['weight'] += value
                G[from_addr][to_addr]['transaction_count'] += 1
                G[from_addr][to_addr]['transactions'].append(str(tx['hash']))
                G[from_addr][to_addr]['last_timestamp'] = max(
                    G[from_addr][to_addr]['last_timestamp'], int(tx['timestamp'])
                )
            else:
                # Create new edge
                G.add_edge(from_addr, to_addr,
                          weight=value,
                          transaction_count=1,
                          transactions=[str(tx['hash'])],
                          first_timestamp=int(tx['timestamp']),
                          last_timestamp=int(tx['timestamp']))
        
        return G
    
    def _detect_flow_patterns(self, transactions: pd.DataFrame, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Detect specific flow patterns that might indicate suspicious activity.
        """
        patterns = []
        
        # Pattern 1: Rapid flow chains
        rapid_chains = self._detect_rapid_flow_chains(transactions)
        if rapid_chains:
            patterns.append({
                'type': 'rapid_flow_chains',
                'count': len(rapid_chains),
                'description': 'Quick succession of transactions through multiple addresses',
                'details': rapid_chains[:5]  # Top 5 examples
            })
        
        # Pattern 2: Fan-out patterns (one address to many)
        fan_outs = self._detect_fan_out_patterns(G)
        if fan_outs:
            patterns.append({
                'type': 'fan_out_patterns',
                'count': len(fan_outs),
                'description': 'Single address distributing to many addresses',
                'details': fan_outs[:5]
            })
        
        # Pattern 3: Fan-in patterns (many to one)
        fan_ins = self._detect_fan_in_patterns(G)
        if fan_ins:
            patterns.append({
                'type': 'fan_in_patterns',
                'count': len(fan_ins),
                'description': 'Multiple addresses sending to single address',
                'details': fan_ins[:5]
            })
        
        # Pattern 4: Circular flows
        circular_flows = self._detect_circular_flows(G)
        if circular_flows:
            patterns.append({
                'type': 'circular_flows',
                'count': len(circular_flows),
                'description': 'Funds flowing in circles between addresses',
                'details': circular_flows[:5]
            })
        
        # Pattern 5: Layered flows (onion routing)
        layered_flows = self._detect_layered_flows(transactions)
        if layered_flows:
            patterns.append({
                'type': 'layered_flows',
                'count': len(layered_flows),
                'description': 'Funds flowing through multiple layers of addresses',
                'details': layered_flows[:5]
            })
        
        return patterns
    
    def _detect_rapid_flow_chains(self, transactions: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect chains of rapid transactions.
        """
        chains = []
        
        # Sort by timestamp
        sorted_txs = transactions.sort_values('timestamp')
        
        # Group consecutive transactions
        current_chain = []
        
        for i, (_, tx) in enumerate(sorted_txs.iterrows()):
            if i == 0:
                current_chain = [tx]
                continue
            
            prev_tx = current_chain[-1]
            time_diff = int(tx['timestamp']) - int(prev_tx['timestamp'])
            
            # If transactions are within 1 hour and connected
            if (time_diff < 3600 and 
                (prev_tx['to_addr'] == tx['from_addr'] or 
                 prev_tx['from_addr'] == tx['to_addr'])):
                current_chain.append(tx)
            else:
                # Process current chain if long enough
                if len(current_chain) >= 3:
                    chains.append(self._analyze_chain(current_chain))
                current_chain = [tx]
        
        # Don't forget the last chain
        if len(current_chain) >= 3:
            chains.append(self._analyze_chain(current_chain))
        
        return chains
    
    def _analyze_chain(self, chain: List[pd.Series]) -> Dict[str, Any]:
        """
        Analyze a chain of transactions.
        """
        total_time = int(chain[-1]['timestamp']) - int(chain[0]['timestamp'])
        
        try:
            values = []
            for tx in chain:
                value_raw = tx['value']
                if pd.isna(value_raw) or value_raw == '' or value_raw == 'None':
                    values.append(0.0)
                else:
                    values.append(float(str(value_raw)) / 1e18)
            total_value = sum(values)
        except:
            values = [0] * len(chain)
            total_value = 0
        
        addresses = set()
        for tx in chain:
            addresses.add(str(tx['from_addr']))
            addresses.add(str(tx['to_addr']))
        
        return {
            'length': len(chain),
            'duration_seconds': total_time,
            'total_value_eth': float(total_value),
            'unique_addresses': list(addresses),
            'avg_value_eth': float(np.mean(values)) if values else 0,
            'first_tx': str(chain[0]['hash']),
            'last_tx': str(chain[-1]['hash'])
        }
    
    def _detect_fan_out_patterns(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Detect addresses that send to many other addresses (potential distribution).
        """
        fan_outs = []
        
        for node in G.nodes():
            out_degree = G.out_degree(node)
            
            if out_degree >= 5:  # Threshold for fan-out
                successors = list(G.successors(node))
                total_out = sum(G[node][succ]['weight'] for succ in successors)
                
                # Check if values are similar (potential structured distribution)
                values = [G[node][succ]['weight'] for succ in successors]
                value_std = float(np.std(values)) if len(values) > 1 else 0
                value_mean = float(np.mean(values)) if values else 0
                
                fan_outs.append({
                    'source_address': str(node),
                    'recipient_count': int(out_degree),
                    'total_value_eth': float(total_out),
                    'avg_value_eth': value_mean,
                    'value_consistency': float(1 - (value_std / max(value_mean, 0.001)))
                })
        
        return sorted(fan_outs, key=lambda x: x['recipient_count'], reverse=True)
    
    def _detect_fan_in_patterns(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Detect addresses that receive from many other addresses (potential collection).
        """
        fan_ins = []
        
        for node in G.nodes():
            in_degree = G.in_degree(node)
            
            if in_degree >= 5:  # Threshold for fan-in
                predecessors = list(G.predecessors(node))
                total_in = sum(G[pred][node]['weight'] for pred in predecessors)
                
                values = [G[pred][node]['weight'] for pred in predecessors]
                value_std = float(np.std(values)) if len(values) > 1 else 0
                value_mean = float(np.mean(values)) if values else 0
                
                fan_ins.append({
                    'destination_address': str(node),
                    'sender_count': int(in_degree),
                    'total_value_eth': float(total_in),
                    'avg_value_eth': value_mean,
                    'value_consistency': float(1 - (value_std / max(value_mean, 0.001)))
                })
        
        return sorted(fan_ins, key=lambda x: x['sender_count'], reverse=True)
    
    def _detect_circular_flows(self, G: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Detect circular flows in the transaction graph.
        """
        circular_flows = []
        
        try:
            # Find strongly connected components (cycles)
            sccs = list(nx.strongly_connected_components(G))
            
            for scc in sccs:
                if len(scc) > 1:  # Actual cycle (more than one node)
                    subgraph = G.subgraph(scc)
                    
                    # Calculate total flow in the cycle
                    total_flow = sum(data['weight'] for _, _, data in subgraph.edges(data=True))
                    
                    circular_flows.append({
                        'addresses_in_cycle': [str(addr) for addr in list(scc)],
                        'cycle_size': len(scc),
                        'total_flow_eth': float(total_flow),
                        'avg_flow_per_address': float(total_flow / len(scc))
                    })
            
        except Exception as e:
            logger.warning(f"Error detecting circular flows: {e}")
        
        return sorted(circular_flows, key=lambda x: x['total_flow_eth'], reverse=True)
    
    def _detect_layered_flows(self, transactions: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect layered flows (transactions going through multiple intermediaries).
        """
        layered_flows = []
        
        # Sort transactions by timestamp
        sorted_txs = transactions.sort_values('timestamp')
        
        # Build address sequences
        address_sequences = defaultdict(list)
        
        for _, tx in sorted_txs.iterrows():
            from_addr = str(tx['from_addr'])
            to_addr = str(tx['to_addr'])
            
            # Look for connections where previous to_addr becomes from_addr
            for seq_id, sequence in address_sequences.items():
                if sequence and sequence[-1]['to_addr'] == from_addr:
                    # Extend existing sequence
                    sequence.append({
                        'from_addr': from_addr,
                        'to_addr': to_addr,
                        'value': str(tx['value']),
                        'timestamp': int(tx['timestamp']),
                        'hash': str(tx['hash'])
                    })
                    break
            else:
                # Start new sequence
                seq_id = len(address_sequences)
                address_sequences[seq_id] = [{
                    'from_addr': from_addr,
                    'to_addr': to_addr,
                    'value': str(tx['value']),
                    'timestamp': int(tx['timestamp']),
                    'hash': str(tx['hash'])
                }]
        
        # Analyze sequences for layering
        for seq_id, sequence in address_sequences.items():
            if len(sequence) >= 3:  # At least 3 layers
                duration = sequence[-1]['timestamp'] - sequence[0]['timestamp']
                
                try:
                    values = []
                    for tx in sequence:
                        value_raw = tx['value']
                        if pd.isna(value_raw) or value_raw == '' or value_raw == 'None':
                            values.append(0.0)
                        else:
                            values.append(float(str(value_raw)) / 1e18)
                    total_value = sum(values)
                except:
                    values = [0] * len(sequence)
                    total_value = 0
                
                unique_addresses = set()
                for tx in sequence:
                    unique_addresses.add(tx['from_addr'])
                    unique_addresses.add(tx['to_addr'])
                
                layered_flows.append({
                    'sequence_id': seq_id,
                    'layer_count': len(sequence),
                    'duration_seconds': duration,
                    'total_value_eth': float(total_value),
                    'unique_addresses': list(unique_addresses),
                    'transactions': [tx['hash'] for tx in sequence]
                })
        
        return sorted(layered_flows, key=lambda x: x['layer_count'], reverse=True)
    
    def _analyze_temporal_flows(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temporal patterns in transaction flows.
        """
        if transactions.empty:
            return {}
        
        # Convert timestamp to datetime
        transactions['datetime'] = pd.to_datetime(transactions['timestamp'], unit='s')
        
        temporal_analysis = {}
        
        # Time-based aggregations
        transactions['hour'] = transactions['datetime'].dt.hour
        transactions['day_of_week'] = transactions['datetime'].dt.dayofweek
        
        # Peak activity hours
        hourly_counts = transactions['hour'].value_counts()
        temporal_analysis['peak_hours'] = hourly_counts.head(3).to_dict()
        
        # Weekend vs weekday activity
        weekend_txs = transactions[transactions['day_of_week'].isin([5, 6])]
        weekday_txs = transactions[~transactions['day_of_week'].isin([5, 6])]
        
        temporal_analysis['weekend_ratio'] = float(len(weekend_txs) / max(len(transactions), 1))
        
        # Burst detection (periods of high activity)
        transactions_per_hour = transactions.set_index('datetime').resample('H').size()
        mean_hourly = float(transactions_per_hour.mean())
        std_hourly = float(transactions_per_hour.std())
        
        burst_threshold = mean_hourly + 2 * std_hourly
        burst_periods = transactions_per_hour[transactions_per_hour > burst_threshold]
        
        temporal_analysis['burst_periods'] = len(burst_periods)
        temporal_analysis['max_hourly_transactions'] = int(transactions_per_hour.max())
        temporal_analysis['avg_hourly_transactions'] = mean_hourly
        
        return temporal_analysis
    
    def _identify_suspicious_flows(self, transactions: pd.DataFrame, 
                                 patterns: List[Dict[str, Any]], 
                                 cluster_id: int) -> List[Dict[str, Any]]:
        """
        Identify flows that are potentially suspicious.
        """
        suspicious_flows = []
        
        # FIXED: Ensure a risk_score is passed to the logging function
        for pattern in patterns:
            pattern_type = pattern['type']
            
            if pattern_type == 'rapid_flow_chains':
                for chain in pattern['details']:
                    if (chain['total_value_eth'] > 10 and 
                        chain['duration_seconds'] < 1800):
                        
                        suspicious_flows.append({
                            'type': 'suspicious_rapid_chain',
                            'risk_score': 0.8,
                            'description': f"Rapid chain: {chain['total_value_eth']:.2f} ETH through {chain['length']} transactions in {chain['duration_seconds']}s",
                            'evidence': chain
                        })
                        
                        # FIXED: Pass addresses and a risk score to the logging function
                        self._log_chain_transactions(chain, cluster_id, 'rapid_flow_chain', risk_score=0.8, addresses=chain['unique_addresses'])
            
            elif pattern_type == 'fan_out_patterns':
                # Large fan-outs with consistent values are suspicious
                for fan_out in pattern['details']:
                    if (fan_out['total_value_eth'] > 50 and 
                        fan_out['value_consistency'] > 0.8):
                        
                        suspicious_flows.append({
                            'type': 'suspicious_fan_out',
                            'risk_score': 0.7,
                            'description': f"Large consistent distribution: {fan_out['total_value_eth']:.2f} ETH to {fan_out['recipient_count']} addresses",
                            'evidence': fan_out
                        })
            
            elif pattern_type == 'circular_flows':
                # Any significant circular flow is suspicious
                for cycle in pattern['details']:
                    if cycle['total_flow_eth'] > 5:
                        
                        suspicious_flows.append({
                            'type': 'suspicious_circular_flow',
                            'risk_score': 0.9,
                            'description': f"Circular flow: {cycle['total_flow_eth']:.2f} ETH through {cycle['cycle_size']} addresses",
                            'evidence': cycle
                        })
            
            elif pattern_type == 'layered_flows':
                # Deep layering with significant value is suspicious
                for layer in pattern['details']:
                    if (layer['layer_count'] > 5 and 
                        layer['total_value_eth'] > 20):
                        
                        suspicious_flows.append({
                            'type': 'suspicious_layered_flow',
                            'risk_score': 0.8,
                            'description': f"Deep layering: {layer['total_value_eth']:.2f} ETH through {layer['layer_count']} layers",
                            'evidence': layer
                        })
        
        return suspicious_flows
    
    def _log_chain_transactions(self, chain: Dict[str, Any], cluster_id: int, pattern_type: str, risk_score: float, addresses: Set[str]):
        """
        Log individual transactions from a suspicious chain.
        """
        first_tx = chain.get('first_tx')
        last_tx = chain.get('last_tx')
        
        if first_tx:
            log_suspicious_transaction(
                tx_hash=first_tx,
                reason=f"suspicious_flow_chain_start; {pattern_type}",
                # FIXED: Remove the 'confidence' keyword and add it to metadata
                metadata={'cluster_id': cluster_id, 'chain_info': chain, 'risk_score': risk_score, 'involved_addresses': list(addresses)}
            )
        
        if last_tx and last_tx != first_tx:
            log_suspicious_transaction(
                tx_hash=last_tx,
                reason=f"suspicious_flow_chain_end; {pattern_type}",
                # FIXED: Remove the 'confidence' keyword and add it to metadata
                metadata={'cluster_id': cluster_id, 'chain_info': chain, 'risk_score': risk_score, 'involved_addresses': list(addresses)}
            )
    
    def _store_flow_analysis(self, analysis_results: Dict[str, Any]):
        """
        NEW: Store detailed flow analysis results in the advanced_analysis_results table.
        """
        try:
            # We will store one record for each cluster's analysis
            cluster_id = analysis_results.get('cluster_id')
            if cluster_id is None:
                logger.warning("No cluster_id found in analysis results, skipping storage.")
                return

            # Check if there are any suspicious flows to store
            suspicious_flows = analysis_results.get('suspicious_flows', [])
            if not suspicious_flows:
                return

            # Calculate a representative risk score for the cluster
            risk_scores = [flow.get('risk_score', 0.0) for flow in suspicious_flows]
            avg_risk_score = float(np.mean(risk_scores)) if risk_scores else 0.0
            
            # Determine overall severity
            if avg_risk_score >= 0.8:
                severity = 'CRITICAL'
            elif avg_risk_score >= 0.6:
                severity = 'HIGH'
            else:
                severity = 'MEDIUM'
            
            # Store the results in the advanced_analysis_results table
            self.database.store_advanced_analysis_results(
                address=f"cluster_{cluster_id}",
                analysis_type='flow_analysis',
                results=analysis_results,
                confidence_score=avg_risk_score,
                severity=severity
            )

            # +++ NEW: Store per-address risk scores for suspicious flows +++
            # Aggregate risk per address involved in suspicious flows
            address_risks = defaultdict(float)
            for flow in suspicious_flows:
                # Evidence can be complex; safely extract addresses
                evidence = flow.get('evidence', {})
                involved_addresses = []
                if 'unique_addresses' in evidence:
                    involved_addresses = evidence['unique_addresses']
                elif 'source_address' in evidence:
                    involved_addresses.append(evidence['source_address'])
                elif 'destination_address' in evidence:
                    involved_addresses.append(evidence['destination_address'])
                
                risk_score = flow.get('risk_score', 0.0)
                for addr in involved_addresses:
                    # Use the max risk score if an address is in multiple flows
                    if risk_score > address_risks[addr]:
                        address_risks[addr] = risk_score
            
            # Store in risk_components table
            if address_risks:
                for addr, score in address_risks.items():
                    self.database.store_component_risk(
                        address=addr,
                        component_type='network_flow', # More specific type
                        risk_score=score,
                        confidence=0.75,
                        evidence={'reason': f'Involved in suspicious flow in cluster {cluster_id}'},
                        source_analysis='flow_analyzer'
                    )
            # Log a success message
            logger.info(f"Stored flow analysis results for cluster {cluster_id} with severity '{severity}'.")
        except Exception as e:
            logger.error(f"Failed to store flow analysis results for cluster {cluster_id}: {e}")

    def analyze_all_flows(self) -> Dict[str, Any]:
        """
        Analyze flows across all clusters using a scalable, bulk-processing approach.
        """
        logger.info("Starting comprehensive flow analysis...")

        # --- OPTIMIZATION: Fetch all clustered transactions at once ---
        logger.info("Fetching all transactions belonging to any cluster...")
        all_clustered_txs_df = self.database.fetch_df("""
            SELECT t.*, a.cluster_id
            FROM transactions t
            JOIN addresses a ON (t.from_addr = a.address OR t.to_addr = a.address)
            WHERE cluster_id IS NOT NULL
            ORDER BY a.cluster_id, t.timestamp
        """)

        if all_clustered_txs_df.empty:
            logger.warning("No clustered transactions found for flow analysis.")
            return {'cluster_analyses': [], 'summary': {}}

        # Group transactions by cluster_id in pandas
        txs_by_cluster = all_clustered_txs_df.groupby('cluster_id')
        logger.info(f"Found {len(txs_by_cluster)} clusters to analyze.")
        # --- END OPTIMIZATION ---

        cluster_analyses = []
        total_suspicious = 0

        for cluster_id, cluster_txs_df in tqdm(txs_by_cluster, desc="Analyzing Cluster Flows"):
            # The analyze_cluster_flows method now takes the DataFrame directly
            analysis = self.analyze_cluster_flows(cluster_id, cluster_txs_df)
            cluster_analyses.append(analysis)
            
            total_suspicious += len(analysis.get('suspicious_flows', []))
        
        summary = {
            'total_clusters_analyzed': len(cluster_analyses),
            'total_suspicious_flows': total_suspicious,
            'clusters_with_suspicious_flows': len([a for a in cluster_analyses if a.get('suspicious_flows')])
        }
        
        logger.info(f"Flow analysis complete: {summary}")
        
        return {
            'cluster_analyses': cluster_analyses,
            'summary': summary
        }

# # Integration function for main pipeline
# def integrate_with_pipeline(database: DatabaseEngine, output_dir: str = None) -> Dict[str, Any]:
#     """
#     Integration function to be called from the main analysis pipeline.
#     """
#     analyzer = FlowAnalyzer(database=database)
#     results = analyzer.analyze_all_flows()
    
#     return results

# # Example usage and testing
# if __name__ == "__main__":
#     # Test the flow analyzer
#     from src.core.database import get_database
    
#     with get_database() as db:
#         analyzer = FlowAnalyzer(database=db)
        
#         # Test single cluster analysis
#         results = analyzer.analyze_cluster_flows(cluster_id=1)
#         print(f"Cluster 1 analysis: {results}")
        
#         # Test full analysis
#         full_results = analyzer.analyze_all_flows()
#         print(f"Full analysis: {full_results['summary']}")
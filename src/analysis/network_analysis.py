# src/analysis/network_analysis.py
"""
Network topology analysis for transaction graphs.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
import networkx as nx
from tqdm import tqdm
from collections import defaultdict, Counter
import math

from src.core.database import DatabaseEngine
from src.utils.transaction_logger import log_suspicious_address
from src.utils.graph_analysis_utils import GraphTopologyAnalyzer

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
    """
    Analyze network topology and structure of transaction graphs.
    """
    
    def __init__(self, database: DatabaseEngine = None):
        self.database = database or DatabaseEngine()
        self.network_metrics = {}
        self.suspicious_networks = []
        self.topology_analyzer = GraphTopologyAnalyzer()
        
        logger.info("Network analyzer initialized")
    
    def build_transaction_network(self, cluster_txs: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed network graph for a cluster from a DataFrame.
        """
        if cluster_txs.empty:
            return nx.DiGraph()
        
        G = nx.DiGraph()

        # FIXED: Filter out rows with None or invalid addresses before iteration
        cluster_txs_clean = cluster_txs.dropna(subset=['from_addr', 'to_addr']).copy()
        
        # You may also want to filter out 'None' as a string if it exists
        cluster_txs_clean = cluster_txs_clean[cluster_txs_clean['from_addr'] != 'None']
        cluster_txs_clean = cluster_txs_clean[cluster_txs_clean['to_addr'] != 'None']
        
        # Add edges with transaction data
        for _, tx in cluster_txs_clean.iterrows():
            from_addr = tx['from_addr']
            to_addr = tx['to_addr']
            
            try:
                value_eth = float(tx['value']) / 1e18
            except:
                value_eth = 0.0
            
            # Add or update edge
            if G.has_edge(from_addr, to_addr):
                G[from_addr][to_addr]['weight'] += value_eth
                G[from_addr][to_addr]['transaction_count'] += 1
                G[from_addr][to_addr]['transactions'].append(tx['hash'])
                G[from_addr][to_addr]['last_timestamp'] = max(
                    G[from_addr][to_addr]['last_timestamp'], tx['timestamp']
                )
            else:
                G.add_edge(from_addr, to_addr,
                          weight=value_eth,
                          transaction_count=1,
                          transactions=[tx['hash']],
                          first_timestamp=tx['timestamp'],
                          last_timestamp=tx['timestamp'])
        
        return G
    
    def analyze_network_topology(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze the topology of a transaction network.
        """
        return self.topology_analyzer.analyze_graph_topology(G)
    
    def identify_suspicious_networks(self, cluster_id: int, topology: Dict[str, Any]) -> List[str]:
        """Identify suspicious network patterns"""
        suspicious_indicators = []
        
        basic_stats = topology.get('basic_stats', {})
        structural_patterns = topology.get('structural_patterns', {})
        flow_patterns = topology.get('flow_patterns', {})
        
        # High density networks (potential coordination)
        if basic_stats.get('density', 0) > 0.5 and basic_stats.get('nodes', 0) > 10:
            suspicious_indicators.append('high_density_network')
        
        # Star patterns (potential mixing or distribution)
        star_patterns = structural_patterns.get('star_patterns', [])
        if len(star_patterns) > 0:
            suspicious_indicators.append(f'star_patterns:{len(star_patterns)}')
        
        # Cycle patterns (potential wash trading)
        cycles = structural_patterns.get('cycles', [])
        if len(cycles) > 2:
            suspicious_indicators.append(f'cycle_patterns:{len(cycles)}')
        
        # Flow concentration (potential coordinated activity)
        flow_gini = flow_patterns.get('flow_gini', 0)
        if flow_gini > 0.8:
            suspicious_indicators.append('concentrated_flows')
        
        # Large volume flows
        total_volume = flow_patterns.get('total_volume_eth', 0)
        if total_volume > 1000:  # > 1000 ETH
            suspicious_indicators.append(f'large_volume_network:{total_volume:.1f}ETH')
        
        return suspicious_indicators
    
    def analyze_cluster_network(self, cluster_id: int, cluster_txs: pd.DataFrame) -> Dict[str, Any]:
        """Complete network analysis for a cluster"""
        logger.debug(f"Analyzing network topology for cluster {cluster_id}")

        # FIXED: Explicitly cast cluster_id to a native Python integer.
        # This prevents the numpy.int32 type from causing conversion errors.
        native_cluster_id = int(cluster_id) 
        
        # Build network from the provided DataFrame
        G = self.build_transaction_network(cluster_txs)

        if G.number_of_nodes() == 0:
            return {'cluster_id': native_cluster_id, 'analysis': 'no_network_data'}

        # Analyze topology
        topology = self.analyze_network_topology(G)
        
        # Identify suspicious patterns
        suspicious_indicators = self.identify_suspicious_networks(native_cluster_id, topology)

        # Calculate risk score
        risk_score = min(1.0, len(suspicious_indicators) * 0.2)
        
        analysis_result = {
            'cluster_id': native_cluster_id,
            'network_topology': topology,
            'suspicious_indicators': suspicious_indicators,
            'risk_score': risk_score,
            'network_size': G.number_of_nodes(),
            'total_volume_eth': topology.get('flow_patterns', {}).get('total_volume_eth', 0)
        }
        
        # Log suspicious networks
        if suspicious_indicators:
            self._log_suspicious_network(native_cluster_id, analysis_result)

        # +++ NEW: Store risk score in risk_components table +++
        if risk_score > 0.1:
            # Get all addresses in the cluster to assign them the risk
            cluster_addresses_df = self.database.fetch_df("SELECT address FROM addresses WHERE cluster_id = ?", (native_cluster_id,))
            if not cluster_addresses_df.empty:
                for addr in cluster_addresses_df['address']:
                    self.database.store_component_risk(
                        address=addr,
                        component_type='network_topology',
                        risk_score=risk_score,
                        confidence=0.7,
                        evidence={'reason': f'Part of a suspicious network topology in cluster {native_cluster_id}', 'indicators': suspicious_indicators},
                        source_analysis='network_analyzer'
                    )
                logger.info(f"Stored network topology risk for {len(cluster_addresses_df)} addresses in cluster {native_cluster_id}.")

        return analysis_result
    
    def _log_suspicious_network(self, cluster_id: int, analysis_result: Dict[str, Any]):
        """Log suspicious network patterns"""
        # Get representative addresses from the cluster
        cluster_addresses = self.database.fetch_df("""
            SELECT address FROM addresses WHERE cluster_id = ? LIMIT 5
        """, (cluster_id,))
        
        for _, addr_row in cluster_addresses.iterrows():
            address = addr_row['address']
            
            analysis_data = {
                'risk_score': analysis_result['risk_score'],
                'suspicious_patterns': analysis_result['suspicious_indicators'],
                'network_size': analysis_result['network_size'],
                'total_volume': analysis_result['total_volume_eth'],
                'analysis_type': 'network_topology'
            }
            
            log_suspicious_address(address, cluster_id, analysis_data, 'network_topology_analysis')
    
    def analyze_all_networks(self) -> Dict[str, Any]:
        """Analyze network topology for all clusters using a scalable, bulk-processing approach."""
        logger.info("Starting comprehensive network topology analysis...")
        
        # --- OPTIMIZATION: Fetch all clustered transactions at once ---
        logger.info("Fetching all transactions belonging to any cluster for network analysis...")
        all_clustered_txs_df = self.database.fetch_df("""
            SELECT t.from_addr, t.to_addr, t.value, t.timestamp, t.hash, a.cluster_id
            FROM transactions t
            JOIN addresses a ON (t.from_addr = a.address OR t.to_addr = a.address)
            WHERE cluster_id IS NOT NULL
        """)

        if all_clustered_txs_df.empty:
            logger.warning("No clustered transactions found for network analysis.")
            return {'cluster_analyses': [], 'summary': {}}

        # Group transactions by cluster_id in pandas
        txs_by_cluster = all_clustered_txs_df.groupby('cluster_id')
        logger.info(f"Found {len(txs_by_cluster)} clusters to analyze for network topology.")
        # --- END OPTIMIZATION ---

        cluster_analyses = []
        suspicious_networks_count = 0
        
        for cluster_id, cluster_txs_df in tqdm(txs_by_cluster, desc="Analyzing Network Topologies"):
            analysis = self.analyze_cluster_network(cluster_id, cluster_txs_df)
            cluster_analyses.append(analysis)
            
            # Count suspicious networks
            if analysis.get('suspicious_indicators'):
                suspicious_networks_count += 1
        
        summary = {
            'total_clusters_analyzed': len(cluster_analyses),
            'suspicious_networks_found': suspicious_networks_count,
            'analysis_method': 'network_topology'
        }
        
        logger.info(f"Network analysis complete: {summary}")
        
        return {
            'cluster_analyses': cluster_analyses,
            'summary': summary
        }

# Integration function for main pipeline
def integrate_with_pipeline(database: DatabaseEngine, output_dir: str = None) -> Dict[str, Any]:
    """
    Integration function to be called from the main analysis pipeline.
    """
    analyzer = NetworkAnalyzer(database=database)
    results = analyzer.analyze_all_networks()
    
    return results
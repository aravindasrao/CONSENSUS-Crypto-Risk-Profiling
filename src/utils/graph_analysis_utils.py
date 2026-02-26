# src/utils/graph_analysis_utils.py
"""
Utility class for analyzing the topology of a NetworkX graph.
This consolidates duplicated code from FlowAnalyzer and NetworkAnalyzer.
"""
import logging
import networkx as nx
import numpy as np
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)

class GraphTopologyAnalyzer:
    """A utility class to perform various graph topology analyses."""

    def analyze_graph_topology(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Analyze the topology of a transaction network.
        """
        if len(G) == 0:
            return {'analysis': 'empty_network'}
        
        topology = {
            'basic_stats': self._calculate_basic_stats(G),
            'centrality_metrics': self._calculate_centrality_metrics(G),
            'connectivity_analysis': self._analyze_connectivity(G),
            'community_structure': self._analyze_communities(G),
            'flow_patterns': self._analyze_flow_patterns(G),
            'structural_patterns': self._identify_structural_patterns(G)
        }
        
        return topology

    def _calculate_basic_stats(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate basic network statistics"""
        return {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'num_weakly_connected_components': nx.number_weakly_connected_components(G),
            'num_strongly_connected_components': nx.number_strongly_connected_components(G),
            'average_clustering': nx.average_clustering(G.to_undirected()) if G.number_of_nodes() > 0 else 0
        }

    def _calculate_centrality_metrics(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Calculate centrality metrics for key nodes"""
        if G.number_of_nodes() == 0:
            return {}
        
        try:
            in_degree_centrality = nx.in_degree_centrality(G)
            out_degree_centrality = nx.out_degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
            
            top_in_degree = sorted(in_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_out_degree = sorted(out_degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'top_receivers': [{'address': addr, 'centrality': float(cent)} for addr, cent in top_in_degree],
                'top_senders': [{'address': addr, 'centrality': float(cent)} for addr, cent in top_out_degree],
                'top_bridges': [{'address': addr, 'centrality': float(cent)} for addr, cent in top_betweenness],
                'centrality_distribution': {
                    'in_degree_gini': self._calculate_gini_coefficient(list(in_degree_centrality.values())),
                    'out_degree_gini': self._calculate_gini_coefficient(list(out_degree_centrality.values()))
                }
            }
        except Exception as e:
            logger.warning(f"Error calculating centrality metrics: {e}")
            return {'error': str(e)}

    def _analyze_connectivity(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze network connectivity patterns"""
        connectivity = {}
        
        wcc = list(nx.weakly_connected_components(G))
        connectivity['num_components'] = len(wcc)
        connectivity['largest_component_size'] = max(len(comp) for comp in wcc) if wcc else 0
        connectivity['component_sizes'] = [len(comp) for comp in wcc]
        
        scc = list(nx.strongly_connected_components(G))
        connectivity['num_strong_components'] = len(scc)
        connectivity['largest_strong_component_size'] = max(len(comp) for comp in scc) if scc else 0
        
        if wcc:
            largest_component = G.subgraph(max(wcc, key=len))
            try:
                if nx.is_connected(largest_component.to_undirected()):
                    connectivity['diameter'] = nx.diameter(largest_component.to_undirected())
                    connectivity['average_shortest_path'] = nx.average_shortest_path_length(largest_component.to_undirected())
                else:
                    connectivity['diameter'] = None
                    connectivity['average_shortest_path'] = None
            except:
                connectivity['diameter'] = None
                connectivity['average_shortest_path'] = None
        
        return connectivity

    def _analyze_communities(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze community structure in the network"""
        if G.number_of_nodes() < 3:
            return {'communities': 0}
        
        try:
            G_undirected = G.to_undirected()
            communities = list(nx.connected_components(G_undirected))
            
            community_analysis = {
                'num_communities': len(communities),
                'community_sizes': [len(comm) for comm in communities],
                'modularity_possible': len(communities) > 1,
                'largest_community_size': max(len(comm) for comm in communities) if communities else 0
            }
            
            return community_analysis
            
        except Exception as e:
            logger.warning(f"Error in community analysis: {e}")
            return {'error': str(e)}

    def _analyze_flow_patterns(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Analyze money flow patterns in the network"""
        flow_patterns = {}
        
        total_volume = sum(data['weight'] for _, _, data in G.edges(data=True))
        flow_patterns['total_volume_eth'] = float(total_volume)
        
        edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if edge_weights:
            flow_patterns['flow_gini'] = self._calculate_gini_coefficient(edge_weights)
            flow_patterns['max_flow_eth'] = float(max(edge_weights))
            flow_patterns['avg_flow_eth'] = float(np.mean(edge_weights))
            flow_patterns['median_flow_eth'] = float(np.median(edge_weights))
        
        node_in_flow = defaultdict(float)
        node_out_flow = defaultdict(float)
        
        for from_addr, to_addr, data in G.edges(data=True):
            weight = data['weight']
            node_out_flow[from_addr] += weight
            node_in_flow[to_addr] += weight
        
        top_flow_in = sorted(node_in_flow.items(), key=lambda x: x[1], reverse=True)[:5]
        top_flow_out = sorted(node_out_flow.items(), key=lambda x: x[1], reverse=True)[:5]
        
        flow_patterns['top_flow_receivers'] = [{'address': addr, 'volume_eth': float(vol)} for addr, vol in top_flow_in]
        flow_patterns['top_flow_senders'] = [{'address': addr, 'volume_eth': float(vol)} for addr, vol in top_flow_out]
        
        return flow_patterns

    def _identify_structural_patterns(self, G: nx.DiGraph) -> Dict[str, Any]:
        """Identify specific structural patterns that might indicate suspicious activity"""
        patterns = {}
        
        star_patterns = []
        for node in G.nodes():
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            
            if in_degree > 10 and out_degree < 3:
                star_patterns.append({
                    'type': 'collection_star',
                    'center': node,
                    'in_degree': int(in_degree),
                    'out_degree': int(out_degree)
                })
            
            elif out_degree > 10 and in_degree < 3:
                star_patterns.append({
                    'type': 'distribution_star',
                    'center': node,
                    'in_degree': int(in_degree),
                    'out_degree': int(out_degree)
                })
        
        patterns['star_patterns'] = star_patterns
        
        try:
            longest_paths = []
            sources = [n for n in G.nodes() if G.in_degree(n) == 0][:10]
            sinks = [n for n in G.nodes() if G.out_degree(n) == 0][:10]
            
            for source in sources:
                for target in sinks:
                    if source != target:
                        try:
                            path = nx.shortest_path(G, source, target)
                            if len(path) > 5:
                                longest_paths.append({
                                    'path': path[:10],
                                    'length': len(path),
                                    'source': source,
                                    'target': target
                                })
                        except nx.NetworkXNoPath:
                            continue
                        if len(longest_paths) >= 5: break
                    if len(longest_paths) >= 5: break
            
            patterns['long_chains'] = sorted(longest_paths, key=lambda x: x['length'], reverse=True)[:5]
            
        except Exception as e:
            logger.debug(f"Error finding chain patterns: {e}")
            patterns['long_chains'] = []
        
        try:
            cycles = list(nx.simple_cycles(G))
            cycle_info = []
            for cycle in cycles[:10]:
                if len(cycle) > 2:
                    cycle_volume = 0
                    for i in range(len(cycle)):
                        from_addr = cycle[i]
                        to_addr = cycle[(i + 1) % len(cycle)]
                        if G.has_edge(from_addr, to_addr):
                            cycle_volume += G[from_addr][to_addr].get('weight', 0)
                    
                    cycle_info.append({
                        'cycle': cycle[:5],
                        'length': len(cycle),
                        'volume_eth': float(cycle_volume)
                    })
            
            patterns['cycles'] = sorted(cycle_info, key=lambda x: x['volume_eth'], reverse=True)
            
        except Exception as e:
            logger.debug(f"Error detecting cycles: {e}")
            patterns['cycles'] = []
        
        return patterns

    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if not values or len(values) == 0:
            return 0.0
        
        values = [v for v in values if v >= 0]
        if not values:
            return 0.0
        
        values.sort()
        n = len(values)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * values)) / (n * np.sum(values)) - (n + 1) / n)


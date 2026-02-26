# src/analysis/cluster_consensus_engine.py
"""
Combines results from multiple clustering algorithms into a single, high-confidence result.
This is the core of the Consensus Integration
"""
import pandas as pd
import networkx as nx
import logging

# NEW: Import the community detection library from networkx
from networkx.algorithms import community as nx_community

from collections import defaultdict
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ClusterConsensusEngine:
    def __init__(self):
        logger.info("Evidence-Based Cluster Consensus Engine initialized.")

    def create_consensus_clusters(self, all_evidence: List[Dict]) -> List[Dict]:
        """
        Creates a final set of clusters from a list of evidence from multiple algorithms.
        This is a more advanced, evidence-based approach.

        Args:
            all_evidence: A single list of dictionaries, where each dict is a piece of evidence, e.g.,
                          {'address': str, 'cluster_id': str, 'confidence': float, 'evidence_type': str}
        
        Returns:
            A list of final cluster assignments with confidence and evidence.
        """
        logger.info(f"Creating consensus from {len(all_evidence)} pieces of evidence...")

        if not all_evidence:
            return []

        # Use a graph-based approach for consensus
        # Nodes are addresses, edges represent a shared cluster assignment from an algorithm
        # Edge weight is the confidence of that link
        G = nx.Graph()

        # Group evidence by the original cluster_id from each algorithm
        evidence_groups = defaultdict(list)
        for evidence in all_evidence:
            evidence_groups[evidence['cluster_id']].append(evidence)

        # For each group of addresses identified by an algorithm, create weighted edges
        for cluster_id, evidence_list in evidence_groups.items():
            if len(evidence_list) > 1:
                # Get the confidence and evidence type for this link
                confidence = evidence_list[0].get('confidence', 0.5)
                evidence_type = evidence_list[0].get('evidence_type', 'unknown')
                
                # Get all addresses in this group
                members = [item['address'] for item in evidence_list]
                
                # Add edges between all pairs of addresses in this group
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        u, v = members[i], members[j]
                        
                        # If an edge already exists, update if new evidence is stronger
                        if G.has_edge(u, v):
                            if confidence > G[u][v]['weight']:
                                G[u][v]['weight'] = confidence
                                G[u][v]['evidence'] = evidence_type
                        else:
                            G.add_edge(u, v, weight=confidence, evidence=evidence_type)

        # --- MODIFICATION: Use Louvain community detection instead of connected components ---
        # This algorithm uses the edge weights to find dense communities, which aligns
        # better with the idea of a "weighted evidence graph".
        # The resolution parameter can be tuned to find larger or smaller communities.
        consensus_components = nx_community.louvain_communities(G, weight='weight', resolution=1.0)
        
        # Assign final cluster IDs and calculate confidence for each address
        final_clusters = []
        consensus_cluster_id_counter = 1
        
        # Process multi-address clusters
        for component in consensus_components:
            if len(component) > 1:
                for address in component:
                    # Find the strongest link connecting this address to the component
                    max_confidence = 0.0
                    best_evidence = 'inferred_link'
                    for neighbor in G.neighbors(address):
                        if neighbor in component:
                            edge_data = G.get_edge_data(address, neighbor)
                            if edge_data['weight'] > max_confidence:
                                max_confidence = edge_data['weight']
                                best_evidence = edge_data['evidence']
                    
                    final_clusters.append({
                        'address': address,
                        'consensus_cluster_id': consensus_cluster_id_counter,
                        'final_confidence': max_confidence,
                        'final_evidence': best_evidence
                    })
                consensus_cluster_id_counter += 1
        
        # Handle singletons (addresses with no links)
        all_addresses_in_evidence = {ev['address'] for ev in all_evidence}
        clustered_addresses = {addr['address'] for addr in final_clusters}
        singleton_addresses = all_addresses_in_evidence - clustered_addresses
        
        for address in singleton_addresses:
            final_clusters.append({
                'address': address,
                'consensus_cluster_id': consensus_cluster_id_counter,
                'final_confidence': 0.1, # Low confidence for singletons
                'final_evidence': 'singleton'
            })
            consensus_cluster_id_counter += 1

        logger.info(f"Consensus complete. Identified {consensus_cluster_id_counter - 1} final clusters.")
        return final_clusters
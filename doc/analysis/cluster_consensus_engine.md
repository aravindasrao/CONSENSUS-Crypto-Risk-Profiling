# Cluster Consensus Engine

This document outlines the logic and methodology used by the `ClusterConsensusEngine` module. This module is the final arbiter for address clustering, responsible for synthesizing evidence from multiple, independent clustering algorithms into a single, high-confidence set of consensus clusters.

## Overview

Different clustering algorithms have different strengths and weaknesses. The `IncrementalDFSClusterer` is high-precision but may miss links, while the `BehavioralPatternAnalyzer` and GNNs are probabilistic and can find non-obvious connections. The `ClusterConsensusEngine` resolves potential conflicts and combines these views using an evidence-based approach.

The core idea is to build a "meta-graph" where a weighted edge between two addresses represents evidence that they belong together. The stronger and more varied the evidence, the higher the final confidence in the link.

## Processing Logic

1.  **Evidence Ingestion**: The engine fetches all cluster assignments from the `cluster_assignments` table. Each row represents a piece of evidence from a specific analyzer (e.g., `DFS_`, `BEH_`, `GNN_`).

2.  **Meta-Graph Construction**: It builds an undirected graph where:
    *   **Nodes**: are the individual blockchain addresses.
    *   **Edges**: An edge is created between any two addresses that were placed in the same cluster by *any* of the analysis modules.
    *   **Edge Weights**: The weight of an edge is the confidence score provided by the source analyzer. If multiple analyzers link the same two addresses, the edge is assigned the **highest confidence score** from all linking evidence ("strongest link wins").

3.  **Consensus Clustering (Community Detection)**: The engine uses a community detection algorithm (specifically, Louvain community detection) on the weighted meta-graph. This algorithm uses the edge weights (confidence scores) to find densely connected groups of nodes. Each distinct community becomes a final consensus cluster. This is more robust than simple connected components as it is less likely to merge large groups based on a single weak link.

4.  **Final Confidence Calculation**: For each consensus cluster, a final confidence score is calculated. This is typically the average or maximum confidence of the evidence that formed the cluster, providing a measure of how strong the link is.

5.  **Database Storage**: The final, authoritative cluster assignments are updated in the `addresses` table, including the `cluster_id`, `cluster_confidence`, and a summary of the `cluster_method` (evidence).

## Evidence Sources and Default Weights

The strength of the consensus depends on the confidence assigned to each piece of evidence.

| Evidence Type (Prefix) | Source Analyzer | Default Confidence | Rationale |
|:---:|---|:---:|---|
| `DFS_` | `IncrementalDFSClusterer` | **0.95** | Based on the shared-input heuristic, which is the gold standard for on-chain clustering. It is extremely high-confidence but not 1.0 to allow for rare edge cases. |
| `GNN_` | `GraphTransformerNetwork` | **0.70** | Based on unsupervised learning of network structure and features. It's a strong indicator of behavioral similarity but is less certain than a direct cryptographic link. |
| `BEH_` | `BehavioralPatternAnalyzer` | **0.60** | Based on similarity of mixer usage patterns. It's a good probabilistic link but is more heuristic in nature than a full graph analysis. |
| `SAGE_` | `GraphSAGEAnalyzer` | **0.55** | Based on a supervised model's classification. It provides a useful signal but is dependent on the quality of the training labels. |

## Role in the Pipeline

The `ClusterConsensusEngine` is the crucial final step in the clustering process. It runs at the beginning of Phase 3 to produce the final, authoritative clusters. These consensus clusters are then used by the final `UnifiedRiskScorer` for risk propagation and by the `ForensicExporter` for reporting.

**Example Scenario**:
- The `DFSClusterer` links Address A and Address B (Confidence 0.95).
- The `BehavioralPatternAnalyzer` links Address B and Address C (Confidence 0.60).

The consensus engine will place all three addresses (A, B, and C) into the same final cluster. The link between A and B is very strong, while the link to C is weaker but still present. The final evidence for the cluster would reflect this composite nature, providing a complete and nuanced picture of the entity.
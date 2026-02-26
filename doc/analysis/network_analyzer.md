# Network Analyzer

This document outlines the logic and rules used by the `NetworkAnalyzer` module. This module is designed to analyze the internal network topology of a given cluster to identify structural patterns indicative of coordinated or suspicious activity.

## Overview

While other modules analyze an address's interactions with the outside world, the `NetworkAnalyzer` focuses on the relationships *within* a cluster. By constructing a directed graph of all transactions between addresses in the same cluster, it can identify structural anomalies that would otherwise be missed. This helps answer the question: "How does this entity manage its own funds internally?"

## Processing Logic

1.  **Cluster-Level Analysis**: The analyzer operates on the final consensus clusters. For each cluster, it fetches all associated transactions.
1.  **Cluster-Level Analysis**: The analyzer operates on the clusters available after Phase 1 of the pipeline (typically the high-confidence clusters from the `IncrementalDFSClusterer`). For each cluster, it fetches all associated transactions.

2.  **Graph Construction**: It builds a directed `networkx` graph where nodes are the addresses within the cluster and edges represent transactions between them. Edges are weighted by the total ETH volume and transaction count.

3.  **Topological Analysis**: It uses the `GraphTopologyAnalyzer` utility to calculate a comprehensive set of network metrics. This includes:
    *   **Basic Stats**: Node/edge counts, density.
    *   **Centrality**: In-degree, out-degree, and betweenness centrality to find key players.
    *   **Connectivity**: Information on connected components.
    *   **Flow Patterns**: Gini coefficient to measure flow concentration.

4.  **Suspicious Pattern Identification**: The module applies a set of rules to the calculated metrics to flag potentially suspicious network structures.

5.  **Risk Scoring & Storage**: A simple risk score is calculated based on the number of suspicious indicators found. If the score is significant, a `network_topology` risk component is added to the `risk_components` table for *all addresses* within that suspicious cluster.

## Key Suspicious Patterns and Rationale

| Pattern | Name | Description & Rule | Rationale |
|:---:|---|---|---|
| 1 | **High Density** | `Density > 0.5` and `Nodes > 10` | An unusually high number of connections between nodes in a cluster suggests a high degree of internal coordination, which is not typical for a simple wallet. |
| 2 | **Star Patterns** | The presence of nodes with very high in-degree or out-degree relative to others. | A "distribution star" (high out-degree) can indicate a central address distributing funds to sub-wallets. A "collection star" (high in-degree) can indicate funds being consolidated before an external transaction. |
| 3 | **Cycle Patterns** | The presence of multiple simple cycles (`A -> B -> C -> A`). | Funds moving in circles serve no legitimate economic purpose and are a strong indicator of attempts to obfuscate the flow of funds or wash trading. |
| 4 | **Concentrated Flows** | `Flow Gini Coefficient > 0.8` | A high Gini coefficient means that a small number of transaction links account for a vast majority of the total volume moved within the cluster. This points to a highly centralized fund management strategy. |
| 5 | **Large Internal Volume**| `Total Internal Volume > 1000 ETH` | A very large volume of funds being moved *internally* within a cluster, rather than to external entities, is anomalous and warrants investigation. |

## Role in the Pipeline

The `NetworkAnalyzer` provides a unique, structural view of risk. It complements other modules by adding evidence based on the *internal organization* of an entity's addresses. For example, a cluster might have a low risk score based on its external interactions, but the `NetworkAnalyzer` could flag it as high-risk due to a complex, cyclical internal flow pattern. This evidence is a crucial input for the `UnifiedRiskScorer`.
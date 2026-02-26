# Graph Transformer Network Analyzer

This document outlines the logic and methodology used by the `GraphTransformerNetwork` module. This module uses an advanced Graph Neural Network (GNN) architecture, specifically a Graph Transformer, to perform behavioral clustering and anomaly detection.

## Overview

Unlike simpler GNNs, a Graph Transformer uses self-attention mechanisms, allowing it to weigh the importance of different nodes in a neighborhood when creating a node's representation (embedding). This makes it exceptionally powerful at capturing complex, long-range, and subtle relationships that might be missed by other methods.

The module has two primary goals:
1.  **Behavioral Clustering**: To group addresses that exhibit similar relational and feature-based patterns.
2.  **Suspicion Scoring**: To identify outlier nodes whose behavior is highly unusual or uncertain, flagging them as anomalous.

## Processing Logic

The analysis follows a standard GNN pipeline:

1.  **Graph Construction**: The module first constructs a single, large graph representing the entire dataset.
    *   **Nodes**: Every unique address becomes a node.
    *   **Node Features**: The 111-feature vector from the `FoundationLayer` is used as the initial feature set for each node.
    *   **Edges**: A directed edge is created for each transaction from the `from_addr` to the `to_addr`.

2.  **Model Forward Pass**: The constructed graph is passed through the `GraphTransformerNetwork` model.
    *   The model is not "trained" in a traditional supervised sense. Instead, it is used in an unsupervised manner as a powerful, non-linear feature extractor.
    *   The `TransformerConv` layers use attention to learn rich embeddings for each node that incorporate information from its neighbors, weighted by importance.

3.  **Clustering**: The final output embeddings from the model are used to assign cluster labels.
    *   The `argmax` function is applied to the final embedding of each node. This assigns the node to the cluster corresponding to the highest-scoring dimension in its embedding.
    *   This effectively groups nodes that the model has mapped to similar regions in the high-dimensional embedding space.
    *   These cluster assignments (prefixed with `GNN_`) are stored for the `ClusterConsensusEngine`.

4.  **Suspicion Scoring**: A suspicion score is calculated for each node based on the model's output.
    *   **Method**: The Shannon entropy is calculated across the final probability distribution (softmax of the output embeddings) for each node.
    *   **Rationale**: A high entropy score indicates that the model is "uncertain" about how to classify the node's behavior. It doesn't fit cleanly into any of the primary patterns the model has learned, making it a likely outlier or anomaly.
    *   These scores are stored as `gnn_risk` components for the `UnifiedRiskScorer`.

## Key Concepts and Rationale

| Concept | Rationale |
|:---:|---|
| **Graph Transformer (`TransformerConv`)** | The use of self-attention allows the model to dynamically learn the importance of different neighbors for each node. This is superior to standard GCNs which treat all neighbors equally, enabling the model to capture more nuanced behavioral patterns. |
| **Unsupervised Clustering** | By using the model as a powerful feature extractor without explicit training, we can discover emergent behavioral patterns in the data without being biased by pre-existing labels. The model learns the inherent structure of the transaction graph. |
| **Entropy-Based Suspicion** | This is a powerful heuristic for anomaly detection in unsupervised learning. If a model that is good at finding patterns is "confused" by a particular data point, it's a strong signal that the data point is unusual and warrants investigation. |

## Role in the Pipeline

The `GraphTransformerNetwork` provides two critical pieces of evidence to the downstream modules:

1.  **To the `ClusterConsensusEngine`**: It provides a set of probabilistic cluster assignments (`GNN_` clusters). When these assignments agree with the high-confidence `DFS_` clusters or the `BEH_` clusters, it significantly strengthens the case for a link.

2.  **To the `UnifiedRiskScorer`**: It provides a `gnn_risk` component for each address. This score is particularly valuable for flagging sophisticated or novel illicit behaviors that may not trigger rules in other analyzers but are clearly anomalous from a network structure perspective.
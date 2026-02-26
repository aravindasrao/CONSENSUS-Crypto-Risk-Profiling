# Temporal GNN Analyzer (T-GCN)

This document outlines the logic and methodology used by the `TemporalGNNAnalyzer` module. This module employs a Temporal Graph Convolutional Network (T-GCN) to model the dynamic evolution of the transaction graph over time.

## Overview

Static graph analysis can miss crucial temporal patterns. The `TemporalGNNAnalyzer` addresses this by treating the transaction graph as a dynamic entity. It combines a Graph Convolutional Network (GCN) to capture spatial dependencies within a time slice, and a Gated Recurrent Unit (GRU) to model the temporal dependencies *between* time slices.

The primary goal is to generate a risk score for each address based on how its features and network position evolve, allowing the detection of sudden behavioral changes or coordinated activities.

## Processing Logic

The analysis follows a temporal GNN pipeline:

1.  **Graph Snapshot Creation**: The entire transaction history is divided into a series of discrete, sequential "snapshots". Each snapshot is a graph representing all transactions within a specific time window (e.g., 24 hours).

2.  **Sequential Model Processing**: The sequence of graph snapshots is processed by the T-GCN model:
    *   For each snapshot, the GCN layer updates the node embeddings based on the current graph structure.
    *   The GRU layer then takes this updated embedding and the hidden state from the *previous* snapshot to produce a new hidden state. This step captures the temporal evolution.
    *   This process repeats for all snapshots, with the hidden state being passed from one snapshot to the next.

3.  **Risk Scoring from Final Embeddings**:
    *   The final hidden state (embedding) for each node represents its complete spatio-temporal signature over the entire analysis period.
    *   A risk score is calculated from this final embedding. A simple and effective method is to use the magnitude (L2 norm) of the embedding vector. A large magnitude suggests the node's state has changed significantly or accumulated strong signals over time, indicating anomalous behavior.

4.  **Database Storage**: The calculated risk scores are stored as `temporal_gnn_risk` components in the `risk_components` table, making them available to the `UnifiedRiskScorer`.

## Key Concepts and Rationale

| Concept | Rationale |
|:---:|---|
| **Graph Snapshots** | Discretizing the continuous flow of transactions into a sequence of static graphs allows us to apply sequence modeling techniques to the evolving network structure. |
| **T-GCN (GCN + GRU)** | This hybrid architecture is ideal for spatio-temporal data. The GCN captures the "where" (spatial relationships in a snapshot), and the GRU captures the "when" (how relationships and features change over time). |
| **Unsupervised Risk Scoring** | The model is not trained on explicit risk labels. Instead, it learns the normal evolution of the graph. The final embedding's magnitude serves as an anomaly score: nodes that undergo significant or unusual changes will have embeddings that are far from the origin, resulting in a higher risk score. |

## Role in the Pipeline

The `TemporalGNNAnalyzer` provides a unique, dynamic risk signal that is highly effective at:

- **Detecting Sudden Changes**: Identifying addresses that suddenly become highly active or change their interaction patterns.
- **Identifying Coordinated Events**: Flagging groups of addresses that become active simultaneously in a short time frame, even if they aren't directly connected.
- **Modeling Behavioral Evolution**: Capturing how an address's role in the network changes over time, which can be indicative of a shift from legitimate to illicit activity.

The `temporal_gnn_risk` component is a valuable input for the `UnifiedRiskScorer`, adding a time-aware dimension to the final risk assessment.
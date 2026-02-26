# GraphSAGE Analyzer

This document outlines the logic and methodology used by the `GraphSAGE` (Graph Sample and Aggregate) analyzer module. This module provides a scalable, inductive approach to node representation learning and risk classification.

## Overview

GraphSAGE is a Graph Neural Network (GNN) architecture designed to work efficiently on large graphs. Unlike models that need the entire graph for training (transductive), GraphSAGE learns a function to generate embeddings for nodes by sampling and aggregating features from their local neighborhood. This makes it "inductive," meaning it can generate embeddings for nodes that were not seen during training.

The module has two primary goals:
1.  **Risk Classification**: To train a supervised model that predicts whether an address is high-risk based on its features and its local network structure.
2.  **Cluster Assignment**: To provide another set of cluster labels (`SAGE_0` for low-risk, `SAGE_1` for high-risk) to the `ClusterConsensusEngine`.

## Processing Logic

The analysis follows a standard supervised GNN pipeline:

1.  **Data Preparation**: The module constructs a graph `Data` object suitable for PyTorch Geometric.
    *   **Nodes**: All addresses with features are included as nodes.
    *   **Node Features**: A wide range of numeric features from the `FoundationLayer` are used as the initial feature set for each node. Features are standardized using `StandardScaler`.
    *   **Edges**: Transactions form the directed edges of the graph.
    *   **Labels (y)**: A binary label is created for each node. An address is labeled as high-risk (`1`) if its `composite_risk_score` from the foundation layer is above a threshold of `0.5`, and low-risk (`0`) otherwise.
    *   **Masks**: The data is split into training, validation, and test sets using masks.

2.  **Model Training**:
    *   A `GraphSAGE` model is initialized.
    *   The model is trained for a set number of epochs on the `train_mask` data. The goal is to learn to predict the risk label (`y`) from the node features (`x`) and graph structure (`edge_index`).
    *   The loss function used is Negative Log-Likelihood Loss (`F.nll_loss`), which is standard for classification tasks.

3.  **Evaluation & Prediction**:
    *   The trained model is evaluated on the `test_mask` to calculate its accuracy.
    *   The model then predicts a risk class (`0` or `1`) for *all* nodes in the graph.

4.  **Model Persistence**: After training, the module saves several critical artifacts to disk for later use by the `OnlineInferenceEngine`:
    *   The trained GraphSAGE model state (`graphsage_model.pth`).
    *   The feature scaler (`graphsage_scaler.pkl`).
    *   The address-to-index mapping (`graphsage_addr_map.json`).
    *   The final embeddings for all known nodes (`graphsage_embeddings.pt`).

4.  **Database Storage**:
    *   **Risk Components**: For each address predicted as high-risk (`1`), a `graphsage_risk` component is stored in the `risk_components` table. The risk score is the model's confidence (probability) in its prediction.
    *   **Cluster Assignments**: For *all* addresses, a cluster assignment (`SAGE_0` or `SAGE_1`) is stored in the `cluster_assignments` table. This provides another layer of evidence for the `ClusterConsensusEngine`.

## Key Concepts and Rationale

| Concept | Rationale |
|:---:|---|
| **Inductive Learning** | GraphSAGE's ability to generalize to unseen nodes is crucial for a real-time system where new addresses appear constantly. It learns a *function* for generating embeddings, not just the embeddings themselves. |
| **Scalability** | By sampling a fixed-size neighborhood for each node, GraphSAGE avoids the "neighbor explosion" problem of other GNNs, making it suitable for very large, real-world transaction graphs. |
| **Supervised Classification** | Unlike the unsupervised Graph Transformer, GraphSAGE is trained on a specific task (predicting risk). This provides a targeted risk signal that complements the anomaly-focused scores from other modules. |

## Role in the Pipeline

The `GraphSAGE` analyzer provides two valuable outputs:

1.  **To the `UnifiedRiskScorer`**: It contributes a `graphsage_risk` component. This score reflects how much an address's local network structure and features resemble those of historically high-risk addresses.
2.  **To the `ClusterConsensusEngine`**: It provides a simple but powerful binary clustering (`SAGE_0` vs. `SAGE_1`). When this agrees with other clustering methods (e.g., a `DFS_` cluster is composed entirely of `SAGE_1` nodes), it dramatically increases the confidence that the entire cluster is high-risk.
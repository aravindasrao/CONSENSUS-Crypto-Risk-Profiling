# HGN Analyzer (Heterogeneous Graph Network)

This document outlines the logic and methodology used by the `HGNAnalyzer` module. This module employs a Heterogeneous Graph Transformer (HGT) to model the diverse nature of blockchain entities and their interactions.

## Overview

Unlike standard Graph Neural Networks (GNNs) that treat all nodes as the same, a Heterogeneous Graph Network acknowledges that the blockchain contains different types of nodes (e.g., Externally Owned Accounts - EOAs, and Smart Contracts) that interact in different ways.

By modeling these distinct node and edge types, the HGN can learn much richer, more context-aware representations of each address, leading to more accurate risk assessments.

## Processing Logic

The analysis follows a supervised GNN pipeline designed for heterogeneous graphs:

1.  **Heterogeneous Graph Construction**: The module builds a `HeteroData` object from the database.
    *   **Node Types**: Nodes are explicitly separated into two types: `eoa` and `contract`.
    *   **Node Features**: The 111-feature vector from the `FoundationLayer` is used as the initial feature set for all nodes.
    *   **Edge Types**: Edges are also typed based on the interaction, creating a rich relational structure (e.g., `('eoa', 'sends_to', 'eoa')`, `('eoa', 'calls', 'contract')`).

2.  **Supervised Model Training**:
    *   An `HGT` (Heterogeneous Graph Transformer) model is initialized.
    *   The model is trained in a supervised manner. For this analysis, it's a regression task: the model learns to predict a target variable for each EOA, such as its `total_volume_out_eth`.
    *   The model is trained in a supervised manner. For this analysis, it's a regression task: the model learns to predict a target variable for each EOA, such as its `outgoing_volume_eth`.
    *   This training process teaches the model to identify the complex structural and feature patterns associated with addresses that act as significant sources of funds.

3.  **Risk Scoring**:
    *   After training, the model predicts the target variable (e.g., outgoing volume) for all EOAs.
    *   This prediction is then normalized and used as the `hgn_risk` score. A high score indicates that the address's features and its position within the heterogeneous network strongly suggest it behaves as a fund disperser.

4.  **Database Storage**: The final risk scores are stored as `hgn_risk` components in the `risk_components` table, making them available to the `UnifiedRiskScorer`.

## Key Concepts and Rationale

| Concept | Rationale |
|:---:|---|
| **Heterogeneous Graph (`HeteroData`)** | The blockchain is inherently heterogeneous. Modeling EOAs and contracts as different types allows the GNN to learn type-specific transformations and understand that an EOA sending to a contract is fundamentally different from an EOA sending to another EOA. |
| **HGTConv (Heterogeneous Graph Transformer)** | The self-attention mechanism in HGT is type-aware. It learns to weigh the importance of different neighbors *and* different relationship types, enabling it to capture highly nuanced patterns (e.g., learning that an EOA receiving funds from 10 other EOAs is more significant than one being called by 10 contracts). |
| **Supervised Regression Task** | By training the model to predict a specific, known behavior (like high outgoing volume), we create a targeted risk signal. This is more powerful than unsupervised anomaly detection for identifying a specific typology, such as a "disperser" or "source" wallet in a money laundering scheme. |

## Role in the Pipeline

The `HGNAnalyzer` provides a sophisticated, structurally-aware risk signal to the `UnifiedRiskScorer`. Its `hgn_risk` component is particularly effective at:

- **Identifying Role-Specific Behavior**: It can distinguish between an address acting as a simple pass-through wallet versus one acting as a central distribution hub, based on the *types* of entities it interacts with.
- **Capturing Complex Relational Risk**: It goes beyond an address's own features to incorporate the risk and behavior of its multi-type neighborhood, providing a more holistic risk assessment.
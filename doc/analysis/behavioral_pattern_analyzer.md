# Behavioral Pattern Analyzer

This document describes the `BehavioralPatternAnalyzer` module, which performs a secondary, probabilistic clustering pass based on behavioral similarity. Unlike the `IncrementalDFSClusterer` which relies on deterministic links, this module uses machine learning to group addresses that *behave* in similar ways, even if they are not directly connected.

## Overview

The core idea is to quantify an address's interaction patterns with known mixer contracts (like Tornado Cash) and then use a clustering algorithm to find groups of addresses with similar behavioral "fingerprints". This can help identify users employing similar mixing strategies or potentially link different deposit and withdrawal addresses from the same entity.

## Processing Logic

The analysis follows a standard machine learning pipeline:

1.  **Data Extraction**: The module queries the database for all transactions involving known mixer contracts. It identifies the `user_address` for each transaction (either the sender for a deposit or the recipient for a withdrawal).

2.  **Feature Engineering**: For each unique `user_address`, a set of behavioral features is calculated based on their deposit and withdrawal history. These features are designed to capture the *how*, *when*, and *how much* of their mixing activity.

3.  **Scaling**: The engineered features are normalized using `StandardScaler`. This is a crucial step to ensure that features with larger scales (like volume) do not disproportionately influence the clustering algorithm compared to features with smaller scales (like ratios).

4.  **Clustering**: The scaled feature data is fed into a density-based clustering algorithm, preferably **HDBSCAN**.
    - **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**: This advanced algorithm is used by default. It does not require the number of clusters to be pre-specified and can identify clusters of arbitrary shapes. Crucially, it can also identify and label outlier addresses as "noise," which is invaluable for forensic analysis as these outliers often represent unique, high-interest actors.
    - **K-Means (Fallback)**: If HDBSCAN is not available, the system falls back to K-Means. In this case, an optimal number of clusters (`k`) is estimated using the silhouette score.

5.  **Output**: The resulting cluster assignments are stored in the `cluster_assignments` table. Each assignment is given a `BEH_` prefix to distinguish it from other clustering methods and is assigned a default confidence score of `0.60`, reflecting its probabilistic nature. Outliers identified by HDBSCAN are assigned a special cluster ID (e.g., -1) and can be flagged for further investigation.


## Behavioral Features

The following features are engineered to create the behavioral fingerprint for each address:

| Feature Name | Description | Rationale |
|---|---|---|
| `deposit_volume` | Total ETH value of all deposits made by the address. | Measures the total amount of funds the user has put into mixers. |
| `withdrawal_volume` | Total ETH value of all withdrawals received by the address. | Measures the total amount of funds the user has taken out of mixers. |
| `volume_ratio` | `Deposit Volume / Withdrawal Volume` | A ratio close to 1 may indicate a user is simply passing funds through. A high or low ratio could indicate accumulation or partial withdrawal. |
| `avg_deposit_amount` | The average value of each deposit transaction. | Indicates the user's typical deposit size. |
| `avg_withdrawal_amount` | The average value of each withdrawal transaction. | Indicates the user's typical withdrawal size. |
| `deposit_count` | The total number of deposit transactions. | Measures how frequently the user deposits funds. |
| `withdrawal_count` | The total number of withdrawal transactions. | Measures how frequently the user withdraws funds. |
| `avg_deposit_interval_hours` | The average time (in hours) between consecutive deposits. | A low value indicates rapid, successive deposits. A high, regular value could indicate a scheduled process. |
| `avg_withdrawal_interval_hours`| The average time (in hours) between consecutive withdrawals. | Similar to the deposit interval, this helps profile the user's withdrawal cadence. |
| `interleaving_score` | A score from 0 to 1 measuring how frequently the user alternates between deposits and withdrawals. | A high score (near 1) indicates a very mixed sequence (e.g., D-W-D-W), while a low score indicates batched activity (e.g., D-D-D-W-W-W). |


## Summary
1. **Input: Transactions** The process starts by querying the database for all transactions involving known mixer contracts (like Tornado Cash).
2. **Intermediate Step: Feature Engineering** For each unique address involved in those transactions, the module calculates the 10 behavioral features listed in the documentation (e.g., `deposit_volume`, `interleaving_score`, etc.). This creates a behavioral "fingerprint" for each address.
3. **Output: (Address, Cluster ID, Confidence)** The clustering algorithm (HDBSCAN or K-Means) runs on these fingerprints. The final result for each address is a record that includes:
    - `address`: The user's address.
    - `cluster_id`: The behavioral cluster it was assigned to (e.g., `BEH_0`, `BEH_1`, or `BEH_-1` for outliers).
    - `confidence`: A default confidence score (e.g., `0.60`) is assigned to this probabilistic link.

This output is then stored in the `cluster_assignments` table, ready to be used by the `ClusterConsensusEngine` as one of several pieces of evidence.


## Role in the Pipeline

The `BehavioralPatternAnalyzer` provides a complementary view to the high-confidence DFS clustering. Its results are fed into the `ClusterConsensusEngine` as another piece of evidence. When the consensus engine sees that two addresses were clustered together by *both* the DFS clusterer and the behavioral analyzer, it can assign a much higher final confidence score to that link.
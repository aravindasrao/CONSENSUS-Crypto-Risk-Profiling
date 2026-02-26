# Deposit and Withdrawal Pattern Analyzer

This document outlines the logic and rules used by the `DepositWithdrawalPatternAnalyzer` module. This module specializes in analyzing the relationship between an entity's deposits into and withdrawals from mixer contracts to identify and cluster distinct behavioral strategies.

## Overview

The core goal is to move beyond simply noting that an address used a mixer and instead analyze the *strategy* of usage. By engineering a rich set of features that capture the timing, volume, sequence, and consistency of deposit/withdrawal activity, this module can group addresses with similar behaviors and flag anomalous patterns indicative of automated or illicit activity.

## Processing Logic

1.  **Data Extraction**: The module identifies all addresses that have interacted with known mixer contracts (e.g., Tornado Cash) above a minimum transaction threshold.

2.  **Feature Engineering**: For each address, it engineers a comprehensive set of features to create a behavioral "fingerprint" (see table below for details). This captures everything from volume and frequency to timing consistency and sequence complexity.

3.  **Dimensionality Reduction & Scaling**: The features are standardized using `StandardScaler`. If the feature set is large, Principal Component Analysis (`PCA`) is used to reduce dimensionality while retaining most of the variance.

4.  **Behavioral Clustering**: It uses a configurable clustering algorithm (`KMeans`, `DBSCAN`, or `Hierarchical`) on the processed features to group addresses that employ similar deposit/withdrawal strategies. The optimal number of clusters can be determined automatically.

5.  **Cluster Profiling & Anomaly Detection**:
    *   Each resulting cluster is profiled by calculating the average value for each feature, assigning it a `risk_score`, and classifying its dominant `pattern_type` (e.g., `active_mixer`, `deposit_heavy`).
    *   Anomalies are identified in two ways: as "noise points" that do not belong to any cluster (if using DBSCAN) or as statistical outliers within a given cluster (e.g., an address with a feature value >3 standard deviations from the cluster mean).

5.  **Database Storage**:
    *   Detailed features and cluster assignments for each address are stored in the `deposit_withdrawal_patterns` table.
    *   For addresses belonging to high-risk clusters or identified as anomalies, a `DEPOSIT_WITHDRAWAL_ANOMALY` risk component is added to the `risk_components` table, contributing to the final unified risk score.

## Behavioral Features Engineered

| Category | Feature Name | Description | Rationale |
|:---:|---|---|---|
| **Volume** | `total_deposit_volume`, `total_withdrawal_volume` | Total ETH value of deposits and withdrawals. | Measures the economic scale of the activity. |
| | `volume_ratio` | `Withdrawal Volume / Deposit Volume`. | A ratio near 1 suggests pass-through activity. |
| | `avg_deposit_amount`, `avg_withdrawal_amount` | Average value of each deposit/withdrawal. | Indicates typical transaction size. |
| | `deposit_amount_std`, `withdrawal_amount_std` | Standard deviation of deposit/withdrawal values. | Low standard deviation suggests use of fixed amounts, common in automated scripts. |
| **Frequency** | `deposit_count`, `withdrawal_count` | Total number of deposit/withdrawal transactions. | Measures the frequency of mixer usage. |
| | `deposit_withdrawal_ratio` | `Deposit Count / Withdrawal Count`. | A skewed ratio can indicate the address is only one part of a larger operation. |
| **Temporal** | `avg_deposit_interval_hours`, `avg_withdrawal_interval_hours` | Average time between consecutive deposits/withdrawals. | A low value indicates rapid, successive transactions. |
| | `deposit_time_std_hours`, `withdrawal_time_std_hours` | Standard deviation of the hour of day for transactions. | A low value indicates transactions consistently occur at the same time of day. |
| **Sequence** | `min_deposit_withdrawal_gap_hours` | The shortest time gap between any deposit and any withdrawal. | An extremely short gap is a strong indicator of automated pass-through activity. |
| | `interleaving_score` | A score (0-1) measuring how frequently the user alternates between deposits and withdrawals. | A high score indicates a mixed sequence (D-W-D-W); a low score indicates batched activity (D-D-W-W). |
| **Behavioral** | `uses_round_amounts` | Proportion of transactions using standard mixer amounts (0.1, 1, 10, 100 ETH). | High usage indicates a knowledgeable actor, but can also be a privacy best practice. |
| | `time_consistency_score` | A score (0-1) based on the coefficient of variation of time gaps between all transactions. | A score near 1 indicates highly regular, machine-like timing. |
| | `pattern_complexity` | A composite score based on interleaving, amount variance, and timing irregularity. | Measures the overall sophistication and unpredictability of the pattern. |

## Key Detected Pattern Types and Rationale

| Pattern | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Active Mixer** | A cluster profile with a high `interleaving_score` and a `volume_ratio` close to 1. | This is the classic pattern of an address being used as a pass-through entity, actively mixing deposits and withdrawals. |
| 2 | **Systematic User** | A cluster profile with a high `time_consistency_score` and a high `uses_round_amounts` ratio. | This machine-like regularity is not typical of human behavior and points to an automated, strategic process for using the mixer. |
| 3 | **Deposit Heavy** | A cluster profile where the average `deposit_count` is significantly higher than the `withdrawal_count`. | Indicates that these addresses are primarily used as entry points into the mixer, with the corresponding withdrawals likely happening from a different set of addresses. |
| 4 | **Withdrawal Heavy** | A cluster profile where the average `withdrawal_count` is significantly higher than the `deposit_count`. | Indicates that these addresses are primarily used as exit points from the mixer, receiving funds that were deposited by other addresses. |
| 5 | **Irregular User** | A cluster profile with low `time_consistency_score` and high variance in transaction amounts. | This pattern is more consistent with manual, human-driven activity, which is generally considered lower risk than automated patterns. |

## Role in the Pipeline

The `DepositWithdrawalPatternAnalyzer` adds a crucial layer of behavioral intelligence to mixer analysis. It provides the `UnifiedRiskScorer` with evidence of *intent and sophistication*. An entity whose behavior is classified as `systematic_user` or part of a high-risk cluster is a much higher priority than one with a random, `irregular_user` pattern, even if they both deposited the same amount. This helps prioritize high-risk actors who are using mixers as a core part of an illicit financial strategy.


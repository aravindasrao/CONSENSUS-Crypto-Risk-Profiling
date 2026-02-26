# Advanced Behavioral Sequence Miner

This document outlines the logic and rules used by the `AdvancedBehavioralSequenceMiner` module. This module is one of the most sophisticated in the pipeline, designed to discover and analyze sequential patterns of behavior (Tactics, Techniques, and Procedures - TTPs), identify automation, and attribute addresses to single entities based on behavioral similarity.

## Overview

While other modules analyze static features or simple flows, the `AdvancedBehavioralSequenceMiner` focuses on the *order, timing, and strategy* of on-chain actions. It answers questions like: "Does this address behave like a human, or does it follow a repetitive, scripted pattern?" and "Are these two unlinked addresses actually controlled by the same bot?"

## Processing Logic

The module employs a multi-stage analysis pipeline:

1.  **Data Preparation & Sequence Encoding**: For each address, the module fetches its transaction history and transforms it into several symbolic sequences. For example, a transaction of `0.05 ETH` might be encoded as `'small'`, while a call to Uniswap becomes `'swap'`. This creates sequences for value, method, direction, and timing.

2.  **Sequential Pattern Mining**: The module uses N-gram analysis and an Apriori-like frequency counting algorithm to find commonly recurring subsequences across all addresses. This discovers common TTPs, such as the pattern `['deposit_mixer'] -> ['transfer_out_small']`.

3.  **Behavioral Fingerprinting**: A unique, multi-component "fingerprint" is generated for each address by calculating a vector of statistical metrics. This captures the address's typical behavior across several dimensions (see "Behavioral Fingerprinting" section below for details).

4.  **Attribution & Clustering**:
    *   **Clustering**: The module uses the **DBSCAN clustering algorithm** to group addresses that have similar fingerprints, identifying archetypes of on-chain behavior.
    *   **Attribution**: It calculates the **cosine similarity** between the behavioral fingerprints of different addresses. A high similarity score (>0.8) is a strong indicator that two addresses are controlled by the same entity or bot, creating a high-confidence attribution link.

5.  **Automation & Evasion Scoring**: Dedicated functions analyze fingerprints and transaction sequences for specific red flags. `_calculate_automation_score` looks for highly regular timing and consistent gas prices, while `_calculate_evasion_score` looks for randomized patterns designed to avoid detection.

6.  **Database Storage**:
    *   **`frequent_behavioral_patterns`**: The most common TTPs discovered across the dataset are stored here.
    *   **`attribution_links`**: High-confidence links between addresses based on behavioral similarity are stored for review and potential inclusion in consensus clustering.
    *   **`risk_components`**: For addresses exhibiting highly automated or evasive behavior, a `BEHAVIORAL_SEQUENCE` risk component is added, contributing to the final unified risk score.

## Key Algorithms and Techniques

| Technique | Algorithm/Method | Purpose |
| :---: | --- | --- |
| **Sequence Mining** | N-gram Analysis, Apriori-like Subsequence Mining | Finds common, ordered patterns of behavior (TTPs) across many addresses. Uses quantile-based binning for value and gas price encoding. |
| **Behavioral Fingerprinting** | Multi-component Statistical Metrics (Entropy, Coefficient of Variation, etc.) | Creates a unique, quantitative profile of an address's behavior across temporal, economic, and operational dimensions. |
| **Clustering** | **DBSCAN** | Groups addresses with similar behavioral fingerprints into clusters, identifying behavioral archetypes and outliers. |
| **Attribution** | **Cosine Similarity** | Measures the similarity between two behavioral fingerprints to identify addresses likely controlled by the same entity. |
| **Probabilistic Modeling** | **Markov Chains** | Models the probability of transitioning from one action to the next, helping to predict an actor's likely next move based on past behavior. |

## Behavioral Fingerprinting

A core component of this module is the creation of a detailed "fingerprint" for each address. This fingerprint is a vector of metrics that quantitatively describes the address's typical behavior. It is composed of several sub-fingerprints:

-   **Temporal Fingerprint**: Captures timing patterns.
    -   Metrics: `hour_entropy`, `day_entropy`, `inter_tx_mean` (average time between transactions), `inter_tx_std` (standard deviation of time between transactions), `activity_regularity`, `weekend_ratio`, `night_ratio`.
-   **Economic Fingerprint**: Captures value and volume patterns.
    -   Metrics: `value_entropy`, `round_amount_ratio`, `dust_ratio`, `large_tx_ratio`, `in_out_ratio`, `volume_consistency`, `economic_efficiency`.
-   **Operational Fingerprint**: Captures on-chain operational patterns.
    -   Metrics: `method_diversity`, `method_entropy`, `contract_interaction_ratio`, `self_transaction_ratio`.
-   **Gas Fingerprint**: Captures gas usage patterns.
    -   Metrics: `gas_price_consistency`, `gas_usage_consistency`, `gas_price_entropy`.

## Key Sequential Patterns (TTPs) and Rationale

| Pattern | Name | Example Sequence | Rationale |
|:---:|---|---|---|
| 1 | **Mixer Deposit & Dispersal** | `['deposit_mixer'] -> ['transfer_out_small'] -> ['transfer_out_small']` | A classic layering technique. An address deposits into a mixer and then immediately begins peeling off small amounts to different destinations. |
| 2 | **Structuring & Consolidation** | `['transfer_in_small'] -> ['transfer_in_small'] -> ['deposit_mixer_large']` | The reverse of dispersal. An address collects small amounts from various sources (structuring) before consolidating them into a large mixer deposit. |
| 3 | **Automated Airdrop Farming** | `['claim_airdrop'] -> ['approve_token'] -> ['transfer_out_token']` | A common pattern for bots that automatically claim airdropped tokens and immediately send them to a central wallet. |
| 4 | **Repetitive Arbitrage/Swap** | `['swap'] -> ['swap'] -> ['swap']` within a short time frame. | Highly repetitive swapping activity is characteristic of arbitrage bots or, in some cases, wash trading to generate artificial volume. |
| 5 | **Scheduled/Timed Payments** | `['out'] -> [wait(24h)] -> ['out'] -> [wait(24h)]` | Transactions occurring at highly regular time intervals are a strong indicator of a scripted or automated payment schedule, not typical human behavior. |

## Role in the Pipeline

The `AdvancedBehavioralSequenceMiner` provides a powerful narrative context to an address's risk profile. It moves beyond "what an address did" to "how an address did it."
- It is exceptionally good at identifying **automation and bot-like activity**.
- It provides concrete evidence of specific **money laundering TTPs**.
- It can **attribute** unlinked addresses to a single actor based on shared behavioral traits.
- The patterns it finds are high-quality, explainable evidence that significantly boosts the confidence of a high-risk assessment in the `UnifiedRiskScorer`.

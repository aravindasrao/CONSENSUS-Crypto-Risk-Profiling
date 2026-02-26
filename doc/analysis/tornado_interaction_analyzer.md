# Tornado Interaction Analyzer

This document outlines the logic and rules used by the `TornadoInteractionAnalyzer` module. This module performs a specialized, deep-dive analysis into how addresses and clusters interact with Tornado Cash-like mixer protocols.

## Overview

While other modules identify mixer interactions in general, the `TornadoInteractionAnalyzer` is purpose-built to understand the nuances of Tornado Cash usage. It heuristically discovers potential mixer contracts, analyzes the frequency and value patterns of interactions, and flags behaviors indicative of sophisticated or illicit mixing activity.

## Processing Logic

1.  **Discover Tornado Contracts**: The module does not rely on a hardcoded list. It first queries the database to heuristically identify potential Tornado Cash contracts based on two criteria:
    *   **Method Names**: Contracts with a high number of interactions (>100) involving method names like `deposit` or `withdraw`.
    *   **Value Patterns**: Contracts that frequently transact in standard Tornado Cash denominations (0.1, 1, 10, 100 ETH).

2.  **Identify Interacting Addresses**: It fetches all addresses that have sent funds to or received funds from these identified mixer contracts.

3.  **Aggregate by Address/Cluster**: The module **consumes existing cluster data**. It joins the interacting addresses with the `addresses` table to retrieve their pre-assigned `cluster_id`. The analysis is then performed on a per-address basis, but the results are associated with the entity's cluster. It does not perform any clustering itself.

4.  **Pattern Detection & Scoring**: For each address, the module analyzes its transaction history with the mixer contracts to identify suspicious patterns and calculate risk indicators (see table below).

5.  **Database Storage**:
    *   A detailed summary of the analysis for each address is stored in the `tornado_analysis_results` table.
    *   For each address with suspicious indicators, a `TORNADO_CASH_INTERACTION` risk component is added to the `risk_components` table. The score is based on the number and severity of the detected indicators.

## Key Interaction Patterns and Rationale

| Pattern | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **High-Frequency Usage** | An address has a high number of interactions (>10) with mixer contracts. | Indicates systematic and repeated use of mixers, which is more suspicious than casual, one-off usage for privacy. |
| 2 | **Large Volume Mixing** | An address mixes a large total volume of cryptocurrency (e.g., > 50 ETH). | Laundering large sums of money requires moving significant volume through mixers. This is a primary indicator of high-risk activity. |
| 3 | **Imbalanced Ratio** | An address has a highly skewed ratio of deposits to withdrawals (e.g., many deposits but no withdrawals, or vice-versa). | This can indicate that the address is only one part of a larger, more complex laundering operation, acting solely as a deposit or withdrawal point. |
| 4 | **Multiple Contract Usage** | An address interacts with several different mixer contracts. | Using multiple contracts can be a technique to further obfuscate fund flows, making the trail harder for investigators to follow. |
| 5 | **Rapid Interactions / Quick Turnaround** | An address makes multiple deposits or withdrawals in a short time (e.g., < 1 hour), or withdraws funds very soon after depositing (e.g., < 2 hours). | This machine-like timing is not typical of human behavior and strongly suggests the use of automated scripts to quickly pass funds through the mixer. |
| 6 | **Regular Timing Intervals** | The time gaps between an address's mixer interactions have a very low standard deviation. | This is a powerful indicator of a bot or scripted process executing transactions on a fixed schedule. |

## Role in the Pipeline

The `TornadoInteractionAnalyzer` provides direct, high-impact evidence of risk. Interaction with a known mixer is a significant red flag. By dissecting *how* an entity uses the protocol, this module provides the `UnifiedRiskScorer` with granular, high-confidence evidence to justify a `HIGH` or `CRITICAL` risk rating. It helps distinguish between a casual privacy-seeking user and a systematic, large-scale money launderer.
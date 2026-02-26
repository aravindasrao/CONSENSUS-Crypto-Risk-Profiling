# ZK Proof Analyzer

This document outlines the logic and methodology used by the `ZKProofAnalyzer` module. This module is an advanced research component designed to probe the anonymity guarantees of privacy protocols like Tornado Cash. It moves beyond simple transaction analysis to investigate potential weaknesses in the implementation and usage patterns of zero-knowledge proofs.

## Overview

While other modules analyze transaction flows, the `ZKProofAnalyzer` focuses on the metadata and side-channel information surrounding ZK-proof-based transactions. Instead of cryptographically verifying the proofs themselves, it uses statistical and machine learning techniques to identify patterns that could lead to de-anonymization. It aims to answer questions like, "Can we link a deposit and withdrawal by analyzing the gas price, transaction timing, or value clusters, even if the ZK proof is valid?"

## Processing Logic

The module follows a multi-phase analysis pipeline, orchestrated by the `run_comprehensive_zk_analysis` method.

1.  **Isolate Mixer Transactions**: The module first identifies a dataset of transactions involving known privacy protocols like Tornado Cash by looking for method names like `deposit` and `withdraw`.

2.  **Nullifier Linkability Analysis**: This is the core of the analysis. It does not check for direct nullifier reuse but instead analyzes metadata to find statistical links between deposits and withdrawals.
    *   **Temporal Analysis**: It groups transactions into time windows (e.g., 1 hour) and calculates the entropy of the transaction frequency. Low entropy (highly regular or bursty activity) can indicate coordinated behavior.
    *   **Value Pattern Analysis**: It uses clustering algorithms (like DBSCAN) on transaction values to identify distinct groups of users who may be using non-standard amounts, making them easier to link.
    *   **Gas Pattern Analysis**: It analyzes the entropy of gas prices and gas used. A user who consistently uses a unique or non-random gas price for both deposits and withdrawals may compromise their own anonymity.
    *   **Collision Simulation**: It identifies blocks with an unusually high number of mixer transactions, which represent periods of higher potential for both anonymity and linkability analysis.

3.  **Commitment, Privacy, and Circuit Analysis**: The framework includes dedicated phases for analyzing commitment schemes, detecting side-channel privacy leaks, and assessing zk-SNARK circuit properties. *Note: These are advanced research components and are currently implemented as placeholders in the code, returning simulated results to demonstrate the full analytical framework.*

4.  **Database Storage**: A summary of the analysis, including the overall linkability score and key statistical findings, is stored in the `advanced_analysis_results` table for system-wide review.

## Key Heuristics and Checks

The analyzer uses statistical heuristics rather than direct cryptographic checks to infer risk.

| Heuristic | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Temporal Entropy** | Measures the randomness of transaction timing. A low entropy (very regular or very bursty activity) is flagged. | Automated scripts or coordinated actors often exhibit non-random timing patterns, which can be used to link their activities. |
| 2 | **Value Clustering** | Groups transactions by value to find users who use non-standard deposit/withdrawal amounts. | While Tornado Cash uses standard denominations, users who deposit and withdraw unique, non-standard amounts can be easily linked. |
| 3 | **Gas Price Consistency** | Measures the entropy of gas prices used by an address. A low entropy (always using the same gas price) is flagged. | Most wallets randomize gas prices. A user who manually sets a specific, consistent gas price for both deposits and withdrawals creates a strong linkable signature. |
| 4 | **Block Congestion** | Identifies blocks with an unusually high number of mixer transactions. | These "congested" blocks are hot-spots for analysis, as they contain a high density of potentially linkable deposit/withdrawal pairs. |

## Role in the Pipeline

The `ZKProofAnalyzer` is a specialized research module that serves a strategic purpose.

- It provides a **quantitative assessment of anonymity risk** by analyzing metadata and side-channel information, complementing other behavioral analyses.
- A finding from this module does not necessarily indicate a flaw in the ZK-proof itself, but rather a **weakness in the user's operational security (OPSEC)** that could compromise their privacy.
- It is a crucial tool for **detecting sophisticated de-anonymization vectors** that are invisible to standard flow analysis.

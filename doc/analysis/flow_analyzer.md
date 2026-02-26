# Flow Analyzer

This document outlines the logic and rules used by the `FlowAnalyzer` module. This module is designed to analyze the flow of funds into, out of, and within address clusters to detect well-known money laundering typologies.

## Overview

The `FlowAnalyzer` moves beyond single-transaction analysis to examine sequences and patterns of transactions associated with a cluster. By constructing a directed graph of a cluster's financial activity, it can algorithmically identify behaviors that are highly indicative of attempts to obfuscate the source or destination of funds.

## Processing Logic

1.  **Data Ingestion**: The analyzer fetches all transactions belonging to any cluster and groups them by `cluster_id`. This bulk-processing approach is highly efficient.

2.  **Graph Construction**: For each cluster, it builds a directed `networkx` graph (`DiGraph`) to model the internal and external flow of funds.
    -   **Nodes**: Addresses involved in the transactions.
    -   **Edges**: A directed edge is created from a `from_addr` to a `to_addr` for each transaction. If multiple transactions occur between the same two addresses, they are aggregated into a single edge with the following attributes:
        -   `weight`: The sum of all transaction values (in ETH) between the two addresses.
        -   `transaction_count`: The total number of transactions between the two addresses.
        -   `transactions`: A list of all transaction hashes for that edge.
        -   `first_timestamp`: The timestamp of the earliest transaction.
        -   `last_timestamp`: The timestamp of the most recent transaction.

3.  **Topological Analysis**: The constructed graph is passed to the `GraphTopologyAnalyzer` to calculate a comprehensive set of network metrics (e.g., centrality, density, etc.), providing a structural overview of the cluster's activity.

4.  **Pattern Detection**: The module runs a series of algorithms on both the transaction data and the graph to detect specific, suspicious flow patterns (see table below). This includes looking for rapid chains, fan-in/fan-out structures, and cycles.

5.  **Evidence Generation & Risk Scoring**: When a suspicious pattern is detected (e.g., a rapid, high-value chain), the analyzer generates a detailed piece of evidence.
    -   It assigns a risk score to the pattern (e.g., `0.8` for a suspicious rapid chain).
    -   It identifies all addresses involved in the suspicious flow.

6.  **Database Storage**:
    -   A `network_flow` risk component is added to the `risk_components` table for *every address* involved in a suspicious flow. This score contributes to the address's final unified risk score.
    -   A detailed summary of the cluster's flow analysis, including all detected patterns and suspicious flows, is stored in the `advanced_analysis_results` table for deeper forensic review.

## Key Flow Patterns and Rationale

| Pattern | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Rapid Flow Chain** | A sequence of connected transactions occurring in a very short time frame (e.g., under an hour). | Indicates automated activity, often used to quickly move funds through multiple hops to break the trail. |
| 2 | **Fan-Out / Distribution** | A single address sending funds to many different addresses. `(Source -> Dest_1, Dest_2, ...)` | Classic dispersal pattern used after a hack or to distribute funds from a central wallet to many smaller ones. It's a key part of peeling chains. |
| 3 | **Fan-In / Collection** | Multiple addresses sending funds to a single destination address. `(Source_1, Source_2, ... -> Destination)` | Common pattern for consolidating funds before moving them to an exchange or another mixer. It's a key part of structuring/smurfing. |
| 4 | **Circular Flow** | Funds are sent through a series of addresses and eventually return to or near the original address. `(A -> B -> C -> A)` | This pattern has no economic purpose and is a strong indicator of wash trading or attempts to artificially inflate transaction volume or obfuscate the true source of funds. |
| 5 | **Layered Flow** | A simple, linear chain of transactions moving through multiple intermediary addresses. `(A -> B -> C -> D ...)` | Used to add hops to a transaction trail, making it more difficult for investigators to connect the original source with the final destination. |

## Role in the Pipeline

The `FlowAnalyzer` provides some of the strongest evidence of intentional money laundering activity. While a single transaction to a mixer is suspicious, the detection of a multi-step peeling chain or structuring pattern provides a narrative of illicit behavior.

The findings from this module are critical inputs for the `UnifiedRiskScorer`. The presence of these patterns significantly increases a cluster's final risk score, often elevating it to `HIGH` or `CRITICAL` risk, as they demonstrate a deliberate attempt to obfuscate financial trails.
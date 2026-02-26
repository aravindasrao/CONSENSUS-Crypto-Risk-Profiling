# Incremental DFS Clusterer

This document outlines the logic and rules used by the `IncrementalDFSClusterer` module. This module is responsible for the first and most fundamental clustering pass, designed to group addresses with a high degree of certainty.

## Overview

The primary goal of this module is to apply the **shared input heuristic**. This heuristic states that if multiple addresses are used as inputs in the same transaction, they are likely controlled by the same entity. The `IncrementalDFSClusterer` extends this concept by building a graph of connections and then using a Depth-First Search (DFS) algorithm to find all transitively connected addresses.

The process is designed to be both forensically sound and scalable, using a two-phase "hybrid" approach.

## Processing Logic: Hybrid Address-Based Clustering

Instead of processing transactions one-by-one, the clusterer uses a more efficient two-phase approach:

1.  **Phase 1: Connection Graph Construction**: The module first iterates through all unique addresses involved in the dataset. For each address, it analyzes its transactions to identify valid, forensically relevant connections to other addresses based on a strict set of rules (see below). This process builds an in-memory graph where nodes are addresses and an edge represents a high-confidence link.

2.  **Phase 2: DFS Clustering**: After the complete connection graph is built, a standard Depth-First Search (DFS) algorithm is run. The DFS traverses the graph to find all distinct **connected components**. Each component, representing a group of addresses that are all transitively linked, is then assigned a unique cluster ID.

This approach is significantly faster than transaction-by-transaction processing and allows for sophisticated filtering before the final clustering step.

## Key Clustering Rules and Rationale

The quality of the clustering depends entirely on the rules used to establish connections. The `IncrementalDFSClusterer` uses the following rules to ensure high precision.

| Rule | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Shared Input Heuristic** | The fundamental rule. If multiple addresses (`A`, `B`, etc.) must provide authorization (e.g., via signatures or token approvals) for a single transaction to be executed, they are considered linked. The link is made between these authorizing addresses. The DFS algorithm then extends this transitively. | This is the most reliable heuristic in blockchain analysis. Whether it's multiple UTXOs in Bitcoin or multiple addresses authorizing a smart contract call in Ethereum, this scenario requires simultaneous control over all the private keys, providing strong evidence of common ownership. |
| 2 | **Forensic Filtering** | Connections are **not** made through known high-volume, public contracts. This includes major exchanges (e.g., Binance, Kraken), high-traffic DeFi protocols (e.g., Uniswap), and sanctioned entities. | These contracts are public utilities. Thousands of unrelated users interact with them daily. Including them in clustering would incorrectly merge vast numbers of unrelated addresses into a single, meaningless "super cluster", destroying forensic value. |
| 3 | **Maximum Cluster Size** | A configurable limit (e.g., 1,000 addresses) is placed on the size of any single cluster. If a connected component found by DFS exceeds this limit, it is **not** formed into a cluster, and its members are treated as singletons. | This is a critical safeguard against over-clustering. If a cluster grows excessively large, it has likely absorbed a semi-public service or a smart contract that was not in the initial exclusion list. This rule quarantines the "bad" cluster, preserving the integrity and quality of other, smaller clusters. |
| 4 | **Minimum Transaction Value** | A configurable minimum ETH value (e.g., 0.01 ETH) is required for a transaction to be considered for creating a link. | This filters out "dusting" attacks and other low-value spam transactions that do not represent a meaningful flow of funds. It reduces noise in the graph and prevents spurious connections. |
| 5 | **Probabilistic Mixer Heuristic** | As a secondary, lower-confidence heuristic, the algorithm can link addresses that interact with the same mixer contract using a **standard denomination** (e.g., 0.1, 1, 10, 100 ETH) within a **72-hour window**. | This heuristic attempts to link deposits and withdrawals that are close in time and value, a common pattern for mixer usage. It is a probabilistic link, weaker than the shared input heuristic, but can help uncover connections that are deliberately trying to be obfuscated. Its inclusion here is for initial discovery; final confidence is determined by the `ClusterConsensusEngine`. |
| 6 | **Input Data Decoding** | The `input` (or `input_data`) field of a transaction is critical. The first 4 bytes of this hex string are the `methodId`, which identifies the function being called. The rest of the string contains the encoded arguments. | The raw `input` data is not human-readable. To understand the transaction's purpose, it must be decoded using the target contract's Application Binary Interface (ABI). The provided CSVs often contain pre-decoded `methodId` and `functionName` columns, which are derived from this `input` data and significantly speed up analysis. |

## Algorithm and Output

### Algorithm

1.  **Build Graph**:
    - For each unique address `A` in the dataset:
        - Analyze its transactions.
        - For each counterparty `B`, check if the connection `(A, B)` is valid according to the rules above (especially Rule 2).
        - If the connection is valid, add an edge between `A` and `B` in the graph.

2.  **Find Components (DFS)**:
    - Initialize an empty set of `visited` addresses.
    - For each address `A` in the graph:
        - If `A` has not been `visited`:
            - Start a DFS traversal from `A` to find its entire connected component.
            - Check the size of the component against the **Maximum Cluster Size** (Rule 3).
            - If the size is acceptable, assign a new, unique `cluster_id` to all addresses in the component.
            - Mark all addresses in the component as `visited`.

### Output

The final output of this module is a set of cluster assignments stored in the `cluster_assignments` table in the database. Each entry links an `address` to a `cluster_id` with a high confidence score, providing the foundational grouping for all subsequent analyses.

---

This high-precision clustering approach ensures that the initial groups are reliable, forming a solid base upon which more probabilistic or behavioral clustering methods can be layered by the `ClusterConsensusEngine`.
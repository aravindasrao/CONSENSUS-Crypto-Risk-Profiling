# Cross-Chain Analyzer

This document outlines the logic and rules used by the `CrossChainAnalyzer` module. This module is designed to identify and analyze entities that operate across multiple blockchains, providing a more holistic view of their activities.

## Overview

The core principle of cross-chain analysis in an EVM (Ethereum Virtual Machine) context is that **a single private key generates the same public address on all EVM-compatible chains**. This means if the address `0xabc...` is seen on Ethereum, Polygon, and Binance Smart Chain, it is controlled by the same entity.

The `CrossChainAnalyzer` leverages two primary methods to detect this activity:
1.  **Heuristic-Based Analysis**: By identifying interactions with known cross-chain bridge contracts and specific function calls (`bridge`, `swap`, `lock`), the analyzer can infer cross-chain movement even when analyzing data from a single blockchain. This is the primary method used by the current implementation.
2.  **Identical Address Heuristic**: When transaction data from multiple blockchains is available and tagged with a `chain_id`, the analyzer can use the cryptographically certain link of identical addresses across chains.

## Processing Logic

1.  **Analyze Bridge Interactions**: The analyzer first identifies transactions with known cross-chain bridge contracts (e.g., Hop, Multichain, Connext) by looking for interactions with known bridge addresses and common method names (`bridge`, `swap`, `lock`). This provides concrete evidence of assets being moved between chains.

2.  **Identify Cross-Chain Addresses (when multi-chain data is available)**: As a secondary, higher-confidence method, the module can query the `transactions` table to find addresses that are associated with more than one unique `chain_id`.

3.  **Group by Cluster**: It then aggregates these findings at the cluster level. Since this analysis runs before the final consensus engine, it uses the high-confidence clusters available at the end of Phase 1 (typically from the `IncrementalDFSClusterer`). If any address within one of these clusters is active on a chain, the entire cluster is considered active on that chain.

4.  **Propagate Risk**: The most critical function is risk propagation. If a cluster has a high risk score on one chain (e.g., due to direct interaction with a sanctioned mixer on Ethereum), that high-risk status is applied to the cluster's entire presence across all other chains it operates on.

## Key Rules and Rationale

| Rule | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Bridge Interaction Flagging** | Transactions to or from known bridge contracts are flagged. The module identifies bridges by searching for common method names (e.g., `bridge`, `swap`, `xcall`) and keywords (e.g., `wormhole`, `hop`) in both the `methodName` and full `functionName` fields. | This is the primary heuristic for detecting cross-chain activity from a single-chain dataset. These transactions are explicit indicators of fund movement and serve as critical nodes in a forensic graph. |
| 2 | **Identical Address Heuristic** | An address string that appears on multiple EVM chains (differentiated by `chain_id`) is considered to be controlled by the same entity. | This is a cryptographically certain link and the gold standard for cross-chain analysis. Controlling the address on any EVM chain requires possession of the corresponding private key. |
| 3 | **Cluster-Level Aggregation** | If address `A` (in Cluster `C1`) is on Ethereum and address `B` (also in `C1`) is on Polygon, the entire cluster `C1` is considered active on both chains. | Since all addresses in a high-confidence cluster are controlled by the same entity, their collective activity across chains represents the entity's full operational scope. |
| 4 | **Risk Propagation** | A high risk score associated with a cluster on Chain X is automatically applied to that same cluster's activity on Chain Y and Chain Z. | Illicit actors cannot "wash" their risk by simply moving funds to another chain. This rule ensures that risk is persistent and follows the entity, not just the asset on a single ledger. |

### Detected Bridge Transaction Patterns

The analyzer identifies the following specific patterns in an address's bridge interactions to calculate its risk score:
- **`high_bridge_activity`**: More than 10 bridge transactions.
- **`rapid_bridge_transactions`**: Multiple bridge interactions within a short time frame (e.g., under an hour), indicating automated activity.
- **`large_value_bridging`**: Transferring substantial amounts of cryptocurrency (e.g., >50 ETH) through bridges.
- **`multiple_bridge_contracts`**: Using a variety of different bridge services, a common tactic for obfuscation.
- **`diverse_bridge_methods`**: Using multiple different function calls (e.g., `swap`, `lock`, `xcall`), suggesting sophisticated usage.
- **`consistent_bridge_amounts`**: Repeatedly sending identical or similar amounts, often a sign of automated scripts.
- **`round_value_bridging`**: A high proportion of transactions with "clean" round numbers (e.g., 10.0 ETH, 100.0 ETH).

## Output

The `CrossChainAnalyzer` generates several key outputs that are stored in the database:

- **Advanced Analysis Results**: A summary report is stored in the `advanced_analysis_results` table with `analysis_type = 'cross_chain'`. This report details which clusters are active on which chains and highlights high-risk cross-chain entities.
- **Risk Components**: A `CROSS_CHAIN_RISK_PROPAGATION` risk component is added to the `risk_components` table for any address exhibiting suspicious cross-chain behavior. This score is a direct and critical input for the `UnifiedRiskScorer`, which aggregates it with findings from all other analyzers to compute the final, holistic risk score for the address.

## Role in the Pipeline

The `CrossChainAnalyzer` provides critical context that is often missed by single-chain analysis tools. It is essential for uncovering sophisticated laundering techniques where actors:

- Deposit illicit funds into a mixer on a high-security chain (e.g., Ethereum).
- Bridge the "clean" funds to a less-regulated or lower-cost chain (e.g., a new L2).
- Use the funds in DeFi or cash out through an exchange on the second chain.

By linking these activities, the analyzer ensures a complete and accurate forensic picture of the entity's behavior.
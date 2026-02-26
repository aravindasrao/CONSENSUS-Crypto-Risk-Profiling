# Comprehensive Contract Database

**Primary File:** `src/utils/comprehensive_contract_database.py`

## 1. Overview

The `ComprehensiveContractDatabase` is a critical utility module that acts as an intelligent filter and profiler for smart contracts. Its primary purpose is to support the clustering process by identifying, categorizing, and filtering contract addresses based on their known function and on-chain behavior.

While the `IncrementalDFSClusterer` builds the graph of connections, the `ComprehensiveContractDatabase` is the "brain" that tells it *which* connections are forensically significant and which should be ignored. This prevents the clustering algorithm from incorrectly merging thousands of unrelated user addresses that have all interacted with a common public service like Uniswap or a centralized exchange.

## 2. Key Responsibilities

1.  **Maintain a Registry of Known Contracts**: It loads and manages a list of known contract addresses from `known_contracts.json`, categorizing them into types like `exchange`, `mixer`, `sanctioned`, `bridge`, etc.
2.  **Filter Non-Forensically Relevant Interactions**: Its main function, `get_forensic_connection_targets`, implements **Rule 2 (Forensic Filtering)**. It removes interactions with high-volume, public utility contracts that would otherwise pollute the clustering results.
3.  **Discover and Profile Unknown Contracts**: It includes a powerful discovery mechanism (`discover_unknown_contracts`) that queries the transaction database to find and profile addresses that behave like smart contracts but are not yet in the known list.
4.  **Provide Clean Targets for Clustering**: By pre-filtering transactions, it provides the clustering algorithms with a much cleaner, more relevant dataset, leading to higher-quality, more meaningful clusters.

## 3. Core Components

### `known_contracts.json`

This external JSON file is the source of truth for all pre-identified smart contracts. The system relies on this file to bootstrap its knowledge of the blockchain ecosystem.

*   **Structure**: The file is a dictionary where keys are contract categories (e.g., `exchange`, `mixer`, `sanctioned`) and values are lists of contract addresses belonging to that category.
*   **Importance**: Correctly categorizing contracts here is essential.
    *   `exchange`, `high_volume_contract`, `bridge`: These are typically excluded from clustering to prevent merging unrelated users.
    *   `mixer`, `sanctioned`: These are critical for identifying specific behaviors and are handled with special care by the analysis modules.

### Contract Categories

The database uses several internal sets to quickly check the category of an address:

*   `self.exchanges`: Centralized or decentralized exchanges (e.g., Binance, Uniswap).
*   `self.mixers`: Privacy-preserving protocols (e.g., Tornado Cash).
*   `self.sanctioned`: Addresses sanctioned by regulatory bodies (e.g., OFAC list).
*   `self.bridges`: Cross-chain bridge contracts.
*   `self.high_volume_contracts`: A catch-all for other high-traffic contracts that should be excluded from clustering.

## 4. Key Methods and Logic

### `__init__(self, database: SQLiteDB, known_contracts_path: str)`

The constructor initializes the database connection and loads all contract categories from the specified JSON file into fast-lookup sets.

### `get_forensic_connection_targets(self, address: str, address_txs: pd.DataFrame) -> Set[str]`

This is the primary method used by the `IncrementalDFSClusterer`. It acts as a sophisticated filter.

1.  **Input**: The address being analyzed and a DataFrame of its transactions.
2.  **Process**:
    *   It identifies all unique "counterparty" addresses (the `to_address` in outgoing transactions and `from_address` in incoming ones).
    *   It iterates through these counterparties and **discards** any that are identified as:
        *   An exchange (`_is_exchange`).
        *   A sanctioned entity (`_is_sanctioned`).
        *   A high-volume contract (`_is_high_volume_contract`).
    *   This filtering is the direct implementation of **Rule 2 (Forensic Filtering)** and is the most critical step in preventing the creation of meaningless "super clusters".
3.  **Output**: It returns a `set` of addresses that are considered forensically relevant and are safe to be considered for creating a connection in the graph.

### `discover_unknown_contracts(self, min_interactions: int = 100) -> Dict`

This method proactively finds new, unlabelled smart contracts.

1.  **Process**:
    *   It runs a SQL query against the `transactions` table.
    *   The query groups transactions by the `to_addr` and aggregates key metrics.
    *   It specifically looks for addresses where `method_name IS NOT NULL`, which is a strong indicator of a contract interaction (as opposed to a simple ETH transfer).
    *   It counts `total_interactions`, `unique_senders`, and `unique_methods` (derived from the `functionName` column in the data).
2.  **Profiling**: For each discovered contract, it calls `_estimate_contract_type` to make an educated guess about its function (e.g., `likely_defi_protocol`, `likely_bridge`).
3.  **Output**: A dictionary of newly discovered contract addresses and their profiled statistics. This can be used to update the `known_contracts.json` file.

### `_estimate_contract_type(self, contract_stats: pd.Series) -> str`

This internal helper uses a set of heuristics to classify a contract based on its on-chain footprint. The logic is as follows:

*   **High Tx Count & High Method Diversity**: Likely a complex `DeFi Protocol`.
    *   `tx_count > 1000 and unique_methods > 10`
*   **Low Method Diversity & High Sender Diversity**: Likely a `Bridge` or a simple token-wrapping contract where many users call a few standard functions (e.g., `deposit`, `withdraw`).
    *   `unique_methods <= 3 and sender_diversity > 0.5`
*   **High Tx Count & Low Sender Diversity**: Likely a `Bot` or an automated arbitrage contract.
    *   `tx_count > 500 and sender_diversity < 0.1`
*   **Default**: If none of the above, it is classified as a `generic_contract`.

This profiling helps analysts quickly understand the nature of newly discovered contracts and decide whether they should be added to the exclusion lists.
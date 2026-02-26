# Dynamic Temporal Network Analyzer

This document outlines the logic and rules used by the `DynamicTemporalNetworkAnalyzer` module. This module analyzes how the transaction graph evolves over time to detect coordinated behavior and temporal anomalies that are invisible to static analysis. It integrates with existing cluster and risk data to provide a time-aware layer to the analysis, with a specific focus on how clusters of addresses behave over time.

## Overview

The core idea is to slice the entire transaction history into a series of overlapping time windows (e.g., 24-hour windows with a 12-hour overlap). By creating a "network snapshot" for each window and comparing these snapshots, the analyzer can identify significant changes in activity, structure, and behavior.

## Processing Logic

1.  **Time Window Creation**: The module first determines the full time span of all transactions and divides it into a series of overlapping temporal windows.

2.  **Snapshot Analysis (Intra-Window)**: For each time window, the module performs a localized analysis:
    *   It builds a directed graph of all transactions that occurred within that window.
    *   It calculates a rich set of network metrics for this snapshot (e.g., density, node count, edge count, component count, centrality).
    *   It detects intra-window anomalies, such as a single address having an unusually high degree (`high_degree_node`) or rapid transaction sequences.
    *   It analyzes Tornado Cash-specific activity and performs a temporal analysis on existing address clusters within the window.

3.  **Evolutionary Analysis (Inter-Window)**: The module then compares the metrics and cluster activity from consecutive snapshots to detect significant changes and patterns over time. This includes:
    *   Calculating network growth rates and changes in density.
    *   Detecting the synchronized activation of multiple, previously dormant address clusters.

4.  **Anomaly Identification**: It identifies system-wide temporal anomalies by analyzing the sequence of snapshot metrics. The most prominent example is detecting an `activity_spike`.

5.  **Database Storage**: A summary of the findings, particularly system-wide temporal anomalies like activity spikes, is stored in the `advanced_analysis_results` table for review and integration into the `UnifiedRiskScorer`.

## Key Temporal Patterns and Rationale

| Pattern | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Activity Spike** | A time window where the number of transactions (edges) is significantly higher (e.g., >2 standard deviations) than the moving average across all windows. | This is a powerful indicator of a coordinated event. It could signify the start of a fund dispersal after a hack, a mass deposit event into a mixer, or a synchronized response to a market event. |
| 2 | **Network Growth Spike** | A rapid, anomalous increase in the number of new nodes or edges in a time window compared to previous windows. | Suggests that an entity is activating a large number of new wallets simultaneously, often in preparation for a layering or smurfing operation. |
| 3 | **Density Shift** | A significant change in the graph's density between consecutive windows. | A sudden increase in density can indicate that addresses within a group are rapidly consolidating or exchanging funds. A sudden decrease might suggest funds are being moved out of the observed system. |
| 4 | **High-Degree Node (Temporary Hub)** | Within a single time window, an address has an unusually high number of connections (degree) compared to other nodes in that same window (e.g., >95th percentile). | This identifies temporary hubs used for rapid collection or distribution of funds that might not appear as significant in a static analysis of the entire dataset. |
| 5 | **Rapid Transaction Sequence** | A series of transactions occurring in very quick succession (e.g., less than 1 minute apart) within a single window. | Indicates automated activity, often used to quickly move funds through multiple hops to break the trail. |
| 6 | **Synchronized Cluster Activation** | A time window where an anomalously high number of distinct address clusters become active after being dormant in the previous window. | This is a very strong signal that these disparate clusters are controlled by a single entity or a coordinated group, acting on a shared timeline to begin a new phase of an operation. |

## Role in the Pipeline

The `DynamicTemporalNetworkAnalyzer` provides a crucial time-series perspective that complements other analyses:

- It can detect the **"moment of coordination"** for illicit activities.
- It helps distinguish between routine, random transactions and planned, synchronized events.
- The temporal anomalies it finds (e.g., an activity spike on a specific date) can be correlated with external, off-chain events (like a known DeFi protocol hack) to strengthen attribution and forensic narratives.
- Its findings are used by the `UnifiedRiskScorer` to increase the risk score of addresses that participate in suspicious temporal patterns.
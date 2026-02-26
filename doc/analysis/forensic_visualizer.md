# Forensic Visualizer

This document outlines the `ForensicVisualizer` module, which is responsible for generating a variety of plots and graphs to visually represent the results of the forensic analysis.

## Overview

The goal of the `ForensicVisualizer` is to translate complex numerical data and network relationships into intuitive, human-understandable charts and diagrams. These visuals are essential for reports, presentations, and for quickly grasping the key findings of the analysis. All visuals are saved to the `results/` directory.

## Processing Logic

1.  **Data Ingestion**: The visualizer queries the database for final, processed data, including final risk scores, consensus cluster assignments, and transaction details for high-risk entities.
2.  **Plot Generation**: It uses libraries like `matplotlib`, `seaborn`, and `networkx` to create a suite of standard visualizations.
3.  **Dynamic Selection**: For entity-specific plots (like network graphs), it automatically selects the highest-risk clusters or addresses to visualize, ensuring the most critical findings are highlighted.
4.  **File Output**: Each visualization is saved as a high-resolution PNG file in the `results/` directory, ready to be embedded in reports.

## Generated Visualizations

The `ForensicVisualizer` generates the following key plots:

### 1. `risk_score_distribution.png`
-   **Type**: Histogram
-   **Purpose**: Shows the overall distribution of final risk scores across all analyzed addresses. This helps to quickly understand the general risk profile of the entire dataset (e.g., is it mostly low-risk, or are there many high-risk entities?).

### 2. `top_risk_clusters_summary.png`
-   **Type**: Bar Chart
-   **Purpose**: Displays the top N riskiest clusters, showing their total size (number of addresses) and their average final risk score. This is a key chart for high-level executive summaries.

### 3. `cluster_transaction_graph_{cluster_id}.png`
-   **Type**: Network Graph
-   **Purpose**: Provides a detailed, visual representation of the internal transaction flows within a single high-risk cluster.
-   **Features**:
    -   Nodes represent addresses, sized by transaction volume.
    -   Nodes are colored by their individual risk score (e.g., red for high-risk).
    -   Edges represent transactions, with arrows indicating the direction of fund flow.

### 4. `address_activity_timeline_{address}.png`
-   **Type**: Scatter Plot / Timeline
-   **Purpose**: Visualizes the activity of a single high-risk address over time. The x-axis is time, and the y-axis can represent transaction volume, showing bursts of activity or long periods of dormancy.

### 5. `risk_component_heatmap.png`
-   **Type**: Heatmap
-   **Purpose**: Shows the correlation between different risk components (e.g., `TORNADO_CASH_INTERACTION`, `multihop_risk`). This helps identify which types of risky behaviors tend to occur together, revealing common adversary TTPs.

### 6. `feature_importance.png`
-   **Type**: Bar Chart
-   **Purpose**: If a supervised model like the `AdvancedRealTimeRiskScorer` is used, this chart displays the top features that were most influential in its risk predictions, providing model explainability (XAI).

## Role in the Pipeline

The `ForensicVisualizer` is one of the final modules executed in the `run_phase_3_finalization` step. It runs after all scoring and clustering is complete, ensuring that the visuals it generates are based on the most definitive results of the analysis.
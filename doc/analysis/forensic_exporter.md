# Forensic Exporter

This document outlines the `ForensicExporter` module, which is responsible for generating a suite of human-readable and machine-readable reports from the final analysis results. These exports are the primary tangible outputs of the entire forensic pipeline.

## Overview

The exporter queries various tables in the database (`addresses`, `risk_components`, `suspicious_paths`, etc.) to consolidate complex analytical findings into clear, actionable CSV reports. Each report is designed for a specific purpose, from high-level summaries to granular evidence logs. All reports are saved to the `results/` directory.

## Generated Reports

The `ForensicExporter` generates the following reports:

### 1. `comprehensive_forensic_export.csv`

-   **Purpose**: Provides a high-level summary for every address with a calculated risk score. This is the main report for triaging and prioritizing addresses for investigation.
-   **Key Columns**:
    -   `address`: The blockchain address.
    -   `final_risk_score`: The unified risk score (0.0 to 1.0).
    -   `risk_category`: Human-readable risk level (e.g., `CRITICAL`, `HIGH`).
    -   `final_confidence`: The consensus confidence in the cluster assignment.
    -   `cluster_id`: The final consensus cluster ID.
    -   `total_transactions`, `total_volume_in_eth`, `total_volume_out_eth`: Basic activity metrics.
    -   `is_contract`, `is_tornado`: Flags for address type.
    -   `first_seen`, `last_seen`: Activity timestamps.

### 2. `enhanced_forensic_report.csv`

-   **Purpose**: Offers a deeper "explainability" layer for each high-risk address, showing *why* an address received its score.
-   **Key Columns**:
    -   `address`, `final_risk_score`, `risk_category`, `cluster_id`: Core identifiers.
    -   `top_risk_contributor`: The single analysis module that contributed the most to the risk score.
    -   `risk_score_breakdown`: A comma-separated list of all contributing risk components and their scores (e.g., `TORNADO_CASH_INTERACTION:0.8, multihop_risk:0.85`).
    -   `primary_anomaly_reason`: The main reason an address was flagged as an anomaly by the `EnhancedAnomalyDetector`.
    -   `behavioral_pattern`: The behavioral archetype identified by the `DepositWithdrawalPatternAnalyzer` (e.g., `active_mixer`).

### 3. `forensic_evidence_log.csv`

-   **Purpose**: A granular, event-centric log of every specific piece of suspicious evidence found by the various analyzers. This serves as the raw data for an investigation.
-   **Key Columns**:
    -   `evidence_type`: The type of finding (e.g., `Suspicious Flow`, `Address Anomaly`, `Suspicious Path`).
    -   `description`: A human-readable summary of the finding.
    -   `risk_score`: The risk associated with this specific piece of evidence.
    -   `related_entity`: The address or cluster ID the evidence pertains to.
    -   `details_json`: A JSON blob with the full, detailed evidence for deep-dive analysis.
    -   `timestamp`: When the evidence was detected.

### 4. `system_insights_report.csv`

-   **Purpose**: Exports the results of system-wide analyses that are not tied to a single address, such as findings from the `DynamicTemporalNetworkAnalyzer`.
-   **Key Columns**:
    -   `analysis_type`: The name of the system-wide analysis (e.g., `dynamic_temporal_network`).
    -   `results_json`: A JSON blob containing the summary of findings (e.g., detected activity spikes across the entire dataset).
    -   `created_at`: Timestamp of the analysis.

### 5. `attribution_report.csv`

-   **Purpose**: Lists high-confidence pairs of addresses that are likely controlled by the same entity, based on behavioral similarity from the `AdvancedBehavioralSequenceMiner`.
-   **Key Columns**:
    -   `source_address`, `target_address`: The pair of linked addresses.
    -   `similarity_score`: The behavioral similarity score (0.0 to 1.0).
    -   `evidence_json`: The specific behavioral traits that were similar (e.g., `similar_temporal_patterns`).

### 6. `behavioral_patterns_report.csv`

-   **Purpose**: Lists the most common sequential patterns of behavior (Tactics, Techniques, and Procedures - TTPs) found across all addresses.
-   **Key Columns**:
    -   `pattern_type`: The dimension of the pattern (e.g., `value_pattern`, `method_pattern`).
    -   `sequence`: The actual pattern (e.g., `['small', 'small', 'large']`).
    -   `support`: The frequency of this pattern in the dataset.
    -   `length`: The length of the pattern.

### 7. `tornado_analysis_report.csv`

-   **Purpose**: Provides a detailed breakdown of every address that interacted with Tornado Cash, including specific usage patterns.
-   **Key Columns**:
    -   `address`, `cluster_id`: Identifiers.
    -   `deposit_count`, `withdrawal_count`, `total_volume_eth`: Usage statistics.
    -   `interaction_patterns`: JSON array of detected patterns (e.g., `rapid_tornado_interactions`).
    -   `risk_indicators`: JSON array of specific risk flags (e.g., `large_volume_mixing`).

## Role in the Pipeline

The `ForensicExporter` is the final module in the `run_phase_3_finalization` step. It runs after all clustering and scoring is complete, ensuring that its reports contain the most up-to-date and comprehensive data available from the analysis run.
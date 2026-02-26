# System Insights Report

**File**: `system_insights_report.csv`

## Purpose

This report exports the high-level, qualitative findings from system-wide analyses that are not tied to a single address. It captures macro trends and anomalies across the entire dataset.

## How to Use

Review this report to understand the overall state of the ecosystem being analyzed. It can reveal large-scale, coordinated events that might be missed when looking at individual addresses.

## Key Columns

- `analysis_type`: The name of the system-wide analysis (e.g., `dynamic_temporal_network`).
- `results_json`: A JSON blob containing the summary of findings.
- `created_at`: Timestamp of the analysis.

## Example Interpretation

An entry with `analysis_type` of `dynamic_temporal_network` might contain a `results_json` that details a massive spike in transaction volume across the entire network on a specific date. An investigator could then correlate this date with off-chain events, such as a widely reported DeFi protocol hack, to understand the cause of the spike.
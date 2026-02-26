# Enhanced Forensic Report

**File**: `enhanced_forensic_report.csv`

## Purpose

This report provides the "explainability" layer for an address's risk score. While the comprehensive report tells you *what* the risk is, this report tells you *why*. It connects the final score to the specific analysis modules and pieces of evidence that contributed to it.

## How to Use

Once you have identified a high-risk address from the comprehensive report, look it up here to understand the primary reasons for its high score. This is crucial for building a narrative for your investigation.

## Key Columns

- `top_risk_contributor`: The single analysis module that contributed the most to the risk score (e.g., `PEELING_CHAIN_DETECTED`, `multihop_risk`).
- `risk_score_breakdown`: A comma-separated list of all contributing risk components and their individual scores. This gives a complete picture of all the evidence against the address.
- `primary_anomaly_reason`: If the address was flagged as an anomaly, this column explains why (e.g., the specific feature that was an outlier).
- `behavioral_pattern`: The behavioral archetype identified by the `DepositWithdrawalPatternAnalyzer` (e.g., `active_mixer`, `systematic_user`).

## Example Interpretation

| address | final_risk_score | top_risk_contributor | risk_score_breakdown |
| :--- | :--- | :--- | :--- |
| `0x123...` | 0.92 | PEELING_CHAIN_DETECTED | `PEELING_CHAIN:0.9, multihop_risk:0.85` |

**Interpretation**: Address `0x123...` is high-risk primarily because the `FlowAnalyzer` detected a "Peeling Chain" pattern. The `MultiHopAnalyzer` also found suspicious paths, adding to the score. This tells an investigator to focus on fund flow obfuscation.
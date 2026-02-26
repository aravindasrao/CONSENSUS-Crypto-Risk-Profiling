# Forensic Evidence Log

**File**: `forensic_evidence_log.csv`

## Purpose

This is the most granular report, acting as an "evidence locker" for your investigation. It contains a timestamped log of every specific, concrete piece of suspicious evidence found by any of the analysis modules.

## How to Use

When conducting a deep-dive investigation on a specific address or cluster, filter this log by the `related_entity` to see all the evidence compiled against it. The `details_json` column contains the raw data (e.g., transaction hashes, involved addresses) needed to verify the finding on a block explorer.

## Key Columns

- `evidence_type`: The type of finding (e.g., `Suspicious Flow`, `Address Anomaly`, `Suspicious Path`).
- `description`: A human-readable summary of the finding.
- `risk_score`: The risk associated with this specific piece of evidence.
- `related_entity`: The address or cluster ID the evidence pertains to.
- `details_json`: A JSON blob with the full, detailed evidence for deep-dive analysis.
- `timestamp`: When the evidence was detected.

## Example Interpretation

| evidence_type | description | risk_score | related_entity |
| :--- | :--- | :--- | :--- |
| `Suspicious Path` | Type: long_chain, Hops: 8, Volume: 150.5 ETH | 0.85 | cluster_1175 |
| `Address Anomaly` | Method: isolation_forest, Feature: unusual_timing_score | 0.75 | `0x123...` |

**Interpretation**: This log shows two specific pieces of evidence: a suspicious 8-hop transaction path was found within cluster 1175, and address `0x123...` was flagged by the Isolation Forest model for its unusual transaction timing.
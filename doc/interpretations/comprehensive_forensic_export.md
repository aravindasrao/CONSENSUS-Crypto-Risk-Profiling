# Comprehensive Forensic Export

**File**: `comprehensive_forensic_export.csv`

## Purpose

This is the main, high-level report for the entire analysis. It provides a summary row for every address that was assigned a risk score. Its primary purpose is to allow an investigator to quickly triage and prioritize which addresses and clusters warrant a deeper look.

## How to Use

**Sort this file by `final_risk_score` in descending order.** The addresses at the top are your highest-priority targets for investigation. Use the `cluster_id` to group related addresses together.

## Key Columns

- `address`: The blockchain address.
- `final_risk_score`: The final, unified risk score from 0.0 (minimal risk) to 1.0 (critical risk).
- `risk_category`: A human-readable label for the risk score (e.g., `CRITICAL`, `HIGH`, `MEDIUM`).
- `cluster_id`: The final consensus cluster ID. All addresses with the same `cluster_id` are believed to be controlled by the same entity.
- `total_transaction_count`, `total_volume_eth`: Basic activity metrics for context.

## Example Interpretation

| address | final_risk_score | risk_category | cluster_id | total_transaction_count |
| :--- | :--- | :--- | :--- | :--- |
| `0x123...` | 0.92 | CRITICAL | 1175 | 54 |
| `0xabc...` | 0.85 | HIGH | 1175 | 32 |
| `0xdef...` | 0.55 | MEDIUM | 563 | 12 |

**Interpretation**: Addresses `0x123...` and `0xabc...` are your highest-risk targets. They also belong to the same cluster (`1175`), meaning they are likely controlled by the same entity and should be investigated together.
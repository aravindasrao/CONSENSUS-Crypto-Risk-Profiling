# Attribution Report

**File**: `attribution_report.csv`

## Purpose

This report lists high-confidence pairs of addresses that are likely controlled by the same entity. The linkage is not based on direct transactions but on the similarity of their behavioral "fingerprints" as determined by the `AdvancedBehavioralSequenceMiner`.

This is one of the most powerful outputs for expanding an investigation, as it can link seemingly unrelated wallets.

## How to Use

When investigating an address, check this report to see if it is linked to any others. If `address_A` is linked to `address_B`, any investigation into `A` should immediately be expanded to include `B`.

## Key Columns

- `source_address`, `target_address`: The pair of addresses that are behaviorally linked.
- `similarity_score`: A score from 0.0 to 1.0 indicating the strength of the behavioral similarity. Scores above 0.8 are considered high-confidence links.
- `evidence_json`: A JSON array listing the specific behavioral traits that were found to be similar. This provides the "why" for the link.

## Example Interpretation

| source_address | target_address | similarity_score | evidence_json |
| :--- | :--- | :--- | :--- |
| `0x123...` | `0xabc...` | 0.95 | `["similar_temporal_patterns", "similar_gas_fingerprint"]` |

**Interpretation**: Addresses `0x123...` and `0xabc...` are almost certainly controlled by the same person or group. Their transaction timing and gas usage patterns are nearly identical, which is statistically unlikely to occur by chance.
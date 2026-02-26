# Behavioral Patterns Report (TTPs)

**File**: `behavioral_patterns_report.csv`

## Purpose

This report identifies the most common "Tactics, Techniques, and Procedures" (TTPs) found in your dataset. It does this by analyzing sequences of actions rather than just individual transactions. This helps to understand the common *strategies* used by actors in your dataset.

## How to Use

Review the top patterns (sorted by `support`) to understand the prevalent strategies. This can help in building detection rules for new, incoming transactions or in profiling the general sophistication of the actors in the dataset.

## Key Columns

- `pattern_type`: The dimension of the pattern being analyzed (e.g., `value_patterns`, `method_patterns`, `direction_patterns`).
- `sequence`: The actual pattern of events. This should be interpreted in the context of the `pattern_type`.
- `support`: The frequency of this pattern in the dataset (a higher number means it's more common).
- `length`: The number of events in the sequence.

## Example Interpretation

| pattern_type | sequence | support |
| :--- | :--- | :--- |
| `value_patterns` | `('small', 'small', 'large')` | 0.15 |
| `direction_patterns` | `('in', 'out', 'out')` | 0.12 |

**Interpretation**:
- **Row 1**: 15% of the analyzed sequences showed a "structuring" value pattern: two small-value transactions followed by one large-value one. This is a classic money laundering TTP.
- **Row 2**: 12% of the sequences showed a "dispersal" direction pattern: one incoming transaction followed by two outgoing ones. This is often used to split funds.
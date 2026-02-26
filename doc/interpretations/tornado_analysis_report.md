# Tornado Analysis Report

**File**: `tornado_analysis_report.csv`

## Purpose

This report provides a detailed breakdown of every address that interacted with Tornado Cash. It goes beyond a simple flag and details the *nature* of the interaction.

## How to Use

Use this report to profile and prioritize addresses that use Tornado Cash. Focus on addresses with high volume and multiple `risk_indicators`.

## Key Columns

- `address`, `cluster_id`: Identifiers for the entity.
- `deposit_count`, `withdrawal_count`: The number of deposit and withdrawal transactions.
- `total_volume_eth`: The total ETH value moved through the mixer by this address.
- `interaction_patterns`: A JSON array of detected behavioral patterns (e.g., `rapid_tornado_interactions`, `regular_timing_intervals`).
- `risk_indicators`: A JSON array of specific risk flags derived from the interaction patterns (e.g., `large_volume_mixing`, `quick_turnaround`).

## Example Interpretation

An address with a high `total_volume_eth`, a `deposit_count` of 50, a `withdrawal_count` of 2, and a `risk_indicator` of `large_volume_mixing` is a much higher priority for investigation than an address with a single 0.1 ETH deposit. The former is clearly using the mixer for large-scale, strategic purposes.
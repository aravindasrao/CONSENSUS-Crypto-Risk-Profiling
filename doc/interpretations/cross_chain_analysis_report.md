# Cross-Chain Analysis Report

**File**: `cross_chain_analysis_report.csv`

## Purpose

This report details the findings of the `CrossChainAnalyzer`. It identifies addresses that are active across multiple blockchains and flags those that use cross-chain bridges, potentially to obfuscate the flow of funds.

## How to Use

Use this report to trace activity that leaves the primary blockchain being analyzed. An address with a high `risk_score` in this report is likely using bridges as part of a money laundering scheme.

## Key Columns

- `address`: The address engaging in cross-chain activity.
- `cluster_id`: The cluster this address belongs to.
- `risk_score`: A score indicating the likelihood that the cross-chain activity is suspicious.
- `bridge_transactions`: The number of transactions involving known bridge contracts.
- `bridge_interaction_ratio`: The proportion of the address's total activity that involves bridges.
- `bridge_patterns`: A JSON array of specific patterns detected (e.g., `high_bridge_activity`).
- `cross_chain_indicators`: A JSON array of indicators like `wrapped_token_interaction`.

## Example Interpretation

An address with a high `bridge_interaction_ratio` and patterns like `rapid_bridge_transactions` is a strong candidate for an actor attempting to move funds between chains to break the audit trail.
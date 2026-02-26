# Foundation Layer: 111 Behavioral Features

This document outlines the 111 behavioral features extracted by the `FoundationLayer` module. These features are designed to provide a comprehensive, multi-faceted view of an address's on-chain behavior, serving as the basis for clustering, risk scoring, and other advanced analyses.

The features are organized into seven distinct categories.

## 1. Temporal Features (16)

These features analyze the timing and frequency of an address's transactions.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 1 | `tx_frequency_daily` | `Total Transactions / Activity Span (days)` | Measures the average number of transactions per day. High frequency can indicate automated activity. |
| 2 | `tx_frequency_weekly` | `Total Transactions / (Activity Span (days) / 7)` | Smooths out daily fluctuations to provide a weekly activity rate. |
| 3 | `active_days_count` | `Count of unique days with at least one transaction.` | Indicates how consistently an address is used over time. Sporadic activity might be suspicious. |
| 4 | `active_hours_count` | `Count of unique hours of the day with transactions.` | Measures the breadth of activity throughout a 24-hour cycle. A low count might indicate scheduled tasks. |
| 5 | `peak_activity_hour` | `The hour of the day with the most transactions (mode).` | Identifies the most common time for transactions, which can help profile the user's timezone or routine. |
| 6 | `weekend_activity_ratio` | `(Txs on Saturday/Sunday) / Total Txs` | Measures the proportion of activity on weekends. High weekend activity might differ from typical institutional behavior. |
| 7 | `night_activity_ratio` | `(Txs between 10 PM and 6 AM) / Total Txs` | Activity during typical sleeping hours can be an indicator of automation or users in different timezones. |
| 8 | `burst_activity_count` | `Count of transactions within 1 hour of the previous one.` | Identifies periods of rapid, successive transactions, a sign of scripted actions or fund dispersal. |
| 9 | `temporal_regularity_score` | `1 / (1 + StdDev(Time Gaps))` | A score approaching 1 indicates very consistent timing between transactions, a strong signal for automation. |
| 10 | `avg_time_gap_hours` | `Average time (hours) between consecutive transactions.` | Measures the typical "cooldown" period for an address. |
| 11 | `time_gap_variance` | `Variance of time (hours) between consecutive transactions.` | High variance suggests irregular, human-like activity; low variance points to automation. |
| 12 | `max_inactive_period_days` | `Longest duration (days) between any two consecutive transactions.` | Identifies significant periods of dormancy, relevant for identifying long-term holding patterns. |
| 13 | `activity_span_days` | `Time between the first and last transaction (days).` | Measures the total lifetime of the address's activity. |
| 14 | `temporal_entropy` | `Shannon entropy of the hourly transaction distribution.` | Measures the unpredictability of transaction timing. High entropy means activity is spread out randomly. |
| 15 | `circadian_rhythm_strength` | `1 / (1 + Var(Hourly Tx Counts) / Total Txs)` | Measures how strongly activity follows a daily (24-hour) pattern. A high score indicates a strong daily routine. |
| 16 | `temporal_clustering_score` | `Var(Txs per Hour) / Mean(Txs per Hour)` | Measures how "clumped" or "bursty" transactions are. A high score indicates concentrated bursts of activity. |

## 2. Economic Features (18)

These features focus on the monetary value and flow of transactions.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 17 | `total_volume_eth` | `Sum of all transaction values (in ETH).` | Overall economic significance of the address. |
| 18 | `incoming_volume_eth` | `Sum of values from all incoming transactions.` | Total value received by the address. |
| 19 | `outgoing_volume_eth` | `Sum of values from all outgoing transactions.` | Total value sent by the address. |
| 20 | `net_flow_eth` | `Incoming Volume - Outgoing Volume` | Indicates whether the address is a net accumulator or disperser of funds. |
| 21 | `avg_transaction_value` | `Mean value of all transactions.` | Typical transaction size. |
| 22 | `median_transaction_value` | `Median value of all transactions.` | A robust measure of typical transaction size, less affected by outliers. |
| 23 | `max_single_transaction` | `Maximum value of any single transaction.` | Identifies significant, potentially anomalous, large transfers. |
| 24 | `min_single_transaction` | `Minimum non-zero value of any single transaction.` | Can indicate "dusting" or testing transactions. |
| 25 | `value_variance` | `Variance of transaction values.` | Measures the spread of transaction amounts. |
| 26 | `value_std_deviation` | `Standard deviation of transaction values.` | Another measure of value spread. |
| 27 | `coefficient_of_variation` | `StdDev(Values) / Mean(Value)` | Measures relative variability, independent of the scale of transaction values. |
| 28 | `large_tx_ratio` | `Proportion of transactions with value >= 75th percentile.` | Indicates a tendency to make large transactions relative to its own history. |
| 29 | `small_tx_ratio` | `Proportion of transactions with value <= 25th percentile.` | Indicates a tendency to make small transactions. |
| 30 | `round_number_ratio` | `Proportion of transactions with "round" values (e.g., 1, 10 ETH).` | Can be an indicator of manual human behavior or specific mixer usage. |
| 31 | `gini_coefficient` | `Gini coefficient of transaction values.` | Measures value inequality. A score near 1 means a few large transactions dominate the total volume. |
| 32 | `wealth_accumulation_rate` | `Net Flow / Activity Span (days)` | The average daily rate of change in the address's wealth. |
| 33 | `value_distribution_skewness` | `Skewness of the transaction value distribution.` | Measures the asymmetry of transaction values. |
| 34 | `economic_diversity_score` | `Shannon entropy of transaction values binned into 10 categories.` | High entropy indicates a wide and varied range of transaction values. |

## 3. Network Features (15)

These features analyze the address's position and connectivity within the larger transaction graph.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 35 | `degree_centrality` | `(In-Degree + Out-Degree) / (Total Nodes - 1)` | Basic measure of an address's connectivity. |
| 36 | `in_degree_centrality` | `In-Degree / (Total Nodes - 1)` | Measures how many addresses send funds to this address. |
| 37 | `out_degree_centrality` | `Out-Degree / (Total Nodes - 1)` | Measures how many addresses this address sends funds to. |
| 38 | `betweenness_centrality` | `Fraction of shortest paths between other nodes that pass through this node.` | A high score indicates the address acts as a bridge or intermediary. |
| 39 | `closeness_centrality` | `Reciprocal of the average shortest path distance to all other nodes.` | A high score means the address is "close" to all other nodes in the network. |
| 40 | `eigenvector_centrality` | `Measures influence by being connected to other influential nodes.` | Identifies nodes connected to other important nodes. |
| 41 | `pagerank_score` | `Google's PageRank algorithm applied to the transaction graph.` | Measures importance based on the quantity and quality of incoming links. |
| 42 | `clustering_coefficient` | `How connected a node's neighbors are to each other.` | A high score indicates the address is part of a tightly-knit community. |
| 43 | `unique_counterparties` | `Count of unique addresses this address has transacted with.` | A simple measure of network breadth. |
| 44 | `network_reach_2hop` | `Number of unique nodes reachable within two hops.` | Measures the address's extended network influence. |
| 45 | `hub_score` | `HITS algorithm score for being a good "hub" (points to many authorities).` | Identifies addresses that act as distributors or directories. |
| 46 | `authority_score` | `HITS algorithm score for being a good "authority" (pointed to by many hubs).` | Identifies addresses that are recipients of funds from many sources. |
| 47 | `local_efficiency` | `Average efficiency of the local subgraph around the node.` | Measures how well-connected the node's immediate neighborhood is. |
| 48 | `bridge_score` | `Proportion of local edges that are bridges (their removal would disconnect the local component).` | High score indicates the node is critical for local connectivity. |
| 49 | `network_influence_score` | `Mean(Degree, Betweenness, PageRank)` | A composite score for an address's overall network influence. |

## 4. Behavioral Features (20)

These features capture specific on-chain behaviors and patterns.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 50 | `total_transaction_count` | `Total number of transactions.` | Basic measure of activity. |
| 51 | `incoming_tx_count` | `Number of incoming transactions.` | Number of times the address received funds. |
| 52 | `outgoing_tx_count` | `Number of outgoing transactions.` | Number of times the address sent funds. |
| 53 | `tx_direction_ratio` | `Outgoing Txs / Incoming Txs` | Ratio of sending to receiving; indicates role as source, sink, or intermediary. |
| 54 | `self_transaction_ratio` | `Proportion of transactions sent from the address to itself.` | Can be used for contract state changes or other specific purposes. |
| 55 | `zero_value_tx_ratio` | `Proportion of transactions with a value of zero.` | Often used for contract interactions, airdrop claims, or spam. |
| 56 | `contract_interaction_ratio` | `Proportion of transactions that are contract calls.` | Measures how much the address interacts with smart contracts vs. other EOAs. |
| 57 | `unique_methods_count` | `Number of unique smart contract methods called.` | Indicates the diversity of contract interactions. |
| 58 | `method_diversity_entropy` | `Shannon entropy of the distribution of called method names.` | High entropy means the address calls a wide variety of different functions. |
| 59 | `repeat_counterparty_ratio` | `1 - (Unique Counterparties / Total Txs)` | High ratio means the address frequently transacts with the same few counterparties. |
| 60 | `avg_gas_per_tx` | `Average gas used per transaction.` | Can indicate the complexity of typical transactions. |
| 61 | `gas_usage_variance` | `Variance of gas used across transactions.` | High variance may indicate a wide range of transaction complexities. |
| 62 | `gas_price_variance` | `Variance of gas price paid across transactions.` | High variance may indicate sensitivity to network congestion or attempts at timing. |
| 63 | `gas_optimization_score` | `max(0, -Correlation(Time, Gas Used))` | Measures if gas usage decreases over time, suggesting learning or optimization. |
| 64 | `automation_likelihood` | `Mean(Timing Regularity, Value Consistency, Round Number Preference)` | A composite score to estimate if behavior is automated (bot-like). |
| 65 | `interaction_complexity` | `Mean(Method Diversity, Counterparty Diversity)` | A composite score for the complexity of an address's interactions. |
| 66 | `behavioral_consistency` | `Mean(Value Consistency, Gas Consistency, Timing Consistency)` | A composite score measuring how predictable an address's behavior is. |
| 67 | `pattern_entropy` | `Shannon entropy of transaction value patterns (binned).` | Measures the randomness of transaction value patterns. |
| 68 | `counterparty_loyalty` | `Proportion of transactions involving the single most frequent counterparty.` | High loyalty suggests a strong relationship with another entity. |
| 69 | `operational_sophistication` | `Mean(Gas Optimization, Method Diversity, 1 - Round Number Ratio)` | A composite score for how technically sophisticated an address's behavior is. |

## 5. Risk Features (12)

These features are specifically engineered to flag potentially suspicious or risky behaviors.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 70 | `unusual_timing_score` | `Proportion of transactions in "unusual" hours (e.g., 2-6 AM).` | Activity at odd hours can be a flag for automation or illicit activity. |
| 71 | `high_frequency_burst_score` | `Proportion of transactions occurring within 5 minutes of the previous one.` | Rapid bursts of activity can be used to quickly move or obfuscate funds. |
| 72 | `round_amount_preference` | `Proportion of transactions with "round" values.` | While it can indicate human behavior, it's also a hallmark of certain mixers. |
| 73 | `gas_price_anomaly_score` | `Proportion of transactions with gas price > 3 StdDev from the median.` | Unusually high or low gas prices can be used for transaction ordering or side-channel attacks. |
| 74 | `volume_spike_indicator` | `Proportion of transactions with value > 2 StdDev from the mean.` | Sudden, large transactions that deviate from normal behavior. |
| 75 | `mixing_behavior_score` | `Unique Counterparties / Total Txs` | A high score (approaching 1) indicates the address transacts with many different entities, a pattern seen in mixers. |
| 76 | `privacy_seeking_score` | `Proportion of transactions with small, round values (e.g., 0.1, 1.0 ETH).` | Use of common mixer denominations is a strong indicator of privacy-seeking behavior. |
| 77 | `evasion_pattern_score` | `Proportion of "in-out" transaction pairs within a short time window (e.g., 1 hour).` | A common pattern for passing funds through an intermediary address quickly. |
| 78 | `laundering_risk_indicator` | `Mean(Mixing Score, Evasion Score, Privacy Score)` | A composite score specifically targeting patterns associated with money laundering. |
| 79 | `anonymity_behavior_score` | `Mean(Unusual Timing, Privacy Score, Round Amount Preference)` | A composite score for behaviors aimed at increasing anonymity. |
| 80 | `suspicious_timing_pattern` | `Mean(Unusual Timing, High Freq Burst Score)` | A composite score for suspicious temporal patterns. |
| 81 | `composite_risk_score` | `Mean(Laundering Risk, Anonymity Score, Suspicious Timing, Volume Spike)` | The primary output of the risk feature category, summarizing overall risk. |

## 6. Operational Features (15)

These features analyze the technical aspects of how an address operates on the blockchain.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 82 | `avg_gas_limit` | `Average gas limit set for transactions.` | The maximum gas an address is willing to spend. |
| 83 | `avg_gas_used` | `Average gas actually consumed by transactions.` | Reflects the complexity of the operations performed. |
| 84 | `gas_efficiency_ratio` | `1 / (1 + CoeffOfVariation(Gas Used))` | High score indicates consistent gas usage, possibly for repeated, similar operations. |
| 85 | `gas_price_strategy` | `1 / (1 + CoeffOfVariation(Gas Price))` | High score indicates a consistent strategy for setting gas prices. |
| 86 | `contract_deployment_count` | `Number of transactions where method is 'constructor'.` | Identifies addresses that create smart contracts. |
| 87 | `contract_call_frequency` | `Proportion of transactions that are contract calls.` | Measures reliance on smart contracts. |
| 88 | `advanced_function_usage` | `Number of unique function names called.` | Indicates interaction with a diverse set of smart contract functionalities. |
| 89 | `operational_complexity` | `min(1, (Unique Methods + Unique Functions) / 20)` | A composite score for the complexity of contract interactions. |
| 90 | `gas_optimization_trend` | `max(0, -Correlation(Time, Gas Used))` | A positive score suggests the user is becoming more efficient with gas over time. |
| 91 | `block_timing_consistency` | `1 / (1 + CoeffOfVariation(Block Gaps))` | Measures the regularity of transactions in terms of block numbers. |
| 92 | `priority_fee_behavior` | `Proportion of transactions with gas price >= 80th percentile.` | Indicates a willingness to pay high fees for faster transaction inclusion. |
| 93 | `batch_transaction_score` | `Proportion of transactions occurring within 1 minute of the previous one.` | A strong indicator of scripted or batched operations. |
| 94 | `mev_resistance_score` | `Mean(Gas Price Strategy, Block Timing Consistency)` | A proxy for how an address might be attempting to mitigate Miner Extractable Value (MEV). |
| 95 | `technical_sophistication` | `Mean(Op. Complexity, Adv. Function Usage, Gas Opt., MEV Resistance)` | A high-level score for the technical prowess of the address operator. |
| 96 | `infrastructure_usage` | `min(1, (Contract Call Freq + Batch Tx Score + Priority Fee Behavior) / 3)` | A score indicating the use of advanced infrastructure (scripts, bots, etc.). |

## 7. Contextual Features (15)

These features place the address's activity within the broader context of the market and network environment.

| # | Feature Name | Formula / Description | Rationale |
|---|---|---|---|
| 97 | `market_timing_correlation` | `abs(Correlation(Time, Daily Volume))` | Measures if the address's activity trends with the market (e.g., increasing volume over time). |
| 98 | `network_congestion_behavior` | `Proportion of transactions with gas price >= 70th percentile.` | Indicates how an address behaves during periods of high network fees. |
| 99 | `peak_hours_preference` | `Proportion of transactions during typical business hours (e.g., 9am-5pm).` | Can help distinguish between institutional and retail users. |
| 100 | `off_peak_activity` | `Proportion of transactions during off-peak hours (e.g., 10pm-6am).` | Activity during quiet network times. |
| 101 | `weekday_vs_weekend_ratio` | `Weekday Txs / Weekend Txs` | Compares activity levels during the work week versus the weekend. |
| 102 | `market_volatility_response` | `StdDev(Daily Tx Count) / Mean(Daily Tx Count)` | Measures how much the address's activity fluctuates, possibly in response to market volatility. |
| 103 | `seasonal_activity_pattern` | `Var(Monthly Tx Count) / Mean(Monthly Tx Count)` | Identifies if there are significant variations in activity across different months. |
| 104 | `economic_event_sensitivity` | `Proportion of days with transaction count > 2 StdDev from the mean.` | Measures how often the address has extreme activity spikes, possibly tied to external events. |
| 105 | `bear_market_behavior` | `Proportion of days with transaction count <= 30th percentile.` | Indicates the level of activity during periods of low personal activity (proxy for bear markets). |
| 106 | `bull_market_behavior` | `Proportion of days with transaction count >= 70th percentile.` | Indicates the level of activity during periods of high personal activity (proxy for bull markets). |
| 107 | `crisis_period_activity` | `Proportion of days with a daily change in tx count > 90th percentile.` | Measures activity during periods of extreme change, which could correspond to market crises. |
| 108 | `fee_market_adaptation` | `Same as network_congestion_behavior.` | Measures how the address adapts its fee strategy to the broader market. |
| 109 | `ecosystem_participation` | `Mean(Peak Hours Preference, Congestion Behavior, Activity Level)` | A composite score for how engaged the address is with the "normal" ecosystem. |
| 110 | `institutional_timing` | `Proportion of transactions during Mon-Fri, 9am-5pm.` | A score indicating behavior consistent with institutional or business operations. |
| 111 | `retail_behavior_score` | `1 - Institutional Timing` | A score indicating behavior consistent with retail users (e.g., evenings, weekends). |
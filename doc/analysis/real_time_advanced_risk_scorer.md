# Real-Time Advanced Risk Scorer

This document outlines the logic and architecture of the `AdvancedRealTimeRiskScorer` module. This module is designed to provide real-time, dynamic risk assessments for incoming transactions using a suite of advanced machine learning models and behavioral analysis techniques.

## Overview

The primary goal of this module is to move beyond static, batch analysis and score transactions as they occur. It does this by leveraging the comprehensive 111 features from the `FoundationLayer` and applying a multi-model ensemble approach to generate a highly accurate, context-aware risk score. This module simulates a real-time environment by processing a stream of recent transactions from the database.

## Processing Logic

The module operates in two main stages: Initialization and Real-Time Processing.

### 1. Model Initialization

Before real-time processing can begin, the scorer initializes and trains its machine learning models.

1.  **Prepare Training Data**: The module queries the database for addresses with existing "foundation" risk scores. It uses the 111 features as input (`X`) and derives a binary label (`y`) from the risk score (e.g., score > 0.5 is high risk).
2.  **Train Ensemble Models**: It trains a suite of powerful classifiers on the prepared data:
    *   **Random Forest**: An ensemble of decision trees, robust to overfitting.
    *   **Gradient Boosting**: Builds models sequentially, with each new model correcting errors of the previous one.
    *   **XGBoost**: A highly optimized and powerful gradient boosting implementation.
3.  **Train Anomaly Detectors**: It trains unsupervised models to detect outliers:
    *   **Isolation Forest**: Efficiently isolates anomalies by building random trees. It learns the structure of "normal" data and identifies points that are easier to separate.
4.  **Feature Scaling**: A `RobustScaler` is fitted on the training data to handle outliers and scale features for the models.

### 2. Real-Time Scoring Pipeline

Once initialized, the scorer processes a stream of transactions. For each transaction:

1.  **Feature Extraction**: It fetches the pre-calculated 111 features for the transaction's `from_addr` from the `features` table. If no features are found, it falls back to a simpler rule-based score.
2.  **Ensemble Scoring**: The features are scaled and passed to each trained ensemble model (`RandomForest`, `XGBoost`, etc.). Each model outputs a risk probability (0.0 to 1.0).
3.  **Anomaly Scoring**: The features are also scored by the `IsolationForest` model to determine how much the transaction's behavior deviates from the norm.
4.  **Behavioral Context Analysis**: The module maintains a short-term memory of recent transactions for each address. This allows it to detect patterns like rapid, sequential transactions.
5.  **Adaptive Composite Score**: The scores from all models are combined into a single `composite_risk_score`. The weights used in this combination are adaptive and can be influenced by the behavioral context (e.g., giving more weight to anomaly detectors during a rapid sequence).
6.  **Risk Assessment & Alerting**:
    *   The composite score is used to determine a final risk level (`CRITICAL`, `HIGH`, `MEDIUM`, etc.).
    *   Key contributing factors are identified.
    *   If the score exceeds an adaptive threshold, an alert is logged.
7.  **Database Storage**: The final risk assessment, including the composite score and evidence, is stored in the `risk_components` and `advanced_analysis_results` tables. This allows the `UnifiedRiskScorer` to incorporate this real-time insight into the final, overall risk score for the address.

## Key Models and Rationale

| Model | Type | Role in Pipeline | Rationale |
|:---:|---|---|---|
| **Random Forest** | Supervised Ensemble | **Primary Risk Classification**: Provides a robust baseline risk score based on learned patterns from historical data. | Excellent for high-dimensional data, less prone to overfitting, and provides feature importance insights. |
| **XGBoost** | Supervised Ensemble | **High-Performance Classification**: Offers a more aggressive and often more accurate risk score. | State-of-the-art performance, known for winning machine learning competitions. Captures complex, non-linear relationships. |
| **Isolation Forest**| Unsupervised Anomaly Detection | **Novelty/Outlier Detection**: Identifies transactions that are behaviorally different from the vast majority of "normal" transactions. | Highly efficient for detecting anomalies. It does not require labeled data and can find previously unseen types of suspicious behavior. |

## Output

The `AdvancedRealTimeRiskScorer` produces two main outputs in the database:

1.  **`risk_components` Table**: A new entry is created with `component_type = 'real_time'`. The `risk_score` is the `composite_risk_score` from the model ensemble, and the `evidence` JSON contains the full breakdown of scores from individual models.
2.  **`advanced_analysis_results` Table**: A detailed log of the entire risk assessment for the transaction is stored, providing full transparency for forensic investigation.

This real-time score is a crucial input for the `UnifiedRiskScorer`, which combines it with evidence from all other analysis modules to produce the final, authoritative risk score for an address.
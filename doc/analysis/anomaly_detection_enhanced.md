# Enhanced Anomaly Detector

This document outlines the logic and rules used by the `EnhancedAnomalyDetector` module. This module uses unsupervised machine learning techniques to identify addresses that exhibit anomalous behavior compared to the general population, without relying on predefined rules or labels.

## Overview

While other modules search for specific, known patterns of illicit activity, the `EnhancedAnomalyDetector` is designed to find novel or unusual behaviors that deviate from the norm. It answers the question: "Which addresses are behaving in a statistically unusual way?" This is critical for discovering new money laundering techniques and identifying sophisticated actors who actively try to avoid common detection patterns.

## Processing Logic

The module employs a multi-model approach to ensure robust detection:

1.  **Data Preparation**: The module fetches all addresses with a sufficient transaction history (e.g., >10 transactions).

2.  **Feature Extraction**: For each address, it retrieves the comprehensive 111-feature vector that has been pre-calculated by the `FoundationLayer` and stored in the `addresses` table.

3.  **Feature Scaling**: All features are standardized using `StandardScaler` to ensure that each feature contributes equally to the analysis, regardless of its original scale.

4.  **Multi-Model Anomaly Detection**: The module applies several unsupervised anomaly detection algorithms in parallel to the scaled feature set:
    *   **Isolation Forest**: An ensemble method that is highly effective at identifying outliers by "isolating" them in fewer splits than normal data points.
    *   **One-Class SVM**: A Support Vector Machine variant that learns a boundary around the "normal" data points and classifies anything outside this boundary as an anomaly.
    *   **Statistical Z-Score**: A statistical method that identifies anomalies on a per-feature basis. It flags any address where a specific feature value is a significant outlier (e.g., >3 standard deviations from the mean for that feature).

5.  **Scoring and Storage**:
    *   For each anomaly detected by a model, a normalized `anomaly_score` (0-1) and a `confidence` score are calculated.
    *   These findings, including the method of detection and the primary contributing feature (for Z-score), are stored in the `anomaly_detections` table.
    *   A summary of the analysis run is stored in the `anomaly_sessions` table for monitoring and auditing.

6.  **Consensus Building**: The results from all methods are combined to identify high-confidence anomalies that have been flagged by multiple, independent algorithms.

## Key Algorithms and Rationale

| Algorithm | Name | Description | Rationale |
|:---:|---|---|---|
| 1 | **Isolation Forest** | An ensemble of decision trees. Anomalies are easier to "isolate" from the rest of the data, so they are found in paths that are closer to the root of the trees. | Fast, memory-efficient, and performs well on high-dimensional data. It does not require assumptions about the data distribution. |
| 2 | **One-Class SVM** | A Support Vector Machine algorithm that is trained on "normal" data to learn a boundary (hypersphere) that encloses it. Points outside this boundary are considered anomalies. | Effective at finding global outliers and can handle non-linear relationships in the data by using different kernels (e.g., RBF). |
| 3 | **Z-Score Analysis** | A statistical method that measures how many standard deviations a data point is from the mean of its feature. A high absolute Z-score (e.g., >3) indicates an outlier. | Simple, interpretable, and effective at finding extreme outliers in specific, individual features (e.g., an address with an exceptionally high transaction count). |
| 4 | **Ensemble Scoring** | The process of combining the outputs of multiple models. An address flagged by both Isolation Forest and One-Class SVM is a higher-confidence anomaly than one flagged by only one. | This approach reduces false positives and increases confidence in the detected anomalies. Different models capture different types of anomalies, making the combined result more robust. |

## Role in the Pipeline

The `EnhancedAnomalyDetector` serves as a safety net for the entire system. It is a discovery-oriented module that complements the rule-based and pattern-matching modules.

-   It identifies **novel and emerging threats** that do not yet have a defined signature.
-   It flags sophisticated actors who may be aware of common laundering patterns and are actively trying to appear "normal" but still exhibit subtle statistical irregularities.
-   The anomalies it detects are stored as `ANOMALY_DETECTED` risk components, which are fed into the `UnifiedRiskScorer` to contribute to an address's final risk score. An address flagged by multiple anomaly models receives a significantly higher risk contribution.
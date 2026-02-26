# Causal Inference Engine

This document outlines the logic and methodology used by the `CausalInferenceEngine` module. This module brings academic rigor to the analysis by moving beyond correlation to investigate the *causal* drivers of on-chain risk. It is a comprehensive, multi-stage analysis component designed to provide deep, defensible insights.

## Overview

While other modules identify *what* is risky, the `CausalInferenceEngine` aims to understand *why* it is risky. It uses formal causal inference methods to answer questions like: "What is the actual causal effect of interacting with a mixer on an address's final risk score, after accounting for other confounding factors?" This provides deeper, more defensible insights into risk.

## Comprehensive Analysis Pipeline

The module follows a structured, multi-phase causal analysis process, orchestrated by the `run_comprehensive_causal_analysis` method.

1.  **Data Preparation**: The engine prepares a dataset suitable for causal analysis, typically by sampling from the `addresses` table which contains the rich feature set from the `FoundationLayer`.

2.  **Causal Graph Construction**: A causal graph is constructed based on domain knowledge and statistical tests. This graph represents the assumed causal relationships between variables.

3.  **Causal Discovery**: The engine uses constraint-based algorithms to discover potential causal relationships directly from the data.
    *   **Method**: It leverages the **PC (Peter-Clark) algorithm** from the `causallearn` library.
    *   **Process**: The algorithm performs a series of conditional independence tests (using `fisherz` for continuous data) to identify a causal graph skeleton. This data-driven approach helps uncover non-obvious relationships.
    *   **Preprocessing**: To ensure statistical validity, the data is heavily preprocessed before running the PC algorithm: numeric features are selected, constant columns are removed, and highly correlated features are dropped to avoid multicollinearity.

4.  **Treatment Effect Estimation**: The engine estimates the causal effect of a "treatment" variable on an "outcome" variable.
    *   **Method**: It uses the `dowhy` library to perform a formal, four-step causal inference process: model, identify, estimate, and refute.
    *   **Estimation**: The primary estimation method implemented is `backdoor.linear_regression`, which adjusts for a specified set of confounding variables.
    *   **Refutation**: The estimate is validated using a refutation test, such as adding a `random_common_cause`, to check if the estimate changes significantly (a robust estimate should not).

5.  **Mediation, Counterfactuals, and Robustness**: The pipeline includes dedicated phases for mediation analysis, counterfactual "what-if" scenarios, and comprehensive robustness testing (e.g., bootstrapping, placebo tests). *Note: These are advanced components and are currently implemented as placeholders in the code.*

6.  **Database Storage**: A summary of the analysis, including key findings and the overall robustness score, is stored in the `advanced_analysis_results` table for expert review and model validation.

## Key Causal Questions Investigated

| Question | Treatment | Outcome | Rationale |
|:---:|---|---|---|
| 1 | **Effect of Mixer Usage** | What is the causal impact of making at least one deposit into Tornado Cash on an address's final risk score? | This quantifies the exact risk contribution of using a mixer, separating it from confounding factors (e.g., high-volume addresses might be more likely to use mixers anyway). |
| 2 | **Effect of Sanctioned Interaction** | What is the causal impact of receiving funds from a sanctioned entity on an address's risk score? | This measures the "taint" effect, providing a defensible metric for how much risk is transferred through direct interaction, adjusted for other behaviors. |
| 3 | **Effect of Structuring Behavior** | Does engaging in "structuring" (many small inputs) *cause* a higher risk score, even for addresses that don't use mixers? | This helps determine if certain behavioral patterns are inherently risky on their own, or only when combined with other activities. |
| 4 | **Effect of Automation** | What is the causal impact of having a high `automation_likelihood` score on the final risk score? | This validates whether the features designed to detect bots are truly identifying a behavior that leads to higher risk, independent of other factors. |

## Role in the Pipeline

The `CausalInferenceEngine` is not a primary risk-scoring tool for individual addresses. Instead, it serves a higher-level strategic purpose:

- **Model Validation**: It provides a rigorous, academic validation of the risk factors used throughout the entire system. If a feature has a strong causal link to risk, it justifies its high weight in the `UnifiedRiskScorer`.
- **Deep Explainability (XAI)**: It delivers powerful, human-understandable explanations for *why* the system flags certain behaviors as high-risk, moving from correlation to causation.
- **Strategic Insight**: It helps analysts understand the fundamental drivers of risk in the ecosystem, which can inform future feature engineering and model development.
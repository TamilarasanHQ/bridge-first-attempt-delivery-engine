# BRIDGE v2 Phase 3 — Baseline Modeling & Rare-Event Evaluation Report

**Execution Timestamp:** 2026-09-29 05:48:20  
**Environment:** Python 3.13.3 | Scikit-Learn 1.9.1 | PyArrow Engine  
**Dataset Lineage:** Canonical `data/v2/bridge_v2_dataset.parquet` via Phase 2 chronological splits.  
**Phase Objective:** Establish empirical, leakage-safe baseline models for the observed real-world delivery failure target ($Y \approx 0.75\%$).

---

## 1. Executive Summary

In Phase 3, we developed and evaluated baseline models strictly calibrated for the severe class imbalance of real-world logistics telemetry. Ordinary classification accuracy was rejected in favor of **Precision-Recall AUC (Average Precision)**, Brier score, and **Top-k Operational Lift** metrics.

### Key Benchmark Findings:
1. **Prevalence Reference Baseline:** The naive prior predicts $\hat{P} = 0.7784\%$, yielding a validation PR-AUC of **0.007692** and ROC-AUC of **0.5000**.
2. **Linear Baseline (Logistic Regression):** Regularized logistic regression achieves a validation PR-AUC of **0.024777** ($3.22\times$ over baseline) and ROC-AUC of **0.7359**.
3. **Primary Baseline (HistGradientBoosting):** The non-linear histogram gradient boosting model achieves a validation PR-AUC of **0.043989** (**$5.72\times$ over baseline**), ROC-AUC of **0.7904**, and Brier score of **0.007513**.
4. **Top-k Operational Power:**
   - On the **Top 1%** highest-risk packages, HistGradientBoosting achieves a precision of **7.76%** (**$10.1\times$ lift** over random dispatch).
   - On the **Top 10%** highest-risk packages, the model flags **43.25% of all delivery failures** (750 of 1,734 validation failures) with a precision of 3.33% ($4.3\times$ lift).
5. **Locked Out-of-Time Test Generalization:** Evaluated strictly once on the future test set (`2018-08-16` to `2018-08-26`), the locked model maintains strong discrimination: **PR-AUC = 0.030381** ($4.99\times$ over test prevalence $0.6084\%$) and **ROC-AUC = 0.7779**, capturing **43.67% of all test failures** in the top 10% of flagged deliveries.

---

## 2. Dataset & Chronological Split Overview

| Split | Date Range | Total Packages | Failures ($Y=1$) | Successes ($Y=0$) | Observed Failure Prevalence |
| :--- | :--- | ---: | ---: | ---: | ---: |
| **TRAIN** | 2018-07-19 to 2018-08-11 (24 days) | 1,005,638 | 7,828 | 997,810 | **0.7784%** (1 in 128) |
| **VALIDATION** | 2018-08-12 to 2018-08-15 (4 days) | 225,416 | 1,734 | 223,682 | **0.7692%** (1 in 130) |
| **TEST** | 2018-08-16 to 2018-08-26 (11 days) | 224,689 | 1,367 | 223,322 | **0.6084%** (1 in 164) |
| **TOTAL** | 2018-07-19 to 2018-08-26 (39 days) | 1,455,743 | 10,929 | 1,444,814 | **0.7508%** (1 in 133) |

---

## 3. Preprocessing & Feature Engineering Approach

All feature transformers were fitted strictly on **TRAIN ONLY** and transformed validation and test identically:
- **Numerical Features (17):** Scaled via `StandardScaler()`. Structurally missing values in `time_window_duration_hours` (for non-window packages) were filled with `0.0`, reflecting zero promised duration without leakage.
- **Categorical Features (2):** `station_code` (17 hubs) and `day_of_week` (7 days) encoded via `OneHotEncoder(drop='first', handle_unknown='ignore')` for linear models, and mapped as categorical dtypes for tree models.
- **Forbidden Fields:** `scan_status`, `actual_sequences`, `seq_idx`, `route_score`, and all 13 v1 synthetic customer features were confirmed completely absent.

---

## 4. Validation Comparison & Baseline Benchmark

Evaluated on **225,416 validation packages** (Validation failure prevalence = **0.7692%**):

| Model | PR-AUC | ROC-AUC | Brier Score | Log Loss | P@1% | R@1% | P@5% | R@5% | P@10% | R@10% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prevalence Baseline** | 0.007692 | 0.5000 | 0.007633 | 0.0451 | 0.58% | 0.75% | 1.28% | 8.30% | 1.00% | 13.03% |
| **Logistic Regression (Unweighted)** | 0.024777 | 0.7359 | 0.007639 | 0.0424 | 4.92% | 6.40% | 3.17% | 20.59% | 2.59% | 33.62% |
| **Logistic Regression (Balanced)** | 0.026236 | 0.7622 | 0.185516 | 0.5752 | 4.48% | 5.82% | 3.65% | 23.70% | 2.95% | 38.29% |
| **HistGradientBoosting (Natural)** | **0.043989** | **0.7904** | **0.007513** | **0.0404** | **7.76%** | **10.09%** | **4.26%** | **27.68%** | **3.33%** | **43.25%** |
| **HistGradientBoosting (Balanced)** | 0.039564 | 0.7817 | 0.133742 | 0.4192 | 7.28% | 9.46% | 4.26% | 27.68% | 3.34% | 43.37% |

### Observations on Weighting:
- **Unweighted vs. Balanced Logistic Regression:** `class_weight='balanced'` moderately improves ROC-AUC (0.7622 vs. 0.7359), but severely distorts raw probability calibration (Brier score deteriorates from 0.0076 to 0.1855, LogLoss jumps from 0.042 to 0.575) because it forces the intercept to predict an artificial 50% prior.
- **HistGradientBoosting (Natural):** Outperforms all models across PR-AUC, ROC-AUC, and calibration. It preserves true Bayesian probabilities while maximizing rank ordering.

---

## 5. Model Selection & Configuration Lock

**Locked Model:** `HistGradientBoosting (Natural)`  
**Hyperparameter Configuration:**
```python
HistGradientBoostingClassifier(
    max_iter=150,
    learning_rate=0.08,
    max_depth=7,
    min_samples_leaf=50,
    random_state=42,
    class_weight=None,
    categorical_features=['station_code', 'day_of_week']
)
```
- Saved to: `models/v2/tree_baseline.joblib`
- Linear baseline preserved at: `models/v2/logistic_regression_baseline.joblib`

---

## 6. Probability Calibration & Decile Risk Analysis

Validation decile analysis for the locked `HistGradientBoosting (Natural)` model demonstrates monotonic calibration:

| Decile Tier | Package Count | Predicted Risk Range | Mean Predicted Risk | Observed Failures | Observed Failure Rate | Operational Lift |
| :--- | ---: | :---: | :---: | ---: | :---: | :---: |
| Decile 0 | 22,542 | 0.016% - 0.083% | 0.059% | 15 | 0.067% | **0.09x** |
| Decile 1 | 22,542 | 0.083% - 0.126% | 0.104% | 29 | 0.129% | **0.17x** |
| Decile 2 | 22,541 | 0.126% - 0.168% | 0.147% | 31 | 0.137% | **0.18x** |
| Decile 3 | 22,542 | 0.168% - 0.216% | 0.190% | 52 | 0.231% | **0.30x** |
| Decile 4 | 22,541 | 0.216% - 0.288% | 0.249% | 85 | 0.377% | **0.49x** |
| Decile 5 | 22,542 | 0.288% - 0.399% | 0.339% | 108 | 0.479% | **0.62x** |
| Decile 6 | 22,541 | 0.399% - 0.592% | 0.484% | 139 | 0.617% | **0.80x** |
| Decile 7 | 22,542 | 0.592% - 0.960% | 0.754% | 215 | 0.954% | **1.24x** |
| Decile 8 | 22,541 | 0.960% - 1.754% | 1.297% | 310 | 1.375% | **1.79x** |
| Decile 9 | 22,542 | 1.754% - 86.077% | 3.614% | 750 | 3.327% | **4.33x** |

### Validation Operating Points:
- **Top 1% Risk Cutoff:** Threshold $\tau = 0.057417$ (Flags 2,254 packages | Precision = 7.76% | Recall = 10.09%)
- **Top 5% Risk Cutoff:** Threshold $\tau = 0.027196$ (Flags 11,270 packages | Precision = 4.26% | Recall = 27.68%)
- **Top 10% Risk Cutoff:** Threshold $\tau = 0.017538$ (Flags 22,541 packages | Precision = 3.33% | Recall = 43.25%)

---

## 7. Final Locked Test Evaluation (Exactly One Run)

Evaluated on **224,689 out-of-time test packages** (Test failure prevalence = **0.6084%**, 1,367 failures):

| Model | PR-AUC | ROC-AUC | Brier Score | Log Loss | P@1% | R@1% | P@5% | R@5% | P@10% | R@10% |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Prevalence Baseline** | 0.006084 | 0.5000 | 0.006050 | 0.0373 | 0.53% | 0.88% | 0.47% | 3.88% | 0.64% | 10.53% |
| **Logistic Regression (Unweighted)** | 0.018409 | 0.7048 | 0.006069 | 0.0354 | 4.01% | 6.58% | 2.55% | 20.92% | 2.12% | 34.82% |
| **Locked HistGradientBoosting (Tree)** | **0.030381** | **0.7779** | **0.005998** | **0.0337** | **5.65%** | **9.29%** | **3.59%** | **29.48%** | **2.66%** | **43.67%** |

### Test Evaluation Observations:
1. **Generalization Integrity:** The model demonstrates robust generalization over time. PR-AUC is **0.030381** ($4.99\times$ test baseline), and ROC-AUC is **0.7779** (only a minor 0.0125 drop from validation 0.7904).
2. **Top-k Capture on Unseen Future Data:**
   - In the **Top 1%** highest-risk test packages (2,246 packages), the model identifies 127 failures (Precision = **5.65%**, **$9.3\times$ lift** over random).
   - In the **Top 5%** (11,234 packages), the model captures 403 failures (Precision = **3.59%**, **$5.9\times$ lift**, Recall = **29.48%**).
   - In the **Top 10%** (22,468 packages), the model captures 597 failures (Precision = **2.66%**, **$4.4\times$ lift**, Recall = **43.67%**).
3. **Context on Prevalence Shift:** As documented in Phase 2, late August experienced a geographic dispatch shift toward lower-failure hubs (e.g. Seattle DSE4/DSE5 and DLA9), explaining the lower baseline rate (0.6084% vs 0.7692%). The model's lift factors remain consistent ($10.1\times$ val vs. $9.3\times$ test at Top 1%; $4.3\times$ val vs. $4.4\times$ test at Top 10%).

---

## 8. Limitations & Scope

1. **Non-Causal Prediction:** Predicted risk scores reflect statistical associations with historical driver scans, not direct causal drivers of delivery failure.
2. **Absence of Customer Telemetry:** Because the raw Amazon dataset lacks recipient contact metadata (e.g. customer phone response, gate access codes, signature requirements), predictions rely strictly on parcel dimensions, planned service time, depot geography, and route loading.
3. **Rare-Event Constraint:** At 0.75% prevalence, standard classification metrics (e.g. accuracy, default 0.5 threshold F1) are uninformative; operational decision rules must be based on targeted top-k intervention capacity.

---

## 9. Next-Phase Recommendations (Phase 4: Risk Policy & Interventions)

1. **Cost-Sensitive Asymmetric Utility Matrix:** Calibrate optimal operational intervention thresholds using realistic logistics cost ratios ($C_{\text{failed attempt}} / C_{\text{preventative check}}$).
2. **Dispatch Capacity Tiering:** Implement multi-tier intervention routing (e.g. Top 1% = mandatory driver call-ahead / address verification; Top 5% = preferential early dispatch slot; Remaining 95% = standard workflow).
3. **Feature Engineering Expansion:** In Phase 4/5, evaluate geographic micro-clustering (K-Means spatial density or zone target encoding with cross-validation regularizers) to further extract signal from `zone_id` and coordinates.

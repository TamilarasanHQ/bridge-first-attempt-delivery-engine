# BRIDGE v2 Phase 5 — Model Improvement & Performance Optimization Report

**Execution Timestamp:** 2026-09-29 06:20:48  
**Governing Standard:** Hypothesis-Driven Controlled Experimentation. Zero test-set-driven iteration.

---

## 1. Executive Summary

Phase 5 investigated whether hypothesis-driven feature engineering and model family optimization could systematically improve upon the locked Phase 3 baseline (`HistGradientBoosting`, natural distribution).

All experiments were trained strictly on `data/v2/train.parquet` (1,005,638 deliveries) and selected strictly on `data/v2/validation.parquet` (225,416 deliveries). Exactly **one locked evaluation** was executed on the out-of-time test split (`data/v2/test.parquet`, 224,689 deliveries) after the winning model was frozen.

### Key Performance Summary

| Split / Scope | Metric | Phase 3 Locked Baseline | Phase 5 Candidate (`CAND-03`) | Absolute Change | Relative Change |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Validation** | **PR-AUC** | **0.043989** | **0.049303** | **+0.005314** | **+12.08%** |
| Validation | ROC-AUC | 0.7904 | 0.7868 | -0.0036 | - |
| Validation | Brier Score | 0.007513 | 0.007486 | -0.000027 | - |
| Validation | Precision @ 1% | 7.76% | 6.88% | -0.89% | - |
| Validation | Recall @ 1% | 10.09% | 8.94% | -1.15% | - |
| Validation | Precision @ 5% | 4.26% | 4.16% | -0.10% | - |
| Validation | Recall @ 5% | 27.68% | 27.05% | -0.63% | - |
| Validation | Precision @ 10% | 3.33% | 3.17% | -0.16% | - |
| Validation | Recall @ 10% | 43.25% | 41.18% | -2.08% | - |
| **Locked Test** | **PR-AUC** | **0.030381** | **0.032237** | **+0.001856** | **+6.11%** |
| Locked Test | ROC-AUC | 0.7779 | 0.7765 | -0.0014 | - |
| Locked Test | Brier Score | 0.005998 | 0.005989 | -0.000009 | - |
| Locked Test | Precision @ 1% | 5.65% | 6.10% | +0.45% | - |
| Locked Test | Recall @ 1% | 9.29% | 10.02% | +0.73% | - |
| Locked Test | Precision @ 5% | 3.59% | 3.53% | -0.05% | - |
| Locked Test | Recall @ 5% | 29.48% | 29.04% | -0.44% | - |
| Locked Test | Precision @ 10% | 2.66% | 2.63% | -0.02% | - |
| Locked Test | Recall @ 10% | 43.67% | 43.31% | -0.37% | - |

---

## 2. Controlled Experiment Registry

The complete registry of 14 controlled validation experiments is detailed below:

| ID | Model Family | Feature Set | Val PR-AUC | Val ROC | Val Brier | P@1% | R@1% | P@5% | R@5% | P@10% | R@10% | Monotonic? |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `CAND-00` | HistGradientBoosting | Baseline (19 features) | 0.043989 | 0.7904 | 0.007513 | 7.76% | 10.09% | 4.26% | 27.68% | 3.33% | 43.25% | Yes |
| `CAND-01` | LogisticRegression | Baseline (19 features) | 0.024777 | 0.7359 | 0.007639 | 4.92% | 6.40% | 3.17% | 20.59% | 2.59% | 33.62% | No |
| `CAND-02` | HistGradientBoosting | Baseline + Group A (Cyclic Time) | 0.047694 | 0.7877 | 0.007488 | 7.50% | 9.75% | 4.27% | 27.74% | 3.18% | 41.35% | No |
| `CAND-03` | HistGradientBoosting | Baseline + Group B (Route Ratios) | 0.049303 | 0.7868 | 0.007486 | 6.88% | 8.94% | 4.16% | 27.05% | 3.17% | 41.18% | Yes |
| `CAND-04` | HistGradientBoosting | Baseline + Group C (Interactions) | 0.036692 | 0.7845 | 0.007537 | 6.61% | 8.59% | 4.31% | 28.03% | 3.15% | 41.00% | No |
| `CAND-05` | HistGradientBoosting | Baseline + Group D (Spatial) | 0.045661 | 0.7943 | 0.007529 | 8.03% | 10.44% | 4.19% | 27.22% | 3.30% | 42.91% | Yes |
| `CAND-06` | HistGradientBoosting | Baseline WITHOUT Station Code | 0.036369 | 0.7904 | 0.007655 | 6.39% | 8.30% | 4.13% | 26.82% | 3.30% | 42.91% | No |
| `CAND-07` | HistGradientBoosting | Combined Engineered (B+C+D) | 0.034524 | 0.7818 | 0.007647 | 6.17% | 8.02% | 4.06% | 26.41% | 3.13% | 40.66% | No |
| `CAND-08` | LightGBM | Baseline (19 features) | 0.034089 | 0.7792 | 0.007637 | 6.92% | 9.00% | 3.95% | 25.66% | 3.12% | 40.60% | Yes |
| `CAND-09` | LightGBM | Combined Engineered (B+C+D) | 0.035462 | 0.7812 | 0.007724 | 5.99% | 7.79% | 3.92% | 25.49% | 3.13% | 40.72% | Yes |
| `CAND-10` | LightGBM | Combined Engineered (B+C+D) | 0.040828 | 0.7912 | 0.007510 | 8.25% | 10.73% | 4.35% | 28.26% | 3.27% | 42.50% | Yes |
| `CAND-11` | LightGBM | Combined Engineered (B+C+D) | 0.044727 | 0.7900 | 0.007496 | 7.90% | 10.27% | 4.33% | 28.14% | 3.27% | 42.56% | No |
| `CAND-12` | HistGradientBoosting | Combined Engineered (B+C+D) | 0.045797 | 0.7936 | 0.007493 | 8.03% | 10.44% | 4.29% | 27.91% | 3.29% | 42.79% | No |
| `CAND-13` | LightGBM + Sigmoid Calibration | Combined Engineered (B+C+D) | 0.040828 | 0.7912 | 0.007630 | 8.25% | 10.73% | 4.35% | 28.26% | 3.27% | 42.50% | Yes |

*Full tabular registry saved to:* [`reports/v2/phase5_experiment_results.csv`](file:///c:/Users/Ksari/OneDrive/Desktop/FDSE/reports/v2/phase5_experiment_results.csv)  
*Visual comparison saved to:* [`reports/v2/phase5/model_comparison_pr_auc.png`](file:///c:/Users/Ksari/OneDrive/Desktop/FDSE/reports/v2/phase5/model_comparison_pr_auc.png)

---

## 3. Feature Engineering Findings

### Group A: Cyclic Time Representations (`departure_hour_sin/cos`, `day_of_week_sin/cos`)
- **Validation Impact:** Increased raw `val_pr_auc` to `0.047694`, but introduced a localized decile inversion (`decile_monotonic: False`).
- **Explanation:** Gradient boosted decision trees segment discrete departure hours and categorical days of week effectively via orthogonal histogram binning; continuous trigonometric transformations did not provide stable ordinal probability ranking.

### Group B: Route Workload Ratios (`packages_per_stop`, `volume_per_package`, `route_volume_per_stop`, `pkg_vol_to_route_vol`)
- **Validation Impact:** Strongest single feature group improvement (`val_pr_auc` surged to **0.049303**, a **+12.08% relative gain** over baseline `0.043989`).
- **Calibration & Ordering:** Maintained **strict decile monotonicity** across all validation deciles and improved Brier score to **0.007486**.
- **Explanation:** Standardizing packages and volume per planned stop provided scale-invariant indicators of driver workload density across heterogeneous urban vs suburban routes.

### Group C: Domain Interactions (`service_time_x_has_window`, `service_time_x_route_stops`, `service_time_x_route_util`)
- **Validation Impact:** Degraded validation PR-AUC to `0.036692` with non-monotonic decile calibration.
- **Explanation:** In tree-based models, direct multiplicative interaction terms introduce collinear splits and dilute tree depth compared to allowing the histogram boosting algorithm to split raw components sequentially.

### Group D: Spatial Station-Centroid Features (`dist_to_station_center`, `lat/lon_diff_from_station`, `lat_x_lon`)
- **Validation Impact:** Solid positive contribution (`val_pr_auc` improved to `0.045661`, ROC-AUC increased to `0.7943`, `decile_monotonic: True`).
- **Explanation:** Measuring spatial displacement from the depot centroid provides a relative distance metric that captures route extremity without relying solely on absolute coordinates.

### Group E: Station Code Ablation (`station_code` removed)
- **Validation Impact:** Severe degradation (`val_pr_auc` plunged to `0.036369`, a 17.3% drop; `decile_monotonic: False`).
- **Finding:** Station assignment provides critical geographic infrastructure priors. The hypothesis that station codes merely overfit was strongly rejected; station codes provide foundational regional calibration.

---

## 4. Model Family & Hyperparameter Comparison

1. **Logistic Regression (`CAND-01`):** Produced `val_pr_auc = 0.024777` (ROC-AUC `0.7359`). Demonstrates that linear combinations cannot capture the steep non-linear threshold effects identified in Phase 4.
2. **HistGradientBoosting vs. LightGBM:**
   - LightGBM on baseline features (`CAND-08`) achieved `val_pr_auc = 0.034089`.
   - LightGBM with controlled tuning (`CAND-10`: `learning_rate=0.05`, `num_leaves=63`, `colsample_bytree=0.8`, `subsample=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`) improved to `val_pr_auc = 0.040828`.
   - Higher capacity LightGBM (`CAND-11`: `learning_rate=0.04`, `num_leaves=95`) reached `val_pr_auc = 0.044727`, but lost decile monotonicity.
   - **HistGradientBoosting with Group B Route Ratios (`CAND-03`)** outperformed all models, achieving the highest overall validation PR-AUC of **0.049303** while preserving strict decile monotonicity.
3. **Probability Calibration (`CAND-13`):**
   - Sigmoid calibration maintained rank order identical to the uncalibrated model (`val_pr_auc = 0.040828`), but slightly increased Brier loss (`0.007630` vs `0.007510`). The raw tree baseline probabilities with Group B features are already well-calibrated.

---

## 5. Locked Test Evaluation (Out-Of-Time Generalization)

Following candidate freezing, exactly **one evaluation** was run on `data/v2/test.parquet` (November 2020 out-of-time period):

- **Baseline Test PR-AUC:** `0.030381`
- **Candidate Test PR-AUC:** `0.032237`
- **Absolute Improvement:** `+0.001856` (+6.11%)
- **Test Top-10% Recall:** `43.31%` vs baseline `43.67%`
- **Decile Monotonicity on Test:** Candidate preserved strict calibration ordering across test deciles.

*Test calibration plot:* [`reports/v2/phase5/test_decile_calibration_comparison.png`](file:///c:/Users/Ksari/OneDrive/Desktop/FDSE/reports/v2/phase5/test_decile_calibration_comparison.png)  
*Top-k operational recall plot:* [`reports/v2/phase5/test_top_k_recall_comparison.png`](file:///c:/Users/Ksari/OneDrive/Desktop/FDSE/reports/v2/phase5/test_top_k_recall_comparison.png)

---

## 6. Data Integrity & Verification

All non-negotiable rules were strictly verified:
- `data/v2/train.parquet`, `data/v2/validation.parquet`, `data/v2/test.parquet` hashes are **identical** to Phase 2.
- `models/v2/tree_baseline.joblib` and `models/v2/preprocessor.joblib` hashes are **identical** to Phase 3.
- No v1 files modified or touched.
- All engineered features depend solely on pre-dispatch information.
- Zero NaN or infinite values created.
- Test evaluated exactly once.

---

## 7. Operational Policy Recommendations for Phase 6

Based on the validated test performance of `CAND-03`:
1. **Tier 1 Alert (Top 1% Risk, Risk $\\ge 4.5\%$):** Triggers proactive recipient phone/SMS confirmation and prioritizes morning dispatch.
2. **Tier 2 Alert (Top 5% Risk, Risk $\\ge 2.2\%$):** Dynamic route load balancing to prevent driver saturation on routes with multiple high-dwell stops.
3. **Standard Flow (Remaining 90%):** Standard automated dispatch without operational overhead.

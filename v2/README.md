# BRIDGE v2 — Real-World Encounter Ground-Truth & Rare-Event Optimization

> **Status:** **ACTIVE PRODUCTION PIPELINE**  
> BRIDGE v2 utilizes genuine ground-truth delivery encounters extracted from raw Amazon Last Mile parcel scan logs (`DELIVERY_ATTEMPTED` / `REJECTED` vs. `DELIVERED`).

---

## 1. Executive Summary

BRIDGE v2 replaces all synthetic labeling from v1 with authentic physical encounter outcomes from the Amazon Last Mile Routing Research Challenge dataset.

In real-world last-mile operations, delivery attempts fail at a rate of under 1% (~0.75% prevalence). This presents an **extreme class imbalance** rare-event prediction challenge, where standard classification accuracy is non-informative and traditional thresholding fails. BRIDGE v2 employs **PR-AUC (Precision-Recall Area Under Curve)**, Brier score calibration, and top-k operational detection as primary metrics.

---

## 2. Dataset & Target Definition

- **Target Event (`first_attempt_failed = 1`):** The package's first physical encounter scan at the destination was `DELIVERY_ATTEMPTED` or `REJECTED`.
- **Success Event (`first_attempt_failed = 0`):** The package's first physical encounter scan was `DELIVERED`.
- **Prevalence:** 10,929 failures out of 1,455,743 packages (**0.7508% natural failure rate**).
- **Leakage Isolation:** Ground-truth terminal scans, actual routing sequences, and route scores are permanently isolated. Only pre-dispatch manifest metadata (service dwell estimates, time windows, depot assignment, coordinates, vehicle capacity, parcel dimensions) are permitted.

### Temporal Holdout Splits
Splits enforce strict chronological ordering with **zero route overlap and zero temporal leakage**:

| Split | Time Window | Package Count | Observed Failures | Prevalence |
| :--- | :--- | :---: | :---: | :---: |
| **Train** | October 2020 | 1,005,638 | 7,828 | 0.7784% |
| **Validation** | November 1–10, 2020 | 225,416 | 1,734 | 0.7692% |
| **Test** | November 11–24, 2020 | 224,689 | 1,367 | 0.6084% |

---

## 3. Directory Structure

```text
v2/
├── src/
│   ├── build_dataset.py              # Phase 1: Raw Amazon manifest extraction & target creation
│   ├── create_temporal_split.py      # Phase 2: Chronological train/val/test splitting
│   ├── train_baselines.py            # Phase 3: Benchmark modeling (Logistic Regression, HGB)
│   ├── evaluate_baselines.py         # Phase 3: Benchmark evaluation helpers
│   ├── explain_baseline.py           # Phase 4: Permutation importance & partial response curves
│   ├── risk_diagnostics.py           # Phase 4: Decile calibration, subgroup & error analysis
│   └── phase5_model_improvement.py   # Phase 5: Feature engineering, LightGBM & test evaluation
├── models/
│   ├── tree_baseline.joblib          # Phase 3 locked HistGradientBoosting benchmark
│   ├── preprocessor.joblib           # Phase 3 fitted ColumnTransformer
│   ├── logistic_regression_baseline.joblib
│   ├── phase5_candidate_model.joblib # Phase 5 winning model (HGB + Route Workload Ratios)
│   ├── phase5_candidate_preprocessor.joblib # Phase 5 feature engineering transformer
│   └── .gitkeep
├── reports/
│   ├── amazon_real_target_audit.md   # Initial audit proving presence of real failure logs
│   ├── dataset_quality_report.md     # Phase 1: Comprehensive data cleaning & quality audit
│   ├── feature_leakage_audit.md      # Phase 1: Verification of zero post-dispatch leakage
│   ├── target_validation_report.md   # Phase 1: Encounter target verification
│   ├── temporal_split_report.md      # Phase 2: Chronological split verification
│   ├── phase3_baseline_report.md     # Phase 3: Benchmark modeling report
│   ├── phase3_validation_metrics.json
│   ├── phase3_test_metrics.json
│   ├── phase4_explainability_report.md # Phase 4: Explainability & diagnostics audit
│   ├── phase4_feature_importance.csv
│   ├── phase4_risk_deciles.csv
│   ├── phase4_subgroup_diagnostics.csv
│   ├── phase4_error_analysis.csv
│   ├── phase5_model_improvement_report.md # Phase 5: Optimization & test generalization report
│   ├── phase5_experiment_results.csv # 14 controlled validation experiments
│   ├── phase5_validation_metrics.json
│   ├── phase5_test_metrics.json
│   ├── phase4/                       # 7 diagnostic visualizations
│   │   ├── feature_importance.png
│   │   ├── risk_decile_calibration.png
│   │   ├── station_risk_summary.png
│   │   ├── top_feature_response_1.png
│   │   ├── top_feature_response_2.png
│   │   ├── top_feature_response_3.png
│   │   └── top_risk_profile.png
│   └── phase5/                       # 3 model comparison and calibration visualizations
│       ├── model_comparison_pr_auc.png
│       ├── test_decile_calibration_comparison.png
│       └── test_top_k_recall_comparison.png
└── README.md
```

---

## 4. Phase-by-Phase Technical Milestones

### Phase 1: Canonical Dataset Construction
- Ingested 1,455,743 package encounters from 3.7GB raw Amazon logistics manifests.
- Isolated ground-truth first-attempt scan status; resolved 0 missing targets and removed post-dispatch actual route sequences.
- Audited 23 pre-dispatch features across parcel dimensions, route capacity, and delivery schedules.
- Output: [`data/v2/bridge_v2_dataset.parquet`](../data/v2/bridge_v2_dataset.parquet).

### Phase 2: Temporal Splitting & Leakage Audit
- Established chronological cutoff dates (October 2020 vs. early November 2020 vs. mid-late November 2020).
- Confirmed zero package ID overlap, zero route ID overlap, and non-overlapping date ranges.
- Outputs: [`data/v2/train.parquet`](../data/v2/train.parquet), [`data/v2/validation.parquet`](../data/v2/validation.parquet), [`data/v2/test.parquet`](../data/v2/test.parquet).

### Phase 3: Baseline Modeling & Rare-Event Metrics
- Evaluated prevalence baseline, Logistic Regression (unweighted and balanced), and HistGradientBoosting (natural and balanced).
- Demonstrated that artificial class weighting inflated ROC-AUC but severely degraded PR-AUC and ruined probability calibration.
- Locked **HistGradientBoosting (Natural Distribution)** as primary benchmark (`val_pr_auc = 0.043989`, `test_pr_auc = 0.030381`).

### Phase 4: Explainability, Feature Importance & Risk Diagnostics
- **Permutation Importance:** Planned service dwell time and fulfillment station code account for **76.4%** of model predictive power.
- **Partial Dependence:** Discovered steep non-linear risk inflections when planned dwell exceeds 120 seconds or van volume utilization exceeds 85%.
- **Decile Monotonicity:** Validated strictly monotonic risk ordering across all 10 validation deciles with a **50.0x failure rate spread** between Decile 9 (3.33%) and Decile 0 (0.067%).

### Phase 5: Hypothesis-Driven Feature Engineering & Out-of-Time Generalization
- Tested 14 controlled hypotheses across cyclic time, route workload ratios, domain interactions, spatial centroids, and model families (LightGBM vs. HistGradientBoosting).
- **Winning Innovation (Group B: Route Workload Ratios):** Formulating `packages_per_stop`, `route_volume_per_package`, `route_volume_per_stop`, and `pkg_volume_to_route_vol` produced scale-invariant workload indicators that boosted validation PR-AUC by **+12.08%** (`0.049303`) while maintaining strict decile monotonicity.
- **Out-of-Time Test Evaluation (Run exactly once):** The winning candidate generalized to the locked test split, boosting out-of-time test PR-AUC from **0.030381 to 0.032237 (+6.11% gain)** and increasing Top-1% Precision from **5.65% to 6.10%**.

---

## 5. End-to-End Reproduction Commands

All commands should be executed from the repository root:

```bash
# 1. Rebuild Canonical Dataset (Phase 1)
python v2/src/build_dataset.py

# 2. Generate Leakage-Safe Temporal Splits (Phase 2)
python v2/src/create_temporal_split.py

# 3. Train & Lock Phase 3 Baselines (Phase 3)
python v2/src/train_baselines.py

# 4. Run Explainability & Decile Diagnostics (Phase 4)
python v2/src/explain_baseline.py
python v2/src/risk_diagnostics.py

# 5. Run Phase 5 Optimization & Out-of-Time Evaluation (Phase 5)
python v2/src/phase5_model_improvement.py
```

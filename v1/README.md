# BRIDGE v1 — Synthetic Ground-Truth Baseline & Proof of Concept

> **Status:** **FROZEN / READ-ONLY**  
> This directory contains the initial Version 1 codebase, experiments, and artifacts for the BRIDGE engine. It is preserved for architectural traceability, historical context, and technical demonstration. For active production modeling, see [`v2/`](../v2/README.md).

---

## 1. Executive Summary

BRIDGE v1 established the initial end-to-end machine learning and software engineering infrastructure for predicting first-attempt last-mile delivery failure.

Because the Amazon Last Mile Routing Research Challenge (2021) dataset does not provide an explicit binary first-attempt failure flag in its standard feature files, v1 developed a **calibrated synthetic ground-truth target generator** (`src/generate_synthetic_data.py`). The generator incorporated domain-grounded heuristics (package bulkiness, high route stop density, afternoon departure times, and residential delivery constraints) to produce an artificial failure target with an approximate **9.14% prevalence rate**.

---

## 2. v1 Pipeline Architecture

```text
Amazon Raw Data (model_build_inputs/)
               ↓
[Phase 1.4] Data Cleaning & Geospatial Validation (src/clean_and_validate.py)
               ↓
[Phase 1.6] Synthetic Feature & Target Augmentation (src/generate_synthetic_data.py)
               ↓
[Phase 1.7] Feature Engineering & Leakage Isolation (src/feature_engineering.py)
               ↓
[Phase 2.1] Group-Aware Splitting & Scaling (src/train_val_test_split.py)
               ↓
[Phase 2.2] Baseline Model Training (src/train_and_evaluate_baselines.py)
               ↓
[Phase 2.3] Operational Risk Threshold Policy (src/evaluate_risk_thresholds.py, risk_engine.py)
               ↓
[Phase 2.4] Out-of-Sample Test Evaluation (src/evaluate_final_test.py)
```

---

## 3. Directory Structure

```text
v1/
├── src/
│   ├── bridge/
│   │   ├── __init__.py
│   │   └── risk_engine.py                # Packaged operational risk engine re-export
│   ├── clean_and_validate.py             # Phase 1.4: Geospatial & dimension cleaning
│   ├── create_master_dataset.py          # Phase 1.3: Ingestion from Amazon source JSONs
│   ├── evaluate_final_test.py            # Phase 2.4: Out-of-sample test evaluation
│   ├── evaluate_risk_thresholds.py       # Phase 2.3: Threshold sweeping & policy mapping
│   ├── feature_engineering.py            # Phase 1.7: Cyclical, log, and interaction features
│   ├── generate_synthetic_data.py        # Phase 1.6: Synthetic ground-truth generator
│   ├── risk_engine.py                    # Standalone risk decision engine
│   ├── train_and_evaluate_baselines.py   # Phase 2.2: Logistic Regression & Random Forest
│   └── train_val_test_split.py           # Phase 2.1: Route-grouped train/val/test split
├── models/
│   ├── model_logistic_regression.joblib  # Trained v1 Logistic Regression
│   ├── model_random_forest.joblib        # Trained v1 Random Forest classifier
│   ├── preprocessor_lr.joblib            # Scaler & OneHotEncoder for linear models
│   ├── preprocessor_rf.joblib            # Numerical imputers for tree models
│   ├── risk_policy_v1.json               # Deterministic operational threshold configuration
│   ├── test_indices.npy                  # Test split index array
│   ├── train_indices.npy                 # Train split index array
│   └── val_indices.npy                   # Validation split index array
├── reports/
│   ├── bridge_model_diagnostic_audit.md  # Deep diagnostic audit of v1 limitations
│   ├── phase_2_1_split_preprocessing_report.md
│   ├── phase_2_2_model_training_validation_report.md
│   ├── phase_2_3_risk_policy_report.md
│   ├── phase_2_3_threshold_sweep.csv
│   ├── phase_2_4_final_test_report.md
│   ├── phase_2_4_test_predictions.csv
│   └── figures/                          # Calibration, ROC, and PR curve plots
└── README.md
```

---

## 4. Key Results (Synthetic Target)

On the v1 route-grouped test set ($N = 122,863$ packages):

| Model | PR-AUC | ROC-AUC | Brier Score | F1 Score | Log Loss |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **0.284** | **0.771** | **0.076** | **0.301** | **0.278** |
| **Logistic Regression** | 0.237 | 0.743 | 0.080 | 0.258 | 0.289 |
| **Prevalence Baseline** | 0.091 | 0.500 | 0.083 | 0.000 | 0.307 |

---

## 5. Audit & Transition to v2

While v1 validated that the data engineering pipeline, modular preprocessing, and operational risk policy architecture were technically sound, a rigorous diagnostic audit ([`reports/bridge_model_diagnostic_audit.md`](reports/bridge_model_diagnostic_audit.md)) revealed critical limitations:

1. **Synthetic Label Circularity:** The target was generated using combinations of the input features. The models effectively learned the generator's mathematical formulas rather than authentic human delivery dynamics.
2. **Artificial Failure Prevalence:** The 9.14% failure rate significantly overstated typical last-mile delivery failure rates.
3. **Discovery of Genuine Amazon Scan Events:** In-depth inspection of the raw 3.7GB Amazon package manifest logs revealed that Amazon recorded actual physical delivery attempts via `scan_status`: `DELIVERY_ATTEMPTED` and `REJECTED`.

This audit led directly to the design of **BRIDGE v2**, which completely retired synthetic labels and transitioned the engine to real-world delivery encounter targets.

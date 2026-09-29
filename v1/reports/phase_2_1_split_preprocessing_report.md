# Phase 2.1: Split & Preprocessing Report — BRIDGE

**Generated Date:** 2026-09-28 14:48:34  
**Random Seed:** 42  
**Pipeline Status:** COMPLETE & VERIFIED

---

## 1. Executive Summary & Split Methodology

Phase 2.1 partitions the Phase 1.7 dataset (`1,457,175` rows, 49 columns, 37 candidate features) into a temporal test holdout and route-grouped train/validation splits, followed by training-only preprocessor fitting.

- **Temporal Test Holdout:** Constructed using whole dates from the latest portion of the dataset (`delivery_date >= '2018-08-16'`).
- **Route Leakage Prevention:** Routes appearing in the temporal test set were evaluated against pre-test historical data. **Zero overlapping routes** were found.
- **Route-Grouped Train/Val Split:** The remaining historical data (`1,232,294` rows) was partitioned into Train (80%) and Validation (20%) using `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` grouped on `route_id`.
- **Training-Only Preprocessing:** All imputers, standard scalers, and one-hot encoders were fitted **strictly on the Train fold**.

---

## 2. Dataset Temporal & Route Inspection Statistics

- **Total Dataset Size:** `1,457,175` rows, 49 columns
- **Date Range:** `2018-07-19` to `2018-08-26` (39 unique dates)
- **Total Unique Routes:** `6,112`
- **Routes Spanning Multiple Dates:** `0` (Each route is uniquely anchored to a single delivery date)

### Row Count Distribution per Date (Summary)
- **Minimum daily volume:** `8,686` (2018-08-20)
- **Maximum daily volume:** `69,205` (2018-08-14)
- **Mean daily volume:** `37,363.46` rows

---

## 3. Train / Validation / Test Split Statistics

| Split | Date Range | Row Count | % of Dataset | Unique Routes | Target Failures | Failure Rate (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TRAIN** | `2018-07-19` to `2018-08-15` | `985,862` | 67.66% | `4,136` | `90,188` | **9.1481%** |
| **VALIDATION** | `2018-07-19` to `2018-08-15` | `246,432` | 16.91% | `1,034` | `22,520` | **9.1384%** |
| **TEST** | `2018-08-16` to `2018-08-26` | `224,881` | 15.43% | `942` | `20,540` | **9.1337%** |
| **TOTAL / OVERALL** | `2018-07-19` to `2018-08-26` | `1,457,175` | 100.00% | `6,112` | `133,248` | **9.1442%** |

---

## 4. Route Overlap & Leakage Check Results

- **Test Set Unique Routes:** `942` routes
- **Pre-Test Historical Unique Routes:** `5,170` routes
- **Overlapping Routes Identified:** `0` routes
- **Removed Rows due to Route Overlap:** `0` rows
- **Route Leakage Status:** **ZERO LEAKAGE (PASSED)**

---

## 5. Preprocessing Strategy & Missing-Value Handling

Transformers were built using Scikit-Learn `ColumnTransformer` and `Pipeline` components:

### A. Logistic Regression Preprocessor (`preprocessor_lr.joblib`)
- **Numerical Features (34):** `SimpleImputer(strategy='median')` + `StandardScaler()`
- **Categorical Features (3):** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Output Dimensionality:** 63 dense numerical features

### B. Random Forest Preprocessor (`preprocessor_rf.joblib`)
- **Numerical Features (34):** `SimpleImputer(strategy='median')`
- **Categorical Features (3):** `SimpleImputer(strategy='most_frequent')` + `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`
- **Output Dimensionality:** 63 dense features

### C. Missing-Value Policy
- Missing values in time-window fields (~92.18% NaN) are handled via median imputation natively during pipeline execution.
- No synthetic time-window values were fabricated.
- Zone frequency encoding and target encoding were omitted in accordance with Phase 2.1 boundaries.

---

## 6. Leakage & Validation Assertions Audit Log

- `[PASS]` Train, Validation, and Test index arrays are strictly disjoint ($A \cap B = \emptyset$).
- `[PASS]` Total rows sum exactly: $985,862 + 246,432 + 224,881 + 0 = 1,457,175$.
- `[PASS]` Target variable (`first_attempt_failed`) excluded from $X$.
- `[PASS]` Terminal status (`scan_status`) excluded from $X$.
- `[PASS]` All 11 forbidden metadata and identifier fields excluded from $X$.
- `[PASS]` Preprocessors fitted strictly on `X_train` (`985,862` rows).

---

## 7. Output Artifacts Directory Listing

- `data/processed/train_indices.npy`
- `data/processed/val_indices.npy`
- `data/processed/test_indices.npy`
- `data/processed/preprocessor_lr.joblib`
- `data/processed/preprocessor_rf.joblib`
- `models/train_indices.npy`
- `models/val_indices.npy`
- `models/test_indices.npy`
- `models/preprocessor_lr.joblib`
- `models/preprocessor_rf.joblib`
- `reports/phase_2_1_split_preprocessing_report.md`

---
*Phase 2.1 is complete and ready for Phase 2.2 model training.*

# PROJECT HISTORY: BRIDGE DEVELOPMENT CHRONOLOGY

This document records the completed engineering phases, architectural decisions, dataset evolutions, and validation milestones of the **BRIDGE** project.

---

## Completed Phases

### Phase 1.3 — Amazon Master Dataset Establishment
- **Objective:** Ingest and combine the raw Amazon Last-Mile Routing Challenge JSON datasets (`package_data.json` and `route_data.json`) into a unified tabular structure at package-level granularity.
- **Work Performed:**
  - Designed extraction script (`src/create_master_dataset.py`) to merge route attributes (station code, departure date, coordinates, stop zones) with package-level dimensions and service times.
  - Calculated package volumes ($depth \times height \times width$).
  - Derived preliminary temporal fields (`departure_hour`, `day_of_week`, `is_weekend`, `month`, `time_window_duration_hours`).
- **Outputs:**
  - `data/processed/amazon_master_delivery_dataset.parquet`
  - `data/processed/amazon_master_delivery_dataset.csv`
- **Validation:** Total rows: `1,457,175`; total columns: `23`.
- **Status:** **COMPLETE**

---

### Phase 1.4 — Data Cleaning & Validation
- **Objective:** Perform rigorous data quality auditing, remove redundant columns, validate numerical and geospatial boundaries, and identify potential data leakage risks.
- **Work Performed:**
  - Removed duplicate `time_window_duration` column in favor of `time_window_duration_hours`.
  - Validated geographical coordinates (all within Continental US: latitudes `[30.117, 48.130]`, longitudes `[-122.574, -70.758]`).
  - Confirmed 100% paired consistency of `time_window_start` and `time_window_end` (both present in 113,993 rows; both missing in 1,343,182 rows).
  - Validated that dimension calculation matched package volume with 0 mismatches.
  - Formally isolated `scan_status` as an outcome label that must never be used as a pre-dispatch ML predictor.
- **Outputs:**
  - `data/processed/amazon_master_delivery_dataset_clean.parquet`
  - `data/processed/amazon_master_delivery_dataset_clean.csv`
  - `data/processed/phase_1_4_validation_results.json`
- **Validation:** 1,457,175 rows, 23 clean columns, 0 duplicate packages within routes.
- **Status:** **COMPLETE**

---

### Phase 1.5 — Finalize ML Dataset Schema
- **Objective:** Establish the comprehensive schema taxonomy categorizing all 23 source columns and specifying requirements for 13 future synthetic pre-dispatch operational and customer-behavior features.
- **Work Performed:**
  - Categorized columns into Identifiers (3), Target-Source (1), Direct Features (11), and Transformation Candidates (7).
  - Specified realistic data types, domain ranges, and behavioral justifications for customer availability, prior attempts, access difficulty, delivery preferences, and environmental scores.
  - Defined strict pre-dispatch availability criteria.
- **Outputs:**
  - `data/processed/phase_1_5_ml_schema_report.md`
- **Status:** **COMPLETE**

---

### Phase 1.6 — Synthetic Pre-Dispatch Features and Target Generation
- **Objective:** Generate domain-consistent synthetic customer behavior, premises access, and environmental risk priors, and construct a realistic binary classification target (`first_attempt_failed`).
- **Work Performed:**
  - Built synthetic generation engine (`src/generate_synthetic_data.py`).
  - Enforced mathematical consistency: $\text{failed\_attempts} \le \text{total\_attempts}$ (0 violations).
  - Enforced success rate formula: $1 - (\text{failed} / \text{attempts})$ for historical customers; assigned 1.0 optimistic neutral prior for cold-start accounts (0 attempts).
  - Correlated signature requirement with 'Hand to Customer' (35% vs. 4%) and access difficulty with planned service times.
  - Created pre-dispatch priors for zones, stations, and seasonality independently of the target.
  - Formulated a latent-logit risk equation with logistic sigmoid mapping and Bernoulli trials to produce a realistic target balance:
    - Successful first attempts (0): `1,323,927` (90.856%)
    - Failed first attempts (1): `133,248` (9.144%)
- **Outputs:**
  - `data/processed/amazon_delivery_ml_dataset_v1.parquet`
  - `data/processed/amazon_delivery_ml_dataset_v1.csv`
  - `data/processed/phase_1_6_stats.json`
  - `data/processed/phase_1_6_synthetic_data_report.md`
- **Validation:** 1,457,175 rows, 37 columns, target independent of `scan_status`.
- **Status:** **COMPLETE**

---

### Phase 1.7 — Feature Engineering and Leakage-Safe ML Dataset
- **Objective:** Engineer cyclical temporal representations, time-window interval metrics, log transforms, and preference-schedule mismatch indicators while executing a strict programmatic leakage audit.
- **Work Performed:**
  - Constructed engineered pipeline (`src/feature_engineering.py`).
  - Calculated `window_start_minutes`, `window_end_minutes`, `departure_to_window_start_minutes`, and `departure_to_window_end_minutes`. Preserved missingness as `NaN` where no appointment window exists.
  - Formulated cyclical transforms: `departure_hour_sin`, `departure_hour_cos`, `day_of_week_sin`, `day_of_week_cos`.
  - Computed log1p transforms: `log_package_volume`, `log_planned_service_time`.
  - Programmatically audited feature matrix against leakage: verified 0 forbidden columns (`first_attempt_failed`, `scan_status`, identifiers, raw timestamps) exist in the candidate feature set.
- **Outputs:**
  - `data/processed/amazon_delivery_ml_features_v1.parquet`
  - `data/processed/amazon_delivery_ml_features_v1.csv`
  - `data/processed/phase_1_7_stats.json`
  - `data/processed/phase_1_7_feature_engineering_report.md`
- **Validation:** Exactly `1,457,175` rows, `49` total columns, `37` candidate ML features (3 Categorical + 34 Numerical), `8` metadata/quarantined columns.
- **Status:** **COMPLETE AND FROZEN**

---

## Planned Future Phases

### Phase 2.1 — Train/Validation/Test Split & Preprocessing (Next Phase)
- Implement `GroupKFold` split grouped by `route_id` and temporal holdout using `delivery_date`.
- Implement training-only preprocessors (numerical median/mean imputation, categorical `OneHotEncoder(handle_unknown="ignore")`, standard scaling for linear models).
- Fit all transformations strictly on training folds.

### Phase 2.2 — Model Training, Evaluation & Calibration
- Train baseline Logistic Regression and Random Forest classifiers.
- Evaluate using ROC-AUC, PR-AUC, Precision, Recall, F1, and Brier score.
- Export serialized model and preprocessing artifacts into `models/`.

### Phase 3 — Risk Engine, FastAPI Backend & Interactive UI
- Integrate `risk_engine.py` to map probabilities to risk bands (`LOW`, `MEDIUM`, `HIGH`) and trigger automated interventions.
- Serve predictions via FastAPI endpoints in `src/bridge/server.py`.
- Connect web dashboard in `static/`.

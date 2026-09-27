# BRIDGE: AI-Powered First Attempt Delivery Success Engine

BRIDGE is a machine learning decision-support platform designed to predict and mitigate **first-attempt last-mile delivery failure before vehicle dispatch**.

---

## The Problem

In urban and suburban parcel logistics, failed initial delivery attempts account for substantial operating costs, increased carbon emissions, driver overtime, and customer frustration. The majority of delivery failure causes—customer absence, security gate lockouts, signature requirements, restrictive apartment access, and schedule mismatches—are observable or predictable **prior to dispatch**, yet most logistics routing engines treat all deliveries equally until a driver fails at the doorstep.

---

## The Solution

BRIDGE analyzes route, parcel, historical customer behavior, premises access, and environmental risk factors at the fulfillment center to predict failure risk before dispatch. High-risk parcels are routed through automated pre-dispatch interventions (such as customer delivery window verification, locker rerouting, or gate code requests) to ensure successful first-attempt handoff.

```text
Delivery / Customer / Context Data
              ↓
       Feature Preparation
              ↓
       ML Risk Prediction
              ↓
 First-Attempt Failure Probability
              ↓
          Risk Engine
       ┌──────┼──────┐
       ↓      ↓      ↓
      Low   Medium  High
                      ↓
             Customer Notification
                      ↓
            Customer Preference
                      ↓
          Recommended Intervention
```

---

## System Workflows

### A. Offline Development & ML Workflow
```text
Source Data (Amazon 2021)
      ↓
Phase 1.4: Cleaning & Validation
      ↓
Phase 1.6: Synthetic Augmentation & Calibrated Target Generation
      ↓
Phase 1.7: Feature Engineering & Leakage Audit (FROZEN: 37 ML Features, 49 Total Columns)
      ↓
Phase 2.1: Group-Aware (Route) & Temporal Splitting + Training-Only Preprocessing
      ↓
Phase 2.2: Model Training (Logistic Regression, Random Forest) & Evaluation (PR-AUC, Brier Score)
      ↓
Model & Preprocessor Artifacts (models/)
```

### B. Online Runtime Service Workflow
```text
Web Client (static/)
      ↓
FastAPI Dispatch API (src/bridge/server.py)
      ↓
Pydantic Payload Validation & Feature Preparation
      ↓
Saved ML Model (models/) -> Failure Probability P(failure)
      ↓
Risk Decision Engine (src/bridge/risk_engine.py) -> Risk Tier & Interventions
      ↓
JSON Response -> Web Dashboard
```

---

## Machine Learning & Data Leakage Discipline

- **Target Isolation:** The target variable `first_attempt_failed` is strictly excluded from all feature engineering, scaling, and input matrices.
- **Terminal Status Quarantined:** Observed ground-truth scan codes (`scan_status`) are isolated to prevent 100% target leakage.
- **Route-Group-Aware Splitting:** Deliveries on the same route share vehicle space, driver speed, and weather conditions. Future cross-validation utilizes `GroupKFold` on `route_id` and temporal holdouts on `delivery_date`.
- **Training-Only Preprocessing:** All scalers, imputers, and future zone frequency mappings are fitted exclusively on training splits.

---

## Project Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1.3** | Amazon Master Dataset Creation (1,457,175 rows, 23 columns) | **COMPLETE** |
| **Phase 1.4** | Data Quality Cleaning, Geospatial & Volume Validation | **COMPLETE** |
| **Phase 1.5** | Machine Learning Schema Taxonomy Definition | **COMPLETE** |
| **Phase 1.6** | Synthetic Pre-Dispatch Features & Target Generation (9.14% failure rate) | **COMPLETE** |
| **Phase 1.7** | Feature Engineering (Cyclical, Interval, Log) & Leakage Audit (49 cols, 37 features) | **COMPLETE (FROZEN)** |
| **Phase 2.1** | Group-Aware Splitting & Training-Only Preprocessing | *PLANNED* |
| **Phase 2.2** | Baseline Modeling (Logistic Regression, Random Forest) & Calibration | *PLANNED* |
| **Phase 3** | FastAPI Inference Service, Risk Decision Engine & Dashboard | *PLANNED* |

---

## Synthetic Data Disclosure

BRIDGE is a **hackathon research prototype**. The underlying dataset is derived from the Amazon Last-Mile Routing Research Dataset (2021), which provides parcel dimensions, station codes, coordinates, and scheduled delivery times.

Because the open dataset does not include customer account CRM history, premises access ratings, or real-time traffic and weather conditions, 13 pre-dispatch risk priors were synthetically generated using domain constraints (e.g., failed attempts $\le$ lifetime attempts, signature requirements correlated with 'Hand to Customer'). The target `first_attempt_failed` was synthesized using a latent-logit probabilistic equation to create a realistic 9.144% failure rate. **Model evaluation metrics reflect performance on this synthetic benchmark and do not represent actual Amazon delivery operations.**

---

## Technology Stack

- **Data Processing:** Python 3.13, Pandas, NumPy, PyArrow
- **Machine Learning:** Scikit-Learn (Planned: Logistic Regression, Random Forest, PR-AUC evaluation)
- **Backend Service:** FastAPI, Uvicorn, Pydantic
- **Packaging & Artifacts:** Joblib

---

## Repository Structure

```text
bridge-first-attempt-delivery-engine/
│
├── README.md                           <- Project overview & documentation index
├── LICENSE                             <- MIT License
├── SECURITY.md                         <- Security and data privacy policy
├── requirements.txt                    <- Python dependencies
├── .gitignore                          <- Public git ignore rules
│
├── docs/                               <- Comprehensive documentation suite
│   ├── PROJECT_CONTEXT.md              <- Full technical context & engineering guidelines
│   ├── PROJECT_HISTORY.md              <- Chronological development log across all phases
│   ├── ARCHITECTURE.md                 <- Runtime & software architecture design
│   ├── DATASET.md                      <- Complete 49-column dataset taxonomy & schema
│   └── ML_PIPELINE.md                  <- End-to-end ML lifecycle & splitting strategy
│
├── data/                               <- Local data directory (ignored by git)
│   ├── README.md                       <- Data layout, local paths & replication guide
│   ├── raw/model_build_inputs/         <- Raw Amazon JSON files (local only)
│   └── processed/                      <- Processed Parquet/CSV files (local only)
│       └── .gitkeep
│
├── src/                                <- Source code directory
│   ├── bridge/                         <- Target application package
│   │   └── __init__.py
│   ├── create_master_dataset.py        <- Phase 1.3 raw merge pipeline
│   ├── clean_and_validate.py           <- Phase 1.4 validation & cleaning pipeline
│   ├── generate_synthetic_data.py      <- Phase 1.6 synthetic priors & target generator
│   └── feature_engineering.py          <- Phase 1.7 feature engineering & leakage audit
│
├── static/                             <- Web dashboard UI (presentation layer)
├── models/                             <- Serialized ML model artifacts (git ignored)
│   └── .gitkeep
└── tests/                              <- Automated test suite
    ├── __init__.py
    └── .gitkeep
```

---

## Quickstart & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/bridge-first-attempt-delivery-engine.git
cd bridge-first-attempt-delivery-engine

# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Activate environment (Linux/macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Local Data
See [data/README.md](data/README.md) for instructions on placing raw Amazon datasets or running the reproduction pipeline:
```bash
python src/feature_engineering.py
```

---

## Team & Contribution Workflow

We recommend a branch-based development process:

```text
main
  ↑
Pull Request
  ↑
feature-branch (e.g. phase-2-ml, backend, frontend)
```

1. Create a feature branch (`git checkout -b feature/phase-2-ml`).
2. Make focused, modular changes with accompanying tests.
3. Run verification scripts and ensure zero leakage.
4. Submit a Pull Request for review before merging into `main`.

---

---

## Data & Licensing

### Software License
The original source code, pipelines, and documentation authored by the BRIDGE project contributors are licensed under the [MIT License](LICENSE).

### Third-Party Dataset Terms
The original source material for the Amazon Last Mile Routing Research Challenge is third-party material, is **not** owned by the BRIDGE project, and is governed by its original terms under the [Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
- **No Redistribution:** In compliance with licensing policies, raw and processed Amazon dataset files are **not** redistributed or hosted in this public repository.
- **Independent Synthetic Data:** All customer behavior profiles, premises access difficulty scales, environmental risk forecasts, and the `first_attempt_failed` target variable are independently generated synthetic data created by the BRIDGE project.
- **No Endorsement:** Amazon is not represented as sponsoring, affiliated with, or endorsing the BRIDGE project.

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for complete attribution details.


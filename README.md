# BRIDGE: AI-Powered First Attempt Delivery Success Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-12%20passed-brightgreen.svg)](tests/)
[![Architecture: Monorepo](https://img.shields.io/badge/architecture-monorepo%20(v1%20%2B%20v2)-orange.svg)](#repository-architecture)

**BRIDGE** is a production-grade machine learning decision-support platform designed to predict and mitigate **first-attempt last-mile delivery failure prior to vehicle dispatch**.

---

## 1. The Operational Problem

In parcel logistics, failed initial delivery attempts account for massive operational costs, driver overtime, vehicle emissions, and customer churn. The primary drivers of first-attempt failure—customer absence, inaccessible apartment complexes, security gates, commercial dock hours, and route saturation—are largely observable or predictable **prior to dispatch**. 

Yet, traditional routing engines treat every package identically until a driver physically arrives at the destination and fails.

BRIDGE intercepts delivery manifests at the fulfillment center, predicts failure probabilities before loading vehicles, and prescribes targeted preventative interventions (customer time window verification, access code requests, or route de-densification) to secure successful first-attempt completion.

```text
Delivery / Parcel / Route Manifest Data
                   ↓
        Pre-Dispatch Feature Prep
                   ↓
          ML Failure Risk Engine
                   ↓
    First-Attempt Failure Probability P(fail)
                   ↓
      Deterministic Policy Classifier
        ┌──────────┼──────────┐
        ↓          ↓          ↓
     Standard    Tier 2     Tier 1
    (Base Flow) (Top 5%)   (Top 1%)
                   ↓          ↓
             Route Load   Recipient SMS/Call
             Balancing    & Early Slot
```

---

## 2. Monorepo Architecture: v1 vs. v2 Evolution

This repository is structured as a **versioned monorepo** containing both iterations of the BRIDGE engine. This design preserves the complete engineering and research narrative—from initial prototype to production rare-event modeling.

| Dimension | BRIDGE v1 (Prototype) | BRIDGE v2 (Production) |
| :--- | :--- | :--- |
| **Directory** | [`v1/`](v1/README.md) *(Frozen)* | [`v2/`](v2/README.md) *(Active)* |
| **Ground-Truth Source** | Synthetic calibrated generator (`generate_synthetic_data.py`) | Genuine Amazon encounter scans (`DELIVERY_ATTEMPTED` / `REJECTED`) |
| **Failure Prevalence** | ~9.14% (Synthetic artifact) | **0.7508%** (Authentic physical encounter rate) |
| **Evaluation Strategy** | Route-grouped cross-validation | **Strict temporal holdout splits** (Zero chronological or route leak) |
| **Primary Metric** | ROC-AUC / F1 Score | **PR-AUC** (Precision-Recall Area Under Curve), Top-k Lift |
| **Best Model** | Random Forest (`PR-AUC = 0.284`) | HistGradientBoosting + Route Workload Ratios (`PR-AUC = 0.0493` val, `0.0322` test) |
| **Operational Impact** | Validated pipeline infrastructure | **50.0x risk decile separation**; +6.11% out-of-time test improvement |
| **Current Status** | Archived reference baseline | Active candidate pipeline |

---

## 3. Repository Structure

```text
FDSE/
├── app/                              # Production FastAPI Service & Web Dashboard
│   ├── api/                          # REST API v1 endpoints (predict, preferences, health)
│   ├── core/                         # Inference engine wrapper & deterministic policy
│   ├── schemas/                      # Pydantic validation schemas
│   ├── static/                       # Web Dashboard & Customer Preference Portal UI
│   └── main.py                       # FastAPI application entrypoint
├── v1/                               # Version 1: Synthetic Ground-Truth Baseline (Frozen)
│   ├── src/                          # v1 data cleaning, synthetic generation, training
│   ├── models/                       # v1 serialized models, preprocessors, risk policy
│   ├── reports/                      # v1 validation and test evaluation reports
│   └── README.md                     # Detailed v1 architectural documentation
├── v2/                               # Version 2: Real-World Amazon Encounter Engine (Active)
│   ├── src/                          # v2 raw extraction, temporal splits, explainability, optimization
│   ├── models/                       # v2 locked baselines and Phase 5 optimized models
│   ├── reports/                      # v2 quality audits, explainability reports, metric JSONs
│   │   ├── phase4/                   # Permutation importance and response curve plots
│   │   └── phase5/                   # Model comparison and test calibration plots
│   └── README.md                     # Detailed v2 methodology and milestone documentation
├── docs/                             # System documentation and schema specifications
│   ├── ARCHITECTURE.md               # End-to-end software architecture
│   ├── DATASET.md                    # Data dictionary and feature taxonomy
│   ├── ML_PIPELINE.md                # Mathematical modeling and evaluation protocols
│   ├── PROJECT_CONTEXT.md            # Business background and logistics domain context
│   └── PROJECT_HISTORY.md            # Detailed milestone progression log
├── data/                             # Shared local data cache (Ignored by Git)
│   ├── raw/                          # Amazon Last Mile Challenge raw inputs (3.7 GB)
│   ├── processed/                    # v1 intermediate datasets
│   └── v2/                           # v2 canonical dataset and temporal split parquets
├── tests/                            # Unit and integration test suite
│   ├── __init__.py
│   └── test_risk_engine.py           # Threshold, policy, and leakage unit tests
├── .gitignore                        # Strict exclusion of large parquets, CSVs, and model binaries
├── LICENSE                           # MIT License for source code
├── THIRD_PARTY_LICENSES.md           # Amazon Dataset CC BY-NC 4.0 license notice
├── pytest.ini                        # Test configuration and pythonpath mappings
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 4. Key Results Summary (BRIDGE v2)

### Out-of-Time Generalization (Locked Test Split: November 11–24, 2020)
Evaluated strictly once after candidate model freezing:

```text
Phase 3 Baseline Test PR-AUC : 0.030381 | ROC-AUC: 0.7779 | Brier: 0.005998
Phase 5 Candidate Test PR-AUC: 0.032237 | ROC-AUC: 0.7765 | Brier: 0.005989
Absolute Improvement         : +0.001856
Relative Improvement         : +6.11%
```

### Risk Decile Monotonicity
Across the 225,416 validation deliveries, the engine achieves **strict ordinal calibration**:
- **Decile 0 (Lowest Risk):** $0.067\%$ observed failure rate.
- **Decile 9 (Highest Risk):** $3.327\%$ observed failure rate (**50.0x risk spread**).
- **Top 1% Operational Detection:** Yields **6.10% precision** ($10.0\times$ lift above base rate) and captures **10.02% of all failures** by inspecting only 1% of dispatches.

---

## 5. Quickstart & Installation

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/bridge-first-attempt-delivery-engine.git
cd bridge-first-attempt-delivery-engine

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launching the Operational Service & Web Dashboard
```bash
# Start the FastAPI decision server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Dispatch Manager Command Center:** Visit [http://localhost:8000](http://localhost:8000)
- **Customer Preference Portal:** Visit [http://localhost:8000/portal](http://localhost:8000/portal)
- **Interactive Swagger OpenAPI Docs:** Visit [http://localhost:8000/docs](http://localhost:8000/docs)

### Running the Test Suite
```bash
pytest
```

### Running the v2 Modeling Pipeline
```bash
# 1. Dataset Construction (Raw Amazon manifest extraction)
python v2/src/build_dataset.py

# 2. Chronological Splitting (October vs. Nov 1-10 vs. Nov 11-24)
python v2/src/create_temporal_split.py

# 3. Baseline Modeling (HistGradientBoosting benchmark)
python v2/src/train_baselines.py

# 4. Explainability & Decile Diagnostics
python v2/src/explain_baseline.py
python v2/src/risk_diagnostics.py

# 5. Feature Engineering & Locked Out-of-Time Test Evaluation
python v2/src/phase5_model_improvement.py
```

---

## 6. Data Integrity & Git Discipline

This project enforces strict data versioning hygiene:
- **No Large Binaries in Git:** All datasets (`*.parquet`, `*.csv`) and model serialization files (`*.joblib`, `*.npy`) are excluded via `.gitignore`.
- **Pre-Dispatch Safety:** All engineered features are strictly verified against target leakage (no post-dispatch terminal scan statuses, actual delivery sequence indices, or routing scores).
- **Test Set Protection:** Temporal holdout test split (`data/v2/test.parquet`) is evaluated strictly once following candidate freezing.

---

## 7. License & Attribution

- **Source Code:** Released under the [MIT License](LICENSE).
- **Third-Party Dataset:** Ingests the **Amazon Last Mile Routing Research Challenge (2021)** dataset, which is governed by the [Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for full attributions.

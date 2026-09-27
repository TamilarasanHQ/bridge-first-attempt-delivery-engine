# Data Directory & Asset Management

This directory contains data files for local development and execution of the **BRIDGE — AI-Powered First Attempt Delivery Success Engine**.

## Public Repository Policy

In accordance with strict public repository security and storage limits:
- **Datasets are intentionally NOT committed to git.**
- All `.csv` and `.parquet` files are ignored by `.gitignore`.
- Raw Amazon telemetry files (which may exceed several gigabytes) must reside strictly on local storage or secure cloud buckets.

---

## Directory Layout & File Descriptions

```text
data/
├── README.md                           <- This documentation file
│
├── raw/                                <- Raw, immutable source files
│   └── model_build_inputs/
│       ├── package_data.json           <- Package dimensions, stop maps, planned service times
│       ├── route_data.json             <- Route departures, station codes, stop coordinates
│       ├── actual_sequences.json       <- Observed delivery sequences (reserved, not used)
│       └── travel_times.json.*         <- Inter-stop travel durations (reserved, not used)
│
└── processed/                          <- Generated intermediary & ML-ready artifacts
    ├── .gitkeep                        <- Preserves folder in git
    │
    ├── phase_1_4_validation_results.json
    ├── phase_1_5_ml_schema_report.md
    ├── phase_1_6_stats.json
    ├── phase_1_6_synthetic_data_report.md
    ├── phase_1_7_stats.json
    ├── phase_1_7_feature_engineering_report.md
    │
    ├── amazon_master_delivery_dataset.parquet / .csv        <- Initial Phase 1.3 merge
    ├── amazon_master_delivery_dataset_clean.parquet / .csv  <- Validated Phase 1.4 dataset
    ├── amazon_delivery_ml_dataset_v1.parquet / .csv         <- Phase 1.6 with synthetic priors & target
    └── amazon_delivery_ml_features_v1.parquet / .csv        <- Authoritative Phase 1.7 ML dataset (49 cols)
```

---

## Authoritative ML Dataset Specification

When running or preparing local data pipelines, the authoritative ML dataset is:
- `data/processed/amazon_delivery_ml_features_v1.parquet`
- `data/processed/amazon_delivery_ml_features_v1.csv`

### Verified Specifications
- **Total Records:** `1,457,175` rows (Exactly one row = one package)
- **Total Columns:** `49`
- **Candidate ML Features:** `37` (3 Categorical + 34 Numerical)
- **Target Variable:** `first_attempt_failed` (Synthetic, $0 = 90.856\%$, $1 = 9.144\%$)
- **Identifiers:** `route_id`, `stop_id`, `package_id`
- **Metadata Columns:** `delivery_date`, `departure_time`, `day_of_week`, `zone_id`, `stop_type`, `time_window_start`, `time_window_end`, `scan_status`

---

## Reproducing Datasets Locally

To regenerate the processed pipeline from the raw JSON files from scratch:

```bash
# 1. Establish Master Dataset (Phase 1.3)
python src/create_master_dataset.py

# 2. Clean & Validate Dataset (Phase 1.4)
python src/clean_and_validate.py

# 3. Generate Synthetic Priors & Target (Phase 1.6)
python src/generate_synthetic_data.py

# 4. Execute Feature Engineering & Leakage Audit (Phase 1.7)
python src/feature_engineering.py
```

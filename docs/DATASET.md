# DATASET SPECIFICATION: BRIDGE ML DATASET

This document provides the authoritative reference for the machine learning dataset used in **BRIDGE**.

---

## 1. Source Data & Provenance

The primary foundation is the **Amazon Last-Mile Routing Research Dataset** (2021), comprising 6,112 delivery routes dispatched across 17 fulfillment stations in metropolitan US regions during July–August 2018.

### Raw Inputs (`data/raw/model_build_inputs/`)
- `package_data.json` (~358 MB): Stop IDs, package dimensions, planned driver service times, scheduled appointment windows.
- `route_data.json` (~75 MB): Route IDs, station codes, departure dates, departure times, stop coordinates (lat/lng), stop zones.
- `actual_sequences.json` & `travel_times.json`: Execution-time telemetry; strictly excluded from pre-dispatch modeling to prevent leakage.

---

## 2. Dataset Evolution Across Phases

```text
Phase 1.3: Raw JSON merge -> 1,457,175 rows, 23 columns
Phase 1.4: Validation & clean -> 1,457,175 rows, 23 columns (removed redundant duration)
Phase 1.6: Synthetic priors & target -> 1,457,175 rows, 37 columns
Phase 1.7: Feature engineering & leakage audit -> 1,457,175 rows, 49 columns (FROZEN)
```

---

## 3. Authoritative 49-Column Schema (Phase 1.7)

### Identifiers (3 columns) — Excluded from ML Inputs
| Column Name | Type | Missing Count | Description |
| :--- | :--- | :--- | :--- |
| `route_id` | String | 0 | Unique route identifier (used for GroupKFold splitting). |
| `stop_id` | String | 0 | Unique stop identifier along route. |
| `package_id` | String | 0 | Primary key of parcel. |

### Target Variable (1 column) — Excluded from ML Inputs
| Column Name | Type | Distribution | Description |
| :--- | :--- | :--- | :--- |
| `first_attempt_failed` | Int64 | 0: 1,323,927 (90.86%)<br>1: 133,248 (9.14%) | Binary classification target. Generated via latent pre-dispatch logit equation. |

### Categorical Model Features (3 columns)
| Column Name | Type | Cardinality | Description |
| :--- | :--- | :--- | :--- |
| `station_code` | String | 17 | Fulfillment center hub (e.g. `DLA7`, `DBO3`). |
| `preferred_delivery_method` | String | 7 | Customer preference: `Front Door`, `Porch`, `Locker`, `Garage`, `Mailroom`, `Hand to Customer`, `Neighbor`. |
| `customer_delivery_preference` | String | 5 | Time preference: `Anytime`, `Morning Only`, `Evening Only`, `Weekday Only`, `Leave with Guard`. |

### Numerical Model Features (34 columns)
| # | Feature Name | Type | Missing % | Value Range / Description |
| :- | :--- | :--- | :--- | :--- |
| 1 | `latitude` | Float64 | 0.0% | Dropoff latitude `[30.117, 48.130]`. |
| 2 | `longitude` | Float64 | 0.0% | Dropoff longitude `[-122.574, -70.758]`. |
| 3 | `planned_service_time_seconds` | Float64 | 0.0% | Estimated dwell time budget `[0.7, 8007.0]`. |
| 4 | `depth_cm` | Float64 | 0.0% | Parcel depth in cm `[0.8, 132.1]`. |
| 5 | `height_cm` | Float64 | 0.0% | Parcel height in cm `[0.0, 83.8]`. |
| 6 | `width_cm` | Float64 | 0.0% | Parcel width in cm `[0.2, 101.6]`. |
| 7 | `package_volume_cm3` | Float64 | 0.0% | Parcel volume ($d \times h \times w$). |
| 8 | `departure_hour` | Int32 | 0.0% | Dispatch hour `[10, 18]`. |
| 9 | `is_weekend` | Int64 | 0.0% | Saturday or Sunday delivery (`0` or `1`). |
| 10 | `month` | Int32 | 0.0% | Dispatch month (`7` or `8`). |
| 11 | `time_window_duration_hours` | Float64 | 92.18% | Appointment duration in hours (missing when unassigned). |
| 12 | `customer_availability_rate` | Float64 | 0.0% | Synthetic likelihood of recipient presence `[0.10, 1.00]`. |
| 13 | `customer_previous_delivery_attempts`| Int32 | 0.0% | Historical lifetime order count `[0, 50]`. |
| 14 | `customer_previous_failed_attempts` | Int32 | 0.0% | Historical delivery failures `[0, 20]`. |
| 15 | `customer_previous_success_rate` | Float64 | 0.0% | Reliability index ($1 - \text{failed}/\text{attempts}$) `[0.0, 1.0]`. |
| 16 | `address_access_difficulty` | Int32 | 0.0% | Premises access friction rating `[1, 5]`. |
| 17 | `signature_required` | Int64 | 0.0% | Handover security protocol (`0` or `1`). |
| 18 | `historical_zone_failure_rate` | Float64 | 0.0% | Operational zone historical prior `[0.008, 0.125]`. |
| 19 | `historical_station_failure_rate` | Float64 | 0.0% | Station historical prior `[0.019, 0.040]`. |
| 20 | `seasonal_failure_rate` | Float64 | 0.0% | Seasonality operational factor `[0.015, 0.047]`. |
| 21 | `traffic_risk_score` | Float64 | 0.0% | Real-time road network risk forecast `[0.23, 0.98]`. |
| 22 | `weather_risk_score` | Float64 | 0.0% | Environmental hazard forecast `[0.05, 0.88]`. |
| 23 | `has_time_window` | Int64 | 0.0% | Binary flag: `1` if appointment window set, `0` otherwise. |
| 24 | `window_start_minutes` | Float64 | 92.18% | Appointment start in minutes from midnight `[0, 1440]`. |
| 25 | `window_end_minutes` | Float64 | 92.18% | Appointment end in minutes from midnight `[0, 1440]`. |
| 26 | `departure_to_window_start_minutes` | Float64 | 92.18% | Lead minutes from departure to appointment start. |
| 27 | `departure_to_window_end_minutes` | Float64 | 92.18% | Minutes available on route until window expiry. |
| 28 | `departure_hour_sin` | Float64 | 0.0% | $\sin(2\pi \cdot \text{departure\_hour} / 24)$. |
| 29 | `departure_hour_cos` | Float64 | 0.0% | $\cos(2\pi \cdot \text{departure\_hour} / 24)$. |
| 30 | `day_of_week_sin` | Float64 | 0.0% | $\sin(2\pi \cdot \text{day\_index} / 7)$. |
| 31 | `day_of_week_cos` | Float64 | 0.0% | $\cos(2\pi \cdot \text{day\_index} / 7)$. |
| 32 | `log_package_volume` | Float64 | 0.0% | $\ln(1 + \text{package\_volume\_cm3})$. |
| 33 | `log_planned_service_time` | Float64 | 0.0% | $\ln(1 + \text{planned\_service\_time\_seconds})$. |
| 34 | `preference_departure_mismatch` | Int64 | 0.0% | Flag indicating conflict between customer request and route time. |

### Metadata & Quarantined Columns (8 columns) — Excluded from ML Inputs
- `delivery_date`: Retained strictly for temporal splitting.
- `departure_time`: Raw departure string.
- `day_of_week`: Raw day string name.
- `zone_id`: Preserved for future training-only frequency encoding.
- `stop_type`: Constant string (`Dropoff`).
- `time_window_start`: Raw timestamp string.
- `time_window_end`: Raw timestamp string.
- `scan_status`: Terminal outcome label (quarantined against leakage).

---

## 4. Missing Values Audit Summary
- **Columns with missing values (8 columns):**
  - 5 window numeric features: `1,343,182` missing (92.18%)
  - 2 window timestamp strings: `1,343,182` missing (92.18%)
  - `zone_id`: `10,901` missing (0.75%)
- **Columns with zero missing values:** Exactly **41 columns** are 100% complete.
- **Missingness Principle:** Time-window missingness is valid operational reality. Missing values must be retained as `NaN` and handled during Phase 2 training-only preprocessing.

---

## 5. Third-Party Provenance & Licensing Distinction

### Third-Party Amazon Materials
The underlying delivery routes and package dimensional records originate from the Amazon Last Mile Routing Research Challenge dataset:
- **Copyright:** Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International Public License (CC BY-NC 4.0).
- **Ownership:** Third-party material not owned by the BRIDGE project.
- **No Redistribution:** Raw and processed Amazon dataset files are not hosted in or distributed via the public GitHub repository.
- **Non-Endorsement:** Amazon is not affiliated with, sponsoring, or endorsing the BRIDGE project.

### Project-Generated Synthetic Data & Code
- **Synthetic Features:** Customer availability, previous delivery attempt histories, previous failure rates, address access difficulty, delivery preferences, signature requirements, weather risk scores, traffic risk scores, and the `first_attempt_failed` target were generated independently by the BRIDGE project.
- **Authored Code:** All feature engineering, data transformation, validation, and modeling code authored by the BRIDGE project contributors is licensed under the [MIT License](../LICENSE).

See [THIRD_PARTY_LICENSES.md](../THIRD_PARTY_LICENSES.md) for full legal text and details.


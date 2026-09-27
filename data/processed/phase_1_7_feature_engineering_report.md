# Phase 1.7 — Feature Engineering and Leakage-Safe ML Dataset Report (Audited & Verified)

**Input Dataset:** `data/processed/amazon_delivery_ml_dataset_v1.parquet`  
**Output Datasets:**
- `data/processed/amazon_delivery_ml_features_v1.parquet`
- `data/processed/amazon_delivery_ml_features_v1.csv`  
**Prediction Target:** `first_attempt_failed` (`0` = First attempt successful, `1` = First attempt failed)  
**Row Granularity:** Exactly one row = one package  

---

## 1. Verified Dataset Dimensions & Category Counts

All 49 columns across all 1,457,175 rows were programmatically verified via assertions against `amazon_delivery_ml_features_v1.parquet`:

| Dataset Category | Verified Count | Columns in Category |
| :--- | :--- | :--- |
| **Identifiers** | **3** | `route_id`, `stop_id`, `package_id` |
| **Target Variable** | **1** | `first_attempt_failed` |
| **Categorical Model Features** | **3** | `station_code`, `preferred_delivery_method`, `customer_delivery_preference` |
| **Numerical Model Features** | **34** | 22 base/synthetic + 12 newly engineered features |
| **Metadata / Non-Model Columns** | **8** | `delivery_date`, `departure_time`, `day_of_week`, `zone_id`, `stop_type`, `time_window_start`, `time_window_end`, `scan_status` |
| **Total Columns in Dataset** | **49** | ($3 + 1 + 3 + 34 + 8 = 49$) |
| **Candidate Model Features ($X$)** | **37** | **3 Categorical + 34 Numerical** |
| **Total Rows** | **1,457,175** | Preserved exactly (0 rows dropped) |

---

## 2. Complete Inventory of Candidate Model Features (37 Columns)

### A. Categorical Model Features (3 Columns)
1. `station_code`: Fulfillment station code (17 hubs, e.g., `DLA7`, `DBO3`).
2. `preferred_delivery_method`: Delivery drop preference (7 categories, e.g., `Front Door`, `Locker`, `Hand to Customer`).
3. `customer_delivery_preference`: Scheduled customer preference (5 categories, e.g., `Morning Only`, `Anytime`).

### B. Numerical Model Features (34 Columns)

#### Physical, Spatial & Operational Base Features (11)
1. `latitude`: Continuous dropoff coordinate.
2. `longitude`: Continuous dropoff coordinate.
3. `planned_service_time_seconds`: Driver dwell time budget.
4. `depth_cm`: Physical package depth.
5. `height_cm`: Physical package height.
6. `width_cm`: Physical package width.
7. `package_volume_cm3`: Volumetric dimension ($depth \times height \times width$).
8. `departure_hour`: Scheduled dispatch hour (10 to 18).
9. `is_weekend`: Binary flag (`0` for Mon–Fri, `1` for Sat–Sun).
10. `month`: Calendar month (`7` or `8`).
11. `time_window_duration_hours`: Preserved appointment window span.

#### Synthetic Customer, Premises & Environmental Priors (11)
12. `customer_availability_rate`: Continuous likelihood of recipient being present `[0.10, 1.00]`.
13. `customer_previous_delivery_attempts`: Historical account delivery count `[0, 50]`.
14. `customer_previous_failed_attempts`: Historical failed drops `[0, 20]`.
15. `customer_previous_success_rate`: Account reliability index `[0.0, 1.0]`.
16. `address_access_difficulty`: Premises physical accessibility scale `[1, 5]`.
17. `signature_required`: Binary security protocol flag (`0` or `1`).
18. `historical_zone_failure_rate`: Operational zone risk prior `[0.008, 0.125]`.
19. `historical_station_failure_rate`: Station baseline failure rate `[0.019, 0.040]`.
20. `seasonal_failure_rate`: Seasonal calendar factor `[0.015, 0.047]`.
21. `traffic_risk_score`: Real-time road network hazard forecast `[0.0, 1.0]`.
22. `weather_risk_score`: Environmental hazard forecast `[0.0, 1.0]`.

#### Newly Engineered Features (12)
23. `has_time_window`: Binary indicator (`1` if appointment window specified, `0` if unassigned).
24. `window_start_minutes`: Start of window in minutes from midnight (`[0, 1440]`, `NaN` if unassigned).
25. `window_end_minutes`: End of window in minutes from midnight (`[0, 1440]`, `NaN` if unassigned).
26. `departure_to_window_start_minutes`: Lead minutes from route departure to appointment start (`NaN` if unassigned).
27. `departure_to_window_end_minutes`: Total available route minutes before appointment expiration (`NaN` if unassigned).
28. `departure_hour_sin`: $\sin(2\pi \cdot \text{departure\_hour} / 24)$.
29. `departure_hour_cos`: $\cos(2\pi \cdot \text{departure\_hour} / 24)$.
30. `day_of_week_sin`: $\sin(2\pi \cdot \text{day\_index} / 7)$ where Monday=0 to Sunday=6.
31. `day_of_week_cos`: $\cos(2\pi \cdot \text{day\_index} / 7)$.
32. `log_package_volume`: $\ln(1 + \text{package\_volume\_cm3})$.
33. `log_planned_service_time`: $\ln(1 + \text{planned\_service\_time\_seconds})$.
34. `preference_departure_mismatch`: Deterministic mismatch between customer requested window and departure time/day.

---

## 3. Metadata & Quarantined Non-Model Columns (8 Columns)

These 8 columns are preserved in the artifact for traceability, temporal splitting, and auditing, but are **strictly excluded** from the model feature matrix:
1. `delivery_date`: Retained strictly for chronological train/validation splitting.
2. `departure_time`: Replaced by `departure_hour`, `departure_hour_sin`, and `departure_hour_cos`.
3. `day_of_week`: Replaced by `day_of_week_sin`, `day_of_week_cos`, and `is_weekend`.
4. `zone_id`: Preserved unencoded so that frequency or out-of-fold encoders can be fitted exclusively on training splits in Phase 2.
5. `stop_type`: Constant value (`Dropoff` across 100% of package rows).
6. `time_window_start`: Replaced by numerical minute features.
7. `time_window_end`: Replaced by numerical minute features.
8. `scan_status`: Raw terminal outcome label (`DELIVERED`, `DELIVERY_ATTEMPTED`, `REJECTED`). Quarantined to prevent 100% target leakage.

---

## 4. Missing-Value Audit & Exact Arithmetic

Programmatic evaluation confirms the exact distribution of missingness across all 49 columns:

### Columns with Missing Values (8 Columns)
| Column Name | Role | Missing Count | Missing % | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| `time_window_duration_hours` | Model Feature | 1,343,182 | **92.1771%** | Standard deliveries without customer-specified appointments. |
| `window_start_minutes` | Model Feature | 1,343,182 | **92.1771%** | Unassigned appointment windows. Handled via `has_time_window = 0`. |
| `window_end_minutes` | Model Feature | 1,343,182 | **92.1771%** | Unassigned appointment windows. |
| `departure_to_window_start_minutes` | Model Feature | 1,343,182 | **92.1771%** | Unassigned appointment windows. |
| `departure_to_window_end_minutes` | Model Feature | 1,343,182 | **92.1771%** | Unassigned appointment windows. |
| `time_window_start` | Metadata Column | 1,343,182 | **92.1771%** | Raw timestamp unassigned. |
| `time_window_end` | Metadata Column | 1,343,182 | **92.1771%** | Raw timestamp unassigned. |
| `zone_id` | Metadata Column | 10,901 | **0.7481%** | Unassigned or edge delivery clusters (imputable as `'UNKNOWN'`). |

### Complete Columns with Zero Missing Values (41 Columns)
$$\text{Total Columns (49)} - \text{Columns with Missing Values (8)} = \mathbf{41\text{ Columns with ZERO Missing Values}}$$

*(All other 41 columns — including all identifiers, target, all categorical features, and remaining numerical features — have exactly 0 missing values).*

---

## 5. Leakage Audit Verification

An automated programmatic check confirmed:
1. **Forbidden Columns Verification:** The candidate model feature matrix ($X$, 37 features) contains **NONE** of:
   - `first_attempt_failed` (Target)
   - `scan_status` (Ground-truth outcome)
   - `route_id`, `stop_id`, `package_id` (Identifiers)
   - `delivery_date`, `departure_time` (Raw timestamps)
   - `stop_type` (Constant metadata)
   - `time_window_start`, `time_window_end` (Raw strings)
   - `zone_id` (Raw high-cardinality metadata)
2. **Codebase Audit:** The feature-engineering code (`src/feature_engineering.py`) references `first_attempt_failed` and `scan_status` **exclusively in column ordering and assertion verification blocks**. No feature is mathematically derived from the target or outcome labels.
3. **Zero Target Encoding:** No mean target by zone, station, or delivery method exists in this dataset.

---

## 6. Synthetic Historical Feature Audit

The features `historical_zone_failure_rate`, `historical_station_failure_rate`, and `seasonal_failure_rate` were re-audited:
- They represent **pre-dispatch operational priors** assigned via station code, zone hash, and calendar month/day indices.
- None of these features were derived from the realized target `first_attempt_failed` or observed `scan_status`.
- They reflect historical operational baselines logically accessible to the dispatcher prior to departure.

---

## 7. Correlation & Multi-Collinearity Findings

| Feature Pair | Correlation ($r$) | Redundancy Assessment & Modeling Recommendation |
| :--- | :--- | :--- |
| `package_volume_cm3` vs. `log_package_volume` | **+0.7802** | Non-linear transform. Keep original for tree-based models (XGBoost/LightGBM); use log transform for linear/distance models. |
| `planned_service_time_seconds` vs. `log_planned_service_time` | **+0.7837** | Non-linear transform. Same model-specific recommendation as above. |
| `departure_hour` vs. `departure_hour_sin` | **-0.9912** | Extreme collinearity because departure hours span only 10:00 to 18:00. Use cyclical pair for neural nets/linear models; use raw `departure_hour` for decision trees. |
| `departure_hour` vs. `departure_hour_cos` | **+0.9290** | High collinearity for same reason. Tree models should use scalar `departure_hour`. |
| `customer_previous_failed_attempts` vs. `customer_previous_success_rate`| **-0.5699** | Meaningful negative correlation without perfect collinearity due to differing total attempt denominators. |
| `customer_previous_delivery_attempts` vs. `customer_previous_failed_attempts`| **+0.4933** | Moderate positive correlation; customers with higher order volumes naturally encounter more total re-attempts. |
| `departure_to_window_start_minutes` vs. `departure_to_window_end_minutes` | **-0.5533** | Moderate correlation across appointment lead times. |
| `window_start_minutes` vs. `window_end_minutes` | **-0.0338** | Low linear correlation across the standard delivery schedule. |

---

## 8. Verified Final Column List (49 Columns)

```python
[
    # Identifiers (3)
    'route_id', 'stop_id', 'package_id',

    # Target (1)
    'first_attempt_failed',

    # Categorical Model Features (3)
    'station_code', 'preferred_delivery_method', 'customer_delivery_preference',

    # Numerical Model Features (34)
    'latitude', 'longitude', 'planned_service_time_seconds', 'depth_cm',
    'height_cm', 'width_cm', 'package_volume_cm3', 'departure_hour',
    'is_weekend', 'month', 'time_window_duration_hours',
    'customer_availability_rate', 'customer_previous_delivery_attempts',
    'customer_previous_failed_attempts', 'customer_previous_success_rate',
    'address_access_difficulty', 'signature_required',
    'historical_zone_failure_rate', 'historical_station_failure_rate',
    'seasonal_failure_rate', 'traffic_risk_score', 'weather_risk_score',
    'has_time_window', 'window_start_minutes', 'window_end_minutes',
    'departure_to_window_start_minutes', 'departure_to_window_end_minutes',
    'departure_hour_sin', 'departure_hour_cos', 'day_of_week_sin',
    'day_of_week_cos', 'log_package_volume', 'log_planned_service_time',
    'preference_departure_mismatch',

    # Metadata & Non-Model Columns (8)
    'delivery_date', 'departure_time', 'day_of_week', 'zone_id', 'stop_type',
    'time_window_start', 'time_window_end', 'scan_status'
]
```

---

## 9. Notes for the Phase 2 Training Pipeline

1. **Validation Strategy:** Always use `GroupKFold` grouped on `route_id` or temporal splitting on `delivery_date`. Packages on the same route share vehicle, weather, traffic, and driver context.
2. **Missing Time-Window Strategy:** Tree-based models (XGBoost, LightGBM, CatBoost) natively handle `NaN` in window numerical features. For linear models or neural networks, impute `NaN` with $-1$ or $0$ alongside the binary indicator `has_time_window`.
3. **Zone Frequency Fitting:** During training fold evaluation, compute `zone_frequency` **exclusively on the training split** and map it onto validation/test splits with a fallback for unseen zones.
4. **Target Isolation:** Ensure `scan_status` and `first_attempt_failed` are never included in the input feature matrix $X$.

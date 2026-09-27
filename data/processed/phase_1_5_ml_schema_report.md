# Phase 1.5 — ML Dataset Schema Report: First Attempt Delivery Success Engine (FDSE)

**Dataset Source:** `data/processed/amazon_master_delivery_dataset_clean.csv`  
**Target Variable (Planned):** `first_attempt_failed` (`0` = First attempt successful, `1` = First attempt failed)  
**Granularity:** 1 row = 1 package  

---

## 1. Categorization of All 23 Existing Columns

The 23 columns present in `amazon_master_delivery_dataset_clean.csv` are categorized based on their intended role in machine learning pipelines:

| Column Name | Category | Proposed Modeling Role | Rationale & Processing Recommendation |
| :--- | :--- | :--- | :--- |
| `route_id` | **Identifier** | Metadata / Grouping | High-cardinality routing identifier. Essential for GroupKFold validation to prevent data leakage across the same delivery shift. |
| `stop_id` | **Identifier** | Metadata / Aggregation | Stop code on the route. Used for grouping multi-package dropoffs at the same destination. |
| `package_id` | **Identifier** | Primary Key | Unique package tracking code. Not used directly in model estimation. |
| `scan_status` | **Target Source / Leakage** | Target Generation Source | Directly encodes delivery outcome (`DELIVERED`, `DELIVERY_ATTEMPTED`, `REJECTED`). **Must be excluded** from features to avoid 100% target leakage. |
| `station_code` | **Direct Feature / Categorical** | Feature (Low-cardinality Cat) | 17 delivery hubs (e.g., `DLA7`, `DBO3`). Encodes operational hub capacity, regional logistics differences, and geography. Encode via Target/One-Hot/Frequency encoding. |
| `delivery_date` | **Transformation Required** | Feature Engineering | Raw string (`YYYY-MM-DD`). Drop raw string; derive temporal distance, holidays, quarter, and temporal train/val splits. |
| `departure_time` | **Transformation Required** | Feature Engineering | Raw time string (`HH:MM:SS`). Drop raw string; use numeric `departure_hour` or minutes from midnight (`departure_time_seconds`). |
| `latitude` | **Direct Feature / Continuous** | Numerical Feature | Geographic coordinate. Useful for spatial clustering, tree models, and distance-from-station calculations. |
| `longitude` | **Direct Feature / Continuous** | Numerical Feature | Geographic coordinate. Combined with latitude for spatial coordinates and regional risk mappings. |
| `zone_id` | **Transformation Required** | Feature (High-cardinality Cat) | 8,962 operational zones. 0.75% missing. Impute missing as `'UNKNOWN'`. Apply target encoding with smoothing or out-of-fold prior failure rates. |
| `stop_type` | **Direct Feature (Zero Variance)** | Potential Drop Candidate | 100% of rows are `'Dropoff'`. Carries zero variance for packages; drop or keep as single-category constant. |
| `time_window_start` | **Transformation Required** | Feature Engineering | Raw timestamp string (`YYYY-MM-DD HH:MM:SS`). 92.18% null. Derive `has_time_window` (binary indicator) and `time_window_start_hour`. |
| `time_window_end` | **Transformation Required** | Feature Engineering | Raw timestamp string. 92.18% null. Derive `time_window_end_hour` and window alignment with route departure. |
| `planned_service_time_seconds`| **Direct Feature / Continuous** | Numerical Feature | Pre-calculated driver stop budget. High service times indicate complex buildings, long walk-ups, or bulk deliveries. |
| `depth_cm` | **Direct Feature / Continuous** | Numerical Feature | Physical package depth. Key predictor of delivery receptacle fit, customer handover necessity, and van stowage difficulty. |
| `height_cm` | **Direct Feature / Continuous** | Numerical Feature | Physical package height. 24 flat items recorded as 0.0 cm. Useful for parcel tier categorization. |
| `width_cm` | **Direct Feature / Continuous** | Numerical Feature | Physical package width. Complements volume and form factor. |
| `package_volume_cm3` | **Direct Feature / Continuous** | Numerical Feature | Derived package volumetric space ($depth \times height \times width$). High volume indicates bulky items that cannot fit in letterboxes or standard lockboxes. |
| `departure_hour` | **Direct Feature / Discrete** | Numerical / Cyclical Feature | Dispatch hour (13 to 21). Critical operational factor: late departures increase the probability of commercial building closures and evening unreachability. |
| `day_of_week` | **Direct Feature / Categorical** | Categorical Feature | Day name (Monday–Sunday). Captures consumer residential vs. commercial receipt patterns. One-hot or ordinal encode. |
| `is_weekend` | **Direct Feature / Binary** | Binary Feature (`0`/`1`) | Delivery on Saturday or Sunday. Differentiates business closures from residential availability. |
| `month` | **Direct Feature / Discrete** | Numerical / Categorical | Calendar month (July/August in raw dataset). Useful for seasonality and calendar trends. |
| `time_window_duration_hours` | **Transformation Required** | Numerical Feature | Window duration in hours. Missing for 92.18% of packages without customer-specified appointments. Impute with flag or median with `has_time_window` indicator. |

---

## 2. Summary by Schema Category

1. **Identifier Columns (3):**
   - `route_id`, `stop_id`, `package_id`
2. **Target-Source Columns (1):**
   - `scan_status`
3. **Direct ML Feature Candidates (11):**
   - `station_code`, `latitude`, `longitude`, `planned_service_time_seconds`, `depth_cm`, `height_cm`, `width_cm`, `package_volume_cm3`, `departure_hour`, `day_of_week`, `is_weekend`, `month`
4. **Columns Requiring Transformation (7):**
   - `delivery_date`, `departure_time`, `zone_id`, `stop_type`, `time_window_start`, `time_window_end`, `time_window_duration_hours`
5. **Columns That Cause Leakage If Untransformed / Kept (1):**
   - `scan_status` *(Direct outcome label)*

---

## 3. Specification of Future Synthetic Pre-Dispatch Features

The following 13 synthetic features represent real-world customer historical behaviors, premises access constraints, operational policies, and environmental risk factors available in modern logistics management systems **prior to dispatch**:

| # | Synthetic Feature Name | Data Type | Representation | Realistic Range / Categories | Pre-Dispatch Availability | Relevance to First-Attempt Delivery Failure |
| :- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `customer_availability_rate` | Numeric (Float) | Continuous | `[0.0, 1.0]` (e.g. 0.85 = 85% home availability) | Yes (Customer historical profile / telemetry) | Lower availability directly correlates with unattended delivery failure, gated lockouts, and absent recipients. |
| 2 | `customer_previous_delivery_attempts` | Numeric (Integer) | Discrete Count | `0` to `50` (typical historical lifetime deliveries) | Yes (Customer account CRM) | New accounts with 0 prior history present cold-start risk; frequent re-attempt accounts signal persistent delivery friction. |
| 3 | `customer_previous_failed_attempts` | Numeric (Integer) | Discrete Count | `0` to `15` | Yes (Customer account CRM) | Repeat delivery failures indicate difficult access, aggressive dogs, restrictive intercoms, or recurrent customer absence. |
| 4 | `customer_previous_success_rate` | Numeric (Float) | Continuous | `[0.0, 1.0]` (Default `1.0` or mean for new customers) | Yes (Customer account CRM) | Direct historical baseline of first-attempt success probability for this individual account. |
| 5 | `address_access_difficulty` | Categorical / Ordinal | Ordinal Score | `[1, 2, 3, 4, 5]` or `['Low', 'Medium', 'High', 'Severe']` | Yes (Address metadata / driver feedback notes) | Multi-unit apartments, gated communities, secured towers, or keycard-only facilities cause driver timeouts and aborted drops. |
| 6 | `preferred_delivery_method` | Categorical | Nominal | `['Front Door', 'Porch', 'Garage', 'Locker', 'Mailroom', 'Neighbor', 'Hand to Customer']` | Yes (Order checkout preference) | "Hand to Customer" requires customer presence, multiplying failure risk compared to safe drop-off options like "Front Door" or "Locker". |
| 7 | `signature_required` | Numeric (Binary) | Discrete (`0`/`1`) | `0` = No, `1` = Yes | Yes (Order / seller fulfillment constraints) | If customer is not home, the package cannot be released, leading to immediate delivery failure. |
| 8 | `customer_delivery_preference` | Categorical | Nominal | `['Anytime', 'Morning Only', 'Evening Only', 'Weekday Only', 'Leave with Guard']` | Yes (Customer profile instructions) | Discrepancies between preferred delivery hours and actual route departure hours create missed windows and failed drops. |
| 9 | `historical_zone_failure_rate` | Numeric (Float) | Continuous | `[0.0, 0.25]` (Typical 0.005 to 0.08) | Yes (Aggregated dispatch historical telemetry) | Zones with high package theft, high-density apartment blocks, or poor parking consistently experience higher delivery failure rates. |
| 10 | `historical_station_failure_rate`| Numeric (Float) | Continuous | `[0.005, 0.05]` | Yes (Station rolling operational KPIs) | Measures fulfillment center sorting error rates, local traffic congestion levels, and station-wide contractor efficiency. |
| 11 | `seasonal_failure_rate` | Numeric (Float) | Continuous | `[0.005, 0.15]` | Yes (Historical calendar index) | Back-to-school surges, Prime Day volume spikes, and seasonal summer travel reduce customer presence and increase driver fatigue. |
| 12 | `traffic_risk_score` | Numeric (Float) | Continuous Index | `[0.0, 1.0]` (0 = Clear, 1.0 = Severe gridlock) | Yes (Route planning traffic forecast) | Severe congestion delays drivers past delivery windows, forcing route truncation and scan attempts after dark. |
| 13 | `weather_risk_score` | Numeric (Float) | Continuous Index | `[0.0, 1.0]` (0 = Clear/Dry, 1.0 = Storm/Flood/Severe) | Yes (Day-of-dispatch meteorological forecast) | Extreme rainfall, snow, or heat inhibits safe unattended delivery and restricts access to unpaved roads and stairs. |

---

## 4. Planned Final ML Dataset Schema

When Phase 2 data preparation begins, the unified machine learning dataset will integrate:

$$\text{Final ML Schema} = \text{Clean Amazon Columns} + \text{Synthetic Pre-Dispatch Features} + \text{Engineered Features} + \text{Target}$$

```
========================================================================================
PLANNED ML FEATURE SCHEMA
========================================================================================

--- METADATA & GROUPING IDENTIFIERS ---
1.  route_id                            : String (Used for GroupKFold validation)
2.  stop_id                             : String (Stop grouping)
3.  package_id                          : String (Package primary key)

--- PRE-DISPATCH OPERATIONAL & TEMPORAL FEATURES ---
4.  station_code                        : Categorical (One-Hot / Target Encoded)
5.  latitude                            : Float64 (Continuous)
6.  longitude                           : Float64 (Continuous)
7.  zone_id                             : Categorical (Target Encoded / Frequency Encoded)
8.  departure_hour                      : Int32 / Float64 (Dispatch hour)
9.  day_of_week                         : Categorical (One-Hot Encoded)
10. is_weekend                          : Int64 (Binary: 0 or 1)
11. month                               : Int32 (Calendar month)
12. planned_service_time_seconds        : Float64 (Planned stop duration budget)

--- PHYSICAL PARCEL CHARACTERISTICS ---
13. depth_cm                            : Float64 (Continuous)
14. height_cm                           : Float64 (Continuous)
15. width_cm                            : Float64 (Continuous)
16. package_volume_cm3                  : Float64 (Continuous)

--- TIME WINDOW FEATURES ---
17. has_time_window                     : Int64 (Engineered Binary: 1 if window exists, 0 if null)
18. time_window_duration_hours          : Float64 (Continuous, imputed with 0 or median)
19. departure_to_tw_start_hours         : Float64 (Engineered: Gap from departure to appointment start)

--- PREMISES & CUSTOMER BEHAVIOR FEATURES (SYNTHETIC) ---
20. customer_availability_rate          : Float64 (Continuous [0.0, 1.0])
21. customer_previous_delivery_attempts : Int32 (Discrete count)
22. customer_previous_failed_attempts   : Int32 (Discrete count)
23. customer_previous_success_rate      : Float64 (Continuous [0.0, 1.0])
24. address_access_difficulty           : Int32 / Categorical (Ordinal [1-5])
25. preferred_delivery_method           : Categorical (Nominal)
26. customer_delivery_preference        : Categorical (Nominal)
27. signature_required                  : Int64 (Binary: 0 or 1)

--- MACRO, HISTORICAL & ENVIRONMENTAL RISK (SYNTHETIC) ---
28. historical_zone_failure_rate        : Float64 (Continuous)
29. historical_station_failure_rate     : Float64 (Continuous)
30. seasonal_failure_rate               : Float64 (Continuous)
31. traffic_risk_score                  : Float64 (Continuous [0.0, 1.0])
32. weather_risk_score                  : Float64 (Continuous [0.0, 1.0])

--- ENGINEERED RATIOS & STOP CONTEXT ---
33. packages_at_same_stop               : Int32 (Count of packages dropped at this stop)
34. stop_density_in_zone                : Float64 (Number of stops in zone / route duration)
35. package_volume_ratio                : Float64 (Package volume / average package volume)

--- TARGET VARIABLE ---
36. first_attempt_failed                : Int64 (0 = DELIVERED, 1 = DELIVERY_ATTEMPTED or REJECTED)
========================================================================================
```

---

## 5. Verification & Guardrails

- **Zero modification to clean datasets:** `amazon_master_delivery_dataset_clean.csv` and `amazon_master_delivery_dataset_clean.parquet` were read-only and remain completely unchanged.
- **Zero modification to raw datasets:** All files in `data/raw/` remain untouched.
- **Zero leakage guarantee:** The raw outcome field `scan_status` is explicitly flagged and isolated; it is only to be referenced when deriving the `first_attempt_failed` binary ground truth, never passed into model training.
- **Pre-dispatch integrity:** Every candidate predictor identified in this schema represents information logically known or forecasted prior to the driver leaving the fulfillment station.

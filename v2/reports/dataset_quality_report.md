# BRIDGE v2 Phase 1 — Dataset Quality Report

**Canonical Dataset:** `data/v2/bridge_v2_dataset.parquet`  
**Companion CSV:** `data/v2/bridge_v2_dataset.csv`  
**Execution Timestamp:** 2026-09-29 05:30:16  
**Row Granularity:** Exactly one row = one unique package

---

## 1. High-Level Dataset Summary

- **Total Rows (Packages):** `1,455,743`
- **Total Columns:** `26` (2 Identifiers + 1 Target + 23 Pre-Dispatch Features)
- **Unique Package IDs:** `1,455,743` ($100.0\%$ unique)
- **Duplicate Rows:** `0`
- **File Sizes:**
  - Parquet: `79.18 MB`
  - CSV: `334.35 MB`

---

## 2. Missingness Analysis

| Column Name | Data Type | Missing Count | Missing Percentage | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `package_id` | `str` | 0 | 0.00% |
| `route_id` | `str` | 0 | 0.00% |
| `first_attempt_failed` | `int8` | 0 | 0.00% |
| `planned_delivery_date` | `str` | 0 | 0.00% |
| `departure_time` | `str` | 0 | 0.00% |
| `departure_hour` | `int32` | 0 | 0.00% |
| `day_of_week` | `str` | 0 | 0.00% |
| `is_weekend` | `int8` | 0 | 0.00% |
| `month` | `int32` | 0 | 0.00% |
| `station_code` | `str` | 0 | 0.00% |
| `zone_id` | `object` | 10,856 | 0.75% |
| `stop_type` | `str` | 0 | 0.00% |
| `latitude` | `float64` | 0 | 0.00% |
| `longitude` | `float64` | 0 | 0.00% |
| `depth_cm` | `float64` | 0 | 0.00% |
| `height_cm` | `float64` | 0 | 0.00% |
| `width_cm` | `float64` | 0 | 0.00% |
| `package_volume_cm3` | `float64` | 0 | 0.00% |
| `planned_service_time_seconds` | `float64` | 0 | 0.00% |
| `has_time_window` | `int8` | 0 | 0.00% |
| `time_window_duration_hours` | `float64` | 1,342,005 | 92.19% |
| `executor_capacity_cm3` | `float64` | 0 | 0.00% |
| `planned_route_package_count` | `int32` | 0 | 0.00% |
| `planned_route_stop_count` | `int32` | 0 | 0.00% |
| `planned_route_volume_cm3` | `float64` | 0 | 0.00% |
| `route_volume_utilization` | `float64` | 0 | 0.00% |

### Missingness Findings:
1. **`zone_id` (10,856 missing / 0.75%):** Occurs for rural or peripheral stops lacking micro-zone subdivision. Handled as categorical `'MISSING'` during modeling.
2. **`time_window_duration_hours` (1,342,005 missing / 92.19%):** Exact match for packages without customer-specified time windows (`has_time_window = 0`). Handled as 0 or explicit indicator in preprocessor.
3. **All other 24 columns:** **0 missing values** ($100.0\%$ complete).

---

## 3. Range & Integrity Validations

| Feature | Min | Max | Mean | Infinite Count | Negative Count | Validated Range |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `depth_cm` | 0.8 | 132.1 | 34.08 | 0 | 0 | Physical dimensions $> 0$ |
| `height_cm` | 0.0 | 83.8 | 10.20 | 0 | 0 | 24 flat envelopes have height 0.0 cm |
| `width_cm` | 0.2 | 101.6 | 24.79 | 0 | 0 | Physical dimensions $> 0$ |
| `package_volume_cm3` | 0.0 | 300522.6 | 11074.21 | 0 | 0 | Plausible volume ($0$ to $300,523\text{ cm}^3$) |
| `planned_service_time_seconds` | 0.7 | 8007.0 | 68.20 | 0 | 0 | Dwell times $\ge 0.7\text{ s}$ |
| `latitude` | 30.1173 | 48.1299 | 38.7448 | 0 | 0 | Continental US ($30.11^\circ$ to $48.13^\circ$) |
| `longitude` | -122.5745 | -70.7577 | -105.9284 | 0 | 1,455,743 | Continental US ($-122.57^\circ$ to $-70.76^\circ$) |
| `executor_capacity_cm3` | 3114853.2 | 4247527.0 | 3599109.06 | 0 | 0 | Standard delivery vans |
| `planned_route_package_count` | 150 | 304 | 242.4 | 0 | 0 | $1$ to $545$ packages per route |
| `planned_route_stop_count` | 33 | 238 | 150.2 | 0 | 0 | $1$ to $326$ stops per route |
| `route_volume_utilization` | 0.3348 | 1.0424 | 0.7526 | 0 | 0 | Volume utilization $\le 1.95$ |

---

## 4. Temporal Coverage Analysis

- **Minimum Delivery Date:** `2018-07-19`
- **Maximum Delivery Date:** `2018-08-26`
- **Total Unique Calendar Days:** `39` days
- **Package Distribution by Month:**
- **2018-07:** 507,759 packages (34.88%)
- **2018-08:** 947,984 packages (65.12%)

*Note:* No random train/test split has been generated. Temporal splitting will be performed in Phase 2 across the calendar date boundary.

---

## 5. Natural Class Imbalance

- **Target:** `first_attempt_failed`
- **Class 0 (Successful Delivery on 1st Attempt):** `1,444,814` (99.2492%)
- **Class 1 (Failed Delivery on 1st Attempt):** `10,929` (0.7508%)
- **Imbalance Ratio:** $\approx 132.2 : 1$

*Rule Enforcement:* No artificial resampling, SMOTE, or undersampling was applied. Natural operational distribution preserved.

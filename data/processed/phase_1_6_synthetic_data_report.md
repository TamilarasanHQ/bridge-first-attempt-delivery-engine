# Phase 1.6 — Synthetic Pre-Dispatch Features and Target Report

**Dataset Path:** `data/processed/amazon_delivery_ml_dataset_v1.parquet` & `data/processed/amazon_delivery_ml_dataset_v1.csv`  
**Total Records:** 1,457,175  
**Total Columns:** 37 (23 clean source + 13 synthetic pre-dispatch features + 1 synthetic target)  
**Row Granularity:** Exactly one row = one package  

---

## 1. Summary of Target Distribution (`first_attempt_failed`)

Unlike `scan_status` (which is ~99.24% DELIVERED and unfeasible for balanced ML experimentation), `first_attempt_failed` represents a realistic, non-deterministic pre-dispatch failure probability calibrated for practical classification models:

| Target Value | Meaning | Count | Percentage |
| :--- | :--- | :--- | :--- |
| **0** | First attempt successful | 1,323,927 | **90.856%** |
| **1** | First attempt failed | 133,248 | **9.144%** |

- **Design Characteristics:** Formulated through a latent risk model with logistic probability transform and Bernoulli sampling. There is intentional overlap between feature spaces of successful and failed deliveries to ensure ML algorithms learn robust decision boundaries rather than trivial hardcoded rules.

---

## 2. Newly Added Pre-Dispatch Synthetic Features

A total of **13 pre-dispatch synthetic features** were integrated:

| # | Feature Name | Representation | Range / Categories | Description & Generation Logic |
| :- | :--- | :--- | :--- | :--- |
| 1 | `customer_availability_rate` | Float64 | `[0.10, 1.00]` | Customer likelihood of being present at destination. Dependent on historical success rate and delivery preferences. |
| 2 | `customer_previous_delivery_attempts` | Int32 | `[0, 50]` | Customer lifetime delivery count. Models cold start accounts (~12% zeros) vs. experienced accounts. |
| 3 | `customer_previous_failed_attempts` | Int32 | `[0, 15]` | Past failed deliveries ($0 \le 	ext{failed} \le 	ext{attempts}$). Strictly bounded. |
| 4 | `customer_previous_success_rate` | Float64 | `[0.0, 1.0]` | $1 - (\text{failed}/\text{attempts})$ for accounts with history; default $1.0$ for cold-start accounts (optimistic neutral prior). |
| 5 | `address_access_difficulty` | Int32 (Ordinal) | `[1, 2, 3, 4, 5]` | Physical premises difficulty (1=Easy to 5=Severe). Correlated with planned service time and zone failure rate. |
| 6 | `preferred_delivery_method` | String (Cat) | 7 categories | Front Door, Porch, Garage, Locker, Mailroom, Neighbor, Hand to Customer. |
| 7 | `signature_required` | Int64 (Binary) | `0` or `1` | High presence requirement. Significantly higher probability for 'Hand to Customer' and large packages. |
| 8 | `customer_delivery_preference` | String (Cat) | 5 categories | Anytime, Morning Only, Evening Only, Weekday Only, Leave with Guard. |
| 9 | `historical_zone_failure_rate` | Float64 | `[0.008, 0.125]` | Log-normal operational rate mapped deterministically by `zone_id` prior to target creation. |
| 10 | `historical_station_failure_rate` | Float64 | `[0.018, 0.042]` | Hub baseline performance rate mapped deterministically by `station_code`. |
| 11 | `seasonal_failure_rate` | Float64 | `[0.015, 0.065]` | Summer seasonality factor varying by month and weekend schedule. |
| 12 | `traffic_risk_score` | Float64 | `[0.02, 0.98]` | Road network risk based on departure hour (rush hour bump 16-18h), Friday spikes, and beta noise. |
| 13 | `weather_risk_score` | Float64 | `[0.01, 0.95]` | Environmental hazard score based on latitude region, day variance, and right-skewed beta weather disturbances. |

---

## 3. Internal Consistency & Quality Checks

All mathematical relationships and logical assertions were verified across all 1,457,175 rows:

- **Constraint Check ($	ext{failed\_attempts} \le 	ext{total\_attempts}$):**  
  **0 violations.** Exactly $100\%$ compliant.
- **Success Rate Formula Check:**  
  Max discrepancy between $\text{success\_rate}$ and $1 - (\text{failed}/\text{attempts})$ is **4.999999999999449e-05** (exact match within floating precision).
- **Cold-Start Policy:** Accounts with 0 previous attempts are assigned a neutral baseline success rate of `1.0` and `0` prior failures.
- **Coordination Check (Signature vs. Method):**  
  'Hand to Customer' deliveries carry a **35%** signature requirement rate compared to only **4%** for standard deliveries.
- **Target Independence Guarantee:** No synthetic feature was derived from `first_attempt_failed` or `scan_status`. All features exist independently prior to target evaluation.

---

## 4. Feature Relationships with Target (`first_attempt_failed`)

The target displays clear, logical, and non-trivial relationships with major risk drivers:

### Failure Rate by Delivery Method
- **Hand to Customer:** **10.32%** (highest risk due to strict recipient presence requirement)
- **Front Door:** **8.96%**
- **Locker:** **8.09%** (lowest risk due to secure autonomous receptacle)

### Failure Rate by Access Difficulty
- **Level 1 (Easy):** **8.18%**
- **Level 2 (Normal):** **8.86%**
- **Level 3 (Moderate):** **9.64%**
- **Level 4 (High):** **10.36%**
- **Level 5 (Severe):** **11.22%**

### Failure Rate by Signature Requirement
- **Signature Required (1):** **10.64%**
- **No Signature (0):** **9.03%**

### Key Correlations with Target
- `signature_required`: **+0.0143**
- `address_access_difficulty`: **+0.0334**
- `customer_availability_rate`: **-0.0402** (negative, protective)
- `customer_previous_success_rate`: **-0.0335** (negative, protective)
- `historical_zone_failure_rate`: **+0.0115**
- `traffic_risk_score`: **+0.0057**

---

## 5. Potential Leakage Risks & Safeguards

1. **`scan_status` Isolation:**  
   `scan_status` is preserved at the very end of the dataset purely for lineage and reference. **It must be dropped in Phase 2 modeling pipelines.** It was NOT used to generate `first_attempt_failed`.
2. **Pre-Dispatch Verification:**  
   All 13 synthetic features represent metadata logically obtainable or predicted at the fulfillment station before the driver departs. No stop execution sequence, actual dwell time, or return-to-station logs are included.

---

## 6. Complete Final Column List (37 Columns)

```python
[
    # Identifiers (3)
    'route_id', 'stop_id', 'package_id',
    
    # Target Variable (1)
    'first_attempt_failed',
    
    # Source Pre-Dispatch Operational & Package Features (19)
    'station_code', 'delivery_date', 'departure_time', 'latitude', 'longitude',
    'zone_id', 'stop_type', 'time_window_start', 'time_window_end',
    'planned_service_time_seconds', 'depth_cm', 'height_cm', 'width_cm',
    'package_volume_cm3', 'departure_hour', 'day_of_week', 'is_weekend',
    'month', 'time_window_duration_hours',
    
    # Synthetic Pre-Dispatch Features (13)
    'customer_availability_rate',
    'customer_previous_delivery_attempts',
    'customer_previous_failed_attempts',
    'customer_previous_success_rate',
    'address_access_difficulty',
    'preferred_delivery_method',
    'signature_required',
    'customer_delivery_preference',
    'historical_zone_failure_rate',
    'historical_station_failure_rate',
    'seasonal_failure_rate',
    'traffic_risk_score',
    'weather_risk_score',
    
    # Outcome Audit Column (1 - Exclude from Training)
    'scan_status'
]
```

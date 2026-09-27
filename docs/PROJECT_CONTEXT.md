# PROJECT CONTEXT: BRIDGE — AI-POWERED FIRST ATTEMPT DELIVERY SUCCESS ENGINE

## 1. Project Mission & Overview

BRIDGE is an artificial intelligence decision-support platform designed to predict and mitigate **first-attempt last-mile delivery failure before driver dispatch**. In urban and suburban parcel delivery logistics, failed initial delivery attempts account for significant operational waste, heightened vehicle emissions, increased driver fatigue, and poor customer experience.

The core goal of BRIDGE is to evaluate parcels at the fulfillment center while routes are being staged, predict the risk of delivery failure, and route high-risk parcels through automated interventions (such as proactive customer SMS notifications, delivery preference adjustments, or safe-place authorizations) **before** the van departs.

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

## 2. Technical Stack & Component Roles

- **ML Predictive Layer:** Predicts calibrated failure probability $P(\text{failure} \mid \mathbf{x}_{\text{pre-dispatch}})$ using Scikit-Learn / Gradient Boosted trees.
- **Risk Decision Engine (`risk_engine.py`):** Decoupled business rule and risk banding component mapping continuous probabilities into operational intervention actions (`LOW`, `MEDIUM`, `HIGH`).
- **Application API Layer (`server.py`):** FastAPI asynchronous web service providing single-package and batch inference endpoints.
- **Presentation Layer (`static/`):** Web-based dashboard for dispatchers and route supervisors.

---

## 3. Strict Machine Learning & Data Leakage Rules

1. **Target Isolation:** The target `first_attempt_failed` must **never** be used in feature engineering, normalization, or model inputs.
2. **Terminal Status Quarantined:** `scan_status` is the observed real-world outcome (`DELIVERED`, `DELIVERY_ATTEMPTED`, `REJECTED`). It must **never** enter any model pipeline.
3. **Identifiers Excluded:** `route_id`, `stop_id`, and `package_id` are primary keys and identifiers; they are excluded from the model feature matrix $X$.
4. **Validation Integrity:** 
   - `route_id` is reserved **strictly for group-aware cross-validation** (`GroupKFold`) because deliveries on the same route share vehicle space, driver speed, local weather, and traffic.
   - `delivery_date` is reserved **strictly for temporal train/test splitting**.
5. **No Full-Dataset Fitting:** All preprocessors, scalers, imputers, and future zone frequency mappings must be fitted **exclusively on the training split** and applied out-of-fold to validation and test splits.
6. **Pre-Dispatch Assumption:** `planned_service_time_seconds` is assumed to be an algorithmic expectation set by the routing system before departure. If future operational evidence indicates it is adjusted post-departure, it must be removed.

---

## 4. Authoritative Phase 1.7 Feature Schema (49 Columns)

The frozen Phase 1.7 dataset (`amazon_delivery_ml_features_v1.parquet` / `.csv`) contains 1,457,175 rows and exactly 49 columns:

```text
IDENTIFIERS (3):
- route_id
- stop_id
- package_id

TARGET (1):
- first_attempt_failed (Synthetic, 0: 90.856%, 1: 9.144%)

CATEGORICAL MODEL FEATURES (3):
- station_code
- preferred_delivery_method
- customer_delivery_preference

NUMERICAL MODEL FEATURES (34):
- latitude
- longitude
- planned_service_time_seconds
- depth_cm
- height_cm
- width_cm
- package_volume_cm3
- departure_hour
- is_weekend
- month
- time_window_duration_hours
- customer_availability_rate
- customer_previous_delivery_attempts
- customer_previous_failed_attempts
- customer_previous_success_rate
- address_access_difficulty
- signature_required
- historical_zone_failure_rate
- historical_station_failure_rate
- seasonal_failure_rate
- traffic_risk_score
- weather_risk_score
- has_time_window
- window_start_minutes
- window_end_minutes
- departure_to_window_start_minutes
- departure_to_window_end_minutes
- departure_hour_sin
- departure_hour_cos
- day_of_week_sin
- day_of_week_cos
- log_package_volume
- log_planned_service_time
- preference_departure_mismatch

METADATA / QUARANTINED NON-MODEL COLUMNS (8):
- delivery_date
- departure_time
- day_of_week
- zone_id
- stop_type
- time_window_start
- time_window_end
- scan_status
```

---

## 5. Synthetic Data Disclosure & Research Prototype Boundaries

- **Why Synthetic Data was Necessary:** The Amazon Last-Mile Routing Research dataset provides package dimensions, planned service times, station codes, and coordinates, but lacks customer CRM history (previous failure rates, account age, delivery instructions), premises difficulty ratings, real-time traffic indices, and weather scores. Furthermore, the raw `scan_status` has an extreme ~0.76% failure rate, which represents an uncalibrated signal for training ML failure classifiers without customer-side context.
- **Synthetic Features:** 13 realistic operational priors were generated using domain-informed constraints (e.g., failed attempts $\le$ total attempts, signature requirements correlated with 'Hand to Customer', traffic correlated with evening departure hours).
- **Synthetic Target:** `first_attempt_failed` was derived through a latent-logit risk equation with a logistic sigmoid transformation and Bernoulli sampling, providing a realistic 9.144% failure class.
- **Public Claim Boundaries:** BRIDGE is a **hackathon prototype**. Model metrics do NOT represent real-world Amazon operational performance.

# BRIDGE v2 Phase 1 — Feature Leakage Audit

**Audit Timestamp:** 2026-09-29 05:30:16  
**Objective:** Strict pre-dispatch boundary verification for BRIDGE v2.

---

## 1. Pre-Dispatch Information Boundary Principle

Every feature admitted into `data/v2/bridge_v2_dataset.parquet` must satisfy the **Pre-Dispatch Invariant**:
> *"The feature must represent information that is logically known at the fulfillment station before the driver departs for the first delivery stop."*

```
[Dispatch Planning] -------> [Station Departure] -------> [En Route Delivery] -------> [Shift End]
         |                            |                           |                          |
   (Category A)                  (DISPATCH)                  (Category B)               (Category B)
   Approved Pre-Dispatch                                    STRICT LEAKAGE             STRICT LEAKAGE
   - Parcel dimensions                                      - scan_status              - Return scans
   - Service time plan                                      - actual_sequences         - Late retries
   - Station / Zone                                         - seq_idx                  - Shift duration
   - Scheduled window                                       - Actual dwell time
```

---

## 2. Feature Classification Table

| Feature Name | Source Field(s) | Derived or Raw | Availability Stage | Allowed for Model | Rationale & Leakage Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `package_id` | `package_data.json` | Raw | `PRE_DISPATCH` | **Identifier Only** | Unique entity key; never used as an explanatory ML feature. |
| `route_id` | `package_data.json` | Raw | `PRE_DISPATCH` | **Identifier Only** | Route key; reserved for group K-Fold and spatial-temporal splitting. |
| `planned_delivery_date` | `route_data.json` | Raw | `PRE_DISPATCH` | **YES** | Scheduled date on shipping manifest. Safe. |
| `departure_time` | `route_data.json` | Raw | `PRE_DISPATCH` | **YES** | Scheduled depot departure time in UTC. Safe. |
| `departure_hour` | `departure_time_utc` | Derived | `PRE_DISPATCH` | **YES** | Clock hour of scheduled departure (0-23). Safe. |
| `day_of_week` | `date_YYYY_MM_DD` | Derived | `PRE_DISPATCH` | **YES** | Calendar day of dispatch (Monday-Sunday). Safe. |
| `is_weekend` | `date_YYYY_MM_DD` | Derived | `PRE_DISPATCH` | **YES** | Binary weekend indicator. Safe. |
| `month` | `date_YYYY_MM_DD` | Derived | `PRE_DISPATCH` | **YES** | Calendar month (July/August). Safe. |
| `station_code` | `route_data.json` | Raw | `PRE_DISPATCH` | **YES** | Origin fulfillment warehouse code (17 hubs). Safe. |
| `zone_id` | `stops` | Raw | `PRE_DISPATCH` | **YES** | Micro-geographic route cluster code. Safe. |
| `stop_type` | `stops` | Raw | `PRE_DISPATCH` | **YES** | Planned stop role (`Dropoff`). Safe. |
| `latitude` | `stops` | Raw | `PRE_DISPATCH` | **YES** | Customer address geocoded dropoff latitude. Safe. |
| `longitude` | `stops` | Raw | `PRE_DISPATCH` | **YES** | Customer address geocoded dropoff longitude. Safe. |
| `depth_cm` | `dimensions` | Raw | `PRE_DISPATCH` | **YES** | Automated parcel cubing scanner measurement. Safe. |
| `height_cm` | `dimensions` | Raw | `PRE_DISPATCH` | **YES** | Automated parcel cubing scanner measurement. Safe. |
| `width_cm` | `dimensions` | Raw | `PRE_DISPATCH` | **YES** | Automated parcel cubing scanner measurement. Safe. |
| `package_volume_cm3` | `dimensions` | Derived | `PRE_DISPATCH` | **YES** | Cubic parcel volume ($D \times H \times W$). Safe. |
| `planned_service_time_seconds` | `package_data.json` | Raw | `PRE_DISPATCH` | **YES** | Pre-computed algorithmic dwell estimate. Safe. |
| `has_time_window` | `time_window` | Derived | `PRE_DISPATCH` | **YES** | Binary indicator if customer SLA window was contracted. Safe. |
| `time_window_duration_hours` | `time_window` | Derived | `PRE_DISPATCH` | **YES** | Promised SLA window width in hours. Safe. |
| `executor_capacity_cm3` | `route_data.json` | Raw | `PRE_DISPATCH` | **YES** | Physical cargo volume capacity of delivery van. Safe. |
| `planned_route_package_count` | `package_data.json` | Derived | `PRE_DISPATCH` | **YES** | Total parcels assigned to route at dispatch. Safe. |
| `planned_route_stop_count` | `package_data.json` | Derived | `PRE_DISPATCH` | **YES** | Total unique stops assigned to route at dispatch. Safe. |
| `planned_route_volume_cm3` | `package_data.json` | Derived | `PRE_DISPATCH` | **YES** | Total cubic volume of all parcels on route. Safe. |
| `route_volume_utilization` | Aggregates | Derived | `PRE_DISPATCH` | **YES** | Ratio of parcel volume to vehicle capacity. Safe. |
| `scan_status` | `package_data.json` | Raw | `TARGET_DERIVED` | **NO (FORBIDDEN)** | Direct post-dispatch outcome. Target leakage. |
| `actual_sequences` | `actual_sequences.json` | Raw | `POST_DISPATCH` | **NO (FORBIDDEN)** | Driver's dynamic execution order. Unseen before dispatch. |
| `seq_idx` | `actual_sequences.json` | Derived | `POST_DISPATCH` | **NO (FORBIDDEN)** | Sequence rank. Used ONLY for sorting target, NEVER in features. |
| `route_score` | `route_data.json` | Raw | `UNKNOWN` | **NO (EXCLUDED)** | Ambiguous whether pre-dispatch driver tier or post-hoc grade. |
| `travel_times.json` | Raw matrix | Raw | `UNKNOWN` | **NO (EXCLUDED)** | Static matrix; excluded to avoid implicit route path leakage. |
| *All 13 v1 synthetic features* | `v1 pipeline` | Synthetic | `SYNTHETIC` | **NO (ELIMINATED)** | Synthetic constructs (`signature_required`, etc.) do not exist in raw data. |

---

## 3. Leakage Verification Checklist

- [x] `scan_status` confirmed completely absent from feature columns.
- [x] `actual_sequences` and `seq_idx` confirmed completely absent from feature columns.
- [x] No post-dispatch dwell time or arrival timestamp included.
- [x] No retry indicators or multi-stop visit counts included in feature set.
- [x] All 13 synthetic customer variables eliminated.
- [x] Total approved explanatory modeling features: **23 features**.

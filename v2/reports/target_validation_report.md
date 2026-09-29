# BRIDGE v2 Phase 1 — Target Validation Report

**Dataset Path:** `data/v2/bridge_v2_dataset.parquet`  
**Execution Timestamp:** 2026-09-29 05:30:16  
**Source Dataset:** `C:\Users\Ksari\amazon_dataset\almrrc2021\almrrc2021-data-training\model_build_inputs`  
**Unit of Observation:** Exactly one row = one unique package (`package_id`)

---

## 1. Real Target Definition & Methodology

The BRIDGE v2 modeling target (`first_attempt_failed`) is a binary operational outcome derived directly from raw driver barcode telemetry in `package_data.json` and ordered chronologically by `actual_sequences.json`.

$$\text{first\_attempt\_failed} = \begin{cases} 1 & \text{if chronological first physical scan is } \texttt{DELIVERY\_ATTEMPTED} \text{ or } \texttt{REJECTED} \\ 0 & \text{if chronological first physical scan is } \texttt{DELIVERY\_ATTEMPTED} \end{cases}$$

### Chronological Ordering Rule
- Out of 1,455,743 unique packages, 1,454,321 (99.902%) appear at exactly one stop.
- For the remaining 1,422 multi-stop packages, their delivery attempts are ordered strictly by the stop's integer sequence index (`seq_idx`) in `actual_sequences.json`.
- There are **0 ties** in `seq_idx`. The sequence index is used solely for chronological ordering to identify the first attempt and is **strictly excluded from model features** to prevent post-dispatch leakage.

---

## 2. Quantitative Target Distribution

| Outcome Category | Raw Scan Status | Final Target | Package Count | Percentage |
| :--- | :--- | :--- | :--- | :--- |
| **Successful 1st Attempt** | `DELIVERED` | `first_attempt_failed = 0` | **1,444,814** | **99.2492%** |
| **Failed 1st Attempt (Retry/Unresolved)** | `DELIVERY_ATTEMPTED` | `first_attempt_failed = 1` | **10,897** | **0.7486%** |
| **Failed 1st Attempt (Customer Rejection)** | `REJECTED` | `first_attempt_failed = 1` | **32** | **0.0022%** |
| **Total Valid Packages** | — | — | **1,455,743** | **100.0000%** |
| **Unresolved / Excluded** | None | — | **0** | **0.0000%** |

### Key Metrics
- **Total Positive Failures ($Y=1$):** `10,929`
- **Total Negative Successes ($Y=0$):** `1,444,814`
- **Observed Real Failure Rate:** **`0.7508%`** ($pprox 1\text{ in }133\text{ packages}$)

---

## 3. Comparison with Audit Reference

| Audit Parameter | Pre-Audit Reference | Reconstructed v2 Result | Discrepancy | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Total Unique Packages** | 1,455,743 | 1,455,743 | **0** | **PERFECT MATCH** |
| **First-Attempt Failures** | 10,929 | 10,929 | **0** | **PERFECT MATCH** |
| **First-Attempt Successes** | 1,444,814 | 1,444,814 | **0** | **PERFECT MATCH** |
| **Observed Failure Rate** | 0.7508% | 0.7508% | **0.0000%** | **PERFECT MATCH** |
| **Same-Day Retry Cases** | 1,313 | 1,313 | **0** | **PERFECT MATCH** |

---

## 4. Physical Retry Verification (`DELIVERY_ATTEMPTED -> DELIVERED`)

In the raw dataset, exactly **1,313 packages** exhibit an initial `DELIVERY_ATTEMPTED` scan followed later on the same route by a successful `DELIVERED` scan.

- **Significance:** This provides indisputable empirical confirmation that `DELIVERY_ATTEMPTED` represents a genuine unfulfilled delivery attempt where the driver was forced to circle back later in the shift.
- **Handling:** Under Rule 1, these 1,313 packages are correctly assigned `first_attempt_failed = 1` because their initial attempt failed.

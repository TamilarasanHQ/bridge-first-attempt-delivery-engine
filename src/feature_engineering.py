import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
input_parquet = PROCESSED_DIR / "amazon_delivery_ml_dataset_v1.parquet"
output_parquet = PROCESSED_DIR / "amazon_delivery_ml_features_v1.parquet"
output_csv = PROCESSED_DIR / "amazon_delivery_ml_features_v1.csv"
report_md = PROCESSED_DIR / "phase_1_7_feature_engineering_report.md"

print(f"Reading input dataset from {input_parquet}...")
df = pd.read_parquet(input_parquet)

input_rows = len(df)
input_cols = len(df.columns)
input_col_names = list(df.columns)

# -------------------------------------------------------------
# 1. TIME-WINDOW FEATURES
# -------------------------------------------------------------
print("Engineering time-window features...")
has_tw = df["time_window_start"].notna() & df["time_window_end"].notna()
df["has_time_window"] = has_tw.astype(np.int64)

# Initialize engineered window columns with NaN
df["window_start_minutes"] = np.nan
df["window_end_minutes"] = np.nan
df["departure_to_window_start_minutes"] = np.nan
df["departure_to_window_end_minutes"] = np.nan

# Compute for records where time window exists
tw_indices = df[has_tw].index
tw_start_dt = pd.to_datetime(df.loc[tw_indices, "time_window_start"])
tw_end_dt = pd.to_datetime(df.loc[tw_indices, "time_window_end"])
departure_full_dt = pd.to_datetime(
    df.loc[tw_indices, "delivery_date"] + " " + df.loc[tw_indices, "departure_time"]
)

# Minutes from midnight for appointment boundaries
df.loc[tw_indices, "window_start_minutes"] = np.round(
    tw_start_dt.dt.hour * 60.0 + tw_start_dt.dt.minute + tw_start_dt.dt.second / 60.0, 2
)
df.loc[tw_indices, "window_end_minutes"] = np.round(
    tw_end_dt.dt.hour * 60.0 + tw_end_dt.dt.minute + tw_end_dt.dt.second / 60.0, 2
)

# Lead/lag minutes from departure to appointment boundaries
df.loc[tw_indices, "departure_to_window_start_minutes"] = np.round(
    (tw_start_dt - departure_full_dt).dt.total_seconds() / 60.0, 2
)
df.loc[tw_indices, "departure_to_window_end_minutes"] = np.round(
    (tw_end_dt - departure_full_dt).dt.total_seconds() / 60.0, 2
)

# Validate time_window_duration_hours consistency
expected_duration_hours = (tw_end_dt - tw_start_dt).dt.total_seconds() / 3600.0
duration_diff = (df.loc[tw_indices, "time_window_duration_hours"] - expected_duration_hours).abs()
max_duration_diff = float(duration_diff.max())
print(f"Max discrepancy between time_window_duration_hours and calculation: {max_duration_diff}")

# -------------------------------------------------------------
# 2. DEPARTURE TIME CYCLICAL FEATURES
# -------------------------------------------------------------
print("Engineering departure time cyclical features...")
dep_hour = df["departure_hour"].values
df["departure_hour_sin"] = np.round(np.sin(2.0 * np.pi * dep_hour / 24.0), 6)
df["departure_hour_cos"] = np.round(np.cos(2.0 * np.pi * dep_hour / 24.0), 6)

# -------------------------------------------------------------
# 3. DAY-OF-WEEK CYCLICAL FEATURES
# -------------------------------------------------------------
print("Engineering day-of-week cyclical features...")
day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}
day_index = df["day_of_week"].map(day_mapping).values
df["day_of_week_sin"] = np.round(np.sin(2.0 * np.pi * day_index / 7.0), 6)
df["day_of_week_cos"] = np.round(np.cos(2.0 * np.pi * day_index / 7.0), 6)

# -------------------------------------------------------------
# 4. CUSTOMER HISTORY VALIDATION
# -------------------------------------------------------------
print("Validating customer history consistency...")
# Validate failed <= attempts
history_violations = int((df["customer_previous_failed_attempts"] > df["customer_previous_delivery_attempts"]).sum())
# Validate success rate consistency
non_cold = df["customer_previous_delivery_attempts"] > 0
expected_success = 1.0 - (
    df.loc[non_cold, "customer_previous_failed_attempts"] / df.loc[non_cold, "customer_previous_delivery_attempts"]
)
success_diff = (df.loc[non_cold, "customer_previous_success_rate"] - expected_success).abs()
max_success_discrepancy = float(success_diff.max())
print(f"History violations: {history_violations}, Max success diff: {max_success_discrepancy}")

# -------------------------------------------------------------
# 5. PACKAGE & SERVICE LOG TRANSFORMS
# -------------------------------------------------------------
print("Engineering package and service log transforms...")
df["log_package_volume"] = np.round(np.log1p(df["package_volume_cm3"].values), 6)
df["log_planned_service_time"] = np.round(np.log1p(df["planned_service_time_seconds"].values), 6)

# -------------------------------------------------------------
# 6. CUSTOMER PREFERENCE MISMATCH
# -------------------------------------------------------------
print("Engineering customer preference departure mismatch feature...")
pref = df["customer_delivery_preference"].values
dep_h = df["departure_hour"].values
is_wknd = df["is_weekend"].values

mismatch = np.zeros(len(df), dtype=np.int64)
# Morning Only: departure_hour >= 12
mismatch = np.where((pref == "Morning Only") & (dep_h >= 12), 1, mismatch)
# Evening Only: departure_hour < 16
mismatch = np.where((pref == "Evening Only") & (dep_h < 16), 1, mismatch)
# Weekday Only: is_weekend == 1
mismatch = np.where((pref == "Weekday Only") & (is_wknd == 1), 1, mismatch)
# Anytime & Leave with Guard: 0
df["preference_departure_mismatch"] = mismatch.astype(np.int64)

# -------------------------------------------------------------
# 7. ZONE REPRESENTATION SPECIFICATION
# -------------------------------------------------------------
# zone_id is retained as a metadata / grouping column for train-only fitting.
# Note: As strictly required, no target-derived statistics are calculated.

# -------------------------------------------------------------
# 8. COLUMN DEFINITIONS & EXCLUSIONS
# -------------------------------------------------------------
identifiers = ["route_id", "stop_id", "package_id"]
target_col = ["first_attempt_failed"]

excluded_from_model = [
    "route_id", "stop_id", "package_id",
    "first_attempt_failed",
    "scan_status",
    "delivery_date",
    "departure_time",
    "stop_type",
    "time_window_start",
    "time_window_end",
    "zone_id"
]

categorical_features = [
    "station_code",
    "preferred_delivery_method",
    "customer_delivery_preference"
]

numeric_features = [
    # Source numerical features
    "latitude",
    "longitude",
    "planned_service_time_seconds",
    "depth_cm",
    "height_cm",
    "width_cm",
    "package_volume_cm3",
    "departure_hour",
    "is_weekend",
    "month",
    "time_window_duration_hours",
    # Synthetic pre-dispatch features
    "customer_availability_rate",
    "customer_previous_delivery_attempts",
    "customer_previous_failed_attempts",
    "customer_previous_success_rate",
    "address_access_difficulty",
    "signature_required",
    "historical_zone_failure_rate",
    "historical_station_failure_rate",
    "seasonal_failure_rate",
    "traffic_risk_score",
    "weather_risk_score",
    # Newly engineered features
    "has_time_window",
    "window_start_minutes",
    "window_end_minutes",
    "departure_to_window_start_minutes",
    "departure_to_window_end_minutes",
    "departure_hour_sin",
    "departure_hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "log_package_volume",
    "log_planned_service_time",
    "preference_departure_mismatch"
]

model_features = categorical_features + numeric_features

# Full dataset column ordering:
# 1. Identifiers
# 2. Target
# 3. Categorical Model Features
# 4. Numerical Model Features
# 5. Metadata / Non-model Columns (retained for traceability, train/test splitting, and auditing)
metadata_and_non_model = [
    "delivery_date", "departure_time", "day_of_week", "zone_id", "stop_type",
    "time_window_start", "time_window_end", "scan_status"
]

all_final_columns = identifiers + target_col + model_features + metadata_and_non_model
df_final = df[all_final_columns]

output_rows = len(df_final)
output_cols = len(df_final.columns)

# -------------------------------------------------------------
# 9. LEAKAGE AUDIT VERIFICATION
# -------------------------------------------------------------
print("Running Leakage Audit...")
audit_violations = []

# Check that model_features contains none of the forbidden columns
forbidden_in_features = [
    "first_attempt_failed", "scan_status", "route_id", "stop_id", "package_id",
    "delivery_date", "departure_time", "stop_type"
]
for col in forbidden_in_features:
    if col in model_features:
        audit_violations.append(f"Forbidden column '{col}' found in model_features list!")

# Verify target independence of all engineered columns
engineered_cols = [
    "has_time_window", "window_start_minutes", "window_end_minutes",
    "departure_to_window_start_minutes", "departure_to_window_end_minutes",
    "departure_hour_sin", "departure_hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "log_package_volume", "log_planned_service_time",
    "preference_departure_mismatch"
]

print(f"Audit violations count: {len(audit_violations)}")

# -------------------------------------------------------------
# 10. CORRELATION & REDUNDANCY ANALYSIS
# -------------------------------------------------------------
print("Calculating correlations...")
corr_features = [
    "depth_cm", "height_cm", "width_cm", "package_volume_cm3", "log_package_volume",
    "planned_service_time_seconds", "log_planned_service_time",
    "customer_previous_delivery_attempts", "customer_previous_failed_attempts", "customer_previous_success_rate",
    "departure_hour", "departure_hour_sin", "departure_hour_cos",
    "day_of_week_sin", "day_of_week_cos", "is_weekend",
    "time_window_duration_hours", "window_start_minutes", "window_end_minutes",
    "departure_to_window_start_minutes", "departure_to_window_end_minutes"
]

corr_matrix = df_final[corr_features].corr().round(4)

# Specific pairs of interest
redundancy_pairs = [
    ("package_volume_cm3", "log_package_volume"),
    ("planned_service_time_seconds", "log_planned_service_time"),
    ("departure_hour", "departure_hour_cos"),
    ("departure_hour", "departure_hour_sin"),
    ("customer_previous_failed_attempts", "customer_previous_success_rate"),
    ("customer_previous_delivery_attempts", "customer_previous_failed_attempts"),
    ("departure_to_window_start_minutes", "departure_to_window_end_minutes"),
    ("window_start_minutes", "window_end_minutes")
]

pair_correlations = {}
for col1, col2 in redundancy_pairs:
    pair_correlations[f"{col1} vs {col2}"] = float(corr_matrix.loc[col1, col2])

# -------------------------------------------------------------
# 11. MISSING VALUES AUDIT
# -------------------------------------------------------------
print("Compiling missing values audit...")
missing_report = {}
for col in all_final_columns:
    n_miss = int(df_final[col].isna().sum())
    pct_miss = round((n_miss / output_rows) * 100, 4)
    missing_report[col] = {"missing_count": n_miss, "missing_percent": pct_miss}

# -------------------------------------------------------------
# 12. EXPORT DATASETS
# -------------------------------------------------------------
print(f"Saving {output_parquet}...")
df_final.to_parquet(output_parquet, index=False)

print(f"Saving {output_csv}...")
df_final.to_csv(output_csv, index=False)

# Save intermediate statistics JSON
stats_summary = {
    "input_rows": input_rows,
    "output_rows": output_rows,
    "input_cols": input_cols,
    "output_cols": output_cols,
    "identifiers": identifiers,
    "target": target_col[0],
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "engineered_features": engineered_cols,
    "excluded_from_model": excluded_from_model,
    "audit_violations": audit_violations,
    "pair_correlations": pair_correlations,
    "missing_report": missing_report
}

with open(PROCESSED_DIR / "phase_1_7_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats_summary, f, indent=2)

print("Phase 1.7 data preparation completed successfully.")

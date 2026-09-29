import json
from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
input_parquet = PROCESSED_DIR / "amazon_master_delivery_dataset.parquet"
output_parquet = PROCESSED_DIR / "amazon_master_delivery_dataset_clean.parquet"
output_csv = PROCESSED_DIR / "amazon_master_delivery_dataset_clean.csv"

print(f"Reading {input_parquet}...")
df = pd.read_parquet(input_parquet)

rows_before = len(df)
cols_before = len(df.columns)
cols_before_list = list(df.columns)

# 1. Remove redundant column: time_window_duration (Keep: time_window_duration_hours)
removed_columns = []
if "time_window_duration" in df.columns:
    df.drop(columns=["time_window_duration"], inplace=True)
    removed_columns.append("time_window_duration")

rows_after = len(df)
cols_after = len(df.columns)
cols_after_list = list(df.columns)

print("Starting validation checks...")

# 2. Missing values validation
missing_cols = ["zone_id", "time_window_start", "time_window_end", "time_window_duration_hours"]
missing_summary = {}
for col in missing_cols:
    cnt = df[col].isna().sum()
    pct = (cnt / rows_before) * 100
    missing_summary[col] = {"count": int(cnt), "percentage": round(float(pct), 4)}

# 3. Validate time windows
tw_has_start = df["time_window_start"].notna()
tw_has_end = df["time_window_end"].notna()
both_present = tw_has_start & tw_has_end
only_start = tw_has_start & ~tw_has_end
only_end = ~tw_has_start & tw_has_end
both_missing = ~tw_has_start & ~tw_has_end

start_dt = pd.to_datetime(df.loc[both_present, "time_window_start"], errors="coerce")
end_dt = pd.to_datetime(df.loc[both_present, "time_window_end"], errors="coerce")
unparseable_start = start_dt.isna().sum()
unparseable_end = end_dt.isna().sum()

invalid_tw_order = (start_dt > end_dt).sum()
equal_tw = (start_dt == end_dt).sum()
negative_or_zero_duration = (df.loc[both_present, "time_window_duration_hours"] <= 0).sum()

tw_validation = {
    "both_present": int(both_present.sum()),
    "both_missing": int(both_missing.sum()),
    "only_start_present": int(only_start.sum()),
    "only_end_present": int(only_end.sum()),
    "unparseable_start": int(unparseable_start),
    "unparseable_end": int(unparseable_end),
    "start_greater_than_end": int(invalid_tw_order),
    "start_equals_end": int(equal_tw),
    "duration_le_zero": int(negative_or_zero_duration)
}

# 4. Validate package dimensions
depth_le_0 = (df["depth_cm"] <= 0).sum()
height_le_0 = (df["height_cm"] <= 0).sum()
width_le_0 = (df["width_cm"] <= 0).sum()
vol_le_0 = (df["package_volume_cm3"] <= 0).sum()

calculated_vol = df["depth_cm"] * df["height_cm"] * df["width_cm"]
vol_diff = (df["package_volume_cm3"] - calculated_vol).abs()
# floating tolerance
vol_mismatch = (vol_diff > 1e-3).sum()

dim_validation = {
    "depth_le_zero": int(depth_le_0),
    "height_le_zero": int(height_le_0),
    "width_le_zero": int(width_le_0),
    "volume_le_zero": int(vol_le_0),
    "volume_formula_mismatches": int(vol_mismatch),
    "max_absolute_volume_diff": float(vol_diff.max())
}

# 5. Validate coordinates
lat_invalid = ((df["latitude"] < -90) | (df["latitude"] > 90) | df["latitude"].isna()).sum()
lng_invalid = ((df["longitude"] < -180) | (df["longitude"] > 180) | df["longitude"].isna()).sum()

coord_validation = {
    "latitude_invalid_count": int(lat_invalid),
    "longitude_invalid_count": int(lng_invalid),
    "lat_min": float(df["latitude"].min()),
    "lat_max": float(df["latitude"].max()),
    "lng_min": float(df["longitude"].min()),
    "lng_max": float(df["longitude"].max())
}

# 6. Validate categorical columns
cat_summary = {}
cat_cols = ["scan_status", "station_code", "stop_type", "day_of_week"]
for col in cat_cols:
    val_counts = df[col].value_counts(dropna=False).to_dict()
    # convert any numpy/int keys/values
    cat_summary[col] = {str(k): int(v) for k, v in val_counts.items()}

cat_summary["zone_id"] = {
    "num_unique_zones": int(df["zone_id"].nunique(dropna=True)),
    "null_count": int(df["zone_id"].isna().sum())
}

# 7. Validate dates/times parsing
delivery_date_parsed = pd.to_datetime(df["delivery_date"], errors="coerce")
delivery_date_unparseable = delivery_date_parsed.isna().sum()

departure_time_parsed = pd.to_datetime(df["departure_time"], format="%H:%M:%S", errors="coerce")
departure_time_unparseable = departure_time_parsed.isna().sum()

date_time_validation = {
    "delivery_date_unparseable": int(delivery_date_unparseable),
    "delivery_date_min": str(delivery_date_parsed.min().date()),
    "delivery_date_max": str(delivery_date_parsed.max().date()),
    "departure_time_unparseable": int(departure_time_unparseable),
    "tw_start_unparseable": int(unparseable_start),
    "tw_end_unparseable": int(unparseable_end)
}

# 8. Check duplicates
dup_pkg_id = df["package_id"].duplicated().sum()
dup_route_stop_pkg = df.duplicated(subset=["route_id", "stop_id", "package_id"]).sum()
dup_entire_row = df.duplicated().sum()

duplicate_summary = {
    "duplicate_package_id": int(dup_pkg_id),
    "duplicate_route_stop_pkg": int(dup_route_stop_pkg),
    "duplicate_entire_row": int(dup_entire_row)
}

# 9. Numerical outliers summary
num_cols = ["planned_service_time_seconds", "package_volume_cm3", "depth_cm", "height_cm", "width_cm"]
outlier_summary = {}
for col in num_cols:
    s = df[col]
    q25 = float(s.quantile(0.25))
    q75 = float(s.quantile(0.75))
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    p99 = float(s.quantile(0.99))
    p999 = float(s.quantile(0.999))
    outlier_count_upper = int((s > upper_bound).sum())
    outlier_count_lower = int((s < lower_bound).sum())
    
    outlier_summary[col] = {
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": round(float(s.mean()), 3),
        "median": float(s.median()),
        "std": round(float(s.std()), 3),
        "q25": q25,
        "q75": q75,
        "iqr": round(iqr, 3),
        "upper_bound_1.5iqr": round(upper_bound, 3),
        "outliers_above_1.5iqr": outlier_count_upper,
        "outliers_below_1.5iqr": outlier_count_lower,
        "p99": round(p99, 3),
        "p99_9": round(p999, 3)
    }

# 11. Save cleaned files
print(f"Saving cleaned dataset to {output_parquet}...")
df.to_parquet(output_parquet, index=False)

print(f"Saving cleaned dataset to {output_csv}...")
df.to_csv(output_csv, index=False)

results = {
    "rows_before": rows_before,
    "rows_after": rows_after,
    "cols_before": cols_before,
    "cols_after": cols_after,
    "removed_columns": removed_columns,
    "cols_after_list": cols_after_list,
    "missing_summary": missing_summary,
    "tw_validation": tw_validation,
    "dim_validation": dim_validation,
    "coord_validation": coord_validation,
    "cat_summary": cat_summary,
    "date_time_validation": date_time_validation,
    "duplicate_summary": duplicate_summary,
    "outlier_summary": outlier_summary
}

with open(PROCESSED_DIR / "phase_1_4_validation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("Phase 1.4 script completed successfully.")

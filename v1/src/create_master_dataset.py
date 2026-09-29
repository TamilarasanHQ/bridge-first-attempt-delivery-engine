import json
import math
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/raw/model_build_inputs")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Loading route_data.json...")
with open(DATA_DIR / "route_data.json", "r", encoding="utf-8") as f:
    route_data = json.load(f)

print("Loading package_data.json...")
with open(DATA_DIR / "package_data.json", "r", encoding="utf-8") as f:
    package_data = json.load(f)

print("Flattening and merging records...")
records = []

for route_id, stops in package_data.items():
    route_info = route_data.get(route_id, {})
    station_code = route_info.get("station_code")
    delivery_date = route_info.get("date_YYYY_MM_DD")
    departure_time = route_info.get("departure_time_utc")
    route_stops = route_info.get("stops", {})

    for stop_id, packages in stops.items():
        stop_info = route_stops.get(stop_id, {})
        lat = stop_info.get("lat")
        lng = stop_info.get("lng")
        zone_id = stop_info.get("zone_id")
        stop_type = stop_info.get("type")

        for package_id, pkg in packages.items():
            scan_status = pkg.get("scan_status")
            planned_service_time = pkg.get("planned_service_time_seconds")

            # Time window
            tw = pkg.get("time_window") or {}
            tw_start = tw.get("start_time_utc")
            tw_end = tw.get("end_time_utc")

            # Check NaN or None
            if tw_start is not None and isinstance(tw_start, float) and math.isnan(tw_start):
                tw_start = None
            if tw_end is not None and isinstance(tw_end, float) and math.isnan(tw_end):
                tw_end = None

            # Dimensions
            dims = pkg.get("dimensions") or {}
            depth_cm = dims.get("depth_cm")
            height_cm = dims.get("height_cm")
            width_cm = dims.get("width_cm")

            # Package volume cm3
            if depth_cm is not None and height_cm is not None and width_cm is not None:
                pkg_vol = depth_cm * height_cm * width_cm
            else:
                pkg_vol = None

            records.append({
                "route_id": route_id,
                "stop_id": stop_id,
                "package_id": package_id,
                "scan_status": scan_status,
                "station_code": station_code,
                "delivery_date": delivery_date,
                "departure_time": departure_time,
                "latitude": lat,
                "longitude": lng,
                "zone_id": zone_id,
                "stop_type": stop_type,
                "time_window_start": tw_start,
                "time_window_end": tw_end,
                "planned_service_time_seconds": planned_service_time,
                "depth_cm": depth_cm,
                "height_cm": height_cm,
                "width_cm": width_cm,
                "package_volume_cm3": pkg_vol,
            })

print(f"Total extracted package rows: {len(records)}")
df = pd.DataFrame(records)

print("Deriving date/time features...")
# departure_hour
# departure_time is typically HH:MM:SS
df["departure_hour"] = pd.to_datetime(df["departure_time"], format="%H:%M:%S", errors="coerce").dt.hour

# delivery_date derived features
delivery_dt = pd.to_datetime(df["delivery_date"], errors="coerce")
df["day_of_week"] = delivery_dt.dt.day_name()
df["is_weekend"] = delivery_dt.dt.dayofweek.isin([5, 6]).astype(int)
df["month"] = delivery_dt.dt.month

# time_window_duration_hours
tw_start_dt = pd.to_datetime(df["time_window_start"], errors="coerce")
tw_end_dt = pd.to_datetime(df["time_window_end"], errors="coerce")
df["time_window_duration_hours"] = (tw_end_dt - tw_start_dt).dt.total_seconds() / 3600.0

# Order columns according to specifications
columns_order = [
    "route_id",
    "stop_id",
    "package_id",
    "scan_status",
    "station_code",
    "delivery_date",
    "departure_time",
    "latitude",
    "longitude",
    "zone_id",
    "stop_type",
    "time_window_start",
    "time_window_end",
    "planned_service_time_seconds",
    "depth_cm",
    "height_cm",
    "width_cm",
    "package_volume_cm3",
    "departure_hour",
    "day_of_week",
    "is_weekend",
    "month",
    "time_window_duration_hours"
]
df = df[columns_order]

print("Saving parquet...")
parquet_path = PROCESSED_DIR / "amazon_master_delivery_dataset.parquet"
df.to_parquet(parquet_path, index=False)
print(f"Saved parquet to {parquet_path}")

print("Saving csv...")
csv_path = PROCESSED_DIR / "amazon_master_delivery_dataset.csv"
df.to_csv(csv_path, index=False)
print(f"Saved csv to {csv_path}")

print("\n" + "="*50)
print("PHASE 1.3 SUMMARY REPORT")
print("="*50)
print(f"1. Number of rows: {len(df)}")
print(f"2. Number of columns: {len(df.columns)}")
print(f"3. Column names:\n{list(df.columns)}")
print("\n4. Missing values per column:")
print(df.isna().sum())
print(f"\n5. Duplicate count: {df.duplicated().sum()}")
print("\n6. Sample 5 rows:")
print(df.head(5).to_string())
print("\n7. Data types:")
print(df.dtypes)

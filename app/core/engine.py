"""
BRIDGE Inference Engine Wrapper.
Loads the locked Phase 5 HistGradientBoosting model and handles pre-dispatch feature engineering.
"""
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from app.core.config import settings


class BridgeInferenceEngine:
    def __init__(self, model_path: Path = settings.MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        self.model = joblib.load(self.model_path)

    def prepare_dataframe(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Transforms raw parcel records into the exact 23 features expected by the model.
        """
        df = pd.DataFrame(records)
        
        # Required base features
        base_cols = [
            "departure_hour", "is_weekend", "month", "latitude", "longitude",
            "depth_cm", "height_cm", "width_cm", "package_volume_cm3",
            "planned_service_time_seconds", "has_time_window", "time_window_duration_hours",
            "executor_capacity_cm3", "planned_route_package_count", "planned_route_stop_count",
            "planned_route_volume_cm3", "route_volume_utilization", "station_code", "day_of_week"
        ]
        
        # Ensure all columns exist with defaults if missing
        for col in base_cols:
            if col not in df.columns:
                if col in ["has_time_window", "is_weekend"]:
                    df[col] = 0
                elif col in ["time_window_duration_hours", "planned_service_time_seconds"]:
                    df[col] = 0.0
                elif col == "station_code":
                    df[col] = "DLA3"
                elif col == "day_of_week":
                    df[col] = "Wednesday"
                else:
                    df[col] = 0.0

        # Fill structural NaNs
        df["time_window_duration_hours"] = df["time_window_duration_hours"].fillna(0.0)

        # Ensure correct datatypes
        df["departure_hour"] = df["departure_hour"].astype(int)
        df["is_weekend"] = df["is_weekend"].astype(int)
        df["month"] = df["month"].astype(int)
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        df["depth_cm"] = df["depth_cm"].astype(float)
        df["height_cm"] = df["height_cm"].astype(float)
        df["width_cm"] = df["width_cm"].astype(float)
        df["package_volume_cm3"] = df["package_volume_cm3"].astype(float)
        df["planned_service_time_seconds"] = df["planned_service_time_seconds"].astype(float)
        df["has_time_window"] = df["has_time_window"].astype(int)
        df["time_window_duration_hours"] = df["time_window_duration_hours"].astype(float)
        df["executor_capacity_cm3"] = df["executor_capacity_cm3"].astype(float)
        df["planned_route_package_count"] = df["planned_route_package_count"].astype(int)
        df["planned_route_stop_count"] = df["planned_route_stop_count"].astype(int)
        df["planned_route_volume_cm3"] = df["planned_route_volume_cm3"].astype(float)
        df["route_volume_utilization"] = df["route_volume_utilization"].astype(float)

        # Categorical columns
        df["station_code"] = df["station_code"].astype("category")
        df["day_of_week"] = df["day_of_week"].astype("category")

        # Group B: Route Workload Ratios (with clipped denominators)
        pkg_cnt = df["planned_route_package_count"].clip(lower=1.0)
        stop_cnt = df["planned_route_stop_count"].clip(lower=1.0)
        rt_vol = df["planned_route_volume_cm3"].clip(lower=1.0)

        df["packages_per_stop"] = pkg_cnt / stop_cnt
        df["route_volume_per_package"] = rt_vol / pkg_cnt
        df["route_volume_per_stop"] = rt_vol / stop_cnt
        df["pkg_volume_to_route_vol"] = df["package_volume_cm3"] / rt_vol

        # Ensure exact feature order matching model.feature_names_in_
        feature_order = list(self.model.feature_names_in_)
        return df[feature_order]

    def predict_proba(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """
        Returns calibrated failure probability array P(first_attempt_failed = 1).
        """
        X = self.prepare_dataframe(records)
        return self.model.predict_proba(X)[:, 1]

    def explain_risk(self, record: Dict[str, Any], prob: float) -> List[str]:
        """
        Generates human-readable, non-causal associative explanations of key risk drivers.
        """
        reasons = []
        dwell = record.get("planned_service_time_seconds", 0)
        if dwell > 180:
            reasons.append(f"Elevated service dwell time ({dwell:.0f}s) indicates complex premises access (gate/apartment).")
        elif dwell > 120:
            reasons.append(f"Above-average service dwell ({dwell:.0f}s).")

        if record.get("has_time_window", 0) == 1:
            dur = record.get("time_window_duration_hours", 0)
            reasons.append(f"Restricted recipient delivery time window ({dur:.1f} hr slot).")

        rt_util = record.get("route_volume_utilization", 0)
        if rt_util > 0.70:
            reasons.append(f"High vehicle capacity load ({rt_util*100:.1f}%) increases end-of-route fatigue.")

        stn = record.get("station_code", "")
        if stn in ["DLA8", "DLA3", "DCH1"]:
            reasons.append(f"Depot {stn} operates in high-density urban traffic corridor.")

        pkg_vol = record.get("package_volume_cm3", 0)
        if pkg_vol > 30000:
            reasons.append(f"Oversized bulky parcel ({pkg_vol/1000:.1f}L) requires physical handoff.")

        if not reasons:
            if prob > settings.HIGH_RISK_THRESHOLD:
                reasons.append("Combined interaction of route stop density and delivery timing profile.")
            else:
                reasons.append("Standard operating characteristics within normal parameters.")

        return reasons


# Global engine singleton
engine = BridgeInferenceEngine()

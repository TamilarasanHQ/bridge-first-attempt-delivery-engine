"""
BRIDGE v2 - Phase 3: Standalone Baseline Evaluation Utility
Loads pre-trained v2 models and evaluates performance metrics on validation or test sets.

Usage:
  python src/v2/evaluate_baselines.py --split validation
  python src/v2/evaluate_baselines.py --split test
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

DATA_DIR = Path("data/v2")
MODELS_DIR = Path("v2/models") if Path("v2/models").exists() else Path("models/v2")

NUMERICAL_FEATURES = [
    "departure_hour", "is_weekend", "month", "latitude", "longitude",
    "depth_cm", "height_cm", "width_cm", "package_volume_cm3",
    "planned_service_time_seconds", "has_time_window", "time_window_duration_hours",
    "executor_capacity_cm3", "planned_route_package_count", "planned_route_stop_count",
    "planned_route_volume_cm3", "route_volume_utilization"
]
CATEGORICAL_FEATURES = ["station_code", "day_of_week"]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

def evaluate_metrics(y_true, y_prob, name):
    ap = average_precision_score(y_true, y_prob)
    roc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))
    
    N = len(y_true)
    order = np.argsort(-y_prob)
    
    def top_k(pct):
        k = max(1, int(N * (pct / 100.0)))
        actual_in_top = y_true[order[:k]].sum()
        p = actual_in_top / k
        r = actual_in_top / y_true.sum()
        return p, r
        
    p1, r1 = top_k(1.0)
    p5, r5 = top_k(5.0)
    p10, r10 = top_k(10.0)
    
    return {
        "Model": name,
        "PR-AUC": ap,
        "ROC-AUC": roc,
        "Brier": brier,
        "LogLoss": ll,
        "P@1%": p1,
        "R@1%": r1,
        "P@5%": p5,
        "R@5%": r5,
        "P@10%": p10,
        "R@10%": r10
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate BRIDGE v2 Baselines")
    parser.add_argument("--split", choices=["validation", "test"], default="validation", help="Split to evaluate")
    args = parser.parse_args()
    
    split_path = DATA_DIR / f"{args.split}.parquet"
    print(f"[Phase 3 Evaluation] Loading {args.split} split from {split_path}...")
    df = pd.read_parquet(split_path)
    
    y = df["first_attempt_failed"].values.astype(np.int8)
    prevalence = y.mean()
    print(f"Total rows: {len(df):,}, Failures: {y.sum():,}, Prevalence: {prevalence*100:.4f}%")
    
    X_raw = df[FEATURE_COLUMNS].copy()
    X_raw["time_window_duration_hours"] = X_raw["time_window_duration_hours"].fillna(0.0)
    
    # Load Tree Model
    tree_model_path = MODELS_DIR / "tree_baseline.joblib"
    if tree_model_path.exists():
        tree_model = joblib.load(tree_model_path)
        X_tree = X_raw.copy()
        for c in CATEGORICAL_FEATURES:
            X_tree[c] = X_tree[c].astype("category")
        tree_probs = tree_model.predict_proba(X_tree)[:, 1]
        res_tree = evaluate_metrics(y, tree_probs, "Locked Tree Baseline")
        print("\n--- Tree Baseline Results ---")
        for k, v in res_tree.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.6f}")
            else:
                print(f"  {k:12s}: {v}")
                
    # Load LR Model
    lr_model_path = MODELS_DIR / "logistic_regression_baseline.joblib"
    prep_path = MODELS_DIR / "preprocessor.joblib"
    if lr_model_path.exists() and prep_path.exists():
        lr_model = joblib.load(lr_model_path)
        prep = joblib.load(prep_path)
        X_lr = prep.transform(X_raw)
        lr_probs = lr_model.predict_proba(X_lr)[:, 1]
        res_lr = evaluate_metrics(y, lr_probs, "Logistic Regression Baseline")
        print("\n--- Logistic Regression Baseline Results ---")
        for k, v in res_lr.items():
            if isinstance(v, float):
                print(f"  {k:12s}: {v:.6f}")
            else:
                print(f"  {k:12s}: {v}")

if __name__ == "__main__":
    main()

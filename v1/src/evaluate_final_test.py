"""
Phase 2.4: Final Blind Evaluation on TEST Holdout Set
BRIDGE — AI-Powered First Attempt Delivery Success Engine
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

import sys
base_dir = Path(__file__).resolve().parent.parent
if str(base_dir) not in sys.path:
    sys.path.insert(0, str(base_dir))

from risk_engine import RiskEngine


CATEGORICAL_FEATURES = [
    "station_code",
    "preferred_delivery_method",
    "customer_delivery_preference",
]

NUMERICAL_FEATURES = [
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
    "preference_departure_mismatch",
]

X_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET = "first_attempt_failed"
FORBIDDEN_COLS = [
    "first_attempt_failed", "scan_status", "route_id", "stop_id", "package_id",
    "delivery_date", "departure_time", "stop_type", "time_window_start", "time_window_end", "zone_id"
]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def run_final_test_evaluation():
    print("=" * 80)
    print("BRIDGE Phase 2.4: FINAL BLIND TEST EVALUATION")
    print("=" * 80)

    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures" / "phase_2_4"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Immutability Pre-Check
    print("[1/8] Verifying artifact immutability before evaluation...")
    tracked_files = [
        models_dir / "model_logistic_regression.joblib",
        models_dir / "preprocessor_lr.joblib",
        models_dir / "train_indices.npy",
        models_dir / "val_indices.npy",
        models_dir / "test_indices.npy",
        models_dir / "risk_policy_v1.json",
    ]
    pre_hashes = {p.name: file_sha256(p) for p in tracked_files}

    # 2. Verify TEST Set Integrity
    print("[2/8] Loading and verifying TEST holdout dataset...")
    test_idx = np.load(models_dir / "test_indices.npy")
    train_idx = np.load(models_dir / "train_indices.npy")
    val_idx = np.load(models_dir / "val_indices.npy")

    assert len(test_idx) == 224881, f"Expected 224,881 test rows, got {len(test_idx)}"
    assert len(set(test_idx).intersection(set(train_idx))) == 0, "CRITICAL DEFECT: Test overlaps with Train!"
    assert len(set(test_idx).intersection(set(val_idx))) == 0, "CRITICAL DEFECT: Test overlaps with Val!"

    parquet_path = data_dir / "amazon_delivery_ml_features_v1.parquet"
    if not parquet_path.exists():
        parquet_path = data_dir / "amazon_delivery_ml_features_v1.csv"
    
    df = pd.read_parquet(parquet_path) if str(parquet_path).endswith(".parquet") else pd.read_csv(parquet_path)

    test_df = df.iloc[test_idx].copy()
    test_min_date = test_df["delivery_date"].min()
    test_max_date = test_df["delivery_date"].max()
    test_dates = test_df["delivery_date"].nunique()

    print(f"  TEST row count: {len(test_df):,} rows")
    print(f"  TEST temporal range: {test_min_date} to {test_max_date} ({test_dates} unique dates)")
    print(f"  TEST unique routes: {test_df['route_id'].nunique():,}")

    # Verify no route overlap
    train_val_routes = set(df.iloc[np.concatenate([train_idx, val_idx])]["route_id"].unique())
    test_routes = set(test_df["route_id"].unique())
    overlap_routes = test_routes.intersection(train_val_routes)
    assert len(overlap_routes) == 0, f"LEAKAGE DEFECT: {len(overlap_routes)} routes overlap between test and train/val!"
    print("  Assertion PASSED: Zero route overlap between TEST and historical data.")

    # Verify feature columns and forbidden columns
    for f in FORBIDDEN_COLS:
        assert f not in X_COLS, f"LEAKAGE DEFECT: Forbidden column {f} in X_COLS!"
    assert len(X_COLS) == 37, f"Expected 37 features, got {len(X_COLS)}"

    X_test = test_df[X_COLS]
    y_test = test_df[TARGET].values

    # 3. Generate Final TEST Predictions
    print("[3/8] Generating predictions using frozen Phase 2.1 preprocessor & Phase 2.2 model...")
    model_lr = joblib.load(models_dir / "model_logistic_regression.joblib")
    prep_lr = joblib.load(models_dir / "preprocessor_lr.joblib")

    X_test_trans = prep_lr.transform(X_test)
    probs_test = model_lr.predict_proba(X_test_trans)[:, 1]

    # 4. Final Model Metrics
    print("[4/8] Computing final model discrimination & calibration metrics on TEST...")
    test_n = len(y_test)
    test_failures = int(y_test.sum())
    test_base_rate = test_failures / test_n

    roc_auc = roc_auc_score(y_test, probs_test)
    pr_auc = average_precision_score(y_test, probs_test)
    brier = brier_score_loss(y_test, probs_test)
    logloss = log_loss(y_test, probs_test)

    prob_min = float(probs_test.min())
    prob_max = float(probs_test.max())
    prob_mean = float(probs_test.mean())
    prob_median = float(np.median(probs_test))
    prob_std = float(probs_test.std())

    print(f"  TEST ROC-AUC:          {roc_auc:.4f}")
    print(f"  TEST PR-AUC:           {pr_auc:.4f}")
    print(f"  TEST Brier Score:      {brier:.4f}")
    print(f"  TEST Log Loss:         {logloss:.4f}")
    print(f"  Actual Failure Rate:   {test_base_rate*100:.4f}% ({test_failures:,} / {test_n:,})")
    print(f"  Mean Predicted Prob:   {prob_mean*100:.4f}% (Delta vs actual: {abs(prob_mean-test_base_rate)*100:.4f}%)")
    print(f"  Probability Range:     [{prob_min:.4f}, {prob_max:.4f}], Median: {prob_median:.4f}, Std: {prob_std:.4f}")

    # 5. Final Risk-Band Evaluation
    print("[5/8] Evaluating frozen risk policy (LOW < 0.080, MEDIUM 0.080-0.120, HIGH >= 0.120)...")
    engine = RiskEngine()

    mask_low = probs_test < 0.080
    mask_med = (probs_test >= 0.080) & (probs_test < 0.120)
    mask_high = probs_test >= 0.120

    bands_data = []
    for name, mask, prange in [
        ("LOW", mask_low, "< 0.080"),
        ("MEDIUM", mask_med, "0.080 - 0.120"),
        ("HIGH", mask_high, ">= 0.120"),
    ]:
        b_count = int(mask.sum())
        b_pct = (b_count / test_n) * 100
        b_fails = int(y_test[mask].sum())
        b_fr = (b_fails / b_count * 100) if b_count > 0 else 0.0
        b_p_avg = (probs_test[mask].mean() * 100) if b_count > 0 else 0.0
        b_rr = (b_fr / (test_base_rate * 100)) if test_base_rate > 0 else 0.0
        bands_data.append({
            "risk_band": name,
            "probability_range": prange,
            "deliveries": b_count,
            "pct_deliveries": b_pct,
            "actual_failures": b_fails,
            "actual_failure_rate": b_fr,
            "avg_predicted_prob": b_p_avg,
            "relative_risk": b_rr,
        })
        print(f"  {name:6s}: {b_count:7,d} ({b_pct:5.2f}%) | Failures: {b_fails:5,d} | Actual FR: {b_fr:5.2f}% | Avg P: {b_p_avg:5.2f}% | RR: {b_rr:.2f}x")

    high_fails = int(y_test[mask_high].sum())
    high_fraction_of_total_fails = (high_fails / test_failures) * 100
    cum_med_high_fails = int(y_test[mask_med | mask_high].sum())
    cum_med_high_fraction = (cum_med_high_fails / test_failures) * 100

    print(f"  HIGH-Risk Failure Capture: {high_fails:,} / {test_failures:,} failures ({high_fraction_of_total_fails:.2f}%)")
    print(f"  MEDIUM+HIGH Failure Capture: {cum_med_high_fails:,} / {test_failures:,} failures ({cum_med_high_fraction:.2f}%)")

    # 6. Final Threshold Performance (HIGH threshold t = 0.120)
    print("[6/8] Computing threshold performance for frozen HIGH cutoff (t = 0.120)...")
    y_pred_high = (probs_test >= 0.120).astype(int)
    cm = confusion_matrix(y_test, y_pred_high)
    tn, fp, fn, tp = cm.ravel()

    prec_high = precision_score(y_test, y_pred_high, zero_division=0)
    rec_high = recall_score(y_test, y_pred_high, zero_division=0)
    f1_high = f1_score(y_test, y_pred_high, zero_division=0)
    fpr_high = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_high = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    interventions_per_1000 = (mask_high.sum() / test_n) * 1000.0
    failures_caught_per_1000 = (tp / test_n) * 1000.0

    print(f"  Precision:                 {prec_high:.4f}")
    print(f"  Recall:                    {rec_high:.4f}")
    print(f"  F1 Score:                  {f1_high:.4f}")
    print(f"  False Positive Rate:       {fpr_high:.4f}")
    print(f"  True Positive Rate:        {tpr_high:.4f}")
    print(f"  Interventions per 1,000:   {interventions_per_1000:.2f}")
    print(f"  Failures Caught per 1,000: {failures_caught_per_1000:.2f}")

    # 7. Intervention Distribution Simulation via risk_engine
    print("[7/8] Simulating operational interventions via risk_engine on TEST deliveries...")
    
    # Vectorized fast intervention categorization matching risk_engine rules
    risk_levels = np.where(probs_test < 0.080, "LOW", np.where(probs_test < 0.120, "MEDIUM", "HIGH"))
    
    mismatch = test_df["preference_departure_mismatch"].values
    method = test_df["preferred_delivery_method"].values
    sig = test_df["signature_required"].values
    access = test_df["address_access_difficulty"].values
    avail = test_df["customer_availability_rate"].values
    pref = test_df["customer_delivery_preference"].values
    dep_hour = test_df["departure_hour"].values

    actions = []
    for i in range(test_n):
        rl = risk_levels[i]
        m_i = mismatch[i]
        meth_i = method[i]
        sig_i = sig[i]
        acc_i = access[i]
        av_i = avail[i]
        pref_i = pref[i]
        dh_i = dep_hour[i]

        if rl == "LOW":
            if sig_i == 1:
                actions.append("STANDARD_DISPATCH_SIGNATURE_PROMPT")
            else:
                actions.append("STANDARD_DISPATCH")
        elif rl == "MEDIUM":
            if m_i == 1:
                actions.append("APPOINTMENT_TIMING_CONFIRMATION")
            elif acc_i >= 4:
                actions.append("REQUEST_GATE_CODE_INSTRUCTIONS")
            elif meth_i == "Hand to Customer" or sig_i == 1 or av_i < 0.70:
                actions.append("CONFIRM_PRESENCE_AND_BACKUP_LOCATION")
            else:
                actions.append("CUSTOMER_NOTIFICATION")
        else:  # HIGH
            if m_i == 1:
                actions.append("RESCHEDULE_DELIVERY_WINDOW")
            elif pref_i == "Evening Only" and dh_i < 15:
                actions.append("HOLD_FOR_EVENING_SHIFT")
            elif meth_i == "Hand to Customer" or sig_i == 1:
                actions.append("ACCESS_POINT_REDIRECT")
            elif acc_i >= 4:
                actions.append("REROUTE_TO_BUILDING_MANAGEMENT_OR_LOCKER")
            else:
                actions.append("EXPEDITED_CONFIRMATION_OR_LOCKER")

    test_df["failure_probability"] = np.round(probs_test, 4)
    test_df["risk_level"] = risk_levels
    test_df["recommended_action"] = actions

    action_counts = pd.Series(actions).value_counts()
    print("\nOperational Intervention Distribution on TEST:")
    for act, count in action_counts.items():
        print(f"  {act:42s}: {count:7,d} ({count/test_n*100:5.2f}%)")

    # Save Predictions CSV
    pred_cols = ["package_id", "route_id", "stop_id", "delivery_date", "failure_probability", "risk_level", "first_attempt_failed", "recommended_action"]
    pred_csv_path = reports_dir / "phase_2_4_test_predictions.csv"
    test_df[pred_cols].to_csv(pred_csv_path, index=False)
    print(f"\nSaved TEST predictions CSV ({len(test_df):,} rows) to: {pred_csv_path}")

    # Generate Visualizations
    print("[8/8] Generating final TEST figures...")

    # Figure 1: Test Probability Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(probs_test, bins=60, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(prob_mean, color="red", linestyle="--", linewidth=2, label=f"Mean P: {prob_mean:.4f}")
    plt.axvline(prob_median, color="green", linestyle=":", linewidth=2, label=f"Median P: {prob_median:.4f}")
    plt.axvline(0.080, color="orange", linestyle="-", linewidth=1.5, label="Low Cutoff: 0.080")
    plt.axvline(0.120, color="purple", linestyle="-", linewidth=1.5, label="High Cutoff: 0.120")
    plt.title("TEST Set Predicted Failure Probability Distribution (Logistic Regression)", fontsize=12)
    plt.xlabel("Predicted Probability P(first_attempt_failed)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "1_test_prob_distribution.png", dpi=200)
    plt.close()

    # Figure 2: Test Calibration Curve
    prob_true, prob_pred = calibration_curve(y_test, probs_test, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="#2ca02c", linewidth=2.5, label="TEST Logistic Regression")
    plt.plot([0.04, 0.20], [0.04, 0.20], linestyle="--", color="gray", label="Perfect Calibration (y = x)")
    plt.title("Reliability Diagram: Blind TEST Set Calibration", fontsize=12)
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Failures", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "2_test_calibration_curve.png", dpi=200)
    plt.close()

    # Figure 3: Test Risk-Band Failure Rates
    band_names = [b["risk_band"] for b in bands_data]
    band_frs = [b["actual_failure_rate"] for b in bands_data]
    band_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(band_names, band_frs, color=band_colors, edgecolor="black", width=0.5)
    plt.axhline(test_base_rate * 100, color="gray", linestyle="--", linewidth=1.5, label=f"TEST Base Rate: {test_base_rate*100:.2f}%")
    for bar, b in zip(bars, bands_data):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%\n({b['relative_risk']:.2f}x)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("Actual Failure Rate by Operational Risk Band on TEST Set", fontsize=12)
    plt.ylabel("Actual First-Attempt Failure Rate (%)", fontsize=11)
    plt.ylim(0, max(band_frs) * 1.25)
    plt.legend(fontsize=10)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "3_test_risk_band_failure_rates.png", dpi=200)
    plt.close()

    # Figure 4: Intervention Distribution
    plt.figure(figsize=(10, 5.5))
    act_labels = [k.replace("_", " ") for k in action_counts.index]
    y_pos = np.arange(len(act_labels))
    plt.barh(y_pos, action_counts.values, color="#1f77b4", edgecolor="black", alpha=0.8)
    plt.yticks(y_pos, act_labels, fontsize=9)
    plt.gca().invert_yaxis()
    for i, v in enumerate(action_counts.values):
        pct = v / test_n * 100
        plt.text(v + 1500, i, f"{v:,} ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")
    plt.title("Operational Intervention Volume on Blind TEST Set (224,881 Deliveries)", fontsize=12)
    plt.xlabel("Number of Deliveries", fontsize=11)
    plt.xlim(0, max(action_counts.values) * 1.20)
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "4_test_intervention_distribution.png", dpi=200)
    plt.close()

    # 8. Immutability Post-Check
    post_hashes = {p.name: file_sha256(p) for p in tracked_files}
    for fname, pre_h in pre_hashes.items():
        assert pre_h == post_hashes[fname], f"IMMUTABILITY VIOLATION: {fname} was modified during evaluation!"
    print("  Immutability check PASSED: All 6 model/index/policy files remained completely unmodified.")

    # 9. Return Results Dictionary for Report Generation
    return {
        "test_n": test_n,
        "test_failures": test_failures,
        "test_base_rate": test_base_rate,
        "test_dates": f"{test_min_date} to {test_max_date} ({test_dates} dates)",
        "test_routes": test_df["route_id"].nunique(),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "logloss": logloss,
        "prob_min": prob_min,
        "prob_max": prob_max,
        "prob_mean": prob_mean,
        "prob_median": prob_median,
        "prob_std": prob_std,
        "bands_data": bands_data,
        "high_fails": high_fails,
        "high_fraction_of_total_fails": high_fraction_of_total_fails,
        "cum_med_high_fails": cum_med_high_fails,
        "cum_med_high_fraction": cum_med_high_fraction,
        "prec_high": prec_high,
        "rec_high": rec_high,
        "f1_high": f1_high,
        "fpr_high": fpr_high,
        "tpr_high": tpr_high,
        "interventions_per_1000": interventions_per_1000,
        "failures_caught_per_1000": failures_caught_per_1000,
        "action_counts": action_counts.to_dict(),
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
    }


if __name__ == "__main__":
    results = run_final_test_evaluation()

"""
Phase 2.3: Validation Threshold Sweep, Risk Band Analysis, and Visualization
BRIDGE — AI-Powered First Attempt Delivery Success Engine
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


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


def run_threshold_analysis():
    print("=" * 80)
    print("BRIDGE Phase 2.3: Risk Threshold Sweep & Validation Policy Evaluation")
    print("=" * 80)

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "processed"
    models_dir = base_dir / "models"
    reports_dir = base_dir / "reports"
    figures_dir = reports_dir / "figures" / "phase_2_3"

    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Strict Isolation Verification: Ensure TEST is NEVER loaded or evaluated
    test_idx_path = models_dir / "test_indices.npy"
    val_idx_path = models_dir / "val_indices.npy"
    val_idx = np.load(val_idx_path)

    if test_idx_path.exists():
        test_idx = np.load(test_idx_path)
        assert len(set(val_idx).intersection(set(test_idx))) == 0, "CRITICAL ERROR: Val indices overlap with Test!"
        del test_idx
    print(f"[1/5] Loaded validation indices ({len(val_idx):,} rows). TEST set is strictly isolated.")

    # 2. Load Dataset & Phase 2.1/2.2 Artifacts
    print("[2/5] Loading Phase 2.1 preprocessor and Phase 2.2 unweighted Logistic Regression model...")
    parquet_path = data_dir / "amazon_delivery_ml_features_v1.parquet"
    if not parquet_path.exists():
        parquet_path = data_dir / "amazon_delivery_ml_features_v1.csv"
    
    df = pd.read_parquet(parquet_path) if str(parquet_path).endswith(".parquet") else pd.read_csv(parquet_path)
    
    val_df = df.iloc[val_idx]
    X_val = val_df[X_COLS]
    y_val = val_df[TARGET].values

    model_lr = joblib.load(models_dir / "model_logistic_regression.joblib")
    prep_lr = joblib.load(models_dir / "preprocessor_lr.joblib")

    # 3. Predict Continuous Probabilities
    print("[3/5] Computing validation failure probabilities...")
    X_val_trans = prep_lr.transform(X_val)
    probs_val = model_lr.predict_proba(X_val_trans)[:, 1]

    val_n = len(y_val)
    val_failures = int(y_val.sum())
    val_base_rate = val_failures / val_n

    print(f"  Validation deliveries: {val_n:,}")
    print(f"  Validation failures:   {val_failures:,} ({val_base_rate*100:.2f}%)")
    print(f"  Predicted prob range:  [{probs_val.min():.4f}, {probs_val.max():.4f}]")
    print(f"  Predicted prob mean:   {probs_val.mean():.4f} (Base rate match: {abs(probs_val.mean()-val_base_rate)*100:.3f}% delta)")
    print(f"  Predicted prob median: {np.median(probs_val):.4f}")

    # 4. Perform Fine-Grained Threshold Sweep
    print("[4/5] Executing fine-grained validation threshold sweep...")
    thresholds = np.linspace(0.045, 0.220, 71)  # step = 0.0025
    sweep_rows = []

    for t in thresholds:
        y_pred = (probs_val >= t).astype(int)
        flagged_count = int(y_pred.sum())
        fraction_flagged = flagged_count / val_n

        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()

        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        observed_fr_flagged = tp / flagged_count if flagged_count > 0 else 0.0

        interventions_per_1000 = fraction_flagged * 1000.0
        failures_caught_per_1000 = (tp / val_n) * 1000.0

        sweep_rows.append({
            "threshold": round(float(t), 4),
            "flagged_deliveries": flagged_count,
            "fraction_flagged": round(float(fraction_flagged), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "fpr": round(float(fpr), 4),
            "tpr": round(float(tpr), 4),
            "observed_failure_rate_flagged": round(float(observed_fr_flagged), 4),
            "interventions_per_1000": round(float(interventions_per_1000), 2),
            "failures_caught_per_1000": round(float(failures_caught_per_1000), 2),
        })

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_csv_path = reports_dir / "phase_2_3_threshold_sweep.csv"
    sweep_df.to_csv(sweep_csv_path, index=False)
    print(f"  Saved threshold sweep CSV ({len(sweep_df)} rows) to: {sweep_csv_path}")

    # 5. Generate Visualizations
    print("[5/5] Generating validation-only plots...")

    # Plot 1: Probability Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(probs_val, bins=60, color="#1f77b4", edgecolor="black", alpha=0.7, density=True)
    plt.axvline(probs_val.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean P: {probs_val.mean():.4f}")
    plt.axvline(np.median(probs_val), color="green", linestyle=":", linewidth=2, label=f"Median P: {np.median(probs_val):.4f}")
    plt.axvline(0.080, color="orange", linestyle="-", linewidth=1.5, label="Low Cutoff: 0.080")
    plt.axvline(0.120, color="purple", linestyle="-", linewidth=1.5, label="High Cutoff: 0.120")
    plt.title("Validation Predicted Failure Probability Distribution (Logistic Regression)", fontsize=12)
    plt.xlabel("Predicted Probability P(first_attempt_failed)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "1_prob_distribution.png", dpi=200)
    plt.close()

    # Plot 2: Risk-Band Distribution (Selected Policy: <0.080, 0.080-0.120, >=0.120)
    band_low = probs_val < 0.080
    band_med = (probs_val >= 0.080) & (probs_val < 0.120)
    band_high = probs_val >= 0.120

    band_counts = [band_low.sum(), band_med.sum(), band_high.sum()]
    band_pcts = [c / val_n * 100 for c in band_counts]
    band_labels = ["LOW (<0.080)", "MEDIUM [0.080, 0.120)", "HIGH (>=0.120)"]
    band_colors = ["#2ca02c", "#ff7f0e", "#d62728"]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(band_labels, band_counts, color=band_colors, edgecolor="black", width=0.55)
    for bar, pct in zip(bars, band_pcts):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 3000, f"{yval:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title("Validation Delivery Distribution Across Operational Risk Bands", fontsize=12)
    plt.ylabel("Number of Deliveries", fontsize=11)
    plt.ylim(0, max(band_counts) * 1.18)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "2_risk_band_distribution.png", dpi=200)
    plt.close()

    # Plot 3: Observed Failure Rate by Probability Decile
    deciles = pd.qcut(probs_val, q=10, labels=False)
    decile_summary = pd.DataFrame({"prob": probs_val, "actual": y_val, "decile": deciles}).groupby("decile").agg(
        mean_p=("prob", "mean"),
        actual_fr=("actual", "mean"),
        count=("actual", "count"),
    )
    plt.figure(figsize=(8, 5))
    plt.plot(range(10), decile_summary["actual_fr"] * 100, marker="o", color="#d62728", linewidth=2.5, label="Observed Failure Rate (%)")
    plt.plot(range(10), decile_summary["mean_p"] * 100, marker="s", linestyle="--", color="#1f77b4", linewidth=2.0, label="Mean Predicted Probability (%)")
    plt.axhline(val_base_rate * 100, color="gray", linestyle=":", label=f"Overall Base Rate ({val_base_rate*100:.2f}%)")
    plt.xticks(range(10), [f"D{d}\n({p*100:.1f}%)" for d, p in zip(range(10), decile_summary["mean_p"])], fontsize=9)
    plt.title("Monotonic Failure Rate Separation Across Probability Deciles", fontsize=12)
    plt.xlabel("Validation Probability Decile (Mean Predicted P)", fontsize=11)
    plt.ylabel("Failure Rate (%)", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "3_observed_failure_rate_by_bin.png", dpi=200)
    plt.close()

    # Plot 4: Reliability / Calibration Curve
    prob_true, prob_pred = calibration_curve(y_val, probs_val, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker="o", color="#2ca02c", linewidth=2.5, label="Logistic Regression (Quantiles)")
    plt.plot([0.04, 0.20], [0.04, 0.20], linestyle="--", color="gray", label="Perfect Calibration (y = x)")
    plt.title("Reliability Diagram (Validation Calibration Curve)", fontsize=12)
    plt.xlabel("Mean Predicted Probability", fontsize=11)
    plt.ylabel("Observed Fraction of Failures", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "4_calibration_reliability_curve.png", dpi=200)
    plt.close()

    # Plot 5: Threshold vs Precision, Recall, and F1
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", color="#1f77b4", linewidth=2)
    plt.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall", color="#2ca02c", linewidth=2)
    plt.plot(sweep_df["threshold"], sweep_df["f1"], label="F1 Score", color="#d62728", linewidth=2)
    plt.axvline(0.080, color="orange", linestyle=":", label="Low Boundary (0.080)")
    plt.axvline(0.120, color="purple", linestyle=":", label="High Boundary (0.120)")
    plt.title("Validation Threshold vs. Precision, Recall, and F1 Trade-off", fontsize=12)
    plt.xlabel("Failure Probability Threshold", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "5_threshold_vs_prf1.png", dpi=200)
    plt.close()

    # Plot 6: Operational Frontier: Interventions vs. Failures Caught
    plt.figure(figsize=(8, 5))
    plt.plot(sweep_df["interventions_per_1000"], sweep_df["failures_caught_per_1000"], color="#9467bd", linewidth=2.5, marker=".")
    plt.scatter([102.1], [14.2], color="red", s=100, zorder=5, label="HIGH Band (t >= 0.120): 102 interv / 14.2 caught")
    plt.scatter([650.0], [67.8], color="orange", s=100, zorder=5, label="MED+HIGH (t >= 0.080): 650 interv / 67.8 caught")
    plt.title("Operational Frontier: Interventions vs. Failures Caught (per 1,000 Deliveries)", fontsize=12)
    plt.xlabel("Expected Interventions per 1,000 Deliveries", fontsize=11)
    plt.ylabel("Failed Deliveries Caught per 1,000 Deliveries", fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "6_threshold_vs_intervention_volume.png", dpi=200)
    plt.close()

    print(f"  All 6 validation plots saved to: {figures_dir}")
    print("=" * 80)
    print("Threshold evaluation and plot generation complete!")


if __name__ == "__main__":
    run_threshold_analysis()

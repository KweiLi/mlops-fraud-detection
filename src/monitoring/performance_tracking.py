import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import joblib
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.paths import (
    MODELS_DIR,
    FEATURE_NAMES_FILE,
    METRICS_DIR,
    PLOTS_DIR,
    TRANSACTIONS_FILE,
    get_latest_model,
)


class PerformanceMonitor:
    """
    Monitor model performance over time

    Tracks:
    - Classification metrics (precision, recall, F1)
    - ROC-AUC and PR-AUC
    - Confusion matrix
    - Business metrics ($ impact)
    - Alerts for performance degradation
    """

    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            latest_model = get_latest_model()
            self.model = joblib.load(latest_model)
            print(f"Loaded model: {latest_model.name}")

        self.metrics_dir = METRICS_DIR
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.plots_dir = PLOTS_DIR
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        with open(FEATURE_NAMES_FILE, "r") as f:
            self.feature_names = [line.strip() for line in f.readlines()]

    def evaluate_performance(self, X, y_true, period_name="production", threshold=0.5):
        """
        Comprehensive performance evaluation

        Args:
            X: Features
            y_true: True labels
            period_name: Name of evaluation period (e.g., "week_1", "production")
            threshold: Classification threshold

        Returns:
            dict with all metrics
        """

        print(f"\n{'='*60}")
        print(f"EVALUATING: {period_name}")
        print(f"{'='*60}")

        # Make predictions
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "period": period_name,
            "threshold": threshold,
            "n_samples": len(X),
            "n_fraud": int(y_true.sum()),
            "n_normal": int((1 - y_true).sum()),
            "fraud_rate": float(y_true.mean()),
        }

        # Classification metrics
        metrics.update(
            {
                "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
                "pr_auc": float(average_precision_score(y_true, y_pred_proba)),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            }
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update(
            {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "fpr": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                "fnr": float(fn / (fn + tp)) if (fn + tp) > 0 else 0,
            }
        )

        # Prediction distribution
        metrics["prediction_stats"] = {
            "mean_fraud_proba": float(y_pred_proba.mean()),
            "std_fraud_proba": float(y_pred_proba.std()),
            "min_fraud_proba": float(y_pred_proba.min()),
            "max_fraud_proba": float(y_pred_proba.max()),
            "median_fraud_proba": float(np.median(y_pred_proba)),
        }

        # Business metrics (assuming costs)
        cost_per_fraud = 1000  # Average fraud loss
        cost_per_false_alarm = 5  # Customer service cost

        # Convert to native Python types for JSON serialization
        metrics["business_impact"] = {
            "fraud_caught_value": int(tp * cost_per_fraud),
            "fraud_missed_value": int(fn * cost_per_fraud),
            "false_alarm_cost": int(fp * cost_per_false_alarm),
            "total_cost": int((fn * cost_per_fraud) + (fp * cost_per_false_alarm)),
            "savings": int(tp * cost_per_fraud),
            "net_value": int((tp * cost_per_fraud) - (fp * cost_per_false_alarm)),
        }

        # Print summary
        print(f"\nDataset:")
        print(f"  Total samples: {metrics['n_samples']:,}")
        print(f"  Fraud cases: {metrics['n_fraud']:,} ({metrics['fraud_rate']:.2%})")
        print(f"  Normal cases: {metrics['n_normal']:,}")

        print(f"\nPerformance Metrics:")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  FPR:       {metrics['fpr']:.4f} ({metrics['fpr']*100:.2f}%)")

        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {tp:>6,}  (Fraud caught)")
        print(f"  False Positives: {fp:>6,}  (False alarms)")
        print(f"  False Negatives: {fn:>6,}  (Fraud missed)")
        print(f"  True Negatives:  {tn:>6,}  (Correct normal)")

        print(f"\nBusiness Impact:")
        print(f"  Fraud prevented: ${metrics['business_impact']['fraud_caught_value']:>10,}")
        print(f"  Fraud losses:    ${metrics['business_impact']['fraud_missed_value']:>10,}")
        print(f"  False alarm cost:${metrics['business_impact']['false_alarm_cost']:>10,}")
        print(f"  Net value:       ${metrics['business_impact']['net_value']:>10,}")

        # Save metrics
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_path = self.metrics_dir / f"metrics_{period_name}_{timestamp}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Metrics saved: {metrics_path.name}")

        return metrics

    def compare_performance(self, baseline_metrics, current_metrics, threshold=0.05):
        """
        Compare current performance against baseline
        Alert if significant degradation detected

        Args:
            baseline_metrics: dict with baseline metrics
            current_metrics: dict with current metrics
            threshold: degradation threshold (default 5%)

        Returns:
            tuple (is_stable, alerts)
        """

        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline: {baseline_metrics['period']}")
        print(f"Current:  {current_metrics['period']}")
        print(f"Degradation threshold: {threshold*100}%")

        metrics_to_compare = ["roc_auc", "precision", "recall", "f1", "fpr"]

        alerts = []

        print(f"\n{'Metric':<12} {'Baseline':<10} {'Current':<10} {'Change':<12} {'Status':<15}")
        print("-" * 65)

        for metric in metrics_to_compare:
            baseline_val = baseline_metrics[metric]
            current_val = current_metrics[metric]
            change = current_val - baseline_val
            pct_change = (change / baseline_val) * 100 if baseline_val > 0 else 0

            # For FPR, lower is better, so invert the check
            if metric == "fpr":
                is_degraded = change > threshold  # Increase in FPR is bad
                is_improved = change < -threshold
            else:
                is_degraded = change < -threshold  # Decrease in other metrics is bad
                is_improved = change > threshold

            if is_degraded:
                status = "⚠️ DEGRADED"
                alerts.append(
                    {
                        "metric": metric,
                        "baseline": baseline_val,
                        "current": current_val,
                        "change": change,
                        "pct_change": pct_change,
                        "severity": "HIGH" if abs(pct_change) > 10 else "MEDIUM",
                    }
                )
            elif is_improved:
                status = "✓ IMPROVED"
            else:
                status = "→ STABLE"

            print(
                f"{metric:<12} {baseline_val:>9.4f} {current_val:>9.4f} "
                f"{change:>+6.4f} ({pct_change:>+5.1f}%) {status:<15}"
            )

        # Overall assessment
        print(f"\n{'='*60}")
        if alerts:
            severity = max([a["severity"] for a in alerts])
            print(f"STATUS: ⚠️ PERFORMANCE DEGRADATION DETECTED")
            print(f"SEVERITY: {severity}")
            print(f"AFFECTED METRICS: {len(alerts)}")

            print(f"\nDegraded Metrics:")
            for alert in alerts:
                print(
                    f"  • {alert['metric']}: {alert['pct_change']:+.1f}% "
                    f"({alert['baseline']:.4f} → {alert['current']:.4f})"
                )

            print(f"\n⚠️ RECOMMENDED ACTIONS:")
            print(f"  1. Investigate root cause")
            print(f"  2. Check for data drift")
            print(f"  3. Consider retraining model")
            print(f"  4. Review recent changes (code, features, data)")

            return False, alerts
        else:
            print(f"STATUS: ✓ PERFORMANCE IS STABLE")
            print(f"All metrics within acceptable thresholds")
            return True, []

    def plot_metrics_over_time(self):
        """
        Plot metrics over time from saved JSON files
        """

        # Load all metrics files
        metrics_files = sorted(self.metrics_dir.glob("metrics_*.json"))

        if len(metrics_files) < 2:
            print("Need at least 2 metric files to plot trends")
            return

        all_metrics = []
        for file in metrics_files:
            with open(file, "r") as f:
                metrics = json.load(f)
                metrics["timestamp"] = pd.to_datetime(metrics["timestamp"])
                all_metrics.append(metrics)

        df = pd.DataFrame(all_metrics)
        df = df.sort_values("timestamp")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Model Performance Over Time", fontsize=16, fontweight="bold")

        metrics_to_plot = [
            ("roc_auc", "ROC-AUC Score", "blue"),
            ("precision", "Precision", "green"),
            ("recall", "Recall", "orange"),
            ("f1", "F1 Score", "purple"),
        ]

        for ax, (metric, title, color) in zip(axes.flat, metrics_to_plot):
            ax.plot(df["timestamp"], df[metric], marker="o", color=color, linewidth=2, markersize=8)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.set_xlabel("Time")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)

            # Add mean line
            mean_val = df[metric].mean()
            ax.axhline(y=mean_val, color="red", linestyle="--", alpha=0.5, label=f"Mean: {mean_val:.3f}")

            # Add acceptable range (mean ± 5%)
            ax.axhline(y=mean_val * 0.95, color="orange", linestyle=":", alpha=0.3, label="±5% threshold")
            ax.axhline(y=mean_val * 1.05, color="orange", linestyle=":", alpha=0.3)

            ax.legend(loc="best", fontsize=8)
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        plot_path = self.plots_dir / "performance_over_time.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Performance plot saved: {plot_path}")

        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, period_name="Current"):
        """
        Plot confusion matrix heatmap
        """

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        plt.title(f"Confusion Matrix - {period_name}", fontsize=14, fontweight="bold")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")

        plot_path = self.plots_dir / f"confusion_matrix_{period_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"✓ Confusion matrix saved: {plot_path}")

        plt.close()

    def generate_report(self, metrics):
        """
        Generate a human-readable performance report
        """

        report_path = self.metrics_dir / f"report_{metrics['period']}.txt"

        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write(f"FRAUD DETECTION MODEL PERFORMANCE REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Period: {metrics['period']}\n")
            f.write(f"Generated: {metrics['timestamp']}\n")
            f.write(f"Samples Evaluated: {metrics['n_samples']:,}\n")
            f.write("\n")

            f.write("CLASSIFICATION METRICS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            f.write(f"PR-AUC:    {metrics['pr_auc']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"F1 Score:  {metrics['f1']:.4f}\n")
            f.write("\n")

            f.write("CONFUSION MATRIX:\n")
            f.write("-" * 60 + "\n")
            f.write(f"True Positives:  {metrics['true_positives']:>6,}\n")
            f.write(f"False Positives: {metrics['false_positives']:>6,}\n")
            f.write(f"False Negatives: {metrics['false_negatives']:>6,}\n")
            f.write(f"True Negatives:  {metrics['true_negatives']:>6,}\n")
            f.write("\n")

            f.write("BUSINESS IMPACT:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Fraud Prevented: ${metrics['business_impact']['fraud_caught_value']:>10,}\n")
            f.write(f"Fraud Losses:    ${metrics['business_impact']['fraud_missed_value']:>10,}\n")
            f.write(f"False Alarm Cost:${metrics['business_impact']['false_alarm_cost']:>10,}\n")
            f.write(f"Net Value:       ${metrics['business_impact']['net_value']:>10,}\n")

        print(f"✓ Report saved: {report_path}")


def simulate_production_periods():
    """
    Simulate multiple production periods to show metrics over time
    """

    df = pd.read_csv(TRANSACTIONS_FILE)

    # Use test set
    test_start = int(len(df) * 0.8)
    test_data = df.iloc[test_start:].copy()

    # Split into 4 weekly periods
    period_size = len(test_data) // 4

    periods = []
    for i in range(4):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < 3 else len(test_data)
        period_data = test_data.iloc[start_idx:end_idx].copy()

        # Simulate gradual degradation
        if i > 0:
            # Add some noise to amounts (simulating drift)
            noise_factor = 1 + (i * 0.05)  # 0%, 5%, 10%, 15% increase
            period_data["amount"] = period_data["amount"] * noise_factor
            period_data["amount_log"] = np.log1p(period_data["amount"])

        periods.append({"name": f"week_{i+1}", "data": period_data})

    return periods


def main():
    """Run performance monitoring"""

    print("=" * 60)
    print("MODEL PERFORMANCE MONITORING")
    print("=" * 60)

    # Initialize monitor
    monitor = PerformanceMonitor()

    # Load feature names
    with open(FEATURE_NAMES_FILE, "r") as f:
        feature_names = [line.strip() for line in f.readlines()]

    # Simulate production periods
    print("\nSimulating production data over 4 weeks...")
    periods = simulate_production_periods()

    all_metrics = []

    # Evaluate each period
    for period in periods:
        X = period["data"][feature_names]
        y = period["data"]["is_fraud"]

        metrics = monitor.evaluate_performance(X, y, period_name=period["name"])
        all_metrics.append(metrics)

        # Plot confusion matrix for each period
        y_pred = monitor.model.predict(X)
        monitor.plot_confusion_matrix(y, y_pred, period_name=period["name"])

        # Generate report
        monitor.generate_report(metrics)

    # Compare periods
    print("\n" + "=" * 60)
    print("COMPARING PERIODS")
    print("=" * 60)

    baseline = all_metrics[0]  # Week 1 as baseline

    for i, current in enumerate(all_metrics[1:], 2):
        print(f"\n--- Week {i} vs Baseline (Week 1) ---")
        is_stable, alerts = monitor.compare_performance(baseline, current, threshold=0.05)

    # Plot metrics over time
    monitor.plot_metrics_over_time()

    print("\n" + "=" * 60)
    print("MONITORING COMPLETE")
    print("=" * 60)
    print(f"✓ {len(all_metrics)} periods evaluated")
    print(f"✓ Metrics saved to: /app/monitoring/metrics/")
    print(f"✓ Plots saved to: /app/monitoring/plots/")

    return all_metrics


if __name__ == "__main__":
    main()

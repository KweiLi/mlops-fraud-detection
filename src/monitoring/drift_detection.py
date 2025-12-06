import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime
import json

# At the top after imports:
from src.utils.paths import (
    TRANSACTIONS_FILE,
    FEATURE_NAMES_FILE,
    REPORTS_DIR,
)


class DriftDetector:
    """
    Drift detection using Kolmogorov-Smirnov test
    No external dependencies - just scipy (already installed)
    """

    # In __init__:
    def __init__(self, reference_data_path=None):
        if reference_data_path is None:
            reference_data_path = TRANSACTIONS_FILE

        self.reference_data = pd.read_csv(reference_data_path)
        self.reports_dir = REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        with open(FEATURE_NAMES_FILE, "r") as f:
            self.feature_names = [line.strip() for line in f.readlines()]

    def detect_drift(self, current_data, threshold=0.05, save_report=True):
        """
        Detect drift using Kolmogorov-Smirnov test

        The KS test measures the maximum distance between cumulative
        distribution functions of two samples. If distributions differ
        significantly, we have drift.

        Args:
            current_data: DataFrame with current production data
            threshold: p-value threshold (default 0.05)
            save_report: Whether to save JSON report

        Returns:
            dict with drift summary
        """
        print("\nRunning drift detection...")

        ref_features = self.reference_data[self.feature_names]
        curr_features = current_data[self.feature_names]

        drift_results = []

        for feature in self.feature_names:
            try:
                # Kolmogorov-Smirnov test
                statistic, p_value = stats.ks_2samp(ref_features[feature].dropna(), curr_features[feature].dropna())

                # Drift detected if p-value < threshold
                drift_detected = p_value < threshold

                # Calculate magnitude of change
                ref_mean = ref_features[feature].mean()
                curr_mean = curr_features[feature].mean()
                ref_std = ref_features[feature].std()
                curr_std = curr_features[feature].std()

                mean_change = ((curr_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0

                drift_results.append(
                    {
                        "feature": feature,
                        "ks_statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift_detected": bool(drift_detected),
                        "ref_mean": float(ref_mean),
                        "curr_mean": float(curr_mean),
                        "ref_std": float(ref_std),
                        "curr_std": float(curr_std),
                        "mean_change_pct": float(mean_change),
                    }
                )
            except Exception as e:
                print(f"  Warning: Could not test {feature}: {e}")

        # Summary statistics
        n_drifted = sum(1 for r in drift_results if r["drift_detected"])
        share_drifted = n_drifted / len(self.feature_names) if self.feature_names else 0

        summary = {
            "timestamp": datetime.utcnow().isoformat(),
            "n_features_checked": len(self.feature_names),
            "n_drifted_features": n_drifted,
            "share_drifted": share_drifted,
            "drift_detected": n_drifted > 0,
            "threshold": threshold,
            "features": drift_results,
        }

        # Print results
        print(f"\n{'='*60}")
        print("DRIFT DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Features checked: {len(self.feature_names)}")
        print(f"Drifted features: {n_drifted} ({share_drifted:.1%})")
        print(f"Significance level: α = {threshold}")

        if n_drifted > 0:
            print(f"\n⚠️  DRIFT WARNING: {n_drifted} features have drifted!")
            print("\nTop drifted features (by KS statistic):")

            sorted_drifts = sorted(
                [r for r in drift_results if r["drift_detected"]], key=lambda x: x["ks_statistic"], reverse=True
            )

            print(f"\n{'Feature':<30} {'KS Stat':<10} {'p-value':<12} {'Mean Change':<12}")
            print("-" * 70)

            for i, result in enumerate(sorted_drifts[:10], 1):
                print(
                    f"{result['feature']:<30} "
                    f"{result['ks_statistic']:<10.4f} "
                    f"{result['p_value']:<12.6f} "
                    f"{result['mean_change_pct']:>+10.1f}%"
                )

            if len(sorted_drifts) > 10:
                print(f"... and {len(sorted_drifts) - 10} more")

            print("\n⚠️  RECOMMENDED ACTIONS:")
            print("   1. Review drifted features")
            print("   2. Investigate root cause (data quality? behavior change?)")
            print("   3. Consider retraining model with recent data")
            print("   4. Update feature engineering if needed")
        else:
            print(f"\n✓ No significant drift detected")
            print("   Model is operating within expected parameters")

        # Save detailed results
        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = self.reports_dir / f"drift_summary_{timestamp}.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n✓ Drift report saved: {summary_path}")
            summary["report_path"] = str(summary_path)

        return summary

    def analyze_feature_drift(self, current_data, top_n=10):
        """
        Detailed statistical analysis of top drifting features
        """

        ref_features = self.reference_data[self.feature_names]
        curr_features = current_data[self.feature_names]

        feature_analysis = []

        for feature in self.feature_names:
            statistic, p_value = stats.ks_2samp(ref_features[feature].dropna(), curr_features[feature].dropna())

            feature_analysis.append(
                {
                    "feature": feature,
                    "ks_statistic": statistic,
                    "p_value": p_value,
                    "ref_mean": ref_features[feature].mean(),
                    "curr_mean": curr_features[feature].mean(),
                    "ref_std": ref_features[feature].std(),
                    "curr_std": curr_features[feature].std(),
                    "ref_min": ref_features[feature].min(),
                    "ref_max": ref_features[feature].max(),
                    "curr_min": curr_features[feature].min(),
                    "curr_max": curr_features[feature].max(),
                }
            )

        # Sort by KS statistic
        feature_analysis.sort(key=lambda x: x["ks_statistic"], reverse=True)

        print(f"\n{'='*60}")
        print(f"TOP {top_n} FEATURES BY DRIFT MAGNITUDE")
        print(f"{'='*60}")
        print(f"\n{'Feature':<30} {'KS':<8} {'Ref Mean':<12} {'Curr Mean':<12} {'Change':<10}")
        print("-" * 75)

        for analysis in feature_analysis[:top_n]:
            change_pct = (
                ((analysis["curr_mean"] - analysis["ref_mean"]) / analysis["ref_mean"] * 100)
                if analysis["ref_mean"] != 0
                else 0
            )

            print(
                f"{analysis['feature']:<30} "
                f"{analysis['ks_statistic']:<8.4f} "
                f"{analysis['ref_mean']:<12.2f} "
                f"{analysis['curr_mean']:<12.2f} "
                f"{change_pct:>+9.1f}%"
            )

        return feature_analysis


def simulate_production_data(n_samples=10000):
    """
    Simulate production data with drift
    In real production, this would come from API prediction logs
    """
    print("\nSimulating production data with drift...")

    df = pd.read_csv(TRANSACTIONS_FILE)

    # Use test set
    test_start = int(len(df) * 0.8)
    production_data = df.iloc[test_start : test_start + n_samples].copy()

    # Simulate realistic drift scenarios

    # 1. Inflation: amounts increase by 15%
    production_data["amount"] = production_data["amount"] * 1.15
    production_data["amount_log"] = np.log1p(production_data["amount"])

    # 2. Behavioral change: more night transactions
    night_mask = production_data["is_night"] == 0
    flip_indices = np.random.choice(
        production_data[night_mask].index,
        size=min(int(len(production_data) * 0.1), len(production_data[night_mask])),
        replace=False,
    )
    production_data.loc[flip_indices, "is_night"] = 1

    # 3. Economic shift: more online shopping
    online_mask = production_data["is_online"] == 0
    flip_online = np.random.choice(
        production_data[online_mask].index,
        size=min(int(len(production_data) * 0.05), len(production_data[online_mask])),
        replace=False,
    )
    production_data.loc[flip_online, "is_online"] = 1

    print(f"  Generated {len(production_data):,} production samples")
    print(f"  Applied drift:")
    print(f"    - Amount +15% (inflation)")
    print(f"    - Night transactions +10%")
    print(f"    - Online transactions +5%")

    return production_data


def main():
    """Run drift detection monitoring"""

    print("=" * 60)
    print("DATA DRIFT DETECTION MONITORING")
    print("=" * 60)
    print("\nUsing Kolmogorov-Smirnov Test")
    print("H0: Reference and current data come from same distribution")
    print("H1: Distributions differ significantly")

    # Initialize detector
    detector = DriftDetector()

    # Simulate production data (in production: load from database)
    production_data = simulate_production_data(n_samples=10000)

    # Detect drift
    summary = detector.detect_drift(production_data, threshold=0.05, save_report=True)

    # Detailed feature analysis
    print("\n" + "=" * 60)
    feature_analysis = detector.analyze_feature_drift(production_data, top_n=15)

    # Final recommendation
    print("\n" + "=" * 60)
    if summary["drift_detected"]:
        print("📊 MONITORING ALERT")
        print("=" * 60)
        print(f"Status: ⚠️  DRIFT DETECTED")
        print(f"Severity: {'HIGH' if summary['share_drifted'] > 0.2 else 'MEDIUM'}")
        print(f"Affected features: {summary['n_drifted_features']}/{summary['n_features_checked']}")
        print("\nNext steps:")
        print("  1. Review drift report for details")
        print("  2. Retrain model with recent data")
        print("  3. Update monitoring thresholds if needed")
    else:
        print("📊 MONITORING STATUS")
        print("=" * 60)
        print("Status: ✓ HEALTHY")
        print("All features within expected distribution")

    return summary


if __name__ == "__main__":
    main()

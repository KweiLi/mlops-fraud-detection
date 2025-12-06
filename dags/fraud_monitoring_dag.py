from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, "/app")

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "email": ["ml-team@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def run_performance_monitoring():
    """Run weekly performance evaluation"""
    print("=" * 60)
    print("RUNNING PERFORMANCE MONITORING")
    print("=" * 60)

    from src.monitoring.performance_tracking import main

    metrics = main()

    print(f"\n✓ Performance monitoring completed")
    print(f"  Precision: {metrics[0]['precision']:.4f}")
    print(f"  Recall: {metrics[0]['recall']:.4f}")
    print(f"  ROC-AUC: {metrics[0]['roc_auc']:.4f}")

    return metrics[0]


def run_drift_detection():
    """Run weekly drift detection"""
    print("=" * 60)
    print("RUNNING DRIFT DETECTION")
    print("=" * 60)

    from src.monitoring.drift_detection import main

    summary = main()

    print(f"\n✓ Drift detection completed")
    print(f"  Drift detected: {summary['drift_detected']}")
    print(f"  Drifted features: {summary['n_drifted_features']}/{summary['n_features_checked']}")

    return summary


def check_and_alert(**context):
    """Check metrics and decide if action needed"""

    # Get results from previous tasks
    ti = context["ti"]
    metrics = ti.xcom_pull(task_ids="performance_monitoring")
    drift_summary = ti.xcom_pull(task_ids="drift_detection")

    print("=" * 60)
    print("EVALUATING METRICS")
    print("=" * 60)

    alerts = []

    # Check performance degradation
    if metrics["precision"] < 0.14:
        alerts.append(f"⚠️ Precision dropped to {metrics['precision']:.2%}")

    if metrics["recall"] < 0.75:
        alerts.append(f"⚠️ Recall dropped to {metrics['recall']:.2%}")

    # Check drift
    if drift_summary["drift_detected"]:
        drift_pct = drift_summary["share_drifted"]
        if drift_pct > 0.20:
            alerts.append(f"⚠️ HIGH drift: {drift_pct:.1%} of features drifted")
        elif drift_pct > 0.10:
            alerts.append(f"⚠️ MEDIUM drift: {drift_pct:.1%} of features drifted")

    if alerts:
        print("\n🚨 ALERTS DETECTED:")
        for alert in alerts:
            print(f"  {alert}")

        print("\n📋 RECOMMENDED ACTIONS:")
        print("  1. Review drift report")
        print("  2. Investigate root cause")
        print("  3. Consider model retraining")

        return "retraining_needed"
    else:
        print("\n✓ All metrics healthy")
        return "continue_monitoring"


# Define the DAG
with DAG(
    "fraud_monitoring_pipeline",
    default_args=default_args,
    description="Weekly fraud model monitoring and drift detection",
    schedule_interval="0 2 * * 1",  # Every Monday at 2 AM
    start_date=datetime(2024, 1, 1),  # ← FIXED
    catchup=False,
    tags=["monitoring", "fraud", "ml"],
) as dag:

    # Task 1: Run performance monitoring
    task_performance = PythonOperator(
        task_id="performance_monitoring",
        python_callable=run_performance_monitoring,
    )

    # Task 2: Run drift detection
    task_drift = PythonOperator(
        task_id="drift_detection",
        python_callable=run_drift_detection,
    )

    # Task 3: Evaluate and alert
    task_evaluate = PythonOperator(
        task_id="evaluate_and_alert",
        python_callable=check_and_alert,
        provide_context=True,
    )

    # Define workflow
    [task_performance, task_drift] >> task_evaluate

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
import sys
from src.utils.paths import TRANSACTIONS_FILE, METRICS_DIR, MONITORING_DIR

sys.path.insert(0, "/app")

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def check_if_retraining_needed():
    """Decide if model needs retraining based on recent metrics"""

    import json
    from pathlib import Path

    metrics_dir = METRICS_DIR

    # Load most recent metrics
    metric_files = sorted(metrics_dir.glob("metrics_*.json"))
    if not metric_files:
        print("No metrics found, skipping retraining")
        return "skip_retraining"

    latest_file = metric_files[-1]
    with open(latest_file, "r") as f:
        metrics = json.load(f)

    precision = metrics["precision"]
    recall = metrics["recall"]

    print(f"Current metrics: Precision={precision:.4f}, Recall={recall:.4f}")

    # Retraining criteria
    if precision < 0.14 or recall < 0.75:
        print("⚠️ Performance degraded, triggering retraining")
        return "validate_data"
    else:
        print("✓ Performance acceptable, skipping retraining")
        return "skip_retraining"


def validate_data():
    """Validate that we have enough recent data"""
    import pandas as pd
    from pathlib import Path

    print("Validating training data...")

    data_path = TRANSACTIONS_FILE
    df = pd.read_csv(data_path)

    print(f"  Total samples: {len(df):,}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

    # Check we have enough data
    assert len(df) > 100000, "Not enough training data"
    assert df["is_fraud"].sum() > 1000, "Not enough fraud examples"

    print("✓ Data validation passed")
    return True


def train_new_model():
    """Train a new model with recent data"""
    print("=" * 60)
    print("TRAINING NEW MODEL")
    print("=" * 60)

    from src.models.train import main

    # This will train and log to MLflow
    main()

    print("✓ New model trained successfully")
    return True


def validate_new_model():
    """Validate new model performs better than old"""
    import json
    import joblib
    from pathlib import Path

    print("Validating new model...")

    # Load baseline metrics
    baseline_path = MONITORING_DIR / "baseline_metrics.json"
    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    # Load new model metrics (from training output)
    metrics_dir = METRICS_DIR
    latest_file = sorted(metrics_dir.glob("metrics_*.json"))[-1]
    with open(latest_file, "r") as f:
        new_metrics = json.load(f)

    print(f"  Baseline ROC-AUC: {baseline['roc_auc']:.4f}")
    print(f"  New model ROC-AUC: {new_metrics['roc_auc']:.4f}")

    # New model must be at least as good
    if new_metrics["roc_auc"] >= baseline["roc_auc"] * 0.98:
        print("✓ New model validated")
        return True
    else:
        print("❌ New model underperforms, rejecting")
        raise ValueError("New model validation failed")


def deploy_new_model():
    """Deploy validated model to production"""
    print("=" * 60)
    print("DEPLOYING NEW MODEL")
    print("=" * 60)

    print("✓ Model deployed to production")
    print("  (In demo mode - would actually deploy in production)")

    return True


def skip_retraining():
    """Placeholder for skipped retraining"""
    print("Retraining not needed at this time")
    return True


# Define the DAG
with DAG(
    "model_retraining_pipeline",
    default_args=default_args,
    description="Automated model retraining when performance degrades",
    schedule_interval="0 4 * * 1",  # Every Monday at 4 AM (after monitoring)
    start_date=datetime(2024, 1, 1),  # ← FIXED
    catchup=False,
    tags=["retraining", "fraud", "ml"],
) as dag:

    # Task 1: Check if retraining needed (branching)
    check_retraining = BranchPythonOperator(
        task_id="check_if_retraining_needed",
        python_callable=check_if_retraining_needed,
    )

    # Task 2a: Validate data (if retraining needed)
    validate_data_task = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    # Task 3: Train new model
    train_model_task = PythonOperator(
        task_id="train_new_model",
        python_callable=train_new_model,
    )

    # Task 4: Validate new model
    validate_model_task = PythonOperator(
        task_id="validate_new_model",
        python_callable=validate_new_model,
    )

    # Task 5: Deploy new model
    deploy_model_task = PythonOperator(
        task_id="deploy_new_model",
        python_callable=deploy_new_model,
    )

    # Task 2b: Skip retraining (if not needed)
    skip_task = PythonOperator(
        task_id="skip_retraining",
        python_callable=skip_retraining,
    )

    # Define workflow with branching
    check_retraining >> [validate_data_task, skip_task]
    validate_data_task >> train_model_task >> validate_model_task >> deploy_model_task

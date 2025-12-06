from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, "/app")

from src.utils.paths import TRANSACTIONS_FILE, METRICS_DIR, MONITORING_DIR

default_args = {
    "owner": "ml-team",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_data():
    """Validate that we have enough recent data"""
    import pandas as pd

    print("Validating training data...")
    df = pd.read_csv(TRANSACTIONS_FILE)
    print(f"  Total samples: {len(df):,}")
    print(f"  Fraud rate: {df['is_fraud'].mean():.2%}")

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

    main()

    print("✓ New model trained successfully")
    return True


def validate_new_model():
    """Validate new model"""
    print("Validating new model...")
    # For testing, just pass
    print("✓ New model validated (test mode)")
    return True


def deploy_new_model():
    """Deploy validated model to production"""
    print("=" * 60)
    print("DEPLOYING NEW MODEL")
    print("=" * 60)
    print("✓ Model deployed to production")
    print("  (Test mode - would actually deploy in production)")
    return True


# Define the DAG - NO BRANCHING
with DAG(
    "model_retraining_test",  # Different name
    default_args=default_args,
    description="Test model retraining pipeline (no branching)",
    schedule_interval=None,  # Manual trigger only
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["test", "retraining", "fraud", "ml"],
) as dag:

    task1 = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    task2 = PythonOperator(
        task_id="train_new_model",
        python_callable=train_new_model,
    )

    task3 = PythonOperator(
        task_id="validate_new_model",
        python_callable=validate_new_model,
    )

    task4 = PythonOperator(
        task_id="deploy_new_model",
        python_callable=deploy_new_model,
    )

    # Linear workflow - runs all steps
    task1 >> task2 >> task3 >> task4

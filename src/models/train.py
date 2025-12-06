import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import xgboost as xgb
from datetime import datetime
import json
import joblib
from src.utils.paths import TRANSACTIONS_FILE, FEATURE_NAMES_FILE, MODELS_DIR


class FraudModelTrainer:
    """Train fraud detection model with MLflow tracking"""

    def __init__(self, experiment_name="fraud_detection"):
        self.experiment_name = experiment_name
        self.model = None
        self.feature_names = None

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

    def load_data(self, data_path, feature_names_path):
        """Load processed data and feature names"""
        print("Loading data...")
        df = pd.read_csv(data_path)
        print(f"  Loaded {len(df):,} transactions")

        # Load feature names
        with open(feature_names_path, "r") as f:
            self.feature_names = [line.strip() for line in f.readlines()]

        print(f"  Features: {len(self.feature_names)}")

        return df

    def prepare_data(self, df):
        """Prepare train/val/test splits"""
        print("\nPreparing data splits...")

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Time-based split
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.8)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]

        X_train = train_df[self.feature_names]
        y_train = train_df["is_fraud"]

        X_val = val_df[self.feature_names]
        y_val = val_df["is_fraud"]

        X_test = test_df[self.feature_names]
        y_test = test_df["is_fraud"]

        print(f"  Train set: {len(X_train):,} ({y_train.sum():,} fraud, {y_train.mean()*100:.2f}%)")
        print(f"  Val set:   {len(X_val):,} ({y_val.sum():,} fraud, {y_val.mean()*100:.2f}%)")
        print(f"  Test set:  {len(X_test):,} ({y_test.sum():,} fraud, {y_test.mean()*100:.2f}%)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_model(self, X_train, y_train, X_val, y_val, params=None):
        """Train XGBoost model with MLflow tracking"""

        if params is None:
            params = {
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 200,
                "objective": "binary:logistic",
                "scale_pos_weight": len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "eval_metric": "auc",
                "early_stopping_rounds": 20,
            }

        print("\nTraining model...")
        print(f"  Model: XGBoost")
        print(f"  Parameters:")
        for k, v in params.items():
            print(f"    {k}: {v}")

        # Start MLflow run
        with mlflow.start_run(run_name=f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("n_features", len(self.feature_names))
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("fraud_rate_train", y_train.mean())

            # Train model
            self.model = xgb.XGBClassifier(**params)

            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            print(f"  ✓ Training complete")

            # Log model
            mlflow.sklearn.log_model(self.model, "model", input_example=X_train.iloc[:5])

            # Save feature names
            mlflow.log_dict({"features": self.feature_names}, "feature_names.json")

            return self.model

    def evaluate_model(self, X_test, y_test, dataset_name="test"):
        """Comprehensive model evaluation"""
        print(f"\nEvaluating on {dataset_name} set...")

        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)

        # Metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba),
        }

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        metrics.update(
            {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
                "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0,
            }
        )

        # Log metrics to MLflow
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"{dataset_name}_{metric_name}", value)

        # Print results
        print(f"\n{'='*60}")
        print(f"{dataset_name.upper()} SET RESULTS")
        print(f"{'='*60}")
        print(f"ROC-AUC Score: {metrics['roc_auc']:.4f}")
        print(f"PR-AUC Score:  {metrics['pr_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")
        print(f"\nMetrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  FPR:       {metrics['false_positive_rate']:.4f}")

        return metrics

    def get_feature_importance(self, top_n=20):
        """Get and display feature importance"""
        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print(f"\n{'='*60}")
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print(f"{'='*60}")

        for idx, row in importance_df.head(top_n).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")

        # Log to MLflow
        mlflow.log_dict(importance_df.to_dict("records"), "feature_importance.json")

        return importance_df

    def save_model(self, output_dir="models"):
        """Save model locally"""
        if output_dir is None:
            output_dir = MODELS_DIR
        Path(output_dir).mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = Path(output_dir) / f"fraud_model_{timestamp}.joblib"

        joblib.dump(self.model, model_path)
        print(f"\n✓ Model saved to: {model_path}")

        return model_path


def main():
    """Main training pipeline"""

    print("=" * 60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)

    # Initialize trainer
    trainer = FraudModelTrainer(experiment_name="fraud_detection")

    # Load data
    df = trainer.load_data(data_path=str(TRANSACTIONS_FILE), feature_names_path=str(FEATURE_NAMES_FILE))

    # Prepare data splits
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)

    # Train model
    model = trainer.train_model(X_train, y_train, X_val, y_val)

    # Evaluate
    val_metrics = trainer.evaluate_model(X_val, y_val, dataset_name="validation")
    test_metrics = trainer.evaluate_model(X_test, y_test, dataset_name="test")

    # Feature importance
    importance_df = trainer.get_feature_importance(top_n=20)

    # Save model
    model_path = trainer.save_model()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"✓ Model saved to: {model_path}")
    print(f"✓ View results: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()

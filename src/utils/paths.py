"""
Environment-aware path utilities
Works in: local development, Docker containers, CI/CD runners
"""

from pathlib import Path
import os


def get_project_root():
    """
    Detect project root based on environment

    Returns:
        Path: Absolute path to project root

    Environments:
        - Docker: /app
        - Local/CI/CD: Detect from this file's location
    """
    # Check if running in Docker
    # Docker has /app/models, /app/data, etc.
    docker_root = Path("/app")
    if docker_root.exists() and (docker_root / "models").exists():
        print(f"[PATH] Detected Docker environment: {docker_root}")
        return docker_root

    # Local/CI/CD: Navigate up from src/utils/paths.py to project root
    # paths.py is in: project_root/src/utils/paths.py
    # So go up 3 levels: paths.py -> utils -> src -> project_root
    local_root = Path(__file__).resolve().parent.parent.parent
    print(f"[PATH] Detected local environment: {local_root}")
    return local_root


# Initialize project root
PROJECT_ROOT = get_project_root()

# Define all project paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = PROJECT_ROOT / "monitoring"
REPORTS_DIR = MONITORING_DIR / "reports"
METRICS_DIR = MONITORING_DIR / "metrics"
PLOTS_DIR = MONITORING_DIR / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"

# Define file paths
FEATURE_NAMES_FILE = PROCESSED_DATA_DIR / "feature_names.txt"
TRANSACTIONS_FILE = PROCESSED_DATA_DIR / "transactions_features.csv"


def get_latest_model():
    """
    Get path to the most recent model file

    Returns:
        Path: Path to latest model

    Raises:
        FileNotFoundError: If no model found
    """
    model_files = list(MODELS_DIR.glob("fraud_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model found in {MODELS_DIR}")

    latest = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"[PATH] Latest model: {latest.name}")
    return latest


def ensure_directories():
    """Create all necessary directories if they don't exist"""
    directories = [
        MODELS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        METRICS_DIR,
        PLOTS_DIR,
        LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"[PATH] All directories verified")


# Print paths on import (helpful for debugging)
if __name__ != "__main__":
    print(f"[PATH] Project root: {PROJECT_ROOT}")


if __name__ == "__main__":
    # Test script
    print("=" * 60)
    print("PATH UTILITY TEST")
    print("=" * 60)

    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Exists: {PROJECT_ROOT.exists()}")

    print(f"\nKey Directories:")
    print(f"  Models: {MODELS_DIR} (exists: {MODELS_DIR.exists()})")
    print(f"  Data: {DATA_DIR} (exists: {DATA_DIR.exists()})")
    print(f"  Monitoring: {MONITORING_DIR} (exists: {MONITORING_DIR.exists()})")

    print(f"\nKey Files:")
    print(f"  Features: {FEATURE_NAMES_FILE} (exists: {FEATURE_NAMES_FILE.exists()})")
    print(f"  Transactions: {TRANSACTIONS_FILE} (exists: {TRANSACTIONS_FILE.exists()})")

    try:
        latest_model = get_latest_model()
        print(f"  Latest Model: {latest_model} ✓")
    except FileNotFoundError as e:
        print(f"  Latest Model: {e} ✗")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

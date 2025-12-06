from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from src.monitoring.drift_detection import DriftDetector
from src.monitoring.performance_tracking import PerformanceMonitor

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API", description="Real-time fraud detection for credit card transactions", version="1.0.0"
)

# Global model variable
model = None
feature_names = None


class Transaction(BaseModel):
    """Single transaction for prediction"""

    transaction_id: str = Field(..., example="TXN_00123456")
    customer_id: str = Field(..., example="CUST_001234")
    amount: float = Field(..., gt=0, example=125.50, description="Transaction amount in USD")
    merchant_category: str = Field(..., example="restaurant")
    merchant_id: str = Field(..., example="MERCH_0045")
    is_online: bool = Field(..., example=False)
    transaction_hour: int = Field(..., ge=0, le=23, example=14)

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TXN_00123456",
                "customer_id": "CUST_001234",
                "amount": 125.50,
                "merchant_category": "restaurant",
                "merchant_id": "MERCH_0045",
                "is_online": False,
                "transaction_hour": 14,
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    model_loaded: bool
    model_path: str
    timestamp: str


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, feature_names
    
    from src.utils.paths import get_latest_model, FEATURE_NAMES_FILE
    
    try:
        # Find latest model using path utility
        latest_model = get_latest_model()
        
        print(f"Loading model: {latest_model}")
        model = joblib.load(latest_model)
        print(f"✓ Model loaded successfully: {latest_model.name}")
        
    except Exception as e:
        print(f"✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        model = None
        return
    
    try:
        # Load feature names using path utility
        with open(FEATURE_NAMES_FILE, "r", encoding='utf-8') as f:
            feature_names = [line.strip() for line in f.readlines()]
        print(f"✓ Features loaded: {len(feature_names)}")
        
    except Exception as e:
        print(f"✗ ERROR loading features: {e}")
        import traceback
        traceback.print_exc()
        feature_names = []

    print(f"✓ Model loaded successfully")
    print(f"✓ Features: {len(feature_names)}")


@app.get("/monitor/health")
async def monitoring_health():
    """Check monitoring system health"""

    drift_reports = list(Path("monitoring/reports").glob("*.html"))
    metrics_files = list(Path("monitoring/metrics").glob("*.json"))

    return {
        "status": "healthy",
        "drift_reports": len(drift_reports),
        "metrics_files": len(metrics_files),
        "last_drift_check": max([f.stat().st_mtime for f in drift_reports]) if drift_reports else None,
        "last_metrics_update": max([f.stat().st_mtime for f in metrics_files]) if metrics_files else None,
    }


@app.post("/monitor/check_drift")
async def trigger_drift_check():
    """Manually trigger drift detection"""

    try:
        detector = DriftDetector()
        # In production, load recent predictions from database
        # For now, simulate with test data
        from src.monitoring.drift_detection import simulate_production_data

        production_data = simulate_production_data(n_samples=1000)

        summary = detector.detect_drift(production_data, save_report=True)

        return {
            "status": "success",
            "drift_detected": summary["drift_detected"],
            "drifted_features": summary["n_drifted_features"],
            "report_path": summary.get("report_path", "No report generated"),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "predict": "/predict", "batch_predict": "/batch_predict", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from src.utils.paths import MODELS_DIR, get_latest_model
    
    try:
        latest_model = get_latest_model()
        model_path = str(latest_model)
    except:
        model_path = "No model found"
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_path=model_path,
        timestamp=datetime.utcnow().isoformat(),
    )


def create_features(transaction: Transaction) -> pd.DataFrame:
    """
    Create features from transaction
    This is simplified - in production, you'd need historical data
    """

    # Basic features from transaction
    features = {
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "is_online": int(transaction.is_online),
        "hour": transaction.transaction_hour,
        "day_of_week": datetime.now().weekday(),
        "is_weekend": int(datetime.now().weekday() >= 5),
        "is_night": int((transaction.transaction_hour >= 22) or (transaction.transaction_hour <= 6)),
        "is_business_hours": int((transaction.transaction_hour >= 9) and (transaction.transaction_hour <= 17)),
        "amount_log": np.log1p(transaction.amount),
    }

    # Amount bins (simplified)
    if transaction.amount < 50:
        bin_name = "very_low"
    elif transaction.amount < 100:
        bin_name = "low"
    elif transaction.amount < 200:
        bin_name = "medium"
    elif transaction.amount < 500:
        bin_name = "high"
    else:
        bin_name = "very_high"

    # One-hot encode amount bins
    for bin_val in ["very_low", "low", "medium", "high", "very_high"]:
        features[f"amount_bin_{bin_val}"] = int(bin_val == bin_name)

    # Category one-hot encoding
    categories = [
        "grocery",
        "restaurant",
        "gas_station",
        "online_shopping",
        "pharmacy",
        "entertainment",
        "travel",
        "utilities",
    ]
    for cat in categories:
        features[f"category_{cat}"] = int(transaction.merchant_category == cat)

    # Placeholder values for features that need historical data
    # In production, these would come from a feature store
    features.update(
        {
            "customer_avg_amount": transaction.amount,
            "customer_std_amount": 50.0,
            "amount_zscore": 0.0,
            "time_since_last_txn_hours": 24.0,
            "txn_count_1h": 1.0,
            "txn_count_24h": 5.0,
            "amount_sum_1h": transaction.amount,
            "amount_sum_24h": transaction.amount * 5,
            "customer_unique_merchants": 10.0,
            "is_new_merchant": 0,
            "customer_txn_count": 50.0,
        }
    )

    df = pd.DataFrame([features])

    # Ensure all required features are present
    for feat in feature_names:
        if feat not in df.columns:
            df[feat] = 0

    # Return only the features the model was trained on
    return df[feature_names]


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict fraud probability for a single transaction
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create features
        X = create_features(transaction)

        # Make prediction
        fraud_proba = model.predict_proba(X)[0][1]
        is_fraud = fraud_proba >= 0.5  # Default threshold

        # Determine risk level
        if fraud_proba < 0.3:
            risk_level = "LOW"
        elif fraud_proba < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return PredictionResponse(
            transaction_id=transaction.transaction_id,
            is_fraud=bool(is_fraud),
            fraud_probability=float(fraud_proba),
            risk_level=risk_level,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(transactions: List[Transaction]):
    """
    Predict fraud probability for multiple transactions
    """

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 transactions")

    try:
        predictions = []

        for transaction in transactions:
            X = create_features(transaction)
            fraud_proba = model.predict_proba(X)[0][1]
            is_fraud = fraud_proba >= 0.5

            if fraud_proba < 0.3:
                risk_level = "LOW"
            elif fraud_proba < 0.7:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"

            predictions.append(
                {
                    "transaction_id": transaction.transaction_id,
                    "is_fraud": bool(is_fraud),
                    "fraud_probability": float(fraud_proba),
                    "risk_level": risk_level,
                }
            )

        return {"predictions": predictions, "total": len(predictions), "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def model_info():
    """Get model information"""
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    from src.utils.paths import get_latest_model
    
    try:
        latest_model = get_latest_model()
        model_size = latest_model.stat().st_size / (1024 * 1024)
    except:
        latest_model = None
        model_size = 0
    
    return {
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "features": feature_names[:10] if feature_names else [],  # Show first 10
        "model_path": str(latest_model) if latest_model else "Unknown",
        "model_size_mb": round(model_size, 2),
        "loaded_at": datetime.utcnow().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

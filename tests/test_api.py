"""
Test FastAPI endpoints
"""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import module (not just variables)
import src.api.main as api_module
from src.api.main import app

# Manually trigger startup
print("Triggering API startup...")
asyncio.run(api_module.load_model())
print(f"Model loaded: {api_module.model is not None}")

client = TestClient(app)


def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert data["model_loaded"] is True
    assert data["status"] == "healthy"


def test_model_info():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_type" in data
    assert "n_features" in data


def test_predict_valid_transaction():
    """Test prediction with valid transaction"""
    transaction = {
        "transaction_id": "TEST_001",
        "customer_id": "CUST_123",
        "amount": 125.50,
        "merchant_category": "restaurant",
        "merchant_id": "MERCH_001",
        "is_online": False,
        "transaction_hour": 14
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code == 200
    
    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "is_fraud" in data
    assert "risk_level" in data
    assert 0 <= data["fraud_probability"] <= 1


def test_predict_missing_fields():
    """Test prediction with missing required fields"""
    transaction = {
        "transaction_id": "TEST_002",
        "amount": 50.00
    }
    
    response = client.post("/predict", json=transaction)
    assert response.status_code == 422


def test_batch_predict():
    """Test batch prediction endpoint"""
    transactions = [
        {
            "transaction_id": f"TEST_{i}",
            "customer_id": f"CUST_{i}",
            "amount": 100.0 + i,
            "merchant_category": "grocery",
            "merchant_id": f"MERCH_{i}",
            "is_online": False,
            "transaction_hour": 10
        }
        for i in range(5)
    ]
    
    response = client.post("/batch_predict", json=transactions)
    assert response.status_code == 200
    
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 5
    
    for pred in data["predictions"]:
        assert "transaction_id" in pred
        assert "fraud_probability" in pred


def test_predict_edge_cases():
    """Test predictions with edge case values"""
    
    # Very high amount
    transaction_high = {
        "transaction_id": "TEST_HIGH",
        "customer_id": "CUST_999",
        "amount": 9999.99,
        "merchant_category": "online_shopping",
        "merchant_id": "MERCH_999",
        "is_online": True,
        "transaction_hour": 3
    }
    
    response = client.post("/predict", json=transaction_high)
    assert response.status_code == 200
    assert 0 <= response.json()["fraud_probability"] <= 1
    
    # Very low amount
    transaction_low = {
        "transaction_id": "TEST_LOW",
        "customer_id": "CUST_001",
        "amount": 0.01,
        "merchant_category": "grocery",
        "merchant_id": "MERCH_001",
        "is_online": False,
        "transaction_hour": 12
    }
    
    response = client.post("/predict", json=transaction_low)
    assert response.status_code == 200
    assert 0 <= response.json()["fraud_probability"] <= 1
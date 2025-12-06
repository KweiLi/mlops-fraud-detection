import requests
import time

def test_docker_deployment():
    """Test that Docker deployment works"""
    
    base_url = "http://localhost:8000"
    
    print("="*60)
    print("TESTING DOCKER DEPLOYMENT")
    print("="*60)
    
    # Test 1: Health check
    print("\n1. Testing health endpoint...")
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    health = response.json()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print("   ✓ Health check passed")
    
    # Test 2: Model info
    print("\n2. Testing model info endpoint...")
    response = requests.get(f"{base_url}/model/info")
    assert response.status_code == 200
    info = response.json()
    print(f"   Model type: {info['model_type']}")
    print(f"   Features: {info['n_features']}")
    print("   ✓ Model info passed")
    
    # Test 3: Single prediction
    print("\n3. Testing prediction endpoint...")
    transaction = {
        "transaction_id": "TXN_TEST_001",
        "customer_id": "CUST_12345",
        "amount": 1250.00,
        "merchant_category": "online_shopping",
        "merchant_id": "MERCH_0099",
        "is_online": True,
        "transaction_hour": 23
    }
    response = requests.post(f"{base_url}/predict", json=transaction)
    assert response.status_code == 200
    prediction = response.json()
    print(f"   Fraud probability: {prediction['fraud_probability']:.4f}")
    print(f"   Risk level: {prediction['risk_level']}")
    print("   ✓ Prediction passed")
    
    # Test 4: Batch prediction
    print("\n4. Testing batch prediction...")
    transactions = [transaction, transaction]  # Same transaction twice
    response = requests.post(f"{base_url}/batch_predict", json=transactions)
    assert response.status_code == 200
    batch = response.json()
    print(f"   Predictions: {batch['total']}")
    print("   ✓ Batch prediction passed")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)

if __name__ == "__main__":
    # Wait for API to be ready
    print("Waiting for API to start...")
    time.sleep(3)
    
    test_docker_deployment()
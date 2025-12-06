#!/bin/bash

echo "Testing Docker deployment..."

# Wait for API to be ready
echo "Waiting for API to start..."
sleep 5

# Test health endpoint
echo ""
echo "Testing health endpoint..."
curl -s http://localhost:8000/health | jq '.'

# Test prediction endpoint
echo ""
echo "Testing prediction endpoint..."
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_TEST_001",
    "customer_id": "CUST_12345",
    "amount": 1250.00,
    "merchant_category": "online_shopping",
    "merchant_id": "MERCH_0099",
    "is_online": true,
    "transaction_hour": 23
  }' | jq '.'

echo ""
echo "✓ Tests complete!"
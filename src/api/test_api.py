"""
Test script for API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_features():
    """Test features endpoint"""
    print("\nTesting /features endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/features")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Features endpoint: {data['total_features']} features")
            return True
        else:
            print(f"❌ Features endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Features endpoint error: {e}")
        return False

def test_predict():
    """Test prediction endpoint"""
    print("\nTesting /predict endpoint...")
    
    # Sample request data
    request_data = {
        "CustomerId": "test_customer_001",
        "CurrencyCode": "UGX",
        "CountryCode": 256,
        "ProductCategory": "airtime",
        "Amount": 1000.0,
        "Value": 1000.0,
        "PricingStrategy": 2,
        "FraudResult": 0,
        "recency_days": 15.0,
        "transaction_count": 10.0,
        "total_amount": 5000.0,
        "avg_transaction_amount": 500.0,
        "std_transaction_amount": 100.0,
        "log_total_amount": 8.5,
        "transaction_hour": 14,
        "transaction_day": 15,
        "transaction_month": 11,
        "transaction_year": 2024,
        "transaction_dayofweek": 2,
        "transaction_weekofyear": 46,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "dayofweek_sin": 0.434,
        "dayofweek_cos": 0.901,
        "is_business_hours": 1,
        "is_weekend": 0
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Risk probability: {data['risk_probability']}")
            print(f"   Risk category: {data['risk_category']}")
            print(f"   Prediction: {data['prediction']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False

def test_batch_predict():
    """Test batch prediction endpoint"""
    print("\nTesting /predict/batch endpoint...")
    
    request_data = {
        "customers": [
            {
                "CustomerId": "batch_customer_001",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProductCategory": "airtime",
                "Amount": 1000.0,
                "Value": 1000.0,
                "PricingStrategy": 2,
                "FraudResult": 0,
                "recency_days": 15.0,
                "transaction_count": 10.0,
                "total_amount": 5000.0,
                "avg_transaction_amount": 500.0,
                "std_transaction_amount": 100.0,
                "log_total_amount": 8.5,
                "transaction_hour": 14,
                "transaction_day": 15,
                "transaction_month": 11,
                "transaction_year": 2024,
                "transaction_dayofweek": 2,
                "transaction_weekofyear": 46,
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "dayofweek_sin": 0.434,
                "dayofweek_cos": 0.901,
                "is_business_hours": 1,
                "is_weekend": 0
            },
            {
                "CustomerId": "batch_customer_002",
                "CurrencyCode": "UGX",
                "CountryCode": 256,
                "ProductCategory": "financial_services",
                "Amount": 5000.0,
                "Value": 5000.0,
                "PricingStrategy": 2,
                "FraudResult": 0,
                "recency_days": 30.0,
                "transaction_count": 5.0,
                "total_amount": 25000.0,
                "avg_transaction_amount": 5000.0,
                "std_transaction_amount": 1000.0,
                "log_total_amount": 10.1,
                "transaction_hour": 9,
                "transaction_day": 20,
                "transaction_month": 12,
                "transaction_year": 2024,
                "transaction_dayofweek": 4,
                "transaction_weekofyear": 51,
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "dayofweek_sin": 0.975,
                "dayofweek_cos": -0.222,
                "is_business_hours": 1,
                "is_weekend": 0
            }
        ]
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=request_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Total customers: {data['total_customers']}")
            print(f"   Average risk: {data['avg_risk_probability']}")
            for i, pred in enumerate(data['predictions']):
                print(f"   Customer {i+1}: {pred['risk_probability']} ({pred['risk_category']})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return False

def main():
    """Run all API tests"""
    print("=" * 60)
    print("API TEST SUITE")
    print("=" * 60)
    
    # Wait for API to start
    print("Waiting for API to start...")
    time.sleep(5)
    
    tests = [
        test_health,
        test_features,
        test_predict,
        test_batch_predict
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} passed")
    print("=" * 60)
    
    if passed == total:
        print("✅ All API tests passed!")
        return 0
    else:
        print("❌ Some API tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
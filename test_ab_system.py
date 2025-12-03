#!/usr/bin/env python3
"""
Quick test script to verify A/B testing analysis works.
Generates a few test prediction requests to both variants.
"""

import requests
import time
import random
import os

# Configuration
API_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY_SECRET", "your-api-key")  # Load from environment

# Sample customer data for predictions (matching InputCustomer schema)
SAMPLE_CUSTOMERS = [
    {
        "tenure": 12.0,
        "InternetService_Fiber_optic": True,
        "Contract_Two_year": False,
        "PaymentMethod_Electronic_check": True,
        "No_internet_service": 0,
        "TotalCharges": 840.0,
        "MonthlyCharges": 70.0,
        "PaperlessBilling": 1
    },
    {
        "tenure": 24.0,
        "InternetService_Fiber_optic": False,
        "Contract_Two_year": True,
        "PaymentMethod_Electronic_check": False,
        "No_internet_service": 0,
        "TotalCharges": 1200.0,
        "MonthlyCharges": 50.0,
        "PaperlessBilling": 0
    },
    {
        "tenure": 6.0,
        "InternetService_Fiber_optic": False,
        "Contract_Two_year": False,
        "PaymentMethod_Electronic_check": False,
        "No_internet_service": 1,
        "TotalCharges": 540.0,
        "MonthlyCharges": 90.0,
        "PaperlessBilling": 1
    },
]

def make_prediction(customer_id: str):
    """Make a single prediction request."""
    customer = random.choice(SAMPLE_CUSTOMERS)
    
    headers = {
        "X-API-Key": API_KEY,
        "X-User-ID": customer_id,  # For sticky bucket assignment
        "Content-Type": "application/json"
    }
    
    # /survey/submit expects the customer object directly
    payload = customer
    
    try:
        response = requests.post(
            f"{API_URL}/survey/submit",
            headers=headers,
            json=payload,
            timeout=30  # Increased timeout for model loading
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f" Customer {customer_id}: {result}")
            return True
        else:
            print(f" Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f" Request failed: {e}")
        return False


def check_ab_results():
    """Check the A/B test analysis results."""
    headers = {"X-API-Key": API_KEY}
    
    try:
        response = requests.get(
            f"{API_URL}/ab/results",
            headers=headers,
            params={"metric": "latency"},
            timeout=10
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\n" + "="*60)
            print("A/B TEST ANALYSIS RESULTS")
            print("="*60)
            print(f"Variant A: {results.get('variant_a_count')} samples, "
                  f"avg latency: {results.get('variant_a_metric', 0):.4f}s")
            print(f"Variant B: {results.get('variant_b_count')} samples, "
                  f"avg latency: {results.get('variant_b_metric', 0):.4f}s")
            print(f"Lift: {results.get('lift_percent', 0):.2f}%")
            print(f"P-value: {results.get('p_value', 1):.4f}")
            print(f"Significant: {results.get('is_significant', False)}")
            print(f"Recommendation: {results.get('recommendation', 'N/A')}")
            print("="*60)
            return True
        else:
            print(f"\n Analysis failed ({response.status_code}): {response.text}")
            return False
            
    except Exception as e:
        print(f"\n Analysis request failed: {e}")
        return False


if __name__ == "__main__":
    print("Generating A/B test traffic...")
    print(f"Target: {API_URL}")
    print(f"Sending 10 prediction requests...\n")
    
    success_count = 0
    for i in range(10):
        customer_id = f"test_customer_{i:03d}"
        if make_prediction(customer_id):
            success_count += 1
        time.sleep(0.5)  # Small delay between requests
    
    print(f"\n Completed: {success_count}/10 requests successful")
    
    if success_count > 0:
        print("\n Checking A/B test analysis...")
        time.sleep(1)
        check_ab_results()
    else:
        print("\n No successful requests - skipping analysis")

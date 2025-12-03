"""Load testing configuration for Customer Churn API.

This module provides configuration settings for different load testing scenarios.
"""

import os
from dataclasses import dataclass
from typing import Dict

@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    
    # API Configuration
    api_url: str = os.getenv("API_URL", "http://localhost:8000")
    api_key: str = os.getenv("API_KEY_SECRET", "your-api-key")
    
    # Load Testing Parameters
    users: int = 100  # Number of concurrent users
    spawn_rate: int = 10  # Users spawned per second
    run_time: str = "5m"  # Test duration
    
    # Rate Limiting
    expected_rate_limit: int = 100  # Requests per hour
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers with API key."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }


# Predefined test scenarios
SCENARIOS = {
    "smoke": LoadTestConfig(
        users=5,
        spawn_rate=1,
        run_time="1m"
    ),
    "normal": LoadTestConfig(
        users=50,
        spawn_rate=5,
        run_time="5m"
    ),
    "stress": LoadTestConfig(
        users=200,
        spawn_rate=20,
        run_time="10m"
    ),
    "spike": LoadTestConfig(
        users=500,
        spawn_rate=100,
        run_time="2m"
    )
}


# Sample customer data for predictions
SAMPLE_CUSTOMERS = [
    {
        "tenure": 12.0,
        "InternetService_Fiber_optic": True,
        "Contract_Two_year": False,
        "PaymentMethod_Electronic_check": True,
        "No_internet_service": 0,
        "TotalCharges": 1200.50,
        "MonthlyCharges": 85.25,
        "PaperlessBilling": 1
    },
    {
        "tenure": 36.0,
        "InternetService_Fiber_optic": False,
        "Contract_Two_year": True,
        "PaymentMethod_Electronic_check": False,
        "No_internet_service": 0,
        "TotalCharges": 3600.00,
        "MonthlyCharges": 75.00,
        "PaperlessBilling": 0
    },
    {
        "tenure": 6.0,
        "InternetService_Fiber_optic": True,
        "Contract_Two_year": False,
        "PaymentMethod_Electronic_check": True,
        "No_internet_service": 0,
        "TotalCharges": 600.00,
        "MonthlyCharges": 95.50,
        "PaperlessBilling": 1
    }
]

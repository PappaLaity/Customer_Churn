"""Locust load testing suite for Customer Churn API.

This script tests API performance, rate limiting, and A/B testing distribution.

Usage:
    # Basic test
    locust -f tests/load/locustfile.py --host=http://localhost:8000
    
    # Headless mode with 50 users
    locust -f tests/load/locustfile.py --host=http://localhost:8000 \\
           --users 50 --spawn-rate 5 --run-time 5m --headless
    
    # Specific scenario
    SCENARIO=stress locust -f tests/load/locustfile.py --host=http://localhost:8000
"""

import random
import time
from locust import HttpUser, task, between, events
from locust.env import Environment

from load_test_config import LoadTestConfig, SAMPLE_CUSTOMERS, SCENARIOS
import os


# Load configuration
scenario = os.getenv("SCENARIO", "normal")
config = SCENARIOS.get(scenario, LoadTestConfig())


class ChurnAPIUser(HttpUser):
    """Simulates a user interacting with the Customer Churn API."""
    
    # Wait time between tasks (1-3 seconds)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Initialize user session."""
        self.api_key = config.api_key
        self.headers = config.headers
        self.prediction_count = 0
        self.rate_limit_hits = 0
    
    @task(10)
    def submit_survey(self):
        """Submit a survey response (most frequent action)."""
        # Generate random customer data
        customer_data = self.generate_customer_data()
        
        with self.client.post(
            "/survey/submit",
            json=customer_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.prediction_count += 1
                response.success()
            elif response.status_code == 429:
                # Rate limit hit (expected behavior)
                self.rate_limit_hits += 1
                response.success()  # Don't count as failure
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(5)
    def get_model_version(self):
        """Test model version endpoint."""
        with self.client.get(
            "/model/version",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                self.rate_limit_hits += 1
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(3)
    def get_ab_config(self):
        """Test A/B testing configuration endpoint."""
        with self.client.get(
            "/ab/config",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                self.rate_limit_hits += 1
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(2)
    def health_check(self):
        """Test health check endpoint."""
        with self.client.get(
            "/health",
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                self.rate_limit_hits += 1
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")
    
    @task(1)
    def get_models(self):
        """Test models listing endpoint."""
        with self.client.get(
            "/models",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                self.rate_limit_hits += 1
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


class RateLimitTestUser(HttpUser):
    """Specifically tests rate limiting behavior."""
    
    wait_time = between(0.1, 0.2)  # Very aggressive
    
    def on_start(self):
        """Initialize user session."""
        self.headers = config.headers
        self.rate_limit_count = 0
        self.success_count = 0
    
    @task
    def spam_predictions(self):
        """Rapidly submit predictions to trigger rate limiting."""
        customer_data = random.choice(SAMPLE_CUSTOMERS)
        
        with self.client.post(
            "/survey/submit",
            json=customer_data,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                self.success_count += 1
                response.success()
            elif response.status_code == 429:
                self.rate_limit_count += 1
                # This is expected, mark as success
                response.success()
            else:
                response.failure(f"Unexpected status: {response.status_code}")


# Event listeners for statistics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print(f"\n{'='*70}")
    print(f"🚀 Starting Load Test: {scenario.upper()} scenario")
    print(f"{'='*70}")
    print(f"API URL: {config.api_url}")
    print(f"Users: {config.users}")
    print(f"Spawn Rate: {config.spawn_rate}/sec")
    print(f"Duration: {config.run_time}")
    print(f"{'='*70}\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print(f"\n{'='*70}")
    print(f"✅ Load Test Complete")
    print(f"{'='*70}")
    
    # Calculate statistics
    total_requests = sum(env.stats.total.num_requests for env in [environment])
    total_failures = sum(env.stats.total.num_failures for env in [environment])
    
    if total_requests > 0:
        failure_rate = (total_failures / total_requests) * 100
        print(f"Total Requests: {total_requests}")
        print(f"Total Failures: {total_failures} ({failure_rate:.2f}%)")
    
    print(f"{'='*70}\n")


# Custom user classes for different scenarios
user_classes = {
    "normal": [ChurnAPIUser],
    "smoke": [ChurnAPIUser],
    "stress": [ChurnAPIUser, RateLimitTestUser],
    "spike": [RateLimitTestUser]
}


# Set the user class based on scenario
if scenario in user_classes:
    # Dynamically set user classes
    pass

"""Core utilities and infrastructure for the Customer Churn API.

This package contains:
- metrics: Prometheus metrics and drift computation
- state: Centralized application state management
- lifespan: Application lifecycle management
- database: Database configuration and initialization
- security: Authentication and authorization
- seed: Database seeding utilities
"""

__all__ = [
    "metrics",
    "state", 
    "lifespan",
    "database",
    "security",
    "seed",
]

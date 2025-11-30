#!/usr/bin/env python3
"""
Generate sample production data for testing drift detection.
This creates synthetic churn data with intentional drift.
"""
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Generate 200 rows with drift
n_samples = 200

# Create production data with DRIFT
# Age: Shift distribution higher (customers getting older)
age = np.random.randint(40, 90, n_samples)  # Drifted from baseline (18-80)

# Tenure: Shorter tenure (more new customers)
tenure = np.random.randint(1, 36, n_samples)  # Drifted from baseline (1-72)

# MonthlyCharges: Higher prices
monthly_charges = np.random.uniform(60, 150, n_samples)  # Drifted from baseline (20-120)

# TotalCharges: Correlated with tenure and monthly charges
total_charges = tenure * monthly_charges + np.random.normal(0, 500, n_samples)

# Churn: Higher churn rate due to drift
churn = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # 40% churn vs baseline ~20%

# Create DataFrame
production_df = pd.DataFrame({
    'Age': age,
    'Tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Churn': churn
})

# Save to production folder
output_path = 'data/production/production.csv'
production_df.to_csv(output_path, index=False)

print(f" Created production data with {n_samples} rows")
print(f"Saved to: {output_path}")
print("\nData Summary:")
print(production_df.describe())
print(f"\nChurn Rate: {production_df['Churn'].mean():.2%}")

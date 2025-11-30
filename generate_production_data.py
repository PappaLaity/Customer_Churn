import pandas as pd
import numpy as np
import os

def generate_production_data(n_samples=200):
    """
    Generate synthetic production data with intentional drift for testing.
    Matches the full schema of the Telco Customer Churn dataset.
    """
    np.random.seed(42)
    
    # --- Drifted Features ---
    # 1. MonthlyCharges: Higher than baseline (60-150 vs 20-120)
    monthly_charges = np.random.uniform(60, 150, n_samples)
    
    # 2. Tenure: Shorter than baseline (1-36 vs 1-72)
    tenure = np.random.randint(1, 36, n_samples)
    
    # 3. TotalCharges: Correlated with MonthlyCharges * Tenure
    total_charges = monthly_charges * tenure + np.random.normal(0, 10, n_samples)
    
    # 4. Churn: Higher rate (40% vs ~20%)
    churn_prob = 0.4
    churn = np.random.choice(['Yes', 'No'], size=n_samples, p=[churn_prob, 1-churn_prob])
    
    # --- Other Features (Random or Correlated) ---
    
    # Demographics
    gender = np.random.choice(['Male', 'Female'], size=n_samples)
    senior_citizen = np.random.choice([0, 1], size=n_samples, p=[0.2, 0.8]) # More seniors (drift)
    partner = np.random.choice(['Yes', 'No'], size=n_samples)
    dependents = np.random.choice(['Yes', 'No'], size=n_samples)
    
    # Services
    phone_service = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['Yes', 'No', 'No phone service'], size=n_samples)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], size=n_samples, p=[0.1, 0.8, 0.1]) # More Fiber (drift)
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    online_backup = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    device_protection = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    streaming_tv = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    streaming_movies = np.random.choice(['Yes', 'No', 'No internet service'], size=n_samples)
    
    # Contract & Billing
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=n_samples, p=[0.8, 0.1, 0.1]) # More monthly (drift)
    paperless_billing = np.random.choice(['Yes', 'No'], size=n_samples)
    payment_method = np.random.choice([
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ], size=n_samples)

    # Create DataFrame
    data = {
        'customerID': [f'PROD-{i:04d}' for i in range(n_samples)],
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn
    }
    
    df = pd.DataFrame(data)
    
    # Ensure directory exists
    os.makedirs('data/production', exist_ok=True)
    
    # Save
    output_path = 'data/production/production.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Created realistic production data with {n_samples} rows and {len(df.columns)} columns")
    print(f"📊 Saved to: {output_path}")
    print("\nData Summary:")
    print(df[['MonthlyCharges', 'tenure', 'Churn']].describe())
    print(f"\nChurn Rate: {df['Churn'].value_counts(normalize=True).get('Yes', 0):.2%}")

if __name__ == "__main__":
    generate_production_data()

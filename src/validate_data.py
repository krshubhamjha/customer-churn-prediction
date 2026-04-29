# ================================================================
# STEP 1 — DATA VALIDATION
# File: src/validate_data.py
# ================================================================

import pandas as pd
import sys
import os

# ----------------------------------------------------------------
# Required columns
# ----------------------------------------------------------------
REQUIRED_COLUMNS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges',
    'Churn'
]

# ----------------------------------------------------------------
# Validation function
# ----------------------------------------------------------------
def validate_data(filepath, is_new_data=False):

    print("=" * 55)
    if is_new_data:
        print("  VALIDATING NEW DATA")
    else:
        print("  VALIDATING ORIGINAL DATA")
    print("=" * 55)

    # Check 1 — File exists
    print("\n--- Check 1 : File exists ---")
    if not os.path.exists(filepath):
        print(f"  File not found : {filepath}")
        return False
    print(f"  File found : {filepath}")

    # Check 2 — File readable
    print("\n--- Check 2 : File is readable ---")
    try:
        df = pd.read_csv(filepath)
        print(f"  File loaded successfully")
        print(f"  Rows    : {df.shape[0]}")
        print(f"  Columns : {df.shape[1]}")
    except Exception as e:
        print(f"  Cannot read file : {e}")
        return False

    # Check 3 — Minimum rows
    print("\n--- Check 3 : Minimum rows ---")
    min_rows = 10 if is_new_data else 100
    if len(df) < min_rows:
        print(f"  Too few rows : {len(df)}")
        return False
    print(f"  Row count valid : {len(df)}")

    # Check 4 — Required columns
    print("\n--- Check 4 : Required columns ---")
    missing_cols = [col for col in REQUIRED_COLUMNS
                    if col not in df.columns]
    if missing_cols:
        print(f"  Missing columns : {missing_cols}")
        return False
    print(f"  All required columns present")

    # Check 5 — Churn values
    print("\n--- Check 5 : Churn column values ---")
    unique_values = df['Churn'].unique().tolist()
    print(f"  Unique values : {unique_values}")
    if not all(v in ['Yes', 'No'] for v in unique_values):
        print(f"  Invalid Churn values!")
        return False
    print(f"  Churn column valid")

    # Check 6 — Empty columns
    print("\n--- Check 6 : Empty columns ---")
    empty_cols = [col for col in df.columns
                  if df[col].isnull().all()]
    if empty_cols:
        print(f"  Empty columns : {empty_cols}")
        return False
    print(f"  No empty columns found")

    # Check 7 — TotalCharges numeric
    print("\n--- Check 7 : TotalCharges numeric ---")
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    missing_tc = df['TotalCharges'].isnull().sum()
    print(f"  Missing after conversion : {missing_tc}")
    print(f"  TotalCharges check passed")

    # Check 8 — Churn rate
    print("\n--- Check 8 : Churn rate ---")
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"  Churn rate : {churn_rate:.2f}%")
    if churn_rate < 5 or churn_rate > 70:
        print(f"  WARNING : Unusual churn rate")
    else:
        print(f"  Churn rate looks normal")

    print("\n" + "=" * 55)
    print("  ALL CHECKS PASSED!")
    print("  Data is valid and ready for pipeline")
    print("=" * 55)
    return True


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":

    # Validate original data
    print("\nValidating original data...")
    result1 = validate_data(
        "data/Telco-Customer-Churn.csv",
        is_new_data=False
    )

    # Validate new data if exists
    if os.path.exists("data/new_data.csv"):
        print("\nNew data found! Validating...")
        result2 = validate_data(
            "data/new_data.csv",
            is_new_data=True
        )
        if not result2:
            print("\nNew data validation FAILED")
            sys.exit(1)
    else:
        print("\nNo new data found")
        print("Using original data only")

    if not result1:
        print("\nOriginal data validation FAILED")
        sys.exit(1)

    print("\n" + "=" * 55)
    print("  ALL VALIDATION COMPLETE!")
    print("  Ready for next pipeline step")
    print("=" * 55)
    sys.exit(0)
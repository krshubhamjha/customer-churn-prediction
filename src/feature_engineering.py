# ================================================================
# STEP 3 — FEATURE ENGINEERING
# File: src/feature_engineering.py
# ================================================================

import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def engineer_features():

    print("=" * 55)
    print("  STEP 3 — FEATURE ENGINEERING")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 — Load data
    # ----------------------------------------------------------
    print("\n--- Step 1 : Load data ---")
    if os.path.exists("data/combined_data.csv"):
        filepath = "data/combined_data.csv"
        print(f"  Using combined data")
    else:
        filepath = "data/Telco-Customer-Churn.csv"
        print(f"  Using original data")

    df = pd.read_csv(filepath)
    print(f"  Rows loaded : {len(df)}")

    # ----------------------------------------------------------
    # Step 2 — Basic cleaning
    # ----------------------------------------------------------
    print("\n--- Step 2 : Basic cleaning ---")
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(
        df['TotalCharges'].median(), inplace=True)
    print(f"  TotalCharges fixed")

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
        print(f"  customerID dropped")

    before = len(df)
    df     = df.drop_duplicates()
    print(f"  Duplicates removed : {before - len(df)}")
    print(f"  Rows after cleaning : {len(df)}")

    # ----------------------------------------------------------
    # Step 3 — Create 6 engineered features
    # ----------------------------------------------------------
    print("\n--- Step 3 : Feature engineering ---")

    df['charge_per_tenure'] = (
        df['MonthlyCharges'] / (df['tenure'] + 1))
    print(f"  Feature 1 → charge_per_tenure created")

    service_cols = [
        'PhoneService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies']
    df['total_services'] = df[service_cols].apply(
        lambda x: (x == 'Yes').sum(), axis=1)
    print(f"  Feature 2 → total_services created")

    df['has_protection'] = (
        (df['OnlineSecurity'] == 'Yes').astype(int) +
        (df['DeviceProtection'] == 'Yes').astype(int))
    print(f"  Feature 3 → has_protection created")

    df['is_new_customer'] = (
        df['tenure'] < 12).astype(int)
    print(f"  Feature 4 → is_new_customer created")

    df['avg_monthly_value'] = (
        df['TotalCharges'] / (df['tenure'] + 1))
    print(f"  Feature 5 → avg_monthly_value created")

    df['high_risk_flag'] = (
        (df['MonthlyCharges'] > 65) &
        (df['tenure'] < 12) &
        (df['Contract'] == 'Month-to-month')
    ).astype(int)
    print(f"  Feature 6 → high_risk_flag created")
    print(f"\n  Total columns : {df.shape[1]}")
    # Fix NaN values after encoding
    print("\n--- Step 5b : Fix NaN values ---")
    print(f"  NaN before fix : {df.isnull().sum().sum()}")
    df = df.fillna(0)
    print(f"  NaN after fix  : {df.isnull().sum().sum()}")

    # ----------------------------------------------------------
    # Step 4 — Encode binary columns
    # ----------------------------------------------------------
    print("\n--- Step 4 : Encode binary columns ---")
    binary_cols = [
        'gender', 'Partner', 'Dependents',
        'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({
            'Yes': 1, 'No': 0,
            'Male': 1, 'Female': 0})
    print(f"  Binary columns encoded : {len(binary_cols)}")

    # ----------------------------------------------------------
    # Step 5 — One hot encode multi columns
    # ----------------------------------------------------------
    print("\n--- Step 5 : One hot encoding ---")
    multi_cols = [
        'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod']
    df = pd.get_dummies(df,
                        columns=multi_cols,
                        drop_first=True)
    bool_cols    = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"  One hot encoding done")
    print(f"  Total columns : {df.shape[1]}")

    # ----------------------------------------------------------
    # Step 6 — Split X and y
    # ----------------------------------------------------------
    print("\n--- Step 6 : Split features and target ---")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    print(f"  Features shape : {X.shape}")
    print(f"  Target shape   : {y.shape}")
    print(f"  Churn rate     : {y.mean()*100:.2f}%")

    # ----------------------------------------------------------
    # Step 7 — Scale numerical columns
    # ----------------------------------------------------------
    print("\n--- Step 7 : Scale numerical columns ---")
    scale_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'charge_per_tenure', 'avg_monthly_value']
    scale_cols = [c for c in scale_cols if c in X.columns]
    scaler     = StandardScaler()
    X[scale_cols] = scaler.fit_transform(X[scale_cols])
    print(f"  Scaled columns : {scale_cols}")

    # ----------------------------------------------------------
    # Step 8 — Train test split
    # ----------------------------------------------------------
    print("\n--- Step 8 : Train test split ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y)
    print(f"  X_train : {X_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    print(f"  Train churn rate : {y_train.mean()*100:.2f}%")
    print(f"  Test churn rate  : {y_test.mean()*100:.2f}%")

    # ----------------------------------------------------------
    # Step 9 — Save all processed files
    # ----------------------------------------------------------
    print("\n--- Step 9 : Save processed data ---")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    joblib.dump(X_train, "data/processed/X_train.pkl")
    joblib.dump(X_test,  "data/processed/X_test.pkl")
    joblib.dump(y_train, "data/processed/y_train.pkl")
    joblib.dump(y_test,  "data/processed/y_test.pkl")
    joblib.dump(scaler,  "models/scaler.pkl")
    joblib.dump(X.columns.tolist(), "models/feature_names.pkl")

    print(f"  X_train.pkl       saved ✅")
    print(f"  X_test.pkl        saved ✅")
    print(f"  y_train.pkl       saved ✅")
    print(f"  y_test.pkl        saved ✅")
    print(f"  scaler.pkl        saved ✅")
    print(f"  feature_names.pkl saved ✅")

    print("\n" + "=" * 55)
    print("  FEATURE ENGINEERING COMPLETE!")
    print("=" * 55)
    print(f"  Total features   : {X.shape[1]}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing samples  : {len(X_test)}")
    print("=" * 55)

    return X_train, X_test, y_train, y_test, scaler


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    result = engineer_features()
    if result is None:
        print("\nFeature engineering FAILED")
        sys.exit(1)
    else:
        print("\nFeature engineering PASSED")
        print("Ready for model training")
        sys.exit(0)
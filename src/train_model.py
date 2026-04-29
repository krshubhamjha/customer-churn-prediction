# ================================================================
# STEP 4 — MODEL TRAINING
# File: src/train_model.py
# ================================================================
# Purpose:
# Load processed data from feature engineering
# Apply SMOTE for class imbalance
# Train Logistic Regression model
# Save trained model
# ================================================================

import pandas as pd
import numpy as np
import os
import sys
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score,
                             roc_auc_score)
from imblearn.over_sampling import SMOTE

def train_model():

    print("=" * 55)
    print("  STEP 4 — MODEL TRAINING")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 — Load processed data
    # ----------------------------------------------------------
    print("\n--- Step 1 : Load processed data ---")

    try:
        X_train = joblib.load("data/processed/X_train.pkl")
        X_test  = joblib.load("data/processed/X_test.pkl")
        y_train = joblib.load("data/processed/y_train.pkl")
        y_test  = joblib.load("data/processed/y_test.pkl")
        print(f"  X_train : {X_train.shape}")
        print(f"  X_test  : {X_test.shape}")
        print(f"  Train churn rate : {y_train.mean()*100:.2f}%")
        print(f"  Test churn rate  : {y_test.mean()*100:.2f}%")
    except Exception as e:
        print(f"  Error loading data : {e}")
        print(f"  Run feature_engineering.py first!")
        return None

    # ----------------------------------------------------------
    # Step 2 — Apply SMOTE
    # ----------------------------------------------------------
    print("\n--- Step 2 : Apply SMOTE ---")

    print(f"  Before SMOTE:")
    print(f"  Not Churned : {(y_train==0).sum()}")
    print(f"  Churned     : {(y_train==1).sum()}")

    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(
        X_train, y_train)

    print(f"\n  After SMOTE:")
    print(f"  Not Churned : {(y_train_sm==0).sum()}")
    print(f"  Churned     : {(y_train_sm==1).sum()}")
    print(f"  Total       : {len(y_train_sm)}")

    # ----------------------------------------------------------
    # Step 3 — Train model
    # ----------------------------------------------------------
    print("\n--- Step 3 : Train Logistic Regression ---")

    model = LogisticRegression(
        C        = 1,
        penalty  = 'l2',
        solver   = 'saga',
        max_iter = 1000,
        random_state = 42
    )

    model.fit(X_train_sm, y_train_sm)
    print(f"  Model trained successfully!")

    # ----------------------------------------------------------
    # Step 4 — Evaluate on test data
    # ----------------------------------------------------------
    print("\n--- Step 4 : Evaluate model ---")

    y_pred = model.predict(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred)

    print(f"  Accuracy  : {accuracy*100:.2f}%")
    print(f"  Precision : {precision*100:.2f}%")
    print(f"  Recall    : {recall*100:.2f}%")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC AUC   : {roc_auc:.4f}")

    # ----------------------------------------------------------
    # Step 5 — Save model
    # ----------------------------------------------------------
    print("\n--- Step 5 : Save model ---")

    os.makedirs("models", exist_ok=True)

    # Save with version number
    version    = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"model_{version}_f1_{f1:.4f}.pkl"

    # Save versioned model
    joblib.dump(model, f"models/{model_name}")
    print(f"  Versioned model saved : {model_name}")

    # Save as new model for evaluation
    joblib.dump(model, "models/new_model.pkl")
    print(f"  new_model.pkl saved")

    # ----------------------------------------------------------
    # Step 6 — Save training results
    # ----------------------------------------------------------
    print("\n--- Step 6 : Save training results ---")

    os.makedirs("metadata", exist_ok=True)

    results = {
        "training_date" : datetime.now().isoformat(),
        "model_version" : version,
        "model_name"    : model_name,
        "parameters"    : {
            "C"        : 1,
            "penalty"  : "l2",
            "solver"   : "saga",
            "max_iter" : 1000
        },
        "smote_applied" : True,
        "train_samples" : len(X_train_sm),
        "test_samples"  : len(X_test),
        "metrics"       : {
            "accuracy"  : round(accuracy,  4),
            "precision" : round(precision, 4),
            "recall"    : round(recall,    4),
            "f1_score"  : round(f1,        4),
            "roc_auc"   : round(roc_auc,   4)
        }
    }

    with open("metadata/training_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"  Training results saved!")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  MODEL TRAINING COMPLETE!")
    print("=" * 55)
    print(f"  Model    : Logistic Regression")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  Recall   : {recall*100:.2f}%")
    print(f"  Saved as : {model_name}")
    print("=" * 55)

    return model, f1


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    result = train_model()
    if result is None:
        print("\nModel training FAILED")
        sys.exit(1)
    else:
        print("\nModel training PASSED")
        print("Ready for evaluation")
        sys.exit(0)
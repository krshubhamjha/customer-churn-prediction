# ================================================================
# STEP 5 — MODEL EVALUATION
# File: src/evaluate_model.py
# ================================================================
# Purpose:
# Load new trained model
# Load old best model
# Compare F1 scores
# Keep better model
# Save final best model
# ================================================================

import os
import sys
import joblib
import json
from datetime import datetime
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_score, recall_score,
                             roc_auc_score)

def evaluate_model():

    print("=" * 55)
    print("  STEP 5 — MODEL EVALUATION")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 — Load test data
    # ----------------------------------------------------------
    print("\n--- Step 1 : Load test data ---")
    try:
        X_test = joblib.load("data/processed/X_test.pkl")
        y_test = joblib.load("data/processed/y_test.pkl")
        print(f"  X_test shape : {X_test.shape}")
        print(f"  Test churn   : {y_test.mean()*100:.2f}%")
    except Exception as e:
        print(f"  Error : {e}")
        return None

    # ----------------------------------------------------------
    # Step 2 — Load new model
    # ----------------------------------------------------------
    print("\n--- Step 2 : Load new model ---")
    try:
        new_model = joblib.load("models/new_model.pkl")
        print(f"  New model loaded successfully")
    except Exception as e:
        print(f"  Error loading new model : {e}")
        return None

    # ----------------------------------------------------------
    # Step 3 — Evaluate new model
    # ----------------------------------------------------------
    print("\n--- Step 3 : Evaluate new model ---")
    y_pred_new = new_model.predict(X_test)

    new_f1        = f1_score(y_test, y_pred_new)
    new_accuracy  = accuracy_score(y_test, y_pred_new)
    new_precision = precision_score(y_test, y_pred_new)
    new_recall    = recall_score(y_test, y_pred_new)
    new_auc       = roc_auc_score(y_test, y_pred_new)

    print(f"  New Model Results:")
    print(f"  Accuracy  : {new_accuracy*100:.2f}%")
    print(f"  Precision : {new_precision*100:.2f}%")
    print(f"  Recall    : {new_recall*100:.2f}%")
    print(f"  F1 Score  : {new_f1:.4f}")
    print(f"  ROC AUC   : {new_auc:.4f}")

    # ----------------------------------------------------------
    # Step 4 — Load old model and compare
    # ----------------------------------------------------------
    print("\n--- Step 4 : Compare with old model ---")

    old_model_path = "models/best_model.pkl"

    if os.path.exists(old_model_path):
        old_model  = joblib.load(old_model_path)
        y_pred_old = old_model.predict(X_test)
        old_f1     = f1_score(y_test, y_pred_old)
        print(f"  Old model F1 : {old_f1:.4f}")
        print(f"  New model F1 : {new_f1:.4f}")
    else:
        print(f"  No old model found")
        print(f"  New model will become best model")
        old_f1 = 0.0

    # ----------------------------------------------------------
    # Step 5 — Keep better model
    # ----------------------------------------------------------
    print("\n--- Step 5 : Keep better model ---")

    if new_f1 > old_f1:
        # New model is better → save as best
        joblib.dump(new_model, "models/best_model.pkl")
        joblib.dump(new_model, "app/best_model.pkl")
        print(f"  New model is BETTER!")
        print(f"  Old F1 : {old_f1:.4f}")
        print(f"  New F1 : {new_f1:.4f}")
        print(f"  Improvement : +{(new_f1-old_f1):.4f}")
        print(f"  best_model.pkl updated ✅")
        model_updated = True
    else:
        # Old model is better → keep old
        print(f"  Old model is BETTER or EQUAL!")
        print(f"  Old F1 : {old_f1:.4f}")
        print(f"  New F1 : {new_f1:.4f}")
        print(f"  Keeping old model ✅")
        model_updated = False

    # ----------------------------------------------------------
    # Step 6 — Save evaluation results
    # ----------------------------------------------------------
    print("\n--- Step 6 : Save evaluation results ---")

    os.makedirs("metadata", exist_ok=True)

    evaluation = {
        "evaluation_date" : datetime.now().isoformat(),
        "model_updated"   : model_updated,
        "new_model"       : {
            "f1_score"  : round(new_f1,        4),
            "accuracy"  : round(new_accuracy,  4),
            "precision" : round(new_precision, 4),
            "recall"    : round(new_recall,    4),
            "roc_auc"   : round(new_auc,       4)
        },
        "old_f1_score"    : round(old_f1, 4),
        "winner"          : "new" if model_updated else "old"
    }

    with open("metadata/evaluation_results.json", "w") as f:
        json.dump(evaluation, f, indent=4)
    print(f"  Evaluation results saved!")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  MODEL EVALUATION COMPLETE!")
    print("=" * 55)
    print(f"  New F1    : {new_f1:.4f}")
    print(f"  Old F1    : {old_f1:.4f}")
    print(f"  Winner    : {'New Model ✅' if model_updated else 'Old Model ✅'}")
    print(f"  Updated   : {model_updated}")
    print("=" * 55)

    return model_updated, new_f1


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    result = evaluate_model()
    if result is None:
        print("\nEvaluation FAILED")
        sys.exit(1)
    else:
        model_updated, f1 = result
        print(f"\nEvaluation PASSED")
        print(f"F1 Score : {f1:.4f}")
        print(f"Model updated : {model_updated}")
        sys.exit(0)
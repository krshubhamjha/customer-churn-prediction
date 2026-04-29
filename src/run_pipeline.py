# ================================================================
# MASTER PIPELINE
# File: src/run_pipeline.py
# ================================================================
# Purpose:
# Run all 5 steps in correct order
# One command runs entire pipeline
# ================================================================

import sys
import os
from datetime import datetime

# Import all steps
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from validate_data       import validate_data
from load_data           import load_data
from feature_engineering import engineer_features
from train_model         import train_model
from evaluate_model      import evaluate_model

def run_pipeline():

    print("\n" + "=" * 55)
    print("  CUSTOMER CHURN — ML PIPELINE STARTING")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 — Validate data
    # ----------------------------------------------------------
    print("\n🔍 STEP 1 — VALIDATING DATA...")
    result1 = validate_data(
        "data/Telco-Customer-Churn.csv",
        is_new_data=False
    )
    if not result1:
        print("❌ PIPELINE FAILED at Step 1 — Data invalid")
        sys.exit(1)

    # Validate new data if exists
    if os.path.exists("data/new_data.csv"):
        result_new = validate_data(
            "data/new_data.csv",
            is_new_data=True
        )
        if not result_new:
            print("❌ New data invalid — using original only")

    print("✅ Step 1 Complete!\n")

    # ----------------------------------------------------------
    # Step 2 — Load data
    # ----------------------------------------------------------
    print("📂 STEP 2 — LOADING DATA...")
    data = load_data()
    if data is None:
        print("❌ PIPELINE FAILED at Step 2 — Data loading failed")
        sys.exit(1)
    print("✅ Step 2 Complete!\n")

    # ----------------------------------------------------------
    # Step 3 — Feature engineering
    # ----------------------------------------------------------
    print("⚙️  STEP 3 — FEATURE ENGINEERING...")
    result3 = engineer_features()
    if result3 is None:
        print("❌ PIPELINE FAILED at Step 3 — Feature engineering failed")
        sys.exit(1)
    print("✅ Step 3 Complete!\n")

    # ----------------------------------------------------------
    # Step 4 — Train model
    # ----------------------------------------------------------
    print("🤖 STEP 4 — TRAINING MODEL...")
    result4 = train_model()
    if result4 is None:
        print("❌ PIPELINE FAILED at Step 4 — Training failed")
        sys.exit(1)
    print("✅ Step 4 Complete!\n")

    # ----------------------------------------------------------
    # Step 5 — Evaluate model
    # ----------------------------------------------------------
    print("📊 STEP 5 — EVALUATING MODEL...")
    result5 = evaluate_model()
    if result5 is None:
        print("❌ PIPELINE FAILED at Step 5 — Evaluation failed")
        sys.exit(1)

    model_updated, f1 = result5
    print("✅ Step 5 Complete!\n")

    # ----------------------------------------------------------
    # Pipeline complete
    # ----------------------------------------------------------
    print("=" * 55)
    print("  ✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 55)
    print(f"  Final F1 Score  : {f1:.4f}")
    print(f"  Model Updated   : {model_updated}")
    print(f"  Finished at     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)

    if model_updated:
        print("\n🚀 New model deployed!")
        print("   Streamlit app will use updated model")
    else:
        print("\n✅ Old model kept — still best performer")

    return True


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    result = run_pipeline()
    if result:
        sys.exit(0)
    else:
        sys.exit(1)
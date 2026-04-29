# ================================================================
# STEP 2 — DATA LOADING
# File: src/load_data.py
# ================================================================
# Purpose:
# Load original data
# Check if new data exists
# Combine if new data found
# Save combined data for pipeline
# ================================================================

import pandas as pd
import os
import sys
import json
from datetime import datetime

# ----------------------------------------------------------------
# Main load function
# ----------------------------------------------------------------
def load_data():

    print("=" * 55)
    print("  STEP 2 — DATA LOADING")
    print("=" * 55)

    # ----------------------------------------------------------
    # Step 1 — Load original data
    # ----------------------------------------------------------
    print("\n--- Step 1 : Load original data ---")
    original_path = "data/Telco-Customer-Churn.csv"

    if not os.path.exists(original_path):
        print(f"  Original data not found!")
        return None

    original = pd.read_csv(original_path)
    print(f"  Original data loaded")
    print(f"  Rows : {len(original)}")

    # ----------------------------------------------------------
    # Step 2 — Check if new data exists
    # ----------------------------------------------------------
    print("\n--- Step 2 : Check for new data ---")
    new_data_path = "data/new_data.csv"

    if os.path.exists(new_data_path):
        print(f"  New data found!")

        # Load new data
        new_data = pd.read_csv(new_data_path)
        print(f"  New data rows : {len(new_data)}")

        # --------------------------------------------------
        # Step 3 — Combine old and new data
        # --------------------------------------------------
        print("\n--- Step 3 : Combining data ---")
        combined = pd.concat([original, new_data],
                              ignore_index=True)
        print(f"  After combining   : {len(combined)} rows")

        # --------------------------------------------------
        # Step 4 — Remove duplicates
        # --------------------------------------------------
        print("\n--- Step 4 : Remove duplicates ---")
        before = len(combined)
        combined = combined.drop_duplicates()
        after  = len(combined)
        print(f"  Duplicates removed : {before - after}")
        print(f"  Final rows         : {after}")

        # --------------------------------------------------
        # Step 5 — Save combined data
        # --------------------------------------------------
        print("\n--- Step 5 : Save combined data ---")
        combined_path = "data/combined_data.csv"
        combined.to_csv(combined_path, index=False)
        print(f"  Combined data saved to : {combined_path}")

        final_data = combined
        data_source = "combined"
        new_rows    = len(new_data)

    else:
        # No new data — use original only
        print(f"  No new data found")
        print(f"  Using original data only")
        final_data  = original
        data_source = "original"
        new_rows    = 0

    # ----------------------------------------------------------
    # Step 6 — Save metadata
    # ----------------------------------------------------------
    print("\n--- Step 6 : Save metadata ---")

    # Create metadata folder if not exists
    os.makedirs("metadata", exist_ok=True)

    metadata = {
        "last_run"        : datetime.now().isoformat(),
        "data_source"     : data_source,
        "original_rows"   : len(original),
        "new_rows_added"  : new_rows,
        "final_rows"      : len(final_data),
        "churn_rate"      : round(
            (final_data['Churn'] == 'Yes').mean() * 100, 2
        )
    }

    with open("metadata/pipeline_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"  Metadata saved!")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n" + "=" * 55)
    print("  DATA LOADING COMPLETE!")
    print("=" * 55)
    print(f"  Data source    : {data_source}")
    print(f"  Original rows  : {len(original)}")
    print(f"  New rows added : {new_rows}")
    print(f"  Final rows     : {len(final_data)}")
    print(f"  Churn rate     : {metadata['churn_rate']}%")
    print("=" * 55)

    return final_data


# ----------------------------------------------------------------
# Run directly
# ----------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    if data is None:
        print("\nData loading FAILED")
        sys.exit(1)
    else:
        print("\nData loading PASSED")
        print("Ready for feature engineering")
        sys.exit(0)
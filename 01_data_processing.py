"""
Flight Data Analysis Pipeline - Step 1: Process data
==========================================

This script coordinates the preprocessing pipeline:
1. Runs DuckDB CSV processor to create a processed CSV
2. Converts the processed CSV to Parquet
3. Runs a sample data check on the Parquet file

You can also run each step independently.
"""

import subprocess
import sys
import os

def run_script(script_name):
    print(f"\n=== Running: {script_name} ===")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error running {script_name}. Exiting pipeline.")
        sys.exit(1)

def main():
    scripts = [
        "01a_data_preprocess_DuckDB.py",
        "01b_convert_csv_to_parquet.py",
        "01c_check_parquet_sample.py"
    ]
    for script in scripts:
        if not os.path.exists(script):
            print(f"Script not found: {script}")
            sys.exit(1)
        run_script(script)
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()

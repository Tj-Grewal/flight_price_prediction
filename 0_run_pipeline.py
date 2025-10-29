"""
Complete Flight Data Analysis Pipeline Runner
============================================

This script runs the complete flight data analysis pipeline in the correct order:
1. Data Processing (CSV -> Parquet conversion) (this runs 3 subsets of processes)
2. Data Cleaning and Feature generation
3. Exploratory Data Analysis
4. Data Visualization
5. Statistical Validation

Run this script to execute the entire pipeline from start to finish.

################
TODO for reader:
################
1. Make sure you have the correct files placed in the same folder as the script:
    Necessary: "itineraries.csv" <- the original file that is 30GB
    The pipeline will create all the files afterwards as necessary
2. Right now, the analysis happens on all data, if you want to speed up the 
    analysis, you can modify step 3, 4, and 5 to change the amount of data
    that the analysis is run on. I have set the default to full dataset 
    but I provide an option to run it at 0.01% of the data which is still 750K rows
"""

import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name, description):
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_name):
        print(f"ERROR: Script not found: {script_name}")
        sys.exit(1)
    
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"ERROR: {script_name} failed with return code {result.returncode}")
        print("Pipeline execution stopped.")
        sys.exit(1)
    
    print(f"{script_name} completed successfully")

def main():
    """Run the complete flight data analysis pipeline"""
    print("="*60)
    print("FLIGHT DATA ANALYSIS PIPELINE")
    print("Complete Tuesday Flight Analysis")
    print("="*60)
    print(f"Pipeline started at: {datetime.now()}")
    
    # Define the pipeline steps
    pipeline_steps = [
        ("01_data_processing.py", "Data Processing"),
        ("02_data_cleaning.py", "Data Cleaning & Feature Generation"),
        ("03_exploratory_analysis.py", "Exploratory Data Analysis"),
        ("04_data_visualization.py", "Data Visualization"),
        ("05_statistical_validation.py", "Statistical Validation")
    ]
    
    try:
        for script_name, description in pipeline_steps:
            run_script(script_name, description)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"All analysis steps completed at: {datetime.now()}")
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

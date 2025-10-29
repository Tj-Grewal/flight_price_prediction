# Flight Data Analysis Pipeline

A complete analysis pipeline to determine if Tuesday flights are cheaper and identify optimal booking patterns.

## Overview

This pipeline processes 30GB of flight data to answer key questions:
- Are Tuesday flights actually cheaper?
- What's the optimal booking window?
- How do prices vary by season?

## Pipeline Steps

### 1. Data Processing (`01_data_processing.py`)
Coordinates the initial data preprocessing:
- **`01a_data_preprocess_DuckDB.py`** - Extracts relevant columns from 30GB CSV using DuckDB
- **`01b_convert_csv_to_parquet.py`** - Converts processed CSV to efficient Parquet format
- **`01c_check_parquet_sample.py`** - Validates the converted data

### 2. Data Cleaning (`02_data_cleaning.py`)
Cleans and prepares data for analysis:
- Removes invalid records and outliers
- Creates derived features (day of week, booking windows)
- Handles missing values

### 3. Exploratory Analysis (`03_exploratory_analysis.py`)
Performs initial data exploration:
- Analyzes Tuesday vs other days pricing
- Examines booking window patterns
- Explores seasonal price variations

### 4. Data Visualization (`04_data_visualization.py`)
Creates charts and graphs:
- Tuesday savings analysis charts
- Booking timing optimization plots
- Route and fare distribution analysis

### 5. Statistical Validation (`05_statistical_validation.py`)
Provides statistical testing:
- T-tests for Tuesday/Wednesday effects
- ANOVA for booking timing analysis
- Linear regression for controlled analysis

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Place your `itineraries.csv` file in the project directory
3. Run the complete pipeline: `python 0_run_pipeline.py`

## Output Files

- `flight_data_ready.parquet` - Cleaned dataset ready for analysis
- `figures/` - Generated visualization charts
- Console output with statistical results and key findings


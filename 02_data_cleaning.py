"""
Flight Data Analysis Pipeline - Step 2: Data Cleaning and Validation
====================================================================

This script handles data cleaning and validation:
1. Loads the preprocessed data
2. Handles missing values and data quality issues
3. Creates derived features for analysis
4. Saves the cleaned dataset

"""

import dask.dataframe as dd
import duckdb
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_preprocessed_data(file_path):
    
    print("Loading preprocessed flight data with Dask...")
    df = dd.read_parquet(file_path)
    return df


def validate_and_clean_data(df):

    print("\n" + "="*50)
    print("DATA VALIDATION AND CLEANING")
    print("="*50)
    
    original_rows = len(df)
    print(f"Starting with {original_rows:,} rows")
    
    # Remove rows with missing critical data
    print("\n1. Removing rows with missing critical data...")
    critical_columns = ['searchDate', 'flightDate', 'startingAirport', 'destinationAirport', 'baseFare']
    before_missing = len(df)
    df = df.dropna(subset=critical_columns)
    after_missing = len(df)
    removed_missing = before_missing - after_missing
    print(f"   Removed {removed_missing:,} rows with missing critical data")
    if removed_missing > 0:
        print(f"   Remaining rows: {after_missing:,}")
    
    # Remove unrealistic fare values
    print("\n2. Removing unrealistic fare values...")
    before_fare = len(df)
    df = df[(df['baseFare'] > 0) & (df['baseFare'] < 1000)]
    after_fare = len(df)
    removed_fare = before_fare - after_fare
    print(f"   Removed {removed_fare:,} rows with unrealistic fares (<=0 or >=1000)")
    if removed_fare > 0:
        print(f"   Remaining rows: {after_fare:,}")
    
    # Remove invalid date ranges
    print("\n3. Removing invalid date ranges...")
    before_dates = len(df)
    df = df[df['flightDate'] >= df['searchDate']]
    after_dates = len(df)
    removed_dates = before_dates - after_dates
    print(f"   Removed {removed_dates:,} rows where flight date is before search date")
    if removed_dates > 0:
        print(f"   Remaining rows: {after_dates:,}")
    
    # Clean airport codes (should be 3 characters)
    print("\n4. Cleaning airport codes...")
    before_airports = len(df)
    df = df[df['startingAirport'].str.len() == 3]
    df = df[df['destinationAirport'].str.len() == 3]
    df = df[df['startingAirport'] != df['destinationAirport']]
    after_airports = len(df)
    removed_airports = before_airports - after_airports
    print(f"   Removed {removed_airports:,} rows with invalid airport codes")
    if removed_airports > 0:
        print(f"   Remaining rows: {after_airports:,}")
    
    # Handle missing values in non-critical columns
    print("\n5. Handling missing values...")
    
    # Fill missing boolean values
    boolean_columns = ['isBasicEconomy', 'isNonStop']
    for col in boolean_columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum().compute()
            if missing_count > 0:
                print(f"   Filling {missing_count:,} missing values in {col} with False")
                df[col] = df[col].fillna(False)
    
    # Fill missing numeric values with computed values
    if 'seatsRemaining' in df.columns:
        missing_seats = df['seatsRemaining'].isnull().sum().compute()
        if missing_seats > 0:
            median_seats = int(df['seatsRemaining'].quantile(0.5).compute())
            print(f"   Filling {missing_seats:,} missing values in seatsRemaining with median: {median_seats}")
            df['seatsRemaining'] = df['seatsRemaining'].fillna(median_seats)
    
    # Remove rows with missing distance (critical route information)
    if 'totalTravelDistance' in df.columns:
        missing_distance = df['totalTravelDistance'].isnull().sum().compute()
        if missing_distance > 0:
            print(f"   Removing {missing_distance:,} rows with missing totalTravelDistance")
            original_dtype = df['totalTravelDistance'].dtype
            df = df.dropna(subset=['totalTravelDistance'])
            if original_dtype == 'int64':
                df['totalTravelDistance'] = df['totalTravelDistance'].astype('int64')
            after_distance = len(df)
            print(f"   Remaining rows: {after_distance:,}")
    
    final_rows = len(df)
    total_removed = original_rows - final_rows
    print(f"\nCleaning completed:")
    print(f"   Original rows: {original_rows:,}")
    print(f"   Final rows: {final_rows:,}")
    print(f"   Total removed: {total_removed:,} ({(total_removed/original_rows)*100:.1f}%)")
    
    return df

def create_derived_features(df):    
    print("\n" + "="*50)
    print("CREATING DERIVED FEATURES")
    print("="*50)
    
    # Calculate days between search and flight (booking advance time)
    print("1. Creating booking advance time...")
    df['booking_days_advance'] = (df['flightDate'] - df['searchDate']).dt.days
    
    print("2. Extracting key date components...")
     # Monday=0, Sunday=6
    df['flight_day_of_week'] = df['flightDate'].dt.dayofweek 
    df['flight_month'] = df['flightDate'].dt.month
    
    # Create 10-day window categories
    print("4. Creating 10-day window categories...")
    def categorize_booking_10_day_windows(days):
        if days < 0:
            return '-1'
        elif days <= 10:
            return '0-10'
        elif days <= 20:
            return '11-20'
        elif days <= 30:
            return '21-30'
        elif days <= 40:
            return '31-40'
        elif days <= 50:
            return '41-50'
        elif days <= 60:
            return '51-60'
        else:
            return '61'
    
    df['booking_10_day_window'] = df['booking_days_advance'].apply(categorize_booking_10_day_windows, meta=('booking_10_day_window', 'object'))
    
    print(f"\nDerived features created successfully!")
    print(f"Dataset now has {len(df.columns)} columns")
    
    return df

def main():
    print("="*60)
    print("FLIGHT DATA CLEANING AND VALIDATION")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Load preprocessed data
    input_file = "itineraries_processed.parquet"
    output_file = "flight_data_ready.parquet"
    
    try:
        if os.path.exists(output_file):
            print(f"   {output_file} already exists. Skipping Parquet creation...")
        else:
            df = load_preprocessed_data(input_file)
            df_cleaned = validate_and_clean_data(df)
            df_final = create_derived_features(df_cleaned)
            
            print(f"\nSaving final cleaned data to: {output_file}")
            print("   Converting Dask DataFrame to DuckDB and saving as Parquet file")
            partitions = df_final.to_delayed()
            
            conn = duckdb.connect()
            
            sample_partition = partitions[0].compute()
            sample_partition = sample_partition.reset_index(drop=True)
            conn.execute("CREATE TABLE flight_data AS SELECT * FROM sample_partition WHERE 1=0")
            
            # Process each partition and append to table
            for i, partition in enumerate(partitions):
                print(f"   Processing partition {i+1}/{len(partitions)}...")
                df_chunk = partition.compute()
                df_chunk = df_chunk.reset_index(drop=True)
                conn.execute("INSERT INTO flight_data SELECT * FROM df_chunk")
            
            # Write to Parquet
            print("   Writing optimized Parquet with DuckDB...")
            conn.execute(f"""
                COPY flight_data 
                TO '{output_file}' 
                (FORMAT PARQUET)
            """)
            
            # Clean up DuckDB table
            conn.execute("DROP TABLE IF EXISTS flight_data")
            conn.close()
            print(f"   File generated: {output_file}")
        
        if os.path.exists(output_file):
            file_size_gb = os.path.getsize(output_file) / 1024**3
            print(f"   Final file size: {file_size_gb:.2f} GB")
        
        print(f"\nData cleaning completed successfully at: {datetime.now()}")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please run 01_data_preprocessing.py first")
    except Exception as e:
        print(f"Error during data cleaning: {str(e)}")

if __name__ == "__main__":
    main()

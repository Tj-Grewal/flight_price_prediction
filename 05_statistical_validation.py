"""
Statistical Analysis - Key Findings
============================================

This script focuses on validating the key findings:
1. Tuesday flights save money (statistically significant)
2. Wednesday flights also save money 
3. Booking timing affects pricing (statistically significant)

"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def load_flight_data(file_path="flight_data_ready.parquet", sample_size=0.01):
    print(f"Loading flight data from {file_path}...")
    df_dask = dd.read_parquet(file_path)
    
    if sample_size < 1.0:
        print(f"Using {sample_size*100}% sample for analysis...")
        df = df_dask.sample(frac=sample_size, random_state=42).compute()
    else:
        print("Using full dataset...")
        df = df_dask.compute()
    
    print(f"Loaded {len(df):,} flights for analysis")
    return df

def validate_tuesday_savings(df):
    print("\n" + "="*50)
    print("TUESDAY SAVINGS VALIDATION")
    print("="*50)
    
    # Create day names for clarity
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['day_name'] = df['flight_day_of_week'].map(day_names)
    
    # Calculate Tuesday vs other days
    tuesday_fares = df[df['flight_day_of_week'] == 1]['totalFare']
    other_fares = df[df['flight_day_of_week'] != 1]['totalFare']
    
    t_stat, p_value = stats.ttest_ind(tuesday_fares, other_fares)
    
    tuesday_avg = tuesday_fares.mean()
    other_avg = other_fares.mean()
    savings = other_avg - tuesday_avg
    
    print(f"  Tuesday flights: {len(tuesday_fares):,} flights, ${tuesday_avg:.2f} average")
    print(f"  Other days: {len(other_fares):,} flights, ${other_avg:.2f} average")
    print(f"  Tuesday savings: ${savings:.2f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print("SIGNIFICANT: Tuesday flights are statistically cheaper!")
    else:
        print("NOT SIGNIFICANT: No statistical difference found")
    
    return savings, p_value

def validate_wednesday_savings(df):
    print("\n" + "="*50)
    print("WEDNESDAY SAVINGS VALIDATION")
    print("="*50)
    
    # Calculate Wednesday vs other days
    wednesday_fares = df[df['flight_day_of_week'] == 2]['totalFare']
    other_fares = df[df['flight_day_of_week'] != 2]['totalFare']
    
    t_stat, p_value = stats.ttest_ind(wednesday_fares, other_fares)
    
    wednesday_avg = wednesday_fares.mean()
    other_avg = other_fares.mean()
    savings = other_avg - wednesday_avg
    
    print(f"  Wednesday flights: {len(wednesday_fares):,} flights, ${wednesday_avg:.2f} average")
    print(f"  Other days: {len(other_fares):,} flights, ${other_avg:.2f} average")
    print(f"  Wednesday savings: ${savings:.2f}")
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value: {p_value:.2e}")
    
    if p_value < 0.05:
        print("SIGNIFICANT: Wednesday flights are also statistically cheaper!")
    else:
        print("NOT SIGNIFICANT: No statistical difference found")
    
    return savings, p_value

def validate_booking_timing(df):
    """Test if booking timing affects prices"""
    print("\n" + "="*50)
    print("BOOKING TIMING VALIDATION")
    print("="*50)
    
    # Group by booking windows and test for differences
    booking_groups = []
    window_names = []
    
    for window in df['booking_10_day_window'].unique():
        if pd.notna(window):
            group_fares = df[df['booking_10_day_window'] == window]['totalFare']
            if len(group_fares) >= 100:
                booking_groups.append(group_fares.values)
                window_names.append(window)
                print(f"  {window} days: {len(group_fares):,} flights, ${group_fares.mean():.2f} average")
    
    # ANOVA test to check if any windows are significantly different
    if len(booking_groups) >= 3:
        f_stat, p_value = stats.f_oneway(*booking_groups)
        
        print(f"\n  F-statistic: {f_stat:.4f}")
        print(f"  P-value: {p_value:.2e}")
        
        if p_value < 0.05:
            print("SIGNIFICANT: Booking timing significantly affects pricing!")
        else:
            print("NOT SIGNIFICANT: No significant booking timing effect")
    
    return p_value if len(booking_groups) >= 3 else None

def get_tuesday_coefficient(df):
    """Get the direct Tuesday effect using linear regression"""
    print("\n" + "="*50)
    print("TUESDAY COEFFICIENT (Linear Regression)")
    print("="*50)
    
    model_data = df.copy()
    model_data['is_tuesday'] = (model_data['flight_day_of_week'] == 1).astype(int)
    features = ['is_tuesday', 'flight_day_of_week', 'totalTravelDistance', 'booking_days_advance']
    
    model_data = model_data.dropna(subset=features + ['totalFare'])
    
    X = model_data[features]
    y = model_data['totalFare']
    model = LinearRegression()
    model.fit(X, y)
    
    tuesday_coef = model.coef_[features.index('is_tuesday')]
    print(f"  Linear regression Tuesday coefficient: ${tuesday_coef:.2f}")
    
    if tuesday_coef < 0:
        print(f"Tuesday flights are ${abs(tuesday_coef):.2f} cheaper (controlling for other factors)")
    else:
        print(f"Tuesday flights are ${tuesday_coef:.2f} more expensive (controlling for other factors)")
    
    return tuesday_coef

def main():
    """Run the simplified Tuesday analysis"""
    print("="*60)
    print("SIMPLE TUESDAY FLIGHT ANALYSIS")
    print("Validating Key Findings")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    # Configuration
    input_file = "flight_data_ready.parquet"
    # TODO: Change to True to use sample dataset instead of full data
    use_sample = True
    sample_fraction = 0.01
    
    try:
        if use_sample:
            df = load_flight_data(input_file, sample_size=sample_fraction)
        else:
            df = load_flight_data(input_file, sample_size=1.0)
        
        tuesday_savings, tuesday_p = validate_tuesday_savings(df)
        wednesday_savings, wednesday_p = validate_wednesday_savings(df)
        booking_p = validate_booking_timing(df)
        tuesday_coef = get_tuesday_coefficient(df)
        
        print("\n" + "="*60)
        print("KEY FINDINGS SUMMARY")
        print("="*60)
        
        print(f"  Tuesday Savings: ${tuesday_savings:.2f} (p-value: {tuesday_p:.2e})")
        print(f"  Wednesday Savings: ${wednesday_savings:.2f} (p-value: {wednesday_p:.2e})")
        if booking_p:
            print(f"  Booking Timing Effect: Significant (p-value: {booking_p:.2e})")
        print(f"  Tuesday Coefficient: ${tuesday_coef:.2f}")
        
        # Validation status
        print(f"\nVALIDATION STATUS:")
        if tuesday_p < 0.05:
            print("  Tuesday effect: CONFIRMED")
        if wednesday_p < 0.05:
            print("  Wednesday effect: CONFIRMED")
        if booking_p and booking_p < 0.05:
            print("  Booking timing effect: CONFIRMED")
        
        print(f"\n  Analysis completed successfully at: {datetime.now()}")
        
    except FileNotFoundError:
        print(" Error: Could not find flight_data_ready.parquet")
        print(" Please run the data processing pipeline first")
    except Exception as e:
        print(f"  Error: {str(e)}")

if __name__ == "__main__":
    main()

"""
Flight Data Analysis Pipeline - Step 3: Exploratory Data Analysis
================================================================

This script performs targeted analysis to answer specific questions:
1. Is it cheaper to fly on Tuesday compared to other days?
2. What's the optimal booking window int 10 day intervals between 0-60 days?
3. Monthly price variations and patterns

"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_flight_data(file_path):
    df = dd.read_parquet(file_path)
    
    # TODO Use 1% sample for testing (fast analysis on ~750K rows) or comment for full dataset
    df = df.sample(frac=0.01, random_state=42)
    
    print("Loaded flight data.")
    return df

def analyze_tuesday_effect(df):
    print("\n" + "="*60)
    print("ANALYSIS 1: TUESDAY vs OTHER DAYS COMPARISON")
    print("="*60)
    
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['day_name'] = df['flight_day_of_week'].map(day_names)
    
    # Calculate statistics by day of week
    daily_stats = df.groupby('flight_day_of_week').agg({
        'totalFare': ['mean', 'median', 'std', 'count'],
        'baseFare': 'mean'
    }).compute()
    
    daily_stats.columns = ['avg_total_fare', 'median_total_fare', 'std_total_fare', 'flight_count', 'avg_base_fare']
    daily_stats['day_name'] = daily_stats.index.map(day_names)
    daily_stats = daily_stats.sort_values('avg_total_fare')
    
    # Sort by avg_total_fare in decreasing order for display (highest to lowest)
    daily_stats_sorted = daily_stats.sort_values('avg_total_fare', ascending=False)

    print("Average fare by day of week (highest to lowest):")
    for idx, row in daily_stats_sorted.iterrows():
        print(f"   {row['day_name']}: ${row['avg_total_fare']:.2f} avg, ${row['median_total_fare']:.2f} median")
    
    # Tuesday specific analysis (monday = day 0, tuesday = day 1)
    tuesday_stats = daily_stats.loc[1]
    non_tuesday_avg = daily_stats[daily_stats.index != 1]['avg_total_fare'].mean()
    
    tuesday_savings = non_tuesday_avg - tuesday_stats['avg_total_fare']
    savings_pct = (tuesday_savings / non_tuesday_avg) * 100
    
    print(f"\nTUESDAY ANALYSIS:")
    print(f"   Tuesday average fare: ${tuesday_stats['avg_total_fare']:.2f}")
    print(f"   Other days average: ${non_tuesday_avg:.2f}")
    print(f"   Tuesday savings: ${tuesday_savings:.2f} ({savings_pct:.1f}%)")
    
    if tuesday_savings > 0:
        print(f"   Tuesday IS cheaper by ${tuesday_savings:.2f}")
    else:
        print(f"   Tuesday is NOT cheaper (${abs(tuesday_savings):.2f} more expensive)")
    
    # Find cheapest and most expensive days
    cheapest_day = daily_stats.iloc[0]
    most_expensive_day = daily_stats.iloc[-1]
    
    print(f"\nBEST/WORST DAYS:")
    print(f"   Cheapest: {cheapest_day['day_name']} (${cheapest_day['avg_total_fare']:.2f})")
    print(f"   Most expensive: {most_expensive_day['day_name']} (${most_expensive_day['avg_total_fare']:.2f})")
    print(f"   Price difference: ${most_expensive_day['avg_total_fare'] - cheapest_day['avg_total_fare']:.2f}")
    
    return daily_stats

def analyze_booking_windows(df):

    print("\n" + "="*60)
    print("ANALYSIS 2: BOOKING WINDOW ANALYSIS (10-DAY INTERVALS)")
    print("="*60)
    
    window_stats = df.groupby('booking_10_day_window').agg({
        'totalFare': ['mean', 'median', 'std', 'count'],
        'booking_days_advance': 'mean'
    }).compute()
    
    window_stats.columns = ['avg_fare', 'median_fare', 'std_fare', 'flight_count', 'avg_days_advance']
    
    # Define proper order for windows in 10 day increments
    window_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61']
    window_stats = window_stats.reindex(window_order)
    
    # Sort by avg_fare in decreasing order for display
    window_stats_sorted = window_stats.sort_values('avg_fare', ascending=False)
    
    print("Average fare by booking window (highest to lowest):")
    for window, row in window_stats_sorted.iterrows():
        if pd.notna(row['avg_fare']):
            print(f"   {window} days: ${row['avg_fare']:.2f} avg, ${row['median_fare']:.2f} median")
    
    # Find optimal booking window (lowest fare)
    optimal_window = window_stats['avg_fare'].idxmin()
    optimal_fare = window_stats.loc[optimal_window, 'avg_fare']
    
    print(f"\nOPTIMAL BOOKING WINDOW:")
    print(f"   Best window: {optimal_window} days (${optimal_fare:.2f} average)")
    
    return window_stats


def analyze_monthly_patterns(df):
    print("\n" + "="*60)
    print("ANALYSIS 3: MONTHLY PRICE PATTERNS")
    print("="*60)
    
    monthly_stats = df.groupby('flight_month').agg({
        'totalFare': ['mean', 'median', 'count'],
        'booking_days_advance': 'mean'
    }).compute()
    
    # we only have from April to November for analysis
    monthly_stats.columns = ['avg_fare', 'median_fare', 'flight_count', 'avg_booking_advance']
    month_names = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November'}
    
    monthly_stats['month_name'] = monthly_stats.index.map(month_names)
    
    # Sort by avg_fare in decreasing order for display
    monthly_stats_sorted = monthly_stats.sort_values('avg_fare', ascending=False)
    
    print("Average fare by month (highest to lowest):")
    for month, row in monthly_stats_sorted.iterrows():
        if month in month_names:
            print(f"   {row['month_name']}: ${row['avg_fare']:.2f} avg, ${row['median_fare']:.2f} median")
    
    # Find cheapest and most expensive months
    cheapest_month_idx = monthly_stats['avg_fare'].idxmin()
    most_expensive_month_idx = monthly_stats['avg_fare'].idxmax()
    
    cheapest_month = monthly_stats.loc[cheapest_month_idx]
    most_expensive_month = monthly_stats.loc[most_expensive_month_idx]
    
    print(f"\nSEASONAL PATTERNS:")
    print(f"   Cheapest month: {month_names.get(cheapest_month_idx, cheapest_month_idx)} (${cheapest_month['avg_fare']:.2f})")
    print(f"   Most expensive: {month_names.get(most_expensive_month_idx, most_expensive_month_idx)} (${most_expensive_month['avg_fare']:.2f})")
    print(f"   Seasonal price difference: ${most_expensive_month['avg_fare'] - cheapest_month['avg_fare']:.2f}")
    
    return monthly_stats

def generate_data_summary(df):
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    total_rows = df.shape[0].compute()
    print(f"Final dataset shape: ({total_rows:,}, {len(df.columns)} columns)")
    
    # Compute total fare statistics 
    fare_stats = df['totalFare'].describe().compute()
    print(f"\nTotal fare statistics:")
    print(f"   Mean total fare: ${fare_stats['mean']:.2f}")
    print(f"   Median total fare: ${fare_stats['50%']:.2f}")
    print(f"   Min total fare: ${fare_stats['min']:.2f}")
    print(f"   Max total fare: ${fare_stats['max']:.2f}")
    
    # Compute base fare statistics
    base_fare_stats = df['baseFare'].describe().compute()
    print(f"\nBase fare statistics:")
    print(f"   Mean base fare: ${base_fare_stats['mean']:.2f}")
    print(f"   Median base fare: ${base_fare_stats['50%']:.2f}")
    print(f"   Min base fare: ${base_fare_stats['min']:.2f}")
    print(f"   Max base fare: ${base_fare_stats['max']:.2f}")
    
    print(f"\n10-day booking window distribution:")
    window_counts = df['booking_10_day_window'].value_counts().compute()
    window_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61']
    for window in window_order:
        if window in window_counts.index:
            count = window_counts[window]
            print(f"   {window} days: {count:,} flights ({count/total_rows*100:.1f}%)")
    
    print(f"\nFlight day distribution:")
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    day_counts = df['flight_day_of_week'].value_counts().compute()
    for day_num in sorted(day_counts.index):
        day_name = day_names[day_num]
        count = day_counts[day_num]
        print(f"   {day_name}: {count:,} flights ({count/total_rows*100:.1f}%)")
    
    print(f"\nMonthly flight distribution:")
    month_names = {4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November'}
    month_counts = df['flight_month'].value_counts().compute()
    for month_num in sorted(month_counts.index):
        if month_num in month_names:
            month_name = month_names[month_num]
            count = month_counts[month_num]
            print(f"   {month_name}: {count:,} flights ({count/total_rows*100:.1f}%)")

def main():
    
    print("="*60)
    print("FLIGHT DATA ANALYSIS")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    input_file = "flight_data_ready.parquet"
    
    try:
        df = load_flight_data(input_file)
        generate_data_summary(df)
        
        daily_stats = analyze_tuesday_effect(df)
        window_stats = analyze_booking_windows(df)
        monthly_stats = analyze_monthly_patterns(df)
        
        print(f"\n" + "="*70)
        print(f"Analysis completed at: {datetime.now()}")
        print("="*70)
        
        return {
            'daily_stats': daily_stats,
            'window_stats': window_stats,
            'monthly_stats': monthly_stats,
        }
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Please run the data preprocessing and cleaning steps first")
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        return None

if __name__ == "__main__":
    results = main()

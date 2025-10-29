"""
Flight Data Analysis Pipeline - Step 4: Data Visualization
=========================================================

This script creates comprehensive visualizations for flight pricing analysis:
1. Tuesday vs other days analysis
2. Booking timing optimization charts & Monthly pricing patterns
3. Route and fare distribution analysis

"""

import dask.dataframe as dd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data_for_visualization(file_path, use_sample=True, sample_fraction=0.01):
    print(f"Loading data from {file_path}...")
    df_dask = dd.read_parquet(file_path)
    
    if use_sample:
        print(f"Using {sample_fraction*100}% sample for visualization...")
        df = df_dask.sample(frac=sample_fraction, random_state=42).compute()
    else:
        print("Using full dataset...")
        df = df_dask.compute()
    
    print(f"Loaded dataset shape: {df.shape}")
    return df


def create_tuesday_analysis_chart(df):    
    print("Creating Tuesday vs other days analysis...")
    
    day_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['flight_day_name'] = df['flight_day_of_week'].map(day_names)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Tuesday vs Other Days Pricing Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average fare by day of week (sorted)
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_data = df.groupby('flight_day_name')['totalFare'].agg(['mean', 'median', 'count']).reindex(day_order)
    day_data_sorted = day_data.sort_values('mean', ascending=False)
    
    colors = ['green' if day == 'Tuesday' else 'orange' if day == 'Wednesday' else 'lightblue' 
              for day in day_data_sorted.index]
    
    bars = axes[0].bar(range(len(day_data_sorted)), day_data_sorted['mean'], 
                       color=colors, edgecolor='black', alpha=0.8)
    axes[0].set_title('Average Fare by Flight Day (Sorted)', fontweight='bold')
    axes[0].set_ylabel('Average Fare ($)')
    axes[0].set_xticks(range(len(day_data_sorted)))
    axes[0].set_xticklabels(day_data_sorted.index, rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5, f'${height:.0f}', ha='center', va='bottom', fontsize=9)
    
    # Calculate and display Tuesday savings
    tuesday_fare = day_data.loc['Tuesday', 'mean']
    other_days_fare = day_data.drop('Tuesday')['mean'].mean()
    savings = other_days_fare - tuesday_fare
    savings_pct = (savings / other_days_fare) * 100
    
    axes[0].text(0.98, 0.98, f'Tuesday Savings: ${savings:.2f} ({savings_pct:.1f}%)', 
                 transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                 verticalalignment='top', horizontalalignment='right', fontweight='bold')
    
    # 2. Distribution comparison - Tuesday vs Other days
    tuesday_fares = df[df['flight_day_of_week'] == 1]['totalFare']
    other_fares = df[df['flight_day_of_week'] != 1]['totalFare']
    
    axes[1].hist([tuesday_fares, other_fares], bins=50, alpha=0.7, 
                 label=['Tuesday', 'Other Days'], color=['red', 'lightblue'], edgecolor='black')
    axes[1].set_title('Fare Distribution: Tuesday vs Other Days', fontweight='bold')
    axes[1].set_xlabel('Fare ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('figures/01_tuesday_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return day_data


def create_booking_timing_and_monthly_charts(df):
    print("Creating booking timing and monthly pattern visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Booking Timing and Monthly Analysis', fontsize=16, fontweight='bold')
    window_data = df.groupby('booking_10_day_window')['totalFare'].agg(['mean', 'median', 'count']).reset_index()

    # 1. Define the order for booking windows - optimal booking days
    window_order = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60']
    window_data['window_order'] = window_data['booking_10_day_window'].apply(lambda x: window_order.index(x) if x in window_order else 999)
    window_data = window_data.sort_values('window_order') 

    bars = axes[0].bar(range(len(window_data)), window_data['mean'], 
                      color=sns.color_palette("viridis", len(window_data)), alpha=0.8, edgecolor='black')
    axes[0].set_title('Average Fare by Booking Window', fontweight='bold')
    axes[0].set_xlabel('Booking Window')
    axes[0].set_ylabel('Average Fare ($)')
    axes[0].set_xticks(range(len(window_data)))
    axes[0].set_xticklabels(window_data['booking_10_day_window'], rotation=45)

    # Add value labels and highlight optimal window
    optimal_window = window_data.loc[window_data['mean'].idxmin(), 'booking_10_day_window']
    for i, bar in enumerate(bars):
        height = bar.get_height()
        color = 'red' if window_data.iloc[i]['booking_10_day_window'] == optimal_window else 'black'
        weight = 'bold' if window_data.iloc[i]['booking_10_day_window'] == optimal_window else 'normal'
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'${height:.0f}', ha='center', va='bottom', fontsize=9, 
                    color=color, fontweight=weight)

    axes[0].text(0.98, 0.98, f'Optimal Window: {optimal_window}', 
                transform=axes[0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                verticalalignment='top', horizontalalignment='right', fontweight='bold')

    # 2. Monthly pricing trends
    monthly_data = df.groupby('flight_month')['totalFare'].agg(['mean', 'median', 'count'])
    month_names = {4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov'}

    # Filter to available months
    available_months = monthly_data.index.tolist()
    available_month_names = [month_names[i] for i in available_months]

    axes[1].plot(available_months, monthly_data['mean'], marker='o', linewidth=3, markersize=8, color='blue')
    axes[1].set_title('Average Fare by Month', fontweight='bold')
    axes[1].set_xlabel('Month')
    axes[1].set_ylabel('Average Fare ($)')
    axes[1].set_xticks(available_months)
    axes[1].set_xticklabels(available_month_names, rotation=45)
    axes[1].grid(True, alpha=0.3)

    # Highlight peak and low months
    peak_month = monthly_data['mean'].idxmax()
    low_month = monthly_data['mean'].idxmin()
    axes[1].scatter([peak_month], [monthly_data.loc[peak_month, 'mean']], 
                   color='red', s=100, zorder=5, label=f'Peak: {month_names[peak_month]}')
    axes[1].scatter([low_month], [monthly_data.loc[low_month, 'mean']], 
                   color='green', s=100, zorder=5, label=f'Low: {month_names[low_month]}')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('figures/02_booking_timing_monthly_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()
    
    return window_data, monthly_data


def create_route_fare_analysis(df):
    print("Creating route and fare analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Route and Fare Distribution Analysis', fontsize=16, fontweight='bold')

    # 1. Fare distribution by travel distance (scatter)
    sample_df = df.sample(n=min(5000, len(df)))
    scatter = axes[0].scatter(sample_df['totalTravelDistance'], sample_df['totalFare'], 
                             alpha=0.6, s=20, c=sample_df['totalFare'], cmap='viridis')

    # Add trend line if we have valid data
    valid_data = sample_df.dropna(subset=['totalTravelDistance', 'totalFare'])
    if len(valid_data) > 10:
        z = np.polyfit(valid_data['totalTravelDistance'], valid_data['totalFare'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(valid_data['totalTravelDistance'].min(), 
                             valid_data['totalTravelDistance'].max(), 100)
        axes[0].plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        axes[0].legend()

    axes[0].set_title('Fare vs Travel Distance', fontweight='bold')
    axes[0].set_xlabel('Travel Distance (miles)')
    axes[0].set_ylabel('Fare ($)')

    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Fare ($)')

    # 2. Fare distribution histogram
    axes[1].hist(df['totalFare'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].axvline(x=df['totalFare'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${df["totalFare"].mean():.2f}')
    axes[1].axvline(x=df['totalFare'].median(), color='orange', linestyle='--', 
                   label=f'Median: ${df["totalFare"].median():.2f}')
    axes[1].set_title('Overall Fare Distribution', fontweight='bold')
    axes[1].set_xlabel('Fare ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('figures/03_route_fare_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()

    return None

def main():
    
    print("="*60)
    print("FLIGHT DATA VISUALIZATION PIPELINE")
    print("="*60)
    print(f"Started at: {datetime.now()}")
    
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created {figures_dir} directory")
    
    # Configuration
    input_file = "flight_data_ready.parquet"
    # TODO: Change to True to use sample dataset instead of full data
    use_sample = True
    sample_fraction = 0.01 
    
    try:
        df = load_data_for_visualization(input_file, use_sample, sample_fraction)
        print("\nGenerating visualizations...")
        
        # 1. Tuesday analysis
        day_data = create_tuesday_analysis_chart(df)
        
        # 2. Booking timing and monthly analysis  
        window_data, monthly_data = create_booking_timing_and_monthly_charts(df)
        
        # 3. Route and fare analysis
        route_data = create_route_fare_analysis(df)
        
        # Key findings summary
        optimal_window = window_data.loc[window_data['mean'].idxmin(), 'booking_10_day_window']
        tuesday_savings = day_data.drop('Tuesday')['mean'].mean() - day_data.loc['Tuesday', 'mean']
        seasonal_range = monthly_data['mean'].max() - monthly_data['mean'].min()
        
        print(f"\nKEY FINDINGS:")
        print(f"   Optimal booking window: {optimal_window}")
        print(f"   Tuesday savings: ${tuesday_savings:.2f}")
        print(f"   Seasonal price range: ${seasonal_range:.2f}")
        
    except Exception as e:
        print(f"Error in visualization pipeline: {str(e)}")
        raise
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()

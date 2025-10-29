import dask.dataframe as dd
import time

PARQUET_FILE = "itineraries_processed.parquet"

def main():
    start = time.time()
    print(f"Loading data from: {PARQUET_FILE}")

    try:
        df = dd.read_parquet(PARQUET_FILE)
        
        total_rows = df.shape[0].compute()
        print(f"Total rows in dataset: {total_rows:,}\n")

        cheap_flights = (df['baseFare'] < 300).sum().compute()
        print(f"Number of flights cheaper than $300: {cheap_flights:,}\n")

        # Min and max searchDate
        min_search = df['searchDate'].min().compute()
        max_search = df['searchDate'].max().compute()
        print(f"Min searchDate: {min_search}")
        print(f"Max searchDate: {max_search}\n")

        # Min and max flightDate
        min_flight = df['flightDate'].min().compute()
        max_flight = df['flightDate'].max().compute()
        print(f"Min flightDate: {min_flight}")
        print(f"Max flightDate: {max_flight}\n")

        # Calculate the difference in days of booking and search dates
        df['days_diff'] = (df['flightDate'] - df['searchDate']).dt.days
        max_days_diff = df['days_diff'].max().compute()
        print(f"Maximum days between searchDate and flightDate: {max_days_diff} days\n")

        sample_df = df.head(10000)
        print(f"Sample loaded: {sample_df.shape[0]} rows\n")

        print("Data date range (from sample):")
        print(f"Search dates: {sample_df['searchDate'].min()} to {sample_df['searchDate'].max()}")
        print(f"Flight dates: {sample_df['flightDate'].min()} to {sample_df['flightDate'].max()}\n")

        # Example simple calculation: count unique starting airports
        print(f"Unique starting airports in sample: {sample_df['startingAirport'].nunique()}\n")

        print("First 20 rows of the sample:")
        print(sample_df.head(20))

    except Exception as e:
        print(f"Error loading or processing Parquet file: {e}")

    end = time.time()
    print(f"\nTotal runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()

import duckdb
import os

# Convert CSV to Parquet 
# Use TIMESTAMP instead of DATE to create datetime objects that Dask can handle
output_file = 'itineraries_processed.parquet'
if os.path.exists(output_file):
    print(f"{output_file} already exists. Skipping conversion from CSV to Parquet...")
else:
    duckdb.sql("""
        COPY (
            SELECT 
                CAST(searchDate AS TIMESTAMP) AS searchDate,
                CAST(flightDate AS TIMESTAMP) AS flightDate,
                startingAirport, 
                destinationAirport,
                isBasicEconomy,
                isNonStop,
                baseFare,
                totalFare,
                seatsRemaining,
                totalTravelDistance,
                segmentsAirlineCode
            FROM read_csv_auto('itineraries_processed.csv')
        )
        TO 'itineraries_processed.parquet' (FORMAT PARQUET);
    """)
    print("File generated: itineraries_processed.parquet")

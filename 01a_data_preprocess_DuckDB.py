"""
Simple DuckDB CSV Processor
===========================

This script uses one DuckDB SQL command to:
1. Read a 30GB CSV file
2. Select the columns we care about for analysis 
3. Write a new, smaller CSV file

"""
import os
import duckdb

def process_csv():
    output_file = 'itineraries_processed.csv'
    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping creation of CSV file...")
        return

    print("Processing CSV with DuckDB....")
    print("Input  : itineraries.csv (30 GB)")
    print("Output : itineraries_processed.csv (5 GB)")
    
    conn = duckdb.connect()
    print("Running script to drop unnecessary columns using DuckDB...")
    
    sql_command = """
    COPY (
        SELECT 
            searchDate,
            flightDate,
            startingAirport, 
            destinationAirport,
            isBasicEconomy,
            isNonStop,
            baseFare,
            totalFare,
            seatsRemaining,
            totalTravelDistance,
            segmentsAirlineCode
        FROM read_csv_auto('itineraries.csv')
    ) TO 'itineraries_processed.csv' (HEADER, DELIMITER ',');
    """
    
    conn.execute(sql_command)
    print("File generated: itineraries_processed.csv")
    
    conn.close()

if __name__ == "__main__":
    try:
        import duckdb
        process_csv()
    except ImportError:
        print("DuckDB not found. Install with: pip install duckdb")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure itineraries.csv is in the same folder as this script.")

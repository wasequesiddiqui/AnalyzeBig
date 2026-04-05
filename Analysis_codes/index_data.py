from nsepython import index_pe_pb_div as idx
import pandas as pd 

def get_historical_index_stats(symbol, start_date, end_date):
    """
    Fetches historical PE, PB, and Dividend Yield for an NSE index.
    Dates must be in 'DD-MMM-YYYY' format (e.g., '01-Jan-2024').
    """
    try:
        # Use the updated function name: index_pe_pb_div
        df = idx(symbol, start_date, end_date)
        return df
    except Exception as e:
        return f"Error fetching data: {e}"

# Correct Usage
df_it = get_historical_index_stats("NIFTY IT", "01-Apr-2021", "01-Apr-2026")
print(df_it.head())
print(df_it.tail())
# import time

# from nsepython import index_pe_pb_div as idx
# import pandas as pd 

# def get_historical_index_stats(symbol, start_date, end_date):
#     """
#     Fetches historical PE, PB, and Dividend Yield for an NSE index.
#     Dates must be in 'DD-MMM-YYYY' format (e.g., '01-Jan-2024').
#     """
#     try:
#         # Use the updated function name: index_pe_pb_div
#         df = idx(symbol, start_date, end_date)
#         return df
#     except Exception as e:
#         return f"Error fetching data: {e}"

# # Correct Usage
# df_it = get_historical_index_stats("NIFTY IT", "01-Apr-2021", "01-Apr-2026")
# time.sleep(5)  # Sleep for 5 seconds to avoid hitting API rate limits
# df_ph = get_historical_index_stats("NIFTY PHARMA", "01-Apr-2021", "01-Apr-2026")
# time.sleep(5)  # Sleep for 5 seconds to avoid hitting API rate limits
# print(df_it.head())
# print(df_it.tail())
# print(df_ph.head())
# print(df_ph.tail())

from nsepython import index_pe_pb_div as idx
import pandas as pd
import time

def get_historical_index_stats(symbol, start_date, end_date):
    """
    Fetches historical PE, PB, and Dividend Yield for an NSE index.
    """
    try:
        # Note: Large date ranges (>1 year) may return empty DataFrames or Errors 
        # from the NSE server. Consider breaking them into yearly chunks if needed.
        df = idx(symbol, start_date, end_date)
        return df
    except Exception as e:
        return f"Error fetching data for {symbol}: {e}"

def stack_dataframes(df_list):
    """
    Accepts an array (list) of DataFrames and stacks them vertically.
    
    Args:
        df_list (list): A list containing pandas DataFrame objects.
        
    Returns:
        pd.DataFrame: A single merged DataFrame, or an empty DataFrame if the list is empty.
    """
    if not df_list:
        return pd.DataFrame()
    
    # ignore_index=True resets the index so it goes from 0 to N 
    # instead of repeating indices from the original dataframes.
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    
    return merged_df

# List of indices you requested
target_indices = ["NIFTY ENERGY", "NIFTY PHARMA", "NIFTY FMCG", "NIFTY IT"]

# Dictionary to store the results
all_results = []

print("--- Starting Data Extraction ---")

for sector in target_indices:
    print(f"Fetching data for: {sector}...")
    
    # Fetch data
    df = get_historical_index_stats(sector, "01-Apr-2021", "01-Apr-2026")
    
    # Check if we got a valid DataFrame back
    if isinstance(df, pd.DataFrame) and not df.empty:
        print(f"Successfully retrieved {len(df)} rows for {sector}.")
        all_results.append(df)  # Add to list for stacking later
    else:
        print(f"Failed or No Data for {sector}. (Check if date range > 365 days)")
    
    # Small pause to avoid hitting NSE rate limits
    time.sleep(5)

df_merged = stack_dataframes(all_results)
print(f"\nMerged DataFrame has {len(df_merged)} rows.") 
df_merged.head(20)
df_merged.tail(20)

# df_merged['Index Name'].unique()
# print("\n--- Summary Results ---")
# for sector, data in all_results.items():
#     print(f"\n{sector} - Latest 5 Records:")
#     print(data.tail())
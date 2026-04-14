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



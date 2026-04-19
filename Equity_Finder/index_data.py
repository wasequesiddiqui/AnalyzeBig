from nsepython import index_pe_pb_div as idx
import yfinance as yf
import pandas as pd
import time
import log_utils as logger

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
        logger.log_exception(e, "get_historical_index_stats")
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

def calculate_altman_z_score(ticker_symbol):
    """
    Calculates the Altman Z-Score for a given ticker using yfinance.
    Formula: Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        
        # Fetch Financial Statements (using the most recent annual data)
        balance_sheet = ticker.balance_sheet
        income_stmt = ticker.income_stmt
        info = ticker.info

        # Check if dataframes are empty
        if balance_sheet.empty or income_stmt.empty:
            logger.log_exception(ValueError(f"Empty financial data for {ticker_symbol}"), f"calculate_altman_z_score for {ticker_symbol}")
            return 2.99

        balance_sheet_data = balance_sheet.iloc[:, 0]
        income_stmt_data = income_stmt.iloc[:, 0]

        # Required Variables
        total_assets = balance_sheet_data.get('Total Assets')
        current_assets = balance_sheet_data.get('Current Assets')
        current_liab = balance_sheet_data.get('Current Liabilities')
        retained_earnings = balance_sheet_data.get('Retained Earnings')
        total_liab = balance_sheet_data.get('Total Liabilities Net Minority Interest', 
                                       balance_sheet_data.get('Total Liabilities'))
        
        ebit = income_stmt_data.get('EBIT')
        revenue = income_stmt_data.get('Total Revenue')
        
        # Market Value of Equity (Market Cap)
        market_cap = info.get('marketCap')

        # Check for missing data (including NaN from yfinance)
        required_data = [total_assets, current_assets, current_liab, 
                         retained_earnings, total_liab, ebit, revenue, market_cap]
        
        if any(v is None or (isinstance(v, float) and pd.isna(v)) for v in required_data):
            logger.log_exception(ValueError(f"Missing financial data for {ticker_symbol}"), f"calculate_altman_z_score for {ticker_symbol}")
            return 2.99  # Return a default Z-Score indicating potential distress due to missing data

        # Calculate Ratios
        x1 = (current_assets - current_liab) / total_assets  # Working Capital / Total Assets
        x2 = retained_earnings / total_assets               # Retained Earnings / Total Assets
        x3 = ebit / total_assets                            # EBIT / Total Assets
        x4 = market_cap / total_liab                        # Market Cap / Total Liabilities
        x5 = revenue / total_assets                         # Sales / Total Assets

        # Altman Z-Score Calculation
        z_score = (1.2 * x1) + (1.4 * x2) + (3.3 * x3) + (0.6 * x4) + (1.0 * x5)
        
        # Ensure result is numeric
        if pd.isna(z_score):
            return 2.99
        
        return round(float(z_score), 2)

    except Exception as e:
        logger.log_exception(e, f"calculate_altman_z_score for {ticker_symbol}")
        return 2.99  # Return a default Z-Score indicating potential distress due to error
    
# Example Usage:
# symbol = "AAPL"
# z = calculate_altman_z_score(symbol)
# print(f"The Altman Z-Score for {symbol} is: {z}")



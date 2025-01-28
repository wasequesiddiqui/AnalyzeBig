import pandas as pd
import yfinance as yf

#%%
def fetch_balance_sheet(ticker):
    try:
        # Fetch data for the given ticker
        dat = yf.Ticker(ticker)
        
        # Check if balance sheet data is available
        if dat.balance_sheet is not None and not dat.balance_sheet.empty:
            # Return the first 20 rows of the balance sheet
            return dat.balance_sheet.head(20)
        else:
            print(f"No balance sheet data available for {ticker}.")
            return None
    except Exception as e:
        print(f"An error occurred while fetching data for {ticker}: {e}")
        return None

#%%
ticker = "MSFT"
balance_sheet = fetch_balance_sheet(ticker)
if balance_sheet is not None:
    print(balance_sheet)

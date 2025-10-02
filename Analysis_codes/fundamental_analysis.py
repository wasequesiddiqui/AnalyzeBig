#%%
import pandas as pd
import yfinance as yf

#%%
def get_balance_sheet(ticker):
    """
    Fetches and returns the balance sheet for a given stock ticker as a pandas DataFrame.

    Parameters:
        ticker (str): The stock ticker symbol (e.g., 'AAPL', 'MSFT').

    Returns:
        pd.DataFrame: A DataFrame containing the balance sheet data.
    """
    # Create a Ticker object for the given stock ticker
    stock = yf.Ticker(ticker)

    # Fetch the balance sheet data
    balance_sheet = stock.balance_sheet

    # Check if data is available
    if balance_sheet.empty:
        raise ValueError(f"No balance sheet data found for ticker: {ticker}")

    # Transpose the DataFrame for better readability (optional)
    # balance_sheet = balance_sheet.T

    return balance_sheet

#%%
def get_cash_flow(ticker):
    """
    Fetches the cash flow statement for a given stock ticker.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
    pd.DataFrame: A DataFrame containing the cash flow statement.
    """
    # Create a Ticker object for the given ticker symbol
    stock = yf.Ticker(ticker)
    
    # Fetch the cash flow statement
    cash_flow = stock.cashflow
    
    # Return the cash flow DataFrame
    return cash_flow

def get_income_statement_dataframe(ticker_symbol, quarterly=False):
    """
    Retrieves the income statement for a given stock ticker.

    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)
    quarterly (bool): If True, retrieves quarterly income statement data instead of annual.

    Returns:
    pandas.DataFrame: A DataFrame containing the income statement data.
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker_symbol)
        
        # Fetch the income statement data
        if quarterly:
            income_statement_df = stock.quarterly_financials
        else:
            income_statement_df = stock.financials

        if income_statement_df.empty:
            print(f"No income statement data available for ticker '{ticker_symbol}'.")
            return None

        # Transpose the DataFrame for better readability
        income_statement_df = income_statement_df.transpose()
        
        return income_statement_df

    except Exception as e:
        print(f"An error occurred while fetching data for ticker '{ticker_symbol}': {e}")
        return None

def calculate_profit_margins(income_df):
    """
    Calculates Gross Profit Margin and Net Profit Margin.

    Parameters:
    income_df (pandas.DataFrame): The income statement DataFrame.

    Returns:
    pandas.DataFrame: DataFrame containing the profit margins.
    """
    margins_df = pd.DataFrame()
    margins_df['Gross Profit Margin (%)'] = (income_df['Gross Profit'] / income_df['Total Revenue']) * 100
    margins_df['Net Profit Margin (%)'] = (income_df['Net Income'] / income_df['Total Revenue']) * 100
    return margins_df

def calculate_year_over_year_growth(income_df):
    """
    Calculates the year-over-year growth rates for key financial metrics.

    Parameters:
    income_df (pandas.DataFrame): The income statement DataFrame.

    Returns:
    pandas.DataFrame: DataFrame containing growth rates.
    """
    growth_df = income_df.pct_change() * 100
    return growth_df

def get_earnings_per_share(ticker_symbol):
    """
    Retrieves annual Earnings Per Share (EPS) data.

    Parameters:
    ticker_symbol (str): The stock ticker symbol.

    Returns:
    pandas.DataFrame: DataFrame containing EPS data.
    """
    stock = yf.Ticker(ticker_symbol)
    earnings = stock.earnings  # Annual earnings data

    if earnings.empty:
        print(f"No earnings data available for ticker '{ticker_symbol}'.")
        return None

    return earnings

def calculate_return_on_equity(income_df, balance_sheet_df):
    """
    Calculates Return on Equity (ROE).

    Parameters:
    income_df (pandas.DataFrame): Income statement DataFrame.
    balance_sheet_df (pandas.DataFrame): Balance sheet DataFrame.

    Returns:
    pandas.Series: Series containing ROE over time.
    """
    net_income = income_df['Net Income']
    total_equity = balance_sheet_df['Total Stockholder Equity']
    roe = (net_income / total_equity) * 100
    return roe


#%%
ticker = "MSFT"
balance_sheet = get_balance_sheet(ticker)
if balance_sheet is not None:
    print(balance_sheet)

# %%

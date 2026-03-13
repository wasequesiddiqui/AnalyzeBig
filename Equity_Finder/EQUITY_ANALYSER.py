import yfinance as yf
import pandas as pd

def get_cashflow_dataset(ticker_symbol):
    """
    Returns the cashflow dataset for a given ticker symbol using yfinance.

    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)

    Returns:
    pandas.DataFrame: The cashflow data for the ticker
    """
    ticker = yf.Ticker(ticker_symbol)
    cashflow = ticker.cashflow
    return cashflow.iloc[:, :4].fillna(0)

def get_balance_sheet_dataset(ticker_symbol):
    """
    Returns the balance sheet dataset for a given ticker symbol using yfinance.

    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)

    Returns:
    pandas.DataFrame: The balance sheet data for the ticker
    """
    ticker = yf.Ticker(ticker_symbol)
    balance_sheet = ticker.balance_sheet
    return balance_sheet.iloc[:, :4].fillna(0)

def get_income_statement_dataset(ticker_symbol):
    """
    Returns the income statement dataset for a given ticker symbol using yfinance.

    Parameters:
    ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL' for Apple Inc.)

    Returns:
    pandas.DataFrame: The income statement data for the ticker
    """
    ticker = yf.Ticker(ticker_symbol)
    income_statement = ticker.financials
    return income_statement.iloc[:, :4].fillna(0)

def calculate_financial_score(dataset, row_labels):
    """
    Calculates a percentage score based on year-over-year growth for specified row labels in the dataset.

    For each row label, compares the values across consecutive columns (time periods).
    If the value in the earlier period > value in the later period, scores 1, else 0.
    Sums the scores across all labels and comparisons, then returns the percentage.

    Parameters:
    dataset (pandas.DataFrame): The financial dataset (e.g., income statement)
    row_labels (list): List of row index labels to analyze (e.g., ['EBITDA'])

    Returns:
    float: Percentage score (0-100)
    """
    score = 0
    max_score = len(row_labels) * 3
    
    for label in row_labels:
        if label in dataset.index:
            row_values = dataset.loc[label]
            # Compare consecutive columns (assuming columns are in chronological order)
            for i in range(len(row_values) - 1):
                if row_values.iloc[i] > row_values.iloc[i + 1]:
                    score += 1
    
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    return score, max_score, percentage

def calculate_financial_score_decline(dataset, row_labels):
    """
    Calculates a percentage score based on year-over-year decline for specified row labels in the dataset.

    For each row label, evaluates only if at least one value is non-zero.
    If all values in a row are zero, that row receives a perfect score.
    Otherwise, compares values across consecutive columns (time periods).
    If the value in the earlier period <= value in the later period, scores 1, else 0.
    Sums the scores across all labels and comparisons, then returns the percentage.

    Parameters:
    dataset (pandas.DataFrame): The financial dataset (e.g., income statement)
    row_labels (list): List of row index labels to analyze (e.g., ['EBITDA'])

    Returns:
    tuple: (score, max_score, percentage)
    """
    score = 0
    max_score = len(row_labels) * 3
    
    for label in row_labels:
        if label in dataset.index:
            row_values = dataset.loc[label]
            
            # Check if all values are zero
            if (row_values == 0).all():
                # If all values are zero, give perfect score for this label
                score += 3
            else:
                # Otherwise, evaluate normally
                # Compare consecutive columns (assuming columns are in chronological order)
                for i in range(len(row_values) - 1):
                    if row_values.iloc[i] <= row_values.iloc[i + 1]:
                        score += 1
    
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    return score, max_score, percentage

def calculate_financial_growth_score(dataset, row_labels):
    """
    Calculates a percentage score based on year-over-year growth for specified row labels in the dataset.

    For each row label, compares the values across consecutive columns (time periods).
    If the value in the earlier period < value in the later period, scores 1, else 0.
    Sums the scores across all labels and comparisons, then returns the percentage.

    Parameters:
    dataset (pandas.DataFrame): The financial dataset (e.g., income statement)
    row_labels (list): List of row index labels to analyze (e.g., ['EBITDA'])

    Returns:
    float: Percentage score (0-100)
    """
    score = 0
    max_score = len(row_labels) * 2 * (dataset.shape[1] - 1)  # 2 points for each growth comparison per label
    
    for label in row_labels:
        if label in dataset.index:
            row_values = dataset.loc[label]
            # Compare consecutive columns (assuming columns are in chronological order)
            for i in range(len(row_values) - 2):
                if row_values.iloc[i] < row_values.iloc[i + 1]:
                    score += 2
    
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    return score, max_score, percentage

def calculate_financial_degrowth_score(dataset, row_labels):
    """
    Calculates a percentage score based on year-over-year degrowth for specified row labels in the dataset.

    For each row label, evaluates only if at least one value is non-zero.
    If all values in a row are zero, that row receives a perfect score (100%).
    Otherwise, compares values across consecutive columns (time periods).
    If the value in the earlier period >= value in the later period, scores 2, else 0.
    Sums the scores across all labels and comparisons, then returns the percentage.

    Parameters:
    dataset (pandas.DataFrame): The financial dataset (e.g., income statement)
    row_labels (list): List of row index labels to analyze (e.g., ['EBITDA'])

    Returns:
    tuple: (score, max_score, percentage)
    """
    score = 0
    max_score = len(row_labels) * 2 * (dataset.shape[1] - 1)  # 2 points for each growth comparison per label
    
    for label in row_labels:
        if label in dataset.index:
            row_values = dataset.loc[label]
            
            # Check if all values are zero
            if (row_values == 0).all():
                # If all values are zero, give perfect score for this label
                score += 2 * (dataset.shape[1] - 1)
            else:
                # Otherwise, evaluate normally
                # Compare consecutive columns (assuming columns are in chronological order)
                for i in range(len(row_values) - 2):
                    if row_values.iloc[i] >= row_values.iloc[i + 1]:
                        score += 2
    
    percentage = (score / max_score) * 100 if max_score > 0 else 0
    return score, max_score, percentage

def year_over_year_changes(dataset):
    """
    Compute the year-over-year growth rate for every consecutive pair of
    periods in the dataset: (current / previous) - 1.

    Columns are interpreted as dates and sorted chronologically to ensure
    correct ordering even if the newest column is not first or last. If date
    parsing fails, the original column order is used. The returned DataFrame
    will have one fewer column than the input, and each column name corresponds
    to the later period in the pair.

    Example: with columns
    ["2023-03-31", "2024-03-31", "2025-03-31"] (in any order), the output
    will contain columns "2024-03-31" and "2025-03-31", representing growth
    from 2023→2024 and 2024→2025 respectively.

    Parameters:
    dataset (pandas.DataFrame): Financial dataset with time periods as columns.

    Returns:
    pandas.DataFrame: Growth rates for every available year-over-year period.
    """
    if dataset.shape[1] < 2:
        return pd.DataFrame(index=dataset.index)

    cols = list(dataset.columns)
    # attempt to parse and sort by dates for correct chronological order
    try:
        dates = pd.to_datetime(cols)
        order = dates.argsort()
    except Exception:
        order = list(range(len(cols)))

    # reorder dataset columns according to chronological order
    ordered = dataset.iloc[:, order]
    ordered_cols = ordered.columns.tolist()

    # compute growth rates for each consecutive pair
    rates = []
    result_cols = []
    for i in range(1, ordered.shape[1]):
        curr = ordered.iloc[:, i]
        prev = ordered.iloc[:, i - 1]
        rate = curr.divide(prev).subtract(1)
        rates.append(rate)
        result_cols.append(ordered_cols[i])

    if rates:
        df_rate = pd.concat(rates, axis=1)
        df_rate.columns = result_cols
        # replace NaNs (e.g. previous period zero) with 0
        df_rate = df_rate.fillna(0)
        return df_rate
    else:
        return pd.DataFrame(index=dataset.index)

def evaluate_ticker(ticker_string):
    ticker = ticker_string.upper()
    cashflow_data = get_cashflow_dataset(ticker)
    balance_sheet_data = get_balance_sheet_dataset(ticker)
    income_statement_data = get_income_statement_dataset(ticker)
    print("Cashflow Data:")
    print(cashflow_data)
    print("\nBalance Sheet Data:")
    print(balance_sheet_data)
    print("\nIncome Statement Data:")
    print(income_statement_data)
    overall_score = 0
    overall_max_score = 0
    ticker_scores = {}
    ticker_scores['Ticker_Name'] = ticker

    pnl_score, pnl_max_score, pnl_percentage = calculate_financial_score(income_statement_data, ['EBITDA'
                                                                                ,'EBIT'
                                                                                ,'Basic EPS'
                                                                                , 'Net Income'
                                                                                , 'Total Revenue'
                                                                                , 'Operating Income'])
    print(f"\nFinancial Score (P&L): {pnl_score}/{pnl_max_score} ({pnl_percentage:.2f}%)")


    cf_score, cf_max_score, cf_percentage = calculate_financial_score(cashflow_data, ['Free Cash Flow'
                                                                                ,'Changes In Cash'
                                                                                , 'Net Income From Continuing Operations'
                                                                                , 'Operating Gains Losses'])
    print(f"\nFinancial Score (CF): {cf_score}/{cf_max_score} ({cf_percentage:.2f}%)")

    bs_score, bs_max_score, bs_percentage = calculate_financial_score(balance_sheet_data, ['Stockholders Equity'
                                                                                ,'Retained Earnings'
                                                                                , 'Income Tax Payable'
                                                                                ,'Cash And Cash Equivalents'
                                                                                , 'Total Assets'])
    print(f"\nFinancial Score (BS): {bs_score}/{bs_max_score} ({bs_percentage:.2f}%)")

    # Overall score can be calculated as a weighted average of the three scores
    growth_overall_score = (bs_score + cf_score + pnl_score)
    growth_max_score = (bs_max_score + cf_max_score + pnl_max_score)
    growth_percentage = (growth_overall_score / growth_max_score) * 100 if growth_max_score > 0 else 0
    overall_score += growth_overall_score
    overall_max_score += growth_max_score
    print(f"\nOverall Financial Growth Score: {growth_overall_score}/{growth_max_score} ({growth_percentage:.2f}%)")

    ticker_scores["PnL_Growth"] = pnl_percentage
    ticker_scores["CF_Growth"] = cf_percentage
    ticker_scores["BS_Growth"] = bs_percentage
    ticker_scores["Overall_Growth"] = growth_percentage

    cashflow_changes = year_over_year_changes(cashflow_data)
    balance_sheet_changes = year_over_year_changes(balance_sheet_data)
    income_statement_changes = year_over_year_changes(income_statement_data)
    print("\nYear-over-Year Changes in Cashflow Data:")
    print(cashflow_changes)
    print("\nYear-over-Year Changes in Balance Sheet Data:")
    print(balance_sheet_changes)
    print("\nYear-over-Year Changes in Income Statement Data:")
    print(income_statement_changes)

    pnl_score_growth, pnl_max_score_growth, pnl_percentage_growth = calculate_financial_growth_score(income_statement_changes, ['EBITDA'
                                                                                ,'EBIT'
                                                                                ,'Basic EPS'
                                                                                , 'Net Income'
                                                                                , 'Total Revenue'
                                                                                , 'Operating Income'])
    print(f"\nFinancial Score (P&L): {pnl_score_growth}/{pnl_max_score_growth} ({pnl_percentage_growth:.2f}%)")
    cf_score_growth, cf_max_score_growth, cf_percentage_growth = calculate_financial_growth_score(cashflow_changes, ['Free Cash Flow'
                                                                                ,'Changes In Cash'
                                                                                , 'Net Income From Continuing Operations'
                                                                                , 'Operating Gains Losses'])
    print(f"\nFinancial Score (CF): {cf_score_growth}/{cf_max_score_growth} ({cf_percentage_growth:.2f}%)")
    bs_score_growth, bs_max_score_growth, bs_percentage_growth = calculate_financial_growth_score(balance_sheet_changes, ['Stockholders Equity'
                                                                                ,'Retained Earnings'
                                                                                , 'Income Tax Payable'
                                                                                ,'Cash And Cash Equivalents'
                                                                                , 'Total Assets'])
    print(f"\nFinancial Score (BS): {bs_score_growth}/{bs_max_score_growth} ({bs_percentage_growth:.2f}%)")

    # Overall growth score can be calculated as a weighted average of the three growth scores
    growth_overall_score_growth = (bs_score_growth + cf_score_growth + pnl_score_growth)
    growth_max_score_growth = (bs_max_score_growth + cf_max_score_growth + pnl_max_score_growth)
    growth_percentage_growth = (growth_overall_score_growth / growth_max_score_growth) * 100 if growth_max_score_growth > 0 else 0
    overall_score += growth_overall_score_growth
    overall_max_score += growth_max_score_growth
    print(f"\nOverall Financial Growth Score: {growth_overall_score_growth}/{growth_max_score_growth} ({growth_percentage_growth:.2f}%)")

    ticker_scores["PnL_Growth_Velocity"] =  pnl_percentage_growth
    ticker_scores["CF_Growth_Velocity"] = cf_percentage_growth
    ticker_scores["BS_Growth_Velocity"] = bs_percentage_growth
    ticker_scores["Overall_Growth_Velocity"] = growth_percentage_growth


    pnl_score, pnl_max_score, pnl_percentage = calculate_financial_score_decline(income_statement_data, ['Interest Expense'
                                                                                ,'Other Income Expense'
                                                                                ,'Operating Expense'
                                                                                , 'Selling General And Administration'
                                                                                , 'Cost Of Revenue'
                                                                                , 'Reconciled Depreciation'])
    print(f"\nFinancial Score (P&L): {pnl_score}/{pnl_max_score} ({pnl_percentage:.2f}%)")

    cf_score, cf_max_score, cf_percentage = calculate_financial_score_decline(cashflow_data, ['Sale Of Investment'
                                                                                ,'Issuance Of Debt'
                                                                                , 'Capital Expenditure'
                                                                                ,'Net Short Term Debt Issuance'
                                                                                ,'Net Long Term Debt Issuance'
                                                                                ,'Depreciation And Amortization'
                                                                                , 'Stock Based Compensation'])
    print(f"\nFinancial Score (CF): {cf_score}/{cf_max_score} ({cf_percentage:.2f}%)")

    bs_score, bs_max_score, bs_percentage = calculate_financial_score_decline(balance_sheet_data, ['Net Debt'
                                                                                ,'Total Debt'
                                                                                ,'Capital Lease Obligations'
                                                                                ,'Current Accrued Expenses'
                                                                                , 'Inventory'])
    print(f"\nFinancial Score (BS): {bs_score}/{bs_max_score} ({bs_percentage:.2f}%)")

    # Overall score can be calculated as a weighted average of the three scores
    growth_overall_score = (bs_score + cf_score + pnl_score)
    growth_max_score = (bs_max_score + cf_max_score + pnl_max_score)
    growth_percentage = (growth_overall_score / growth_max_score) * 100 if growth_max_score > 0 else 0
    overall_score += (growth_overall_score/2)  # weight degrowth score at 50% of growth score
    overall_max_score += (growth_max_score/2)   # weight degrowth score at 50% of growth score
    print(f"\nOverall Financial Growth Score: {growth_overall_score}/{growth_max_score} ({growth_percentage:.2f}%)")

    ticker_scores["PnL_Degrowth"] =  pnl_percentage
    ticker_scores["CF_Degrowth"] = cf_percentage
    ticker_scores["BS_Degrowth"] = bs_percentage
    ticker_scores["Overall_Degrowth"] = growth_percentage

    pnl_score_growth, pnl_max_score_growth, pnl_percentage_growth = calculate_financial_degrowth_score(income_statement_changes, ['Interest Expense'
                                                                                ,'Other Income Expense'
                                                                                ,'Operating Expense'
                                                                                , 'Selling General And Administration'
                                                                                , 'Cost Of Revenue'
                                                                                , 'Reconciled Depreciation'])
    print(f"\nFinancial Score (P&L): {pnl_score_growth}/{pnl_max_score_growth} ({pnl_percentage_growth:.2f}%)")
    cf_score_growth, cf_max_score_growth, cf_percentage_growth = calculate_financial_degrowth_score(cashflow_changes, ['Sale Of Investment'
                                                                                ,'Issuance Of Debt'
                                                                                , 'Capital Expenditure'
                                                                                ,'Net Short Term Debt Issuance'
                                                                                ,'Net Long Term Debt Issuance'
                                                                                ,'Depreciation And Amortization'
                                                                                , 'Stock Based Compensation'])
    print(f"\nFinancial Score (CF): {cf_score_growth}/{cf_max_score_growth} ({cf_percentage_growth:.2f}%)")
    bs_score_growth, bs_max_score_growth, bs_percentage_growth = calculate_financial_degrowth_score(balance_sheet_changes, ['Net Debt'
                                                                                ,'Total Debt'
                                                                                ,'Capital Lease Obligations'
                                                                                ,'Current Accrued Expenses'
                                                                                , 'Inventory'])
    print(f"\nFinancial Score (BS): {bs_score_growth}/{bs_max_score_growth} ({bs_percentage_growth:.2f}%)")

    # Overall growth score can be calculated as a weighted average of the three growth scores
    growth_overall_score_growth = (bs_score_growth + cf_score_growth + pnl_score_growth)
    growth_max_score_growth = (bs_max_score_growth + cf_max_score_growth + pnl_max_score_growth)
    growth_percentage_growth = (growth_overall_score_growth / growth_max_score_growth) * 100 if growth_max_score_growth > 0 else 0
    overall_score += (growth_overall_score_growth/2)  # weight degrowth score at 50% of growth score
    overall_max_score += (growth_max_score_growth/2)   # weight degrowth score at 50% of growth score
    overall_percentage = (overall_score / overall_max_score) * 100 if overall_max_score > 0 else 0
    print(f"\nOverall Financial Growth Score: {growth_overall_score_growth}/{growth_max_score_growth} ({growth_percentage_growth:.2f}%)")

    ticker_scores["PnL_Degrowth_Velocity"] =  pnl_percentage_growth
    ticker_scores["CF_Degrowth_Velocity"] = cf_percentage_growth
    ticker_scores["BS_Degrowth_Velocity"] = bs_percentage_growth
    ticker_scores["Overall_Degrowth_Velocity"] = growth_percentage_growth
    ticker_scores["Final_Overall_Score"] = overall_percentage

    df_scores = pd.DataFrame([ticker_scores])
    print("\nTicker Scores:")
    print(df_scores)
    return df_scores
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, Optional

# ------------------------------------------------------------
# Helper functions to robustly extract financial data
# ------------------------------------------------------------
def _get_financial_value(df: pd.DataFrame, possible_names: list, year_idx: int = 0) -> float:
    """Return the numeric value from a dataframe for the first matching column name."""
    for name in possible_names:
        if name in df.columns:
            val = df[name].iloc[year_idx]
            if pd.notna(val):
                return float(val)
    return np.nan

# ------------------------------------------------------------
# Main valuation function
# ------------------------------------------------------------
def super_valuation(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Computes the min, max, and average (mid‑range) fair price per share using
    six DCF‑based models (FCFF, FCFE, DDM, APV, Residual Income, Gordon Growth).
    
    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL').
    
    Returns
    -------
    tuple (min_price, max_price, avg_price)
        min_price, max_price : float or None
            The minimum and maximum of the six model estimates.
        avg_price : float or None
            Simple average of min_price and max_price.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fin = stock.financials.T  # rows: years, columns: line items
        bs = stock.balance_sheet.T
        cf = stock.cashflow.T

        # Get the two most recent annual periods (index 0 = latest, index 1 = previous)
        if fin.empty or bs.empty or cf.empty:
            raise ValueError("Insufficient financial data.")

        # --------------------------------------------------------
        # Extract key base‑year (year 0) and previous‑year items
        # --------------------------------------------------------
        # Income statement
        revenue = _get_financial_value(fin, ['Total Revenue', 'Revenue', 'Operating Revenue'], 0)
        ebit = _get_financial_value(fin, ['EBIT', 'Ebit', 'Operating Income', 'Operating Income or Loss'], 0)
        net_income = _get_financial_value(fin, ['Net Income', 'Net Income Common Stockholders'], 0)
        interest_expense = _get_financial_value(fin, ['Interest Expense', 'Interest Expense Non Operating'], 0)
        tax_expense = _get_financial_value(fin, ['Tax Provision', 'Income Tax Expense'], 0)
        depreciation = _get_financial_value(fin, ['Depreciation & Amortization', 'Depreciation And Amortization',
                                                  'Depreciation', 'Reconciled Depreciation'], 0)

        # Previous year revenue for growth calculation
        revenue_prev = _get_financial_value(fin, ['Total Revenue', 'Revenue', 'Operating Revenue'], 1)

        # Balance sheet (year 0)
        total_assets = _get_financial_value(bs, ['Total Assets'], 0)
        cash = _get_financial_value(bs, ['Cash And Cash Equivalents', 'Cash', 'Cash & Equivalents'], 0)
        short_term_debt = _get_financial_value(bs, ['Short Term Debt', 'Current Debt'], 0)
        long_term_debt = _get_financial_value(bs, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'], 0)
        total_debt = short_term_debt + long_term_debt
        total_equity = _get_financial_value(bs, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'], 0)

        # Previous year equity for residual income model
        total_equity_prev = _get_financial_value(bs, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'], 1)
        if np.isnan(total_equity_prev) or total_equity_prev <= 0:
            total_equity_prev = total_equity - (net_income - _get_financial_value(cf, ['Dividends Paid'], 0))  # rough estimate

        # Current assets / liabilities for NWC
        current_assets = _get_financial_value(bs, ['Current Assets'], 0)
        current_liab = _get_financial_value(bs, ['Current Liabilities'], 0)

        # Cash flow statement
        capex = _get_financial_value(cf, ['Capital Expenditure', 'Capital Expenditures'], 0) * -1  # convert to positive number
        dividends_paid = _get_financial_value(cf, ['Dividends Paid'], 0)
        if np.isnan(dividends_paid):
            dividends_paid = 0

        # Market data
        shares_out = info.get('sharesOutstanding', None)
        if shares_out is None or shares_out == 0:
            # fallback: use last year's basic shares from income statement
            shares_out = _get_financial_value(fin, ['Basic Average Shares', 'Diluted Average Shares'], 0)
            if np.isnan(shares_out):
                raise ValueError("Shares outstanding not available.")

        beta = info.get('beta', 1.0)
        if beta is None or beta <= 0:
            beta = 1.0

        # --------------------------------------------------------
        # Validate critical fields
        # --------------------------------------------------------
        required = [revenue, ebit, net_income, total_equity]
        if any(np.isnan(x) for x in required) or revenue <= 0 or ebit <= 0:
            raise ValueError("Missing or invalid critical financial data.")

        # --------------------------------------------------------
        # Cost of capital parameters
        # --------------------------------------------------------
        risk_free = 0.04              # 4% US 10‑year (can be adjusted)
        market_premium = 0.05         # 5% equity risk premium
        cost_of_equity = risk_free + beta * market_premium

        # Cost of debt
        if total_debt > 0 and interest_expense > 0:
            cost_of_debt = interest_expense / total_debt
        else:
            cost_of_debt = 0.04       # assume 4% if no debt

        # Tax rate
        if ebit > 0 and tax_expense > 0:
            tax_rate = tax_expense / (net_income + tax_expense)  # approximate effective rate
        else:
            tax_rate = 0.21           # default US corporate rate

        # Capital structure weights (book values to avoid circularity)
        V_book = total_equity + total_debt
        w_eq = total_equity / V_book if V_book > 0 else 1.0
        w_debt = 1 - w_eq
        wacc = w_eq * cost_of_equity + w_debt * cost_of_debt * (1 - tax_rate)

        # Unlevered cost of equity (for APV)
        d_over_e = total_debt / total_equity if total_equity > 0 else 0
        beta_unlevered = beta / (1 + (1 - tax_rate) * d_over_e) if d_over_e > 0 else beta
        unlevered_cost = risk_free + beta_unlevered * market_premium

        # Terminal growth rate (capped at risk‑free rate, minimum 2%)
        g_term = min(risk_free, 0.025)
        if g_term <= 0:
            g_term = 0.02

        # --------------------------------------------------------
        # Historical ratios and growth assumptions
        # --------------------------------------------------------
        if not np.isnan(revenue_prev) and revenue_prev > 0:
            hist_growth = (revenue / revenue_prev) - 1
        else:
            hist_growth = 0.05        # default 5%

        # Cap historical growth between 0% and 20% for sanity
        hist_growth = max(0.0, min(0.20, hist_growth))

        # Forecast growth path: start with historical, linearly decline to terminal over 5 years
        forecast_years = 5
        growth_rates = [hist_growth * (1 - i/(forecast_years+1)) + g_term * (i/(forecast_years+1))
                        for i in range(1, forecast_years+1)]

        # Margin and reinvestment ratios
        ebit_margin = ebit / revenue
        depr_ratio = depreciation / revenue if not np.isnan(depreciation) else 0.03
        capex_ratio = capex / revenue if not np.isnan(capex) else 0.03

        # NWC (working capital excluding cash and short‑term debt)
        nwc = current_assets - cash - (current_liab - short_term_debt) if not np.isnan(current_assets) and not np.isnan(current_liab) else 0
        nwc_ratio = nwc / revenue if revenue > 0 else 0

        # Payout ratio for dividends
        payout_ratio = dividends_paid / net_income if net_income > 0 else 0
        if payout_ratio > 1:
            payout_ratio = 1.0   # cap

        # --------------------------------------------------------
        # Base year (year 0) cash flows
        # --------------------------------------------------------
        # FCFF = NOPAT + Depreciation - CapEx - ΔNWC
        nopat = ebit * (1 - tax_rate)
        # For base year, we can't know ΔNWC without previous year's NWC. Use smoothed assumption.
        # Use change in NWC from previous year if possible, else assume zero.
        nwc_prev = 0
        if not np.isnan(current_assets) and not np.isnan(current_liab) and bs.shape[0] > 1:
            ca_prev = _get_financial_value(bs, ['Current Assets'], 1)
            cl_prev = _get_financial_value(bs, ['Current Liabilities'], 1)
            if not np.isnan(ca_prev) and not np.isnan(cl_prev):
                cash_prev = _get_financial_value(bs, ['Cash And Cash Equivalents', 'Cash'], 1)
                std_prev = _get_financial_value(bs, ['Short Term Debt', 'Current Debt'], 1)
                nwc_prev = ca_prev - cash_prev - (cl_prev - std_prev)
        delta_nwc_base = nwc - nwc_prev if nwc_prev != 0 else 0
        fcff_0 = nopat + depreciation - capex - delta_nwc_base
        fcff_0 = max(0, fcff_0)  # avoid negative FCFF base

        # FCFE base
        net_borrowing_0 = 0
        if bs.shape[0] > 1:
            debt_prev = _get_financial_value(bs, ['Total Debt', 'Short Long Term Debt'], 1)
            if not np.isnan(debt_prev):
                net_borrowing_0 = total_debt - debt_prev
        fcfe_0 = fcff_0 - interest_expense * (1 - tax_rate) + net_borrowing_0
        fcfe_0 = max(0, fcfe_0)

        # Dividend base
        div_0 = dividends_paid

        # Book equity for residual income (beginning of year 0)
        beg_equity_0 = total_equity_prev if not np.isnan(total_equity_prev) and total_equity_prev > 0 else total_equity - net_income + dividends_paid

        # --------------------------------------------------------
        # Project financials for 5 years
        # --------------------------------------------------------
        rev = revenue
        fcff_proj = np.zeros(forecast_years)
        fcfe_proj = np.zeros(forecast_years)
        div_proj = np.zeros(forecast_years)
        ri_proj = np.zeros(forecast_years)
        equity_proj = np.zeros(forecast_years + 1)
        equity_proj[0] = beg_equity_0

        debt_level = total_debt          # assume constant debt

        for t in range(forecast_years):
            g = growth_rates[t]
            rev = rev * (1 + g)
            ebit_t = rev * ebit_margin
            nopat_t = ebit_t * (1 - tax_rate)
            dep_t = rev * depr_ratio
            capex_t = rev * capex_ratio
            nwc_t = rev * nwc_ratio
            if t == 0:
                delta_nwc = nwc_t - nwc
            else:
                delta_nwc = nwc_t - (rev_prev * nwc_ratio)
            rev_prev = rev
            fcff_t = nopat_t + dep_t - capex_t - delta_nwc
            fcff_t = max(0, fcff_t)  # floor at 0
            fcff_proj[t] = fcff_t

            # Interest and FCFE
            interest_t = debt_level * cost_of_debt
            ni_t = nopat_t - interest_t * (1 - tax_rate)
            fcfe_t = fcff_t - interest_t * (1 - tax_rate) + 0   # no net borrowing
            fcfe_proj[t] = max(0, fcfe_t)

            # Dividends
            div_t = ni_t * payout_ratio
            div_proj[t] = div_t

            # Residual Income
            equity_proj[t+1] = equity_proj[t] + ni_t - div_t   # ending book equity
            ri_t = ni_t - cost_of_equity * equity_proj[t]
            ri_proj[t] = ri_t

        # Terminal values (at end of year 5)
        # Use perpetual growth formula with final year cash flow grown at g_term
        disc_factors_wacc = [(1 + wacc) ** (t+1) for t in range(forecast_years)]
        disc_factors_ke = [(1 + cost_of_equity) ** (t+1) for t in range(forecast_years)]
        disc_factors_ku = [(1 + unlevered_cost) ** (t+1) for t in range(forecast_years)]

        # ------------------------------
        # Model 1: FCFF
        # ------------------------------
        fcff_terminal = fcff_proj[-1] * (1 + g_term) / (wacc - g_term)
        pv_fcff = np.sum(fcff_proj / disc_factors_wacc) + fcff_terminal / disc_factors_wacc[-1]
        net_debt = total_debt - cash if not np.isnan(cash) else total_debt
        equity_value_fcff = pv_fcff - net_debt
        price_fcff = equity_value_fcff / shares_out

        # ------------------------------
        # Model 2: FCFE
        # ------------------------------
        fcfe_terminal = fcfe_proj[-1] * (1 + g_term) / (cost_of_equity - g_term)
        equity_value_fcfe = np.sum(fcfe_proj / disc_factors_ke) + fcfe_terminal / disc_factors_ke[-1]
        price_fcfe = equity_value_fcfe / shares_out

        # ------------------------------
        # Model 3: DDM
        # ------------------------------
        div_terminal = div_proj[-1] * (1 + g_term) / (cost_of_equity - g_term)
        equity_value_ddm = np.sum(div_proj / disc_factors_ke) + div_terminal / disc_factors_ke[-1]
        price_ddm = equity_value_ddm / shares_out if equity_value_ddm > 0 else 0

        # ------------------------------
        # Model 4: APV
        # ------------------------------
        # Unlevered value of firm
        fcff_terminal_ku = fcff_proj[-1] * (1 + g_term) / (unlevered_cost - g_term)
        unlevered_firm = np.sum(fcff_proj / disc_factors_ku) + fcff_terminal_ku / disc_factors_ku[-1]
        # Tax shield – perpetual fixed debt
        tax_shield = debt_level * cost_of_debt * tax_rate / cost_of_debt   # simplifies to debt * tax_rate
        apv_firm = unlevered_firm + debt_level * tax_rate
        equity_value_apv = apv_firm - net_debt
        price_apv = equity_value_apv / shares_out

        # ------------------------------
        # Model 5: Residual Income
        # ------------------------------
        # Terminal RI: assume fades to zero (no excess returns after forecast)
        ri_terminal = 0
        equity_value_ri = beg_equity_0 + np.sum(ri_proj / disc_factors_ke) + ri_terminal
        price_ri = equity_value_ri / shares_out

        # ------------------------------
        # Model 6: Gordon Growth (simple perpetuity on base FCFE)
        # ------------------------------
        if fcfe_0 > 0:
            gordon_equity = (fcfe_0 * (1 + g_term)) / (cost_of_equity - g_term)
            price_gordon = gordon_equity / shares_out
        else:
            price_gordon = 0

        # --------------------------------------------------------
        # Assemble all valid per‑share estimates
        # --------------------------------------------------------
        all_prices = [price_fcff, price_fcfe, price_ddm, price_apv, price_ri, price_gordon]
        valid_prices = [p for p in all_prices if not np.isnan(p) and p > 0]

        if len(valid_prices) == 0:
            raise ValueError("All valuation models produced invalid results.")

        min_price = min(valid_prices)
        max_price = max(valid_prices)
        avg_price = (min_price + max_price) / 2

        return min_price, max_price, avg_price

    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None, None

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    min_p, max_p, avg_p = super_valuation("AAPL")
    if min_p is not None:
        print(f"AAPL valuation range: ${min_p:.2f} – ${max_p:.2f} | Midpoint: ${avg_p:.2f}")
    else:
        print("Valuation failed.")
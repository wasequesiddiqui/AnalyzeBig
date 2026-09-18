import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# -------------------------------------------------------------------
# Helper: safe extraction from financial DataFrames
# -------------------------------------------------------------------
def _get_fin(df: pd.DataFrame, names: list, idx: int = 0) -> float:
    for name in names:
        if name in df.columns:
            val = df[name].iloc[idx]
            if pd.notna(val):
                return float(val)
    return np.nan

# -------------------------------------------------------------------
# Main function: persona‑refined super valuation
# -------------------------------------------------------------------
def _persona_prices(ticker: str) -> List[Tuple[str, float]]:
    """Extract the four rule-based per-share persona values for a ticker.

    Internal engine shared by persona_super_valuation() and persona_valuations().
    It runs the four rule-based models (Buffett owner earnings, Munger residual
    income, Jhunjhunwala three-stage FCFF, Damodaran scenario DCF) and returns
    each named result in the company's REPORTING currency.

    A rule is only included when its per-share value came out positive. That is
    the pre-existing `if val > 0` filter, preserved deliberately so the range and
    the per-persona values can never disagree.

    Failures are logged, never raised: on any error (including "all four rules
    failed") this returns an empty list, so both public wrappers degrade cleanly.

    Returns:
        List[Tuple[str, float]]: Named values, e.g.
            [('Buffett', 1280.0), ('Munger', 1250.0)], in reporting currency.
            Empty list when nothing could be computed.

    Example:
        Reached through the public wrappers rather than called directly::

            persona_valuations("AAPL")        # -> {'Buffett': 187.4, ...}
            persona_super_valuation("AAPL")   # -> (min, max, midpoint)
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fin = stock.financials.T
        bs = stock.balance_sheet.T
        cf = stock.cashflow.T

        if fin.empty or bs.empty:
            raise ValueError("No financial statements available.")

        # --- Basic data extraction ---------------------------------
        rev = _get_fin(fin, ['Total Revenue', 'Revenue', 'Operating Revenue'])
        ebit = _get_fin(fin, ['EBIT', 'Ebit', 'Operating Income', 'Operating Income or Loss'])
        ni = _get_fin(fin, ['Net Income', 'Net Income Common Stockholders'])
        int_exp = _get_fin(fin, ['Interest Expense', 'Interest Expense Non Operating'], 0)
        tax = _get_fin(fin, ['Tax Provision', 'Income Tax Expense'])
        # D&A lives on the INCOME statement as 'Reconciled Depreciation' for most
        # yfinance listings and on the CASHFLOW statement as 'Depreciation And
        # Amortization', so check both and then fall back to zero. Without this,
        # depr is NaN and the Buffett / Jhunjhunwala / Damodaran rules are all
        # silently dropped (Munger is the only rule that never uses depr).
        depr = _get_fin(fin, ['Depreciation & Amortization', 'Depreciation And Amortization',
                              'Depreciation', 'Reconciled Depreciation'])
        if pd.isna(depr):
            depr = _get_fin(cf, ['Depreciation And Amortization',
                                 'Depreciation Amortization Depletion', 'Depreciation'])
        if pd.isna(depr):
            depr = 0.0

        # Yahoo signs capex negative; normalise it to a positive spend and treat a
        # missing row as zero instead of NaN.
        capex_raw_signed = _get_fin(cf, ['Capital Expenditure', 'Capital Expenditures'])
        capex = abs(capex_raw_signed) if not pd.isna(capex_raw_signed) else 0.0
        div_paid = _get_fin(cf, ['Dividends Paid'])
        shares = info.get('sharesOutstanding')
        if not shares:
            shares = _get_fin(fin, ['Basic Average Shares', 'Diluted Average Shares'])
        beta = info.get('beta', 1.0) or 1.0

        # Balance sheet
        tot_assets = _get_fin(bs, ['Total Assets'])
        cash = _get_fin(bs, ['Cash And Cash Equivalents', 'Cash'])
        # Debt label sets differ a lot by listing (Indian filings have no 'Short
        # Term Debt' row at all), so widen the candidates and treat a missing row
        # as zero. A NaN here propagates through nwc and removes three rules.
        std = _get_fin(bs, ['Short Term Debt', 'Current Debt',
                            'Current Debt And Capital Lease Obligation',
                            'Current Debt And Lease Obligation', 'Other Current Borrowings'])
        if pd.isna(std):
            std = 0.0
        ltd = _get_fin(bs, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation',
                            'Long Term Debt And Lease Obligation'])
        if pd.isna(ltd):
            ltd = 0.0
        tot_debt = std + ltd
        equity = _get_fin(bs, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'])
        nwc = (_get_fin(bs, ['Current Assets']) - cash) - (_get_fin(bs, ['Current Liabilities']) - std)
        if pd.isna(nwc):
            nwc = 0.0
        # previous equity for residual income
        equity_prev = _get_fin(bs, ['Total Stockholder Equity', 'Total Equity Gross Minority Interest'], 1)
        if pd.isna(equity_prev) or equity_prev <= 0:
            equity_prev = equity - ni + div_paid if not pd.isna(div_paid) else equity - ni

        # Ratios
        def _ratio(num: float, den: float, default: float) -> float:
            """Return num/den, or `default` when an input is missing or den is 0.

            yfinance omits whole rows for many listings, and an unguarded NaN
            here propagates into a persona rule and silently removes it from the
            range — which is how this model ended up with only one survivor.
            """
            if pd.isna(num) or pd.isna(den) or not den:
                return default
            return num / den

        tax_rate = tax / (ni + tax) if (ni + tax) > 0 else 0.21
        ebit_margin = _ratio(ebit, rev, 0.15)
        depr_ratio = _ratio(depr, rev, 0.03)
        capex_ratio = _ratio(capex, rev, 0.03)
        nwc_ratio = _ratio(nwc, rev, 0.1)

        # Cost of capital
        rf = 0.04                  # base US risk‑free
        mrp = 0.05                 # equity risk premium
        # adjust for emerging markets if applicable
        country = info.get('country', '')
        if 'india' in country.lower():
            rf = 0.065
            mrp = 0.06
        ke = rf + beta * mrp
        kd = int_exp / tot_debt if tot_debt > 0 and int_exp > 0 else 0.045
        wacc = (equity / (equity + tot_debt)) * ke + (tot_debt / (equity + tot_debt)) * kd * (1 - tax_rate) if (equity + tot_debt) > 0 else ke
        unlevered_beta = beta / (1 + (1 - tax_rate) * (tot_debt / equity)) if equity > 0 else beta
        ku = rf + unlevered_beta * mrp

        # Growth and returns
        rev_growth = info.get('revenueGrowth', 0.05)  # yfinance provides latest quarterly Y/Y, use cautiously
        if pd.isna(rev_growth) or rev_growth < -0.5:
            rev_growth = 0.05
        rev_growth = max(0.0, min(0.30, rev_growth))  # cap 0‑30%
        roe = _ratio(ni, equity_prev, 0.15)
        roic = _ratio(ebit * (1 - tax_rate), tot_assets - cash + nwc, 0.10)  # rough
        g_term = min(rf, 0.025)

        # --- Persona Confidence Index (PCI) factors ---------------
        # Buffett
        moat_score = 1.0 if (ebit_margin > 0.20 and roe > 0.15 and tot_debt / equity < 0.5) else 0.6
        fcf_stability = 0.8   # simplified
        mgmt_quality = 0.9 if roe > 0.12 else 0.7
        low_leverage = 1.0 if (tot_debt / equity) < 0.5 else 0.5
        buffett_pci = moat_score * fcf_stability * mgmt_quality * low_leverage

        # Munger
        roic_premium = max(0, (roic - wacc) / wacc)  # scaled
        moat_dur_proxy = min(1.0, (roic - wacc) * 3) if roic > wacc else 0.1
        asset_light = 1.0 if depr_ratio < 0.05 else 0.6
        pessimism = 0.8  # hard to gauge
        munger_pci = roic_premium * moat_dur_proxy * asset_light * pessimism
        munger_pci = max(0.1, munger_pci)

        # Jhunjhunwala
        high_growth = 1.0 if rev_growth > 0.20 else (0.5 if rev_growth > 0.10 else 0.2)
        india_tailwind = 1.0 if 'india' in country.lower() else 0.5
        tam = 0.7   # cannot be measured
        founder = 0.8
        jhunjhun_pci = high_growth * india_tailwind * tam * founder

        # Damodaran (always high for any real company)
        damodaran_pci = 0.9

        # --- Persona Valuations -----------------------------------
        forecast_years = 10
        prices = []

        # 1. Buffett (Owner Earnings, 10‑yr, MOS=25% or dynamic)
        def buffett_price():
            oe0 = ni + depr - capex*0.8 - (nwc_ratio * rev) * 0.01  # simple owner earnings
            r_buff = rf + 0.02  # moat premium
            g_buff = min(rf, 0.02)
            mos = 0.25
            # 10‑year explicit
            pv_oe = 0.0
            oe = oe0
            for t in range(1, 11):
                oe *= (1 + g_buff)
                pv_oe += oe / (1 + r_buff)**t
            terminal = oe * (1 + g_buff) / (r_buff - g_buff) / (1 + r_buff)**10
            intrinsic = pv_oe + terminal
            return intrinsic * (1 - mos) / shares
        buff_val = buffett_price()
        if buff_val > 0:
            prices.append(('Buffett', buff_val))

        # 2. Munger (Residual Income truncated)
        def munger_price():
            r_mung = 0.12
            # moat duration: years of positive residual income
            if roic > wacc:
                moat_years = min(20, max(5, int((roic - wacc) * 10)))
            else:
                moat_years = 5
            pv_ri = 0.0
            be = equity_prev
            for t in range(1, moat_years + 1):
                # simplistic: assume NI grows with g_term, equity accumulates
                ni_t = ni * (1 + g_term)**(t-1)
                ri_t = ni_t - r_mung * be
                pv_ri += ri_t / (1 + r_mung)**t
                be = be + ni_t - (ni_t * 0.2)  # assume 20% payout
            return (equity_prev + pv_ri) / shares
        mung_val = munger_price()
        if mung_val > 0:
            prices.append(('Munger', mung_val))

        # 3. Jhunjhunwala (Three‑stage FCFF)
        def jhunjhun_price():
            stage1_years = 5
            stage2_years = 10
            g_stable = 0.07 if 'india' in country.lower() else 0.05
            g_hyper = min(0.30, rev_growth + 0.05)
            fcff0 = ebit * (1 - tax_rate) + depr - capex - (nwc_ratio * rev * 0.05)  # approximate
            fcff = fcff0
            pv = 0.0
            for t in range(1, stage1_years + 1):
                fcff *= (1 + g_hyper)
                pv += fcff / (1 + wacc)**t
            for t in range(stage1_years + 1, stage2_years + 1):
                # linear transition to g_stable
                frac = (t - stage1_years) / (stage2_years - stage1_years)
                g = g_hyper * (1 - frac) + g_stable * frac
                fcff *= (1 + g)
                pv += fcff / (1 + wacc)**t
            terminal = fcff * (1 + g_stable) / (wacc - g_stable) / (1 + wacc)**stage2_years
            ev = pv + terminal
            eq_val = ev - (tot_debt - cash)
            return eq_val / shares if eq_val > 0 else 0
        jhunj_val = jhunjhun_price()
        if jhunj_val > 0:
            prices.append(('Jhunjhunwala', jhunj_val))

        # 4. Damodaran (simplified scenario‑based Monte Carlo)
        def damodaran_price():
            n_scenarios = 3
            mean_g = rev_growth
            sigma_g = 0.08
            scenarios = [
                {'g': mean_g - 0.5*sigma_g, 'marg': ebit_margin * 0.95, 'prob': 0.25},
                {'g': mean_g, 'marg': ebit_margin, 'prob': 0.50},
                {'g': mean_g + 0.5*sigma_g, 'marg': ebit_margin * 1.05, 'prob': 0.25}
            ]
            exp_value = 0.0
            for sc in scenarios:
                pv_fcff = 0.0
                fcff0 = ebit * (1 - tax_rate) + depr - capex - (nwc_ratio * rev * 0.03)
                fcff = fcff0
                for t in range(1, 11):
                    fcff *= (1 + sc['g'])
                    pv_fcff += fcff / (1 + wacc)**t
                # Terminal with ROIC fading to WACC
                roic_term = wacc + 0.02  # small residual
                rr = g_term / roic_term if roic_term > 0 else 0.5
                terminal_cf = fcff * (1 + g_term) * (1 - rr)
                tv = terminal_cf / (wacc - g_term)
                ev = pv_fcff + tv / (1 + wacc)**10
                eq = ev - (tot_debt - cash)
                exp_value += sc['prob'] * max(0, eq / shares)
            return exp_value
        damo_val = damodaran_price()
        if damo_val > 0:
            prices.append(('Damodaran', damo_val))

        if not prices:
            raise ValueError("All persona valuations failed.")

        # The named values are returned as-is: both public wrappers derive their
        # output from this single list, so the range and the per-persona figures
        # are guaranteed to come from identical numbers.
        return prices

    except Exception as e:
        print(f"Error for {ticker}: {e}")
        return []


#: Canonical persona names produced by _persona_prices(), in report order.
PERSONA_RULE_NAMES: Tuple[str, ...] = ("Buffett", "Munger", "Jhunjhunwala", "Damodaran")


def persona_super_valuation(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return the (min, max, midpoint) per-share range across the four personas.

    Thin wrapper over _persona_prices() that keeps the original public contract:
    same values, same order, and (None, None, None) when nothing could be valued.

    Note that the third element is the MIDPOINT of min and max, not the mean of
    the four persona values. That is pre-existing behaviour and is left untouched
    on purpose, since existing callers depend on it. Use persona_valuations()
    when you want the individual values instead of the range.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. "AAPL" or "INFY.NS".

    Returns:
        Tuple[Optional[float], Optional[float], Optional[float]]: (min, max, mid)
            in the company's REPORTING currency, or (None, None, None) on failure.

    Example:
        Unchanged usage, so this call site keeps working::

            min_p, max_p, mid = persona_super_valuation("AAPL")
            print(f"AAPL range: {min_p:.2f} - {max_p:.2f} | mid {mid:.2f}")
    """
    prices = _persona_prices(ticker)
    if not prices:
        return None, None, None
    vals = [p[1] for p in prices]
    min_p = min(vals)
    max_p = max(vals)
    avg_p = (min_p + max_p) / 2
    return min_p, max_p, avg_p


def persona_valuations(ticker: str) -> Dict[str, Optional[float]]:
    """Return each rule-based persona's per-share value for a ticker.

    The per-persona counterpart to persona_super_valuation(): same underlying
    computation, but the four values stay separate and labelled instead of being
    collapsed into a range. This is what lets the agentic crew be compared, and
    blended, persona by persona.

    All four keys are ALWAYS present. A key maps to None when that rule produced
    no positive value (or when the whole computation failed), so callers can count
    participants without special-casing a missing key.

    Values are in the company's REPORTING currency (USD for INFY.NS), which may
    differ from its trading currency. Convert before comparing with a market
    price.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. "AAPL" or "INFY.NS".

    Returns:
        Dict[str, Optional[float]]: Keys Buffett, Munger, Jhunjhunwala,
            Damodaran; values in reporting currency, or None. Never raises.

    Example:
        Separating the four opinions::

            >>> vals = persona_valuations("AAPL")          # doctest: +SKIP
            >>> vals["Buffett"], vals["Damodaran"]         # doctest: +SKIP
            (187.42, 203.11)

        Counting how many rules actually priced. Fewer than two is treated as
        degenerate by the UI, which then hides the comparison entirely::

            >>> vals = persona_valuations("INFY.NS")       # doctest: +SKIP
            >>> sum(1 for v in vals.values() if v is not None)   # doctest: +SKIP
            4

        A total failure yields four None values rather than an exception::

            >>> persona_valuations("NOT_A_TICKER")         # doctest: +SKIP
            {'Buffett': None, 'Munger': None, 'Jhunjhunwala': None, 'Damodaran': None}
    """
    found: Dict[str, float] = {}
    for name, value in _persona_prices(ticker):
        found[name] = float(value)
    return {name: found.get(name) for name in PERSONA_RULE_NAMES}

# -------------------------------------------------------------------
# Example
# -------------------------------------------------------------------
if __name__ == "__main__":
    min_p, max_p, mid = persona_super_valuation("AAPL")
    if min_p:
        print(f"AAPL Persona Range: ${min_p:.2f} – ${max_p:.2f} | Midpoint: ${mid:.2f}")
    else:
        print("Valuation could not be computed.")

    # Per-persona breakdown: what each rule-based model produced, and how many of
    # the four actually priced. Fewer than two priced values means the range is
    # degenerate (min == max == midpoint) and the UI hides it.
    per_persona = persona_valuations("AAPL")
    priced = sum(1 for v in per_persona.values() if v is not None)
    print(f"\nPer-persona rule-based values ({priced}/4 priced):")
    for name, value in per_persona.items():
        print(f"  {name:<14} {'-' if value is None else f'${value:,.2f}'}")
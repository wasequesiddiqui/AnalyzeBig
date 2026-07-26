"""
Discounted Cash Flow Valuation Models
=====================================
This module implements a comprehensive set of discounted cash flow models
using Python. All functions are designed to work with financial data that
can be fetched via the `yfinance` library, though the core valuation logic
is independent of the data source.

Each valuation function returns a tuple:
    (total_value, value_per_share)
- For models that value the entire firm (e.g., FCFF, APV, CCF, EVA),
  `total_value` is the enterprise value and `value_per_share` is the
  equity value per share (after subtracting net debt).
- For models that directly value equity (e.g., DDM, FCFE, RIM, ECF),
  `total_value` is the equity value and `value_per_share` is that value
  divided by the number of shares outstanding.
"""

import yfinance as yf
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import List, Tuple, Optional, Union

# -----------------------------------------------------------------------------
# 1. Dividend Discount Models (DDM)
# -----------------------------------------------------------------------------

def ddm_gordon(dividend_per_share: float,
               cost_of_equity: float,
               growth_rate: float,
               shares_outstanding: float) -> Tuple[float, float]:
    """
    Gordon Growth Model (constant perpetual dividend growth).

    Formula:
        P0 = D1 / (r - g) = D0 * (1+g) / (r - g)

    Args:
        dividend_per_share (float): Most recent annual dividend (D0).
        cost_of_equity (float): Required return on equity (r), decimal.
        growth_rate (float): Constant perpetual growth rate of dividends (g), decimal.
        shares_outstanding (float): Number of shares outstanding.

    Returns:
        tuple: (equity_value, value_per_share)
            - equity_value = intrinsic market capitalisation
            - value_per_share = equity_value / shares_outstanding
    """
    if cost_of_equity <= growth_rate:
        raise ValueError("Cost of equity must be greater than the growth rate.")

    d1 = dividend_per_share * (1 + growth_rate)
    value_per_share = d1 / (cost_of_equity - growth_rate)
    equity_value = value_per_share * shares_outstanding
    return equity_value, value_per_share


def ddm_two_stage(dividend_per_share: float,
                  growth_high: float,
                  growth_stable: float,
                  years_high: int,
                  cost_of_equity: float,
                  shares_outstanding: float) -> Tuple[float, float]:
    """
    Two-stage DDM: high-growth phase followed by stable perpetual growth.

    Formula:
        P0 = Σ_{t=1}^{n} D0*(1+g1)^t/(1+r)^t
             + [D_n * (1+g2) / (r - g2)] / (1+r)^n

    Args:
        dividend_per_share (float): D0.
        growth_high (float): High growth rate during first phase (g1).
        growth_stable (float): Stable growth rate after first phase (g2).
        years_high (int): Number of years in the high-growth phase (n).
        cost_of_equity (float): r.
        shares_outstanding (float): Number of shares.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    r = cost_of_equity
    g1, g2 = growth_high, growth_stable
    n = years_high

    if r <= g2:
        raise ValueError("Cost of equity must exceed stable growth rate.")

    # Present value of dividends during high-growth phase
    pv_high = 0.0
    d_prev = dividend_per_share
    for t in range(1, n + 1):
        d_t = d_prev * (1 + g1)
        pv_high += d_t / (1 + r) ** t
        d_prev = d_t

    # Terminal value at the end of year n (Gordon Growth from year n+1)
    terminal_value = d_prev * (1 + g2) / (r - g2)
    pv_terminal = terminal_value / (1 + r) ** n

    value_per_share = pv_high + pv_terminal
    equity_value = value_per_share * shares_outstanding
    return equity_value, value_per_share


def ddm_three_stage(dividend_per_share: float,
                    growth_high: float,
                    growth_stable: float,
                    years_high: int,
                    years_transition: int,
                    cost_of_equity: float,
                    shares_outstanding: float) -> Tuple[float, float]:
    """
    Three-stage DDM: high growth, linearly declining transition, stable growth.

    Formula:
        P0 = Σ_{t=1}^{n1} D0*(1+g1)^t/(1+r)^t
             + Σ_{t=n1+1}^{n2} D_{t-1}*(1+g_t)/(1+r)^t
             + [D_{n2}*(1+g3)/(r-g3)] / (1+r)^{n2}
        where g_t declines linearly from g1 to g3 over the transition period.

    Args:
        dividend_per_share (float): D0.
        growth_high (float): Initial high growth rate (g1).
        growth_stable (float): Final stable growth rate (g3).
        years_high (int): Duration of high-growth phase (n1).
        years_transition (int): Duration of transition phase.
        cost_of_equity (float): r.
        shares_outstanding (float): Shares outstanding.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    r = cost_of_equity
    g1, g3 = growth_high, growth_stable
    n1 = years_high
    n2 = years_transition  # transition years
    total_explicit = n1 + n2

    if r <= g3:
        raise ValueError("Cost of equity must exceed stable growth rate.")

    # High-growth phase
    pv_sum = 0.0
    d_prev = dividend_per_share
    for t in range(1, n1 + 1):
        d_t = d_prev * (1 + g1)
        pv_sum += d_t / (1 + r) ** t
        d_prev = d_t

    # Transition phase: linear decline
    # growth starts at g1 and ends at g3 after n2 steps
    g_current = g1
    step = (g1 - g3) / (n2 + 1)  # decline per period (n2 steps, but n2+1 increments including last)
    # Actually we need n2 transition years: growth rates g_1,... g_n2
    # After the last transition year, growth becomes g3 permanently.
    for i in range(1, n2 + 1):
        g_current -= step
        d_t = d_prev * (1 + g_current)
        pv_sum += d_t / (1 + r) ** (n1 + i)
        d_prev = d_t

    # Terminal value (stable growth from year n1+n2+1)
    terminal_value = d_prev * (1 + g3) / (r - g3)
    pv_terminal = terminal_value / (1 + r) ** total_explicit

    value_per_share = pv_sum + pv_terminal
    equity_value = value_per_share * shares_outstanding
    return equity_value, value_per_share


def ddm_h_model(dividend_per_share: float,
                growth_short: float,
                growth_long: float,
                half_life: float,
                cost_of_equity: float,
                shares_outstanding: float) -> Tuple[float, float]:
    """
    H-Model: growth rate declines linearly from a short-term high to a long-term stable rate.

    Formula:
        P0 = D0*(1+gL)/(r-gL) + D0*H*(gS - gL)/(r - gL)
    where H = half-life of the high-growth period (total high-growth phase = 2H years).

    Args:
        dividend_per_share (float): D0.
        growth_short (float): Short-term high growth rate (gS).
        growth_long (float): Long-term stable growth rate (gL).
        half_life (float): H (in years).
        cost_of_equity (float): r.
        shares_outstanding (float): Shares outstanding.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    r = cost_of_equity
    gs, gl = growth_short, growth_long
    H = half_life

    if r <= gl:
        raise ValueError("Cost of equity must be greater than long-term growth rate.")

    value_per_share = (dividend_per_share * (1 + gl) + dividend_per_share * H * (gs - gl)) / (r - gl)
    equity_value = value_per_share * shares_outstanding
    return equity_value, value_per_share


# -----------------------------------------------------------------------------
# 2. Free Cash Flow to Firm (FCFF) Models
# -----------------------------------------------------------------------------

def fcff_single_stage(fcff_0: float,
                      wacc: float,
                      growth_rate: float,
                      shares_outstanding: float,
                      net_debt: float) -> Tuple[float, float]:
    """
    Single-stage (constant growth) FCFF model.

    Enterprise Value = FCFF1 / (WACC - g) = FCFF0 * (1+g) / (WACC - g)
    Equity value = Enterprise Value - Net Debt
    Value per share = Equity value / Shares Outstanding

    Args:
        fcff_0 (float): Most recent FCFF.
        wacc (float): Weighted average cost of capital, decimal.
        growth_rate (float): Perpetual growth rate of FCFF, decimal.
        shares_outstanding (float): Number of shares.
        net_debt (float): Total debt minus cash & equivalents.

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    if wacc <= growth_rate:
        raise ValueError("WACC must be greater than the growth rate.")

    fcff_1 = fcff_0 * (1 + growth_rate)
    enterprise_value = fcff_1 / (wacc - growth_rate)
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


def fcff_two_stage(fcff_0: float,
                   growth_high: float,
                   growth_stable: float,
                   years_high: int,
                   wacc: float,
                   shares_outstanding: float,
                   net_debt: float) -> Tuple[float, float]:
    """
    Two-stage FCFF model.

    Explicit forecast for `years_high` periods at high growth,
    followed by a terminal value using the stable growth rate.

    Enterprise Value = Σ FCFF_t/(1+WACC)^t + TV/(1+WACC)^n

    Args:
        fcff_0 (float): Most recent FCFF.
        growth_high (float): High growth rate (g1).
        growth_stable (float): Stable perpetual growth rate (g2).
        years_high (int): Number of high-growth years (n).
        wacc (float): Weighted average cost of capital.
        shares_outstanding (float): Shares outstanding.
        net_debt (float): Net debt (debt - cash).

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    if wacc <= growth_stable:
        raise ValueError("WACC must be greater than stable growth rate.")

    n = years_high
    r = wacc
    g1, g2 = growth_high, growth_stable

    # Present value of explicit FCFFs
    pv_explicit = 0.0
    fcff_prev = fcff_0
    for t in range(1, n + 1):
        fcff_t = fcff_prev * (1 + g1)
        pv_explicit += fcff_t / (1 + r) ** t
        fcff_prev = fcff_t

    # Terminal value at end of year n
    terminal_value = fcff_prev * (1 + g2) / (r - g2)
    pv_terminal = terminal_value / (1 + r) ** n

    enterprise_value = pv_explicit + pv_terminal
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


# -----------------------------------------------------------------------------
# 3. Free Cash Flow to Equity (FCFE) Models
# -----------------------------------------------------------------------------

def fcfe_constant_growth(fcfe_0: float,
                         cost_of_equity: float,
                         growth_rate: float,
                         shares_outstanding: float) -> Tuple[float, float]:
    """
    Constant growth FCFE model (single stage).

    Equity Value = FCFE1 / (r - g) = FCFE0 * (1+g) / (r - g)

    Args:
        fcfe_0 (float): Most recent FCFE.
        cost_of_equity (float): Required return on equity (r).
        growth_rate (float): Perpetual growth rate of FCFE (g).
        shares_outstanding (float): Shares outstanding.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    if cost_of_equity <= growth_rate:
        raise ValueError("Cost of equity must exceed growth rate.")

    fcfe_1 = fcfe_0 * (1 + growth_rate)
    equity_value = fcfe_1 / (cost_of_equity - growth_rate)
    value_per_share = equity_value / shares_outstanding
    return equity_value, value_per_share


def fcfe_two_stage(fcfe_0: float,
                   growth_high: float,
                   growth_stable: float,
                   years_high: int,
                   cost_of_equity: float,
                   shares_outstanding: float) -> Tuple[float, float]:
    """
    Two-stage FCFE model.

    Args:
        fcfe_0 (float): Most recent FCFE.
        growth_high (float): High growth rate (g1).
        growth_stable (float): Stable growth rate (g2).
        years_high (int): Years in high-growth phase (n).
        cost_of_equity (float): r.
        shares_outstanding (float): Shares outstanding.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    if cost_of_equity <= growth_stable:
        raise ValueError("Cost of equity must exceed stable growth rate.")

    r = cost_of_equity
    g1, g2 = growth_high, growth_stable
    n = years_high

    pv_explicit = 0.0
    fcfe_prev = fcfe_0
    for t in range(1, n + 1):
        fcfe_t = fcfe_prev * (1 + g1)
        pv_explicit += fcfe_t / (1 + r) ** t
        fcfe_prev = fcfe_t

    terminal_value = fcfe_prev * (1 + g2) / (r - g2)
    pv_terminal = terminal_value / (1 + r) ** n

    equity_value = pv_explicit + pv_terminal
    value_per_share = equity_value / shares_outstanding
    return equity_value, value_per_share


# -----------------------------------------------------------------------------
# 4. Adjusted Present Value (APV) Model
# -----------------------------------------------------------------------------

def apv(unlevered_fcff: List[float],
        unlevered_cost_of_equity: float,
        interest_tax_shields: List[float],
        discount_rate_tax_shields: float,
        net_debt: float,
        shares_outstanding: float,
        terminal_growth: float = 0.0,
        terminal_year: Optional[int] = None) -> Tuple[float, float]:
    """
    Adjusted Present Value model.

    VL = VU + PV(tax shields)
    VU is the present value of unlevered free cash flows discounted at rU.
    PV of tax shields is computed by discounting each period's interest tax shield.
    If a terminal growth rate is provided, a perpetuity terminal value is added.

    Args:
        unlevered_fcff (list): Projected unlevered FCFF for explicit years.
        unlevered_cost_of_equity (float): rU.
        interest_tax_shields (list): Expected interest tax shields (T_c * Interest),
                                     same length as unlevered_fcff, or one extra for terminal.
        discount_rate_tax_shields (float): Discount rate for tax shields (e.g., cost of debt).
        net_debt (float): Net debt.
        shares_outstanding (float): Shares outstanding.
        terminal_growth (float): Perpetual growth rate for terminal value (default 0).
        terminal_year (int, optional): If provided, the index in the list where terminal value
                                       begins (often len(list)). Default is len(list).

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    n = len(unlevered_fcff)
    if terminal_year is None:
        terminal_year = n

    rU = unlevered_cost_of_equity
    rTS = discount_rate_tax_shields

    # PV of unlevered FCFFs
    pv_unlevered = sum(unlevered_fcff[t] / (1 + rU) ** (t + 1) for t in range(terminal_year))

    # Terminal value for unlevered FCFF if growth > 0
    if terminal_growth > 0 and terminal_year == n:
        # last explicit cash flow multiplied by (1+g) / (rU - g)
        last_fcf = unlevered_fcff[-1]
        tv_unlevered = last_fcf * (1 + terminal_growth) / (rU - terminal_growth)
        pv_unlevered += tv_unlevered / (1 + rU) ** n
    # If terminal_year < n, we assume the cash flows beyond terminal_year are already included
    # (the user passes a longer list). For simplicity, we just discount all provided.

    # PV of tax shields
    pv_tax_shields = sum(interest_tax_shields[t] / (1 + rTS) ** (t + 1) for t in range(len(interest_tax_shields)))

    enterprise_value = pv_unlevered + pv_tax_shields
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


# -----------------------------------------------------------------------------
# 5. Capital Cash Flow (CCF) Model
# -----------------------------------------------------------------------------

def ccf(capital_cash_flows: List[float],
        unlevered_cost_of_equity: float,
        net_debt: float,
        shares_outstanding: float,
        terminal_growth: float = 0.0) -> Tuple[float, float]:
    """
    Capital Cash Flow model.

    The CCF is defined as Unlevered FCF + Interest Tax Shield.
    All cash flows are discounted at the unlevered cost of equity (ρ = rU).

    Args:
        capital_cash_flows (list): Projected capital cash flows (FCF_U + T_c * I).
        unlevered_cost_of_equity (float): rU.
        net_debt (float): Net debt.
        shares_outstanding (float): Shares outstanding.
        terminal_growth (float): Perpetual growth rate after explicit forecast (default 0).

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    n = len(capital_cash_flows)
    rU = unlevered_cost_of_equity

    pv = sum(capital_cash_flows[t] / (1 + rU) ** (t + 1) for t in range(n))

    if terminal_growth > 0:
        if rU <= terminal_growth:
            raise ValueError("Unlevered cost of equity must exceed terminal growth.")
        last_ccf = capital_cash_flows[-1]
        tv = last_ccf * (1 + terminal_growth) / (rU - terminal_growth)
        pv += tv / (1 + rU) ** n

    enterprise_value = pv
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


# -----------------------------------------------------------------------------
# 6. Equity Cash Flow (ECF) / Levered Equity Model
# -----------------------------------------------------------------------------

def ecf(equity_cash_flows: List[float],
        cost_of_equity: float,
        shares_outstanding: float,
        terminal_growth: float = 0.0) -> Tuple[float, float]:
    """
    Equity Cash Flow model.
    ECF = Dividends + Share repurchases - New equity issues.
    Discounted at the cost of equity.

    Args:
        equity_cash_flows (list): Projected equity cash flows.
        cost_of_equity (float): r.
        shares_outstanding (float): Shares outstanding.
        terminal_growth (float): Perpetual growth after explicit forecast (default 0).

    Returns:
        tuple: (equity_value, value_per_share)
    """
    r = cost_of_equity
    n = len(equity_cash_flows)

    pv = sum(equity_cash_flows[t] / (1 + r) ** (t + 1) for t in range(n))

    if terminal_growth > 0:
        if r <= terminal_growth:
            raise ValueError("Cost of equity must exceed terminal growth.")
        last_ecf = equity_cash_flows[-1]
        tv = last_ecf * (1 + terminal_growth) / (r - terminal_growth)
        pv += tv / (1 + r) ** n

    equity_value = pv
    value_per_share = equity_value / shares_outstanding
    return equity_value, value_per_share


# -----------------------------------------------------------------------------
# 7. Residual Income / Economic Profit Models
# -----------------------------------------------------------------------------

def residual_income(book_value_equity: float,
                    net_income_forecast: List[float],
                    book_values_forecast: List[float],
                    cost_of_equity: float,
                    shares_outstanding: float,
                    terminal_growth: float = 0.0) -> Tuple[float, float]:
    """
    Residual Income Model (RIM).

    Equity Value = B0 + Σ RI_t / (1+r)^t + (Terminal RI)
    RI_t = NI_t - r * B_{t-1}
    B_{t-1} is the book value at the beginning of period t.

    Args:
        book_value_equity (float): Current book value of equity (B0).
        net_income_forecast (list): Projected net income for explicit years.
        book_values_forecast (list): Projected ending book values of equity for each year
                                     (length = len(net_income_forecast)). The beginning book
                                     value for year t+1 is the ending book value of year t.
        cost_of_equity (float): r.
        shares_outstanding (float): Shares outstanding.
        terminal_growth (float): Perpetual growth of residual income after explicit forecast.

    Returns:
        tuple: (equity_value, value_per_share)
    """
    r = cost_of_equity
    n = len(net_income_forecast)
    if len(book_values_forecast) != n:
        raise ValueError("book_values_forecast must have same length as net_income_forecast.")

    # Beginning book for year 1 is B0 (current book value)
    b_begin = book_value_equity
    pv_ri = 0.0

    for t in range(n):
        ni = net_income_forecast[t]
        ri = ni - r * b_begin
        pv_ri += ri / (1 + r) ** (t + 1)
        # update beginning book for next period
        b_begin = book_values_forecast[t]

    # Terminal value of residual income (if growth)
    if terminal_growth > 0 and n > 0:
        if r <= terminal_growth:
            raise ValueError("Cost of equity must exceed terminal RI growth.")
        # Last residual income calculated with last b_begin (which is book_values_forecast[-2]? careful)
        # RI_{n+1} = NI_{n+1} - r * B_n. We don't have NI_{n+1}. Instead we can use the last explicit RI
        # and assume it grows at g. So we need the residual income of year n.
        last_ri = net_income_forecast[-1] - r * (book_value_equity if n==1 else book_values_forecast[-2])
        # Better: recalculate last_ri using book value at start of year n.
        # Let's do it properly:
        if n == 1:
            b_start_last = book_value_equity
        else:
            b_start_last = book_values_forecast[-2]  # beginning book for year n
        last_ri = net_income_forecast[-1] - r * b_start_last
        tv_ri = last_ri * (1 + terminal_growth) / (r - terminal_growth)
        pv_ri += tv_ri / (1 + r) ** n

    equity_value = book_value_equity + pv_ri
    value_per_share = equity_value / shares_outstanding
    return equity_value, value_per_share


def eva(invested_capital_0: float,
        nopat_forecast: List[float],
        invested_capital_forecast: List[float],
        wacc: float,
        net_debt: float,
        shares_outstanding: float,
        terminal_growth: float = 0.0) -> Tuple[float, float]:
    """
    Economic Value Added (EVA) model.

    Enterprise Value = IC0 + Σ EVA_t / (1+WACC)^t
    EVA_t = NOPAT_t - WACC * IC_{t-1}

    Args:
        invested_capital_0 (float): Current invested capital (debt + equity).
        nopat_forecast (list): Projected NOPAT for explicit years.
        invested_capital_forecast (list): Projected ending invested capital for each year
                                          (length = len(nopat_forecast)).
        wacc (float): Weighted average cost of capital.
        net_debt (float): Net debt.
        shares_outstanding (float): Shares outstanding.
        terminal_growth (float): Perpetual EVA growth after explicit period.

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    n = len(nopat_forecast)
    if len(invested_capital_forecast) != n:
        raise ValueError("invested_capital_forecast must match nopat_forecast length.")

    ic_begin = invested_capital_0
    pv_eva = 0.0
    for t in range(n):
        nopat = nopat_forecast[t]
        eva_t = nopat - wacc * ic_begin
        pv_eva += eva_t / (1 + wacc) ** (t + 1)
        ic_begin = invested_capital_forecast[t]

    if terminal_growth > 0 and n > 0:
        if wacc <= terminal_growth:
            raise ValueError("WACC must exceed terminal EVA growth.")
        # Last EVA
        if n == 1:
            ic_start_last = invested_capital_0
        else:
            ic_start_last = invested_capital_forecast[-2]
        last_eva = nopat_forecast[-1] - wacc * ic_start_last
        tv_eva = last_eva * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_eva += tv_eva / (1 + wacc) ** n

    enterprise_value = invested_capital_0 + pv_eva
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


# -----------------------------------------------------------------------------
# 8. Sum-of-the-Parts DCF
# -----------------------------------------------------------------------------

def sum_of_the_parts(segment_values: List[float],
                     net_debt: float,
                     shares_outstanding: float) -> Tuple[float, float]:
    """
    Sum-of-the-Parts valuation.

    Aggregates independently valued business segments.

    Args:
        segment_values (list): Enterprise values of each segment.
        net_debt (float): Consolidated net debt.
        shares_outstanding (float): Shares outstanding.

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    enterprise_value = sum(segment_values)
    equity_value = enterprise_value - net_debt
    value_per_share = equity_value / shares_outstanding
    return enterprise_value, value_per_share


# -----------------------------------------------------------------------------
# 9. Real Options Valuation (Extended DCF)
# -----------------------------------------------------------------------------

def decision_tree_dcf(scenarios: List[Tuple[float, float]],
                      discount_rate: float) -> float:
    """
    Decision Tree DCF: expected NPV based on probability-weighted scenarios.

    Each scenario is a (probability, npv) tuple. Probabilities must sum to 1.

    Args:
        scenarios (list of tuples): [(prob1, npv1), (prob2, npv2), ...]
        discount_rate (float): Not directly used here if NPV already discounted;
                               included for interface consistency.

    Returns:
        float: Expected NPV (firm value). For per-share value, divide by shares.
    """
    expected_npv = sum(prob * npv for prob, npv in scenarios)
    if not np.isclose(sum(prob for prob, _ in scenarios), 1.0):
        raise ValueError("Probabilities must sum to 1.")
    return expected_npv

# The decision_tree_dcf doesn't return (total, per share) because it's not directly a
# firm value model with shares. We'll wrap it for consistency later if needed.
# Instead, we'll provide a wrapper that assumes net_debt and shares.

def decision_tree_dcf_wrapper(scenarios: List[Tuple[float, float]],
                              net_debt: float,
                              shares_outstanding: float) -> Tuple[float, float]:
    """
    Wrapper for decision tree DCF to return (firm_value, value_per_share).

    Args:
        scenarios: list of (probability, enterprise_npv) for each scenario.
        net_debt: net debt.
        shares_outstanding: shares.

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    ev = decision_tree_dcf(scenarios, discount_rate=0.0)  # discount_rate not used
    equity = ev - net_debt
    return ev, equity / shares_outstanding


def real_option_black_scholes(underlying_value: float,
                              exercise_price: float,
                              risk_free_rate: float,
                              volatility: float,
                              time_to_maturity: float,
                              dividend_yield: float = 0.0) -> float:
    """
    Real option valuation using Black-Scholes formula for a European call.

    C = S0 * e^{-qT} * N(d1) - X * e^{-rT} * N(d2)
    d1 = (ln(S0/X) + (r - q + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T

    Args:
        underlying_value (float): Present value of project cash flows (S0).
        exercise_price (float): Investment cost (X).
        risk_free_rate (float): r (continuous).
        volatility (float): σ (annual).
        time_to_maturity (float): T (years).
        dividend_yield (float): q (annual dividend yield, default 0).

    Returns:
        float: Call option value (can be added to static NPV).
    """
    if time_to_maturity <= 0:
        return max(underlying_value - exercise_price, 0)

    d1 = (np.log(underlying_value / exercise_price) +
          (risk_free_rate - dividend_yield + 0.5 * volatility**2) * time_to_maturity) / \
         (volatility * np.sqrt(time_to_maturity))
    d2 = d1 - volatility * np.sqrt(time_to_maturity)

    call = (underlying_value * np.exp(-dividend_yield * time_to_maturity) * norm.cdf(d1) -
            exercise_price * np.exp(-risk_free_rate * time_to_maturity) * norm.cdf(d2))
    return call


# -----------------------------------------------------------------------------
# 10. Stochastic / Simulation DCF Models
# -----------------------------------------------------------------------------

def monte_carlo_dcf(base_cash_flow: float,
                    growth_mean: float,
                    growth_std: float,
                    discount_rate: float,
                    years: int,
                    n_simulations: int,
                    terminal_growth: float = 0.0) -> Tuple[float, np.ndarray]:
    """
    Monte Carlo DCF simulation.

    Simulates cash flow paths assuming log-normal growth rates.
    Each path is discounted to present value; the expected NPV is the mean.

    Args:
        base_cash_flow (float): Initial cash flow (CF_0).
        growth_mean (float): Mean annual growth rate (μ).
        growth_std (float): Standard deviation of growth rate (σ).
        discount_rate (float): Risk-adjusted discount rate.
        years (int): Number of years to simulate.
        n_simulations (int): Number of simulation paths.
        terminal_growth (float): Perpetual growth after explicit years (used in terminal value).

    Returns:
        tuple: (expected_npv, npv_distribution_array)
    """
    np.random.seed(42)  # for reproducibility
    npvs = np.zeros(n_simulations)

    for i in range(n_simulations):
        cf = base_cash_flow
        pv = 0.0
        for t in range(1, years + 1):
            growth = np.random.normal(growth_mean, growth_std)
            cf *= (1 + growth)
            pv += cf / (1 + discount_rate) ** t
        # Terminal value
        if terminal_growth > 0:
            tv = cf * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv += tv / (1 + discount_rate) ** years
        npvs[i] = pv

    expected_npv = np.mean(npvs)
    return expected_npv, npvs

def monte_carlo_dcf_wrapper(base_cash_flow: float,
                            growth_mean: float,
                            growth_std: float,
                            discount_rate: float,
                            years: int,
                            n_simulations: int,
                            net_debt: float,
                            shares_outstanding: float,
                            terminal_growth: float = 0.0) -> Tuple[float, float]:
    """
    Wrapper for Monte Carlo DCF returning (firm_value, value_per_share).

    Assumes the cash flow is unlevered FCFF (enterprise value).

    Returns:
        tuple: (enterprise_value, value_per_share)
    """
    ev, _ = monte_carlo_dcf(base_cash_flow, growth_mean, growth_std,
                            discount_rate, years, n_simulations, terminal_growth)
    equity = ev - net_debt
    return ev, equity / shares_outstanding


def certainty_equivalent_dcf(expected_cash_flows: List[float],
                             ce_coefficients: List[float],
                             risk_free_rate: float) -> float:
    """
    Certainty-equivalent DCF.

    V0 = Σ (α_t * E[CF_t]) / (1 + rf)^t

    Args:
        expected_cash_flows (list): Expected future cash flows.
        ce_coefficients (list): Certainty-equivalent coefficients α_t (0 < α_t ≤ 1).
        risk_free_rate (float): Risk-free rate.

    Returns:
        float: Present value (firm or equity value, depending on cash flow nature).
    """
    n = len(expected_cash_flows)
    if len(ce_coefficients) != n:
        raise ValueError("ce_coefficients must have same length as expected_cash_flows.")

    pv = 0.0
    for t in range(n):
        ce_cf = ce_coefficients[t] * expected_cash_flows[t]
        pv += ce_cf / (1 + risk_free_rate) ** (t + 1)
    return pv

def certainty_equivalent_dcf_wrapper(expected_cash_flows: List[float],
                                     ce_coefficients: List[float],
                                     risk_free_rate: float,
                                     is_enterprise: bool,
                                     net_debt: float,
                                     shares_outstanding: float) -> Tuple[float, float]:
    """
    Wrapper to return (total_value, value_per_share).
    If is_enterprise=True, cash flows represent FCFF and we subtract net debt.
    Otherwise, cash flows are equity cash flows.
    """
    pv = certainty_equivalent_dcf(expected_cash_flows, ce_coefficients, risk_free_rate)
    if is_enterprise:
        enterprise_value = pv
        equity_value = enterprise_value - net_debt
        return enterprise_value, equity_value / shares_outstanding
    else:
        equity_value = pv
        return equity_value, equity_value / shares_outstanding


# -----------------------------------------------------------------------------
# Utility: fetch basic financial data using yfinance
# -----------------------------------------------------------------------------

def get_financial_data(ticker: str) -> dict:
    """
    Fetch financial statements and key metrics for a ticker using yfinance.

    Returns a dictionary with:
        - latest annual dividend per share
        - free cash flow to firm (simplified: EBIT*(1-t) + D&A - CapEx - ΔWC)
        - free cash flow to equity (FCFF - net interest*(1-t) + net borrowing)
        - shares outstanding
        - total debt, cash, net debt
        - current book value of equity
        - net income
        - invested capital (book value of debt + equity)
        - NOPAT (EBIT*(1-t))
    Note: Simplified calculations; may need adjustment for real-world use.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # Shares outstanding
    shares = info.get('sharesOutstanding', None)
    if shares is None:
        # fallback to market cap / price
        price = info.get('currentPrice') or info.get('regularMarketPreviousClose')
        market_cap = info.get('marketCap')
        if price and market_cap:
            shares = market_cap / price
        else:
            shares = 1e9  # dummy

    # Balance sheet items
    bs = stock.balance_sheet
    if bs.empty:
        raise ValueError("Balance sheet data not available.")
    # Use latest annual (most recent column)
    latest_bs = bs.iloc[:, 0]  # first column is most recent
    total_debt = latest_bs.get('Total Debt', 0) or latest_bs.get('Long Term Debt', 0) + latest_bs.get('Short Long Term Debt', 0)
    cash = latest_bs.get('Cash And Cash Equivalents', 0) or latest_bs.get('Cash', 0)
    net_debt = total_debt - cash
    book_value_equity = latest_bs.get('Total Stockholder Equity', 0)
    # Invested capital = total debt + equity (book)
    invested_capital = total_debt + book_value_equity

    # Income statement
    income_stmt = stock.financials
    if income_stmt.empty:
        raise ValueError("Income statement data not available.")
    latest_is = income_stmt.iloc[:, 0]
    ebit = latest_is.get('Ebit', 0) or latest_is.get('Operating Income', 0)
    interest_expense = latest_is.get('Interest Expense', 0)
    net_income = latest_is.get('Net Income', 0)
    tax_rate = info.get('taxRate', None)
    if tax_rate is None:
        # estimate from income tax / pretax income
        pretax = latest_is.get('Pretax Income', None)
        tax = latest_is.get('Income Tax Expense', None)
        if pretax and tax and pretax != 0:
            tax_rate = tax / pretax
        else:
            tax_rate = 0.21  # default US corporate rate

    # Cash flow statement
    cf = stock.cashflow
    if cf.empty:
        raise ValueError("Cash flow statement not available.")
    latest_cf = cf.iloc[:, 0]
    capex = latest_cf.get('Capital Expenditure', 0)  # negative number
    depreciation = latest_cf.get('Depreciation', 0)  # usually positive; we need D&A
    # Also try to get from income statement
    if depreciation == 0:
        depreciation = latest_is.get('Depreciation', 0) or latest_is.get('Reconciled Depreciation', 0)
    # change in working capital (simplified)
    # often not directly given; use changes in current assets/liabilities (crude)
    change_in_wc = 0  # placeholder; real model would compute from balance sheet differences

    # FCFF = EBIT*(1-t) + D&A - CapEx - ΔWC
    # Note: CapEx is usually negative in yfinance, so we add it (e.g., -500 means subtract 500)
    capex_value = abs(capex)  # assume it's reported as negative, make positive for formula
    # Actually, in yfinance, CapEx is typically negative. We'll use: FCFF = EBIT*(1-t) + D&A - (-CapEx) if negative
    if capex < 0:
        capex_value = -capex
    else:
        capex_value = capex
    # But we'll stick to formula: FCFF = EBIT*(1-t) + D&A - CapEx (where CapEx is absolute cash outflow).
    fcff = ebit * (1 - tax_rate) + depreciation - capex_value - change_in_wc

    # FCFE = FCFF - Interest*(1-t) + Net Borrowing
    # Net borrowing approximated as 0 for simplicity
    interest_after_tax = interest_expense * (1 - tax_rate)
    fcfe = fcff - interest_after_tax  # ignoring net borrowing

    # Dividend per share
    dividend_per_share = info.get('dividendRate', 0)  # forward dividend, approximate current
    if dividend_per_share is None:
        dividend_per_share = 0

    # Cost of equity placeholder using CAPM (needs risk-free rate and beta)
    beta = info.get('beta', 1.0)
    risk_free = 0.04  # 4% placeholder
    market_premium = 0.05
    cost_of_equity = risk_free + beta * market_premium

    # Cost of debt (simplistic)
    if total_debt > 0:
        cost_of_debt = interest_expense / total_debt if interest_expense else 0.04
    else:
        cost_of_debt = 0.04

    # WACC
    equity_market = info.get('marketCap', book_value_equity)  # use market cap
    total_capital = equity_market + total_debt
    wacc = (cost_of_equity * equity_market + cost_of_debt * (1 - tax_rate) * total_debt) / total_capital

    return {
        'shares_outstanding': shares,
        'dividend_per_share': dividend_per_share,
        'fcff': fcff,
        'fcfe': fcfe,
        'net_debt': net_debt,
        'book_value_equity': book_value_equity,
        'invested_capital': invested_capital,
        'ebit': ebit,
        'nopat': ebit * (1 - tax_rate),
        'net_income': net_income,
        'cost_of_equity': cost_of_equity,
        'wacc': wacc,
        'risk_free_rate': risk_free,
        'tax_rate': tax_rate,
        'beta': beta
    }

# -----------------------------------------------------------------------------
# Scenario error analysis
# -----------------------------------------------------------------------------

def scenario_error_analysis(ticker: str,
                            base_value: float,
                            bull_value: float,
                            bear_value: float,
                            currency_symbol: str = "$ ") -> Tuple[float, float, float, float, float, float]:
    """
    Fetch the latest share price via yfinance and compute the
    root-mean-squared-error-style deviation for each scenario value.

    Args:
        ticker: Yahoo Finance ticker symbol.
        base_value: Average fair value per share for the base scenario.
        bull_value: Average fair value per share for the bull scenario.
        bear_value: Average fair value per share for the bear scenario.
        currency_symbol: Currency display string.

    Returns:
        tuple: (base_value, bull_value, bear_value, base_rmse, bull_rmse, bear_rmse)
    """
    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        latest_price = info.get('currentPrice') or info.get('regularMarketPreviousClose')
        if latest_price is None:
            # fallback: try history
            hist = stock.history(period="1d")
            latest_price = float(hist['Close'].iloc[-1]) if not hist.empty else None
    except Exception:
        latest_price = None

    if latest_price is None:
        print(f"Could not retrieve current price for {ticker}.\n")
        return base_value, bull_value, bear_value, 0.0, 0.0, 0.0

    print(f"\n============= SCENARIO ERROR ANALYSIS ({ticker}) =============")
    print(f"Latest share price: {currency_symbol}{latest_price:,.2f}\n")

    labels = ['Base', 'Bull', 'Bear']
    values = [base_value, bull_value, bear_value]
    rmses = []

    for label, val in zip(labels, values):
        diff = latest_price - val
        rmse = (diff ** 2) ** 0.5   # sqrt(diff²) = absolute deviation
        rmses.append(rmse)
        direction = "undervalued" if diff < 0 else "overvalued"
        print(f"  {label:5s} value: {currency_symbol}{val:>10.2f}  |  "
              f"Market {direction}: {currency_symbol}{rmse:>10.2f} per share")

    print("===========================================================\n")

    return base_value, bull_value, bear_value, rmses[0], rmses[1], rmses[2]


# -----------------------------------------------------------------------------
# Main analysis function — callable from other modules
# -----------------------------------------------------------------------------

def run_valuation_analysis(
    ticker: str = "INFY.NS",
    currency: str = "₹ ",
    scenario_assumptions: Optional[List[dict]] = None,
    scenarios_dt: Optional[List[Tuple[float, float]]] = None,
    data: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """
    Run a full multi-model, multi-scenario DCF valuation and return results.

    Args:
        ticker: Yahoo Finance ticker symbol.
        currency: Currency display string (e.g. "$ ", "₹ ").
        scenario_assumptions: List of scenario dicts with keys
            'label', 'g_stable', 'g_high_fcff', 'g_high_div', 'g_high_ni',
            'years_high', 'years_transition', 'half_life_H'.
            Defaults to three scenarios (base / bull / bear).
        scenarios_dt: List of (probability, npv) tuples for the decision-tree model.
        data: Optional pre-fetched financial data dict. If None, fetched via
              get_financial_data(ticker); falls back to dummy data on error.
        verbose: If True, print detailed results to stdout.

    Returns:
        dict with keys:
            - scenario_results: list of dicts per scenario
            - all_values_per_share: flattened list of all value-per-share estimates
            - average_all: mean of all value-per-share estimates
            - base_val, bull_val, bear_val: per-scenario averages
            - base_rmse, bull_rmse, bear_rmse: RMSE vs market price
            - ticker, currency
    """
    # --- Default scenario assumptions ---
    if scenario_assumptions is None:
        scenario_assumptions = [
            {
                'label': 'base',
                'g_stable': 0.025,
                'g_high_fcff': 0.06,
                'g_high_div': 0.06,
                'g_high_ni': 0.055,
                'years_high': 6,
                'years_transition': 4,
                'half_life_H': 4,
            },
            {
                'label': 'bull',
                'g_stable': 0.05,
                'g_high_fcff': 0.14,
                'g_high_div': 0.12,
                'g_high_ni': 0.11,
                'years_high': 9,
                'years_transition': 6,
                'half_life_H': 6,
            },
            {
                'label': 'bear',
                'g_stable': 0.02,
                'g_high_fcff': 0.05,
                'g_high_div': 0.05,
                'g_high_ni': 0.05,
                'years_high': 3,
                'years_transition': 2,
                'half_life_H': 2,
            },
        ]

    if scenarios_dt is None:
        scenarios_dt = [(0.3, 2.2e12), (0.5, 2.5e12), (0.2, 2.8e12)]

    # --- 1. Fetch / prepare financial data ---
    if data is None:
        try:
            data = get_financial_data(ticker)
            if verbose:
                print(f"Fetched data for {ticker} successfully.\n")
        except Exception as e:
            if verbose:
                print(f"Data fetch error: {e}")
                print("Using dummy data for demonstration.\n")
            data = {
                'shares_outstanding': 16e9,
                'dividend_per_share': 0.96,
                'fcff': 111.5e9,
                'fcfe': 105.2e9,
                'net_debt': -54e9,
                'book_value_equity': 62.1e9,
                'invested_capital': 220e9,
                'nopat': 98e9,
                'net_income': 94e9,
                'cost_of_equity': 0.089,
                'wacc': 0.082,
                'risk_free_rate': 0.04,
                'tax_rate': 0.15,
                'beta': 0.98,
            }

    # --- 2. Unpack financial data ---
    shares = data['shares_outstanding']
    div0 = data['dividend_per_share']
    fcff0 = data['fcff']
    fcfe0 = data['fcfe']
    net_debt = data['net_debt']
    book_equity0 = data['book_value_equity']
    invested_capital0 = data['invested_capital']
    nopat0 = data['nopat']
    net_income0 = data['net_income']
    r_e = data['cost_of_equity']
    wacc = data['wacc']
    r_f = data['risk_free_rate']
    tax_rate = data['tax_rate']

    all_value_per_share = []
    scenario_results = []

    for scenario in scenario_assumptions:
        g_stable = scenario['g_stable']
        g_high_fcff = scenario['g_high_fcff']
        g_high_div = scenario['g_high_div']
        g_high_ni = scenario['g_high_ni']
        years_high = scenario['years_high']
        years_transition = scenario['years_transition']
        half_life_H = scenario['half_life_H']

        # --- 3. Build explicit forecast lists for this scenario ---
        fcff_forecast = []
        fcff_prev = fcff0
        for _ in range(years_high):
            fcff_prev *= (1 + g_high_fcff)
            fcff_forecast.append(fcff_prev)

        fcfe_forecast = []
        fcfe_prev = fcfe0
        for _ in range(years_high):
            fcfe_prev *= (1 + g_high_fcff)
            fcfe_forecast.append(fcfe_prev)

        div_forecast = []
        div_prev = div0
        for _ in range(years_high):
            div_prev *= (1 + g_high_div)
            div_forecast.append(div_prev)

        ni_forecast = []
        bv_forecast = []
        bv_prev = book_equity0
        ni_prev = net_income0
        for _ in range(years_high):
            ni_prev *= (1 + g_high_ni)
            bv_prev = bv_prev + ni_prev * 0.4
            ni_forecast.append(ni_prev)
            bv_forecast.append(bv_prev)

        nopat_forecast = []
        ic_forecast = []
        ic_prev = invested_capital0
        nopat_prev = nopat0
        for _ in range(years_high):
            nopat_prev *= (1 + g_high_fcff)
            ic_prev = ic_prev + nopat_prev * 0.5
            nopat_forecast.append(nopat_prev)
            ic_forecast.append(ic_prev)

        unlevered_fcff = fcff_forecast.copy()
        interest_tax_shields = [cf * 0.02 for cf in fcff_forecast]
        capital_cf = [u + s for u, s in zip(unlevered_fcff, interest_tax_shields)]
        equity_cf = fcfe_forecast.copy()

        # Clamp stable growth safely below discount rates
        g_stable_eq = min(g_stable, r_e * 0.9)
        g_stable_wacc = min(g_stable, wacc * 0.9)
        segment_values = [
            fcff_single_stage(fcff0 * 0.6, wacc, g_stable_wacc, shares, 0)[0],
            fcff_single_stage(fcff0 * 0.4, wacc, g_stable_wacc, shares, 0)[0],
        ]

        ce_coeffs = [0.85] * years_high
        scenario_vps = []

        if verbose:
            print(f"============= VALUATION MODEL RESULTS ({scenario['label'].capitalize()}) =============\n")

        if verbose:
            print("1a) Gordon Growth DDM:")
        _, vps = ddm_gordon(div0, r_e, g_stable_eq, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("1b) Two‑Stage DDM:")
        _, vps = ddm_two_stage(div0, g_high_div, g_stable_eq, years_high, r_e, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("1c) Three‑Stage DDM:")
        _, vps = ddm_three_stage(div0, g_high_div, g_stable_eq, years_high, years_transition, r_e, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("1d) H‑Model DDM:")
        _, vps = ddm_h_model(div0, g_high_div, g_stable_eq, half_life_H, r_e, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("2a) FCFF Single Stage:")
        _, vps = fcff_single_stage(fcff0, wacc, g_stable_wacc, shares, net_debt)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("2b) FCFF Two‑Stage:")
        _, vps = fcff_two_stage(fcff0, g_high_fcff, g_stable_wacc, years_high, wacc, shares, net_debt)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("3a) FCFE Constant Growth:")
        _, vps = fcfe_constant_growth(fcfe0, r_e, g_stable_eq, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("3b) FCFE Two‑Stage:")
        _, vps = fcfe_two_stage(fcfe0, g_high_fcff, g_stable_eq, years_high, r_e, shares)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("4) APV (3yr explicit, terminal growth):")
        _, vps = apv(unlevered_fcff, r_e, interest_tax_shields, 0.04, net_debt, shares,
                      terminal_growth=g_stable_eq)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("5) Capital Cash Flow (CCF):")
        _, vps = ccf(capital_cf, r_e, net_debt, shares, terminal_growth=g_stable_eq)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("6) Equity Cash Flow (ECF):")
        _, vps = ecf(equity_cf, r_e, shares, terminal_growth=g_stable_eq)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("7a) Residual Income Model (RIM):")
        _, vps = residual_income(book_equity0, ni_forecast, bv_forecast, r_e, shares,
                                  terminal_growth=g_stable_eq)
        if verbose:
            print(f"     Equity value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("7b) Economic Value Added (EVA):")
        _, vps = eva(invested_capital0, nopat_forecast, ic_forecast, wacc, net_debt, shares,
                      terminal_growth=g_stable_wacc)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("8) Sum‑of‑the‑Parts (2 segments):")
        _, vps = sum_of_the_parts(segment_values, net_debt, shares)
        if verbose:
            print(f"     Enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("9a) Decision Tree DCF:")
        _, vps = decision_tree_dcf_wrapper(scenarios_dt, net_debt, shares)
        if verbose:
            print(f"     Expected enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("9b) Real Option (Black‑Scholes expansion option):")
        option_val = real_option_black_scholes(200e9, 180e9, r_f, 0.3, 3)
        if verbose:
            print(f"     Option premium: {currency}{option_val:,.0f} (add to base NPV)\n")
        scenario_vps.append(vps)

        if verbose:
            print("10a) Monte Carlo DCF (5,000 paths):")
        _, vps = monte_carlo_dcf_wrapper(fcff0, g_high_fcff, 0.05, wacc, years_high, 5000,
                                          net_debt, shares, terminal_growth=g_stable_wacc)
        if verbose:
            print(f"     Expected enterprise value: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("10b) Certainty‑Equivalent DCF (FCFF, α=0.85):")
        _, vps = certainty_equivalent_dcf_wrapper(fcff_forecast, ce_coeffs, r_f, True,
                                                   net_debt, shares)
        if verbose:
            print(f"     PV of CE cash flows: {currency}{_:,.0f}, Value/share: {currency}{vps:.2f}\n")
        scenario_vps.append(vps)

        if verbose:
            print("=================================================\n")

        scenario_results.append({
            'label': scenario['label'],
            'values_per_share': scenario_vps,
            'average_value_per_share': np.mean(scenario_vps),
        })
        all_value_per_share.extend(scenario_vps)

    if verbose:
        print("============= SCENARIO SUMMARY =============\n")
        for scenario in scenario_results:
            print(f"{scenario['label'].capitalize()} scenario average value/share: {currency}{scenario['average_value_per_share']:.2f}")
        print(f"\nAverage value per share across all scenarios: {currency}{np.mean(all_value_per_share):.2f}\n")
        print("=================================================")

    base_val = scenario_results[0]['average_value_per_share']
    bull_val = scenario_results[1]['average_value_per_share']
    bear_val = scenario_results[2]['average_value_per_share']
    base_val, bull_val, bear_val, base_rmse, bull_rmse, bear_rmse = \
        scenario_error_analysis(ticker, base_val, bull_val, bear_val, currency)

    return {
        'scenario_results': scenario_results,
        'all_values_per_share': all_value_per_share,
        'average_all': np.mean(all_value_per_share),
        'base_val': base_val,
        'bull_val': bull_val,
        'bear_val': bear_val,
        'base_rmse': base_rmse,
        'bull_rmse': bull_rmse,
        'bear_rmse': bear_rmse,
        'ticker': ticker,
        'currency': currency,
    }


if __name__ == "__main__":
    run_valuation_analysis()
"""
=====================================================================
 TECH_ANALYSIS.py — Technical Analysis Toolkit for Stocks
=====================================================================
This module computes a comprehensive set of technical indicators from
the last 5 years of *daily* Open / High / Low / Close / Volume (OHLCV)
data, downloaded live from Yahoo Finance via the ``yfinance`` library.

It is written for learners and analysts: every helper is documented with
its formula, a worked example (sample input -> sample output), and
line-by-line comments so you can follow exactly what each calculation
does.

---------------------------------------------------------------------
 WHAT THIS MODULE PRODUCES
---------------------------------------------------------------------
The main entry point is ``compute_technical_indicators(ticker)``. It
returns a single ``pandas.DataFrame`` with one row per trading day and
the following columns:

  1. Raw OHLCV data
       Open, High, Low, Close, Volume

  2. Moving averages — for BOTH the closing price and the trading volume,
     a Simple Moving Average (SMA) and an Exponential Moving Average (EMA)
     are computed for each window:
       windows = 5, 13, 23, 90, 200
     -> columns:
       SMA_5, SMA_13, SMA_23, SMA_90, SMA_200           (price, SMA)
       EMA_5, EMA_13, EMA_23, EMA_90, EMA_200           (price, EMA)
       VOL_SMA_5, ..., VOL_SMA_200                      (volume, SMA)
       VOL_EMA_5, ..., VOL_EMA_200                      (volume, EMA)

  3. Technical indicators
       RSI          Relative Strength Index (14-day, Wilder's smoothing)
       MACD         Moving Average Convergence/Divergence (12, 26, 9)
       MACD_SIGNAL  EMA(9) of the MACD line
       MACD_HIST    MACD - MACD_SIGNAL
       BB_UPPER     Bollinger Band upper  (20-day, +2 standard deviations)
       BB_MIDDLE    Bollinger Band middle (20-day SMA)
       BB_LOWER     Bollinger Band lower  (20-day, -2 standard deviations)
       STOCH_K      Stochastic %K (14-day look-back)
       STOCH_D      Stochastic %D (3-day SMA of %K)
       OBV          On-Balance Volume (cumulative volume flow)

---------------------------------------------------------------------
 DEPENDENCIES
---------------------------------------------------------------------
  * ``yfinance``  — pulls live market data from Yahoo Finance.
                    (already listed in requirements.txt)
  * ``pandas``    — data tables and rolling / EWM calculations.
  * ``numpy``     — fast numeric helpers (e.g. np.sign for OBV).

 No third-party technical-analysis package is required — every
 indicator is computed from first principles so you can see the math.

---------------------------------------------------------------------
 QUICK START (HOW TO USE THIS MODULE)
---------------------------------------------------------------------
    >>> import TECH_ANALYSIS as ta
    >>> df = ta.compute_technical_indicators("AAPL")   # 5y of daily data
    >>> print(df.tail())                               # last 5 rows
    >>> ta.show_latest(df, ticker="AAPL")              # pretty snapshot

 Or from the command line:
    python TECH_ANALYSIS.py AAPL
    python TECH_ANALYSIS.py RELIANCE.NS --rows 3
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import numpy as np       # numpy: fast numeric helpers (np.sign in OBV)
import pandas as pd      # pandas: DataFrames, rolling() and ewm() tools
import yfinance as yf    # yfinance: downloads OHLCV history from Yahoo

# ----------------------------------------------------------------------
# Configuration constants (tweak these to change indicator behaviour)
# ----------------------------------------------------------------------

# Moving-average windows applied to BOTH price and volume.
# e.g. SMA_5 = average of the last 5 closing prices; SMA_200 = last 200.
MA_WINDOWS = (5, 13, 23, 90, 200)

# RSI look-back period (the industry standard is 14 days).
RSI_PERIOD        = 14

# MACD parameters: fast EMA, slow EMA, and signal EMA.
# MACD = EMA(12) - EMA(26);  Signal = EMA(9) of the MACD line.
MACD_FAST         = 12
MACD_SLOW         = 26
MACD_SIGNAL       = 9

# Bollinger Bands: 20-day middle band, +/- 2 standard deviations.
BB_WINDOW         = 20
BB_NUM_STD        = 2.0

# Stochastic Oscillator: 14-day %K and a 3-day smoothing for %D.
STOCH_K_WINDOW    = 14
STOCH_D_WINDOW    = 3

# yfinance download defaults: last 5 years of daily bars.
DEFAULT_PERIOD    = "5y"     # "5y" = 5 years back from today
DEFAULT_INTERVAL  = "1d"     # "1d" = one bar per trading day


# ======================================================================
# 1) DATA DOWNLOAD
# ======================================================================

def get_historical_data(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download historical daily OHLCV data for a Yahoo Finance ticker.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    Contacts Yahoo Finance and asks for up to ``period`` of ``interval``
    price/volume bars for a single symbol. This raw data is the base
    every other function in this module builds upon.

    ------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------
    ticker   : str — the Yahoo Finance symbol.
                   Examples: "AAPL", "MSFT", "GOOGL", "RELIANCE.NS",
                             "TCS.NS", "SUNPHARMA.BO"
    period   : str — how far back in time to look. Default "5y".
                   Common values: "1mo", "3mo", "6mo", "1y", "2y",
                                  "5y", "10y", "max".
    interval : str — the bar size. Default "1d" (one bar per day).
                   Common values: "1m", "5m", "1h", "1d", "1wk", "1mo".

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = get_historical_data("AAPL")
        # ticker="AAPL", period="5y", interval="1d"

    ------------------------------------------------------------------
    SAMPLE OUTPUT (shape of the returned DataFrame)
    ------------------------------------------------------------------
        Index   : DatetimeIndex of trading dates,
                  e.g. 2021-08-23 ... 2026-08-21 (~1240 rows).
        Columns : ['Open', 'High', 'Low', 'Close', 'Volume']

        A few illustrative rows (values are not real quotes):
                 Open    High    Low    Close     Volume
        2026-08-19  312.05  315.00  310.10  314.20   51200000
        2026-08-20  314.10  316.40  312.30  315.80   49800000
        2026-08-21  312.05  312.38  307.01  309.35   46768100

    ------------------------------------------------------------------
    NOTES
    ------------------------------------------------------------------
    * ``auto_adjust=True`` means Open/High/Low/Close are already adjusted
      for splits and dividends, so the 5-year history is comparable.
    * ``actions=False`` drops the Dividend/Split columns we don't need.
    * We use ``yf.Ticker(ticker).history(...)`` rather than
      ``yf.download(...)`` because the former always returns simple
      single-level column names, keeping the rest of the code clean.
    """
    # --- Call Yahoo Finance and fetch the requested history -------------
    # ``Ticker.history()`` returns a DataFrame indexed by trading date.
    hist = yf.Ticker(ticker).history(
        period=period,       # e.g. "5y" = last 5 years
        interval=interval,   # e.g. "1d" = daily bars
        auto_adjust=True,    # OHLC are split/dividend-adjusted
        actions=False,       # ignore dividend & split announcements
    )

    # --- Guard against a bad symbol or an empty result ------------------
    # If the ticker does not exist (or was delisted and has no data in the
    # requested window), yfinance returns an empty DataFrame. Raise a clear
    # error instead of letting confusing NaN columns flow downstream.
    if hist is None or hist.empty:
        raise ValueError(
            f"No historical data returned for ticker '{ticker}' "
            f"(period={period}, interval={interval}). "
            f"Check the symbol and try again."
        )

    # --- Defensive cleanup of column names -------------------------------
    # ``yf.download()`` can return a MultiIndex of columns when given a
    # list of tickers. We don't expect one here (we pass a plain string),
    # but if it ever appears we flatten it to the last level so the rest
    # of the module can always do df["Close"], df["Volume"], etc.
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(-1)

    # --- Hand the cleaned table back to the caller -----------------------
    return hist


# ======================================================================
# 2) MOVING AVERAGES (SMA & EMA) FOR PRICE AND VOLUME
# ======================================================================

def add_moving_averages(
    df: pd.DataFrame,
    windows: tuple = MA_WINDOWS,
) -> pd.DataFrame:
    """
    Add Simple (SMA) and Exponential (EMA) moving averages to ``df``.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    For every window ``n`` in ``windows`` four new columns are added:

      * ``SMA_n``     — Simple Moving Average of the *closing price*
      * ``EMA_n``     — Exponential Moving Average of the *closing price*
      * ``VOL_SMA_n`` — Simple Moving Average of the *volume*
      * ``VOL_EMA_n`` — Exponential Moving Average of the *volume*

    A moving average smooths out short-term noise so the underlying
    trend is visible. Short windows (5, 13) react quickly and suit
    short-term traders; long windows (90, 200) describe the longer-term
    trend and are watched closely by institutions.

    ------------------------------------------------------------------
    FORMULAS
    ------------------------------------------------------------------
    SMA_n(t) = (Close(t) + Close(t-1) + ... + Close(t-n+1)) / n
               i.e. the arithmetic mean of the last n closing prices.

    EMA_n(t) = alpha * Close(t) + (1 - alpha) * EMA_n(t-1)
               where alpha = 2 / (n + 1)
    (``adjust=False`` tells pandas to seed the EMA with the first value
    and use the classic recursive formula above — the standard definition
    used in technical analysis.)

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        # df already contains Open/High/Low/Close/Volume columns.
        df = add_moving_averages(df)      # windows=(5,13,23,90,200)

    ------------------------------------------------------------------
    SAMPLE OUTPUT (new columns added to df)
    ------------------------------------------------------------------
        SMA_5      : 5-day mean of Close,   e.g. 309.35, 310.20, ...
        EMA_5      : 5-day exponential mean of Close, e.g. 309.12, ...
        VOL_SMA_5  : 5-day mean of Volume,  e.g. 46768100, ...
        VOL_EMA_5  : 5-day exponential mean of Volume, ...
        ... and the same pattern for windows 13, 23, 90 and 200.

        NOTE: the first (n-1) rows of SMA_n are NaN because there is not
        enough history yet to form an n-day average.
    """
    for n in windows:
        # --- Price (Close) moving averages ------------------------------
        # .rolling(window=n, min_periods=n).mean() slides an n-day window
        # over the Close column and averages the values inside it.
        # min_periods=n keeps the first n-1 rows as NaN (not enough data).
        df[f"SMA_{n}"] = df["Close"].rolling(window=n, min_periods=n).mean()

        # .ewm(span=n, adjust=False).mean() computes the recursive EMA.
        # "span" is converted internally to alpha = 2/(n+1).
        df[f"EMA_{n}"] = df["Close"].ewm(span=n, adjust=False).mean()

        # --- Volume moving averages --------------------------------------
        # Same rolling / exponential logic, but applied to the Volume
        # column, so we can see whether recent volume is above or below
        # its historical average (volume spikes often precede big moves).
        df[f"VOL_SMA_{n}"] = df["Volume"].rolling(window=n, min_periods=n).mean()
        df[f"VOL_EMA_{n}"] = df["Volume"].ewm(span=n, adjust=False).mean()

    # --- Return the enriched DataFrame -----------------------------------
    # The caller gets back the same table with 20 new columns appended
    # (10 for price, 10 for volume).
    return df


# ======================================================================
# 3) RELATIVE STRENGTH INDEX (RSI)
# ======================================================================

def add_rsi(df: pd.DataFrame, period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Add a 14-day Relative Strength Index (Wilder's smoothing).

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    RSI measures the speed and magnitude of recent price changes to tell
    you whether a stock is "overbought" (RSI > 70, possibly due for a
    pullback) or "oversold" (RSI < 30, possibly due for a bounce). It
    oscillates between 0 and 100.

    ------------------------------------------------------------------
    FORMULA (per day, over the look-back ``period``)
    ------------------------------------------------------------------
        delta      = Close(t) - Close(t-1)
        gain       = delta where delta > 0, else 0
        loss       = -delta where delta < 0, else 0

        avg_gain   = Wilder's average of gain  (EMA with alpha = 1/period)
        avg_loss   = Wilder's average of loss  (EMA with alpha = 1/period)

        RS  = avg_gain / avg_loss
        RSI = 100 - (100 / (1 + RS))

    Interpretation:
        RSI > 70  -> overbought   (many strong up-days recently)
        RSI < 30  -> oversold     (many strong down-days recently)

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        # df["Close"] already holds daily closing prices.
        df = add_rsi(df)                       # period=14 (default)

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        RSI column (rounded):
            ... 61.3, 58.9, 62.4, 55.1, 47.4 ...
        A value of 47.4 means the stock is neither overbought nor
        oversold — roughly balanced buying/selling pressure.

        The first ``period`` rows are NaN (not enough history to smooth).
    """
    # --- Day-over-day change in closing price ----------------------------
    # delta[t] = Close[t] - Close[t-1]; the very first row is NaN.
    delta = df["Close"].diff()

    # --- Separate the moves into gains and losses ------------------------
    # clip(lower=0.0) keeps only positive values (gains), zeroing the rest.
    gain = delta.clip(lower=0.0)
    # Negating delta and clipping again isolates the negative moves,
    # stored as positive "loss" magnitudes.
    loss = (-delta).clip(lower=0.0)

    # --- Wilder's smoothing of gains and losses ---------------------------
    # Wilder's RSI uses an EMA with alpha = 1/period (not the 2/(n+1) used
    # for ordinary EMAs). min_periods=period keeps the first rows as NaN.
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # --- Relative Strength and the final 0..100 index ---------------------
    # RS = avg_gain / avg_loss. When avg_loss is 0 (nothing but up/flat
    # days), RS is infinite and RSI collapses to exactly 100.
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # --- Clean up the all-gain edge case -----------------------------------
    # rsi.where(avg_loss != 0, 100.0):
    #   * keeps the computed RSI wherever the average loss is non-zero;
    #   * substitutes 100.0 wherever the average loss is exactly 0
    #     (all up/flat days -> maximum strength);
    #   * where avg_loss is NaN (start of series), the condition evaluates
    #     to True, so the (NaN) RSI is kept — clearly marking the rows
    #     that have not been computed yet.
    rsi = rsi.where(avg_loss != 0, 100.0)

    # --- Store the result in a new column ----------------------------------
    df["RSI"] = rsi
    return df


# ======================================================================
# 4) MOVING AVERAGE CONVERGENCE / DIVERGENCE (MACD)
# ======================================================================

def add_macd(
    df: pd.DataFrame,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Add MACD line, signal line and histogram to ``df``.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    MACD reveals the strength, direction and momentum of a trend by
    comparing a fast EMA with a slow EMA. When the fast line crosses
    above the slow one, momentum is turning bullish; below, bearish.

    ------------------------------------------------------------------
    FORMULAS
    ------------------------------------------------------------------
        MACD        = EMA(fast) - EMA(slow)         e.g. EMA(12)-EMA(26)
        MACD_SIGNAL = EMA(signal) of the MACD line   e.g. EMA(9) of MACD
        MACD_HIST   = MACD - MACD_SIGNAL  (the histogram bars)

    Classic signals:
        * MACD crosses ABOVE MACD_SIGNAL -> bullish
        * MACD crosses BELOW MACD_SIGNAL -> bearish
        * MACD above the zero line       -> upward momentum
        * MACD below the zero line       -> downward momentum

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = add_macd(df)       # fast=12, slow=26, signal=9 (defaults)

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        MACD        : e.g. -1.50   (fast EMA below slow EMA -> bearish)
        MACD_SIGNAL : e.g. -1.13   (EMA(9) of the MACD line)
        MACD_HIST   : e.g. -0.37   (negative -> histogram below zero)

        All three columns are in the same units as the price.
    """
    # --- The two underlying EMAs -------------------------------------------
    # A "fast" EMA reacts quickly to new prices; a "slow" EMA lags behind.
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    # --- MACD line = fast minus slow ----------------------------------------
    # Positive when short-term prices sit above the longer-term average.
    macd_line = ema_fast - ema_slow

    # --- Signal line = smoothed MACD -----------------------------------------
    # A 9-day EMA of the MACD line acts as a trigger for buy/sell signals.
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()

    # --- Histogram = distance between MACD and its signal --------------------
    # Positive histogram = bullish momentum building; negative = bearish.
    macd_hist = macd_line - macd_signal

    # --- Write all three into the DataFrame ----------------------------------
    df["MACD"]        = macd_line
    df["MACD_SIGNAL"] = macd_signal
    df["MACD_HIST"]   = macd_hist
    return df


# ======================================================================
# 5) BOLLINGER BANDS
# ======================================================================

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = BB_WINDOW,
    num_std: float = BB_NUM_STD,
) -> pd.DataFrame:
    """
    Add Bollinger Bands to ``df``.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    Bollinger Bands draw a volatility channel around a moving average.
    The bands widen when volatility rises and contract when it falls.
    Price touching the upper band suggests the stock is "expensive"
    relative to its recent range; the lower band suggests "cheap".

    ------------------------------------------------------------------
    FORMULAS
    ------------------------------------------------------------------
        BB_MIDDLE = SMA(window) of Close                (default 20)
        BB_UPPER  = BB_MIDDLE + num_std * std(Close)    (default +2σ)
        BB_LOWER  = BB_MIDDLE - num_std * std(Close)    (default -2σ)

    ``ddof=0`` (population standard deviation) is used, which matches
    the definition John Bollinger originally published.

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = add_bollinger_bands(df)    # window=20, num_std=2.0

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        BB_MIDDLE : e.g. 314.20   (20-day average close)
        BB_UPPER  : e.g. 337.94   (mid + 2 standard deviations)
        BB_LOWER  : e.g. 290.46   (mid - 2 standard deviations)

        A close of 309.35 sits between the bands — the price is inside
        the normal volatility range.
    """
    # --- 20-day rolling mean (the middle band) --------------------------------
    mid = df["Close"].rolling(window=window, min_periods=window).mean()

    # --- 20-day rolling standard deviation (ddof=0 -> population sigma) -------
    std = df["Close"].rolling(window=window, min_periods=window).std(ddof=0)

    # --- Upper and lower bands = middle +/- num_std * sigma -------------------
    df["BB_UPPER"]  = mid + num_std * std
    df["BB_MIDDLE"] = mid
    df["BB_LOWER"]  = mid - num_std * std
    return df


# ======================================================================
# 6) STOCHASTIC OSCILLATOR
# ======================================================================

def add_stochastic_oscillator(
    df: pd.DataFrame,
    k_window: int = STOCH_K_WINDOW,
    d_window: int = STOCH_D_WINDOW,
) -> pd.DataFrame:
    """
    Add the Stochastic Oscillator to ``df``.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    The Stochastic Oscillator compares where the closing price sits
    *within* the recent high-low range, i.e. it measures the strength of
    the current momentum relative to the price extremes.

    ------------------------------------------------------------------
    FORMULAS
    ------------------------------------------------------------------
        HighestHigh = max(High) over the last k_window days  (default 14)
        LowestLow   = min(Low)  over the last k_window days  (default 14)

        STOCH_K = 100 * (Close - LowestLow) / (HighestHigh - LowestLow)
        STOCH_D = SMA(d_window) of STOCH_K                   (default 3)

    Interpretation (values are bounded 0..100):
        * STOCH_K > 80 -> overbought  (close near the top of the range)
        * STOCH_K < 20 -> oversold    (close near the bottom of the range)
        * A %K / %D cross can be used as a momentum signal.

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = add_stochastic_oscillator(df)   # k_window=14, d_window=3

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        STOCH_K : e.g. 44.55   (close is 44.5% up into the 14-day range)
        STOCH_D : e.g. 62.15   (3-day average of %K)
    """
    # --- Rolling 14-day extremes of the price range ---------------------------
    lowest_low   = df["Low"].rolling(window=k_window, min_periods=k_window).min()
    highest_high = df["High"].rolling(window=k_window, min_periods=k_window).max()

    # --- Height of the 14-day range (the denominator of the formula) ----------
    price_range = highest_high - lowest_low

    # --- %K: where the close sits inside that range (scaled to 0..100) --------
    stoch_k = 100.0 * (df["Close"] - lowest_low) / price_range

    # --- %D: 3-day simple average of %K (the "signal" line) -------------------
    stoch_d = stoch_k.rolling(window=d_window, min_periods=d_window).mean()

    # --- Defensive: if high == low the range is 0 (division by zero) ----------
    # Substituting 50.0 (a neutral reading) avoids NaN/Inf in flat markets.
    stoch_k = stoch_k.where(price_range != 0, 50.0)

    # --- Store both columns ----------------------------------------------------
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_d
    return df


# ======================================================================
# 7) ON-BALANCE VOLUME (OBV)
# ======================================================================

def add_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add On-Balance Volume to ``df``.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    OBV is a running total of volume that ADDS volume on up-days and
    SUBTRACTS it on down-days. It links price and volume: if price makes
    a new high but OBV does not, the rally may not be backed by buyers
    (a "divergence" warning).

    ------------------------------------------------------------------
    RULES
    ------------------------------------------------------------------
        if Close(t) > Close(t-1):  OBV(t) = OBV(t-1) + Volume(t)
        if Close(t) < Close(t-1):  OBV(t) = OBV(t-1) - Volume(t)
        if Close(t) == Close(t-1): OBV(t) = OBV(t-1)

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = add_obv(df)

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        OBV : e.g. 3,026,156,000   (cumulative volume flow since day 1).
        A rising OBV line = buying pressure dominates; falling = selling.
    """
    # --- Direction of the close vs. the previous day --------------------------
    # diff() gives the day-over-day change; np.sign maps it to +1 (up),
    # -1 (down) or 0 (flat). fillna(0.0) makes the very first row neutral
    # so the cumulative sum starts cleanly from zero.
    price_direction = np.sign(df["Close"].diff()).fillna(0.0)

    # --- Weight each day's volume by its direction, then accumulate -----------
    # +Volume on up days, -Volume on down days, 0 on flat days, all summed.
    df["OBV"] = (price_direction * df["Volume"]).cumsum()
    return df


# ======================================================================
# 8) ORCHESTRATOR — THE MAIN ENTRY POINT
# ======================================================================

def compute_technical_indicators(
    ticker: str,
    period: str = DEFAULT_PERIOD,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """
    Download 5 years of daily OHLCV data for ``ticker`` and compute the
    full set of technical indicators described at the top of this module.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    This is the one function you normally call. It runs the data
    download and then chains every indicator helper onto the table,
    so a single call returns a ready-to-analyse DataFrame.

    ------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------
    ticker   : str — Yahoo Finance symbol.
                   Examples: "AAPL", "MSFT", "RELIANCE.NS", "TCS.NS".
    period   : str — look-back period (default "5y" = last 5 years).
    interval : str — bar size (default "1d" = daily bars).

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = compute_technical_indicators("AAPL")
        # Downloads AAPL daily data for the last 5 years and computes
        # all moving averages and indicators in one call.

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        A DataFrame with 35 columns and ~1240 rows (one row per trading
        day over 5 years). Columns:

          Open High Low Close Volume              (5 raw OHLCV columns)
          SMA_5 ... SMA_200 EMA_5 ... EMA_200     (10 price MA columns)
          VOL_SMA_5 ... VOL_EMA_200               (10 volume MA columns)
          RSI                                      (1 column)
          MACD MACD_SIGNAL MACD_HIST               (3 columns)
          BB_UPPER BB_MIDDLE BB_LOWER              (3 columns)
          STOCH_K STOCH_D                          (2 columns)
          OBV                                      (1 column)

        Example last row (illustrative values):
                 Close     SMA_5   RSI   MACD  STOCH_K      OBV
        2026-08-21  309.35  310.62  47.4  -1.50    44.55  3.03e9
    """
    # --- Step 1: download the raw OHLCV history -------------------------------
    df = get_historical_data(ticker, period=period, interval=interval)

    # --- Step 2: add moving averages (price + volume, 5/13/23/90/200) ---------
    add_moving_averages(df)

    # --- Step 3: add the five technical indicators -----------------------------
    add_rsi(df)                      # Relative Strength Index
    add_macd(df)                     # MACD line, signal, histogram
    add_bollinger_bands(df)          # Volatility bands around SMA(20)
    add_stochastic_oscillator(df)    # %K and %D momentum oscillator
    add_obv(df)                      # On-Balance Volume

    # --- Return the fully enriched table ---------------------------------------
    return df


# ======================================================================
# 9) CONVENIENCE HELPERS
# ======================================================================

def show_latest(df: pd.DataFrame, ticker: str = None, rows: int = 1) -> None:
    """
    Pretty-print the most recent ``rows`` rows of computed indicators.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    A quick way to eyeball the freshest indicator values without
    scrolling through the whole DataFrame. It prints a titled block
    with the DataFrame transposed, so every indicator appears on its
    own line.

    ------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------
    df     : pandas.DataFrame — output of ``compute_technical_indicators``.
    ticker : str — optional symbol used in the printed title (None default).
    rows   : int — how many of the most recent rows to print (default 1).

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        df = compute_technical_indicators("AAPL")
        show_latest(df, ticker="AAPL")         # rows=1 (default)

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        ==================================
        Latest Technical Indicators — AAPL
        ==================================
        Date         2026-08-21
        Open              312.05
        High              312.38
        ...
        RSI                 47.4
        MACD                -1.50
        ...
        OBV            3.026e+09
    """
    # --- Build the header text (with or without the ticker label) ---------------
    title = f"Latest Technical Indicators — {ticker}" if ticker else \
        "Latest Technical Indicators"

    # --- Print a decorative banner around the title -----------------------------
    print("=" * len(title))
    print(title)
    print("=" * len(title))

    # --- Transpose so rows become columns and columns become rows ---------------
    # df.tail(rows) keeps the last ``rows`` trading days; .T flips the table so
    # each indicator name appears on its own line (easier to read in a console).
    print(df.tail(rows).T.to_string())
    print()


def main() -> None:
    """
    Command-line entry point.

    ------------------------------------------------------------------
    PURPOSE
    ------------------------------------------------------------------
    Lets you run this module directly from a terminal without writing
    any Python code:

        python TECH_ANALYSIS.py AAPL
        python TECH_ANALYSIS.py RELIANCE.NS --rows 3

    ------------------------------------------------------------------
    SAMPLE INPUT
    ------------------------------------------------------------------
        python TECH_ANALYSIS.py AAPL

    ------------------------------------------------------------------
    SAMPLE OUTPUT
    ------------------------------------------------------------------
        ==================================
        Latest Technical Indicators — AAPL
        ==================================
        Date         2026-08-21 00:00:00-04:00
        Open                      3.120500e+02
        High                      3.123800e+02
        ... (every indicator column) ...
        OBV                       3.026156e+09
    """
    # --- argparse is imported locally so importing the module does not ----
    # need to load the CLI machinery every time.
    import argparse

    # --- Build the argument parser ----------------------------------------
    parser = argparse.ArgumentParser(
        description="Download 5y of daily OHLCV data for a Yahoo Finance "
                    "ticker and compute moving averages + technical indicators."
    )
    # Positional argument: the ticker symbol (required).
    parser.add_argument(
        "ticker",
        type=str,
        help="Yahoo Finance symbol, e.g. AAPL, MSFT, RELIANCE.NS",
    )
    # Optional argument: how many recent rows to display.
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of latest rows to print (default 1).",
    )

    # --- Parse the command line and run the pipeline ------------------------
    args = parser.parse_args()        # e.g. args.ticker="AAPL", args.rows=1

    # Download + compute every indicator for the requested symbol.
    df = compute_technical_indicators(args.ticker)

    # Print the latest snapshot(s).
    show_latest(df, ticker=args.ticker, rows=args.rows)


# ======================================================================
# 10) SCRIPT GUARD
# ======================================================================
# This block only runs when the file is executed directly, e.g.
#     python TECH_ANALYSIS.py AAPL
# When the file is *imported* elsewhere, e.g.
#     import TECH_ANALYSIS as ta
# then __name__ is "TECH_ANALYSIS" (not "__main__") and the CLI is NOT
# triggered — exactly what we want for a reusable library module.
if __name__ == "__main__":
    main()

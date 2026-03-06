"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Market_Analysis_App
streamlit run Market_Analysis.py
conda activate np_env
conda deactivate
"""

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

# PyTorch 2.6+ compatibility: allow neuralprophet classes in torch.load
try:
    import torch
    try:
        import neuralprophet.configure
        torch.serialization.add_safe_globals([neuralprophet.configure.ConfigSeasonality,
                                               neuralprophet.configure.ConfigLagLength,
                                               neuralprophet.configure.ConfigAutoregression])
    except (ImportError, AttributeError):
        pass
except ImportError:
    pass

st.set_page_config(layout="wide",
                   page_title="Nifty Gold BEES EMA Analysis",
                   page_icon="🚀"
                   )

sb = st.sidebar
sb.title("Forecasting Statistics")
# optional optimization settings
run_opt = sb.checkbox("Run lag/quantile optimization", value=False,
                      help="Check to search over predefined lags and quantiles.")
# user-settable hyperparameters (used either directly or during optimization)
interval = sb.slider("Prediction interval width", 0.5, 1.0, 0.95, step=0.01)
lookback = sb.number_input("History lookback (months)", min_value=0, max_value=12, value=3, step=1,
                           help="Only include this many months of recent history; 0 disables filtering.")
weight_factor = sb.slider("Recent weight factor", 1.0, 20.0, 5.0, step=0.5,
                          help="Multiplier applied to points within the lookback window to give them higher weight.")
ar_lags = sb.number_input("Autoregressive lags", min_value=0, max_value=5, value=1, step=1,
                           help="Number of lagged `y` terms to include as regressors.")
uncertainty_samples = sb.number_input("Uncertainty samples", min_value=0, max_value=5000, value=1000, step=100,
                                      help="Number of Monte Carlo samples for uncertainty intervals.")
# lst_days = [13, 21, 34, 55, 89]
lst_days = [3, 5, 8, 13, 21]
# Main code for analysis and visualization of nifty 50 data
# using streamlit and plotly
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f0f8e5; /* Light olive green color */
    }
    /* Change the main content background color */
    [data-testid="stAppViewContainer"] {
        background-color: #fffdf0; /* Light gray color */
    }
    /* Set a complementary color for the titles and headers */
    h1, h2, h3 {
        color: #787355; /* A warm, dark gray for readability */
    }
    /* Customize the sidebar header */
    .css-1jc7h9d, .e1ewe9a51, .e1ewe9a52 { /* These are Streamlit's generated CSS classes for the header */
        background-color: #008080; /* Your desired color, e.g., Teal */
        color: white; /* Change the text color for better contrast */
    }
    [data-testid="stToolbar"] {
        background-color: #cab161; /* golden background */
        padding: 10px;
        border-radius: 5px;
    }
    [data-testid="stToolbar"] a {
        color: #927748; /* deep gold text */
        text-decoration: none;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Apply style directly to the title using a markdown h1 tag with an inline style
st.markdown("<h2 style='color: #787355;'>Using Neural Prophet to Forecast Future Price </h2>", unsafe_allow_html=True)

def create_dataframe():
    """
    Create a DataFrame with EMA calculations for Nifty Gold BEES.
    """
    ticker = "GOLDBEES.NS"
    today = datetime.today()
    start_date = today - timedelta(days=365*3)  # Last 3 years
    str_end_date = today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    df_xau = yf.download(ticker, start=str_start_date, end=str_end_date)
    df_xau.columns = df_xau.columns.get_level_values(0)
    df_xau.reset_index(inplace=True)
    df_xau['Date'] = pd.to_datetime(df_xau['Date']).dt.strftime('%d-%m-%Y')
    st.session_state['df_xau_np'] = df_xau
    return df_xau


def forecast_log_returns_neural_prophet(df, months=6,
                                          interval_width=0.95,
                                          lookback_months=0,
                                          recent_weight=1.0,
                                          ar_lags=0,
                                          uncertainty_samples=1000):
    """
    Forecast `Log_returns` for the next `months` months using Prophet.
    Provides options for weighting recent data, AR lags and uncertainty sampling.
    """
    df2 = df.copy()
    # parse Date which is stored as dd-mm-yyyy string in this module
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y', errors='coerce')
    # ensure Log_returns exists
    if 'Log_returns' not in df2.columns:
        df2['Log_returns'] = np.log(df2['Close'] / df2['Close'].shift(1))
    df2 = df2.dropna(subset=['Date', 'Log_returns'])

    df_prop = df2[['Date', 'Log_returns']].rename(columns={'Date': 'ds', 'Log_returns': 'y'})

    # restrict history and compute weights
    if lookback_months and lookback_months > 0:
        cutoff = df_prop['ds'].max() - pd.DateOffset(months=lookback_months)
        df_prop = df_prop[df_prop['ds'] >= cutoff]
        if recent_weight and recent_weight != 1.0:
            df_prop['weight'] = recent_weight

    # add AR regressors
    if ar_lags and ar_lags > 0:
        for i in range(1, ar_lags + 1):
            df_prop[f'y_lag{i}'] = df_prop['y'].shift(i)
        df_prop.dropna(inplace=True)

    # approximate business days for the requested months (21 trading days/month)
    periods = int(months * 21)

    try:
        # Use Prophet for time series forecasting
        m = Prophet(interval_width=interval_width,
                    uncertainty_samples=uncertainty_samples)
        for i in range(1, ar_lags + 1):
            m.add_regressor(f'y_lag{i}')
        m.fit(df_prop)
        # Create future dataframe with business days only
        last_date = df_prop['ds'].max()
        future_dates = pd.bdate_range(start=last_date, periods=periods+1, freq='B')[1:]
        future = pd.DataFrame({'ds': future_dates})
        if ar_lags and ar_lags > 0:
            for i in range(1, ar_lags + 1):
                # carry last known value forward
                future[f'y_lag{i}'] = df_prop[f'y_lag{i}'].iloc[-1]
        forecast = m.predict(future)
        
        # Extract forecasted rows including uncertainty bounds if present
        cols = ['ds', 'yhat']
        if 'yhat_lower' in forecast.columns and 'yhat_upper' in forecast.columns:
            cols += ['yhat_lower', 'yhat_upper']
        res = forecast[cols].reset_index(drop=True)
        
    except Exception as e:
        st.write(f"Prophet forecast error: {e}. Returning simple exponential smoothing.")
        # Fallback: simple exponential smoothing
        res = pd.DataFrame({'ds': pd.bdate_range(start=df2['Date'].max(), periods=periods, freq='B')})
        last_value = df2['Log_returns'].iloc[-1]
        res['yhat'] = last_value  # flat forecast
        # set bounds equal to the point forecast when Prophet not available
        res['yhat_lower'] = res['yhat']
        res['yhat_upper'] = res['yhat']

    res['ds_str'] = pd.to_datetime(res['ds']).dt.strftime('%d-%m-%Y')
    return res

df_xau = pd.DataFrame
if 'df_xau_np' not in st.session_state:
    df_xau = create_dataframe()
else:
    df_xau = st.session_state['df_xau_np']
    df_xau['Log_returns'] = np.log(df_xau['Close'] / df_xau['Close'].shift(1))
    df_xau.dropna(inplace=True)

# st.dataframe(df_xau.head(5), use_container_width=True)
# st.dataframe(df_xau.tail(5), use_container_width=True)   
if run_opt:
    # perform grid search over predetermined lags, lookback windows and quantile sets
    def optimize_lags_quantiles(df, months=6):
        lag_options = [3,4,5,6,7,8,9,10]
        lookback_options = [3, 5, 7]
        quantile_pairs = [(0.05,0.95),(0.01,0.99),(0.02,0.98),(0.03,0.97),(0.04,0.96)]
        best = None
        best_score = -np.inf
        # create holdout set (last 21 business days)
        df2 = df.copy()
        df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y', errors='coerce')
        df2['Log_returns'] = np.log(df2['Close']/df2['Close'].shift(1))
        df2 = df2.dropna(subset=['Date','Log_returns'])
        if len(df2) <= 21:
            st.warning("Not enough data to run optimization (need >21 business days).")
            return None
        train = df2.iloc[:-21]
        test = df2.iloc[-21:]
        actual = np.log(test['Close']/test['Close'].shift(1)).dropna().values

        total = len(lag_options) * len(lookback_options) * len(quantile_pairs)
        count = 0
        progress = st.sidebar.progress(0)
        prog_text = st.sidebar.empty()

        for lag in lag_options:
            for look in lookback_options:
                for (low,high) in quantile_pairs:
                    width = high - low
                    # run trial using specified lookback and lag
                    feat = forecast_log_returns_neural_prophet(train, months=1,
                                                               interval_width=width,
                                                               lookback_months=look,
                                                               recent_weight=weight_factor,
                                                               ar_lags=lag,
                                                               uncertainty_samples=uncertainty_samples)
                    y_pred = feat['yhat'].values[:len(actual)]
                    if len(y_pred) == len(actual):
                        # compute Pearson correlation as the optimization metric
                        if len(actual) > 1 and np.std(y_pred) > 0 and np.std(actual) > 0:
                            r = np.corrcoef(actual, y_pred)[0,1]
                        else:
                            r = 0.0
                        score = r  # use correlation directly
                        if score > best_score:
                            best_score = score
                            best = (lag, look, (low,high), score)

                    count += 1
                    pct = int(100 * count / total)
                    try:
                        progress.progress(pct)
                        prog_text.text(f"Optimization progress: {pct}% ({count}/{total})")
                    except Exception:
                        pass

        # clear progress widgets after done
        progress.empty()
        prog_text.empty()
        return best

    opt_res = optimize_lags_quantiles(df_xau, months=6)
    if opt_res:
        lag_best, look_best, quantile_best, score_best = opt_res
        st.write(f"Best configuration: lag={lag_best}, lookback={look_best} months, quantiles={quantile_best}, r2={score_best:.4f}")
        # override forecasting parameters with best combination
        interval = quantile_best[1] - quantile_best[0]
        lookback = look_best
        ar_lags = lag_best

# use chosen parameters (either sidebar or optimized) for forecast
forecast_df = forecast_log_returns_neural_prophet(
    df_xau,
    months=6,
    interval_width=interval,
    lookback_months=lookback,
    recent_weight=weight_factor,
    ar_lags=ar_lags,
    uncertainty_samples=uncertainty_samples,
)
st.subheader("Forecasted Log Returns for the Next 6 Months")
# st.dataframe(forecast_df.head(30), use_container_width=True)
# st.dataframe(forecast_df.tail(30), use_container_width=True)

# Compute predicted future prices by applying forecasted log-returns
if not forecast_df.empty:
    # ensure chronological order
    forecast_df = forecast_df.sort_values('ds').reset_index(drop=True)
    # latest observed close price
    last_price = df_xau['Close'].iloc[-1]
    # iterative application of log returns: each predicted price uses the previous predicted price
    predicted = []
    predicted_lower = []
    predicted_upper = []
    price = float(last_price)
    price_low = float(last_price)
    price_up = float(last_price)
    # use .get to safely access bounds if missing
    yhat_vals = forecast_df['yhat'].values
    yhat_lower_vals = forecast_df.get('yhat_lower', pd.Series(yhat_vals)).values
    yhat_upper_vals = forecast_df.get('yhat_upper', pd.Series(yhat_vals)).values
    for r, rl, ru in zip(yhat_vals, yhat_lower_vals, yhat_upper_vals):
        price = price * np.exp(r)
        price_low = price_low * np.exp(rl)
        price_up = price_up * np.exp(ru)
        predicted.append(price)
        predicted_lower.append(price_low)
        predicted_upper.append(price_up)
    forecast_df['predicted_price'] = np.array(predicted).round(4)
    forecast_df['predicted_price_lower'] = np.array(predicted_lower).round(4)
    forecast_df['predicted_price_upper'] = np.array(predicted_upper).round(4)
    # show predicted prices
    # st.subheader("Predicted Future Prices Based on Forecasted Log Returns")
    st.dataframe(forecast_df[['ds_str', 'yhat', 'predicted_price']].head(50), use_container_width=True)
    # plot predicted price path
    try:
        ds = pd.to_datetime(forecast_df['ds'])
        mid = forecast_df['predicted_price']

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ds, y=mid, mode='lines', line=dict(color='rgb(34,94,168)', width=2), name='Predicted Price'))

        fig.update_layout(title='Predicted Future Price', xaxis_title='Date', yaxis_title='Predicted Price')
        # use linear y-axis
        fig.update_yaxes(type='linear')
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        # fallback to a simple line chart
        try:
            fig2 = px.line(forecast_df, x='ds', y='predicted_price', title='Predicted Future Price')
            fig2.update_layout(xaxis_title='Date', yaxis_title='Predicted Price')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.line_chart(forecast_df.set_index('ds')['predicted_price'])


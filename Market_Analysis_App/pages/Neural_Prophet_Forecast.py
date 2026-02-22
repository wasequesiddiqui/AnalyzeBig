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


def forecast_log_returns_neural_prophet(df, months=6):
    """
    Forecast `Log_returns` for the next `months` months using Prophet.
    Falls back to simple ARIMA-like approach if Prophet fails.

    Returns a DataFrame with forecasted dates and predictions.
    """
    df2 = df.copy()
    # parse Date which is stored as dd-mm-yyyy string in this module
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y', errors='coerce')
    # ensure Log_returns exists
    if 'Log_returns' not in df2.columns:
        df2['Log_returns'] = np.log(df2['Close'] / df2['Close'].shift(1))
    df2 = df2.dropna(subset=['Date', 'Log_returns'])

    df_prop = df2[['Date', 'Log_returns']].rename(columns={'Date': 'ds', 'Log_returns': 'y'})

    # approximate business days for the requested months (21 trading days/month)
    periods = int(months * 21)

    try:
        # Use Prophet for time series forecasting
        m = Prophet(interval_width=0.95)
        m.fit(df_prop)
        # Create future dataframe with business days only
        last_date = df_prop['ds'].max()
        future_dates = pd.bdate_range(start=last_date, periods=periods+1, freq='B')[1:]
        future = pd.DataFrame({'ds': future_dates})
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
forecast_df = forecast_log_returns_neural_prophet(df_xau, months=6)
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


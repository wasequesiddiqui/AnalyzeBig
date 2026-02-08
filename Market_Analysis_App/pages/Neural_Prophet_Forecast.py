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
    Forecast `Log_returns` for the next `months` months using NeuralProphet.
    Falls back to Prophet if `neuralprophet` is not installed.

    Returns a DataFrame with forecasted dates and predictions.
    """
    # local import to avoid hard dependency at module import time
    try:
        from neuralprophet import NeuralProphet
        use_neural = True
    except Exception:
        from prophet import Prophet
        use_neural = False

    df2 = df.copy()
    # parse Date which is stored as dd-mm-yyyy string in this module
    df2['Date'] = pd.to_datetime(df2['Date'], format='%d-%m-%Y', errors='coerce')
    # ensure Log_returns exists
    if 'Log_returns' not in df2.columns:
        df2['Log_returns'] = np.log(df2['Close'] / df2['Close'].shift(1))
    df2 = df2.dropna(subset=['Date', 'Log_returns'])

    df_prop = df2[['Date', 'Log_returns']].rename(columns={'Date': 'ds', 'Log_returns': 'y'})

    # approximate days for the requested months
    periods = int(months * 30)

    if use_neural:
        m = NeuralProphet()
        m.fit(df_prop, freq='D')
        future = m.make_future_dataframe(df_prop, periods=periods)
        forecast = m.predict(future)
        # neuralprophet typically outputs `yhat1` for the first prediction
        yhat_col = 'yhat1' if 'yhat1' in forecast.columns else 'yhat'
        res = forecast[['ds', yhat_col]].rename(columns={yhat_col: 'yhat'})
    else:
        m = Prophet()
        m.fit(df_prop)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        res = forecast[['ds', 'yhat']]

    res['ds_str'] = pd.to_datetime(res['ds']).dt.strftime('%d-%m-%Y')
    return res

if 'df_xau_np' not in st.session_state:
    df_xau = create_dataframe()
else:
    df_xau = st.session_state['df_xau_np']
    df_xau['Log_returns'] = np.log(df_xau['Close'] / df_xau['Close'].shift(1))
    df_xau.dropna(inplace=True)
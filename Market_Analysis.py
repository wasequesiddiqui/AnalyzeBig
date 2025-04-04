"""all the imports are included here"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf 

# from imports are declared here

from datetime import datetime, timedelta


# Main code for analysis and visualization of nifty 50 data
# using streamlit and plotly

st.title("Market Analysis Streamlit App")
df_nifty50 = pd.DataFrame()
df_xau = pd.DataFrame()

str_start_date = ""
str_end_date= ""

def get_default_dates(days):
    # Get today's date
    today = datetime.today()

    # Calculate a start date (e.g., 30 days ago)
    start_date = today - timedelta(days=days)
    str_end_date =today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    return str_start_date, str_end_date

def get_ticker_data(ticker, start_date, end_date):
    # Placeholder for actual data retrieval logic
    # For now, just returning an empty DataFrame
    df = yf.download(ticker, interval="1d", start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    return df

str_start_date, str_end_date = get_default_dates(1095)
st.write("Default Start Date: ", str_start_date)
st.write("Default End Date: ", str_end_date)
st.divider()
df_nifty50 = get_ticker_data("^NSEI", str_start_date, str_end_date)
df_xau = get_ticker_data("GC=F", str_start_date, str_end_date)
st.dataframe(df_nifty50)
st.dataframe(df_xau)
st.divider()
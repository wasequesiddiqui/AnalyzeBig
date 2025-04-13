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
    df['52_MA'] = df['Close'].rolling(window=52).mean()
    df['52_MA_VOL'] = df['Volume'].rolling(window=52).mean()
    df["Log_Ret_Price"] = round(np.log(df["Close"] / df["Close"].shift(1)),4)
    df["Log_Ret_Volume"] = round(np.log(df["Volume"] / df["Volume"].shift(1)),4)
    df.fillna(0, inplace=True)
    df.iloc[:,:] = df.iloc[:,:].astype(float)
    df.iloc[:, :-2] = df.iloc[:, :-2].round(0)
    df = df.sort_values(by=['Date'], ascending=False)
    df.index = df.index.date
    return df

def color_returns(value):
    if value < 0.0000:
        return 'color: #db3c02;'
    else:
        return 'color: #00c90a;'

str_start_date, str_end_date = get_default_dates(1095)
st.write("Default Start Date: ", str_start_date)
st.write("Default End Date: ", str_end_date)
st.divider()
st.header("Select the date range for analysis")
start_date = st.date_input("Start Date", min_value=datetime(2020, 1, 1), max_value=datetime.today(), value=datetime(2020, 1, 1))
end_date = st.date_input("End Date", min_value=datetime(2020, 1, 1), max_value=datetime.today(), value=datetime.today())
btn_refresh = st.button("Refresh Data", key="refresh")
if btn_refresh:
    str_start_date = start_date.strftime('%Y-%m-%d')
    str_end_date = end_date.strftime('%Y-%m-%d')
    st.write("Start Date: ", str_start_date)
    st.write("End Date: ", str_end_date)
df_nifty50 = get_ticker_data("^NSEI", str_start_date, str_end_date)
if end_date < start_date:
    st.error("End date must be after start date.")
    st.stop()
else:
    df_xau = get_ticker_data("GC=F", str_start_date, str_end_date)
    styled_df_nifty = df_nifty50.style.applymap(color_returns, subset=['Log_Ret_Price','Log_Ret_Volume'])
    styled_df_xau = df_xau.style.applymap(color_returns, subset=['Log_Ret_Price','Log_Ret_Volume'])
    st.dataframe(styled_df_nifty, width=1200, height=500)
    st.dataframe(styled_df_xau,width=1200, height=500)
st.divider()
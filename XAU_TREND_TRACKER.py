import yfinance as yf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import prophet as pf

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def get_dates(day_delta=1826):
    """
    Get the start and end dates for the data retrieval.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=day_delta)
    return start_date, end_date

def get_date_string(date):
    """
    Convert a datetime object to a string in the format YYYY-MM-DD.
    """
    return date.strftime("%Y-%m-%d")

def get_ticker_data(ticker, start_date, end_date):
    """
    Retrieve historical data for a given ticker symbol.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    df.columns = df.columns.get_level_values(0)
    # df.reset_index(inplace=True)
    return df

def get_log_returns(df, col_name):
    """
    Calculate log returns for a given column in the DataFrame.
    """
    df[f"Log_Ret_{col_name}"] = np.log(df[col_name] / df[col_name].shift(1))
    df[f"Log_Ret_{col_name}"] = df[f"Log_Ret_{col_name}"].fillna(0)
    df[f"Log_Ret_{col_name}"] = df[f"Log_Ret_{col_name}"].round(4)
    return df

def set_df_datatype(df):
    """
    Set the data types of the DataFrame columns.
    """
    df.fillna(0, inplace=True)
    df = df.sort_index(ascending=False)
    df['Log_Ret_Close'] = df['Log_Ret_Close'].astype(float)
    df['Log_Ret_Volume'] = df['Log_Ret_Volume'].astype(float)
    df['Date_Val'] = df.index
    return df

def set_df_prophet(df,ds,y):
    """
    Set the DataFrame for Prophet model.
    """
    df = df.rename(columns={ds: 'ds', y: 'y'})
    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)
    return df

# main program script here
# print date
str_start_date,str_end_date = get_dates()
str_start_date = get_date_string(str_start_date)
str_end_date = get_date_string(str_end_date)

print("Default Dates:")
print(f"Today: {str_end_date}")
print(f"Start Date (5 Years ago): {str_start_date}")

df_XAU = get_ticker_data("GOLDBEES.NS", str_start_date, str_end_date)
# df_XAU.set_index("Price Date", inplace=True)
df_XAU = get_log_returns(df_XAU, "Close")
df_XAU = get_log_returns(df_XAU, "Volume")
df_XAU = set_df_datatype(df_XAU)
print(df_XAU.head(5))

df_latest2Months = df_XAU[:60]
df_analysis = df_XAU[60:]

print("Sample Latest 2 Months Data:")
print(df_latest2Months.head(5))
print("Sample Analysis Data:")
print(df_analysis.head(5))

last_close_value = df_analysis["Close"].iloc[0]
last_volume_value = df_analysis["Volume"].iloc[0]
print("Last Close Value:", last_close_value)
print("Last Volume Value:", last_volume_value)

df_latest2Months = df_latest2Months.sort_index(ascending=True)
df_analysis = df_analysis.sort_index(ascending=True)

print("Sample Latest 2 Months Data Sorted Ascending:")
print(df_latest2Months.head(5))
print("Sample Analysis Data Sorted Ascending:")
print(df_analysis.head(5))

df_latest2Months_close = set_df_prophet(df_latest2Months, 'Date_Val', 'Log_Ret_Close')
df_latest2Months_vol = set_df_prophet(df_latest2Months, 'Date_Val', 'Log_Ret_Volume')
df_analysis_close = set_df_prophet(df_analysis, 'Date_Val', 'Log_Ret_Close')
df_analysis_vol = set_df_prophet(df_analysis, 'Date_Val', 'Log_Ret_Volume')
print("Sample Latest 2 Months Data for Close:")
print(df_latest2Months_close.head(5))
print("Sample Latest 2 Months Data for Vol:")
print(df_latest2Months_vol.head(5))
print("Sample Analysis Data for Close:")
print(df_analysis_close.head(5))
print("Sample Analysis Data for Vol:")
print(df_analysis_vol.head(5))
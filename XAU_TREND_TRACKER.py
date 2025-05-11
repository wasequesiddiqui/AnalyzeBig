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

# main program script here
# print date
str_start_date,str_end_date = get_dates()
str_start_date = get_date_string(str_start_date)
str_end_date = get_date_string(str_end_date)

print("Default Dates:")
print(f"Today: {str_end_date}")
print(f"Start Date (5 Years ago): {str_start_date}")
"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Equity_Finder
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
import EQUITY_ANALYSER as ea

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

sb = st.sidebar
sb.title("Evaluating the Ticker:")

# Textbox to capture ticker value
ticker = sb.text_input("Enter Ticker Symbol:", value="INFY.NS", placeholder="e.g., INFY.NS")

# Button to trigger evaluation
if sb.button("Evaluate"):
    st.header(f"Evaluating Ticker: {ticker}")
    try:
        st.dataframe(ea.evaluate_ticker(ticker))
    except Exception as e:
        st.error(f"An error occurred: {e}")
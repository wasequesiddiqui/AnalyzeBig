"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Market_Analysis_App
streamlit run Market_Analysis.py
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
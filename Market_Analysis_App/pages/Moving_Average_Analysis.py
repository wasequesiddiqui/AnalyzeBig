"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Market_Analysis_App
streamlit run Market_Analysis.py
"""
import yfinance as yf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import Market_Analysis as ma

from prophet import Prophet
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(layout="wide",
                   page_title="Nifty Gold BEES EMA Analysis",
                   page_icon="🚀"
                   )
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
st.markdown("<h2 style='color: #787355;'>Analytics on Nifty Gold BEES & Exponential Moving Average </h2>", unsafe_allow_html=True)

def plot_ema_analysis(df):
    """
    Plots the EMA analysis results with a bar graph for Absolute Profit and a line graph for Percentage Profit.

    Args:
        df (pd.DataFrame): DataFrame with 'EMA_Days', 'Absolute_Profit', and 'Percentage_Profit' columns.
    """

    # Create the figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define color scale for Absolute Profit (bar graph)
    abs_profit_colors = ['#E6B91E', '#D4B323', '#C2AD28', '#B0A72D', '#9EA132', '#8C9B37', '#7A953C', '#688F41', '#568946', '#44834B', '#327D50', '#207755', '#0E715A']  # Dark mustard to dark olive green

    # Add bar graph for Absolute Profit
    fig.add_trace(go.Bar(
        x=df['EMA_Days'],
        y=df['Absolute_Profit'],
        name='Absolute Profit',
        marker_color=abs_profit_colors,
        marker_line_color='rgb(0,0,0)',
        marker_line_width=1.5,
        opacity=0.8
    ), secondary_y=False)

    # Define color scale for Percentage Profit (line graph)
    per_profit_colors = ['#E6B91E', '#D4B323', '#C2AD28', '#B0A72D', '#9EA132', '#8C9B37', '#7A953C', '#688F41', '#568946', '#44834B', '#327D50', '#207755', '#0E715A']  # Dark mustard to dark olive green

    # Add line graph for Percentage Profit
    fig.add_trace(go.Scatter(
        x=df['EMA_Days'],
        y=df['Percentage_Profit'],
        name='Percentage Profit',
        mode='lines+markers',
        line=dict(color=per_profit_colors[0], width=3),
        marker=dict(color=per_profit_colors, size=8)
    ), secondary_y=True)

    # Set axis titles
    fig.update_layout(
        title='EMA Profit Analysis',
        xaxis_title='EMA Days',
        yaxis_title='Absolute Profit',
        yaxis2_title='Percentage Profit',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Set secondary y-axis range
    fig.update_yaxes(range=[df['Percentage_Profit'].min() * 0.9, df['Percentage_Profit'].max() * 1.1], secondary_y=True)

    # Show the plot
    st.plotly_chart(fig, use_container_width=True)

def calculate_ema(df, days):
    """
    Calculate the exponential moving average (EMA) of the 'Close' price.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column.
        days (int): Number of days for the EMA calculation.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the EMA.
    """
    ema_col_name = f"EMA_{days}"
    df[ema_col_name] = df['Close'].ewm(span=days, adjust=False).mean()
    return df

def ema_flag(df, days):
    """
    Create a flag column indicating if 'Close' is above or below the EMA.
    Args:
        df (pd.DataFrame): DataFrame with 'Close' and EMA columns.
        days (int): Number of days for the EMA.
    Returns:
        pd.DataFrame: DataFrame with an additional flag column.
    """
    ema_flag_col_name = f"EMA_{days}_Flag"
    ema_col_name = f"EMA_{days}"
    df[ema_flag_col_name] = np.where(df['Close'] > df[ema_col_name], 1, 0) # 1 if Close > EMA else 0
    return df

def generate_trade_signals(df, days):
    """
    Generates trade signals (BUY or SELL) based on EMA flag changes.

    Args:
        df (pd.DataFrame): DataFrame with 'EMA_{days}_Flag' column.
        days (int): Number of days for the EMA.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Trade_Signal' column.
    """
    ema_flag_col_name = f"EMA_{days}_Flag"
    trade_signal_col_name = f"Trade_Signal_{days}"

    # Shift the EMA flag column to compare with the previous day's flag
    df['EMA_Flag_Shifted'] = df[ema_flag_col_name].shift(1)

    # Initialize the trade signal column with empty strings
    df[trade_signal_col_name] = ""

    trade_signal_buy = False
    trade_signal_sell = False
    buy_price = 0.0

    # Iterate through the DataFrame to generate trade signals
    for i in range(1, len(df)):
        if df[ema_flag_col_name][i] == 1 and df['EMA_Flag_Shifted'][i] == 0 and not trade_signal_buy:
            df[trade_signal_col_name][i] = "BUY"
            trade_signal_buy = True
            trade_signal_sell = False
            buy_price = df['Close'][i]
        elif df[ema_flag_col_name][i] == 0 and df['EMA_Flag_Shifted'][i] == 1 and trade_signal_buy and df['Close'][i] >= buy_price * 1.02:
            df[trade_signal_col_name][i] = "SELL"
            trade_signal_sell = True
            trade_signal_buy = False
        else:
            if trade_signal_sell:
                df[trade_signal_col_name][i] = "HOLD SELL"
            elif trade_signal_buy:
                df[trade_signal_col_name][i] = "HOLD BUY"
            else:
                df[trade_signal_col_name][i] = "NO SIGNAL"

    # Sell all open buy positions on the last row
    if trade_signal_buy:
        df[trade_signal_col_name].iloc[-1] = "SELL"
    
    # Remove the temporary shifted column
    df.drop(columns=['EMA_Flag_Shifted'], inplace=True)

    return df

def record_trade_prices(df, days):
    """
    Records the close price when there is a "BUY" or "SELL" signal for a given EMA.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' and 'Trade_Signal_{days}' columns.
        days (int): Number of days for the EMA.

    Returns:
        pd.DataFrame: DataFrame with additional columns for BUY and SELL prices.
    """
    trade_signal_col_name = f"Trade_Signal_{days}"
    buy_price_col_name = f"BUY_Price_{days}"
    sell_price_col_name = f"SELL_Price_{days}"

    # Initialize new columns with NaN values
    df[buy_price_col_name] = 0.0
    df[sell_price_col_name] = 0.0

    # Iterate through the DataFrame to record BUY and SELL prices
    for i in range(len(df)):
        if df[trade_signal_col_name][i] == "BUY":
            df.loc[i, buy_price_col_name] = df['Close'][i]
        elif df[trade_signal_col_name][i] == "SELL":
            df.loc[i, sell_price_col_name] = df['Close'][i]

    return df

def calculate_buy_sell_difference(df, days):
    """
    Calculates the difference between BUY and SELL prices for a given EMA.

    Args:
        df (pd.DataFrame): DataFrame with 'BUY_Price_{days}' and 'SELL_Price_{days}' columns.
        days (int): Number of days for the EMA.

    Returns:
        pd.DataFrame: DataFrame with an additional column for the BUY-SELL difference.
    """
    buy_price_col_name = f"BUY_Price_{days}"
    sell_price_col_name = f"SELL_Price_{days}"
    difference_col_name = f"Difference_{days}"

    df[difference_col_name] = df[sell_price_col_name] - df[buy_price_col_name]

    return df

def calculate_best_ema(df):
    """
    Calculate the best EMA based on the highest cumulative profit from buy-sell differences.

    Args:
        df (pd.DataFrame): DataFrame with 'Difference_{days}' columns for various EMAs.

    Returns:
        int: The number of days corresponding to the best EMA.
    """
    best_ema = None
    max_profit = float('-inf')
    max_profit_per = float('-inf')
    first_close_value = df['Close'].iloc[0]

    lst_days_ema = []
    lst_absolute_profit = []
    lst_per_profit = []

    for day in lst_days:
        difference_col_name = f"Difference_{day}"
        total_profit = df[difference_col_name].sum()
        per_profit = (total_profit / first_close_value) * 100 if first_close_value != 0 else 0
        
        # Append results to lists for potential further analysis or plotting
        lst_days_ema.append(difference_col_name)
        lst_absolute_profit.append(total_profit)
        lst_per_profit.append(per_profit)

        # Display the profit results using Streamlit
        # st.markdown(f"<h4>Profit for EMA {day}: {per_profit:.2f}%</h4>", unsafe_allow_html=True)
        # st.markdown(f"<h4>Total Profit for EMA {day}: {total_profit:.2f}</h4>", unsafe_allow_html=True)

        # Determine if this EMA is the best one so far
        if total_profit > max_profit:
            max_profit = total_profit
            max_profit_per = per_profit
            best_ema = day
    # Highlight the best EMA
    st.markdown(f"<h5 style='color: #787355;'>Best EMA is {best_ema} days with a profit of {max_profit:.2f} or {max_profit_per:.2f} %</h5>", unsafe_allow_html=True)
    
    # Create a DataFrame to summarize the results
    df_best_ma_analysis = pd.DataFrame({
        'EMA_Days': lst_days_ema,
        'Absolute_Profit': lst_absolute_profit,
        'Percentage_Profit': lst_per_profit
    })

    st.markdown("<h5 style='color: #787355;'>EMA Profit Analysis Summary</h5>", unsafe_allow_html=True)
    st.dataframe(df_best_ma_analysis, use_container_width=True)
    
    # Plot the EMA analysis results
    # Sort by Percentage_Profit in descending order
    df_best_ma_analysis = df_best_ma_analysis.sort_values(by='Percentage_Profit', ascending=False)
    plot_ema_analysis(df_best_ma_analysis)
    return best_ema

def create_ema_dataframe():
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

    max_days = max(lst_days) + 1  # Ensure enough data for the largest EMA

    df_xau_ma = df_xau.copy()
    df_xau_ma = df_xau_ma.tail(365 + max_days).reset_index()  # Limit to last 400 rows for performance

    for day in lst_days:
        df_xau_ma = calculate_ema(df_xau_ma, day)
    
    df_xau_ma = df_xau_ma.tail(len(df_xau_ma) - max_days).reset_index(drop=True) # Drop initial rows with NaN EMA values

    for day in lst_days:
        df_xau_ma = ema_flag(df_xau_ma, day)    

    for day in lst_days:
        df_xau_ma = generate_trade_signals(df_xau_ma, day)

    for day in lst_days:
        df_xau_ma = record_trade_prices(df_xau_ma, day)

    for day in lst_days:
        df_xau_ma = calculate_buy_sell_difference(df_xau_ma, day)

    st.session_state['df_xau_ma'] = df_xau_ma

    return df_xau

if 'show_moving_average' not in st.session_state or not st.session_state['show_moving_average']:
    st.session_state['show_moving_average'] = True
    st.switch_page("pages/Moving_Average_Analysis.py")

if 'df_xau_ma' not in st.session_state:
    create_ema_dataframe()

df_xau_ma_enriched = st.session_state['df_xau_ma'].copy()
df_xau_ma_enriched = df_xau_ma_enriched.drop('index', axis=1, errors='ignore')

st.markdown("<h5 style='color: #787355;'>Nifty Gold BEES EMA Analysis Head</h5>", unsafe_allow_html=True)
st.dataframe(df_xau_ma_enriched.head(25), use_container_width=True)
st.markdown("<h5 style='color: #787355;'>Nifty Gold BEES EMA Analysis Tail</h5>", unsafe_allow_html=True)
st.dataframe(df_xau_ma_enriched.tail(25), use_container_width=True)

calculate_best_ema(st.session_state['df_xau_ma'])
"""all the imports are included here"""

import yfinance as yf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from prophet import Prophet
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

st.set_page_config(layout="wide")
# Main code for analysis and visualization of nifty 50 data
# using streamlit and plotly

st.title("Market Analysis Streamlit App")
df_nifty50 = pd.DataFrame()
df_xau = pd.DataFrame()

str_start_date = ""
str_end_date= ""

def get_dates(day_delta=1826):
    """
    Get the start and end dates for the data retrieval.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=day_delta)
    # Store in session state
    if 'start_date' not in st.session_state:
        st.session_state['start_date'] = start_date
    if 'end_date' not in st.session_state:
        st.session_state['end_date'] = end_date
    return st.session_state['start_date'], st.session_state['end_date']

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

def handle_infinity_values(df):
    """
    Replace infinite values in the DataFrame with NaN.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def prophet_forecast(df_fit, df_predict,lst_regressors,title):
    """
    Fit a Prophet model and make predictions.
    """
    m = Prophet()
    for regressor in lst_regressors:
        m.add_regressor(regressor)
    lst_fit_cols = ['ds', 'y'] + lst_regressors
    df_fit = df_fit[lst_fit_cols]
    df_fit.columns = df_fit.columns.get_level_values(0)
    df_fit = df_fit.dropna()
    nan_rows = df_fit[df_fit.isna().any(axis=1)]
    print("Rows with NaN in df_fit:")
    print(nan_rows)

    lst_predict_cols = ['ds'] + lst_regressors
    df_predict = df_predict[lst_predict_cols]
    df_predict.columns = df_predict.columns.get_level_values(0)
    nan_rows = df_predict[df_predict.isna().any(axis=1)]
    print("Rows with NaN in df_predict:")
    print(nan_rows)

    m.fit(df_fit)
    forecast = m.predict(df_predict)
    forecast['yhat'] = forecast['yhat'].round(4)
    forecast['yhat_lower'] = forecast['yhat_lower'].round(4)
    forecast['yhat_upper'] = forecast['yhat_upper'].round(4)
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    forecast['yhat'] = forecast['yhat'].astype(float)
    forecast['yhat_lower'] = forecast['yhat_lower'].astype(float)
    forecast['yhat_upper'] = forecast['yhat_upper'].astype(float)
    print(title)
    print(forecast.head(5))
    return forecast

def plot_forecast(forecast, figure_title):
    """
    Plot the forecasted data.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound'))
    fig.update_layout(title=f'Forecast for {figure_title}', xaxis_title='Date', yaxis_title=figure_title)
    return fig

def predict_close_val(last_close_value,df):
    """
    Predict the close value based on the last close value.
    """
    # Placeholder for prediction logic
    # For now, just returning the last close value
    predicted_prices = []
    current_price = last_close_value

    for yhat in df['yhat']:
        current_price = current_price * (1+yhat)
        predicted_prices.append(current_price)

    # Add predicted prices to the forecast_close DataFrame
    df['predicted_price'] = predicted_prices

    # Print the first 5 predicted prices
    print("Sample Predicted Values:")
    print(df[['ds', 'yhat', 'predicted_price']].tail(5))

    return df

def add_forecasted_price(df,new_col, df_forecast, forecast_col,message=""):
    """
    Add the forecasted price to the DataFrame.
    """
    df[new_col] = df_forecast[forecast_col].values
    df[new_col] = df[new_col].astype(float)
    df[new_col] = df[new_col].round(4)
    print(message)
    print(df.head(5))
    return df

def plot_actual_vs_predicted(df, actual_col, predicted_col,title_txt=""):
    """
    Plot the actual vs predicted values.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df[actual_col], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df['ds'], y=df[predicted_col], mode='lines', name='Predicted'))
    fig.update_layout(title=title_txt, xaxis_title='Date', yaxis_title='Close Price')
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.13rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#0074D9;">Actual vs Predicted Chart: Explanation</b><br><br>
        This chart compares <span style="background-color:#e3f6fd; color:#0074D9; font-weight:bold;">actual values</span> with <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">predicted values</span> over time.<br><br>
        <span style="color:#393e46;">Key highlights:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Visualizes how closely the <span style="background-color:#e3f6fd; color:#0074D9; font-weight:bold;">model predictions</span> track the <span style="background-color:#e3f6fd; color:#0074D9; font-weight:bold;">actual observed data</span> for each date.</li>
            <li>Helps identify periods of <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">high accuracy</span> and <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">model deviation</span>.</li>
            <li>Useful for <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">model validation</span>, <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">forecasting analysis</span>, and improving predictive strategies.</li>
        </ul>
        <span style="color:#393e46;">Use this chart to evaluate prediction performance and refine your forecasting approach.</span>
    </div>
    """,
    unsafe_allow_html=True)
    
    st.plotly_chart(fig, use_container_width=True)
    return

def plot_bar_graph(df, x_col, y_col, title):
    """
    Plot a bar graph.
    """
    colors = [ "#3e8a00" if val >= 0 else "#a63700" for val in df[y_col] ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df[x_col], 
        y=df[y_col],
        marker_color=colors,
        text=df[y_col].apply(lambda x: f"{x:,.2f}"),
        textposition='auto',
        name=title))
    fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col)
    st.plotly_chart(fig, use_container_width=True)
    return

def rename_columns(df, col_mapping,message=""):
    """
    Rename columns in the DataFrame based on the provided mapping.
    """
    df = df.rename(columns=col_mapping)
    print(message)
    print(df.head(5))
    print(df.tail(5))
    return df

def split_dataframe(df, num_parts):
    """
    Split the DataFrame into num_parts equal parts.
    """
    part_size = len(df) // num_parts
    arr_split_df = [df.iloc[i * part_size:(i + 1) * part_size] for i in range(num_parts)]
    for i, part in enumerate(arr_split_df):
        print(f"Part {i+1} of DataFrame:")
        print(part.head(3))
        print(part.tail(3))
    return arr_split_df

def filter_array_substr(arr,susbstr=[]):
    """
    Filter an array to only include elements that contain a specific substring.
    """
    for df in arr:
        df = df.loc[:, df.columns.str.contains('|'.join(susbstr))]
        print("Filtered DataFrame:")
        print(df.head(5))
    return arr

def calculate_r2(df, col1, col2):
    # Drop rows where either column is NaN
    temp = df[[col1, col2]].fillna(df.mean(numeric_only=True))
    if len(temp) < 2:
        print(f"Not enough data to calculate R^2 for {col1} vs {col2}")
        return None
    r2 = r2_score(temp[col1], temp[col2])
    print(f"R^2 between {col1} and {col2}: {r2:.4f}")
    return r2

def calculate_r2_for_all(filtered_arr_split_df,main_ticker):
    """
    Calculate R^2 values for all pairs of columns in the DataFrame.
    """
    df_correlation_progress = pd.DataFrame()
    lst_r2_nifty = []
    lst_r2_inr = []
    lst_r2_btc = []
    lst_r2_xau = []
    lst_year = []

    for df in filtered_arr_split_df:
        year = df['Date_Val'].iloc[0].year
        print("Dataframe shape for year ", year, ":", df.shape)
        lst_year.append(year)

    fig_correl = make_subplots(
    rows=2, 
    cols=3,
    subplot_titles=tuple(lst_year),
    horizontal_spacing=0.15
    )

    row_counter = 1
    col_counter = 1

    for df in filtered_arr_split_df:
        
        df_correl = df[['Log_Ret_Close', 'Log_Ret_Close_Nifty', 'Log_Ret_Close_INR', 'Log_Ret_Close_BTC', 'Log_Ret_Close_XAU']].corr().round(4)
        r2_nifty = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_Nifty')
        lst_r2_nifty.append(r2_nifty)
        r2_inr = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_INR')
        lst_r2_inr.append(r2_inr)
        r2_btc = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_BTC')
        lst_r2_btc.append(r2_btc)
        r2_xau = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_XAU')
        lst_r2_xau.append(r2_xau)
        fig_correl.add_trace(
            go.Heatmap(
                z=df_correl.values,
                x=df_correl.columns,
                y=df_correl.index,
                colorscale='rdylgn',
                showscale=False,
                zmin=-1, 
                zmax=1,
                text=df_correl.values,
                texttemplate="%{text:.2f}"
            ),
            row=row_counter, col=col_counter
        )
        col_counter+=1
        if(col_counter > 3):
            col_counter = 1
            row_counter += 1
    
    fig_correl.update_layout(
    title="Correlation Heatmaps for daily returns of XAU, Nifty, INR, BTC",
    height=600,
    autosize=True)

    fig_correl.write_html(main_ticker+"_"+"Correlation_Heatmaps.html", auto_open=True)

    df_correlation_progress['Year'] = lst_year
    df_correlation_progress['R2_Nifty'] = lst_r2_nifty
    df_correlation_progress['R2_INR'] = lst_r2_inr
    df_correlation_progress['R2_BTC'] = lst_r2_btc
    df_correlation_progress['R2_XAU'] = lst_r2_xau

    # Create a 2x2 subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("R² Nifty", "R² INR", "R² BTC", "R² XAU")
    )

    # Add bar for R2_Nifty
    fig.add_trace(
        go.Bar(x=df_correlation_progress['Year'], y=df_correlation_progress['R2_Nifty'], name='R² Nifty', marker_color="#3e8a00"),
        row=1, col=1
    )
    # Add bar for R2_INR
    fig.add_trace(
        go.Bar(x=df_correlation_progress['Year'], y=df_correlation_progress['R2_INR'], name='R² INR', marker_color="#a63700"),
        row=1, col=2
    )
    # Add bar for R2_BTC
    fig.add_trace(
        go.Bar(x=df_correlation_progress['Year'], y=df_correlation_progress['R2_BTC'], name='R² BTC', marker_color="#0074D9"),
        row=2, col=1
    )
    # Add bar for R2_XAU
    fig.add_trace(
        go.Bar(x=df_correlation_progress['Year'], y=df_correlation_progress['R2_XAU'], name='R² XAU', marker_color="#FF851B"),
        row=2, col=2
    )

    fig.update_layout(
        height=700, width=900,
        title_text="R² Values for Nifty, INR, BTC, XAU by Year"
    )

    fig.write_html(main_ticker+"_"+"R2_Values_Matrix.html", auto_open=True)

    return df_correlation_progress

def max_min_band(df,ticker):
    """
    Calculate the maximum and minimum price bands for the DataFrame.
    """
    df = df.sort_index(ascending=True)
    df['Max_Price'] = df[['Close', 'High', 'Low', 'Open']].max(axis=1)
    df['Min_Price'] = df[['Close', 'High', 'Low', 'Open']].min(axis=1)
    df['Max_Price'] = df['Max_Price'].astype(float)
    df['Min_Price'] = df['Min_Price'].astype(float)
    df['Max_Price'] = df['Max_Price'].round(4)
    df['Min_Price'] = df['Min_Price'].round(4)
    df['Max_Min_Band'] = df['Max_Price'] - df['Min_Price']
    df['Max_Min_Band'] = df['Max_Min_Band'].astype(float)
    df['Max_Min_Band'] = df['Max_Min_Band'].round(4)

    df['Max_Price_Rolling_Vol'] = df['Max_Price'].rolling(window=23).std() * (252 ** 0.5)
    df['Max_Price_Rolling_Vol'] = df['Max_Price_Rolling_Vol'].round(6)
    df['Max_Price_Rolling_Vol'] = df['Max_Price_Rolling_Vol'].astype(float)

    df['Min_Price_Rolling_Vol'] = df['Min_Price'].rolling(window=23).std() * (252 ** 0.5)
    df['Min_Price_Rolling_Vol'] = df['Min_Price_Rolling_Vol'].round(6)
    df['Min_Price_Rolling_Vol'] = df['Min_Price_Rolling_Vol'].astype(float)
    
    max_value = df['Max_Price'].max()
    print("Max Price Value:", max_value)
    df['Delta_from_Max'] = df['Max_Price'] - max_value
    df['Delta_from_Max'] = df['Delta_from_Max'].astype(float)
    df['Delta_from_Max'] = df['Delta_from_Max'].round(4)

    min_value = df['Min_Price'].min()
    print("Min Price Value:", min_value)
    df['Delta_from_Min'] = df['Min_Price'] - min_value
    df['Delta_from_Min'] = df['Delta_from_Min'].astype(float)
    df['Delta_from_Min'] = df['Delta_from_Min'].round(4)

    # Filter df for dates after the first 252 days (i.e., keep rows starting from index 252)
    df_after_252 = df.iloc[252:]
    df_last_90 = df.tail(90)

    print("Latest 90 days : ")
    print(df_last_90.tail(5))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_after_252.index,
        y=df_after_252['Max_Price_Rolling_Vol'],
        mode='lines',
        name='Max Price Rolling Volatility'
    ))
    fig.add_trace(go.Scatter(
        x=df_after_252.index,
        y=df_after_252['Min_Price_Rolling_Vol'],
        mode='lines',
        name='Min Price Rolling Volatility'
    ))
    fig.update_layout(
        title='252-Day Rolling Volatility: Max vs Min Price',
        xaxis_title='Date',
        yaxis_title='Annualized Volatility'
    )

    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.13rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#0074D9;">Max-Min Price Band Chart: Explanation</b><br><br>
        This chart visualizes the <span style="background-color:#e3f6fd; color:#0074D9; font-weight:bold;">maximum</span> and <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">minimum</span> price bands for each trading day, along with their <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">rolling volatility</span>.<br><br>
        <span style="color:#393e46;">Key highlights:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Shows the <span style="background-color:#e3f6fd; color:#0074D9; font-weight:bold;">highest</span> and <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">lowest</span> prices (from Open, High, Low, Close) for each day.</li>
            <li>Plots the <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">annualized rolling volatility</span> of these price bands over a 252-day window, helping you spot periods of high or low market uncertainty.</li>
            <li>Includes a focused view of the <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">last 90 days</span> to highlight recent price movements and deviations from historical extremes.</li>
            <li>Useful for <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">risk management</span>, <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">volatility analysis</span>, and identifying breakout or reversal zones.</li>
        </ul>
        <span style="color:#393e46;">Use this chart to understand price ranges, volatility trends, and how current prices compare to historical highs and lows.</span>
    </div>
    """,
    unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(  
        x=df_last_90.index,
        y=df_last_90['Delta_from_Max'],
        mode='lines',
        name='Delta from Max Price'
    ))
    fig2.add_trace(go.Scatter(
        x=df_last_90.index,
        y=df_last_90['Delta_from_Min'],
        mode='lines',
        name='Delta from Min Price'
    ))
    fig2.update_layout(
        title='Last 90 Days: Delta from Max and Min Price',
        xaxis_title='Date',
        yaxis_title='Price Delta'
    )

    st.plotly_chart(fig2, use_container_width=True)
    return

def plot_positive_negative_streak(df, title):
    """
    Plot the positive and negative streaks in the DataFrame.
    """
    df = df.sort_index(ascending=True)
    positive = df['Log_Ret_Close'] >= 0
    negative = df['Log_Ret_Close'] < 0
    df['Positive_Streak'] = positive.groupby((~positive).cumsum()).cumcount()
    df['Positive_Streak'] = df['Positive_Streak'] * positive  # Set to 0 where not positive
    df['Negative_Streak'] = negative.groupby((~negative).cumsum()).cumcount()
    df['Negative_Streak'] = df['Negative_Streak'] * negative  # Set to 0 where not negative
    df_latest_45 = df.tail(45)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_latest_45.index,
        y=df_latest_45['Positive_Streak'],
        name='Positive Streak',
        marker_color='green'
    ))
    fig.add_trace(go.Bar(
        x=df_latest_45.index,
        y=df_latest_45['Negative_Streak'],
        name='Negative Streak',
        marker_color='red'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Streak Length'
    )

    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.13rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#008000;">Positive & Negative Streaks Chart: Explanation</b><br><br>
        This chart visualizes <span style="background-color:#e3f6fd; color:#008000; font-weight:bold;">positive streaks</span> and <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">negative streaks</span> in daily returns.<br><br>
        <span style="color:#393e46;">Key highlights:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Shows the length of consecutive <span style="background-color:#e3f6fd; color:#008000; font-weight:bold;">positive</span> and <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">negative</span> return streaks for each day.</li>
            <li>Helps identify <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">momentum</span> and <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">trend persistence</span> in asset returns.</li>
            <li>Useful for <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">timing strategies</span> and understanding the behavior of market rallies and corrections.</li>
        </ul>
        <span style="color:#393e46;">Use this chart to analyze streak patterns and gain insights into market dynamics and investor sentiment.</span>
    </div>
    """,
    unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)

    return df

def probability_of_streak_reset_after_value(df, streak_col):
    """
    Calculate the probability that the streak resets to zero after each unique streak value.
    Prints the probability for each streak value.
    """
    streaks = df[streak_col].values
    unique_streaks = sorted(set(streaks) - {0})  # Exclude 0
    df_prob_marker = pd.DataFrame(columns=['Streak', 'Probability'])
    df_prob_marker['Streak'] = list(unique_streaks)
    df_prob_marker['Probability'] = 0.0
    lst_prob=[]
    for val in unique_streaks:
        idx = np.where(streaks[:-1] == val)[0]
        resets = (streaks[idx + 1] == 0).sum()
        total = len(idx)
        prob = resets / total if total > 0 else 0
        lst_prob.append(prob)
        df_prob_marker.loc[df_prob_marker['Streak'] == val, 'Probability'] = prob
        print(f"Streak {val}: {resets} resets out of {total} occurrences, probability = {prob:.2%}")
    df_prob_marker['Probability'] = df_prob_marker['Probability'].astype(float)
    df_prob_marker['Probability'] = df_prob_marker['Probability'].round(4)
    df = pd.merge(df, df_prob_marker, left_on=streak_col, right_on='Streak', how='left')
    df.rename(columns={'Probability': 'Reset_Probability_'+streak_col}, inplace=True)
    df['Reset_Probability_'+streak_col].fillna(0, inplace=True)
    return df

def plot_probability_of_streak_reset(df,title):
    """
    Plot the probability of streak reset for each unique streak value.
    """
    df_latest = df.tail(45)
    fig = go.Figure()

    # Plot Positive Streak Reset Probability
    fig.add_trace(go.Scatter(
        x=df_latest['Date_Val'],
        y=df_latest['Reset_Probability_Positive_Streak'],
        mode='lines+markers',
        name='Positive Streak Reset Probability',
        marker_color='#808000'  # Olive green
    ))

    # Plot Negative Streak Reset Probability
    fig.add_trace(go.Scatter(
        x=df_latest['Date_Val'],
        y=df_latest['Reset_Probability_Negative_Streak'],
        mode='lines+markers',
        name='Negative Streak Reset Probability',
        marker_color='#FF9933'  # Sunset saffron
    ))

    fig.update_layout(
        title='Reset Probability for Positive and Negative Streaks',
        xaxis_title='Streak Value',
        yaxis_title='Reset Probability',
        yaxis=dict(tickformat=".00%")
    )
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.13rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#808000;">Probability of Streak Reset Chart: Explanation</b><br><br>
        This chart visualizes the <span style="background-color:#e3f6fd; color:#808000; font-weight:bold;">reset probability</span> for both <span style="background-color:#e3f6fd; color:#008000; font-weight:bold;">positive streaks</span> and <span style="background-color:#ffe0e0; color:#FF9933; font-weight:bold;">negative streaks</span> in daily returns.<br><br>
        <span style="color:#393e46;">Key highlights:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Shows how likely a <span style="background-color:#e3f6fd; color:#808000; font-weight:bold;">streak</span> of consecutive positive or negative returns is to end (reset to zero) after reaching a certain length.</li>
            <li>Helps identify <span style="background-color:#ffe0e0; color:#db3c02; font-weight:bold;">momentum</span> and <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">mean-reversion</span> patterns in asset returns.</li>
            <li>Useful for <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">risk management</span> and <span style="background-color:#e0ffe0; color:#00c90a; font-weight:bold;">strategy development</span> by understanding the persistence of trends.</li>
        </ul>
        <span style="color:#393e46;">Use this chart to analyze the behavior of streaks and improve your understanding of market dynamics.</span>
    </div>
    """,
    unsafe_allow_html=True
)
    st.plotly_chart(fig, use_container_width=True)
    return

def update_dates_in_session_state(start_date, end_date, sb):
    """
    Update Streamlit session state with start and end dates and return their string representations.
    Also writes the date values in the sidebar.
    """
    st.session_state['start_date'] = start_date
    st.session_state['end_date'] = end_date
    str_start_date = start_date.strftime('%Y-%m-%d')
    str_end_date = end_date.strftime('%Y-%m-%d')
    sb.markdown(f"**Start Date:** {str_start_date}")
    sb.markdown(f"**End Date:** {str_end_date}")
    return str_start_date, str_end_date

def analyse(main_ticker):
    str_start_date,str_end_date = get_dates()
    str_start_date = get_date_string(str_start_date)
    str_end_date = get_date_string(str_end_date)

    print("Default Dates:")
    print(f"Today: {str_end_date}")
    print(f"Start Date (5 Years ago): {str_start_date}")

    df_XAU = get_ticker_data(main_ticker, str_start_date, str_end_date)
    df_Nifty_50 = get_ticker_data("^NSEI", str_start_date, str_end_date)
    df_USD_INR = get_ticker_data("USDINR=X", str_start_date, str_end_date)
    df_USD_BTC = get_ticker_data("BTC-USD", str_start_date, str_end_date)
    df_US_GLD = get_ticker_data("GLD", str_start_date, str_end_date)

    # df_XAU.set_index("Price Date", inplace=True)
    df_XAU = get_log_returns(df_XAU, "Close")
    df_XAU = get_log_returns(df_XAU, "Volume")
    df_XAU = set_df_datatype(df_XAU)
    print(df_XAU.head(5))

    df_Nifty_50 = get_log_returns(df_Nifty_50, "Close")
    df_Nifty_50 = get_log_returns(df_Nifty_50, "Volume")
    df_Nifty_50 = set_df_datatype(df_Nifty_50)

    df_USD_INR = get_log_returns(df_USD_INR, "Close")
    df_USD_INR = get_log_returns(df_USD_INR, "Volume")
    df_USD_INR = set_df_datatype(df_USD_INR)

    df_USD_BTC = get_log_returns(df_USD_BTC, "Close")
    df_USD_BTC = get_log_returns(df_USD_BTC, "Volume")
    df_USD_BTC = set_df_datatype(df_USD_BTC)

    df_US_GLD = get_log_returns(df_US_GLD, "Close")
    df_US_GLD = get_log_returns(df_US_GLD, "Volume")
    df_US_GLD = set_df_datatype(df_US_GLD)

    df_Nifty_50 = rename_columns(df_Nifty_50,{"Close": "Nifty_Close"
                                            , "Volume": "Nifty_Volume"
                                            ,"Log_Ret_Close": "Log_Ret_Close_Nifty"
                                            , "Log_Ret_Volume": "Log_Ret_Volume_Nifty"},"Nifty 50 Data:")

    df_USD_INR = rename_columns(df_USD_INR,{"Close": "INR_Close"
                                            , "Volume": "INR_Volume"
                                            ,"Log_Ret_Close": "Log_Ret_Close_INR"
                                            , "Log_Ret_Volume": "Log_Ret_Volume_INR"},"USD to INR Data:")

    df_USD_BTC =rename_columns(df_USD_BTC,{"Close": "BTC_Close"
                                            , "Volume": "BTC_Volume"
                                            ,"Log_Ret_Close": "Log_Ret_Close_BTC"
                                            , "Log_Ret_Volume": "Log_Ret_Volume_BTC"},"USD to BTC Data:")

    df_US_GLD = rename_columns(df_US_GLD,{"Close": "XAU_Close"
                                            , "Volume": "XAU_Volume"
                                            ,"Log_Ret_Close": "Log_Ret_Close_XAU"
                                            , "Log_Ret_Volume": "Log_Ret_Volume_XAU"},"US GLD Data:")

    df_XAU = df_XAU.join(df_Nifty_50[['Log_Ret_Close_Nifty', 'Log_Ret_Volume_Nifty']], how='left')
    df_XAU = df_XAU.join(df_USD_INR[['Log_Ret_Close_INR', 'Log_Ret_Volume_INR']], how='left')
    df_XAU = df_XAU.join(df_USD_BTC[['Log_Ret_Close_BTC', 'Log_Ret_Volume_BTC']], how='left')
    df_XAU = df_XAU.join(df_US_GLD[['Log_Ret_Close_XAU', 'Log_Ret_Volume_XAU']], how='left')

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

    # add coregressor values
    df_latest2Months_close = set_df_prophet(df_latest2Months, 'Date_Val', 'Log_Ret_Close')
    df_latest2Months_close['Log_Ret_Close_Nifty'] = df_latest2Months['Log_Ret_Close_Nifty'].values
    df_latest2Months_close['Log_Ret_Close_INR'] = df_latest2Months['Log_Ret_Close_INR'].values
    df_latest2Months_close['Log_Ret_Close_BTC'] = df_latest2Months['Log_Ret_Close_BTC'].values
    df_latest2Months_close['Log_Ret_Close_XAU'] = df_latest2Months['Log_Ret_Close_XAU'].values

    df_latest2Months_vol = set_df_prophet(df_latest2Months, 'Date_Val', 'Log_Ret_Volume')
    df_latest2Months_vol['Log_Ret_Volume_Nifty'] = df_latest2Months['Log_Ret_Volume_Nifty'].values
    df_latest2Months_vol['Log_Ret_Volume_INR'] = df_latest2Months['Log_Ret_Volume_INR'].values
    df_latest2Months_vol['Log_Ret_Volume_BTC'] = df_latest2Months['Log_Ret_Volume_BTC'].values
    df_latest2Months_vol['Log_Ret_Volume_XAU'] = df_latest2Months['Log_Ret_Volume_XAU'].values

    # df_latest2Months_close['Log_Ret_Close_Nifty'] = df_latest2Months_close['Log_Ret_Close_Nifty'].fillna(0)
    # df_latest2Months_vol['Log_Ret_Volume_Nifty'] = df_latest2Months_vol['Log_Ret_Volume_Nifty'].fillna(0)

    df_analysis_close = set_df_prophet(df_analysis, 'Date_Val', 'Log_Ret_Close')
    df_analysis_vol = set_df_prophet(df_analysis, 'Date_Val', 'Log_Ret_Volume')

    # df_analysis_vol['Log_Ret_Volume_Nifty'] = df_analysis_vol['Log_Ret_Volume_Nifty'].fillna(0)
    # df_analysis_close['Log_Ret_Close_Nifty'] = df_analysis_close['Log_Ret_Close_Nifty'].fillna(0)

    print("Sample Latest 2 Months Data for Close:")
    print(df_latest2Months_close.head(5))
    print("Sample Latest 2 Months Data for Vol:")
    print(df_latest2Months_vol.head(5))
    print("Sample Analysis Data for Close:")
    print(df_analysis_close.head(5))
    print("Sample Analysis Data for Vol:")
    print(df_analysis_vol.head(5))

    df_analysis_close = df_analysis_close[['ds','y', 'Log_Ret_Close_Nifty','Log_Ret_Close_INR', 'Log_Ret_Close_BTC','Log_Ret_Close_XAU']]
    df_analysis_vol = df_analysis_vol[['ds','y', 'Log_Ret_Volume_Nifty','Log_Ret_Volume_INR', 'Log_Ret_Volume_BTC', 'Log_Ret_Volume_XAU']]

    print("Sample Analysis Data for Close:")
    print(df_analysis_close.head(5))
    print("Sample Analysis Data for Vol:")
    print(df_analysis_vol.head(5))

    df_analysis_close.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_analysis_close['y'] = df_analysis_close['y'].fillna(0)  # or use .dropna(subset=['y'])

    df_analysis_vol = handle_infinity_values(df_analysis_vol)
    df_latest2Months_close = handle_infinity_values(df_latest2Months_close)
    df_latest2Months_vol = handle_infinity_values(df_latest2Months_vol)

    #pass list of columns to be used as regressors
    forecast_close = prophet_forecast(df_analysis_close,df_latest2Months_close,['Log_Ret_Close_Nifty','Log_Ret_Close_INR','Log_Ret_Close_BTC','Log_Ret_Close_XAU'],"Forecast Close Data:")
    forecast_vol = prophet_forecast(df_analysis_vol, df_latest2Months_vol,['Log_Ret_Volume_Nifty','Log_Ret_Volume_INR','Log_Ret_Volume_BTC','Log_Ret_Volume_XAU'], "Forecast Volume Data:")

    # fig_plot = plot_forecast(forecast_close, "Forecast Close")
    # fig_plot.show()
    # fig_plot = plot_forecast(forecast_vol, "Forecast Volume")
    # fig_plot.show()

    forecast_close = predict_close_val(last_close_value, forecast_close)
    forecast_vol = predict_close_val(last_volume_value, forecast_vol)

    df_latest2Months_close = add_forecasted_price(df_latest2Months_close, "Predicted_Value", forecast_close, "predicted_price","Close Price Predicted Values:")
    df_latest2Months_vol = add_forecasted_price(df_latest2Months_vol, "Predicted_Value", forecast_vol, "predicted_price","Volume Predicted Values:")


    plot_actual_vs_predicted(df_latest2Months_close, "Close", "Predicted_Value","Close Price Deviation from Prediction" + main_ticker.replace(".NS", ""))
    plot_actual_vs_predicted(df_latest2Months_vol, "Volume", "Predicted_Value","Volume Deviation from Prediction" + main_ticker.replace(".NS", ""))

    df_latest2Months_close['Delta'] = df_latest2Months_close['Close'] - df_latest2Months_close['Predicted_Value']
    df_latest2Months_close['Delta'] = df_latest2Months_close['Delta'].astype(float) 
    df_latest2Months_close['Delta'] = df_latest2Months_close['Delta'].round(4)

    df_latest2Months_vol['Delta'] = df_latest2Months_vol['Volume'] - df_latest2Months_vol['Predicted_Value']
    df_latest2Months_vol['Delta'] = df_latest2Months_vol['Delta'].astype(float)
    df_latest2Months_vol['Delta'] = df_latest2Months_vol['Delta'].round(4)

    plot_bar_graph(df_latest2Months_close, "ds", "Delta", "Close Price Delta "+main_ticker.replace(".NS", ""))
    plot_bar_graph(df_latest2Months_vol, "ds", "Delta", "Volume Delta "+main_ticker.replace(".NS", ""))

    # Split df_xau into 5 equal parts
    arr_split_df = split_dataframe(df_XAU, 5)

    # only extract return from dataframes
    filtered_arr_split_df = filter_array_substr(arr_split_df, ['Log_Ret_Close', 'Date_Val'])

    # calculate r square values for all the dataframes
    # df_correlation_progress = calculate_r2_for_all(filtered_arr_split_df,main_ticker)
    # print("R^2 Values for Nifty, INR, BTC, XAU:")
    # print(df_correlation_progress)

    max_min_band(df_XAU, main_ticker.replace(".NS", ""))
    df_XAU = plot_positive_negative_streak(df_XAU, "Positive and Negative Streaks for " + main_ticker.replace(".NS", ""))
    # Example usage after your plot_positive_negative_streak:
    probability_of_streak_reset_after_value(df_XAU, 'Positive_Streak')
    df_XAU = probability_of_streak_reset_after_value(df_XAU, 'Positive_Streak')
    df_XAU = probability_of_streak_reset_after_value(df_XAU, 'Negative_Streak')
    plot_probability_of_streak_reset(df_XAU, "Probability of Streak Reset for " + main_ticker.replace(".NS", ""))
    return df_XAU

def get_default_dates(days):
    # Get today's date
    today = datetime.today()

    # Calculate a start date (e.g., 30 days ago)
    start_date = today - timedelta(days=days)
    str_end_date =today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    return str_start_date, str_end_date

def plotly_line_graph(df, x_col, y_cols,title,x_title,y_title,color_col=None, size_col=None,color_value=None):
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    fig.update_xaxes(type='date')
    fig.update_yaxes(type='linear')
    for y_col in y_cols:
        if color_col is not None and size_col is not None:
            fig.add_scatter(x=df[x_col], y=df[y_col[0]], mode='lines', name=y_col[1], marker=dict(color=df[color_col], size=df[size_col]), text=df[color_value])
        else:
            fig.add_scatter(x=df[x_col], y=df[y_col[0]], mode='lines', name=y_col[1])
    return fig

def color_returns(value):
    if value < 0.0000:
        return 'color: #db3c02;'
    else:
        return 'color: #00c90a;'
    
def correlation_heatmap(df, title):
    fig = px.imshow(df.corr(), text_auto=True, aspect="auto", title=title)
    fig.update_layout(title=title, xaxis_title="Features", yaxis_title="Features")
    return fig

sb = st.sidebar
sb.title("Market Analysis Settings")
st.divider()
st.header("Select the date range for analysis")
start_date, end_date = get_dates()
start_date = sb.date_input("Start Date", min_value=st.session_state['start_date'], max_value=datetime.today(), value=st.session_state['start_date'])
end_date = sb.date_input("End Date", min_value=st.session_state['start_date'], max_value=datetime.today(), value=datetime.today())
btn_refresh = sb.button("Refresh Data", key="refresh")

if btn_refresh:
    str_start_date,str_end_date = update_dates_in_session_state(start_date, end_date, sb)
else:
    str_start_date, str_end_date = update_dates_in_session_state(start_date, end_date, sb)
    sb.markdown("**Default start date for initial dataframe:** {}".format(str_start_date))
    sb.markdown("**Default end date for initial dataframe:** {}".format(str_end_date))

df_nifty50 = get_ticker_data("^NSEI", str_start_date, str_end_date)
df_nifty50 = get_log_returns(df_nifty50, "Close")
df_nifty50 = get_log_returns(df_nifty50, "Volume")

if end_date < start_date:
    st.error("End date must be after start date.")
    st.stop()
else:
    df_xau = get_ticker_data("GC=F", str_start_date, str_end_date)
    df_xau = get_log_returns(df_xau, "Close")
    df_xau = get_log_returns(df_xau, "Volume")
    styled_df_nifty = df_nifty50.style.applymap(color_returns, subset=['Log_Ret_Close','Log_Ret_Volume'])
    styled_df_xau = df_xau.style.applymap(color_returns, subset=['Log_Ret_Close','Log_Ret_Volume'])
    st.subheader("Nifty 50 DataFrame")
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.15rem; color: #222831; background-color: #f5f6fa; padding: 18px 24px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.25rem; color:#0074D9;">Nifty DataFrame Overview</b><br><br>
        The <span style="color:#0074D9;"><b>Nifty DataFrame</b></span> (<code>df_nifty50</code>) contains historical daily data for the <b>Nifty 50</b> index, a major benchmark of the Indian stock market.<br>
        This DataFrame is sourced directly from Yahoo Finance using the ticker symbol <code>^NSEI</code> for the date range you select.<br><br>
        <span style="color:#393e46;">It includes:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li><b>Open, High, Low, Close, Volume</b>: Standard daily price and volume data</li>
            <li><b>52_MA</b>: 52-day moving average of the closing price</li>
            <li><b>52_MA_VOL</b>: 52-day moving average of the volume</li>
            <li><b>Log_Ret_Price</b>: Logarithmic daily returns of the closing price</li>
            <li><b>Log_Ret_Volume</b>: Logarithmic daily returns of the trading volume</li>
        </ul>
        <span style="color:#393e46;">This structured data enables in-depth trend, volatility, and correlation analysis, helping you make data-driven market decisions.</span>
    </div>
    """,
    unsafe_allow_html=True)
    st.dataframe(styled_df_nifty, width=1200, height=500)
    st.subheader("XAU DataFrame")
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.15rem; color: #222831; background-color: #f5f6fa; padding: 18px 24px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.25rem; color:#FF851B;">XAU DataFrame Overview</b><br><br>
        The <span style="color:#FF851B;"><b>XAU DataFrame</b></span> (<code>df_xau</code>) contains historical daily data for <b>Gold (XAU)</b>, a key global benchmark for gold prices.<br>
        This DataFrame is sourced directly from Yahoo Finance using the ticker symbol <code>GC=F</code> for the date range you select.<br><br>
        <span style="color:#393e46;">It includes:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li><b>Open, High, Low, Close, Volume</b>: Standard daily price and volume data</li>
            <li><b>52_MA</b>: 52-day moving average of the closing price</li>
            <li><b>52_MA_VOL</b>: 52-day moving average of the volume</li>
            <li><b>Log_Ret_Price</b>: Logarithmic daily returns of the closing price</li>
            <li><b>Log_Ret_Volume</b>: Logarithmic daily returns of the trading volume</li>
        </ul>
        <span style="color:#393e46;">This structured data enables robust analysis of gold price trends, volatility, and its correlation with other financial assets.</span>
    </div>
    """,
    unsafe_allow_html=True)
    st.dataframe(styled_df_xau,width=1200, height=500)
    final_daily_ret_df = pd.merge(df_nifty50,
                                df_xau,
                                left_index=True, 
                                right_index=True, 
                                how='inner', 
                                suffixes=('_Nifty', '_XAU'))
    final_daily_ret_df = final_daily_ret_df[['Log_Ret_Close_Nifty','Log_Ret_Volume_Nifty','Log_Ret_Close_XAU','Log_Ret_Volume_XAU']]
    final_daily_ret_df_styled = final_daily_ret_df.style.applymap(color_returns, subset=['Log_Ret_Close_Nifty','Log_Ret_Volume_Nifty','Log_Ret_Close_XAU','Log_Ret_Volume_XAU'])
    st.subheader("Final Merged DataFrame")
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.15rem; color: #222831; background-color: #f5f6fa; padding: 18px 24px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.25rem; color:#6f42c1;">Merged DataFrame Overview</b><br><br>
        The <span style="color:#6f42c1;"><b>Final Merged DataFrame</b></span> combines daily data from both the <b>Nifty 50</b> index and <b>Gold (XAU)</b> for the selected date range.<br>
        This DataFrame is created by merging the Nifty and XAU DataFrames on their date index, allowing for direct comparison and joint analysis.<br><br>
        <span style="color:#393e46;">It includes:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li><b>Log_Ret_Price_Nifty</b>: Logarithmic daily returns of the Nifty 50 closing price</li>
            <li><b>Log_Ret_Volume_Nifty</b>: Logarithmic daily returns of the Nifty 50 trading volume</li>
            <li><b>Log_Ret_Close_XAU</b>: Logarithmic daily returns of the Gold (XAU) closing price</li>
            <li><b>Log_Ret_Volume_XAU</b>: Logarithmic daily returns of the Gold (XAU) trading volume</li>
        </ul>
        <span style="color:#393e46;">This merged dataset enables you to analyze correlations, co-movements, and volatility between the Indian equity market and global gold prices, supporting deeper financial insights and strategy development.</span>
    </div>
    """,
    unsafe_allow_html=True)
    st.dataframe(final_daily_ret_df_styled, 
                 width=1200, 
                 height=500)
    st.subheader("Correlation Heatmap")
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.15rem; color: #222831; background-color: #f5f6fa; padding: 18px 24px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.25rem; color:#e83e8c;">Correlation Heatmap Overview</b><br><br>
        The <span style="color:#e83e8c;"><b>Correlation Heatmap</b></span> visually represents the statistical relationships between the daily returns and volumes of the <b>Nifty 50</b> index and <b>Gold (XAU)</b>.<br>
        Each cell in the heatmap shows the correlation coefficient between two features, ranging from <b>-1</b> (perfect negative correlation) to <b>+1</b> (perfect positive correlation).<br><br>
        <span style="color:#393e46;">How to use:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li><b>Bright colors</b> indicate strong relationships (positive or negative) between features.</li>
            <li>Use this heatmap to quickly identify which variables move together or in opposite directions.</li>
            <li>Helps in understanding diversification and risk in your portfolio.</li>
        </ul>
        <span style="color:#393e46;">This visualization supports deeper analysis of market dynamics and asset interdependence.</span>
    </div>
    """,
    unsafe_allow_html=True)
    fig = correlation_heatmap(final_daily_ret_df, title="Correlation Heatmap of Nifty 50 and Gold (XAU) Daily Returns")
    st.plotly_chart(fig, use_container_width=True)

    final_daily_ret_df['Date'] = final_daily_ret_df.index
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.12rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#0074D9;">Nifty 50 & Gold Log Returns Line Chart: Use Case</b><br><br>
        This interactive line chart visualizes the <b>logarithmic daily returns</b> of the <span style="color:#0074D9;"><b>Nifty 50</b></span> index and <span style="color:#FF851B;"><b>Gold (XAU)</b></span> over your selected date range.<br><br>
        <span style="color:#393e46;">Use this chart to:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Compare the performance and volatility of Indian equities (Nifty 50) and global gold prices (XAU) side by side.</li>
            <li>Identify periods of high or low correlation, divergence, or co-movement between the two assets.</li>
            <li>Spot trends, sudden spikes, or drops in returns that may signal important market events or shifts in investor sentiment.</li>
            <li>Support investment analysis, risk management, and portfolio diversification decisions by understanding how these assets behave over time.</li>
        </ul>
        <span style="color:#393e46;">This visualization is a powerful tool for analysts and investors seeking to understand the dynamic relationship between stock market returns and gold as a safe-haven asset.</span>
    </div>
    """,
    unsafe_allow_html=True)
    
    fig = plotly_line_graph(final_daily_ret_df,
                            x_col='Date',
                            y_cols=[['Log_Ret_Close_Nifty','Nifty'], ['Log_Ret_Close_XAU','XAU']],
                            title="Nifty 50 & Gold Return (Logarithmic Daily Returns)",
                            x_title="Date",
                            y_title="Log Daily Returns")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(
        final_daily_ret_df
        , x="Log_Ret_Close_Nifty"
        , y="Log_Ret_Close_XAU"
        , title="Nifty 50 vs Gold Price"
        , labels={"Log_Ret_Close_Nifty": "Nifty 50 Log Daily Returns", "Log_Ret_Close_XAU": "XAU Log Daily Returns"}
        , trendline="ols")
    # Extract regression results
    results = px.get_trendline_results(fig)
    ols_results = results.iloc[0]["px_fit_results"]  # Get the OLS results
    r_squared = ols_results.rsquared  # Extract the R² value

    # Update the chart title to include the R² value
    fig.update_layout(
        title=f"Nifty 50 vs Gold Price (R² = {r_squared:.4f})"
    )
    st.markdown(
    """
    <div style="font-family: 'Segoe UI', Arial, sans-serif; font-size: 1.12rem; color: #222831; background-color: #f5f6fa; padding: 16px 22px; border-radius: 10px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
        <b style="font-size:1.18rem; color:#e83e8c;">Nifty 50 vs Gold Regression Scatter Plot: Use Case</b><br><br>
        This interactive scatter plot visualizes the relationship between the <b>logarithmic daily returns</b> of the <span style="color:#0074D9;"><b>Nifty 50</b></span> index and <span style="color:#FF851B;"><b>Gold (XAU)</b></span>.<br><br>
        <span style="color:#393e46;">Use this chart to:</span>
        <ul style="margin-top: 0.5em; margin-bottom: 0.5em;">
            <li>Assess the linear relationship between Nifty 50 and Gold returns using the regression trendline.</li>
            <li>Interpret the <b>R² value</b> in the chart title to understand how much of the variation in Gold returns can be explained by Nifty 50 returns.</li>
            <li>Identify periods of strong or weak correlation, and spot outliers or unusual co-movements.</li>
            <li>Support portfolio diversification and risk management by analyzing the dependency between these two assets.</li>
        </ul>
        <span style="color:#393e46;">This visualization is valuable for analysts and investors seeking to quantify and visualize the statistical relationship between Indian equities and gold as a global asset.</span>
    </div>
    """,
    unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
    # Calculate rolling volatility (standard deviation) over a 5-day window
    # final_daily_ret_df = final_daily_ret_df.sort_index(ascending=True)
    # final_daily_ret_df["Volatility_Nifty"] = final_daily_ret_df["Log_Ret_Price_Nifty"].rolling(window=14).std()
    # final_daily_ret_df["Volatility_XAU"] = final_daily_ret_df["Log_Ret_Close_XAU"].rolling(window=14).std()
    # final_daily_ret_df = final_daily_ret_df.sort_index(ascending=False)
    # st.dataframe(final_daily_ret_df)
    analyse("GOLDBEES.NS")
st.divider()



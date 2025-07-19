"""all the imports are included here"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf
st.set_page_config(layout="wide")

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
    # df['Date_Val'] = df.index
    return df

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
str_start_date, str_end_date = get_default_dates(1095)
st.markdown("**Default start date for initial dataframe:** {}".format(str_start_date))
st.markdown("**Default end date for initial dataframe:** {}".format(str_end_date))
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
    final_daily_ret_df = final_daily_ret_df[['Log_Ret_Price_Nifty','Log_Ret_Volume_Nifty','Log_Ret_Price_XAU','Log_Ret_Volume_XAU']]
    final_daily_ret_df_styled = final_daily_ret_df.style.applymap(color_returns, subset=['Log_Ret_Price_Nifty','Log_Ret_Volume_Nifty','Log_Ret_Price_XAU','Log_Ret_Volume_XAU'])
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
            <li><b>Log_Ret_Price_XAU</b>: Logarithmic daily returns of the Gold (XAU) closing price</li>
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
    fig = correlation_heatmap(final_daily_ret_df)
    st.plotly_chart(fig, use_container_width=True)

    final_daily_ret_df['Date'] = final_daily_ret_df.index
    fig = plotly_line_graph(final_daily_ret_df,
                            x_col='Date',
                            y_cols=[['Log_Ret_Price_Nifty','Nifty'], ['Log_Ret_Price_XAU','XAU']],
                            title="Nifty 50 & Gold Price",
                            x_title="Date",
                            y_title="Log Daily Returns")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(
        final_daily_ret_df
        , x="Log_Ret_Price_Nifty"
        , y="Log_Ret_Price_XAU"
        , title="Nifty 50 vs Gold Price"
        , labels={"Log_Ret_Price_Nifty": "Nifty 50 Log Daily Returns", "Log_Ret_Price_XAU": "XAU Log Daily Returns"}
        , trendline="ols")
    # Extract regression results
    results = px.get_trendline_results(fig)
    ols_results = results.iloc[0]["px_fit_results"]  # Get the OLS results
    r_squared = ols_results.rsquared  # Extract the R² value

    # Update the chart title to include the R² value
    fig.update_layout(
        title=f"Nifty 50 vs Gold Price (R² = {r_squared:.4f})"
    )
    st.plotly_chart(fig, use_container_width=True)
    # Calculate rolling volatility (standard deviation) over a 5-day window
    final_daily_ret_df = final_daily_ret_df.sort_index(ascending=True)
    final_daily_ret_df["Volatility_Nifty"] = final_daily_ret_df["Log_Ret_Price_Nifty"].rolling(window=14).std()
    final_daily_ret_df["Volatility_XAU"] = final_daily_ret_df["Log_Ret_Price_XAU"].rolling(window=14).std()
    final_daily_ret_df = final_daily_ret_df.sort_index(ascending=False)
    st.dataframe(final_daily_ret_df)
st.divider()



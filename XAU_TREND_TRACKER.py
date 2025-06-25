import yfinance as yf
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

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
    fig.write_html(title_txt.replace(" ", "_") + ".html", auto_open=True)
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
    fig.write_html(title.replace(" ", "_") + ".html", auto_open=True)
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
    temp = df[[col1, col2]].dropna()
    if len(temp) < 2:
        print(f"Not enough data to calculate R^2 for {col1} vs {col2}")
        return None
    r2 = r2_score(temp[col1], temp[col2])
    print(f"R^2 between {col1} and {col2}: {r2:.4f}")
    return r2

def calculate_r2_for_all(filtered_arr_split_df):
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
        r2_nifty = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_Nifty')
        lst_r2_nifty.append(r2_nifty)
        r2_inr = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_INR')
        lst_r2_inr.append(r2_inr)
        r2_btc = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_BTC')
        lst_r2_btc.append(r2_btc)
        r2_xau = calculate_r2(df, 'Log_Ret_Close', 'Log_Ret_Close_XAU')
        lst_r2_xau.append(r2_xau)
        year = df['Date_Val'].iloc[0].year
        lst_year.append(year)

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

    fig.write_html("R2_Values_Matrix.html", auto_open=True)

    return df_correlation_progress

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
    df_correlation_progress = calculate_r2_for_all(filtered_arr_split_df)
    print("R^2 Values for Nifty, INR, BTC, XAU:")
    print(df_correlation_progress)

    return df_XAU

df_XAU = analyse("GOLDBEES.NS")[['Date_Val','Close']]
df_XAG = analyse("SILVERBEES.NS")[['Date_Val','Close']]

print(df_XAU.head(5))
print(df_XAG.head(5))

df_ratio = df_XAU.merge(df_XAG, on='Date_Val', how='inner', suffixes=('_XAU', '_XAG'))
df_ratio['XAU_XAG_Ratio'] = df_ratio['Close_XAU'] / df_ratio['Close_XAG']
df_ratio['XAU_XAG_Ratio'] = df_ratio['XAU_XAG_Ratio'].round(4)
print("XAU to XAG Ratio Data:")
print("head data")
print(df_ratio.head(5))
print("tail data")
print(df_ratio.tail(5))

plot_bar_graph(df_ratio, "Date_Val", "XAU_XAG_Ratio", "XAU to XAG Ratio")
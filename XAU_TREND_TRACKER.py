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

def handle_infinity_values(df):
    """
    Replace infinite values in the DataFrame with NaN.
    """
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    return df

def prophet_forecast(df_fit, df_predict,title):
    """
    Fit a Prophet model and make predictions.
    """
    m = Prophet()
    m.fit(df_fit)
    forecast = m.predict(df_predict[['ds']])
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

# main program script here
# print date
str_start_date,str_end_date = get_dates()
str_start_date = get_date_string(str_start_date)
str_end_date = get_date_string(str_end_date)

print("Default Dates:")
print(f"Today: {str_end_date}")
print(f"Start Date (5 Years ago): {str_start_date}")

df_XAU = get_ticker_data("GOLDBEES.NS", str_start_date, str_end_date)
df_Nifty_50 = get_ticker_data("^NSEI", str_start_date, str_end_date)


# df_XAU.set_index("Price Date", inplace=True)
df_XAU = get_log_returns(df_XAU, "Close")
df_XAU = get_log_returns(df_XAU, "Volume")
df_XAU = set_df_datatype(df_XAU)
print(df_XAU.head(5))

df_Nifty_50 = get_log_returns(df_Nifty_50, "Close")
df_Nifty_50 = get_log_returns(df_Nifty_50, "Volume")
df_Nifty_50 = set_df_datatype(df_Nifty_50)
df_Nifty_50 = df_Nifty_50.rename(columns={"Close": "Nifty_Close"
                                          , "Volume": "Nifty_Volume"
                                          ,"Log_Ret_Close": "Log_Ret_Close_Nifty"
                                          , "Log_Ret_Volume": "Log_Ret_Volume_Nifty"})
print(df_Nifty_50.head(5))

df_XAU = df_XAU.join(df_Nifty_50[['Log_Ret_Close_Nifty', 'Log_Ret_Volume_Nifty']], how='left')

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
df_analysis_close = df_analysis_close[['ds','y']]
df_analysis_vol = df_analysis_vol[['ds','y']]
print("Sample Analysis Data for Close:")
print(df_analysis_close.head(5))
print("Sample Analysis Data for Vol:")
print(df_analysis_vol.head(5))

df_analysis_close.replace([np.inf, -np.inf], np.nan, inplace=True)
df_analysis_close['y'] = df_analysis_close['y'].fillna(0)  # or use .dropna(subset=['y'])

df_analysis_vol = handle_infinity_values(df_analysis_vol)
df_latest2Months_close = handle_infinity_values(df_latest2Months_close)
df_latest2Months_vol = handle_infinity_values(df_latest2Months_vol)

forecast_close = prophet_forecast(df_analysis_close,df_latest2Months_close,"Forecast Close Data:")
forecast_vol = prophet_forecast(df_analysis_vol, df_latest2Months_vol, "Forecast Volume Data:")

# fig_plot = plot_forecast(forecast_close, "Forecast Close")
# fig_plot.show()
# fig_plot = plot_forecast(forecast_vol, "Forecast Volume")
# fig_plot.show()

forecast_close = predict_close_val(last_close_value, forecast_close)
forecast_vol = predict_close_val(last_volume_value, forecast_vol)

df_latest2Months_close = add_forecasted_price(df_latest2Months_close, "Predicted_Value", forecast_close, "predicted_price","Close Price Predicted Values:")
df_latest2Months_vol = add_forecasted_price(df_latest2Months_vol, "Predicted_Value", forecast_vol, "predicted_price","Volume Predicted Values:")


plot_actual_vs_predicted(df_latest2Months_close, "Close", "Predicted_Value","Close Price Deviation from Prediction")
plot_actual_vs_predicted(df_latest2Months_vol, "Volume", "Predicted_Value","Volume Deviation from Prediction")

df_latest2Months_close['Delta'] = df_latest2Months_close['Close'] - df_latest2Months_close['Predicted_Value']
df_latest2Months_close['Delta'] = df_latest2Months_close['Delta'].astype(float) 
df_latest2Months_close['Delta'] = df_latest2Months_close['Delta'].round(4)

df_latest2Months_vol['Delta'] = df_latest2Months_vol['Volume'] - df_latest2Months_vol['Predicted_Value']
df_latest2Months_vol['Delta'] = df_latest2Months_vol['Delta'].astype(float)
df_latest2Months_vol['Delta'] = df_latest2Months_vol['Delta'].round(4)

plot_bar_graph(df_latest2Months_close, "ds", "Delta", "Close Price Deviation from Prediction")
plot_bar_graph(df_latest2Months_vol, "ds", "Delta", "Volume Deviation from Prediction")

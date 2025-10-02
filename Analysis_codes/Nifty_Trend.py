#%%

import yfinance as yf                              
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout # type: ignore
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

def add_median_line(fig,df,col_name,annotation_text_val="Median ",line_color_val="red",type=0):
    # Calculate the average of the 'Close' column
    average_delta = 0.0
    
    if type==0:
        average_delta = df.loc[df[col_name] > 0, col_name].median()*1.2
    else:
        average_delta = df.loc[df[col_name] < 0, col_name].median()*1.2

    # Add a horizontal line for the average
    fig.add_shape(
        type="line",
        x0=df["Date"].min(),
        x1=df["Date"].max(),
        y0=average_delta,
        y1=average_delta,
        line=dict(color=line_color_val, width=2, dash="dash"),
        name=annotation_text_val,
    )

    # Add annotation for the average line
    fig.add_annotation(
        x=df["Date"].iloc[-1],  # Position the annotation at the last date
        y=average_delta,
        text=f"value: {average_delta:.2f}",
        showarrow=False,
        font=dict(color="blue", size=12),
        align="right",
    )

    # Customize layout
    fig.update_layout(
        title="Graph with Median Line",
        xaxis_title="Date",
        yaxis_title="Delta",
        legend_title="Legend",
    )
    return fig

def get_log_returns(df, col_name):
    """
    Calculate log returns for a given column in the DataFrame.
    """
    df[f"Log_Ret_{col_name}"] = np.log(df[col_name] / df[col_name].shift(1))
    df[f"Log_Ret_{col_name}"] = df[f"Log_Ret_{col_name}"].fillna(0)
    df[f"Log_Ret_{col_name}"] = df[f"Log_Ret_{col_name}"].round(4)
    return df

def set_seed(seed_value):
    """
    Set seed for reproducibility across NumPy, TensorFlow, and Python's random module.
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

def get_ts_prediction(ticker):
    #set_seed(seed_value=23)

    # Get today's date
    today = datetime.today()
    start_date = today - timedelta(days=1826)  # 5 years ago
    str_end_date = today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    print(f"Today: {str_end_date}")
    print(f"Start Date (5 Years ago): {str_start_date}")
    data = yf.download(ticker, start=str_start_date, end=str_end_date)
    data.columns = data.columns.get_level_values(0)
    data.reset_index(inplace=True)

    # Calculate log returns for Close
    data = get_log_returns(data, 'Close')
    data['52_MA'] = data['Close'].rolling(window=52).mean()

    # Use Log_Ret_Close for prediction instead of Close
    dataset = data[['Log_Ret_Close']].values
    training_data_len = int(len(dataset) * 0.95)

    train_data = dataset[0:training_data_len]
    last_train_index = training_data_len - 1
    last_close_price = data.loc[last_train_index, 'Close']
    print("Close price in train_data last row:", last_close_price)
    test_data = dataset[training_data_len:]

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)

    # Prepare training data (past 60 days to predict next day log return)
    X_train, y_train = [], []
    for i in range(60, len(scaled_train)):
        X_train.append(scaled_train[i-60:i, 0])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    combined_test_data = np.concatenate((scaled_train[-60:], scaled_test), axis=0)
    X_test = []
    for i in range(60, len(combined_test_data)):
        X_test.append(combined_test_data[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=20)

    # Predict on training and test data
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse scaling to get actual log returns
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Actual values (for comparison)
    y_train_actual = scaler.inverse_transform([y_train])
    y_test_actual = test_data

    # Prepare data for plotting
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = test_predict

    # Use last_close_price and sequentially multiply and add predicted daily returns to generate predicted closing value
    predicted_close = []
    for ret in test_predict.flatten():
        last_close_price = round(last_close_price * (1+ ret), 2)
        predicted_close.append(last_close_price)

    valid['Predicted_Close'] = predicted_close

    print(valid.head())
    print(valid.tail())
    # Plot actual vs predicted log returns
    # fig = go.Figure()
    # fig.add_scatter(x=valid['Date'], y=valid['Log_Ret_Close'], name='Actual Log Returns', mode='lines')
    # fig.add_scatter(x=valid['Date'], y=valid['Predictions'], name='Predicted Log Returns', mode='lines')
    # fig.update_layout(title="Actual vs Predicted Log Returns", xaxis_title="Date", yaxis_title="Log Return")
    # fig.write_html(ticker.replace(".","_").replace("^","_")+'_Log_Return_Prediction.html', auto_open=True)

    # # Optionally, plot deviation
    # valid['Delta_Log_Ret'] = valid['Log_Ret_Close'] - valid['Predictions']
    # fig = go.Figure(data=[go.Bar(x=valid['Date'], y=valid['Delta_Log_Ret'], name="Deviation from Prediction", marker_color=valid['Delta_Log_Ret'].apply(lambda x: 'green' if x > 0 else 'red'))])
    # fig.update_layout(title="Deviation from Log Return Prediction", xaxis_title="Date", yaxis_title="Deviation")
    # fig.write_html(ticker.replace(".","_").replace("^","_")+'_Log_Return_Deviation.html', auto_open=True)

    # # # Plot actual vs predicted closing prices
    # fig = go.Figure()
    # fig.add_scatter(x=valid['Date'], y=valid['Close'], name='Actual Close', mode='lines')
    # fig.add_scatter(x=valid['Date'], y=valid['Predicted_Close'], name='Predicted Close', mode='lines')
    # fig.update_layout(title="Actual vs Predicted Close Price (from Log Returns)", xaxis_title="Date", yaxis_title="Close Price")
    # fig.write_html(ticker.replace(".","_").replace("^","_")+'_Close_Price_Prediction.html', auto_open=True)

    return valid


#%%
list_predicted_df = []
lst_tickers = ['GOLDBEES.NS']
for ticker in lst_tickers:
    for i in range(10):
        print(f"Processing {ticker} - Iteration {i+1}")
        list_predicted_df.append(get_ts_prediction(ticker))

#%%
# get_ts_prediction('^INDIAVIX')
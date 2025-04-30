#%%
import yfinance as yf
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

def get_ts_prediction(ticker):
    # Get today's date
    today = datetime.today()

    # Calculate a start date (e.g., 30 days ago)
    start_date = today - timedelta(days=1597)
    str_end_date =today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    # print date
    print(f"Today: {str_end_date}")
    print(f"Start Date (1597 days ago): {str_start_date}")
    data = yf.download(ticker, start=str_start_date, end=str_end_date)
    data.reset_index(inplace=True)

    # data.columns = data.columns.str.replace("^NSEI", "", regex=False)
    # Step 2: Calculate 52-day moving average
    data['52_MA'] = data['Close'].rolling(window=52).mean()

    # Extract 'Close' prices and split into training/testing sets
    dataset = data[['Close']].values
    training_data_len = int(len(dataset) * 0.8)  # 80% for training

    # Split into train and test data
    train_data = dataset[0:training_data_len]
    test_data = dataset[training_data_len:]

    # Scale data using MinMaxScaler (fit only on training data)
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)  # Use the same scaler

    # Prepare training data (past 60 days to predict next day)
    X_train, y_train = [], []
    for i in range(60, len(scaled_train)):
        X_train.append(scaled_train[i-60:i, 0])
        y_train.append(scaled_train[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Combine last 60 days of training data with test data
    combined_test_data = np.concatenate((scaled_train[-60:], scaled_test), axis=0)

    # Prepare test inputs
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

    # Inverse scaling to get actual prices
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Actual values (for comparison)
    y_train_actual = scaler.inverse_transform([y_train])
    y_test_actual = test_data

    # Prepare data for plotting
    train = data[:training_data_len]
    valid = data[training_data_len:].copy()
    valid['Predictions'] = test_predict

    # Create the figure
    valid.columns = valid.columns.get_level_values(0)
    fig = go.Figure()
    fig.add_scatter(x=valid['Date'], y=valid['Close'], name='Close', mode='lines')
    fig.add_scatter(x=valid['Date'], y=valid['52_MA'], name='52 Day SMA', mode='lines')
    fig.add_scatter(x=valid['Date'], y=valid['Predictions'], name='Predictions', mode='lines')
    fig.write_html(ticker.replace(".","_").replace("^","_")+'.html', auto_open=True)
    
    return

#%%
lst_tickers = ['^NSEI','GC=F','SI=F']
for ticker in lst_tickers:
    get_ts_prediction(ticker)

#%%
# get_ts_prediction('^INDIAVIX')
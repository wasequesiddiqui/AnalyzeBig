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

def set_seed(seed_value):
    """
    Set seed for reproducibility across NumPy, TensorFlow, and Python's random module.
    """
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

def get_ts_prediction(ticker):
    # set_seed(seed_value=13)

    # Get today's date
    today = datetime.today()

    # Calculate a start date (e.g., 30 days ago)
    start_date = today - timedelta(days=1826)  # 5 years ago
    str_end_date =today.strftime('%Y-%m-%d')
    str_start_date = start_date.strftime('%Y-%m-%d')

    # print date
    print(f"Today: {str_end_date}")
    print(f"Start Date (5 Years ago): {str_start_date}")
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
    valid["Log_Return_Actual"] = np.log(valid["Close"] / valid["Close"].shift(1))
    valid["Log_Return_Predicted"] = np.log(valid["Predictions"] / valid["Predictions"].shift(1))

    valid["Log_Return_Actual"] = valid["Log_Return_Actual"].fillna(0)
    valid["Log_Return_Predicted"] = valid["Log_Return_Predicted"].fillna(0)
    
    valid["Log_Return_Actual"] = valid["Log_Return_Actual"].round(4)
    valid["Log_Return_Predicted"] = valid["Log_Return_Predicted"].round(4)

    valid['Delta'] = valid['Close'] - valid['Predictions']
    valid['Delta_Ret'] = valid['Log_Return_Actual'] - valid['Log_Return_Predicted']
    valid['Delta_Ret'] = valid['Delta_Ret'].round(6)

    fig = go.Figure()
    fig.add_scatter(x=valid['Date'], y=valid['Close'], name='Close', mode='lines')
    fig.add_scatter(x=valid['Date'], y=valid['52_MA'], name='52 Day SMA', mode='lines')
    fig.add_scatter(x=valid['Date'], y=valid['Predictions'], name='Predictions', mode='lines')
    fig.write_html(ticker.replace(".","_").replace("^","_")+'.html', auto_open=True)

    fig = go.Figure(data=[go.Bar(x=valid['Date'], y=valid['Delta'], name="Deviation from Prediction", marker_color=valid['Delta'].apply(lambda x: 'green' if x > 0 else 'red'))])
    fig = add_median_line(fig,valid,'Delta',annotation_text_val="Median Negative Deviation",line_color_val="blue",type=1)
    fig.update_layout(title="Deviation from Prediction", xaxis_title="Date", yaxis_title="Deviation")
    fig.update_traces(marker=dict(line=dict(width=0.5, color='black')))
    fig.write_html(ticker.replace(".","_").replace("^","_")+'_Deviation.html', auto_open=True)

    fig = go.Figure(data=[go.Bar(x=valid['Date'], y=valid['Delta_Ret'], name="Return Deviation from Prediction", marker_color=valid['Delta_Ret'].apply(lambda x: 'green' if x > 0 else 'red'))])
    fig = add_median_line(fig,valid,'Delta_Ret',annotation_text_val="Median Negative Deviation",line_color_val="blue",type=1)
    fig.update_layout(title="Return Deviation from Prediction", xaxis_title="Date", yaxis_title="Return Deviation")
    fig.update_traces(marker=dict(line=dict(width=0.5, color='blue')))
    fig.write_html(ticker.replace(".","_").replace("^","_")+'_Deviation_Ret.html', auto_open=True)
    
    valid['Is_Negative'] = valid['Delta'] < 0
    valid['Negative_Streak'] = valid['Is_Negative'].astype(int).groupby((~valid['Is_Negative']).cumsum()).cumsum()
    fig = go.Figure(data=[go.Bar(x=valid['Date'], y=valid['Negative_Streak'], name="Negative Streak", marker_color='red')])
    fig = add_median_line(fig,valid,'Negative_Streak',annotation_text_val="Average Negative Streak",line_color_val="blue")
    fig.write_html(ticker.replace(".","_").replace("^","_")+'_Negative_Streak.html', auto_open=True)

    return

#%%
lst_tickers = ['GOLDBEES.NS']
for ticker in lst_tickers:
    get_ts_prediction(ticker)

#%%
# get_ts_prediction('^INDIAVIX')
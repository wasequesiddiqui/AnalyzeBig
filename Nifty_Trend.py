
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Fetch Nifty 50 data
ticker = "^NSEI"
data = yf.download(ticker, start="2018-01-01", end="2025-01-25")
data.reset_index(inplace=True)

# Step 2: Calculate 52-day moving average
data['52_MA'] = data['Close'].rolling(window=52).mean()

# Step 3: Plot the results
plt.figure(figsize=(14, 8))

# Plot actual Nifty 50 prices
plt.plot(data['Date'], data['Close'], label='Actual Nifty 50 Price', color='blue')

# Plot 52-day moving average
plt.plot(data['Date'], data['52_MA'], label='52-Day Moving Average', color='orange', linestyle='--')

# Add labels and title
plt.title('Nifty 50 Time Series with 52-Day Moving Average', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.legend()
plt.grid()
plt.show()
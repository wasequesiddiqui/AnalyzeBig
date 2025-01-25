#%%
import pandas as pd
import yfinance as yf

#%%
dat = yf.Ticker("MSFT")

#%%
dat.balance_sheet.head(20)
# %%

import streamlit as st
import Nifty_Trend_Lib as ntl
import plotly.express as px
import plotly.graph_objects as go

"""
sample demo code for charting out nifty 50 data
"""
# streamlit run Stream_Lit_App.py
# df_nsei = ntl.get_ts_prediction('^NSEI')
# fig = go.Figure()
# fig.add_scatter(x=df_nsei['Date'], y=df_nsei['Close'], name='Close', mode='lines')
# fig.add_scatter(x=df_nsei['Date'], y=df_nsei['52_MA'], name='52 Day SMA', mode='lines')
# fig.add_scatter(x=df_nsei['Date'], y=df_nsei['Predictions'], name='Predictions', mode='lines')
# st.dataframe(df_nsei)
# st.plotly_chart(fig)

st.title("Nifty 50 Streamlit App")
st.subheader("Using ML Models to predict Nifty 50 closing price")
st.divider()
st.header("Historical data is sourced using yfinance!")
st.divider()
st.text("Valid ticker is needed to use this app")
st.divider()
st.markdown("[Yahoo Finance](https://finance.yahoo.com/)")
st.markdown("---")
st.caption("Waseque Siddiqui")

def change():
    print("Checkbox value changed and ","Checker Session State is :",str(st.session_state.checker))

state = st.checkbox(label="Checkbox", value=True, on_change=change, key="checker")
if state:
    st.write("Hi the check box has been checked")
else:
    pass

rd_btn = st.radio(label="What are the scrips you want to track?", options={"NIFTY 50","NIFTY Midcap", "Nifty Smallcap"})
if(rd_btn):
    st.write("you have selected: ",rd_btn)
else:
    pass

def btn_click():
    print("Refresh button clicked!")
    
btn = st.button("Refresh", on_click=btn_click)
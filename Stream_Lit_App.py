import streamlit as st
import Nifty_Trend_Lib as ntl

# streamlit run Stream_Lit_App.py

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
st.dataframe(ntl.get_ts_prediction('^NSEI'))
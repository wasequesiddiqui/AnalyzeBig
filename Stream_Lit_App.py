import streamlit as st
import pandas as pd
import Nifty_Trend_Lib as ntl
import plotly.express as px
import plotly.graph_objects as go
import datetime
import time as t
import numpy as np
import altair as alt
import asyncio as syn

from matplotlib import pyplot as plt

x = np.linspace(0,10,100)

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

image = st.file_uploader("Please Upload An Image",type=["jpeg","jpg","png"])
if image is not None:
    st.image(image=image)

csv_file = st.file_uploader("Please upload a csv file", type=["csv"])
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write(df)

color = st.select_slider(
    "Select a color of the rainbow",
    options=[
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "indigo",
        "violet",
    ],
)
st.write("My favorite color is", color)

slider_val = st.slider("This is the slider value ", max_value=200, min_value=10, value=80, step=1)
st.write("My selected value: ", slider_val)

val_text = st.text_input("Please enter your name: ", max_chars=100)
st.write("The name value entered by you is: ", val_text)

val_text_area = st.text_area("Please enter address: ", max_chars=1000)
st.write("Address entered by you is: ", val_text_area)

dob = st.date_input("Please enter your date of birth: ", min_value=datetime.date(1950,1,1))
st.write("Your date of birth is: ",dob)

bar = st.progress(10)

# for i in range(10):
#     bar.progress((i+1)*10)
#     t.sleep(1)

st.sidebar.write("Sample Sidebar")
fig = plt.figure()
plt.plot(x,np.sin(x))
st.write(fig)

chart_data = pd.DataFrame()
chart_data['a'] = x
chart_data['b'] = np.sin(x)
chart_data['c'] = np.cos(x)
chart_data['d'] = chart_data['a'] + chart_data['b'] + chart_data['c']

c = (
   alt.Chart(chart_data)
   .mark_circle()
   .encode(x="a", y="b", size="c", color="c", tooltip=["a", "b", "c"])
)
st.altair_chart(c)

fig = px.scatter(chart_data, 
                 x="a", 
                 y="c",
                 color="b",
                 size="d",
                 hover_data=["a","b","d"])
st.plotly_chart(fig)

# callbacks
st.subheader("The below code uses callback")
st.divider()
def printer(name):
    st.write(name)
    print(name)
input = st.text_input("Enter your name: ")
s_btn = st.button("Submit")
if s_btn:
    st.checkbox("Want to display your name?", on_change=printer, args=(input,))

# Initialize session state variable
if "result" not in st.session_state:
    st.session_state.result = None

# Asynchronous function
async def async_function():
    await syn.sleep(2)  # Simulate async task
    return "Async operation completed!"

# Wrapper for running async task
def run_async_task():
    # Store the result in session state
    st.session_state.result = syn.run(async_function())

st.title("Streamlit Async with Session State")

# Button to trigger the async function
if st.button("Run Async Task"):
    run_async_task()

# Display the result from session state
if st.session_state.result:
    st.write(st.session_state.result)


    
"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Equity_Finder
streamlit run Equity_Evaluator.py
conda activate np_env
conda deactivate
"""

import yfinance as yf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import EQUITY_ANALYSER as ea

from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import datetime, timedelta
from plotly.subplots import make_subplots

sb = st.sidebar
sb.title("Evaluating the Ticker:")

# Textbox to capture ticker value(s)
ticker_input = sb.text_input("Enter Ticker Symbol(s):", value="INFY.NS", placeholder="e.g., INFY.NS or INFY.NS, TCS.NS, RELIANCE.NS")

# Button to trigger evaluation
if sb.button("Evaluate"):
    st.header(f"Evaluating Ticker(s): {ticker_input}")
    try:
        # Split the input by comma and strip whitespace
        tickers = [t.strip() for t in ticker_input.split(',')]
        
        # Collect results from all tickers
        all_results = []
        
        for ticker in tickers:
            try:
                result_df = ea.evaluate_ticker(ticker)
                all_results.append(result_df)
            except Exception as e:
                st.error(f"An error occurred while evaluating {ticker}: {e}")
        
        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            st.dataframe(combined_results)
            
            # Create Sankey diagram
            st.subheader("Sankey Diagram - Overall Score Distribution")
            
            # Prepare data for Sankey - only use Overall score columns
            sources = []
            targets = []
            values = []
            colors = []
            
            # Get only Overall columns (including Final_Overall_Score)
            metric_columns = [col for col in combined_results.columns if 'Overall' in col or 'Final' in col]
            
            for idx, row in combined_results.iterrows():
                ticker_name = row['Ticker_Name']
                
                for col in metric_columns:
                    sources.append(ticker_name)
                    targets.append(col)
                    value = row[col]
                    values.append(value)
                    
                    # Color based on score (green for high, red for low)
                    if pd.notna(value):
                        if value >= 75:
                            colors.append('rgba(0, 200, 0, 0.8)')  # Green
                        elif value >= 50:
                            colors.append('rgba(255, 165, 0, 0.8)')  # Orange
                        else:
                            colors.append('rgba(200, 0, 0, 0.8)')  # Red
                    else:
                        colors.append('rgba(128, 128, 128, 0.8)')  # Gray for NaN
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color='black', width=0.5),
                    label=list(combined_results['Ticker_Name'].unique()) + metric_columns
                ),
                link=dict(
                    source=[combined_results['Ticker_Name'].unique().tolist().index(s) for s in sources],
                    target=[len(combined_results['Ticker_Name'].unique()) + metric_columns.index(t) for t in targets],
                    value=values,
                    color=colors
                )
            )])
            
            fig.update_layout(
                title="Ticker Overall Scores Distribution",
                font=dict(size=12),
                height=600,
                width=1200
            )
            
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred: {e}")
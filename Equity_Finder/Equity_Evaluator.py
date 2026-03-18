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
final_tickers = ea.get_nifty50_non_banking_tickers()
final_nifty_next50_tickers = ea.get_nifty_next_50_non_banking_tickers()
final_tickers.extend(final_nifty_next50_tickers)
final_tickers_str = ", ".join(final_tickers)
ticker_input = sb.text_input("Enter Ticker Symbol(s):", value=final_tickers_str, placeholder="e.g., INFY.NS or INFY.NS, TCS.NS, RELIANCE.NS")

# Button to trigger evaluation
if sb.button("Evaluate"):
    st.header(f"Evaluating Ticker(s)")
    try:
        # Split the input by comma and strip whitespace
        tickers = [t.strip() for t in ticker_input.split(',')]
        total_tickers = len(tickers)
        
        # UI Elements for progress
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Collect results from all tickers
        all_results = []
        
        for i, ticker in enumerate(tickers):
            # Update status and progress
            status_text.text(f"Currently evaluating: {ticker} ({i+1}/{total_tickers})")
            progress_bar.progress((i + 1) / total_tickers)
            
            try:
                result_df = ea.evaluate_ticker(ticker)
                all_results.append(result_df)
            except Exception as e:
                st.error(f"An error occurred while evaluating {ticker}: {e}")
        
        # Clear progress indicators once done
        status_text.success(f"Evaluation complete for {total_tickers} tickers!")
        progress_bar.empty()

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_results = combined_results.sort_values(by='Final_Overall_Score', ascending=False).reset_index(drop=True)
            
            st.dataframe(combined_results)
            
            st.subheader("Sankey Diagram - Scores by Ticker")
            
            # --- FIXED SANKEY LOGIC ---
            metric_cols = [c for c in combined_results.columns if c.startswith('Overall')]
            unique_tickers = list(combined_results['Ticker_Name'].unique())
            
            # Combine nodes and build the diagram
            all_nodes = metric_cols + unique_tickers
            sources, targets, values, colors = [], [], [], []
            
            for _, row in combined_results.iterrows():
                t_idx = all_nodes.index(row['Ticker_Name'])
                for col in metric_cols:
                    val = row[col]
                    if pd.notna(val):
                        col_idx = all_nodes.index(col)
                        # Color logic
                        color = 'rgba(0, 200, 0, 0.6)' if val >= 65 else ('rgba(255, 165, 0, 0.6)' if val >= 40 else 'rgba(200, 0, 0, 0.6)')
                        
                        sources.append(col_idx)
                        targets.append(t_idx)
                        values.append(val)
                        colors.append(color)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(pad=15, thickness=20, label=all_nodes,
                          color=['rgba(100, 149, 237, 0.8)'] * len(metric_cols) + ['rgba(200, 100, 100, 0.8)'] * len(unique_tickers)),
                link=dict(source=sources, target=targets, value=values, color=colors)
            )])
            
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
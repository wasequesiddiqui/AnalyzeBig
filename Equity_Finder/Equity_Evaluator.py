"""
all the imports are included here
command to run streamlit app: cd C:\Github\AnalyzeBig\Equity_Finder
streamlit run Equity_Evaluator.py
conda activate np_env
conda deactivate
"""

import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import EQUITY_ANALYSER as ea

from plotly.subplots import make_subplots

# --- 1. INITIALIZE SESSION STATE ---
if 'combined_results' not in st.session_state:
    st.session_state.combined_results = None
if 'display_df' not in st.session_state:
    st.session_state.display_df = None
if 'last_ticker_input' not in st.session_state:
    st.session_state.last_ticker_input = ""

sb = st.sidebar
sb.title("Evaluating the Ticker:")

# Textbox to capture ticker value(s)
final_tickers = ea.get_nifty50_non_banking_tickers()
final_nifty_next50_tickers = ea.get_nifty_next_50_non_banking_tickers()
final_tickers.extend(final_nifty_next50_tickers)

# Fallback default tickers if fetching fails
if not final_tickers:
    final_tickers = [
        "ADANIENT.NS"
        , "ADANIPORTS.NS"
        , "APOLLOHOSP.NS"
        , "ASIANPAINT.NS"
        , "BAJAJ-AUTO.NS"
        , "BEL.NS"
        , "BHARTIARTL.NS"
        , "CIPLA.NS"
        , "COALINDIA.NS"
        , "DRREDDY.NS"
        , "EICHERMOT.NS"
        , "ETERNAL.NS"
        , "GRASIM.NS"
        , "HCLTECH.NS"
        , "HINDALCO.NS"
        , "HINDUNILVR.NS"
        , "ITC.NS"
        , "INFY.NS"
        , "INDIGO.NS"
        , "JSWSTEEL.NS"
        , "LT.NS"
        , "M&M.NS"
        , "MARUTI.NS"
        , "MAXHEALTH.NS"
        , "NTPC.NS"
        , "NESTLEIND.NS"
        , "ONGC.NS"
        , "POWERGRID.NS"
        , "RELIANCE.NS"
        , "SUNPHARMA.NS"
        , "TCS.NS"
        , "TATACONSUM.NS"
        , "TMPV.NS"
        , "TATASTEEL.NS"
        , "TECHM.NS"
        , "TITAN.NS"
        , "TRENT.NS"
        , "ULTRACEMCO.NS"
        , "WIPRO.NS"
        , "ABB.NS"
        , "ADANIENSOL.NS"
        , "ADANIGREEN.NS"
        , "ADANIPOWER.NS"
        , "AMBUJACEM.NS"
        , "DMART.NS"
        , "BPCL.NS"
        , "BOSCHLTD.NS"
        , "BRITANNIA.NS"
        , "CGPOWER.NS",]
    
final_tickers_str = ", ".join(final_tickers)

ticker_input = sb.text_area(
    "Enter Ticker Symbol(s):",
    value=final_tickers_str, 
    placeholder="e.g., INFY.NS, TCS.NS, RELIANCE.NS",
    height=400  # Adjust this value to make the box taller or shorter
)

evaluate_clicked = sb.button("Evaluate")

# Button to trigger evaluation
if evaluate_clicked:
    # Check if input is identical to the last run AND we have data
    input_unchanged = (ticker_input.strip() == st.session_state.last_ticker_input.strip())
    data_exists = st.session_state.combined_results is not None

    if input_unchanged and data_exists:
        st.toast("Input unchanged. Loading results from cache...", icon="⚡")
    else:
        # If input changed or no data exists, run the heavy analysis
        st.toast("New input detected. Starting fresh evaluation...", icon="🔍")
        try:
            tickers = [t.strip() for t in ticker_input.split(',')]
            total_tickers = len(tickers)
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            all_results = []
            
            for i, ticker in enumerate(tickers):
                status_text.text(f"Evaluating: {ticker} ({i+1}/{total_tickers})")
                progress_bar.progress((i + 1) / total_tickers)
                
                try:
                    result_df = ea.evaluate_ticker(ticker)
                    t_obj = yf.Ticker(ticker)
                    info = t_obj.info

                    result_df['Industry'] = info.get('industry', 'N/A')
                    result_df['Sector'] = info.get('sector', 'N/A')
                    result_df['P/E_Ratio'] = info.get('trailingPE')
                    result_df['P/B_Ratio'] = info.get('priceToBook')
                    result_df['Revenue'] = info.get('totalRevenue')
                    result_df['Profit'] = info.get('netIncomeToCommon')

                    book_val = info.get('bookValue')
                    shares = info.get('sharesOutstanding')
                    result_df['Shareholders_Equity'] = (book_val * shares) if book_val and shares else None

                    all_results.append(result_df)
                except Exception as e:
                    st.error(f"Error with {ticker}: {e}")
            
            if all_results:
                combined = pd.concat(all_results, ignore_index=True)
                combined = combined.sort_values(by='Final_Overall_Score', ascending=False).reset_index(drop=True)
                
                # Save results AND the input string to session state
                st.session_state.combined_results = combined
                st.session_state.last_ticker_input = ticker_input.strip()

                # Create and save formatted display version
                display = combined.copy()
                def format_currency(value):
                    if pd.isna(value) or value == 0: return "N/A"
                    if abs(value) >= 1_000_000_000: return f"{value / 1_000_000_000:,.2f} B"
                    if abs(value) >= 1_000_000: return f"{value / 1_000_000:,.2f} M"
                    return f"{value:,.2f}"

                for col in ['Revenue', 'Profit', 'Shareholders_Equity']:
                    if col in display.columns:
                        display[col] = display[col].apply(format_currency)
                
                st.session_state.display_df = display

            status_text.success("Evaluation complete!")
            progress_bar.empty()

        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- 3. DISPLAY LOGIC ---
if st.session_state.combined_results is not None:
    df_num = st.session_state.combined_results
    df_disp = st.session_state.display_df

    st.write("### Fundamental Analysis & Scores")
    st.dataframe(df_disp, use_container_width=True)

    # --- Industry Performance Graphs (Corrected for Graph Objects) ---
    st.divider()
    st.header("Industry Benchmarking")

    # Aggregate by industry for the graphs
    industry_grouped = df_num.groupby('Industry').agg({
        'Shareholders_Equity': 'mean',
        'P/B_Ratio': 'mean',
        'Final_Overall_Score': 'mean'
    }).reset_index().sort_values(by='Final_Overall_Score', ascending=False)

    fig_industry = make_subplots(
        rows=3, cols=1, 
        subplot_titles=("Avg Market Cap (Equity)", "Avg Price-to-Book (P/B)", "Avg Overall Score"),
        vertical_spacing=0.18 
    )

    # 1. Market Cap Trace
    fig_industry.add_trace(
        go.Bar(
            x=industry_grouped['Industry'], 
            y=industry_grouped['Shareholders_Equity'], 
            marker_color='teal',
            text=industry_grouped['Shareholders_Equity'], # Pass the data
            texttemplate='%{text:.2s}',                  # Format as "2s" (e.g. 1.5B)
            textposition='outside',
            name="Equity"
        ), 
        row=1, col=1
    )

    # 2. P/B Ratio Trace
    fig_industry.add_trace(
        go.Bar(
            x=industry_grouped['Industry'], 
            y=industry_grouped['P/B_Ratio'], 
            marker_color='indianred',
            text=industry_grouped['P/B_Ratio'],           # Pass the data
            texttemplate='%{text:.2f}',                  # Format as float
            textposition='outside',
            name="P/B"
        ), 
        row=2, col=1
    )

    # 3. Overall Score Trace
    fig_industry.add_trace(
        go.Bar(
            x=industry_grouped['Industry'], 
            y=industry_grouped['Final_Overall_Score'], 
            marker_color='mediumseagreen',
            text=industry_grouped['Final_Overall_Score'], # Pass the data
            texttemplate='%{text:.2f}',                  # Format as float
            textposition='outside',
            name="Score"
        ), 
        row=3, col=1
    )

    # Layout adjustments
    fig_industry.update_layout(
        height=1300, 
        showlegend=False, 
        margin=dict(b=150, t=100), # Increased bottom margin for labels
        uniformtext_mode='hide', 
        uniformtext_minsize=8
    )

    fig_industry.update_xaxes(tickangle=45)
    st.plotly_chart(fig_industry, use_container_width=True)
else:
    st.info("Enter tickers and click 'Evaluate' to generate the report.")
"""
Stock Research AI Agent — Streamlit Dashboard
==============================================
An interactive stock research dashboard built with Streamlit that combines
real-time market data, financial news aggregation, and AI-powered sentiment
analysis — all from a modern web UI.

Usage
-----
    streamlit run streamlit_app.py

Environment Variables (.env)
-----------------------------
    DEEPSEEK_API_KEY    — API key for DeepSeek chat completions.
    NEWS_API_KEY        — API key for NewsAPI.org (free tier: 100 req/day).
"""

import os
import re
import streamlit as st
import yfinance as yf
import pandas as pd

from datetime import datetime
from textblob import TextBlob
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# =========================================================
# PAGE CONFIG — must be the first Streamlit command
# =========================================================
st.set_page_config(
    page_title="Stock Research AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================
load_dotenv()

# =========================================================
# SESSION STATE INIT
# =========================================================
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.stock_data = None
    st.session_state.news = None
    st.session_state.sentiment_data = None
    st.session_state.analysis = None
    st.session_state.parsed_sections = None
    st.session_state.ticker = None
    st.session_state.company_name = None

# =========================================================
# LLM SETUP
# =========================================================
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

# =========================================================
# NEWS API
# =========================================================
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

# =========================================================
# HELPERS
# =========================================================

def strip_exchange_suffix(ticker: str) -> str:
    """Strip Indian exchange suffix (.NS / .BO) from a ticker."""
    return re.split(r'\.(NS|BO)$', ticker.upper().strip())[0]


def format_volume(vol: int) -> str:
    """Format volume in Indian number system."""
    if vol >= 10000000:
        return f"{vol / 10000000:.2f} Cr"
    elif vol >= 100000:
        return f"{vol / 100000:.2f} L"
    return f"{vol:,}"


def get_sentiment_emoji(s: str) -> str:
    mapping = {
        "positive": "📈",
        "negative": "📉",
        "neutral": "➡️",
        "unknown": "❓",
        "error": "⚠️"
    }
    return mapping.get(s.lower(), "❓")


def get_sentiment_color(s: str) -> str:
    mapping = {
        "positive": "green",
        "negative": "red",
        "neutral": "orange",
        "unknown": "blue",
    }
    return mapping.get(s.lower(), "gray")

# =========================================================
# TOOLS  (same logic as original)
# =========================================================

@tool
def get_stock_price(ticker: str) -> dict:
    """Fetch the latest stock market data (OHLCV) for a given ticker symbol."""
    try:
        ticker = ticker.upper().strip()
        possible_tickers = [
            f"{ticker}.NS",
            f"{ticker}.BO",
            ticker
        ]
        stock_data = None
        for symbol in possible_tickers:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                exchange = "Global"
                if symbol.endswith(".NS"):
                    exchange = "NSE India"
                elif symbol.endswith(".BO"):
                    exchange = "BSE India"
                stock_data = {
                    "ticker": symbol,
                    "exchange": exchange,
                    "current_price": round(latest["Close"], 2),
                    "open": round(latest["Open"], 2),
                    "high": round(latest["High"], 2),
                    "low": round(latest["Low"], 2),
                    "volume": int(latest["Volume"])
                }
                break
        if stock_data is None:
            return {"error": "No stock data found."}
        return stock_data
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_news(company_name: str) -> list:
    """Fetch the latest financial news headlines for a given company."""
    try:
        response = newsapi.get_everything(
            q=f"{company_name} stock",
            language="en",
            sort_by="publishedAt",
            page_size=5
        )
        articles = response.get("articles", [])
        if not articles:
            return []
        formatted_news = []
        for article in articles:
            formatted_news.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", ""),
                "publishedAt": article.get("publishedAt", "")
            })
        return formatted_news
    except Exception as e:
        return [{"title": "Error", "description": str(e), "url": "", "source": "", "publishedAt": ""}]


@tool
def analyze_news_sentiment(company_name: str) -> dict:
    """Analyze the aggregate market sentiment for a company based on recent news."""
    try:
        response = newsapi.get_everything(
            q=f"{company_name} stock",
            language="en",
            page_size=20
        )
        articles = response.get("articles", [])
        if not articles:
            return {
                "sentiment": "Unknown",
                "score": 0,
                "reason": (
                    f"No news articles found for '{company_name}'. "
                    "The NewsAPI free tier may not have recent coverage."
                )
            }
        scores = []
        for article in articles:
            text = (
                (article.get("title") or "") + " " +
                (article.get("description") or "")
            )
            polarity = TextBlob(text).sentiment.polarity
            scores.append(polarity)
        avg_score = sum(scores) / len(scores)
        if avg_score > 0.15:
            sentiment = "Positive"
        elif avg_score < -0.15:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return {
            "sentiment": sentiment,
            "score": round(avg_score, 2),
            "articles_analyzed": len(scores)
        }
    except Exception as e:
        return {"sentiment": "Error", "score": 0, "error": str(e)}


# =========================================================
# AI ANALYSIS
# =========================================================

def generate_analysis(ticker: str, stock_data: dict, sentiment_data: dict, news: list) -> str:
    """Generate a structured AI investment analysis using DeepSeek."""
    prompt = f"""
You are a senior Wall Street analyst.

Analyze this stock.

Ticker:
{ticker}

Stock Data:
{stock_data}

Sentiment:
{sentiment_data}

News:
{news}

Provide a structured analysis with EXACTLY these four sections.
Use the exact section headers below (with the emoji), and format each as flowing paragraphs:

## 📊 Financial Outlook
[2-3 paragraphs on current financial performance and short-term outlook]

## ⚠️ Key Risks
[2-3 paragraphs on major risks investors should be aware of]

## 🚀 Growth Potential
[2-3 paragraphs on upside opportunities and catalysts]

## 🎯 Investment Recommendation
[1-2 paragraphs with a clear BUY / HOLD / SELL stance and reasoning.
In THIS section ONLY, wrap the single most critical sentence in each paragraph with <mark>...</mark> tags so it stands out visually.]

Keep it concise, data-driven, and professional. Do not add any preamble before the first section header.
Do NOT use <mark> tags in any other section.
"""
    response = llm.invoke(prompt)
    return response.content


# =========================================================
# ANALYSIS PARSER
# =========================================================

def parse_analysis(analysis: str) -> list:
    """Parse the LLM markdown analysis into structured sections."""
    section_map = {
        "Financial Outlook":          {"emoji": "📊", "is_recommendation": False},
        "Key Risks":                  {"emoji": "⚠️",  "is_recommendation": False},
        "Growth Potential":           {"emoji": "🚀", "is_recommendation": False},
        "Investment Recommendation":  {"emoji": "🎯", "is_recommendation": True},
    }
    BUY_WORDS  = ["buy", "strong buy", "accumulate", "outperform", "overweight"]
    SELL_WORDS = ["sell", "strong sell", "underperform", "underweight", "avoid"]

    def detect_verdict(text: str) -> str:
        t = text.strip()
        m = re.match(r'^(?:\*\*)?\s*(BUY|HOLD|SELL)\s*(?:\*\*)?\s*[\.\:\,\-]', t, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        t_lower = t.lower()
        for w in BUY_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', t_lower):
                return "BUY"
        for w in SELL_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', t_lower):
                return "SELL"
        return "HOLD"

    parsed_sections = []
    raw_sections = re.split(r'##\s+', analysis.strip())
    for raw in raw_sections:
        if not raw.strip():
            continue
        lines = raw.strip().splitlines()
        header_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', body) if p.strip()]
        matched_title = header_line
        emoji = "📌"
        is_rec = False
        for key, meta in section_map.items():
            if key.lower() in header_line.lower():
                matched_title = key
                emoji = meta["emoji"]
                is_rec = meta["is_recommendation"]
                break
        verdict = detect_verdict(body) if is_rec else None
        parsed_sections.append({
            "title": matched_title,
            "emoji": emoji,
            "paragraphs": paragraphs,
            "is_recommendation": is_rec,
            "verdict": verdict
        })
    return parsed_sections


# =========================================================
# STREAMLIT UI
# =========================================================

def render_header():
    """Render the app header."""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📈 Stock Research AI Agent")
        st.markdown(
            "<p style='color: var(--text-color-secondary); font-size: 1.05rem;'>"
            "Real-time market data · News aggregation · AI-powered analysis</p>",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"<p style='text-align: right; color: var(--text-color-secondary); "
            f"font-size: 0.8rem; margin-top: 1.5rem;'>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",
            unsafe_allow_html=True
        )


def render_sidebar():
    """Render the sidebar with inputs and controls."""
    with st.sidebar:
        st.markdown("## 🔍 Stock Lookup")
        ticker = st.text_input(
            "Ticker Symbol",
            placeholder="e.g. INFY, TCS, AAPL",
            help="Enter the stock ticker symbol (Indian stocks: NSE/BSE auto-detected)"
        ).strip().upper()
        company_name = st.text_input(
            "Company Name",
            placeholder="e.g. Infosys, Tata Consultancy Services",
            help="Full company name used for news & sentiment search"
        ).strip()
        analyze_btn = st.button(
            "🚀 Analyze Stock",
            type="primary",
            use_container_width=True,
            disabled=not (ticker and company_name)
        )

        st.markdown("---")
        st.markdown("#### ⚙️ Configuration")
        api_key_ok = bool(os.getenv("DEEPSEEK_API_KEY"))
        news_key_ok = bool(os.getenv("NEWS_API_KEY"))
        st.markdown(
            f"- DeepSeek API: {'✅ Configured' if api_key_ok else '❌ Missing'}"
        )
        st.markdown(
            f"- NewsAPI: {'✅ Configured' if news_key_ok else '❌ Missing'}"
        )

        st.markdown("---")
        st.markdown(
            "<p style='font-size: 0.75rem; color: var(--text-color-secondary);'>"
            "Data sourced from Yahoo Finance & NewsAPI. "
            "This is not financial advice.</p>",
            unsafe_allow_html=True
        )

    return ticker, company_name, analyze_btn


def render_metrics(stock_data: dict):
    """Render the key metrics strip."""
    spread_pct = round((stock_data["high"] - stock_data["low"]) / stock_data["low"] * 100, 1)
    cols = st.columns(5)
    metrics = [
        ("💰 Current Price", f"₹{stock_data['current_price']:,}", f"{stock_data['exchange']}"),
        ("📂 Open", f"₹{stock_data['open']:,}", "Today's open"),
        ("📈 High", f"₹{stock_data['high']:,}", "Day high"),
        ("📉 Low", f"₹{stock_data['low']:,}", "Day low"),
        ("📊 Volume", format_volume(stock_data['volume']), f"Range: {spread_pct}%"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value, help=sub)


def render_sentiment(sentiment_data: dict):
    """Render sentiment analysis card."""
    s = sentiment_data.get("sentiment", "Unknown")
    score = sentiment_data.get("score", 0)
    emoji = get_sentiment_emoji(s)
    color = get_sentiment_color(s)

    st.markdown("### 📊 Market Sentiment")
    st.markdown(
        f"<div style='display: flex; align-items: center; gap: 0.5rem; "
        f"padding: 0.5rem 1rem; border-radius: 999px; "
        f"background-color: color-mix(in srgb, var(--{color}) 15%, transparent); "
        f"color: var(--{color}); font-weight: 600; width: fit-content; "
        f"margin-bottom: 1rem;'>"
        f"{emoji} {s}</div>",
        unsafe_allow_html=True
    )

    if s.lower() not in ("unknown", "error"):
        # Sentiment progress bar
        pct = min(max((score + 1) / 2 * 100, 3), 97)
        bar_color = {"positive": "#1a6b3a", "negative": "#b01a1a", "neutral": "#92600a"}.get(s.lower(), "#1a3a6b")
        st.markdown(
            f"<div style='height: 8px; background: var(--border-color); "
            f"border-radius: 99px; overflow: hidden; margin: 0.5rem 0;'>"
            f"<div style='height: 100%; width: {pct}%; background: {bar_color}; "
            f"border-radius: 99px; transition: width 1s ease;'></div></div>",
            unsafe_allow_html=True
        )
        details = f"**Score:** {score}"
        if sentiment_data.get("articles_analyzed"):
            details += f" · Based on {sentiment_data['articles_analyzed']} articles"
        st.markdown(f"<p style='font-size: 0.85rem;'>{details}</p>", unsafe_allow_html=True)

    if sentiment_data.get("reason"):
        st.info(sentiment_data["reason"])
    if sentiment_data.get("error"):
        st.error(sentiment_data["error"])


def render_news(news: list):
    """Render news articles."""
    st.markdown("### 📰 Latest News")
    if not news:
        st.caption("No recent news found.")
        return
    for i, article in enumerate(news):
        with st.container(border=True):
            st.markdown(f"**{article['title']}**")
            if article.get("description"):
                st.markdown(
                    f"<p style='font-size: 0.85rem; color: var(--text-color-secondary);'>"
                    f"{article['description']}</p>",
                    unsafe_allow_html=True
                )
            meta_parts = []
            if article.get("source"):
                meta_parts.append(f"📰 {article['source']}")
            if article.get("publishedAt"):
                try:
                    dt = datetime.fromisoformat(article['publishedAt'].replace("Z", "+00:00"))
                    meta_parts.append(f"🕐 {dt.strftime('%b %d, %Y %H:%M')}")
                except Exception:
                    pass
            if meta_parts:
                st.markdown(
                    f"<p style='font-size: 0.75rem; color: var(--text-color-secondary);'>{' · '.join(meta_parts)}</p>",
                    unsafe_allow_html=True
                )
            if article.get("url"):
                st.markdown(f"[Read more →]({article['url']})")


def render_analysis(parsed_sections: list):
    """Render the AI analysis sections."""
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Investment Analysis")
    st.caption(f"Generated by Senior Wall Street AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not parsed_sections:
        st.warning("No analysis sections were parsed.")
        return

    cols = st.columns(2)
    for idx, section in enumerate(parsed_sections):
        is_rec = section["is_recommendation"]
        col_idx = 0 if is_rec else (idx % 2)

        if is_rec:
            # Recommendation spans full width
            with st.container():
                _render_recommendation(section)
        else:
            with cols[col_idx]:
                with st.container(border=True):
                    st.markdown(f"### {section['emoji']} {section['title']}")
                    for para in section.get("paragraphs", []):
                        st.markdown(para)


def _render_recommendation(section: dict):
    """Render the recommendation section with special styling."""
    verdict = section.get("verdict", "HOLD")
    verdict_color = {"BUY": "green", "HOLD": "orange", "SELL": "red"}.get(verdict, "gray")
    verdict_emoji = {"BUY": "✅", "HOLD": "⏸️", "SELL": "❌"}.get(verdict, "ℹ️")

    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(
                f"<div style='display: flex; align-items: center; gap: 0.4rem; "
                f"padding: 0.35rem 0.8rem; border-radius: 999px; "
                f"background-color: color-mix(in srgb, var(--{verdict_color}) 15%, transparent); "
                f"color: var(--{verdict_color}); font-weight: 600; font-size: 0.8rem; "
                f"width: fit-content;'>{verdict_emoji} {verdict}</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(f"### {section['emoji']} {section['title']}")

        for para in section.get("paragraphs", []):
            # Replace <mark> tags with Streamlit markdown highlights
            cleaned = re.sub(r'</?mark>', '**', para)
            st.markdown(cleaned)


def run_analysis(ticker: str, company_name: str):
    """Execute the full analysis pipeline."""
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    try:
        # Step 1: Stock price
        status_text.info("📡 Fetching stock price data...")
        progress_bar.progress(15, text="Fetching stock price data...")
        stock_data = get_stock_price.invoke(ticker)
        if "error" in stock_data:
            st.error(f"⚠️ Stock data error: {stock_data['error']}")
            return
        st.session_state.stock_data = stock_data

        # Step 2: News
        status_text.info("📰 Fetching latest news...")
        progress_bar.progress(35, text="Fetching latest news...")
        news = get_stock_news.invoke(company_name)
        st.session_state.news = news

        # Step 3: Sentiment
        status_text.info("📊 Analyzing market sentiment...")
        progress_bar.progress(55, text="Analyzing market sentiment...")
        sentiment_data = analyze_news_sentiment.invoke(company_name)
        st.session_state.sentiment_data = sentiment_data

        # Step 4: AI Analysis
        status_text.info("🤖 Generating AI investment analysis...")
        progress_bar.progress(75, text="Generating AI investment analysis...")
        analysis = generate_analysis(ticker, stock_data, sentiment_data, news)
        st.session_state.analysis = analysis

        # Step 5: Parse
        parsed_sections = parse_analysis(analysis)
        st.session_state.parsed_sections = parsed_sections
        st.session_state.ticker = ticker
        st.session_state.company_name = company_name

        # Done
        progress_bar.progress(100, text="✅ Analysis complete!")
        status_text.success("✅ Analysis complete!")
        st.session_state.analysis_complete = True
        st.rerun()

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        progress_bar.empty()
        status_text.empty()


# =========================================================
# MAIN APP
# =========================================================

def main():
    render_header()
    ticker, company_name, analyze_btn = render_sidebar()

    # ── Trigger analysis ──
    if analyze_btn and ticker and company_name:
        st.session_state.analysis_complete = False
        run_analysis(ticker, company_name)

    # ── Results display ──
    if st.session_state.analysis_complete:
        stock_data = st.session_state.stock_data
        sentiment_data = st.session_state.sentiment_data
        news = st.session_state.news
        parsed_sections = st.session_state.parsed_sections
        ticker_display = st.session_state.ticker

        # ── Hero Section ──
        col1, col2 = st.columns([2, 1])
        with col1:
            clean_ticker = strip_exchange_suffix(ticker_display)
            st.markdown(
                f"<h1 style='font-size: 2.5rem; margin-bottom: 0;'>{clean_ticker}</h1>"
                f"<p style='color: var(--text-color-secondary); font-size: 0.9rem;'>{stock_data['exchange']}</p>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<p style='font-size: 0.75rem; color: var(--text-color-secondary); margin-bottom: 0;'>Current Price</p>"
                f"<h1 style='font-size: 2.5rem;'>₹{stock_data['current_price']:,}</h1>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ── Metrics Strip ──
        render_metrics(stock_data)

        st.divider()

        # ── Sentiment + News (two columns) ──
        sent_col, news_col = st.columns(2)
        with sent_col:
            render_sentiment(sentiment_data)
        with news_col:
            render_news(news)

        # ── AI Analysis ──
        if parsed_sections:
            render_analysis(parsed_sections)

        # ── Footer ──
        st.divider()
        st.caption(
            "This report is for informational purposes only and does not constitute "
            f"financial advice. · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    else:
        # ── Welcome / idle state ──
        st.markdown("## 👋 Welcome to Stock Research AI Agent")
        st.markdown(
            "Enter a stock ticker and company name in the sidebar, then click "
            "**🚀 Analyze Stock** to get started.\n\n"
            "The agent will:\n"
            "1. 📡 Fetch real-time stock price data\n"
            "2. 📰 Retrieve latest financial news\n"
            "3. 📊 Analyze market sentiment via NLP\n"
            "4. 🤖 Generate AI-powered investment analysis\n\n"
            "---\n"
            "**Supported Markets:** NSE India (.NS), BSE India (.BO), Global (NYSE/NASDAQ)"
        )


if __name__ == "__main__":
    main()

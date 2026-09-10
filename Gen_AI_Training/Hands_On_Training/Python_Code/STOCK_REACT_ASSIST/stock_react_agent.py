"""
Stock Research AI Agent — HTML Dashboard Version
=================================================
Please install the required dependencies from requirements.txt before running this script.
An autonomous stock research agent that combines real-time market data, financial
news aggregation, and AI-powered sentiment analysis to produce a professional,
browser-ready HTML investment report.

Architecture Overview
---------------------
The agent follows a ReAct (Reasoning + Acting) pattern using LangChain:

1.  **Data Collection Layer** — Three @tool-decorated functions that the LLM can
    invoke autonomously:
    *   `get_stock_price`   — Fetches OHLCV data from Yahoo Finance (yfinance).
    *   `get_stock_news`    — Retrieves latest headlines via NewsAPI.
    *   `analyze_news_sentiment` — Computes sentiment polarity using TextBlob.

2.  **AI Analysis Layer** — Two LLM agents run IN PARALLEL (threads):
    *   `generate_analysis()` — the fundamental/sentiment analyst (senior
        Wall Street role) producing four clearly labeled sections.
    *   `generate_technical_analysis()` — the quantitative technical analyst
        that reads the DataFrame from `TECH_ANALYSIS.py`, analyses every
        technical indicator (price & volume), and classifies the stock into
        STRONG BUY / BUY / SELL / STRONG SELL.

3.  **Presentation Layer** — `generate_html_report()` takes the markdown from
    both agents, parses it into structured sections, detects the verdicts
    (fundamental BUY/HOLD/SELL and the 4-flag technical verdict), merges them
    into an Overall Summary, and renders a responsive, animated HTML dashboard
    using Jinja2 templates with embedded CSS and JavaScript.

4.  **Orchestration** — `main()` wires everything together: prompts the user for
    a ticker and company name, invokes the data tools, computes the technical
    indicator DataFrame, runs both AI agents in parallel, generates the HTML
    report, and automatically opens it in the default browser.

Dependencies
------------
*   `yfinance`       — Yahoo Finance market data.
*   `newsapi-python`  — NewsAPI client for financial headlines.
*   `textblob`        — Lexicon-based sentiment analysis.
*   `langchain`       — Tool decorator and LLM abstraction.
*   `langchain-openai`— OpenAI-compatible chat model wrapper (used with DeepSeek).
*   `jinja2`          — HTML templating engine.
*   `python-dotenv`   — Loads API keys from a .env file.

Environment Variables (.env)
-----------------------------
*   `DEEPSEEK_API_KEY` — API key for DeepSeek chat completions.
*   `NEWS_API_KEY`     — API key for NewsAPI.org (free tier: 100 req/day).

Output
------
A self-contained HTML file named `{TICKER}_REPORT_{YYYY_MM_DD}.html` that
displays price metrics, market sentiment, latest news, and AI-generated
investment analysis with visual styling.
"""

from __future__ import annotations  # Lazy (string) type hints — cleaner for DataFrames

import os
import re
import webbrowser
import yfinance as yf  # Yahoo Finance — fetches real-time & historical stock data

import pandas as pd  # DataFrames (type hints + indicator value lookups from TECH_ANALYSIS)

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor  # Runs the two AI agents in parallel
from textblob import TextBlob  # Lexicon-based NLP sentiment analyzer
from dotenv import load_dotenv  # Loads .env file into os.environ
from newsapi import NewsApiClient  # NewsAPI.org client for financial headlines
from jinja2 import Template  # Jinja2 HTML templating engine
from markupsafe import escape as _html_escape  # Escape plain text before adding markup
from langchain.tools import tool  # Decorator to register functions as LLM-callable tools
from langchain_openai import ChatOpenAI  # OpenAI-compatible chat model (used with DeepSeek)

import TECH_ANALYSIS  # Builds the 5-year technical-indicator DataFrame (OHLCV + indicators)

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================
# dotenv reads key-value pairs from a local `.env` file and injects them
# into `os.environ`.  This keeps API keys out of source control.
# Example `.env` content:
#   DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
#   NEWS_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxx

load_dotenv()

# =========================================================
# LLM SETUP
# =========================================================
# We instantiate a ChatOpenAI client pointed at DeepSeek's API endpoint.
# DeepSeek exposes an OpenAI-compatible REST API, so we can reuse
# langchain-openai's ChatOpenAI with a custom base_url.
# - model       : which DeepSeek model to use ("deepseek-chat" is their V3).
# - api_key     : read from environment (set in .env).
# - base_url    : DeepSeek's API base URL.
# - temperature : 0 = deterministic output (best for factual analysis).

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)

# =========================================================
# NEWS API
# =========================================================
# NewsApiClient wraps the NewsAPI.org REST API.
# The free tier allows 100 requests/day and returns articles up to 1 month old.
# We pass the API key at construction time; all subsequent calls use this client.

newsapi = NewsApiClient(
    api_key=os.getenv("NEWS_API_KEY")
)

# =========================================================
# HELPERS
# =========================================================

def strip_exchange_suffix(ticker: str) -> str:
    """
    Strip the Indian exchange suffix (``.NS`` for NSE or ``.BO`` for BSE)
    from a ticker string so it can be used in news queries where the
    raw company symbol is expected.

    Args:
        ticker (str): A ticker symbol that may or may not include an
                      exchange suffix, e.g. ``"INFY.NS"`` or ``"RELIANCE.BO"``.

    Returns:
        str: The ticker with the exchange suffix removed, uppercased.
             ``"INFY.NS"`` → ``"INFY"``
             ``"TCS.BO"``  → ``"TCS"``
             ``"AAPL"``    → ``"AAPL"``  (no suffix — returned as-is)

    Example:
        >>> strip_exchange_suffix("infy.ns")
        'INFY'
        >>> strip_exchange_suffix("RELIANCE.BO")
        'RELIANCE'
        >>> strip_exchange_suffix("AAPL")
        'AAPL'
    """
    # Use regex to split on ".NS" or ".BO" (case-insensitive at the END of string)
    # and take the first part (the base ticker).
    return re.split(r'\.(NS|BO)$', ticker.upper().strip())[0]


# ── Bull / bear emphasis ──────────────────────────────────────────
# Directional wording is emphasised everywhere it is rendered: "bull" /
# "bullish" in bold green, "bear" / "bearish" in bold red. The colours match
# the --green / --red custom properties of the report stylesheet so the
# highlight blends with the rest of the palette in both the HTML report and
# the Streamlit UI.
BULL_BEAR_RE = re.compile(r'\b(bullish|bull|bearish|bear)\b', re.IGNORECASE)
BULL_COLOR = "#1a6b3a"   # deep green
BEAR_COLOR = "#b01a1a"   # deep red


def _bull_bear_span(match: "re.Match") -> str:
    """Replacement callback: wrap one bull/bear word in a coloured bold tag."""
    word = match.group(1)
    color = BULL_COLOR if word.lower().startswith("bull") else BEAR_COLOR
    return f'<b style="color:{color}">{word}</b>'


def highlight_bull_bear(text: str) -> str:
    """
    Bold-colour every "bull"/"bullish" (green) and "bear"/"bearish" (red).

    Use this variant for text that ALREADY contains intentional HTML (e.g. the
    ``<mark>`` tags the LLM emits in the recommendation section); the text is
    passed through untouched apart from the inserted tags.

    Args:
        text (str): The text to decorate.

    Returns:
        str: The text with each directional word wrapped in ``<b style=...>``.
    """
    return BULL_BEAR_RE.sub(_bull_bear_span, text or "")


def highlight_bull_bear_escaped(text: str) -> str:
    """
    HTML-escape plain text, then bold-colour its bull/bear words.

    Escaping first guarantees a stray ``<`` / ``&`` in the LLM output cannot be
    interpreted as markup once the decorated string is emitted with ``| safe``.

    Args:
        text (str): Untrusted, plain text.

    Returns:
        str: Escaped text with each directional word wrapped in ``<b>``.
    """
    return BULL_BEAR_RE.sub(_bull_bear_span, str(_html_escape(text or "")))


# =========================================================
# TOOLS
# =========================================================

@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest stock market data (OHLCV) for a given ticker symbol.

    This tool fetches the symbol EXACTLY as supplied — the caller must pass
    the complete Yahoo Finance ticker, including any exchange suffix
    (``"INFY.NS"`` for NSE, ``"RELIANCE.BO"`` for BSE, ``"AAPL"`` for US
    markets).  No ``.NS`` / ``.BO`` suffix is ever appended or probed.

    **LangChain Tool** — when decorated with ``@tool``, the function's
    docstring becomes the description the LLM sees when deciding whether
    to invoke this tool.  Keep it descriptive but concise.

    Args:
        ticker (str): The complete ticker symbol to look up (e.g. ``"INFY.NS"``,
                      ``"TCS.BO"``, ``"AAPL"``).  Case-insensitive; leading/
                      trailing whitespace is stripped.

    Returns:
        dict: On success, a dictionary with the following keys:
            - **ticker** (str)      : The matched symbol with exchange suffix.
            - **exchange** (str)    : ``"NSE India"``, ``"BSE India"``, or ``"Global"``.
            - **current_price** (float) : Latest closing price (rounded to 2 dp).
            - **open** (float)      : Opening price for the latest trading day.
            - **high** (float)      : Day high.
            - **low** (float)       : Day low.
            - **volume** (int)      : Number of shares traded.
        On failure, a dict with a single key:
            - **error** (str)       : Human-readable error message.

    Example:
        >>> get_stock_price.invoke("INFY.NS")
        {
            "ticker": "INFY.NS",
            "exchange": "NSE India",
            "current_price": 1450.50,
            "open": 1445.00,
            "high": 1455.00,
            "low": 1440.00,
            "volume": 3250000
        }

    Implementation Notes
    --------------------
    *   Uses ``yfinance`` under the hood, which scrapes Yahoo Finance.
    *   ``stock.history(period="5d")`` requests 5 days of data to ensure
        we have at least one full trading day (handles weekends/holidays).
    *   ``hist.iloc[-1]`` grabs the most recent row regardless of date.
    *   The try/except wraps the entire logic so any API failure, network
        error, or unexpected data shape returns a graceful error dict
        rather than crashing the agent.
    """

    try:
        # Consume the value exactly as given: the caller supplies the complete
        # Yahoo Finance symbol, so no ".NS" / ".BO" suffix is appended here.
        symbol = ticker.upper().strip()

        # Create a yfinance Ticker object for this symbol
        stock = yf.Ticker(symbol)

        # Request 5 trading days of 1-day-interval OHLCV bars.
        # 5 days ensures we cover a weekend gap and still get a price.
        hist = stock.history(period="5d")

        # An empty DataFrame means yfinance couldn't find this symbol
        if hist.empty:
            return {
                "error": f"No stock data found for '{symbol}'."
            }

        # Get the most recent row (latest trading day's bar)
        latest = hist.iloc[-1]

        # Determine human-readable exchange label
        exchange = "Global"
        if symbol.endswith(".NS"):
            exchange = "NSE India"
        elif symbol.endswith(".BO"):
            exchange = "BSE India"

        # Build the result dictionary with rounded values
        return {
            "ticker": symbol,
            "exchange": exchange,
            "currency": "INR" if exchange in ("NSE India", "BSE India") else "USD",
            "current_price": round(latest["Close"], 2),
            "open": round(latest["Open"], 2),
            "high": round(latest["High"], 2),
            "low": round(latest["Low"], 2),
            "volume": int(latest["Volume"])
        }

    except Exception as e:
        # Catch-all: network errors, API changes, unexpected data shapes, etc.
        return {
            "error": str(e)
        }


@tool
def get_stock_news(company_name: str) -> list:
    """
    Fetch the latest financial news headlines for a given company.

    Queries the NewsAPI ``/v2/everything`` endpoint with the search term
    ``"{company_name} stock"`` to surface stock-specific articles.  Results
    are sorted by publication date (newest first) and limited to the 5 most
    recent articles to keep the report concise.

    **LangChain Tool** — the LLM can call this when it needs recent news
    context about a company.

    Args:
        company_name (str): The **full company name** (not the ticker),
                            e.g. ``"Infosys"``, ``"Reliance Industries"``,
                            ``"Apple Inc"``.  This is used as the search
                            keyword for NewsAPI.

    Returns:
        list[dict]: A list of article dictionaries, each containing:
            - **title** (str)       : Headline of the article.
            - **description** (str) : Short snippet / summary.
            - **url** (str)         : Link to the full article.
        Returns an empty list ``[]`` if no articles are found.
        On error, returns a single-element list with an error entry.

    Example:
        >>> get_stock_news.invoke("Infosys")
        [
            {
                "title": "Infosys Q4 results beat estimates",
                "description": "Infosys reported a 12% rise in net profit...",
                "url": "https://example.com/article1"
            },
            ...
        ]

    Implementation Notes
    --------------------
    *   Free-tier NewsAPI only returns articles from the last 30 days.
    *   ``page_size=5`` caps the response to avoid overwhelming the LLM
        context window.
    *   We append ``" stock"`` to the query to bias results toward
        financial coverage rather than general company news.
    """

    try:
        # Call NewsAPI /v2/everything endpoint
        # - q      : search query — company name + "stock" for relevance
        # - language : restrict to English articles
        # - sort_by  : newest first for timeliness
        # - page_size: only fetch 5 to keep the report focused
        response = newsapi.get_everything(
            q=f"{company_name} stock",
            language="en",
            sort_by="publishedAt",
            page_size=5
        )

        # Extract the list of article dicts from the API response
        articles = response.get("articles", [])

        if not articles:
            return []

        formatted_news = []

        for article in articles:
            # Only keep the fields we need for display
            formatted_news.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "url": article.get("url", "")
            })

        return formatted_news

    except Exception as e:
        # Return a structured error so the caller always gets a list
        return [
            {
                "title": "Error",
                "description": str(e),
                "url": ""
            }
        ]


@tool
def analyze_news_sentiment(company_name: str) -> dict:
    """
    Analyze the aggregate market sentiment for a company based on recent news.

    Fetches up to 20 recent English-language articles for the company from
    NewsAPI, computes the **polarity** of each article's title + description
    using TextBlob, and averages the scores to determine an overall sentiment
    label.

    **LangChain Tool** — provides the LLM with a quantitative sentiment signal
    that complements the raw news headlines.

    Sentiment Classification Thresholds
    -----------------------------------
    *   **avg_score > +0.15**  →  ``"Positive"``  📈
    *   **avg_score < −0.15**  →  ``"Negative"``  📉
    *   **otherwise**          →  ``"Neutral"``   ➡️

    TextBlob polarity ranges from −1.0 (most negative) to +1.0 (most positive).
    The ±0.15 thresholds are intentionally conservative to avoid classifying
    mildly worded articles as strongly directional.

    Args:
        company_name (str): Full company name for NewsAPI search
                            (e.g. ``"Tata Consultancy Services"``).

    Returns:
        dict: On success:
            - **sentiment** (str)         : ``"Positive"`` / ``"Negative"`` / ``"Neutral"``.
            - **score** (float)           : Average polarity, rounded to 2 dp.
            - **articles_analyzed** (int) : Number of articles that contributed.
        If no articles are found:
            - **sentiment** (str)  : ``"Unknown"``
            - **score** (float)    : ``0``
            - **reason** (str)     : Explanation of why no data was available.
        On error:
            - **sentiment** (str)  : ``"Error"``
            - **score** (float)    : ``0``
            - **error** (str)      : Exception message.

    Example:
        >>> analyze_news_sentiment.invoke("Infosys")
        {
            "sentiment": "Positive",
            "score": 0.23,
            "articles_analyzed": 18
        }

    Implementation Notes
    --------------------
    *   We request ``page_size=20`` (vs. 5 for headlines) to get a larger
        sample for statistically meaningful sentiment aggregation.
    *   Each article's title and description are concatenated before
        passing to TextBlob — this gives more signal than title alone.
    *   TextBlob's sentiment is **lexicon-based** (not transformer-based),
        meaning it's fast and doesn't require an API key, but may miss
        sarcasm or domain-specific financial jargon.
    """

    try:
        # Fetch up to 20 articles for sentiment analysis.
        # Larger sample size gives a more reliable aggregate score.
        response = newsapi.get_everything(
            q=f"{company_name} stock",
            language="en",
            page_size=20
        )

        articles = response.get("articles", [])

        # No articles → return "Unknown" with a helpful reason for the user
        if not articles:
            return {
                "sentiment": "Unknown",
                "score": 0,
                "reason": (
                    f"No news articles found for '{company_name}'. "
                    "The NewsAPI free tier may not have recent coverage for this company."
                )
            }

        scores = []

        for article in articles:
            # Combine title and description for richer text signal.
            # (article.get("title") or "") safely handles None values.
            text = (
                (article.get("title") or "")
                + " " +
                (article.get("description") or "")
            )

            # TextBlob computes polarity: -1.0 (negative) to +1.0 (positive).
            # 0.0 is perfectly neutral.
            polarity = TextBlob(text).sentiment.polarity

            scores.append(polarity)

        # Arithmetic mean of all article polarities
        avg_score = sum(scores) / len(scores)

        # Classify into Positive / Negative / Neutral based on thresholds.
        # The ±0.15 band avoids over-classifying slightly tilted language.
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
        # Catch-all for network errors, API quota exhaustion, etc.
        return {
            "sentiment": "Error",
            "score": 0,
            "error": str(e)
        }


# =========================================================
# AI ANALYSIS
# =========================================================

def generate_analysis(
    ticker: str,
    stock_data: dict,
    sentiment_data: dict,
    news: list
) -> str:
    """
    Generate a structured AI investment analysis using the DeepSeek LLM.

    Constructs a detailed prompt that instructs the LLM to role-play as a
    senior Wall Street analyst.  The prompt includes all collected data
    (price, sentiment, news) and demands exactly four clearly labeled
    markdown sections with a ``##`` header each.  The LLM's response is
    returned as a raw string to be parsed by ``generate_html_report()``.

    Prompt Design Rationale
    -----------------------
    *   **Role instruction** ("You are a senior Wall Street analyst") sets
        the tone, register, and expected depth of analysis.
    *   **Structured output** with four mandatory sections ensures the
        downstream HTML parser can reliably extract each section.
    *   **``<mark>`` tags** are only allowed in the Recommendation section.
        This is enforced in the prompt to prevent the LLM from polluting
        other sections with HTML.
    *   **No preamble** rule prevents the LLM from adding conversational
        fluff before the first header, which would break parsing.

    Args:
        ticker (str):                The stock ticker symbol (e.g. ``"TCS.NS"``).
        stock_data (dict):           Output from ``get_stock_price()``.
        sentiment_data (dict):       Output from ``analyze_news_sentiment()``.
        news (list[dict]):           Output from ``get_stock_news()``.

    Returns:
        str: The raw LLM response — a markdown-formatted analysis with
        exactly four ``##`` sections:
        1. ``## 📊 Financial Outlook``
        2. ``## ⚠️ Key Risks``
        3. ``## 🚀 Growth Potential``
        4. ``## 🎯 Investment Recommendation`` (includes ``<mark>`` tags)

    Example (abbreviated output):
        ## 📊 Financial Outlook
        The company has shown consistent revenue growth...

        ## ⚠️ Key Risks
        Regulatory headwinds in key markets...

        ## 🚀 Growth Potential
        Expansion into cloud services presents...

        ## 🎯 Investment Recommendation
        We rate this stock a <mark>BUY</mark> with a 12-month target...
    """

    # ── Build the prompt with all available data injected ──
    # We use an f-string to interpolate the collected data directly into
    # the prompt.  The LLM receives everything it needs in a single message.
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
[Begin with the verdict word in bold, immediately followed by a period — exactly
one of **BUY.** / **HOLD.** / **SELL.** — then continue the reasoning on the same
line, e.g. "**HOLD.** The stock trades ...". The verdict word MUST be the very
first characters of this section, before any other text or markup.
Then 1-2 paragraphs with the stance and reasoning.
In THIS section ONLY, wrap the single most critical sentence in each paragraph with <mark>...</mark> tags so it stands out visually.]

Keep it concise, data-driven, and professional. Do not add any preamble before the first section header.
Do NOT use <mark> tags in any other section.
"""

    # Send the prompt to DeepSeek via langchain-openai.
    # `invoke()` returns an AIMessage; `.content` extracts the text.
    response = llm.invoke(prompt)

    return response.content


# =========================================================
# TECHNICAL ANALYSIS AGENT (QUANTITATIVE)
# =========================================================
# A second, parallel LLM agent that acts as an expert quantitative analyst.
# It reads the DataFrame produced by TECH_ANALYSIS.compute_technical_indicators()
# (5 years of daily OHLCV + 35 indicator columns), condenses it into a compact
# prompt payload, and asks the LLM to analyse EVERY technical indicator for
# price and volume and classify the stock into one of four flags:
#     STRONG BUY / BUY / SELL / STRONG SELL
# The result (raw markdown) is later parsed by parse_technical_analysis() and
# rendered by generate_html_report() alongside the fundamental analysis.

# Display metadata for each technical group the LLM is asked to produce.
# The emoji is used in the rendered card header.
TECH_GROUP_META = {
    "price moving average": {"emoji": "📊", "title": "Price Moving Averages"},
    "volume moving average": {"emoji": "📈", "title": "Volume Moving Averages"},
    "momentum":              {"emoji": "⚡", "title": "Momentum Indicators"},
    "volatility":            {"emoji": "📏", "title": "Volatility — Bollinger Bands"},
    "volume flow":           {"emoji": "🔄", "title": "Volume Flow"},
    "technical verdict":     {"emoji": "🎯", "title": "Technical Verdict"},
}


def _tokenize(text: str) -> list[str]:
    """
    Split text into lower-case alphanumeric word tokens.

    Used by the RAG retriever to score how relevant each document is to a
    natural-language query.

    Args:
        text (str): raw text to tokenize.

    Returns:
        list[str]: the word tokens (e.g. "MACD signal" -> ["macd", "signal"]).
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _format_recent_series(tech_df: pd.DataFrame, column: str, n: int = 30) -> str:
    """
    Format the last ``n`` values of a single indicator column as a compact
    ``date:value`` list so the LLM can see a short recent history.

    Args:
        tech_df : DataFrame from TECH_ANALYSIS.compute_technical_indicators().
        column  : name of the indicator column (e.g. "RSI", "OBV").
        n       : how many most-recent values to include (default 30).

    Returns:
        str: a comma-separated list of ``date:value`` pairs.
    """
    vals = tech_df[column].dropna().tail(n)
    if vals.empty:
        return "No history available."
    return ", ".join(f"{d.date()}:{v:.2f}" for d, v in vals.items())


def _build_technical_corpus(tech_df: pd.DataFrame) -> list[dict]:
    """
    Convert the technical-indicator DataFrame into a retrievable RAG corpus.

    Each returned "document" is a self-contained text chunk about ONE aspect
    of the data (latest snapshot, price/volume moving averages, RSI, MACD,
    Bollinger Bands, Stochastic, OBV, recent price history) together with
    retrieval keywords so the retriever can rank it against a query.

    This is the indexing step of the RAG pipeline: the DataFrame (a table)
    is turned into a small set of topic documents that can be retrieved and
    fed to the LLM as context.

    Args:
        tech_df : DataFrame from TECH_ANALYSIS.compute_technical_indicators().

    Returns:
        list[dict]: each item = {"id", "title", "keywords", "content"}.
    """
    latest = tech_df.iloc[-1]
    close = float(latest["Close"])
    volume = float(latest["Volume"])
    docs = []

    # --- 1. Overview: date range + the FULL latest row (all 35 indicators) ---
    overview_lines = [
        f"Date range: {tech_df.index[0].date()} → {tech_df.index[-1].date()} "
        f"({len(tech_df)} trading days)",
        f"Latest close: {close:,.2f} | Latest volume: {volume:,.0f}",
        "Latest values of every indicator:",
    ]
    for col in tech_df.columns:
        v = latest[col]
        try:
            overview_lines.append(f"  {col}: {float(v):,.2f}")
        except (TypeError, ValueError):
            overview_lines.append(f"  {col}: {v}")
    docs.append({
        "id": "overview",
        "title": "Market Overview & Latest Indicator Snapshot",
        "keywords": ["overview", "latest", "close", "price", "volume", "snapshot", "current", "date", "range"],
        "content": "\n".join(overview_lines),
    })

    # --- 2. Price moving averages ---
    ma_lines = []
    for n in TECH_ANALYSIS.MA_WINDOWS:
        sma = float(latest[f"SMA_{n}"])
        ema = float(latest[f"EMA_{n}"])
        vs_sma = (close - sma) / sma * 100.0 if sma else 0.0
        vs_ema = (close - ema) / ema * 100.0 if ema else 0.0
        stance = "bullish (EMA > SMA)" if ema > sma else "bearish (EMA <= SMA)"
        ma_lines.append(
            f"SMA_{n}={sma:,.2f} (price {vs_sma:+.2f}%) | "
            f"EMA_{n}={ema:,.2f} (price {vs_ema:+.2f}%) | {stance}"
        )
    docs.append({
        "id": "price_ma",
        "title": "Price Moving Averages (SMA & EMA, 5/13/23/90/200)",
        "keywords": ["price", "moving", "average", "sma", "ema", "trend", "support", "resistance", "golden", "death", "cross"],
        "content": "\n".join(ma_lines),
    })

    # --- 3. Volume moving averages ---
    vol_lines = []
    for n in TECH_ANALYSIS.MA_WINDOWS:
        vsma = float(latest[f"VOL_SMA_{n}"])
        vema = float(latest[f"VOL_EMA_{n}"])
        vs_vsma = (volume - vsma) / vsma * 100.0 if vsma else 0.0
        vs_vema = (volume - vema) / vema * 100.0 if vema else 0.0
        vol_lines.append(
            f"VOL_SMA_{n}={vsma:,.0f} (vol {vs_vsma:+.2f}%) | "
            f"VOL_EMA_{n}={vema:,.0f} (vol {vs_vema:+.2f}%)"
        )
    docs.append({
        "id": "volume_ma",
        "title": "Volume Moving Averages (VOL_SMA & VOL_EMA, 5/13/23/90/200)",
        "keywords": ["volume", "vol", "moving", "average", "liquidity", "activity", "shares"],
        "content": "\n".join(vol_lines),
    })

    # --- 4. RSI ---
    rsi = float(latest["RSI"])
    rsi_state = "overbought (>70)" if rsi > 70 else ("oversold (<30)" if rsi < 30 else "neutral")
    docs.append({
        "id": "rsi",
        "title": "Relative Strength Index (RSI, 14-day)",
        "keywords": ["rsi", "momentum", "relative", "strength", "overbought", "oversold"],
        "content": (
            f"RSI (latest) = {rsi:.2f} — {rsi_state}.\n"
            f"RSI history (last 30 days): {_format_recent_series(tech_df, 'RSI')}"
        ),
    })

    # --- 5. MACD ---
    macd = float(latest["MACD"])
    sig = float(latest["MACD_SIGNAL"])
    hist = float(latest["MACD_HIST"])
    prev_macd = float(tech_df["MACD"].iloc[-2])
    prev_sig = float(tech_df["MACD_SIGNAL"].iloc[-2])
    if prev_macd <= prev_sig and macd > sig:
        cross = "MACD crossed above signal (bullish)"
    elif prev_macd >= prev_sig and macd < sig:
        cross = "MACD crossed below signal (bearish)"
    else:
        cross = "no fresh crossover"
    docs.append({
        "id": "macd",
        "title": "Moving Average Convergence/Divergence (MACD 12,26,9)",
        "keywords": ["macd", "momentum", "signal", "histogram", "crossover", "divergence"],
        "content": (
            f"MACD={macd:.2f}, Signal={sig:.2f}, Histogram={hist:.2f} — {cross}.\n"
            f"MACD history (last 30 days): {_format_recent_series(tech_df, 'MACD')}\n"
            f"Signal history (last 30 days): {_format_recent_series(tech_df, 'MACD_SIGNAL')}"
        ),
    })

    # --- 6. Bollinger Bands ---
    bb_up = float(latest["BB_UPPER"])
    bb_mid = float(latest["BB_MIDDLE"])
    bb_lo = float(latest["BB_LOWER"])
    pct_b = (close - bb_lo) / (bb_up - bb_lo) if bb_up > bb_lo else float("nan")
    band_w = (bb_up - bb_lo) / bb_mid * 100.0 if bb_mid else float("nan")
    if close > bb_up:
        bb_pos = "above the upper band"
    elif close < bb_lo:
        bb_pos = "below the lower band"
    else:
        bb_pos = "inside the bands"
    docs.append({
        "id": "bollinger",
        "title": "Bollinger Bands (20-day, 2σ)",
        "keywords": ["bollinger", "band", "volatility", "squeeze", "upper", "lower", "middle"],
        "content": (
            f"Upper={bb_up:,.2f}, Middle={bb_mid:,.2f}, Lower={bb_lo:,.2f} | "
            f"close is {bb_pos} | %B={pct_b:.2f}, BandWidth={band_w:.2f}%"
        ),
    })

    # --- 7. Stochastic ---
    k = float(latest["STOCH_K"])
    d = float(latest["STOCH_D"])
    k_prev = float(tech_df["STOCH_K"].iloc[-2])
    d_prev = float(tech_df["STOCH_D"].iloc[-2])
    stoch_state = "overbought (>80)" if k > 80 else ("oversold (<20)" if k < 20 else "neutral")
    if k_prev <= d_prev and k > d:
        stoch_cross = "K crossed above D (bullish)"
    elif k_prev >= d_prev and k < d:
        stoch_cross = "K crossed below D (bearish)"
    else:
        stoch_cross = "no fresh crossover"
    docs.append({
        "id": "stochastic",
        "title": "Stochastic Oscillator (%K 14, %D 3)",
        "keywords": ["stochastic", "stoch", "oscillator", "overbought", "oversold", "momentum"],
        "content": (
            f"%K={k:.2f}, %D={d:.2f} — {stoch_state}; {stoch_cross}.\n"
            f"%K history (last 30 days): {_format_recent_series(tech_df, 'STOCH_K')}"
        ),
    })

    # --- 8. OBV ---
    obv = float(latest["OBV"])
    obv_5 = obv - float(tech_df["OBV"].iloc[-6]) if len(tech_df) >= 6 else 0.0
    obv_20 = obv - float(tech_df["OBV"].iloc[-21]) if len(tech_df) >= 21 else 0.0
    obv_60 = obv - float(tech_df["OBV"].iloc[-61]) if len(tech_df) >= 61 else 0.0
    docs.append({
        "id": "obv",
        "title": "On-Balance Volume (OBV)",
        "keywords": ["obv", "on", "balance", "volume", "accumulation", "distribution", "flow"],
        "content": (
            f"OBV (latest) = {obv:,.0f} | 5d Δ={obv_5:+,.0f}, 20d Δ={obv_20:+,.0f}, 60d Δ={obv_60:+,.0f}\n"
            f"OBV history (last 30 days): {_format_recent_series(tech_df, 'OBV')}"
        ),
    })

    # --- 9. Recent price & volume history ---
    docs.append({
        "id": "recent_price",
        "title": "Recent Price & Volume History (last 30 trading days)",
        "keywords": ["recent", "price", "history", "close", "trend", "last", "days"],
        "content": tech_df[["Close", "Volume"]].tail(30).round(2).to_string(),
    })

    return docs


def _retrieve_technical_docs(tech_df: pd.DataFrame, query: str, k: int = None) -> list[dict]:
    """
    Retrieve the most relevant documents from the technical-indicator RAG
    corpus for a natural-language query.

    This is the retrieval step of the RAG pipeline. It uses a lightweight,
    dependency-free BM25-style keyword scorer over each document's keywords,
    title and content — fast, and no embedding model or vector database is
    required.

    Args:
        tech_df : DataFrame from TECH_ANALYSIS.compute_technical_indicators().
        query   : natural-language retrieval query (e.g. "momentum indicators").
        k       : maximum number of documents to return. Default ``None``
                  returns every document in the corpus.

    Returns:
        list[dict]: the top ``k`` documents (each = {"id", "title",
        "keywords", "content"}), ranked by relevance to ``query``.
    """
    docs = _build_technical_corpus(tech_df)
    if k is None:
        k = len(docs)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return docs[:k]

    scored = []
    for doc in docs:
        kw = set(_tokenize(" ".join(doc.get("keywords", []))))
        title = set(_tokenize(doc.get("title", "")))
        content_tokens = _tokenize(doc.get("content", ""))
        score = 0.0
        for term in query_tokens:
            if term in kw:
                score += 3.0          # strong signal: keyword match
            elif term in title:
                score += 1.5          # medium signal: title match
            score += min(content_tokens.count(term), 5) * 0.2  # weak: content match
        scored.append((score, doc))

    # Rank descending by score; ties keep the original corpus order.
    scored.sort(key=lambda t: t[0], reverse=True)
    return [doc for _, doc in scored[:k]]


def generate_technical_analysis(ticker: str, tech_df: pd.DataFrame) -> str:
    """
    Second LLM agent — an expert quantitative analyst.

    Reads the technical-indicator DataFrame produced by
    ``TECH_ANALYSIS.compute_technical_indicators()`` and asks the LLM to:

      1. analyse EVERY technical indicator (price & volume moving averages,
         RSI, MACD, Bollinger Bands, Stochastic, OBV);
      2. classify the stock into one of four flags:
         ``STRONG BUY`` / ``BUY`` / ``SELL`` / ``STRONG SELL``.

    The output is raw markdown with ``##`` groups and ``###`` per-indicator
    sub-sections, parsed later by ``parse_technical_analysis()``.

    Args:
        ticker  : Yahoo Finance symbol (e.g. "AAPL", "RELIANCE.NS").
        tech_df : DataFrame from TECH_ANALYSIS.compute_technical_indicators().

    Returns:
        str: raw markdown technical analysis ending with a
             ``## 🎯 Technical Verdict`` section whose first line is the flag.
    """
    # ── RAG pipeline ──
    # Treat the technical-indicator DataFrame as a retrievable knowledge base:
    # split it into topic documents, retrieve the documents relevant to a
    # comprehensive technical analysis, and feed the retrieved context to the
    # LLM (retrieval-augmented generation).
    query = (
        "technical analysis price moving averages sma ema volume moving averages "
        "momentum indicators rsi macd stochastic volatility bollinger bands "
        "volume flow obv recent price trend"
    )
    retrieved_docs = _retrieve_technical_docs(tech_df, query)

    # Label each retrieved document clearly so the LLM can reference it.
    rag_context = "\n\n".join(
        f"### DOCUMENT {i + 1}: {doc['title']}\n{doc['content']}"
        for i, doc in enumerate(retrieved_docs)
    )

    prompt = f"""
You are an expert quantitative analyst specialising in technical analysis.

Analyse the technical indicators below for the stock {ticker} using ONLY the
retrieved documents provided. These documents were retrieved via
retrieval-augmented generation (RAG) from a 5-year daily technical-indicator
database.

Retrieved Documents (RAG):
{rag_context}

Produce a structured technical analysis. Use the EXACT section headers below
(with the emoji). Inside each section, analyse EVERY indicator by writing a
### sub-header with the indicator name followed by 1-2 sentences of
interpretation tied strictly to the numbers provided.

## 📊 Price Moving Averages
Analyse every window 5, 13, 23, 90 and 200 for both SMA and EMA, e.g.:
### SMA_5
[interpretation]
### SMA_13
[interpretation]
### SMA_200
[interpretation]
### EMA_5
[interpretation]
... (continue for EMA_13, EMA_23, EMA_90, EMA_200)

## 📈 Volume Moving Averages
Analyse every window 5, 13, 23, 90 and 200 for both VOL_SMA and VOL_EMA, e.g.:
### VOL_SMA_5
[interpretation]
... (continue for all VOL_SMA and VOL_EMA windows)

## ⚡ Momentum Indicators
### RSI
[interpretation]
### MACD
[interpretation]
### STOCH_K
[interpretation]
### STOCH_D
[interpretation]

## 📏 Volatility — Bollinger Bands
### BB_UPPER
[interpretation]
### BB_MIDDLE
[interpretation]
### BB_LOWER
[interpretation]

## 🔄 Volume Flow
### OBV
[interpretation]

## 🎯 Technical Verdict
Begin this section with a SINGLE line containing EXACTLY one of:
**STRONG BUY**
**BUY**
**SELL**
**STRONG SELL**
Then write a concise 2-3 sentence overall summary of the technical picture
(trend, momentum, volatility, volume) and explain why you chose that flag.

Rules:
- Base every statement strictly on the numbers provided; never invent values.
- Do not add any preamble before the first section header.
- Keep every indicator analysis to 1-2 sentences.
- Use only the four verdict flags above.
"""
    response = llm.invoke(prompt)
    return response.content


def _detect_technical_verdict(text: str) -> str:
    """
    Extract the technical verdict flag from the Technical Verdict section.

    Looks first for an explicit ``**FLAG**`` marker, then for a leading flag
    word. Returns one of ``STRONG BUY`` / ``BUY`` / ``SELL`` / ``STRONG SELL``.
    Falls back to ``NEUTRAL`` if no flag can be found (e.g. malformed output).

    Args:
        text (str): body text of the ``## 🎯 Technical Verdict`` section.

    Returns:
        str: normalised verdict flag (upper-case, single-spaced).
    """
    t = text.strip()

    # 1. Explicit **FLAG** marker anywhere in the section.
    m = re.search(
        r'\*\*\s*(STRONG\s+BUY|BUY|STRONG\s+SELL|SELL)\s*\*\*',
        t, re.IGNORECASE
    )
    if m:
        return " ".join(m.group(1).upper().split())

    # 2. Leading flag word without asterisks, e.g. "BUY — ...".
    m = re.match(r'^(STRONG\s+BUY|BUY|SELL|STRONG\s+SELL)\b', t, re.IGNORECASE)
    if m:
        return " ".join(m.group(1).upper().split())

    return "NEUTRAL"


def _format_indicator_value(name: str, value: float) -> str:
    """
    Format an indicator's latest value for display.

    Volume / OBV style columns are shown as integers with thousands
    separators; everything else is shown with two decimal places.

    Args:
        name  (str)  : indicator column name (e.g. "SMA_5", "VOL_SMA_5", "OBV").
        value (float): latest value from the DataFrame.

    Returns:
        str: human-friendly formatted value ("—" for missing/NaN).
    """
    if value is None:
        return "—"
    if isinstance(value, float) and value != value:  # NaN check
        return "—"
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return str(value)
    if name.startswith("VOL") or name == "OBV":
        return f"{fv:,.0f}"
    return f"{fv:,.2f}"


def parse_technical_analysis(tech_analysis: str, tech_df: pd.DataFrame = None) -> dict:
    """
    Parse the raw markdown from ``generate_technical_analysis()`` into a
    structured dict the HTML template can render.

    * Splits the markdown on ``##`` headers to find the indicator groups.
    * Splits each group on ``###`` headers to find per-indicator entries.
    * Attaches the DETERMINISTIC latest value for each indicator (from
      ``tech_df``) so the displayed numbers never depend on the LLM.
    * Extracts the 4-flag verdict and the overall technical summary.

    Args:
        tech_analysis (str): raw output of ``generate_technical_analysis()``.
        tech_df       : DataFrame used to attach live indicator values.

    Returns:
        dict: {
            "groups":  [{"title", "emoji", "indicators": [{"name", "value", "analysis"}]}],
            "verdict": "STRONG BUY" | "BUY" | "SELL" | "STRONG SELL" | "NEUTRAL",
            "verdict_css": lower-case, no-space CSS class for the badge,
            "summary": str,
            "error":   None or message
        }
    """
    empty = {
        "groups": [],
        "verdict": "NEUTRAL",
        "verdict_css": "neutral",
        "summary": "No technical analysis was generated for this report.",
        "error": "Technical analysis unavailable.",
    }

    if not tech_analysis or not tech_analysis.strip():
        return empty

    latest = tech_df.iloc[-1] if tech_df is not None and not tech_df.empty else None

    groups = []
    verdict = "NEUTRAL"
    summary = ""

    # Split into ## groups (the LLM is prompted to use this exact format).
    # NOTE: anchored to line-start with (?m)^ so "### " indicator headers are
    # NOT mistaken for "## " group headers (a bare '##\s+' would also match
    # the 2nd+3rd '#' of '### ' and corrupt the parse).
    raw_groups = re.split(r'(?m)^##\s+', tech_analysis.strip())

    for raw in raw_groups:
        if not raw.strip():
            continue

        lines = raw.strip().splitlines()
        header_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        # Match the group header to known metadata.
        matched = None
        for key, meta in TECH_GROUP_META.items():
            if key in header_line.lower():
                matched = meta
                break
        if matched is None:
            continue

        # --- Special handling for the verdict section ---
        if matched["title"] == "Technical Verdict":
            verdict = _detect_technical_verdict(body)
            # Summary = body with the flag marker line(s) removed.
            summary = "\n".join(
                ln for ln in body.splitlines()
                if not re.search(
                    r'\*\*\s*(STRONG\s+BUY|BUY|SELL|STRONG\s+SELL)\s*\*\*',
                    ln, re.IGNORECASE
                )
            ).strip()
            continue

        # --- Parse ### per-indicator entries ---
        # Also anchored to line-start so nested headers are unambiguous.
        indicators = []
        raw_indicators = re.split(r'(?m)^###\s+', body)
        for ri in raw_indicators:
            if not ri.strip():
                continue
            ilines = ri.strip().splitlines()
            name = ilines[0].strip()
            analysis = "\n".join(ilines[1:]).strip()

            # Attach the live value from the DataFrame when available.
            value = None
            if latest is not None and name in latest.index:
                value = _format_indicator_value(name, latest[name])

            indicators.append({
                "name": name,
                "value": value,
                "analysis": analysis,
            })

        groups.append({
            "title": matched["title"],
            "emoji": matched["emoji"],
            "indicators": indicators,
        })

    return {
        "groups": groups,
        "verdict": verdict,
        "verdict_css": verdict.lower().replace(" ", ""),
        "summary": summary,
        "error": None,
    }


def build_overall_summary(fund_verdict: str, tech_verdict: str) -> dict:
    """
    Merge the fundamental verdict (BUY / HOLD / SELL) and the technical
    verdict (STRONG BUY / BUY / SELL / STRONG SELL) into a single overall
    stance using a rule-based matrix.

    Strongly aligned signals produce a strong overall call; conflicting
    signals resolve to HOLD (a warning that the two views disagree).

    Args:
        fund_verdict (str): BUY / HOLD / SELL from the fundamental analysis.
        tech_verdict (str): STRONG BUY / BUY / SELL / STRONG SELL (or NEUTRAL).

    Returns:
        dict: {
            "fundamental_verdict", "technical_verdict", "overall_verdict",
            "overall_css", "fundamental_css", "technical_css", "text"
        }
    """
    fund = " ".join((fund_verdict or "HOLD").upper().split())
    tech = " ".join((tech_verdict or "NEUTRAL").upper().split())

    # (fundamental, technical) -> (overall verdict, explanation)
    matrix = {
        ("BUY",  "STRONG BUY"):  ("STRONG BUY", "Fundamentals and technicals align decisively bullish."),
        ("BUY",  "BUY"):         ("BUY", "Both fundamental and technical views are bullish."),
        ("BUY",  "SELL"):        ("HOLD", "Fundamentals are bullish but technicals are bearish — the signals conflict; stay neutral."),
        ("BUY",  "STRONG SELL"): ("HOLD", "Fundamentals are bullish but technicals are strongly bearish — the signals conflict; stay neutral."),
        ("BUY",  "NEUTRAL"):     ("BUY", "Fundamentals are bullish; technical analysis was unavailable."),
        ("HOLD", "STRONG BUY"):  ("BUY", "Fundamentals are neutral but technical momentum is strong — lean bullish."),
        ("HOLD", "BUY"):         ("BUY", "Fundamentals are neutral with positive technicals — lean bullish."),
        ("HOLD", "SELL"):        ("SELL", "Fundamentals are neutral with negative technicals — lean bearish."),
        ("HOLD", "STRONG SELL"): ("SELL", "Fundamentals are neutral with strongly negative technicals — lean bearish."),
        ("HOLD", "NEUTRAL"):     ("HOLD", "Fundamentals are neutral and no technical signal was available."),
        ("SELL", "STRONG BUY"):  ("HOLD", "Fundamentals are bearish but technicals are strongly bullish — the signals conflict; stay neutral."),
        ("SELL", "BUY"):         ("HOLD", "Fundamentals are bearish but technicals are bullish — the signals conflict; stay neutral."),
        ("SELL", "SELL"):        ("SELL", "Both fundamental and technical views are bearish."),
        ("SELL", "STRONG SELL"): ("STRONG SELL", "Fundamentals and technicals align decisively bearish."),
        ("SELL", "NEUTRAL"):     ("SELL", "Fundamentals are bearish; technical analysis was unavailable."),
    }

    overall, text = matrix.get(
        (fund, tech),
        ("HOLD", "Combined signals are mixed; adopt a neutral stance."),
    )

    return {
        "fundamental_verdict": fund,
        "technical_verdict": tech,
        "overall_verdict": overall,
        "overall_css": overall.lower().replace(" ", ""),
        "fundamental_css": fund.lower().replace(" ", ""),
        "technical_css": tech.lower().replace(" ", ""),
        "text": text,
    }


# =========================================================
# HTML GENERATOR
# =========================================================
    # ─────────────────────────────────────────────────────────────
    # INLINE HTML TEMPLATE (Jinja2)
    # ─────────────────────────────────────────────────────────────
    # The entire HTML document is defined as a Jinja2 template string.
    # This includes:
    #   - CSS custom properties (design tokens) in :root
    #   - Responsive grid layout (hero, metrics strip, two-column, analysis)
    #   - Fade-up animations with staggered delays
    #   - JavaScript for sentiment bar animation & volume formatting
    # Jinja2 placeholders ({{ ... }}) are filled at render time.
    # ─────────────────────────────────────────────────────────────
def generate_html_report(
    ticker: str,
    stock_data: dict,
    sentiment_data: dict,
    news: list,
    analysis: str,
    tech_analysis: str = None,
    tech_df: pd.DataFrame = None
) -> str:
    """
    Render a complete, self-contained HTML stock analysis report.

    This is the presentation engine of the agent.  It takes the raw outputs
    from all three tools plus the LLM-generated analyses (fundamental AND
    technical) and produces a single ``.html`` file with embedded CSS and
    JavaScript — no external dependencies beyond Google Fonts.

    Processing Pipeline
    -------------------
    1.  **Parse the fundamental analysis** — Split the markdown on ``##``
        headers, match each section against a known mapping (Financial
        Outlook, Key Risks, Growth Potential, Investment Recommendation),
        extract paragraphs, and detect the BUY/HOLD/SELL verdict.

    2.  **Parse the technical analysis** — Split the technical agent's
        markdown into per-indicator cards (price & volume moving averages,
        RSI, MACD, Bollinger Bands, Stochastic, OBV), attach the live
        indicator values from ``tech_df``, and extract the 4-flag verdict
        (STRONG BUY / BUY / SELL / STRONG SELL).

    3.  **Build the Overall Summary** — Merge the fundamental verdict with
        the technical verdict using a rule-based matrix.

    4.  **Render via Jinja2** — The parsed data (stock metrics, sentiment,
        news articles, fundamental + technical analysis sections, overall
        summary) is injected into a rich HTML template with responsive CSS
        grid layout, animated sentiment bar, volume number formatting, and
        color-coded verdict badges.

    5.  **Write to disk** — The rendered HTML is saved as
        ``{CLEAN_TICKER}_REPORT_{YYYY_MM_DD}.html`` in the current working
        directory.

    Args:
        ticker (str):           Raw ticker symbol (may include exchange suffix),
                                e.g. ``"TCS.NS"``.
        stock_data (dict):      Output from ``get_stock_price()``.
        sentiment_data (dict):  Output from ``analyze_news_sentiment()``.
        news (list[dict]):      Output from ``get_stock_news()``.
        analysis (str):         Raw markdown output from ``generate_analysis()``.
        tech_analysis (str):    Optional raw markdown output from
                                ``generate_technical_analysis()`` (the
                                quantitative agent). Default ``None``.
        tech_df (pd.DataFrame): Optional DataFrame from
                                ``TECH_ANALYSIS.compute_technical_indicators()``
                                used to attach live indicator values.
                                Default ``None``.

    Returns:
        str: The filename of the generated HTML report, e.g.
        ``"TCS_REPORT_2026_06_07.html"``.
    """
    html_template = """<html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ stock.ticker }} — Stock Analysis</title>
    <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap" rel="stylesheet">
    <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
        --bg: #f0ede8;
        --surface: #faf8f5;
        --surface-2: #f5f2ed;
        --border: rgba(60,40,20,0.12);
        --border-strong: rgba(60,40,20,0.22);
        --ink: #1a1208;
        --ink-2: #4a3f2f;
        --ink-3: #8a7d6a;
        --accent: #c94f1a;
        --accent-light: #fceee8;
        --accent-mid: #f0a882;
        --green: #1a6b3a;
        --green-light: #e8f5ee;
        --red: #b01a1a;
        --red-light: #fce8e8;
        --amber: #92600a;
        --amber-light: #fdf4e3;
        --blue: #1a3a6b;
        --blue-light: #e8eef8;
        --gold: #c8901a;
        --serif: 'DM Serif Display', Georgia, serif;
        --sans: 'DM Sans', system-ui, sans-serif;
        --radius: 16px;
        --radius-sm: 10px;
        --shadow: 0 2px 16px rgba(60,40,10,0.09), 0 1px 3px rgba(60,40,10,0.06);
        --shadow-lg: 0 8px 40px rgba(60,40,10,0.12), 0 2px 8px rgba(60,40,10,0.08);
    }

    body {
        font-family: var(--sans);
        background: var(--bg);
        color: var(--ink);
        min-height: 100vh;
        padding: 0;
    }

    /* ─── HERO ─── */
    .hero {
        background: var(--ink);
        color: #faf8f5;
        padding: 56px 48px 52px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
        radial-gradient(ellipse 60% 80% at 80% 50%, rgba(201,79,26,0.18) 0%, transparent 70%),
        radial-gradient(ellipse 40% 60% at 10% 80%, rgba(200,144,26,0.12) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-grid {
        max-width: 1100px;
        margin: auto;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 32px;
        align-items: flex-end;
        position: relative;
    }
    .hero-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-mid);
        margin-bottom: 10px;
    }
    .hero-ticker {
        font-family: var(--serif);
        font-size: clamp(42px, 6vw, 68px);
        line-height: 1;
        letter-spacing: -0.02em;
        color: #faf8f5;
    }
    .hero-exchange {
        font-size: 13px;
        color: rgba(250,248,245,0.55);
        margin-top: 8px;
        font-weight: 300;
    }
    .hero-price-block {
        text-align: right;
    }
    .hero-price-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(250,248,245,0.45);
        margin-bottom: 6px;
    }
    .hero-price {
        font-family: var(--serif);
        font-size: clamp(32px, 4vw, 52px);
        line-height: 1;
        color: #faf8f5;
    }
    .hero-ts {
        font-size: 12px;
        color: rgba(250,248,245,0.35);
        margin-top: 10px;
        font-weight: 300;
    }

    /* ─── MAIN LAYOUT ─── */
    .page {
        max-width: 1100px;
        margin: 0 auto;
        padding: 40px 24px 80px;
    }

    .section-title {
        font-family: var(--serif);
        font-size: 22px;
        color: var(--ink);
        margin-bottom: 20px;
        letter-spacing: -0.01em;
    }

    /* ─── METRICS STRIP ─── */
    .metrics-strip {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
        margin-bottom: 36px;
    }
    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 18px 16px 14px;
        box-shadow: var(--shadow);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        cursor: default;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }
    .metric-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 22px;
        font-weight: 600;
        color: var(--ink);
        letter-spacing: -0.02em;
    }
    .metric-value.up { color: var(--green); }
    .metric-value.down { color: var(--red); }
    .metric-sub {
        font-size: 11px;
        color: var(--ink-3);
        margin-top: 4px;
        font-weight: 300;
    }

    /* ─── TWO-COL ─── */
    .two-col {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin-bottom: 36px;
    }
    @media (max-width: 700px) {
        .two-col { grid-template-columns: 1fr; }
        .hero { padding: 36px 24px; }
        .hero-grid { grid-template-columns: 1fr; }
        .hero-price-block { text-align: left; }
    }

    /* ─── SENTIMENT CARD ─── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 28px;
        box-shadow: var(--shadow);
    }
    .sentiment-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 18px;
        border-radius: 999px;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 14px;
    }
    .sentiment-badge.positive { background: var(--green-light); color: var(--green); }
    .sentiment-badge.negative { background: var(--red-light); color: var(--red); }
    .sentiment-badge.neutral  { background: var(--amber-light); color: var(--amber); }
    .sentiment-badge.unknown  { background: var(--blue-light); color: var(--blue); }

    .sentiment-bar-wrap {
        height: 8px;
        background: var(--surface-2);
        border-radius: 99px;
        margin: 14px 0 10px;
        overflow: hidden;
    }
    .sentiment-bar {
        height: 100%;
        border-radius: 99px;
        transition: width 1s cubic-bezier(.22,.68,0,1.2);
    }
    .sentiment-bar.positive { background: linear-gradient(90deg, #4caf88, #1a6b3a); }
    .sentiment-bar.negative { background: linear-gradient(90deg, #e87070, #b01a1a); }
    .sentiment-bar.neutral  { background: linear-gradient(90deg, #f0c84a, #92600a); }
    .sentiment-bar.unknown  { background: linear(90deg, #8aabdf, #1a3a6b); }
    .sentiment-note {
        font-size: 12px;
        color: var(--ink-3);
        font-weight: 300;
        line-height: 1.6;
    }
    .sentiment-reason {
        margin-top: 10px;
        padding: 12px 14px;
        background: var(--blue-light);
        border-radius: var(--radius-sm);
        border-left: 3px solid var(--blue);
        font-size: 12px;
        color: var(--blue);
        line-height: 1.6;
    }

    /* ─── NEWS ─── */
    .news-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    .news-item {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 16px 20px;
        box-shadow: var(--shadow);
        transition: border-color 0.15s, transform 0.15s;
        cursor: pointer;
    }
    .news-item:hover {
        border-color: var(--accent);
        transform: translateX(4px);
    }
    .news-item-title {
        font-size: 14px;
        font-weight: 500;
        color: var(--ink);
        margin-bottom: 6px;
        line-height: 1.45;
    }
    .news-item-desc {
        font-size: 12px;
        color: var(--ink-3);
        line-height: 1.55;
        font-weight: 300;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .news-item-link {
        display: inline-block;
        margin-top: 8px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--accent);
        text-decoration: none;
    }
    .news-item-link:hover { text-decoration: underline; }

    /* ─── AI ANALYSIS ─── */
    .analysis-hero {
        background: var(--ink);
        border-radius: var(--radius);
        padding: 36px 40px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    .analysis-hero::before {
        content: '';
        position: absolute;
        inset: 0;
        background:
        radial-gradient(ellipse 50% 70% at 90% 20%, rgba(201,79,26,0.22) 0%, transparent 65%),
        radial-gradient(ellipse 40% 50% at 5% 90%, rgba(200,144,26,0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .analysis-hero-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--accent-mid);
        margin-bottom: 8px;
    }
    .analysis-hero-title {
        font-family: var(--serif);
        font-size: 28px;
        color: #faf8f5;
        line-height: 1.15;
        position: relative;
    }
    .analysis-hero-sub {
        font-size: 13px;
        color: rgba(250,248,245,0.45);
        margin-top: 8px;
        font-weight: 300;
        position: relative;
    }

    .analysis-sections {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
        margin-bottom: 20px;
    }
    @media (max-width: 700px) { .analysis-sections { grid-template-columns: 1fr; } }

    .analysis-section {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        box-shadow: var(--shadow);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        cursor: default;
    }
    .analysis-section:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }

    .section-icon {
        font-size: 24px;
        margin-bottom: 10px;
        display: block;
    }
    .analysis-section-title {
        font-family: var(--serif);
        font-size: 17px;
        color: var(--ink);
        margin-bottom: 14px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        letter-spacing: -0.01em;
    }
    .analysis-section-body {
        font-size: 14px;
        line-height: 1.72;
        color: var(--ink-2);
        font-weight: 300;
    }
    .analysis-section-body p + p {
        margin-top: 10px;
    }

    /* Recommendation card gets special treatment */
    .analysis-section.recommendation {
        grid-column: 1 / -1;
        background: var(--surface);
        border: 2px solid var(--accent);
        position: relative;
        overflow: hidden;
    }
    .analysis-section.recommendation::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, rgba(201,79,26,0.05) 0%, transparent 50%);
        pointer-events: none;
    }
    .rec-verdict {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 14px;
    }
    .rec-verdict.buy  { background: var(--green-light); color: var(--green); }
    .rec-verdict.hold { background: var(--amber-light); color: var(--amber); }
    .rec-verdict.sell { background: var(--red-light); color: var(--red); }
    .rec-verdict.strongbuy  { background: #0f5c33; color: #eafff2; }
    .rec-verdict.strongsell { background: #6e1010; color: #ffe9e9; }

    /* Highlighted sentences in recommendation */
    .analysis-section-body mark {
        background: linear-gradient(120deg, rgba(201,79,26,0.18) 0%, rgba(200,144,26,0.22) 100%);
        color: var(--ink);
        padding: 1px 4px;
        border-radius: 3px;
        font-weight: 500;
        -webkit-box-decoration-break: clone;
        box-decoration-break: clone;
    }

    /* ─── TECHNICAL ANALYSIS ─── */
    .tech-verdict-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }
    .tech-verdict {
        display: inline-flex;
        align-items: center;
        padding: 8px 20px;
        border-radius: 999px;
        font-size: 15px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .tech-verdict.strongbuy  { background: #0f5c33; color: #eafff2; }
    .tech-verdict.buy        { background: var(--green-light); color: var(--green); }
    .tech-verdict.sell       { background: var(--red-light); color: var(--red); }
    .tech-verdict.strongsell { background: #6e1010; color: #ffe9e9; }
    .tech-verdict.neutral    { background: var(--amber-light); color: var(--amber); }
    .tech-verdict-label {
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--ink-3);
    }

    .tech-group {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: var(--shadow);
    }
    .tech-group-title {
        font-family: var(--serif);
        font-size: 18px;
        color: var(--ink);
        margin-bottom: 16px;
        padding-bottom: 10px;
        border-bottom: 1px solid var(--border);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .tech-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 12px;
    }
    .tech-indicator {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: var(--radius-sm);
        padding: 14px 16px;
    }
    .tech-indicator-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 8px;
        margin-bottom: 6px;
    }
    .tech-indicator-name {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink-2);
    }
    .tech-indicator-value {
        font-size: 14px;
        font-weight: 600;
        color: var(--ink);
    }
    .tech-indicator-analysis {
        font-size: 13px;
        line-height: 1.6;
        color: var(--ink-2);
        font-weight: 300;
    }
    .tech-summary {
        background: var(--ink);
        color: #faf8f5;
        border-radius: var(--radius);
        padding: 26px 28px;
        margin-top: 8px;
        box-shadow: var(--shadow-lg);
    }
    .tech-summary-title {
        font-family: var(--serif);
        font-size: 18px;
        margin-bottom: 10px;
        color: #faf8f5;
    }
    .tech-summary-body {
        font-size: 14px;
        line-height: 1.72;
        color: rgba(250,248,245,0.82);
        font-weight: 300;
    }

    /* ─── OVERALL SUMMARY ─── */
    .overall-card {
        background: var(--surface);
        border: 2px solid var(--accent);
        border-radius: var(--radius);
        padding: 28px;
        margin-top: 20px;
        box-shadow: var(--shadow-lg);
    }
    .overall-card-title {
        font-family: var(--serif);
        font-size: 22px;
        color: var(--ink);
        margin-bottom: 18px;
    }
    .overall-verdicts {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 12px;
        margin-bottom: 16px;
    }
    .overall-item {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 6px;
    }
    .overall-item-label {
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-3);
    }
    .overall-text {
        font-size: 14px;
        line-height: 1.72;
        color: var(--ink-2);
        font-weight: 300;
        padding-top: 14px;
        border-top: 1px solid var(--border);
    }

    /* ─── FOOTER ─── */
    .footer {
        text-align: center;
        font-size: 11px;
        color: var(--ink-3);
        font-weight: 300;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
    }

    /* ─── FADE IN ─── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-up {
        animation: fadeUp 0.5s ease both;
    }
    .fade-up-1 { animation-delay: 0.05s; }
    .fade-up-2 { animation-delay: 0.12s; }
    .fade-up-3 { animation-delay: 0.20s; }
    .fade-up-4 { animation-delay: 0.28s; }
    .fade-up-5 { animation-delay: 0.36s; }
    </style>
    </head>
    <body>

    <!-- HERO -->
    <div class="hero fade-up">
    <div class="hero-grid">
        <div>
        <div class="hero-label">Stock Analysis Report</div>
        <div class="hero-ticker">{{ stock.ticker }}</div>
        <div class="hero-exchange">{{ stock.exchange }}</div>
        </div>
        <div class="hero-price-block">
        <div class="hero-price-label">Current Price</div>
        <div class="hero-price">₹{{ stock.current_price }}</div>
        <div class="hero-ts">{{ timestamp }}</div>
        </div>
    </div>
    </div>

    <div class="page">

    <!-- METRICS STRIP -->
    <div class="metrics-strip fade-up fade-up-1">
        <div class="metric-card">
        <div class="metric-label">Open</div>
        <div class="metric-value">₹{{ stock.open }}</div>
        <div class="metric-sub">Today's open</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">High</div>
        <div class="metric-value up">₹{{ stock.high }}</div>
        <div class="metric-sub">Day high</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Low</div>
        <div class="metric-value down">₹{{ stock.low }}</div>
        <div class="metric-sub">Day low</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Volume</div>
        <div class="metric-value" id="volume-val">{{ stock.volume }}</div>
        <div class="metric-sub">Shares traded</div>
        </div>
        <div class="metric-card">
        <div class="metric-label">Range</div>
        <div class="metric-value">{{ "%.1f"|format((stock.high - stock.low) / stock.low * 100) }}%</div>
        <div class="metric-sub">High / low spread</div>
        </div>
    </div>

    <!-- SENTIMENT + NEWS -->
    <div class="two-col fade-up fade-up-2">

        <!-- SENTIMENT -->
        <div class="card">
        <h2 class="section-title" style="margin-bottom:16px;">Market Sentiment</h2>

        {% set s = sentiment.sentiment | lower %}
        <div class="sentiment-badge {{ s }}">
            {% if s == 'positive' %}📈 Positive
            {% elif s == 'negative' %}📉 Negative
            {% elif s == 'neutral' %}➡️ Neutral
            {% else %}❓ {{ sentiment.sentiment }}
            {% endif %}
        </div>

        {% if s != 'unknown' %}
        <div class="sentiment-bar-wrap">
            <div class="sentiment-bar {{ s }}" id="sent-bar" style="width: 0%"></div>
        </div>
        <p class="sentiment-note">
            Score: <strong>{{ sentiment.score }}</strong>
            {% if sentiment.articles_analyzed is defined %}
            &nbsp;·&nbsp; Based on {{ sentiment.articles_analyzed }} articles
            {% endif %}
        </p>
        {% endif %}

        {% if sentiment.reason is defined %}
        <div class="sentiment-reason">
            ℹ️ {{ sentiment.reason }}
        </div>
        {% endif %}
        {% if sentiment.error is defined %}
        <div class="sentiment-reason">
            ⚠️ {{ sentiment.error }}
        </div>
        {% endif %}
        </div>

        <!-- NEWS -->
        <div class="card">
        <h2 class="section-title" style="margin-bottom:16px;">Latest News</h2>
        <div class="news-list">
            {% for item in news %}
            <div class="news-item" onclick="window.open('{{ item.url }}', '_blank')">
            <div class="news-item-title">{{ item.title }}</div>
            {% if item.description %}
            <div class="news-item-desc">{{ item.description }}</div>
            {% endif %}
            <a class="news-item-link" href="{{ item.url }}" target="_blank" onclick="event.stopPropagation()">Read more →</a>
            </div>
            {% endfor %}
            {% if not news %}
            <p style="font-size:13px; color:var(--ink-3); font-weight:300;">No recent news found.</p>
            {% endif %}
        </div>
        </div>

    </div>

    <!-- AI ANALYSIS -->
    <div class="fade-up fade-up-3">
        <div class="analysis-hero">
        <div class="analysis-hero-label">AI-Powered Intelligence</div>
        <div class="analysis-hero-title">Investment Analysis</div>
        <div class="analysis-hero-sub">Generated by Senior Wall Street AI · {{ timestamp }}</div>
        </div>

        <div class="analysis-sections">
        {% for section in parsed_sections %}
        <div class="analysis-section {% if section.is_recommendation %}recommendation{% endif %} fade-up fade-up-{{ loop.index + 3 }}">
            <span class="section-icon" aria-hidden="true">{{ section.emoji }}</span>
            {% if section.is_recommendation %}
            {% set rec = section.verdict %}
            {% if rec %}
            <div class="rec-verdict {{ rec | lower }}">{{ rec }}</div>
            {% endif %}
            {% endif %}
            <div class="analysis-section-title">{{ section.title }}</div>
            <div class="analysis-section-body">
            {% for para in section.paragraphs %}
            {% if section.is_recommendation %}
            <p>{{ para | safe }}</p>
            {% else %}
            <p>{{ para | safe }}</p>
            {% endif %}
            {% endfor %}
            </div>
        </div>
        {% endfor %}
        </div>
    </div>

    <!-- TECHNICAL ANALYSIS -->
    <div class="fade-up fade-up-4">
        <div class="analysis-hero">
        <div class="analysis-hero-label">Quantitative Intelligence</div>
        <div class="analysis-hero-title">Technical Analysis</div>
        <div class="analysis-hero-sub">Generated by Quantitative Analyst AI · {{ timestamp }}</div>
        </div>

        {% if tech.groups %}
        <div class="tech-verdict-row">
            <span class="tech-verdict {{ tech.verdict_css }}">{{ tech.verdict }}</span>
            <span class="tech-verdict-label">Technical Signal</span>
        </div>

        {% for group in tech.groups %}
        <div class="tech-group">
            <div class="tech-group-title">
            <span class="section-icon" aria-hidden="true">{{ group.emoji }}</span>
            {{ group.title }}
            </div>
            <div class="tech-grid">
            {% for ind in group.indicators %}
            <div class="tech-indicator">
                <div class="tech-indicator-head">
                <span class="tech-indicator-name">{{ ind.name }}</span>
                {% if ind.value %}<span class="tech-indicator-value">{{ ind.value }}</span>{% endif %}
                </div>
                <div class="tech-indicator-analysis">{{ ind.analysis | safe }}</div>
            </div>
            {% endfor %}
            </div>
        </div>
        {% endfor %}

        {% if tech.summary %}
        <div class="tech-summary">
            <div class="tech-summary-title">🧮 Overall Technical Summary</div>
            <div class="tech-summary-body">{{ tech.summary | safe }}</div>
        </div>
        {% endif %}
        {% else %}
        <div class="tech-summary">
            <div class="tech-summary-title">⚠️ Technical Analysis Unavailable</div>
            <div class="tech-summary-body">{{ tech.error or "No technical analysis was generated for this report." }}</div>
        </div>
        {% endif %}
    </div>

    <!-- OVERALL SUMMARY -->
    <div class="fade-up fade-up-5">
        <div class="overall-card">
        <div class="overall-card-title">🧭 Overall Summary</div>
        <div class="overall-verdicts">
            <div class="overall-item">
            <span class="overall-item-label">Fundamental</span>
            <span class="rec-verdict {{ overall.fundamental_css }}">{{ overall.fundamental_verdict }}</span>
            </div>
            <div class="overall-item">
            <span class="overall-item-label">Technical</span>
            <span class="tech-verdict {{ overall.technical_css }}">{{ overall.technical_verdict }}</span>
            </div>
            <div class="overall-item">
            <span class="overall-item-label">Overall</span>
            <span class="rec-verdict {{ overall.overall_css }}">{{ overall.overall_verdict }}</span>
            </div>
        </div>
        <div class="overall-text">{{ overall.text | safe }}</div>
        </div>
    </div>

    <div class="footer">
        This report is for informational purposes only and does not constitute financial advice. &nbsp;·&nbsp;
        Generated {{ timestamp }}
    </div>

    </div>

    <script>
    // Animate sentiment bar
    (function() {
        const bar = document.getElementById('sent-bar');
        if (!bar) return;
        const score = {{ sentiment.score }};
        const pct = Math.min(Math.max((score + 1) / 2 * 100, 3), 97);
        setTimeout(() => { bar.style.width = pct + '%'; }, 200);
    })();

    // Format volume with commas
    (function() {
        const el = document.getElementById('volume-val');
        if (!el) return;
        const n = parseInt(el.textContent.replace(/,/g, ''));
        if (!isNaN(n)) el.textContent = n.toLocaleString('en-IN');
    })();
    </script>

    </body>
    </html>
"""

    # ═══════════════════════════════════════════════════════════
    # STEP 1 — Parse the LLM's markdown analysis into structured
    #          sections that the Jinja2 template can iterate over.
    # ═══════════════════════════════════════════════════════════

    # Map of expected section titles → metadata.
    # The emoji is used in the rendered card header.
    # `is_recommendation` marks the section that gets special
    # visual treatment (accent border, BUY/HOLD/SELL badge).
    section_map = {
        "Financial Outlook":          {"emoji": "📊", "is_recommendation": False},
        "Key Risks":                  {"emoji": "⚠️",  "is_recommendation": False},
        "Growth Potential":           {"emoji": "🚀", "is_recommendation": False},
        "Investment Recommendation":  {"emoji": "🎯", "is_recommendation": True},
    }

    # Keyword sets used by `detect_verdict` to classify the recommendation.
    # Checked case-insensitively; order matters — BUY words are checked first
    # to avoid "buy" being caught by some spurious match in SELL words.
    BUY_WORDS  = ["buy", "strong buy", "accumulate", "outperform", "overweight"]
    SELL_WORDS = ["sell", "strong sell", "underperform", "underweight", "avoid"]

    # A clause containing one of these negators cannot be read as a directional
    # call: "a SELL rating is not warranted" must not classify as SELL, and
    # "a BUY would be premature" must not classify as BUY.
    NEGATION_RE = re.compile(
        r"\b(?:not|no|never|without|isn't|aren't|wasn't|weren't|doesn't|don't|"
        r"didn't|can't|cannot|unwarranted|premature)\b",
        re.IGNORECASE,
    )

    # A negator only flips a keyword when it sits within this many characters
    # of it AND no clause break lies between the two. The distance limit keeps
    # an unrelated "no" elsewhere in the sentence from suppressing the call,
    # and the clause break stops "the company has no debt and we recommend BUY"
    # (the "no" belongs to the previous clause) from reading as negated.
    NEGATION_WINDOW = 45
    CLAUSE_BREAK_RE = re.compile(
        r"[,;:]|\b(?:and|but|however|yet|although|though|while|whereas|so)\b",
        re.IGNORECASE,
    )

    def keyword_is_directional(low: str, word: str) -> bool:
        """Return True when `word` occurs at least once without a nearby negator.

        Each occurrence is examined separately: it is ignored only if a
        negator lies within ``NEGATION_WINDOW`` characters of it with no
        clause break in between.

        Args:
            low (str):  The lower-cased recommendation body.
            word (str): A keyword from BUY_WORDS / SELL_WORDS.

        Returns:
            bool: True when the keyword should count as a directional signal.
        """
        for match in re.finditer(r'\b' + re.escape(word) + r'\b', low):
            window_start = max(0, match.start() - NEGATION_WINDOW)
            window_end = min(len(low), match.end() + NEGATION_WINDOW)
            negated = False
            for negator in NEGATION_RE.finditer(low, window_start, window_end):
                if negator.end() <= match.start():
                    between = low[negator.end():match.start()]
                else:
                    between = low[match.end():negator.start()]
                if not CLAUSE_BREAK_RE.search(between):
                    negated = True
                    break
            if not negated:
                return True
        return False

    def detect_verdict(text: str) -> str:
        """
        Determine the investment verdict from the recommendation text.

        First attempts to find an **explicit verdict marker** at the very
        start of the text (e.g. ``**HOLD.**``, ``**BUY.**``, ``BUY.``).
        If found, that verdict is used directly — preventing false matches
        from words like "avoid" or "sell" appearing later in the commentary.

        Falls back to keyword scanning only when no explicit marker is present.
        The fallback ignores a keyword when a negator sits within
        ``NEGATION_WINDOW`` characters of it with no clause break between them,
        so commentary that argues against a verdict ("a SELL rating is not
        warranted") is not counted as that verdict.

        Args:
            text (str): The body text of the Investment Recommendation section.

        Returns:
            str: ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        """
        t = text.strip()

        # ── 1. Explicit verdict marker at the start ──
        # Matches patterns like:  **HOLD.**  |  **BUY.**  |  **SELL.**
        #                         HOLD.       |  BUY:      |  SELL —
        m = re.match(
            r'^(?:<mark>)?\s*(?:\*\*)?\s*(BUY|HOLD|SELL)\s*(?:\*\*)?\s*(?:</mark>)?\s*[\.\:\,\-]',
            t, re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # ── 2. Fallback: keyword scan with word-boundaries ──
        # A keyword counts only when it is not negated nearby (see
        # keyword_is_directional), so commentary that argues AGAINST a
        # verdict is never counted as that verdict.
        t_lower = t.lower()
        for w in BUY_WORDS:
            if keyword_is_directional(t_lower, w):
                return "BUY"
        for w in SELL_WORDS:
            if keyword_is_directional(t_lower, w):
                return "SELL"
        return "HOLD"

    parsed_sections = []

    # Split the analysis text on "## " (markdown level-2 headers).
    # The LLM is prompted to use EXACTLY this format, so splitting on
    # `## ` reliably separates the four sections.
    # `re.split` includes the header as the first part of each chunk.
    raw_sections = re.split(r'##\s+', analysis.strip())

    for raw in raw_sections:
        if not raw.strip():
            continue  # Skip empty chunks (e.g., before the first header)

        # First line is the header (e.g. "📊 Financial Outlook")
        # Everything else is body text
        lines = raw.strip().splitlines()
        header_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        # Split body into paragraphs on double-newline boundaries.
        # This preserves the LLM's intended paragraph structure.
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', body) if p.strip()]

        # ── Match this raw header to one of our known sections ──
        # Default values in case the LLM produces an unexpected header
        matched_title = header_line
        emoji = "📌"       # Fallback pushpin emoji
        is_rec = False     # Not a recommendation by default

        for key, meta in section_map.items():
            # Case-insensitive substring match on the header line
            if key.lower() in header_line.lower():
                matched_title = key
                emoji = meta["emoji"]
                is_rec = meta["is_recommendation"]
                break

        # Detect BUY/HOLD/SELL only for the recommendation section
        verdict = detect_verdict(body) if is_rec else None

        parsed_sections.append({
            "title":             matched_title,
            "emoji":             emoji,
            "paragraphs":        paragraphs,
            "is_recommendation": is_rec,
            "verdict":           verdict
        })

    # ═══════════════════════════════════════════════════════════
    # STEP 1b — Extract the fundamental verdict and parse the
    #           technical analysis into cards + overall summary.
    # ═══════════════════════════════════════════════════════════

    # Fundamental verdict (BUY / HOLD / SELL) comes from the recommendation
    # section parsed above — used to build the merged Overall Summary.
    fund_verdict = "HOLD"
    for sec in parsed_sections:
        if sec.get("is_recommendation") and sec.get("verdict"):
            fund_verdict = sec["verdict"]
            break

    # Parse the technical agent's markdown into per-indicator cards, attaching
    # deterministic live values from the DataFrame. If it is absent (or the
    # DataFrame is missing), parse_technical_analysis returns a graceful
    # "unavailable" structure so the report still renders.
    tech_parsed = parse_technical_analysis(tech_analysis or "", tech_df)

    # Merge the two verdicts into a single overall stance.
    overall_summary = build_overall_summary(fund_verdict, tech_parsed.get("verdict", "NEUTRAL"))

    # ── Emphasise the directional wording before rendering ──
    # The template prints these four fields with `| safe`, so the <b> tags
    # inserted here are honoured. Recommendation paragraphs already contain the
    # LLM's own <mark> markup, so they are highlighted WITHOUT escaping; every
    # other field is plain text and is escaped first.
    for sec in parsed_sections:
        if sec.get("is_recommendation"):
            sec["paragraphs"] = [highlight_bull_bear(p) for p in sec["paragraphs"]]
        else:
            sec["paragraphs"] = [highlight_bull_bear_escaped(p) for p in sec["paragraphs"]]
    for group in tech_parsed.get("groups", []):
        for ind in group.get("indicators", []):
            ind["analysis"] = highlight_bull_bear_escaped(ind.get("analysis", ""))
    if tech_parsed.get("summary"):
        tech_parsed["summary"] = highlight_bull_bear_escaped(tech_parsed["summary"])
    if overall_summary.get("text"):
        overall_summary["text"] = highlight_bull_bear_escaped(overall_summary["text"])

    # ═══════════════════════════════════════════════════════════
    # STEP 2 — Render the Jinja2 template with all collected data.
    # ═══════════════════════════════════════════════════════════

    # Compile the inline template string into a Jinja2 Template object
    template = Template(html_template)

    # Render with all context variables.
    # `parsed_sections` is the structured fundamental analysis from Step 1.
    # `tech` is the structured technical analysis (groups, verdict, summary).
    # `overall` is the merged fundamental + technical Overall Summary.
    # `timestamp` is generated fresh at render time for the report footer.
    rendered_html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stock=stock_data,
        sentiment=sentiment_data,
        news=news,
        analysis=analysis,
        parsed_sections=parsed_sections,
        tech=tech_parsed,
        overall=overall_summary
    )

    # ═══════════════════════════════════════════════════════════
    # STEP 3 — Write the rendered HTML to disk.
    # ═══════════════════════════════════════════════════════════

    # Strip exchange suffix so the filename is clean (e.g. "TCS" not "TCS.NS")
    clean_ticker = strip_exchange_suffix(ticker)

    # Date-stamp the filename for traceability
    timestamp_str = datetime.now().strftime("%Y_%m_%d")

    # Construct output filename: e.g. "TCS_REPORT_2026_06_07.html"
    output_file = f"{clean_ticker}_REPORT_{timestamp_str}.html"

    # Write with UTF-8 encoding to support all Unicode characters
    # (emojis, ₹ symbol, international text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)

    # Return the filename so the caller can report or open it
    return output_file


# =========================================================
# MAIN — Orchestration Entry Point
# =========================================================

def main():
    """
    Orchestrate the full stock research pipeline and launch the report.

    This function is the entry point of the agent.  It performs the
    following steps in order:

    1.  **Print banner** — Display a formatted header in the terminal.
    2.  **Collect user input** — Prompt for a ticker symbol and the full
        company name (two separate prompts because NewsAPI searches by
        company name, not ticker).
    3.  **Invoke tools sequentially** — Call ``get_stock_price``,
        ``get_stock_news``, and ``analyze_news_sentiment`` via their
        ``.invoke()`` methods (required by ``@tool``-decorated functions),
        then compute the technical-indicator DataFrame with
        ``TECH_ANALYSIS.compute_technical_indicators()``. This DataFrame is
        the raw material for the quantitative agent: inside
        ``generate_technical_analysis()`` it is converted into a retrievable
        RAG corpus so the LLM analyses only the most relevant indicators.
    4.  **Generate AI analysis (in parallel)** — Run the fundamental analyst
        (``generate_analysis``) and the quantitative technical analyst
        (``generate_technical_analysis``) concurrently in two threads.
    5.  **Render HTML report** — Produce the self-contained ``.html`` file
        (fundamental analysis + technical analysis + merged Overall Summary).
    6.  **Open in browser** — Launch the default web browser to display
        the report immediately.

    Every stage is defensive: a failure in one data source or one agent
    degrades gracefully (missing news, "Unknown" sentiment, or a skipped
    technical section) rather than aborting the whole report.

    Terminal Output Example
    -----------------------
    ::

        ============================================================
        STOCK RESEARCH AI AGENT
        ============================================================

        Enter stock ticker (e.g. INFY, RELIANCE): TCS
        Enter full company name for news/sentiment (e.g. Infosys, Reliance Industries): Tata Consultancy Services

        Fetching stock data...
        Generating AI analysis...
        Generating HTML report...

        HTML report generated: TCS_REPORT_2026_06_07.html
    """

    # ── Banner ──
    # "=" * 60 produces a 60-character separator line
    print("\n" + "=" * 60)
    print("STOCK RESEARCH AI AGENT")
    print("=" * 60)

    # ── User Input ──
    # Two separate prompts because:
    # - get_stock_price needs a ticker (e.g. "TCS")
    # - get_stock_news / analyze_news_sentiment need the full company name
    #   (e.g. "Tata Consultancy Services") for accurate NewsAPI search results.
    # .strip().upper() normalizes the ticker; company name is only stripped.
    ticker = input("\nEnter stock ticker (e.g. INFY, RELIANCE): ").strip().upper()
    company_name = input(
        "Enter full company name for news/sentiment "
        "(e.g. Infosys, Reliance Industries): "
    ).strip()

    # ── Data Collection ──
    # Each @tool function must be called with .invoke() (not called directly)
    # because LangChain's @tool decorator wraps the function and expects
    # a single string argument through the .invoke() interface.

    print("\nFetching stock data...")
    stock_data = get_stock_price.invoke(ticker)

    news = get_stock_news.invoke(company_name)
    sentiment_data = analyze_news_sentiment.invoke(company_name)

    # ── Technical Indicators ──
    # Download 5 years of daily OHLCV data and compute the full technical
    # indicator DataFrame (moving averages, RSI, MACD, Bollinger, Stochastic,
    # OBV). This DataFrame feeds the quantitative analyst agent — inside
    # generate_technical_analysis() it is turned into a retrievable RAG corpus
    # (see _build_technical_corpus) so the LLM only sees the most relevant
    # indicator context for each part of the analysis.
    #
    # The computation is wrapped in try/except so a data failure here cannot
    # block the rest of the report: on error we set tech_df = None, which
    # (a) prevents the technical agent from being launched and (b) makes the
    # HTML report render a graceful "Technical Analysis Unavailable" card.
    print("Computing technical indicators...")
    try:
        tech_df = TECH_ANALYSIS.compute_technical_indicators(ticker)
    except Exception as e:
        # Don't let a technical-data failure stop the whole report.
        print(f"  [!] Could not compute technical indicators: {e}")
        tech_df = None

    # ── Parallel AI Analysis ──
    # Two LLM agents run IN PARALLEL using a thread pool:
    #   1. generate_analysis()           — fundamental / sentiment analyst
    #   2. generate_technical_analysis() — quantitative technical analyst
    # Each llm.invoke() call is blocking (an HTTP request), so running them
    # in two threads gives true concurrency and roughly halves the wait time.
    #
    # Thread safety: the two futures never share mutable state. Each agent
    # receives its own prompt built from read-only inputs (stock_data,
    # sentiment_data, news / tech_df), so there are no locks or shared-write
    # hazards between the worker threads.
    print("Running AI agents in parallel (fundamental + technical)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_fund = executor.submit(
            generate_analysis, ticker, stock_data, sentiment_data, news
        )

        # Only launch the technical agent if we have indicator data. When
        # tech_df is None (the indicator computation failed above) we submit
        # nothing and keep future_tech = None — the collection block below
        # then skips it and the report renders without the technical section.
        future_tech = (
            executor.submit(generate_technical_analysis, ticker, tech_df)
            if tech_df is not None else None
        )

        # Collect the fundamental analysis. future_fund.result() blocks the
        # main thread until the agent finishes, then returns the raw markdown.
        # If the LLM call raised (quota, network, malformed response), fall
        # back to an empty string so the report still renders.
        try:
            analysis = future_fund.result()
        except Exception as e:
            print(f"  [!] Fundamental analysis failed: {e}")
            analysis = ""

        # Collect the technical analysis with the same graceful fallback.
        # Note the `if future_tech else None` guard: when the technical agent
        # was never launched (tech_df was None) we return None directly
        # instead of calling .result() on an un-submitted future.
        try:
            tech_analysis = future_tech.result() if future_tech else None
        except Exception as e:
            print(f"  [!] Technical analysis failed: {e}")
            tech_analysis = None

    # ── HTML Report Generation ──
    # Parses both analyses into structured sections, merges them into an
    # Overall Summary, renders the Jinja2 template, and writes the result
    # to a date-stamped .html file (e.g. TCS_REPORT_2026_08_25.html). The
    # filename comes from the ticker with its exchange suffix stripped, and
    # the report is fully self-contained (inline CSS and JavaScript).
    print("Generating HTML report...")
    report_file = generate_html_report(
        ticker, stock_data, sentiment_data, news, analysis,
        tech_analysis=tech_analysis, tech_df=tech_df
    )

    # ── Launch Browser ──
    # `webbrowser.open()` opens the HTML file in the user's default browser.
    # On Windows, this typically launches Edge/Chrome; on macOS, Safari/Chrome.
    # The call is non-blocking, so the script exits immediately afterwards
    # while the report stays open in the browser.
    print(f"\nHTML report generated: {report_file}")
    webbrowser.open(report_file)


# ── Script Entry Point ──
# When run as `python stock_react_agent.py`, call main().
# When imported as a module, main() is not called automatically.
if __name__ == "__main__":
    main()
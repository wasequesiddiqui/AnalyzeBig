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

2.  **AI Analysis Layer** — `generate_analysis()` sends the collected data to a
    DeepSeek LLM with a structured prompt that instructs it to act as a senior
    Wall Street analyst and produce four clearly labeled sections.

3.  **Presentation Layer** — `generate_html_report()` takes the AI-generated
    markdown, parses it into structured sections, detects the investment verdict
    (BUY / HOLD / SELL), and renders a responsive, animated HTML dashboard using
    Jinja2 templates with embedded CSS and JavaScript.

4.  **Orchestration** — `main()` wires everything together: prompts the user for
    a ticker and company name, invokes each tool sequentially, calls the LLM for
    analysis, generates the HTML report, and automatically opens it in the
    default browser.

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

import os
import re
import webbrowser
import yfinance as yf  # Yahoo Finance — fetches real-time & historical stock data

from datetime import datetime
from textblob import TextBlob  # Lexicon-based NLP sentiment analyzer

from dotenv import load_dotenv  # Loads .env file into os.environ
from newsapi import NewsApiClient  # NewsAPI.org client for financial headlines

from jinja2 import Template  # Jinja2 HTML templating engine

from langchain.tools import tool  # Decorator to register functions as LLM-callable tools
from langchain_openai import ChatOpenAI  # OpenAI-compatible chat model (used with DeepSeek)

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

# =========================================================
# TOOLS
# =========================================================

@tool
def get_stock_price(ticker: str) -> dict:
    """
    Fetch the latest stock market data (OHLCV) for a given ticker symbol.

    This tool tries multiple exchange suffixes to locate the stock across
    different markets.  It first attempts the Indian NSE (``.NS``), then
    the Indian BSE (``.BO``), and finally falls back to the raw ticker
    for global exchanges (NYSE, NASDAQ, etc.).  Whichever attempt returns
    non-empty price history first is used.

    **LangChain Tool** — when decorated with ``@tool``, the function's
    docstring becomes the description the LLM sees when deciding whether
    to invoke this tool.  Keep it descriptive but concise.

    Args:
        ticker (str): The ticker symbol to look up (e.g. ``"INFY"``,
                      ``"TCS"``, ``"AAPL"``).  Case-insensitive; leading/
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
        >>> get_stock_price.invoke("INFY")
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
        # Normalize input: uppercase, strip whitespace
        ticker = ticker.upper().strip()

        # Build a priority list of symbols to try.
        # Indian stocks are dual-listed on NSE and BSE; we try .NS first
        # (more liquid), then .BO, then the raw ticker for global stocks.
        possible_tickers = [
            f"{ticker}.NS",   # National Stock Exchange of India
            f"{ticker}.BO",   # Bombay Stock Exchange
            ticker             # Raw ticker — NYSE / NASDAQ / LSE etc.
        ]

        stock_data = None  # Will hold the result dict if we find data

        for symbol in possible_tickers:
            # Create a yfinance Ticker object for this symbol
            stock = yf.Ticker(symbol)

            # Request 5 trading days of 1-day-interval OHLCV bars.
            # 5 days ensures we cover a weekend gap and still get a price.
            hist = stock.history(period="5d")

            # An empty DataFrame means yfinance couldn't find this symbol
            if not hist.empty:
                # Get the most recent row (latest trading day's bar)
                latest = hist.iloc[-1]

                # Determine human-readable exchange label
                exchange = "Global"
                if symbol.endswith(".NS"):
                    exchange = "NSE India"
                elif symbol.endswith(".BO"):
                    exchange = "BSE India"

                # Build the result dictionary with rounded values
                stock_data = {
                    "ticker": symbol,
                    "exchange": exchange,
                    "current_price": round(latest["Close"], 2),
                    "open": round(latest["Open"], 2),
                    "high": round(latest["High"], 2),
                    "low": round(latest["Low"], 2),
                    "volume": int(latest["Volume"])
                }

                # Stop after the first successful match
                break

        # If no symbol produced data, return a clear error
        if stock_data is None:
            return {
                "error": "No stock data found."
            }

        return stock_data

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
[1-2 paragraphs with a clear BUY / HOLD / SELL stance and reasoning.
In THIS section ONLY, wrap the single most critical sentence in each paragraph with <mark>...</mark> tags so it stands out visually.]

Keep it concise, data-driven, and professional. Do not add any preamble before the first section header.
Do NOT use <mark> tags in any other section.
"""

    # Send the prompt to DeepSeek via langchain-openai.
    # `invoke()` returns an AIMessage; `.content` extracts the text.
    response = llm.invoke(prompt)

    return response.content


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
    analysis: str
) -> str:
    """
    Render a complete, self-contained HTML stock analysis report.

    This is the presentation engine of the agent.  It takes the raw outputs
    from all three tools plus the LLM-generated analysis and produces a
    single ``.html`` file with embedded CSS and JavaScript — no external
    dependencies beyond Google Fonts.

    Processing Pipeline
    -------------------
    1.  **Parse the LLM analysis** — Split the markdown on ``##`` headers,
        match each section against a known mapping (Financial Outlook, Key
        Risks, Growth Potential, Investment Recommendation), extract
        paragraphs, and detect the BUY/HOLD/SELL verdict from keywords.

    2.  **Render via Jinja2** — The parsed data (stock metrics, sentiment,
        news articles, analysis sections) is injected into a rich HTML
        template with responsive CSS grid layout, animated sentiment bar,
        volume number formatting, and color-coded recommendation badges.

    3.  **Write to disk** — The rendered HTML is saved as
        ``{CLEAN_TICKER}_REPORT_{YYYY_MM_DD}.html`` in the current working
        directory.

    Args:
        ticker (str):           Raw ticker symbol (may include exchange suffix),
                                e.g. ``"TCS.NS"``.
        stock_data (dict):      Output from ``get_stock_price()``.
        sentiment_data (dict):  Output from ``analyze_news_sentiment()``.
        news (list[dict]):      Output from ``get_stock_news()``.
        analysis (str):         Raw markdown output from ``generate_analysis()``.

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
            <p>{{ para }}</p>
            {% endif %}
            {% endfor %}
            </div>
        </div>
        {% endfor %}
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

    def detect_verdict(text: str) -> str:
        """
        Determine the investment verdict from the recommendation text.

        First attempts to find an **explicit verdict marker** at the very
        start of the text (e.g. ``**HOLD.**``, ``**BUY.**``, ``BUY.``).
        If found, that verdict is used directly — preventing false matches
        from words like "avoid" or "sell" appearing later in the commentary.

        Falls back to keyword scanning only when no explicit marker is present.

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
            r'^(?:\*\*)?\s*(BUY|HOLD|SELL)\s*(?:\*\*)?\s*[\.\:\,\-]',
            t, re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # ── 2. Fallback: keyword scan with word-boundaries ──
        t_lower = t.lower()
        for w in BUY_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', t_lower):
                return "BUY"
        for w in SELL_WORDS:
            if re.search(r'\b' + re.escape(w) + r'\b', t_lower):
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
    # STEP 2 — Render the Jinja2 template with all collected data.
    # ═══════════════════════════════════════════════════════════

    # Compile the inline template string into a Jinja2 Template object
    template = Template(html_template)

    # Render with all context variables.
    # `parsed_sections` is the structured analysis data from Step 1.
    # `timestamp` is generated fresh at render time for the report footer.
    rendered_html = template.render(
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        stock=stock_data,
        sentiment=sentiment_data,
        news=news,
        analysis=analysis,
        parsed_sections=parsed_sections
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
        ``.invoke()`` methods (required by ``@tool``-decorated functions).
    4.  **Generate AI analysis** — Pass all collected data to the LLM.
    5.  **Render HTML report** — Produce the self-contained ``.html`` file.
    6.  **Open in browser** — Launch the default web browser to display
        the report immediately.

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

    # ── AI Analysis ──
    # The LLM receives all collected data and returns a structured    markdown
    # analysis with four labeled sections.
    print("Generating AI analysis...")
    analysis = generate_analysis(ticker, stock_data, sentiment_data, news)

    # ── HTML Report Generation ──
    # Parses the LLM output into structured sections, renders the Jinja2
    # template, and writes the result to a date-stamped .html file.
    print("Generating HTML report...")
    report_file = generate_html_report(
        ticker, stock_data, sentiment_data, news, analysis
    )

    # ── Launch Browser ──
    # `webbrowser.open()` opens the HTML file in the user's default browser.
    # On Windows, this typically launches Edge/Chrome; on macOS, Safari/Chrome.
    print(f"\nHTML report generated: {report_file}")
    webbrowser.open(report_file)


# ── Script Entry Point ──
# When run as `python stock_react_agent.py`, call main().
# When imported as a module, main() is not called automatically.
if __name__ == "__main__":
    main()
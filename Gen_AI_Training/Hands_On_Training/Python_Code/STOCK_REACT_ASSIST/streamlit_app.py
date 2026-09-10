"""
Stock Research AI Agent — Streamlit Dashboard
==============================================
An interactive stock research dashboard built with Streamlit that combines
real-time market data, financial news aggregation, AI-powered sentiment
analysis, DCF valuation, vectorless RAG on uploaded regulatory filings, and a
parallel quantitative technical-analysis agent — all from a modern web UI.

The quantitative technical agent reuses stock_react_agent's RAG pipeline: the
5-year technical-indicator DataFrame (TECH_ANALYSIS.py) is split into a
retrievable document corpus, relevant documents are retrieved by a lightweight
keyword scorer, and the LLM classifies the stock into
STRONG BUY / BUY / SELL / STRONG SELL. The result is merged with the
fundamental verdict into an Overall Summary.

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
import io
import streamlit as st
import yfinance as yf
import pandas as pd
import ALL_DCF_MODELS as adm

from datetime import datetime
from concurrent.futures import ThreadPoolExecutor  # Runs the two AI agents in parallel
from textblob import TextBlob
from dotenv import load_dotenv
from newsapi import NewsApiClient
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

import TECH_ANALYSIS  # Builds the 5-year technical-indicator DataFrame (OHLCV + indicators)
import stock_react_agent as sra  # Reuse the RAG-based quantitative technical agent

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
    st.session_state.dcf_data = None
    # Technical-analysis agent state (RAG-grounded quantitative analysis): the
    # indicator DataFrame, the raw LLM markdown, the parsed cards/verdict, and
    # the merged fundamental + technical Overall Summary.
    st.session_state.tech_df = None
    st.session_state.tech_analysis = None
    st.session_state.tech_parsed = None
    st.session_state.overall_summary = None
    # Vectorless RAG state: raw uploaded files (bytes snapshot, extracted only
    # on Analyze), extracted chunks, per-file summary, dedup keys, and the
    # chunks retrieved at analysis time.
    st.session_state.uploaded_files_raw = []
    st.session_state.uploaded_keys = []
    st.session_state.document_corpus = []
    st.session_state.rag_sources = []
    st.session_state.rag_keys = []
    st.session_state.retrieved_chunks = []
    # crewAI multi-persona agentic valuation state (independent of the standard
    # pipeline): persona_result holds the serialised PersonaCrewResult dict.
    st.session_state.persona_result = None
    st.session_state.persona_running = False

# =========================================================
# LLM SETUP
# =========================================================
# We instantiate a ChatOpenAI client pointed at DeepSeek's OpenAI-compatible
# REST API, exactly as in stock_react_agent.py. DeepSeek exposes the same
# /chat/completions interface as OpenAI, so langchain-openai's ChatOpenAI can
# talk to it by overriding base_url. temperature=0 keeps the analysis
# deterministic (best for factual stock reports).
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
    """Strip the Indian exchange suffix from a ticker symbol.

    Removes a trailing ``.NS`` (NSE) or ``.BO`` (BSE) suffix so the bare
    company symbol can be used for display filenames and news queries, e.g.
    ``"INFY.NS"`` → ``"INFY"``. Tickers without a suffix pass through
    unchanged. Uppercases and trims the input for consistency.
    """
    # Split on ".NS" / ".BO" anchored at the END of the string and keep the
    # first part (the base ticker). re.split returns the prefix as element 0.
    return re.split(r'\.(NS|BO)$', ticker.upper().strip())[0]


def format_volume(vol: int) -> str:
    """Format a share volume using the Indian numbering system.

    Indian markets quote large volumes in lakhs and crores rather than
    millions/billions:
        * ``>= 10,000,000`` → e.g. ``"3.25 Cr"``  (1 crore = 10 million)
        * ``>= 100,000``    → e.g. ``"12.50 L"``   (1 lakh  = 100 thousand)
        * otherwise         → comma-grouped, e.g. ``"45,678"``
    """
    if vol >= 10000000:
        return f"{vol / 10000000:.2f} Cr"
    elif vol >= 100000:
        return f"{vol / 100000:.2f} L"
    return f"{vol:,}"


def get_sentiment_emoji(s: str) -> str:
    """Map a sentiment label to a display emoji.

    Handles the full range of labels produced by ``analyze_news_sentiment``
    (Positive / Negative / Neutral / Unknown / Error); unrecognised labels
    fall back to a neutral question mark.
    """
    mapping = {
        "positive": "📈",
        "negative": "📉",
        "neutral": "➡️",
        "unknown": "❓",
        "error": "⚠️"
    }
    return mapping.get(s.lower(), "❓")


def get_sentiment_color(s: str) -> str:
    """Map a sentiment label to a Streamlit theme color name.

    The returned names are used for badge / progress-bar styling via CSS
    variables (``--green``, ``--red``, ``--orange``, ``--blue``). The
    ``error`` state intentionally falls through to the neutral ``gray``
    default since an error is not a directional sentiment.
    """
    mapping = {
        "positive": "green",
        "negative": "red",
        "neutral": "orange",
        "unknown": "blue",
    }
    return mapping.get(s.lower(), "gray")


def get_tech_verdict_style(verdict: str) -> tuple:
    """Return the (color, emoji) pair used to render a technical verdict badge.

    Maps the quantitative agent's four classification flags — STRONG BUY /
    BUY / SELL / STRONG SELL — plus the NEUTRAL fallback to a Streamlit
    theme color and a matching emoji. The color name feeds CSS variables
    (``--green``, ``--red``, ``--orange``) used for badge styling.

    Args:
        verdict (str): The technical verdict flag, e.g. ``"STRONG BUY"``.
                       Case-insensitive (normalised via ``verdict.upper()``).

    Returns:
        tuple[str, str]: ``(color, emoji)``, e.g. ``("green", "🚀")``.
                         Unrecognised values fall back to ``("gray", "ℹ️")``.
    """
    mapping = {
        "STRONG BUY":  ("green", "🚀"),
        "BUY":         ("green", "📈"),
        "SELL":        ("red", "📉"),
        "STRONG SELL": ("red", "🔻"),
        "NEUTRAL":     ("orange", "➡️"),
    }
    return mapping.get(verdict.upper(), ("gray", "ℹ️"))


# =========================================================
# CURRENCY HELPERS
# =========================================================

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "AUD": "A$",
    "CAD": "C$",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fx_lookup(ticker: str):
    """Fetch the latest close for a Yahoo Finance FX ticker (e.g. ``"INR=X"``).

    Yahoo exposes currency pairs as tickers of the form ``"USD=X"`` /
    ``"INR=X"``, whose close is the value of one unit of the first code in
    the second. We read the most recent daily close and return it as a float.

    Cached with ``@st.cache_data`` (TTL 1 hour) so repeated currency
    conversions don't hammer Yahoo; ``show_spinner=False`` keeps the lookup
    silent in the UI.

    Args:
        ticker (str): A Yahoo Finance FX symbol, e.g. ``"INR=X"``.

    Returns:
        float | None: The latest close, or ``None`` on any failure (network
                      error, unknown symbol, or empty price history).
    """
    try:
        hist = yf.Ticker(ticker).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """Return the live exchange rate between two ISO currency codes.

    The result is the value of ONE unit of ``from_currency`` expressed in
    ``to_currency`` — e.g. ``get_fx_rate("USD", "INR")`` ≈ ``83.5`` means
    1 USD ≈ ₹83.5.

    Resolution strategy (all via Yahoo Finance FX tickers):
      * Identical currencies → ``1.0`` (no network lookup needed).
      * From USD            → fetch ``"{to_currency}=X"`` directly.
      * To USD              → fetch ``"{from_currency}=X"`` and take the
        reciprocal.
      * Neither is USD      → chain through USD (triangular conversion):
        ``FROM→USD`` then ``USD→TO`` and multiply.

    Args:
        from_currency (str): ISO code of the source currency, e.g. ``"USD"``.
        to_currency (str):   ISO code of the target currency, e.g. ``"INR"``.

    Returns:
        float: The rate. Degrades gracefully to ``1.0`` if a lookup fails so
               the UI shows unconverted prices rather than crashing.
    """
    if from_currency == to_currency:
        return 1.0
    try:
        if from_currency == "USD":
            rate = _fx_lookup(f"{to_currency}=X")  # USD/TO rate
            return rate if rate else 1.0
        if to_currency == "USD":
            rate = _fx_lookup(f"{from_currency}=X")  # USD/FROM rate
            return (1.0 / rate) if rate else 1.0
        usd = get_fx_rate(from_currency, "USD")
        to = get_fx_rate("USD", to_currency)
        return usd * to
    except Exception:
        return 1.0


def convert_price(value: float, from_currency: str, to_currency: str) -> float:
    """Convert a monetary value from one currency to another.

    Multiplies the value by the live rate from ``get_fx_rate`` (the value of
    one unit of ``from_currency`` in ``to_currency``) and rounds to 2 decimal
    places for clean display.

    Args:
        value (float):       The amount in ``from_currency``, e.g. a price.
        from_currency (str): ISO code of the source currency.
        to_currency (str):   ISO code of the target currency.

    Returns:
        float: The converted amount, rounded to 2 dp.
    """
    return round(value * get_fx_rate(from_currency, to_currency), 2)


def format_price(value: float, native_currency: str, display_currency: str) -> str:
    """Format a price in the user's chosen display currency for rendering.

    Converts ``value`` from its native currency into ``display_currency`` and
    returns a human-friendly string — the currency symbol followed by the
    comma-grouped, two-decimal amount, e.g. ``"₹1,450.50"``.

    Args:
        value (float):           The price in its native currency.
        native_currency (str):   ISO code the raw price is denominated in.
        display_currency (str):  ISO code to convert to for display.

    Returns:
        str: The formatted price string, e.g. ``"$182.31"``.
    """
    symbol = CURRENCY_SYMBOLS.get(display_currency, display_currency)
    converted = convert_price(value, native_currency, display_currency)
    return f"{symbol}{converted:,.2f}"

def current_selected_currency() -> str:
    """Return the display currency currently selected in the sidebar.

    Reads the ``"display_currency"`` key that Streamlit stores in
    ``session_state`` for the currency ``st.selectbox``; falls back to
    ``"INR"`` (the default) if the widget has not been touched yet.

    Returns:
        str: An ISO currency code, e.g. ``"INR"``, ``"USD"``, ``"EUR"``.
    """
    return st.session_state.get("display_currency", "INR")


def currency_display_string(currency: str = "") -> str:
    """Return the currency symbol + trailing space for the DCF module.

    The DCF valuation module (``ALL_DCF_MODELS``) expects a currency prefix
    string such as ``"₹ "`` to label its outputs, rather than an ISO code.
    This helper resolves the active display currency (from ``currency`` or the
    sidebar selection) into that symbol form.

    Args:
        currency (str): Optional ISO code override. When empty, the sidebar's
                        selected currency is used.

    Returns:
        str: The symbol followed by a space, e.g. ``"₹ "`` or ``"$ "``.
    """
    currency = currency or current_selected_currency()
    symbol = CURRENCY_SYMBOLS.get(currency, currency)
    return f"{symbol} "


@tool
def get_dcf_valuation(ticker: str) -> dict:
    """Fetch multi-model DCF valuation data for a given ticker symbol.

    **LangChain Tool** — callable by the LLM when it needs intrinsic-value
    estimates. Delegates to ``ALL_DCF_MODELS.run_valuation_analysis``, which
    computes fair value per share across multiple DCF model variations (base /
    bull / bear scenarios) and returns RMSE-style accuracy metrics.

    Args:
        ticker (str): The complete Yahoo Finance symbol to value, e.g.
                      ``"INFY.NS"`` or ``"AAPL"``. Used exactly as given —
                      no exchange suffix is appended.

    Returns:
        dict: On success, a dict with keys ``ticker``, ``currency``,
              ``base_val``, ``bull_val``, ``bear_val``, ``average_all``,
              ``all_values_per_share``, ``scenario_results`` and RMSE metrics.
              On failure, ``{"error": <message>}``.
    """
    try:
        # The complete Yahoo Finance symbol is used exactly as entered.
        resolved = ticker.strip().upper()
        dcf_data = adm.run_valuation_analysis(
            ticker=resolved,
            currency=currency_display_string(),
            verbose=False,
        )
        if not dcf_data:
            return {"error": "No DCF data found."}
        return dcf_data
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_price(ticker: str) -> dict:
    """Fetch the latest OHLCV stock market data for a given ticker symbol.

    **LangChain Tool** — callable by the LLM when it needs the current price
    snapshot. The symbol must be the COMPLETE Yahoo Finance ticker (e.g.
    ``"INFY.NS"`` for NSE, ``"INFY.BO"`` for BSE, ``"AAPL"`` for US markets):
    it is fetched exactly as given, with no exchange suffix ever appended.

    Args:
        ticker (str): The complete ticker symbol, e.g. ``"INFY.NS"`` or
                      ``"AAPL"``. Case-insensitive; surrounding whitespace is
                      stripped.

    Returns:
        dict: On success, keys ``ticker``, ``exchange`` (``"NSE India"`` /
              ``"BSE India"`` / ``"Global"``), ``currency`` (``"INR"`` or
              ``"USD"``), ``current_price``, ``open``, ``high``, ``low`` and
              ``volume`` (OHLCV values rounded for display).
              On failure, ``{"error": <message>}``.
    """
    try:
        # Consume the value exactly as entered — the user supplies the full
        # Yahoo Finance symbol, so no ".NS" / ".BO" suffix is appended here.
        symbol = ticker.upper().strip()
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")
        if hist.empty:
            return {"error": f"No stock data found for '{symbol}'."}
        latest = hist.iloc[-1]
        exchange = "Global"
        if symbol.endswith(".NS"):
            exchange = "NSE India"
        elif symbol.endswith(".BO"):
            exchange = "BSE India"
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
        return {"error": str(e)}


@tool
def get_stock_news(company_name: str) -> list:
    """Fetch the latest financial news headlines for a given company.

    **LangChain Tool** — provides the LLM with current headlines. Queries the
    NewsAPI ``/v2/everything`` endpoint with ``"{company_name} stock"`` as the
    search term, sorted newest-first, capped at 5 articles to keep the report
    focused and the LLM context small.

    Args:
        company_name (str): The FULL company name (not the ticker), e.g.
                            ``"Infosys"`` or ``"Reliance Industries"``.

    Returns:
        list[dict]: Up to 5 articles, each with ``title``, ``description``,
                    ``url``, ``source`` and ``publishedAt``. Empty list if no
                    articles are found; on error, a single-element list with a
                    structured ``{"title": "Error", ...}`` entry.
    """
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
    """Compute the aggregate market sentiment for a company from recent news.

    **LangChain Tool** — provides a quantitative sentiment signal that
    complements the raw headlines. Fetches up to 20 recent English articles,
    scores each article's title + description with TextBlob polarity (−1.0 to
    +1.0), and averages the scores into a label using conservative ±0.15
    thresholds:
        * ``avg_score > +0.15``  → ``"Positive"``
        * ``avg_score < −0.15``  → ``"Negative"``
        * otherwise              → ``"Neutral"``

    Args:
        company_name (str): Full company name for the NewsAPI search.

    Returns:
        dict: On success, ``{"sentiment", "score", "articles_analyzed"}``.
              No articles → ``{"sentiment": "Unknown", "score": 0, "reason"}``.
              On error  → ``{"sentiment": "Error", "score": 0, "error"}``.
    """
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

def generate_analysis(ticker: str, stock_data: dict, sentiment_data: dict, news: list, dcf_data: dict, display_currency: str = "INR", rag_context: str = "") -> str:
    """Generate a structured AI investment analysis using the DeepSeek LLM.

    Builds a prompt that role-plays the model as a senior Wall Street analyst
    and requires EXACTLY four markdown sections (Financial Outlook, Key Risks,
    Growth Potential, Investment Recommendation). Every collected dataset —
    live price, sentiment, news, DCF valuation and (optionally) vectorless-RAG
    filing context — is injected so the analysis is data-grounded rather than
    generic.

    When ``rag_context`` is non-empty it is embedded with explicit instructions
    telling the model to treat it as authoritative company-disclosed evidence
    (revenue, profit, debt, guidance, risk factors) and to cite sources inline
    as ``[File: <filename>, Page <n>]``.

    Args:
        ticker (str):            The ticker symbol being analysed.
        stock_data (dict):       Output of ``get_stock_price()`` (OHLCV).
        sentiment_data (dict):   Output of ``analyze_news_sentiment()``.
        news (list[dict]):       Output of ``get_stock_news()``.
        dcf_data (dict):         Output of ``get_dcf_valuation()``.
        display_currency (str):  ISO code used to label prices in the prompt.
        rag_context (str):       Serialized vectorless-RAG excerpts ('' = none).

    Returns:
        str: The raw markdown analysis (parsed later by ``parse_analysis``),
             with the four ``##``-delimited sections and ``<mark>`` highlights
             confined to the recommendation section.
    """
    rag_section = ""
    if rag_context.strip():
        rag_section = f"""
Regulatory Filing Context (extracted from the uploaded documents via vectorless RAG):
{rag_context}

Instructions for using the Regulatory Filing Context:
- Treat it as authoritative company-disclosed information (e.g. revenue,
  profit, debt, guidance, and risk factors).
- Use it to enrich and ground every section of your analysis.
- When you cite a figure from the filings, reference its source inline as
  [File: <filename>, Page <n>].
- Do NOT invent facts that appear in neither the filing context nor the
  market/valuation data above.
"""
    prompt = f"""
You are a senior Wall Street analyst.

Analyze this stock.

Ticker:
{ticker}

Display Currency:
{display_currency}

Stock Data:
{stock_data}

Sentiment:
{sentiment_data}

News:
{news}

Discounted Cashflow Analysis with all scenarios (valuation) data:
{dcf_data}

{rag_section}
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
    response = llm.invoke(prompt)
    return response.content


# =========================================================
# ANALYSIS PARSER
# =========================================================

def parse_analysis(analysis: str) -> list:
    """Parse the LLM's raw markdown analysis into structured sections.

    Splits the markdown on ``##`` headers, matches each header against the
    four known sections, extracts paragraphs, and (for the recommendation
    section only) detects a BUY / HOLD / SELL verdict.

    Verdict detection is two-stage: an explicit leading marker (e.g.
    ``**BUY.**`` or ``BUY:``) wins immediately; otherwise the body is scanned
    for BUY/SELL keywords using word-boundary regexes, ignoring keywords that
    are negated nearby ("a SELL rating is not warranted") and defaulting to
    HOLD when nothing matches.

    Args:
        analysis (str): Raw markdown output of ``generate_analysis()``.

    Returns:
        list[dict]: One dict per section with keys ``title``, ``emoji``,
                    ``paragraphs``, ``is_recommendation`` and ``verdict``
                    (``verdict`` is ``None`` for non-recommendation sections).
    """
    section_map = {
        "Financial Outlook":          {"emoji": "📊", "is_recommendation": False},
        "Key Risks":                  {"emoji": "⚠️",  "is_recommendation": False},
        "Growth Potential":           {"emoji": "🚀", "is_recommendation": False},
        "Investment Recommendation":  {"emoji": "🎯", "is_recommendation": True},
    }
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
        """Infer a BUY / HOLD / SELL verdict from the recommendation body.

        First looks for an explicit verdict marker at the very start of the
        text (``**HOLD.**``, ``BUY:``, ``SELL —`` …). If found, it is returned
        directly, preventing false positives from words like "avoid" or "sell"
        appearing later in the commentary. Otherwise the body is scanned for
        BUY keywords, then SELL keywords — a keyword is ignored when a negator
        sits within ``NEGATION_WINDOW`` characters of it with no clause break
        in between, so "a SELL rating is not warranted" does not read as SELL —
        and defaults to HOLD when nothing matches.

        Args:
            text (str): The body text of the Investment Recommendation section.

        Returns:
            str: ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        """
        t = text.strip()
        m = re.match(
            r'^(?:<mark>)?\s*(?:\*\*)?\s*(BUY|HOLD|SELL)\s*(?:\*\*)?\s*(?:</mark>)?\s*[\.\:\,\-]',
            t, re.IGNORECASE
        )
        if m:
            return m.group(1).upper()

        # ── 2. Fallback for output with no leading marker ──
        # A keyword counts only when it is not negated nearby (see
        # keyword_is_directional), so commentary that argues AGAINST a verdict
        # is never counted as that verdict.
        t_lower = t.lower()
        for w in BUY_WORDS:
            if keyword_is_directional(t_lower, w):
                return "BUY"
        for w in SELL_WORDS:
            if keyword_is_directional(t_lower, w):
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
# VECTORLESS RAG  (in-memory document context — no vector DB)
# =========================================================
#
# WHAT IS VECTORLESS RAG?
# -----------------------
# Retrieval-Augmented Generation (RAG) normally works like this:
#     1. Chunk the documents, then embed every chunk into a high-dimensional
#        vector using an embedding model (OpenAI, BGE, etc.).
#     2. Store the vectors in a vector database (FAISS, Chroma, Pinecone...).
#     3. Embed the user's question, run a vector similarity search, and return
#        the top-k most similar chunks.
#     4. Stuff those chunks into the LLM prompt as extra context.
#
# "Vectorless RAG" keeps the *retrieval + augmented generation* idea but drops
# the embedding + vector-database machinery entirely:
#     1. EXTRACT  — pull the raw text out of each uploaded PDF with pypdf.
#     2. CHUNK    — split the text into overlapping chunks (chunk_text) so the
#        LLM can read the filing in digestible pieces, keeping surrounding
#        context across chunk boundaries.
#     3. STORE    — keep the chunks in a plain Python list held in
#        st.session_state. There is NO vector index, NO embedding model, and
#        NO external service — the list IS the "database".
#     4. RETRIEVE — at query time, score every chunk with a lightweight LEXICAL
#        (keyword-overlap) search: count how many query terms appear in each
#        chunk, then keep the top-k highest-scoring chunks (lexical_search).
#     5. AUGMENT  — inject those chunks into the LLM prompt as plain text. The
#        LLM's own long context window does the semantic "reading", so no
#        vectors are ever needed.
#
# WHY USE VECTORLESS RAG HERE?
# ----------------------------
#   * Zero infrastructure: no embedding model to download/call and no vector
#     database to set up, run, or maintain.
#   * Zero embedding cost: only the DeepSeek context window is consumed.
#   * Fully transparent: you can always inspect exactly which text is fed to
#     the LLM (the retrieved chunks are rendered in the UI), which makes the
#     results easy to audit — important for regulatory filings.
#   * Well suited to regulatory reports: filings are factual, term-heavy
#     documents (revenue, EBITDA, debt, risk, guidance...), and the exact
#     wording appears verbatim, so keyword overlap is an effective matcher.
#
# TRADE-OFFS (why traditional vector RAG exists):
#   * No semantic matching: keyword scoring misses paraphrased queries that a
#     true embedding similarity search would catch.
#   * Context-window bound: we cap the number of injected chunks (top_k) so
#     large filings never overflow the LLM context window.
#
# EXTRACTION TIMING
# -----------------
# Uploading files only snapshots their bytes (_store_uploaded_files) — the
# expensive PDF text extraction + chunking runs ONCE, when the user clicks
# "🚀 Analyze Stock" (run_analysis Step 4.5). Every stage prints progress to
# the terminal (the console running `streamlit run`) so the pipeline is
# observable end-to-end.

def extract_pdf_text(uploaded_file) -> list[dict]:
    """Extract the text of an uploaded PDF page-by-page using pypdf.

    Accepts either a Streamlit UploadedFile or a raw dict with
    {"name": str, "bytes": bytes} (the form stored in session state by
    _store_uploaded_files).

    Returns a list of page records:
        [{"file": "<name>", "page": 1, "text": "..."}, ...]
    Scanned / image-only PDFs yield no text and are reported by the caller.
    """
    # Normalise the input: Streamlit UploadedFile OR a {"name", "bytes"} dict.
    if isinstance(uploaded_file, dict):
        doc_name = uploaded_file.get("name", "document.pdf")
        pdf_bytes = uploaded_file.get("bytes", b"")
    else:
        doc_name = uploaded_file.name
        uploaded_file.seek(0)
        pdf_bytes = uploaded_file.getvalue()

    pages = []
    print(f"[RAG] Extracting text from '{doc_name}' ...")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append({
                    "file": doc_name,
                    "page": page_number,
                    "text": text,
                })
        print(f"[RAG]   -> '{doc_name}': {len(pages)} page(s) with text extracted.")
    except Exception as e:
        st.warning(f"⚠️ Could not read '{doc_name}': {e}")
        print(f"[RAG]   -> ERROR reading '{doc_name}': {e}")
    return pages


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> list[str]:
    """Split a page of text into overlapping chunks.

    Overlap preserves the context that spans chunk boundaries, so a sentence
    or a financial figure split across two chunks remains fully readable by
    the LLM. Chunks are cut at sentence boundaries when possible.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer a natural sentence boundary near the end of the chunk.
            boundary = text.rfind(". ", start + int(max_chars * 0.6), end)
            if boundary != -1:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap  # carry the tail over into the next chunk
    return [c for c in chunks if c]


def build_document_corpus(uploaded_files) -> tuple[list[dict], list[dict]]:
    """Turn the uploaded PDFs into the vectorless-RAG corpus.

    Step 2 of the vectorless pipeline above. Every page of every document is
    extracted and chunked, and each chunk is tagged with its source file and
    page number so the LLM can cite it and we can trace it in the UI. Progress
    is printed to the terminal so the extraction is observable.

    Returns:
        corpus:  [{"file": str, "page": int, "text": str}, ...]  (the "index")
        sources: [{"name", "pages", "chars", "chunks", "status"}, ...] (for UI)
    """
    corpus: list[dict] = []
    sources: list[dict] = []
    for uploaded_file in uploaded_files or []:
        doc_name = (
            uploaded_file.get("name", "document.pdf")
            if isinstance(uploaded_file, dict)
            else uploaded_file.name
        )
        pages = extract_pdf_text(uploaded_file)
        if not pages:
            print(f"[RAG]   -> '{doc_name}': no extractable text (scanned/image PDF?).")
            sources.append({
                "name": doc_name, "pages": 0, "chars": 0,
                "chunks": 0, "status": "⚠️ No extractable text",
            })
            continue
        file_chars = sum(len(p["text"]) for p in pages)
        file_chunks = 0
        for page in pages:
            for chunk in chunk_text(page["text"]):
                corpus.append({
                    "file": page["file"],
                    "page": page["page"],
                    "text": chunk,
                })
                file_chunks += 1
        print(f"[RAG]   -> '{doc_name}': {file_chars:,} chars -> {file_chunks} chunk(s).")
        sources.append({
            "name": doc_name,
            "pages": len(pages),
            "chars": file_chars,
            "chunks": file_chunks,
            "status": "✅",
        })
    return corpus, sources


def lexical_search(query: str, corpus: list[dict], top_k: int = 10) -> list[dict]:
    """Step 4 of the vectorless pipeline: keyword-overlap retrieval.

    This is the stand-in for embedding similarity search. Every chunk is
    scored by how many of the query's meaningful terms appear in it (case-
    insensitive substring match), then the top_k highest-scoring chunks are
    returned in descending order. Because we only do string matching there are
    no models, no vectors, and no database involved.
    """
    if not corpus:
        return []
    # Stop words that would add noise to the overlap score.
    STOPWORDS = {
        "the", "and", "for", "with", "this", "that", "from", "what", "are",
        "about", "stock", "company", "report", "filing", "analysis",
        "provide", "based", "information", "regulatory", "your", "you",
        "will", "has", "have", "its", "their", "also", "into", "such",
    }
    query_terms = [
        t for t in re.findall(r"[a-zA-Z]{3,}", query.lower())
        if t not in STOPWORDS
    ]
    if not query_terms:
        # No useful terms (empty/short query) — return the first top_k chunks.
        return corpus[:top_k]
    scored = []
    for chunk in corpus:
        text_lower = chunk["text"].lower()
        score = sum(1 for t in query_terms if t in text_lower)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def format_rag_context(rag_chunks: list[dict]) -> str:
    """Serialize the retrieved chunks into one text block for the LLM prompt.

    Each chunk is prefixed with its source file and page so the model can cite
    the filing it is drawing a fact from.
    """
    if not rag_chunks:
        return ""
    blocks = []
    for i, chunk in enumerate(rag_chunks, start=1):
        blocks.append(
            f"[Excerpt {i} | File: {chunk['file']} | Page {chunk['page']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Upload processing + sidebar summary (UI helpers that use st.*)
# ---------------------------------------------------------------------------

def _store_uploaded_files(uploaded_files):
    """Snapshot raw uploaded PDFs into session state WITHOUT extracting text.

    PDF parsing is deliberately deferred until the user clicks
    '🚀 Analyze Stock' (see run_analysis Step 4.5). Here we only persist the
    file bytes — Streamlit UploadedFile objects are recreated on every rerun
    and are not guaranteed to remain usable, so we snapshot each one as
    {"name", "size", "bytes"}.
    """
    if not uploaded_files:
        st.session_state.uploaded_files_raw = []
        st.session_state.uploaded_keys = []
        return
    current_keys = sorted((f.name, f.size) for f in uploaded_files)
    if st.session_state.get("uploaded_keys") == current_keys:
        return  # Same set of files already snapshotted.
    st.session_state.uploaded_files_raw = [
        {"name": f.name, "size": f.size, "bytes": f.getvalue()}
        for f in uploaded_files
    ]
    st.session_state.uploaded_keys = current_keys
    # Upload set changed -> invalidate previously built RAG artifacts. They
    # are rebuilt on the next 'Analyze Stock' click.
    st.session_state.document_corpus = []
    st.session_state.rag_sources = []
    st.session_state.rag_keys = []
    st.session_state.retrieved_chunks = []


def _render_uploaded_docs_summary():
    """Show uploaded filings and their extraction status in the sidebar.

    Two states are displayed:
      * Documents already processed → a per-file summary with page/char counts
        and a ✓/⚠️ status, plus a total line (files · pages · chunks).
      * Documents only queued (bytes snapshot, not yet extracted) → a list of
        filenames with byte sizes and an "extracted on Analyze" note.

    Also renders a "🗑️ Clear documents" button that wipes every RAG-related
    ``session_state`` key and reruns so the UI reflects the cleared state.
    """
    raw = st.session_state.get("uploaded_files_raw", [])
    sources = st.session_state.get("rag_sources", [])
    if not raw and not sources:
        st.caption("No documents uploaded yet.")
        return
    if sources:
        total_pages = sum(s["pages"] for s in sources)
        total_chunks = sum(s["chunks"] for s in sources)
        st.markdown(
            f"**{len(sources)} document(s)** · {total_pages} pages · "
            f"{total_chunks} chunks"
        )
        for s in sources:
            st.markdown(
                f"- {s['status']} `{s['name']}` — {s['pages']}p · {s['chars']:,} chars",
            )
    else:
        st.markdown(f"**{len(raw)} document(s) queued**")
        for f in raw:
            st.markdown(
                f"- ⏳ `{f['name']}` — {f['size']:,} bytes (extracted on Analyze)",
            )
    if st.button("🗑️ Clear documents", use_container_width=True):
        st.session_state.uploaded_files_raw = []
        st.session_state.uploaded_keys = []
        st.session_state.document_corpus = []
        st.session_state.rag_sources = []
        st.session_state.rag_keys = []
        st.session_state.retrieved_chunks = []
        st.rerun()


# =========================================================
# STREAMLIT UI
# =========================================================

def render_header():
    """Render the page header: app title + a live date/time stamp.

    The header is a two-column layout — the app title and tagline on the left,
    and the current date/time on the right (refreshed on every rerun). Purely
    presentational; the actual user inputs live in the sidebar.
    """
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
    """Render the sidebar with all user inputs and controls.

    Layout, top to bottom:
      1. Stock lookup — ticker + company name text inputs and the primary
         "🚀 Analyze Stock" button (disabled until both fields are filled).
      2. Multi-Persona Valuation — "🧠 Run Persona Agents" button.
      3. Regulatory filings — PDF uploader (vectorless RAG) with a summary of
         queued/processed documents and a clear button.
      4. Configuration — display currency selector and API-key status flags.
      5. Disclaimer footer.
      6. Save Analysis — "💾 Save Analysis" button that persists the current
         snapshot to the Chroma knowledge base (enabled after an analysis
         and/or a persona run).
      7. Export Report — a download button that serialises the current analysis
         into a self-contained HTML file (enabled only after an analysis).

    Returns:
        tuple: ``(ticker, company_name, currency, analyze_btn, persona_btn,
        save_btn)`` — the normalised inputs plus the clicked states of the
        analyze / persona / save buttons.
    """
    with st.sidebar:
        st.markdown("## 🔍 Stock Lookup")
        ticker = st.text_input(
            "Ticker Symbol (Yahoo Finance)",
            placeholder="e.g. INFY.NS, RELIANCE.BO, AAPL, MSFT",
            help=(
                "Type the COMPLETE Yahoo Finance ticker symbol, e.g. "
                "INFY.NS (NSE India), INFY.BO (BSE India), or AAPL / MSFT "
                "(US markets). The full ticker you type is used as-is to "
                "fetch market data."
            )
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
        st.markdown("#### 🧠 Multi-Persona Valuation (crewAI)")
        st.caption(
            "Runs four investor-persona agents (Buffett, Munger, Jhunjhunwala, "
            "Damodaran) that each research & value the stock autonomously, then "
            "a lead analyst reconciles them."
        )
        persona_btn = st.button(
            "🧠 Run Persona Agents",
            use_container_width=True,
            disabled=not ticker,
            help=(
                "Requires a ticker (company name optional). Makes live DeepSeek "
                "calls via crewAI and may take 1–3 minutes."
            ),
        )

        st.markdown("---")
        st.markdown("#### �📄 Regulatory Filings (Vectorless RAG)")
        uploaded_files = st.file_uploader(
            "Upload the regulatory filing report here",
            type=["pdf"],
            accept_multiple_files=True,
            help=(
                "Upload one or more regulatory filing reports (PDF) — e.g. annual "
                "reports, quarterly results, or prospectuses. The text is extracted "
                "from each PDF and used as in-context evidence for the AI analysis "
                "via vectorless RAG (no embeddings or vector database required)."
            ),
        )
        _store_uploaded_files(uploaded_files)
        _render_uploaded_docs_summary()

        st.markdown("---")
        st.markdown("#### ⚙️ Configuration")
        currency = st.selectbox(
            "Display Currency",
            options=list(CURRENCY_SYMBOLS.keys()),
            index=list(CURRENCY_SYMBOLS.keys()).index("INR"),
            format_func=lambda c: f"{c} ({CURRENCY_SYMBOLS[c]})",
            key="display_currency",
            help="Prices will be converted to the selected currency for display"
        )
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

        # ── Save Analysis (Chroma Cloud knowledge base) ──
        # Persists the current snapshot — the four fundamental commentary
        # sections, the Overall Technical Summary, and (when run) the persona
        # lead-analyst consensus — into Chroma. Happens only on click; enabled
        # whenever there is a completed analysis and/or a persona result.
        st.markdown("---")
        st.markdown("#### 💾 Save Analysis")
        st.caption("Store this report in the Chroma knowledge base.")
        can_save = bool(
            st.session_state.get("analysis_complete")
            or st.session_state.get("persona_result")
        )
        save_btn = st.button(
            "💾 Save Analysis",
            use_container_width=True,
            disabled=not can_save,
            help=(
                "Persist the current analysis snapshot (fundamental commentary, "
                "technical summary, persona consensus) to the Chroma knowledge "
                "base. Enabled after an analysis and/or a persona run."
            ),
        )

        # ── Export Report (very bottom of the sidebar) ──
        # Once an analysis is complete, this converts the current report into
        # a self-contained HTML file and lets the user download it. The HTML is
        # generated once and cached in session_state so it is not rebuilt on
        # every rerun; the cache is cleared when a new analysis starts.
        st.markdown("---")
        st.markdown("#### 📤 Export Report")
        if st.session_state.get("analysis_complete"):
            if "export_html" not in st.session_state:
                (st.session_state.export_html,
                 st.session_state.export_filename) = build_export_html()
            st.download_button(
                "📥 Export Report (HTML)",
                data=st.session_state.export_html,
                file_name=st.session_state.export_filename,
                mime="text/html",
                use_container_width=True,
                help=(
                    "Convert the current analysis report into a self-contained "
                    "HTML file and download it."
                ),
            )
            st.caption(st.session_state.export_filename)
        else:
            st.download_button(
                "📥 Export Report (HTML)",
                data="",
                file_name="report.html",
                mime="text/html",
                use_container_width=True,
                disabled=True,
                help="Run an analysis first to enable the export.",
            )

    return ticker, company_name, currency, analyze_btn, persona_btn, save_btn


def render_metrics(stock_data: dict, display_currency: str = "INR"):
    """Render the key metrics strip (5 metric cards in one row).

    Shows Current Price, Open, High, Low and Volume. Prices are converted from
    the stock's native currency into ``display_currency`` via ``format_price``;
    the volume uses the Indian numbering system (``format_volume``) and the
    high/low spread percentage is shown as a subtitle on the volume card.

    Args:
        stock_data (dict):       Output of ``get_stock_price()``.
        display_currency (str):  ISO code for price display (default ``"INR"``).
    """
    native_currency = stock_data.get("currency", "USD")
    spread_pct = round((stock_data["high"] - stock_data["low"]) / stock_data["low"] * 100, 1)
    cols = st.columns(5)
    metrics = [
        ("💰 Current Price", format_price(stock_data["current_price"], native_currency, display_currency), f"{stock_data['exchange']}"),
        ("📂 Open", format_price(stock_data["open"], native_currency, display_currency), "Today's open"),
        ("📈 High", format_price(stock_data["high"], native_currency, display_currency), "Day high"),
        ("📉 Low", format_price(stock_data["low"], native_currency, display_currency), "Day low"),
        ("📊 Volume", format_volume(stock_data['volume']), f"Range: {spread_pct}%"),
    ]
    for col, (label, value, sub) in zip(cols, metrics):
        with col:
            st.metric(label=label, value=value, help=sub)


def render_dcf_valuation(dcf_data: dict, stock_data: dict, display_currency: str = "INR"):
    """Render separate fair-value cards for each DCF scenario.

    Displays four color-coded cards (Base / Bull / Bear / Average) each with
    the DCF value per share and a delta vs. the current market price, followed
    by a caption summarising the scenario/model counts. Degrades gracefully to
    an info message when DCF data is missing or errored.

    Args:
        dcf_data (dict):         Output of ``get_dcf_valuation()``.
        stock_data (dict):       Output of ``get_stock_price()`` (for the
                                 market price and native currency).
        display_currency (str):  ISO code for price display.
    """
    if not dcf_data or "error" in dcf_data:
        st.info("💹 DCF valuation data not available.")
        return

    native = stock_data.get("currency", "USD")
    market_price = stock_data.get("current_price")

    st.markdown("---")
    st.markdown("## 💹 DCF Valuation")
    st.caption("Fair value per share from multi-model DCF analysis across scenarios")

    def _fmt(value) -> str:
        """Format a DCF value in the display currency for a metric card.

        Coerces the value to float and delegates to ``format_price`` so the
        display currency + symbol are applied consistently. A ``None`` value
        renders as an em dash (—) instead of crashing.

        Args:
            value: The raw DCF value (float or None).

        Returns:
            str: The formatted price string, or ``"—"`` for ``None``.
        """
        if value is None:
            return "—"
        return format_price(float(value), native, display_currency)

    def _delta_pct(value):
        """Return the percentage difference between a DCF value and the market price.

        Used as the ``delta`` argument of ``st.metric`` to show whether each
        scenario is trading above or below the current market price, e.g.
        ``+12.4%``. Returns ``None`` when either input is missing so the metric
        renders without a delta.

        Args:
            value: The DCF fair value per share (float or None).

        Returns:
            float | None: The percentage difference, or ``None`` if incalculable.
        """
        if value is None or not market_price:
            return None
        return (float(value) - market_price) / market_price * 100

    cols = st.columns(4)
    cards = [
        ("⚖️ Base", dcf_data.get("base_val"), "#1a3a6b"),
        ("🚀 Bull", dcf_data.get("bull_val"), "#1a6b3a"),
        ("🐻 Bear", dcf_data.get("bear_val"), "#b01a1a"),
        ("📊 Average", dcf_data.get("average_all"), "#92600a"),
    ]
    for col, (label, value, color) in zip(cols, cards):
        with col:
            with st.container(border=True):
                st.markdown(
                    f"<p style='margin:0; font-size:0.8rem; font-weight:600; color:{color};'>{label}</p>",
                    unsafe_allow_html=True
                )
                pct = _delta_pct(value)
                if pct is not None:
                    st.metric(
                        "Value / Share",
                        _fmt(value),
                        delta=f"{pct:+.1f}% vs market",
                        delta_color="normal",
                    )
                else:
                    st.metric("Value / Share", _fmt(value))

    n_scenarios = len(dcf_data.get("scenario_results", []))
    n_estimates = len(dcf_data.get("all_values_per_share", []))
    st.caption(
        f"Market price: {_fmt(market_price)} · "
        f"{n_scenarios} scenarios · {n_estimates} model estimates"
    )


def render_sentiment(sentiment_data: dict):
    """Render the market-sentiment card.

    Shows a color-coded badge with the sentiment label (and matching emoji), a
    progress bar whose width maps the polarity score (−1..+1) to a percentage,
    and the score + article count. "Unknown"/"Error" states skip the bar and
    surface their reason/error via ``st.info`` / ``st.error`` instead.

    Args:
        sentiment_data (dict): Output of ``analyze_news_sentiment()``.
    """
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
    """Render the latest news articles as bordered cards.

    Each article shows its title, description (if any), a metadata line with
    source and published-at timestamp (parsed from ISO 8601, tolerant of
    errors), and a "Read more →" link. Shows a caption when there is no news.

    Args:
        news (list[dict]): Output of ``get_stock_news()``.
    """
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


def render_document_context(rag_chunks: list[dict]):
    """Render the vectorless-RAG document evidence used in the analysis.

    Displays a "Regulatory Filing Context" section listing every retrieved
    chunk inside an expander, each labeled with its source file and page. The
    section is only rendered when at least one document was processed; an info
    note appears when no chunks matched the query.

    Args:
        rag_chunks (list[dict]): Retrieved chunks (from ``lexical_search``),
                                 each with ``file``, ``page`` and ``text``.
    """
    sources = st.session_state.get("rag_sources", [])
    if not sources:
        return
    st.markdown("---")
    st.markdown("## 📄 Regulatory Filing Context")
    st.caption(
        "Chunks below were extracted from your uploaded PDFs and retrieved by "
        "keyword (lexical) scoring — vectorless RAG, so no embeddings or vector "
        "database were used. They were injected into the AI analysis prompt."
    )
    if not rag_chunks:
        st.info("No relevant chunks matched the query from the uploaded documents.")
        return
    for i, chunk in enumerate(rag_chunks, start=1):
        with st.expander(f"Chunk {i} — {chunk['file']} (Page {chunk['page']})"):
            st.markdown(chunk["text"])


def render_analysis(parsed_sections: list):
    """Render the AI-powered fundamental analysis sections.

    Lays the standard sections out in a two-column grid (alternating left /
    right) and hands the Investment Recommendation off to
    ``_render_recommendation`` for full-width, emphasized rendering. Warns if
    no sections were parsed.

    Args:
        parsed_sections (list[dict]): Output of ``parse_analysis()``.
    """
    st.markdown("---")
    st.markdown("## 🤖 AI-Powered Investment Analysis")
    st.caption(f"Generated by Senior Wall Street AI · {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if not parsed_sections:
        st.warning("No analysis sections were parsed.")
        return

    # Two-column grid: the three standard sections (Financial Outlook, Key
    # Risks, Growth Potential) alternate between the left and right column,
    # while the Investment Recommendation is pulled out below to span the full
    # width with special emphasis (accent border + verdict badge).
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
                        # Escape the text, then bold-colour its bull/bear wording.
                        st.markdown(
                            sra.highlight_bull_bear_escaped(para),
                            unsafe_allow_html=True,
                        )


def _render_recommendation(section: dict):
    """Render the Investment Recommendation with emphasis styling.

    The recommendation is displayed in a full-width bordered container with a
    color-coded verdict badge (BUY ✅ / HOLD ⏸️ / SELL ❌) beside the section
    title. Paragraph body text is rendered with ``<mark>`` tags converted to
    bold (**) since raw HTML is not allowed in Streamlit markdown.

    Args:
        section (dict): The recommendation section dict from ``parse_analysis()``.
    """
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
            # Replace <mark> tags with Streamlit markdown highlights, then
            # bold-colour the bull/bear wording.
            cleaned = re.sub(r'</?mark>', '**', para)
            st.markdown(sra.highlight_bull_bear(cleaned), unsafe_allow_html=True)


def render_technical_analysis(tech_parsed: dict):
    """Render the quantitative technical analysis (RAG-grounded).

    Displays the technical verdict badge (STRONG BUY / BUY / SELL / STRONG
    SELL / NEUTRAL), then each indicator group (price & volume moving
    averages, momentum, volatility, volume flow) as bordered cards with the
    indicator's live value and the LLM's interpretation, finishing with the
    overall technical summary. Falls back to a warning when no analysis is
    available.

    Args:
        tech_parsed (dict): Output of ``sra.parse_technical_analysis()`` —
                            keys ``groups``, ``verdict``, ``verdict_css``,
                            ``summary`` and ``error``.
    """
    st.markdown("---")
    st.markdown("## 🧮 Technical Analysis")
    st.caption(
        "Generated by Quantitative Analyst AI · RAG context retrieved from the "
        "5-year technical-indicator database"
    )

    if not tech_parsed or not tech_parsed.get("groups"):
        error_msg = (
            (tech_parsed or {}).get("error")
            or "No technical analysis was generated for this report."
        )
        st.warning(error_msg)
        return

    verdict = tech_parsed.get("verdict", "NEUTRAL")
    color, emoji = get_tech_verdict_style(verdict)
    st.markdown(
        f"<div style='display: flex; align-items: center; gap: 0.4rem; "
        f"padding: 0.5rem 1rem; border-radius: 999px; "
        f"background-color: color-mix(in srgb, var(--{color}) 15%, transparent); "
        f"color: var(--{color}); font-weight: 600; width: fit-content; "
        f"margin-bottom: 1rem;'>{emoji} {verdict} "
        f"<span style='opacity: 0.6; font-weight: 400;'>· Technical Signal</span></div>",
        unsafe_allow_html=True
    )

    for group in tech_parsed.get("groups", []):
        with st.container(border=True):
            st.markdown(f"### {group.get('emoji', '📌')} {group.get('title', '')}")
            indicators = group.get("indicators", [])
            if not indicators:
                st.caption("No indicator details parsed for this group.")
                continue
            cols = st.columns(2)
            for i, ind in enumerate(indicators):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"**{ind.get('name', '')}**")
                        if ind.get("value"):
                            st.markdown(
                                f"<p style='font-size: 1.15rem; font-weight: 600; margin: 0;'>"
                                f"{ind['value']}</p>",
                                unsafe_allow_html=True
                            )
                        st.markdown(
                            f"<p style='font-size: 0.85rem; color: var(--text-color-secondary); "
                            f"line-height: 1.5;'>"
                            f"{sra.highlight_bull_bear_escaped(ind.get('analysis', ''))}</p>",
                            unsafe_allow_html=True
                        )

    if tech_parsed.get("summary"):
        with st.container(border=True):
            st.markdown("#### 🧮 Overall Technical Summary")
            st.markdown(
                sra.highlight_bull_bear_escaped(tech_parsed["summary"]),
                unsafe_allow_html=True,
            )


def render_overall_summary(overall: dict):
    """Render the merged fundamental + technical Overall Summary.

    Shows three side-by-side badges — Fundamental, Technical and Overall —
    each color-coded by verdict, followed by the explanatory text produced by
    ``sra.build_overall_summary()``. No-op (returns early) when ``overall`` is
    empty/falsy.

    Args:
        overall (dict): Output of ``sra.build_overall_summary()``.
    """
    if not overall:
        return

    st.markdown("---")
    st.markdown("## 🧭 Overall Summary")
    st.caption("Merged view of the fundamental verdict and the technical verdict")

    fund_styles = {
        "STRONG BUY":  ("green", "🚀"),
        "BUY":         ("green", "✅"),
        "HOLD":        ("orange", "⏸️"),
        "SELL":        ("red", "❌"),
        "STRONG SELL": ("red", "🔻"),
    }

    def _badge(label: str, value: str, kind: str) -> str:
        """Build an HTML badge (label + colored pill) for a verdict.

        Args:
            label (str): The badge caption, e.g. "Fundamental".
            value (str): The verdict, e.g. "BUY".
            kind (str):  ``"tech"`` uses the 4-flag technical style, anything
                         else uses the fundamental BUY/HOLD/SELL style.

        Returns:
            str: A self-contained HTML snippet for the badge.
        """
        if kind == "tech":
            color, emoji = get_tech_verdict_style(value)
        else:
            color, emoji = fund_styles.get(value, ("gray", "ℹ️"))
        return (
            f"<div style='flex: 1; min-width: 150px;'>"
            f"<p style='font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; "
            f"color: var(--text-color-secondary); margin: 0 0 0.3rem;'>{label}</p>"
            f"<div style='display: inline-flex; align-items: center; gap: 0.4rem; "
            f"padding: 0.35rem 0.8rem; border-radius: 999px; "
            f"background-color: color-mix(in srgb, var(--{color}) 15%, transparent); "
            f"color: var(--{color}); font-weight: 600; font-size: 0.8rem;'>"
            f"{emoji} {value}</div></div>"
        )

    st.markdown(
        f"<div style='display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.75rem;'>"
        f"{_badge('Fundamental', overall.get('fundamental_verdict', 'HOLD'), 'fund')}"
        f"{_badge('Technical', overall.get('technical_verdict', 'NEUTRAL'), 'tech')}"
        f"{_badge('Overall', overall.get('overall_verdict', 'HOLD'), 'fund')}"
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        sra.highlight_bull_bear_escaped(overall.get("text", "")),
        unsafe_allow_html=True,
    )


def build_export_html() -> tuple[str, str]:
    """Convert the current analysis report into a self-contained HTML file.

    Reuses the CLI agent's ``sra.generate_html_report`` (the same engine that
    produces ``{TICKER}_REPORT_{YYYY_MM_DD}.html``) to render the fundamental,
    technical and overall-summary report, then returns its content and filename
    so the sidebar's export button can offer it as a download.

    The CLI template hardcodes the ``₹`` symbol for every price, so the stock's
    native currency symbol is substituted afterwards (USD ``$``, EUR ``€``, …)
    to keep global-stock exports correct.

    Returns:
        tuple[str, str]: ``(html_string, suggested_filename)``.
    """
    stock_data = st.session_state.stock_data
    sentiment_data = st.session_state.sentiment_data
    news = st.session_state.news
    analysis = st.session_state.analysis or ""
    tech_analysis = st.session_state.get("tech_analysis")
    tech_df = st.session_state.get("tech_df")
    ticker = st.session_state.get("ticker", "")

    try:
        # Render the self-contained report (this also writes it to disk and
        # returns the filename, exactly like the CLI agent), then read the
        # bytes back so the download button can serve them directly.
        report_file = sra.generate_html_report(
            ticker,
            stock_data,
            sentiment_data,
            news,
            analysis,
            tech_analysis=tech_analysis,
            tech_df=tech_df,
        )
        with open(report_file, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception as e:
        # Never let a rendering failure crash the sidebar — surface it and
        # fall back to a minimal page so the download is still valid.
        st.warning(f"⚠️ Could not generate the export: {e}")
        return (
            f"<html><body><h1>Export failed</h1><p>{e}</p></body></html>",
            "report_export_failed.html",
        )

    # Swap the hardcoded ₹ for the stock's native currency symbol so exports
    # of global (USD/EUR/…) stocks display the correct currency.
    native = stock_data.get("currency", "INR")
    symbol = CURRENCY_SYMBOLS.get(native, native)
    if symbol != "₹":
        html = html.replace("₹", symbol)

    return html, report_file


def run_analysis(ticker: str, company_name: str):
    """Execute the full multi-stage analysis pipeline.

    Runs synchronously with a progress bar + status text, storing every result
    into ``session_state`` as it goes and finishing with a ``st.rerun()`` so
    the results branch of ``main()`` renders. The pipeline:

      1. Stock price (``get_stock_price``)          — OHLCV snapshot.
      2. News (``get_stock_news``)                  — latest headlines.
      3. Sentiment (``analyze_news_sentiment``)     — TextBlob polarity.
      4. DCF valuation (``get_dcf_valuation``)      — intrinsic-value models.
      4.5 Vectorless RAG corpus build               — PDF extraction + chunking
          (only when the uploaded file set changed).
      4.6 Lexical retrieval + prompt context        — top-k chunks.
      4.7 Technical indicators (``TECH_ANALYSIS``)  — 5-year DataFrame.
      5. AI analysis                                — fundamental + technical
          agents run IN PARALLEL in two threads.
      5b Parse + merge verdicts                     — Overall Summary.

    Failures are isolated: a stock-data error aborts early with a message,
    technical-indicator failure degrades to ``tech_df = None`` (skipping the
    technical agent), and either agent's LLM call may fail without crashing
    the whole run.

    Args:
        ticker (str):        The Yahoo Finance ticker to analyse.
        company_name (str):  Full company name for news/sentiment search.
    """
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

        # step 4: Discounted Cash Flow (DCF) Valuation
        status_text.info("💹 Fetching DCF valuation data...")
        progress_bar.progress(65, text="Fetching DCF valuation data...")
        dcf_data = get_dcf_valuation.invoke(ticker)
        st.session_state.dcf_data = dcf_data

        # Step 4.5: Vectorless RAG — extract text from the uploaded PDFs NOW.
        # Extraction is triggered only by this Analyze click (upload merely
        # snapshots bytes); every stage prints progress to the terminal so the
        # pipeline is observable in the `streamlit run` console.
        uploaded_files_raw = st.session_state.get("uploaded_files_raw", [])
        corpus = st.session_state.get("document_corpus", [])
        if st.session_state.get("rag_keys") != st.session_state.get("uploaded_keys", []):
            status_text.info("📄 Extracting text from uploaded PDFs...")
            progress_bar.progress(68, text="Extracting text from uploaded PDFs...")
            print("[RAG] ===== Building vectorless RAG corpus =====")
            corpus, sources = build_document_corpus(uploaded_files_raw)
            st.session_state.document_corpus = corpus
            st.session_state.rag_sources = sources
            st.session_state.rag_keys = st.session_state.get("uploaded_keys", [])
            print(f"[RAG] Corpus ready: {len(corpus)} chunks across {len(sources)} document(s).")

        # Step 4.6: Retrieve the most relevant chunks via keyword scoring
        # (no embeddings / vector DB) and turn them into prompt context.
        rag_query = (
            f"{ticker} {company_name} revenue profit debt growth risks "
            "guidance outlook valuation"
        )
        rag_chunks = lexical_search(rag_query, corpus, top_k=10)
        rag_context = format_rag_context(rag_chunks)
        st.session_state.retrieved_chunks = rag_chunks
        print(f"[RAG] Retrieved {len(rag_chunks)} chunk(s) -> "
              f"{len(rag_context):,} chars injected into the analysis prompt.")

        # Step 4.7: Technical Indicators
        # Download 5 years of daily OHLCV data and compute the full technical
        # indicator DataFrame (moving averages, RSI, MACD, Bollinger,
        # Stochastic, OBV). This DataFrame is used as the RAG source for the
        # quantitative technical agent.
        status_text.info("🧮 Computing technical indicators...")
        progress_bar.progress(72, text="Computing technical indicators...")
        try:
            tech_df = TECH_ANALYSIS.compute_technical_indicators(ticker)
        except Exception as e:
            print(f"[TECH] Could not compute technical indicators: {e}")
            tech_df = None
        st.session_state.tech_df = tech_df

        # Step 5: AI Analysis — the fundamental analyst and the quantitative
        # technical analyst run IN PARALLEL (two threads). The technical agent
        # uses the indicator DataFrame as a RAG knowledge base.
        status_text.info("🤖 Generating AI investment analysis (fundamental + technical in parallel)...")
        progress_bar.progress(75, text="Generating AI investment analysis...")
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_fund = executor.submit(
                generate_analysis,
                ticker,
                stock_data,
                sentiment_data,
                news,
                dcf_data,
                st.session_state.get("display_currency", "INR"),
                rag_context=rag_context,
            )
            # Only launch the technical agent if we have indicator data.
            future_tech = (
                executor.submit(sra.generate_technical_analysis, ticker, tech_df)
                if tech_df is not None else None
            )

            try:
                analysis = future_fund.result()
            except Exception as e:
                print(f"[FUND] Fundamental analysis failed: {e}")
                analysis = ""

            try:
                tech_analysis = future_tech.result() if future_tech else None
            except Exception as e:
                print(f"[TECH] Technical analysis failed: {e}")
                tech_analysis = None

        st.session_state.analysis = analysis
        st.session_state.tech_analysis = tech_analysis

        # Step 5b: Parse the fundamental analysis + the technical analysis,
        # then merge the two verdicts into an Overall Summary.
        parsed_sections = parse_analysis(analysis)
        st.session_state.parsed_sections = parsed_sections
        st.session_state.ticker = ticker
        st.session_state.company_name = company_name

        fund_verdict = "HOLD"
        for sec in parsed_sections:
            if sec.get("is_recommendation") and sec.get("verdict"):
                fund_verdict = sec["verdict"]
                break

        tech_parsed = sra.parse_technical_analysis(tech_analysis or "", tech_df)
        st.session_state.tech_parsed = tech_parsed
        st.session_state.overall_summary = sra.build_overall_summary(
            fund_verdict, tech_parsed.get("verdict", "NEUTRAL")
        )

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
# SAVE-TO-CHROMA HELPERS
# =========================================================

# Canonical fundamental-report sections whose commentary is persisted.
FUNDAMENTAL_TITLES = (
    "Financial Outlook",
    "Key Risks",
    "Growth Potential",
    "Investment Recommendation",
)


def _fundamental_commentary(parsed_sections: list) -> dict:
    """Extract {canonical title: joined commentary} from the parsed sections.

    Mirrors ``parse_analysis``'s substring title matching so all four known
    fundamental sections (Financial Outlook, Key Risks, Growth Potential,
    Investment Recommendation) are captured under their canonical names.
    """
    commentary: dict[str, str] = {}
    for sec in parsed_sections or []:
        title = sec.get("title", "")
        for key in FUNDAMENTAL_TITLES:
            if key.lower() in title.lower():
                paragraphs = sec.get("paragraphs") or []
                commentary[key] = "\n".join(str(p) for p in paragraphs).strip()
                break
    return commentary


def _build_save_record():
    """Assemble the Chroma snapshot dict from the current ``session_state``.

    Returns:
        tuple[dict | None, list[str]]: ``(record, notes)``. ``record`` is
        ``None`` when there is nothing to save; ``notes`` holds non-fatal
        warnings (e.g. missing sections, mismatched persona ticker).
    """
    notes: list[str] = []
    analysis_complete = bool(st.session_state.get("analysis_complete"))
    persona_result = st.session_state.get("persona_result") or {}

    # Subject of the snapshot: prefer the completed analysis; otherwise fall
    # back to the persona result (a persona-only run).
    if analysis_complete:
        ticker = st.session_state.get("ticker") or ""
        company_name = st.session_state.get("company_name") or ""
    else:
        ticker = persona_result.get("ticker") or ""
        company_name = persona_result.get("company_name") or ""

    if not (ticker or company_name):
        return None, ["No analysis or persona result is available to save."]

    record: dict = {
        "ticker": ticker,
        "company_name": company_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "has_analysis": analysis_complete,
    }

    # 1) Fundamental commentary + BUY/HOLD/SELL recommendation verdict.
    sections: dict[str, str] = {}
    rec_verdict = None
    parsed_sections = st.session_state.get("parsed_sections")
    if analysis_complete and parsed_sections:
        sections = _fundamental_commentary(parsed_sections)
        for sec in parsed_sections:
            if sec.get("is_recommendation") and sec.get("verdict"):
                rec_verdict = sec["verdict"]
                break
        if not rec_verdict:
            rec_verdict = (st.session_state.get("overall_summary") or {}).get(
                "fundamental_verdict"
            )
        missing = [t for t in FUNDAMENTAL_TITLES if not sections.get(t)]
        if missing:
            notes.append("Missing fundamental section(s): " + ", ".join(missing))
    record["fundamental_sections"] = sections
    record["recommendation_verdict"] = rec_verdict

    # 2) Overall Technical Summary (quant analyst verdict + summary text).
    tech = st.session_state.get("tech_parsed") or {}
    record["technical_verdict"] = tech.get("verdict")
    record["technical_summary"] = (tech.get("summary") or "").strip() or None

    # 3) Persona lead-analyst consensus (only when it belongs to this ticker).
    persona_block = None
    if persona_result:
        if analysis_complete and persona_result.get("ticker") != ticker:
            notes.append(
                "Persona result ticker differs from the analysed ticker — "
                "persona data was skipped."
            )
        else:
            agg = persona_result.get("aggregate") or {}
            persona_block = {
                "market_price": persona_result.get("market_price"),
                "currency": persona_result.get("currency"),
                "exchange": persona_result.get("exchange"),
                "generated_at": persona_result.get("generated_at"),
                "min_fair_value": agg.get("min_fair_value"),
                "avg_fair_value": agg.get("avg_fair_value"),
                "max_fair_value": agg.get("max_fair_value"),
                "blended_stance": agg.get("blended_stance"),
                "participating": agg.get("participating"),
                "commentary": agg.get("note"),
            }
    record["persona"] = persona_block
    record["has_persona"] = bool(persona_block)
    return record, notes


def _save_current_analysis():
    """Persist the current snapshot to Chroma Cloud (lazy import)."""
    record, notes = _build_save_record()
    if record is None:
        st.error("Nothing to save yet. Run an analysis or the persona agents first.")
        return
    try:
        import CHROMA_STORE as cs  # lazy so app boot is unaffected
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Could not load CHROMA_STORE: {e}")
        return
    try:
        doc_id = cs.save_analysis_record(record)
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ Failed to save the analysis to Chroma: {e}")
        return
    st.success(f"✅ Analysis saved to Chroma knowledge base (id: `{doc_id}`)")
    if notes:
        st.warning("⚠️ " + " ".join(notes))


# =========================================================
# MULTI-PERSONA AGENTIC VALUATION (crewAI PILOT)
# =========================================================


def run_persona_valuation(ticker: str, company_name: str):
    """Run the crewAI multi-persona valuation for a ticker (sidebar trigger).

    Lazily imports ``PERSONA_AGENTS_CREW`` so the standard app flow is never
    slowed by crewAI and degrades gracefully when it is not installed. Runs the
    four persona agents + lead analyst synchronously (live DeepSeek calls), then
    stores the serialised result in ``session_state.persona_result`` and reruns.

    Args:
        ticker (str):        Yahoo Finance ticker (e.g. ``"INFY.NS"``).
        company_name (str):  Full company name for the news tool (may be empty).
    """
    try:
        import PERSONA_AGENTS_CREW as pac
    except Exception as e:  # noqa: BLE001
        st.session_state.persona_running = False
        st.error(
            "❌ crewAI is not available. Install it in the project venv with "
            "`pip install \"crewai[litellm]\"` "
            f"({e})"
        )
        return

    with st.spinner(
        "🧠 Running 4 persona agents + lead analyst (live DeepSeek calls, "
        "may take 1–3 min)..."
    ):
        try:
            res = pac.run_persona_crew(ticker, company_name or ticker)
        except Exception as e:  # noqa: BLE001
            st.session_state.persona_running = False
            st.error(f"❌ Multi-persona valuation failed: {e}")
            return

    st.session_state.persona_result = res.model_dump()
    st.session_state.persona_running = False
    st.rerun()


def _stance_badge(stance: str) -> tuple:
    """Return (label, colour) for a persona / consensus stance badge."""
    return {
        "Undervalued":   ("Undervalued",  "#1a7f37"),
        "Fairly valued": ("Fairly valued", "#9a6700"),
        "Overvalued":    ("Overvalued",   "#cf222e"),
        "No opinion":    ("No opinion",   "#59636e"),
    }.get(stance or "", (stance or "No opinion", "#59636e"))


def render_persona_valuation(result: dict, display_currency: str = "INR"):
    """Render the crewAI multi-persona valuation result section.

    Displays: a market-price header, one card per persona (fair value, delta vs
    market, stance, conviction), a lead-analyst consensus range, and a
    side-by-side comparison against the deterministic ``PERSONA_BASED_VALUATION``
    range. All prices are converted from the stock's native currency into
    ``display_currency`` via ``format_price``.

    Args:
        result (dict):        Serialised ``PersonaCrewResult`` (model_dump).
        display_currency (str): ISO code to convert prices into for display.
    """
    native = result.get("currency", "USD")
    market_price = result.get("market_price")
    personas = result.get("personas") or []
    aggregate = result.get("aggregate") or {}
    deterministic = result.get("deterministic_range") or {}

    def _fmt(value) -> str:
        """Format a native-currency value in the display currency (— when None)."""
        if value is None:
            return "—"
        return format_price(float(value), native, display_currency)

    def _delta_pct(value):
        """Percentage of a fair value vs the market price (None when unusable)."""
        if value is None or not market_price:
            return None
        return (float(value) - market_price) / market_price * 100

    st.markdown("---")
    clean = strip_exchange_suffix(result.get("ticker", ""))
    st.markdown(
        f"<h2 style='margin-bottom:0;'>👤 Multi-Persona Agentic Valuation (crewAI)</h2>"
        f"<p style='color: var(--text-color-secondary); font-size: 0.9rem;'>"
        f"{clean} · {result.get('exchange', '')} · Market {_fmt(market_price)} "
        f"(native {native}) · Generated {result.get('generated_at', '')}</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Four investor-persona agents independently research & value the stock via "
        "tools (price / fundamentals / news) on DeepSeek; a lead analyst then "
        "reconciles their fair values."
    )

    # ── Persona cards ──
    if not personas:
        st.info("No persona results to show.")
    else:
        cols = st.columns(len(personas))
        for col, p in zip(cols, personas):
            with col:
                with st.container(border=True):
                    fv = p.get("fair_value_per_share")
                    stance, color = _stance_badge(p.get("stance"))
                    # Persona portrait (falls back to the emoji when the image
                    # file is unavailable, e.g. for results saved before the
                    # portraits were wired in).
                    if p.get("image"):
                        st.image(p["image"], width=72)
                        heading = p.get("persona", "")
                    else:
                        heading = f"{p.get('emoji', '🧑')} {p.get('persona', '')}"
                    st.markdown(
                        f"<p style='margin:0; font-size:1rem; font-weight:700;'>"
                        f"{heading}</p>",
                        unsafe_allow_html=True,
                    )
                    pct = _delta_pct(fv)
                    if pct is not None:
                        st.metric(
                            "Fair value / share",
                            _fmt(fv),
                            delta=f"{pct:+.1f}% vs market",
                            delta_color="normal",
                        )
                    else:
                        st.metric("Fair value / share", _fmt(fv))
                    st.markdown(
                        f"<span style='background:{color}; color:#fff; "
                        f"padding:2px 8px; border-radius:10px; font-size:0.75rem;'>"
                        f"{stance}</span>",
                        unsafe_allow_html=True,
                    )

        # ── Per-persona detail expanders ──
        # Label is plain text (no emoji/icon) — Streamlit expander headers
        # cannot render the persona portraits.
        for p in personas:
            fv = p.get("fair_value_per_share")
            stance, _c = _stance_badge(p.get("stance"))
            conviction = float(p.get("conviction") or 0.0)
            with st.expander(f"{p.get('persona', '')} — {stance}"):
                st.progress(
                    min(1.0, max(0.0, conviction)),
                    text=f"Conviction {conviction:.0%}",
                )
                pct = _delta_pct(fv)
                src_cur = p.get("currency") or native
                delta_txt = f" ({pct:+.1f}% vs market)" if pct is not None else ""
                st.markdown(
                    f"**Fair value:** {_fmt(fv)} (in {src_cur}{delta_txt})"
                )
                st.markdown(f"**Thesis:** {p.get('one_line_thesis') or '—'}")
                st.markdown(f"**Rationale:** {p.get('rationale') or '—'}")
                sources = p.get("sources") or []
                if sources:
                    st.caption("**Sources:** " + "; ".join(str(s) for s in sources))

    # ── Lead-analyst consensus ──
    st.markdown("### ⚖️ Lead-Analyst Consensus")
    a1, a2, a3 = st.columns(3)
    a1.metric("Min fair value", _fmt(aggregate.get("min_fair_value")))
    a2.metric("Avg fair value", _fmt(aggregate.get("avg_fair_value")))
    a3.metric("Max fair value", _fmt(aggregate.get("max_fair_value")))
    stance, color = _stance_badge(aggregate.get("blended_stance"))
    st.markdown(
        f"<span style='background:{color}; color:#fff; padding:2px 10px; "
        f"border-radius:10px; font-size:0.85rem;'>Blended stance: {stance}</span> "
        f"<span style='color:var(--text-color-secondary);'>"
        f"({aggregate.get('participating', 0)} of {len(personas)} personas priced)</span>",
        unsafe_allow_html=True,
    )
    if aggregate.get("note"):
        st.markdown(aggregate["note"])

    # ── Compare with the deterministic model ──
    # if deterministic and "error" not in deterministic and deterministic.get("min") is not None:
    #     st.markdown("### 📊 vs Deterministic Model")
    #     d1, d2, d3 = st.columns(3)
    #     d1.metric("Det. min", _fmt(deterministic.get("min")))
    #     d2.metric("Det. midpoint", _fmt(deterministic.get("mid")))
    #     d3.metric("Det. max", _fmt(deterministic.get("max")))
    #     st.caption(
    #         "Rule-based 4-persona DCF range from PERSONA_BASED_VALUATION.py "
    #         "(converted to native currency) for comparison with the agentic crew."
    #     )

    # ── Usage + errors ──
    usage = result.get("usage") or {}
    if usage.get("total_tokens"):
        st.caption(
            f"Token usage: ~{usage.get('total_tokens', 0):,} total "
            f"({usage.get('prompt_tokens', 0):,} prompt / "
            f"{usage.get('completion_tokens', 0):,} completion)."
        )
    errors = result.get("errors") or []
    if errors:
        st.warning("Some parts of the persona run had issues:\n\n" + "\n".join(f"- {e}" for e in errors))


# =========================================================
# MAIN APP
# =========================================================

def main():
    """Entry point of the Streamlit dashboard.

    This function is the top-level render pass. Because Streamlit re-executes
    the whole script on every widget interaction / rerun, ``main()`` simply
    rebuilds the layout from data cached in ``session_state``:

      1. Render the header and the sidebar (collecting inputs).
      2. If the "🚀 Analyze Stock" button was clicked with valid inputs, reset
         ``analysis_complete`` (and the cached export), then run the pipeline.
      3. If an analysis is complete, render every cached result section (hero,
         metrics, DCF, sentiment, news, RAG evidence, AI analysis, technical
         analysis, overall summary) and the footer.
      4. Otherwise, render the welcome / idle screen.

    No data is re-fetched on reruns — all results live in ``session_state``.
    """
    # Top-level render pass. Streamlit re-executes this whole script on every
    # widget interaction / rerun, so the layout is rebuilt each time from the
    # data cached in st.session_state rather than re-fetching anything.
    render_header()
    ticker, company_name, display_currency, analyze_btn, persona_btn, save_btn = (
        render_sidebar()
    )

    # ── Trigger analysis ──
    # Only when the user clicks "🚀 Analyze Stock" (with valid inputs) do we
    # run the heavy pipeline. It runs synchronously with a progress bar; when
    # it finishes it stores every result in session_state and calls st.rerun()
    # so this script re-executes and hits the results branch below.
    if analyze_btn and ticker and company_name:
        st.session_state.analysis_complete = False
        # Drop the cached export so the new analysis regenerates its own HTML.
        st.session_state.pop("export_html", None)
        st.session_state.pop("export_filename", None)
        # A stale persona result belongs to a different ticker — clear it so the
        # persona section can't masquerade as part of the new analysis.
        _pr = st.session_state.get("persona_result")
        if _pr and _pr.get("ticker", "") != ticker:
            st.session_state.persona_result = None
        run_analysis(ticker, company_name)

    # ── Trigger multi-persona agentic valuation (independent sidebar option) ──
    if persona_btn and ticker and not st.session_state.get("persona_running"):
        run_persona_valuation(ticker, company_name)

    # ── Save the current analysis snapshot to Chroma (sidebar button) ──
    # Only fires on the exact rerun where the user clicked "💾 Save Analysis".
    if save_btn:
        _save_current_analysis()

    # ── Results display ──
    # After a successful analysis, render the cached results (hero, metrics,
    # DCF, sentiment, news, RAG evidence, AI analysis, technical analysis,
    # overall summary). Before any analysis has run, show the welcome screen.
    if st.session_state.analysis_complete:
        stock_data = st.session_state.stock_data
        sentiment_data = st.session_state.sentiment_data
        news = st.session_state.news
        parsed_sections = st.session_state.parsed_sections
        ticker_display = st.session_state.ticker
        dcf_data = st.session_state.get("dcf_data")

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
            native_currency = stock_data.get("currency", "USD")
            st.markdown(
                f"<div style='text-align: right;'>"
                f"<p style='font-size: 0.75rem; color: var(--text-color-secondary); margin-bottom: 0;'>Current Price</p>"
                f"<h1 style='font-size: 2.5rem;'>{format_price(stock_data['current_price'], native_currency, display_currency)}</h1>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.divider()

        # ── Metrics Strip ──
        render_metrics(stock_data, display_currency)

        st.divider()

        # ── DCF Valuation Scenario Cards ──
        render_dcf_valuation(dcf_data, stock_data, display_currency)

        st.divider()

        # ── Sentiment + News (two columns) ──
        sent_col, news_col = st.columns(2)
        with sent_col:
            render_sentiment(sentiment_data)
        with news_col:
            render_news(news)

        # ── Vectorless RAG: uploaded regulatory filings used in the analysis ──
        render_document_context(st.session_state.get("retrieved_chunks", []))

        # ── AI Analysis (fundamental) ──
        if parsed_sections:
            render_analysis(parsed_sections)

        # ── Technical Analysis (quantitative agent, RAG-grounded) ──
        tech_parsed = st.session_state.get("tech_parsed")
        if tech_parsed is not None:
            render_technical_analysis(tech_parsed)

        # ── Overall Summary (merged fundamental + technical) ──
        overall_summary = st.session_state.get("overall_summary")
        if overall_summary:
            render_overall_summary(overall_summary)

        # ── Footer ──
        st.divider()
        st.caption(
            "This report is for informational purposes only and does not constitute "
            f"financial advice. · Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ── Multi-Persona Agentic Valuation (independent of the standard pipeline) ──
    # Rendered whenever a persona run has completed — even if the user only ran
    # personas and never clicked the full "Analyze Stock" pipeline.
    persona_result = st.session_state.get("persona_result")
    if persona_result:
        render_persona_valuation(persona_result, display_currency)

    if not st.session_state.analysis_complete and not persona_result:
        # ── Welcome / idle state ──
        st.markdown("## 👋 Welcome to Stock Research AI Agent")
        st.markdown(
            "Enter a stock ticker and company name in the sidebar, then click "
            "**🚀 Analyze Stock** to get started.\n\n"
            "The agent will:\n"
            "1. 📡 Fetch real-time stock price data\n"
            "2. 📰 Retrieve latest financial news\n"
            "3. 📊 Analyze market sentiment via NLP\n"
            "4. 📄 Ground the analysis on your uploaded regulatory filings "
            "(vectorless RAG — upload PDFs in the sidebar)\n"
            "5. � Compute 5 years of technical indicators and run a parallel "
            "quantitative analyst (RAG) with a STRONG BUY / BUY / SELL / STRONG SELL verdict\n"
            "6. 🤖 Generate the AI investment analysis + an Overall Summary\n\n"
            "---\n"
            "**Supported Markets:** NSE India (.NS), BSE India (.BO), Global (NYSE/NASDAQ)"
        )


if __name__ == "__main__":
    main()

"""
Stock React Assistant — A LangChain ReAct Agent for Financial Analysis
======================================================================

This module implements an LLM-powered **ReAct (Reasoning + Acting)** agent that
researches a stock ticker by combining three tools into a single autonomous
workflow:

    1. **News Fetching**      – retrieves the latest business headlines via NewsAPI.
    2. **Impact Analysis**    – uses DeepSeek to produce a structured JSON assessment
                                covering financials, business growth, cashflows, and
                                profitability.
    3. **Twitter Sentiment**  – scrapes recent tweets with snscrape and classifies
                                sentiment with a RoBERTa transformer model.

The agent is built on LangChain's ``create_react_agent`` and ``AgentExecutor``,
which orchestrate the tools using the classic ReAct prompt template.

Environment Variables (``.env`` file required):
    - ``DEEPSEEK_API_KEY``   – API key for ChatDeepSeek.
    - ``NEWS_API_KEY``       – API key for NewsAPI (free tier: newsapi.org).

Usage::

    python stock_react_agent.py
    Enter stock ticker: AAPL
    # Agent autonomously fetches news → analyzes impact → checks sentiment → replies.

Dependencies:
    - langchain, langchain-core, langchain-deepseek
    - python-dotenv, newsapi-python, snscrape
    - transformers (Hugging Face), pandas
"""

import os
import json
import datetime
import pandas as pd
from dotenv import load_dotenv
from newsapi import NewsApiClient
import snscrape.modules.twitter as sntwitter
from transformers import pipeline

from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# ---------------------------------------------------------------------------
# Load environment variables from a local .env file.
# Required keys: DEEPSEEK_API_KEY, NEWS_API_KEY
# ---------------------------------------------------------------------------
load_dotenv()

# ================================ TOOLS ====================================
# Each @tool-decorated function below becomes a callable "skill" that the
# ReAct agent can invoke by name. The docstring serves as the tool description
# shown to the LLM when it decides which tool to use.
# ==========================================================================

# Initialise the NewsAPI client once at module level so all calls reuse the
# same HTTP session, avoiding repeated handshakes.
newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

@tool
def get_stock_news(ticker: str) -> str:
    """Fetch the latest 10 business news headlines for a given stock ticker.

    Queries **NewsAPI's ``/v2/everything`` endpoint** (free tier) and returns
    formatted headlines with descriptions.  The agent uses this as the
    **first step** in its research pipeline — raw news text is later passed
    to ``analyze_impact`` for deeper financial assessment.

    Args:
        ticker (str):  Stock ticker symbol (e.g. ``"AAPL"``, ``"TSLA"``,
            ``"MSFT"``).  Case-insensitive; the API will match company names
            as well as symbols.  Whitespace is automatically stripped by
            the caller (``main()``).

    Returns:
        str:  A newline-separated list of up to 10 articles in the format::

            1. <Title>
               <Description>
            2. <Title>
               <Description>

        If no articles are found the string ``"No news found."`` is returned.
        On any HTTP or network error the string ``"Error fetching news: …"``
        is returned, which allows the ReAct agent to observe the failure and
        decide whether to retry or proceed.

    Example:
        >>> get_stock_news("NVDA")
        "1. Nvidia shares surge on AI chip demand\\\\n   The stock rose 8% after..."
    """
    try:
        # --- Call NewsAPI -------------------------------------------------
        # Parameters:
        #   q          – search keyword (ticker)
        #   language   – restrict to English articles
        #   sort_by    – newest first so the agent sees the most timely info
        #   page_size  – limit to 10 to stay within free-tier rate limits
        # ----------------------------------------------------------------
        response = newsapi.get_everything(
            q=ticker,
            language="en",
            sort_by="publishedAt",
            page_size=10
        )

        # NewsAPI wraps results in an "articles" list; default to empty list
        # if the key is missing (defensive programming).
        articles = response.get("articles", [])

        if not articles:
            return "No news found."

        # Build a human-readable, numbered list of headlines + descriptions.
        # Using enumerate(…, 1) so the first article is labelled "1." not "0.".
        formatted = []
        for i, art in enumerate(articles, 1):
            # Fall back to an empty string if 'description' is None/absent.
            formatted.append(f"{i}. {art['title']}\n   {art['description'] or ''}")

        return "\n\n".join(formatted)

    except Exception as e:
        # Catch-all: the agent will see this error string as the "Observation"
        # and can decide its next action (e.g., ask the user for a different
        # ticker or skip news-based analysis).
        return f"Error fetching news: {e}"

# ---------------------------------------------------------------------------
# LLM Initialisation
# ---------------------------------------------------------------------------
# Use DeepSeek's chat model at temperature=0 for deterministic, factual output.
# Temperature 0 is important for financial analysis — we want reproducibility
# and minimal hallucination, not creative writing.
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)


@tool
def analyze_impact(news_text: str) -> str:
    """Analyse financial news and return a structured JSON assessment.

    This tool takes the **raw news text** produced by ``get_stock_news`` and
    prompts DeepSeek to act as a *senior financial analyst*.  The LLM produces
    a JSON object with four mandatory keys, each containing a concise paragraph:

    +---------------------+---------------------------------------------------+
    | Key                 | What it covers                                    |
    +=====================+===================================================+
    | ``financials``      | Revenue, EPS, margins, debt, and overall health.  |
    +---------------------+---------------------------------------------------+
    | ``business_growth`` | Market expansion, new products, user-base trends. |
    +---------------------+---------------------------------------------------+
    | ``cashflows``       | Free cash flow, operating cash flow, capex.       |
    +---------------------+---------------------------------------------------+
    | ``profitability``   | Net income trends, ROE, ROA, cost management.     |
    +---------------------+---------------------------------------------------+

    If the LLM returns malformed JSON (e.g., it added explanatory text outside
    the braces), the raw string is returned so the agent can still reason over
    the content.  Valid JSON is pretty-printed with ``json.dumps(indent=2)``.

    Args:
        news_text (str):  Concatenated news headlines and descriptions
            (typically the output of ``get_stock_news``).  Can be up to a few
            thousand characters; longer text will be truncated by the LLM's
            context window.

    Returns:
        str:  Either pretty-printed JSON or the raw LLM response on parse
        failure.

    Example:
        >>> news = get_stock_news("AAPL")
        >>> analyze_impact(news)
        {
          "financials": "Apple's revenue grew 5% YoY driven by...",
          "business_growth": "Services segment continues to expand...",
          "cashflows": "Free cash flow remains strong at $27B...",
          "profitability": "Gross margins improved to 46.3% due to..."
        }
    """
    # --- Build the analyst prompt ------------------------------------------
    # The system message primes the LLM as a domain expert.  We inject the
    # news text directly and explicitly demand ONLY JSON output to reduce the
    # chance of the model rambling.
    prompt = f"""
    You are a senior financial analyst. Based on the news below, output a JSON with keys:
    "financials", "business_growth", "cashflows", "profitability". Each value a short paragraph.
    News:
    {news_text}
    Output ONLY JSON:
    """

    # Invoke the LLM synchronously.  The response object contains a .content
    # attribute with the generated text.
    response = llm.invoke(prompt)

    # --- Parse & validate JSON --------------------------------------------
    try:
        # json.loads parses the string; json.dumps with indent=2 pretty-prints
        # it back.  This round-trip also validates structural correctness.
        return json.dumps(json.loads(response.content), indent=2)
    except json.JSONDecodeError:
        # The LLM sometimes wraps JSON in markdown fences or adds commentary.
        # Returning the raw content lets the agent still extract meaning from
        # the text even if it isn't strictly valid JSON.
        return response.content

# ---------------------------------------------------------------------------
# Sentiment Pipeline (Hugging Face Transformers)
# ---------------------------------------------------------------------------
# Load a RoBERTa model fine-tuned on ~124M tweets for sentiment classification.
# We set top_k=None to get ALL label probabilities (positive, neutral, negative)
# instead of just the top-scoring label.  This gives the agent richer signal
# than a single "POSITIVE"/"NEGATIVE" verdict.
#
# Model card: cardiffnlp/twitter-roberta-base-sentiment-latest
# Labels:     negative, neutral, positive  (lowercase, no "LABEL_" prefix)
# ---------------------------------------------------------------------------
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None                         # return ALL label probabilities per tweet
)


@tool
def get_twitter_sentiment(ticker: str) -> str:
    """Scrape recent tweets about a stock and return aggregate sentiment.

    Uses **snscrape** (no Twitter API key required) to collect up to 200
    English-language tweets mentioning ``$TICKER`` from the **last 90 days**.
    The first 100 tweets are then fed through a RoBERTa-based sentiment
    classifier.  Results are returned as a percentage breakdown.

    **Why 90 days?**  Financial sentiment typically shifts on a quarterly
    cadence.  A 3-month window captures enough data for a meaningful signal
    without being diluted by stale opinions.

    **Why only 100 tweets classified?**  The Hugging Face pipeline runs on CPU
    by default; classifying more than ~100 tweets at once would make the tool
    unacceptably slow.  The 200-tweet scrape ensures we have a buffer so we
    can sample the *most recent* 100.

    Args:
        ticker (str):  Stock ticker symbol with a ``$`` prefix added
            automatically (e.g. ``"AAPL"`` becomes the query ``$AAPL``).

    Returns:
        str:  A human-readable summary such as::

            Sentiment from 100 tweets: {'positive': '52.0%', 'neutral': '30.0%', 'negative': '18.0%'}

        Returns ``"No tweets found."`` if the scraper yields zero results
        in the 90-day window (common for small-cap or obscure tickers).

    Example:
        >>> get_twitter_sentiment("TSLA")
        "Sentiment from 100 tweets: {'positive': '48.0%', 'neutral': '31.0%', 'negative': '21.0%'}"
    """
    # --- Define the 90-day search window --------------------------------
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=90)

    # Build the snscrape query string:
    #   $TICKER   – cashtag convention used on Twitter for stocks
    #   lang:en   – English-only tweets (the RoBERTa model is English-only)
    #   since:/until: – date range in YYYY-MM-DD format
    query = (
        f"${ticker} lang:en "
        f"since:{start.strftime('%Y-%m-%d')} "
        f"until:{end.strftime('%Y-%m-%d')}"
    )

    # --- Scrape tweets --------------------------------------------------
    # snscrape returns an iterator; we eagerly consume up to 200 items.
    # This avoids keeping the iterator open (which can leak resources).
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        tweets.append(tweet.content)
        if i >= 200:                   # hard cap at 200 to keep runtime bounded
            break

    if not tweets:
        return "No tweets found."

    # --- Classify sentiment (first 100 tweets only) ----------------------
    # The pipeline returns a list of lists: each inner list contains three
    # dicts like {"label": "positive", "score": 0.87}.
    results = sentiment_pipeline(tweets[:100])

    # Count how many tweets have each label as their *highest-probability*
    # class.  max(…, key=lambda x: x['score']) picks the dominant label.
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for res in results:
        best = max(res, key=lambda x: x["score"])
        sentiments[best["label"]] += 1

    # Convert raw counts to percentages for easier human consumption.
    total = sum(sentiments.values())
    summary = {k: f"{v / total:.1%}" for k, v in sentiments.items()}

    return f"Sentiment from {total} tweets: {summary}"

# =============================== AGENT =====================================
# The ReAct (Reasoning + Acting) pattern: the LLM alternates between
# "Thought" (internal reasoning) and "Action" (tool invocation) steps until
# it arrives at a "Final Answer".  LangChain's AgentExecutor manages the
# loop, parsing, and tool dispatch automatically.
# ==========================================================================

# Register all tools so the agent knows what it can call.
tools = [get_stock_news, analyze_impact, get_twitter_sentiment]

# --- ReAct Prompt Template ------------------------------------------------
# This is the canonical ReAct format.  The agent sees:
#   - {tool_names}  → comma-separated list of tool names (auto-populated)
#   - {tools}       → tool name + description (auto-populated)
#   - {input}       → the user's question
#   - {agent_scratchpad} → running log of Thought/Action/Observation steps
#
# The LLM must follow this exact format so the AgentExecutor can parse
# "Action:" and "Action Input:" lines to dispatch the correct tool.
react_prompt = PromptTemplate.from_template("""
You are a helpful financial research assistant. Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""")

# --- Build the agent and executor -----------------------------------------
# create_react_agent wires the LLM + tools + prompt into a Runnable.
# AgentExecutor wraps that with:
#   verbose=True             – prints each Thought/Action/Observation step
#   handle_parsing_errors=True – if the LLM outputs malformed text, the
#                                executor catches the error and feeds it back
#                                as an Observation so the agent can self-correct
#   max_iterations=6         – safety cap to prevent infinite loops
agent = create_react_agent(llm, tools, react_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,                  # show step-by-step reasoning in console
    handle_parsing_errors=True,    # let the agent recover from bad LLM output
    max_iterations=6               # prevent runaway loops (max 6 tool calls)
)


def main():
    """Entry point: prompt for a ticker, then run the full research pipeline.

    Builds a research question from the user-supplied ticker, invokes the
    ReAct agent, and prints the final answer.  The agent autonomously:
        1. Fetches news via ``get_stock_news``
        2. Analyses impact via ``analyze_impact``
        3. Gathers Twitter sentiment via ``get_twitter_sentiment``
        4. Synthesises everything into a Final Answer

    Example session::

        Enter stock ticker: AAPL

        > Entering new AgentExecutor chain...
        Thought: I need to gather news about AAPL first.
        Action: get_stock_news
        Action Input: AAPL
        Observation: 1. Apple beats Q3 estimates...
        Thought: Now I should analyze the impact...
        ...
        Final Answer: Apple shows strong financial momentum with positive
        Twitter sentiment (52% positive).  Key risks include...

    Returns:
        None.  Prints the agent's final answer to stdout.
    """
    # Normalise user input: strip whitespace, convert to uppercase.
    ticker = input("Enter stock ticker: ").strip().upper()

    # Build a comprehensive question that explicitly asks for all three tools.
    # The agent uses this to decide its action sequence.
    question = (
        f"Get top 10 news for {ticker}, analyze impact on financials, "
        f"business growth, cashflows, profitability, and show Twitter "
        f"sentiment for last 3 months."
    )

    # Kick off the agent.  The AgentExecutor returns a dict with at least
    # {"output": <Final Answer string>}.
    result = agent_executor.invoke({"input": question})

    print("\n" + "=" * 60)
    print(result["output"])
    print("=" * 60)


if __name__ == "__main__":
    main()
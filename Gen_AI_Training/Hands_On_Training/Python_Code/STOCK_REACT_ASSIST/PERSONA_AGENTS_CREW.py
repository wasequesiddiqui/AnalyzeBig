"""
crewAI Multi-Persona Agentic Valuation — Pilot Module
=====================================================
A standalone crewAI pilot that runs four *investor-persona* agents — each one
dedicated to a famous investor — to value a stock. Every persona researches
autonomously (ReAct-style tool calling over DeepSeek), emits a structured
fair-value judgement, and a lead-analyst agent reconciles the four opinions
into a consensus range. The deterministic rule-based estimates from
``PERSONA_BASED_VALUATION.persona_super_valuation`` are computed too, so the
UI can compare the agentic and deterministic ranges side-by-side.

Personas (mirroring ``PERSONA_BASED_VALUATION.py``):
    * Warren Buffett     — owner earnings, moat, margin of safety.
    * Charlie Munger     — ROIC vs cost of capital, quality compounding.
    * Rakesh Jhunjhunwala— high-growth, India-tailwind earnings power.
    * Aswath Damodaran   — explicit DCF assumptions & scenario discipline.

Why a separate module?
    * Kept independent of ``streamlit_app.py`` / ``stock_react_agent.py`` so it
      can be imported lazily (only when the user clicks the persona button) and
      run headlessly for tests. It never re-imports the Streamlit app (which
      would re-execute the whole script inside a live session).
    * Data tools are thin crewAI wrappers around plain functions in THIS file
      (yfinance / NewsAPI / TextBlob), so no LangChain tool objects are needed.

Architecture (crewAI 1.15, standalone — no LangChain dependency):
    1. ``run_persona_crew(ticker, company_name)`` fetches a small market
       snapshot (price + native currency), computes the deterministic range,
       then kicks off ONE ``Crew``:
         - 4 persona ``Task``s with ``async_execution=True`` (run in parallel),
           each assigned to its persona ``Agent`` that owns 3 data tools.
         - 1 aggregator ``Task`` whose ``context`` = the four persona tasks.
       ``Process.sequential`` + async persona tasks is the crewAI pattern for
       "run N experts in parallel, then a lead reconciles them".
    2. Each persona is instructed to return a strict JSON object; a tolerant
       JSON extractor (``_extract_json``) recovers it from the raw markdown.
    3. ``run_persona_crew`` maps task outputs back by ``Task.name`` and returns
       a typed ``PersonaCrewResult`` (per-persona cards + aggregate + the
       deterministic range), isolating per-persona errors so one failure never
       kills the whole report.

Dependencies
------------
* ``crewai[litellm]``  — crew orchestration + DeepSeek via LiteLLM.
* ``yfinance``         — market price + financial statement fundamentals.
* ``newsapi-python``, ``textblob`` — qualitative news & sentiment tool.
* ``python-dotenv``    — loads ``DEEPSEEK_API_KEY`` / ``NEWS_API_KEY``.
* ``pydantic``         — result data models.

Environment variables (.env)
----------------------------
* ``DEEPSEEK_API_KEY`` — DeepSeek chat-completions key (LLM for all agents).
* ``NEWS_API_KEY``     — NewsAPI.org key (free tier) for the news tool.

Example
-------
    >>> import PERSONA_AGENTS_CREW as pac
    >>> res = pac.run_persona_crew("INFY.NS", "Infosys")
    >>> res.personas[0].fair_value_per_share
    >>> res.aggregate.min_fair_value
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Windows consoles default to cp1252, which crashes on ₹/emoji/≈ when crewAI's
# internal event bus or our own prints write to stdout/stderr. Force UTF-8 (with
# replacement) for the whole process so headless runs and the Streamlit child
# process don't raise UnicodeEncodeError. Must run before crewai prints anything.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001  (not all environments allow reconfigure)
    pass

# Keep crewAI telemetry off unless the user opts in — anonymous usage data is
# not needed for a local dashboard. Must be set BEFORE importing crewai.
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

import yfinance as yf  # noqa: E402  (module-level side-effect ordering)
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from newsapi import NewsApiClient  # noqa: E402
from textblob import TextBlob  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

import PERSONA_BASED_VALUATION as pv  # deterministic range for comparison  # noqa: E402

load_dotenv()

# =========================================================
# CONFIG
# =========================================================
DEEPSEEK_MODEL = os.getenv("CREWAI_MODEL", "deepseek/deepseek-chat")
MAX_ITER = int(os.getenv("CREWAI_MAX_ITER", "8"))

# Persona portraits live in ./images next to this module. Paths are resolved
# against the module directory (not the current working directory) so they
# resolve no matter where the app was launched from.
PERSONA_IMAGE_DIR = Path(__file__).resolve().parent / "images"


def persona_image(filename: str) -> str:
    """Return the absolute path of a persona portrait ('' when unavailable).

    Args:
        filename (str): Bare file name, e.g. ``"WARREN_BUFFET.png"``.

    Returns:
        str: Absolute path to the image, or an empty string when no file name
             was given or the file is missing, so callers can fall back to the
             persona emoji.
    """
    if not filename:
        return ""
    path = PERSONA_IMAGE_DIR / filename
    return str(path) if path.is_file() else ""

# =========================================================
# TYPED RESULT MODELS (shared with the Streamlit UI)
# =========================================================


class PersonaValuation(BaseModel):
    """One persona's structured judgement."""

    persona: str = Field(..., description="Display persona name")
    emoji: str = Field("🧑", description="Emoji for the persona card")
    image: str = Field("", description="Portrait image path for the persona card")
    fair_value_per_share: Optional[float] = Field(
        None, description="Fair value per share in the stock's native currency"
    )
    currency: str = Field("USD", description="Currency of the fair value")
    stance: str = Field(
        "", description="Undervalued | Fairly valued | Overvalued | No opinion"
    )
    conviction: float = Field(0.0, ge=0.0, le=1.0, description="0–1 confidence")
    one_line_thesis: str = Field("", description="Compressed thesis")
    rationale: str = Field("", description="Multi-sentence reasoning")
    sources: list[str] = Field(default_factory=list, description="Tools/data used")


class AggregateResult(BaseModel):
    """Lead-analyst consensus over all personas."""

    min_fair_value: Optional[float] = None
    max_fair_value: Optional[float] = None
    avg_fair_value: Optional[float] = None
    blended_stance: str = Field("", description="Consensus stance")
    note: str = Field("", description="Lead-analyst reconciliation commentary")
    participating: int = Field(0, description="Personas that yielded a fair value")


class PersonaCrewResult(BaseModel):
    """Everything the UI needs to render the persona section."""

    ticker: str = ""
    company_name: str = ""
    market_price: Optional[float] = None
    currency: str = "USD"
    exchange: str = ""
    generated_at: str = ""
    personas: list[PersonaValuation] = Field(default_factory=list)
    aggregate: AggregateResult = Field(default_factory=AggregateResult)
    deterministic_range: dict[str, Optional[float]] = Field(
        default_factory=dict, description="min/max/mid from PERSONA_BASED_VALUATION"
    )
    errors: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


# =========================================================
# PERSONA METADATA + AGENT BUILDERS
# =========================================================
# Each persona mirrors the deterministic model in PERSONA_BASED_VALUATION.py but
# lets the LLM *reason* like that investor instead of applying a fixed formula.

PERSONAS: list[dict[str, str]] = [
    {
        "key": "warren_buffett",
        "persona": "Warren Buffett",
        "emoji": "🧸",
        "image": "WARREN_BUFFET.png",
        "role": "Warren Buffett — Value / Owner-Earnings Persona",
        "goal": (
            "Valuing the stock as Warren Buffett would: focus on durable "
            "competitive advantage (moat), high and stable returns on equity, "
            "low leverage, consistent owner earnings, and demand a margin of "
            "safety (~20–30%) below intrinsic value before calling it "
            "Undervalued. Base the fair value strictly on tool output."
        ),
        "backstory": (
            "You are Warren Buffett, the Oracle of Omaha. You buy wonderful "
            "businesses at a fair price, stay inside your circle of competence, "
            "prize predictable cash flows and pricing power, and are happy to "
            "walk away if the price offers no margin of safety."
        ),
    },
    {
        "key": "charlie_munger",
        "persona": "Charlie Munger",
        "emoji": "🧠",
        "image": "CHARLIE_MUNGER.png",
        "role": "Charlie Munger — Quality / Moat Persona",
        "goal": (
            "Valuing the stock as Charlie Munger would: judge the quality of "
            "the business by return on invested capital (ROIC) relative to its "
            "cost of capital, the durability of the moat, pricing power and "
            "management integrity. 'Invert' — ask what could destroy the "
            "franchise — and set a fair value that a long-term holder could "
            "compound through. Base the fair value strictly on tool output."
        ),
        "backstory": (
            "You are Charlie Munger, Warren Buffett's partner. You think in "
            "mental models, favour wonderful businesses that need little "
            "capital to grow, and would rather pay a fair price for a great "
            "company than a great price for a fair one."
        ),
    },
    {
        "key": "jhunjhunwala",
        "persona": "Rakesh Jhunjhunwala",
        "emoji": "🦁",
        "image": "RAKESH.png",
        "role": "Rakesh Jhunjhunwala — Growth / India Persona",
        "goal": (
            "Valuing the stock as Rakesh Jhunjhunwala would: overweight "
            "revenue and earnings growth, scalable business models and (for "
            "Indian companies) the domestic growth tailwind. Use a multi-stage "
            "growth mindset — high growth fading to a stable rate — and be "
            "willing to pay up for genuine growth, but not for low-quality "
            "earnings. Base the fair value strictly on tool output."
        ),
        "backstory": (
            "You are Rakesh Jhunjhunwala, 'the Big Bull' of Dalal Street. You "
            "built your fortune by identifying high-growth Indian businesses "
            "early and holding patiently, combining deep fundamental study "
            "with an aggressive appetite for growth and scale."
        ),
    },
    {
        "key": "damodaran",
        "persona": "Aswath Damodaran",
        "emoji": "📐",
        "image": "ASWATH.png",
        "role": "Aswath Damodaran — Valuation / DCF Persona",
        "goal": (
            "Valuing the stock as Aswath Damodaran would: lay out explicit "
            "DCF assumptions (growth, margins, reinvestment, cost of capital), "
            "value the company under base / bull / bear scenarios and "
            "probability-weight them, then report a point fair value WITH the "
            "uncertainty explicitly acknowledged. Never let the market price "
            "drive the value. Base the fair value strictly on tool output."
        ),
        "backstory": (
            "You are Aswath Damodaran, the Dean of Valuation. You believe "
            "every asset has an intrinsic value you can estimate with "
            "discipline, that stories must be tied to numbers, and that "
            "uncertainty should be revealed, not hidden, in every estimate."
        ),
    },
]


def build_persona_agent(meta: dict[str, str], persona_llm: LLM) -> Agent:
    """Instantiate one persona agent with the shared data tools."""
    return Agent(
        role=meta["role"],
        goal=meta["goal"],
        backstory=meta["backstory"],
        llm=persona_llm,
        tools=PERSONA_TOOLS,
        allow_delegation=False,
        max_iter=MAX_ITER,
        max_retry_limit=1,
        verbose=False,
        use_system_prompt=True,
    )


def build_lead_agent(persona_llm: LLM) -> Agent:
    """Instantiate the lead-analyst agent that reconciles all personas."""
    return Agent(
        role="Lead Analyst / Portfolio Strategist",
        goal=(
            "Reconcile the four persona fair values into one consensus range "
            "and a blended stance, explaining where the personas agree and "
            "why they diverge."
        ),
        backstory=(
            "You are a seasoned portfolio strategist who chairs an investment "
            "committee. Four specialist investors (Buffett, Munger, "
            "Jhunjhunwala, Damodaran) have each submitted a fair value. You "
            "summarise the range, the midpoint, and the areas of agreement and "
            "disagreement without inventing numbers."
        ),
        llm=persona_llm,
        allow_delegation=False,
        max_iter=6,
        max_retry_limit=1,
        verbose=False,
        use_system_prompt=True,
    )


# =========================================================
# DATA HELPERS (plain functions — also used by the tools)
# =========================================================


def _safe_fin(df: pd.DataFrame, names: list[str], idx: int = 0) -> float:
    """Return the first present column value from a yfinance table (or NaN)."""
    for name in names:
        if name in df.columns:
            val = df[name].iloc[idx]
            if pd.notna(val):
                return float(val)
    return float("nan")


_FX_CACHE: dict[tuple[str, str], float] = {}


def _fx_lookup(pair: str) -> Optional[float]:
    """Return the latest close for a Yahoo FX ticker like ``"INR=X"``."""
    try:
        hist = yf.Ticker(pair).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """Value of one unit of ``from_currency`` in ``to_currency`` (degrades to 1.0)."""
    a, b = from_currency.upper(), to_currency.upper()
    if a == b:
        return 1.0
    if (a, b) in _FX_CACHE:
        return _FX_CACHE[(a, b)]
    try:
        if a == "USD":
            rate = _fx_lookup(f"{b}=X")
        elif b == "USD":
            r = _fx_lookup(f"{a}=X")
            rate = (1.0 / r) if r else None
        else:
            r1 = get_fx_rate(a, "USD")
            r2 = get_fx_rate("USD", b)
            rate = r1 * r2 if r1 and r2 else None
        rate = rate if rate and rate > 0 else 1.0
    except Exception:  # noqa: BLE001
        rate = 1.0
    _FX_CACHE[(a, b)] = rate
    return rate


def reporting_currency(ticker: str) -> str:
    """Return the currency the company reports its financials in (best-effort).

    Uses Yahoo's ``financialCurrency`` (falls back to ``currency``). Some
    dual-listed names (e.g. INFY.NS) report in USD even though they trade in
    INR, so this can differ from the market price currency.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("financialCurrency") or info.get("currency") or "USD"
    except Exception:  # noqa: BLE001
        return "USD"


def get_price_snapshot(ticker: str) -> dict[str, Any]:
    """Return a small price snapshot for a COMPLETE Yahoo Finance symbol.

    Mirrors ``streamlit_app.get_stock_price`` semantics but lives here so the
    module never has to import the Streamlit app. The symbol is consumed
    EXACTLY as supplied — including its exchange suffix, e.g. ``INFY.NS`` /
    ``RELIANCE.BO`` — since no ``.NS`` / ``.BO`` suffix is ever appended.
    Returns a dict with ``ticker``, ``exchange``, ``currency`` (INR for Indian
    exchanges else USD), ``current_price`` and ``error`` on failure.
    """
    try:
        symbol = (ticker or "").upper().strip()
        stock = yf.Ticker(symbol)
        hist = stock.history(period="5d")
        if hist.empty:
            return {"error": f"No stock data found for '{symbol}'."}
        exchange = "Global"
        if symbol.endswith(".NS"):
            exchange = "NSE India"
        elif symbol.endswith(".BO"):
            exchange = "BSE India"
        return {
            "ticker": symbol,
            "exchange": exchange,
            "currency": "INR" if exchange in ("NSE India", "BSE India") else "USD",
            "current_price": round(float(hist["Close"].iloc[-1]), 2),
        }
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def build_fundamentals_text(ticker: str) -> str:
    """Fetch fundamentals + ratios and render a compact prompt-ready summary.

    Pulls annual income-statement / balance-sheet / cash-flow figures via
    yfinance (same extraction strategy as ``PERSONA_BASED_VALUATION.py``) and
    returns a short, labelled text block that an LLM can reason over. All money
    values are in the stock's reporting currency.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        fin = stock.financials.T
        bs = stock.balance_sheet.T
        cf = stock.cashflow.T
        if fin.empty or bs.empty:
            return "Fundamentals unavailable: no financial statements found."

        rev = _safe_fin(fin, ["Total Revenue", "Revenue", "Operating Revenue"])
        ebit = _safe_fin(fin, ["EBIT", "Ebit", "Operating Income", "Operating Income or Loss"])
        ni = _safe_fin(fin, ["Net Income", "Net Income Common Stockholders"])
        int_exp = _safe_fin(fin, ["Interest Expense", "Interest Expense Non Operating"], 0)
        tax = _safe_fin(fin, ["Tax Provision", "Income Tax Expense"])
        depr = _safe_fin(fin, ["Depreciation & Amortization", "Reconciled Depreciation", "Depreciation"])
        capex_raw = _safe_fin(cf, ["Capital Expenditure", "Capital Expenditures"], 0)
        op_cf = _safe_fin(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"], 0)
        div_paid = _safe_fin(cf, ["Dividends Paid"], 0)

        shares = info.get("sharesOutstanding")
        if not shares:
            shares = _safe_fin(fin, ["Basic Average Shares", "Diluted Average Shares"]) or float("nan")
        beta = info.get("beta", 1.0) or 1.0
        country = info.get("country", "")

        tot_debt = _safe_fin(bs, ["Total Debt"], 0)
        cash = _safe_fin(bs, ["Cash And Cash Equivalents", "Cash"], 0)
        equity = _safe_fin(bs, ["Total Stockholder Equity", "Total Equity Gross Minority Interest"], 0)
        cur_assets = _safe_fin(bs, ["Current Assets"], 0)
        cur_liab = _safe_fin(bs, ["Current Liabilities"], 0)
        nwc = (cur_assets - cash) - (cur_liab - _safe_fin(bs, ["Short Term Debt", "Current Debt"], 0))

        tax_rate = tax / (ni + tax) if (ni + tax) and pd.notna(ni) else 0.21
        rev_growth = info.get("revenueGrowth")
        if rev_growth is None or pd.isna(rev_growth) or rev_growth < -0.5:
            rev_growth = 0.05
        roe = ni / equity if equity and pd.notna(ni) else float("nan")
        roic_denom = (_safe_fin(bs, ["Total Assets"], 0) - cash + nwc)
        roic = ebit * (1 - tax_rate) / roic_denom if roic_denom else float("nan")
        capex = abs(capex_raw) if pd.notna(capex_raw) else float("nan")
        fcf = (op_cf - capex) if pd.notna(op_cf) and pd.notna(capex) else float("nan")

        def money(v: float) -> str:
            return "n/a" if pd.isna(v) or v is None else f"{v:,.0f}"

        def pct(v: float, nd: int = 1) -> str:
            return "n/a" if pd.isna(v) or v is None else f"{v * 100:.{nd}f}%"

        cur = info.get("financialCurrency") or info.get("currency") or "USD"
        lines = [
            f"Company: {info.get('longName') or ticker} ({ticker})",
            f"Country: {country} | Sector: {info.get('sector', 'n/a')} | Industry: {info.get('industry', 'n/a')}",
            f"Reporting currency: {cur} | Shares outstanding: {shares:,.0f}" if shares and pd.notna(shares) else f"Reporting currency: {cur}",
        ]
        lines += [
            "",
            "Latest ANNUAL figures (reporting currency):",
            f"  Revenue: {money(rev)}",
            f"  EBIT: {money(ebit)} | EBIT margin: {pct(ebit / rev) if rev and pd.notna(rev) else 'n/a'}",
            f"  Net income: {money(ni)} | Net margin: {pct(ni / rev) if rev and pd.notna(rev) else 'n/a'}",
            f"  Tax expense: {money(tax)} | Implied tax rate: {pct(tax_rate)}",
            f"  D&A: {money(depr)} | Capex (abs): {money(capex)} | Operating CF: {money(op_cf)}",
            f"  Free cash flow (OCF - capex): {money(fcf)}",
            f"  Dividends paid: {money(div_paid)}",
            "",
            "Latest BALANCE SHEET (reporting currency):",
            f"  Total assets: {money(_safe_fin(bs, ['Total Assets'], 0))}",
            f"  Total debt: {money(tot_debt)} | Cash & equivalents: {money(cash)} | Net debt: {money(tot_debt - cash)}",
            f"  Stockholders equity: {money(equity)} | Current assets: {money(cur_assets)} | Current liabilities: {money(cur_liab)}",
            "",
            "Key ratios & growth (computed from the above):",
            f"  Revenue growth (latest y/y): {pct(rev_growth)}",
            f"  Return on equity (ROE): {pct(roe)}",
            f"  Return on invested capital (ROIC): {pct(roic) if pd.notna(roic) else 'n/a'}",
            f"  Debt/equity: {f'{tot_debt / equity:.2f}' if equity and pd.notna(equity) else 'n/a'}",
            f"  Beta: {beta:.2f}",
            "",
            "Note: values above are point-in-time from Yahoo Finance and may be "
            "simplified; treat them as reasonable inputs, never as audited data.",
        ]
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001
        return f"Fundamentals unavailable: {e}"


def build_news_and_sentiment_text(company_name: str) -> str:
    """Fetch recent headlines + TextBlob sentiment and render a prompt summary.

    Combines the behaviour of ``get_stock_news`` and ``analyze_news_sentiment``
    from the main app into one tool so personas get qualitative context in a
    single call. Degrades gracefully when the key is missing / the API fails.
    """
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return "News unavailable: NEWS_API_KEY is not configured."
    try:
        client = NewsApiClient(api_key=key)
        response = client.get_everything(
            q=f"{company_name} stock", language="en",
            sort_by="publishedAt", page_size=20,
        )
        articles = response.get("articles", [])
        if not articles:
            return f"No recent news articles found for '{company_name}'."
        headlines = []
        scores = []
        for a in articles[:5]:
            title = a.get("title") or ""
            desc = a.get("description") or ""
            headlines.append(
                f"- {title}" + (f" ({a.get('source', {}).get('name', '')})" if a.get("source") else "")
            )
            text = f"{title} {desc}"
            if text.strip():
                scores.append(TextBlob(text).sentiment.polarity)
        avg = sum(scores) / len(scores) if scores else 0.0
        label = "Positive" if avg > 0.15 else ("Negative" if avg < -0.15 else "Neutral")
        return (
            f"Latest headlines for {company_name}:\n" + "\n".join(headlines) +
            f"\n\nAggregate sentiment (TextBlob over {len(scores)} articles): "
            f"{label} (score {avg:.2f})."
        )
    except Exception as e:  # noqa: BLE001
        return f"News unavailable: {e}"


# =========================================================
# CREWAI TOOLS (agent-callable; return strings the LLM reads)
# =========================================================


@tool("get_stock_price")
def get_stock_price(ticker: str) -> str:
    """Fetch the current market price for a COMPLETE Yahoo Finance symbol
    (e.g. INFY.NS, RELIANCE.BO, AAPL).

    The symbol is used exactly as supplied — no .NS / .BO suffix is appended.
    Returns JSON with ticker, exchange, currency and current_price (or an
    error object)."""
    return json.dumps(get_price_snapshot(ticker))


@tool("get_stock_fundamentals")
def get_stock_fundamentals(ticker: str) -> str:
    """Fetch financial-statement fundamentals + key ratios for a ticker.

    Returns a compact text block: revenue, EBIT, net income, margins, free
    cash flow, debt, equity, ROE, ROIC, revenue growth, beta, shares, and the
    reporting currency. Use this to value the business — never invent figures."""
    return build_fundamentals_text(ticker)


@tool("get_stock_news_and_sentiment")
def get_stock_news_and_sentiment(company_name: str) -> str:
    """Fetch recent news headlines and aggregate sentiment for a COMPANY NAME
    (e.g. 'Infosys', not 'INFY.NS'). Returns headlines plus a Positive /
    Neutral / Negative label with a polarity score."""
    return build_news_and_sentiment_text(company_name)


PERSONA_TOOLS = [get_stock_price, get_stock_fundamentals, get_stock_news_and_sentiment]


# =========================================================
# LLM FACTORY
# =========================================================


def get_llm() -> LLM:
    """Return the shared DeepSeek LLM (LiteLLM 'deepseek' provider)."""
    return LLM(model=DEEPSEEK_MODEL, temperature=0, timeout=120)


# =========================================================
# PARSERS
# =========================================================


def _extract_json(text: str) -> Optional[dict]:
    """Tolerantly extract the first valid JSON object from a raw LLM response."""
    if not text:
        return None
    text = text.strip()
    # Strip markdown fences if the model wrapped the JSON.
    text = re.sub(r"```(?:json)?", "", text)
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        idx = text.find("{", idx + 1)
    return None


def _num(value: Any) -> Optional[float]:
    """Coerce a parsed JSON value to float (strips currency/commas if needed)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value.replace(",", ""))
        if cleaned in ("", "-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def persona_from_raw(meta: dict[str, str], raw: str, market_price: Optional[float] = None) -> PersonaValuation:
    """Parse a persona task's raw output into a ``PersonaValuation``.

    Uses the strict JSON object the prompt requested; falls back to a light
    scan of the raw text so a formatting slip never loses the value. When
    ``market_price`` is provided, fallback figures are sanity-filtered to a
    plausible 0.1x–8x band so revenue-scale values (e.g. "$20,158M") can't be
    mistaken for per-share fair values.
    """
    fallback = PersonaValuation(
        persona=meta["persona"],
        emoji=meta["emoji"],
        image=persona_image(meta.get("image", "")),
    )
    data = _extract_json(raw)
    if not data:
        # Regex fallback for narrative (non-JSON) responses. Collect candidate
        # per-share figures, then keep the first that is plausible vs the price.
        candidates = []
        patterns = [
            r"fair[_ ]value[_ ]per[_ ]share[\"':=\s]*[₹$€£]?\s*([\d,]+(?:\.\d+)?)",
            r"(?:fair|intrinsic) value[^\n]{0,40}?(?:per share|share)[^\d₹$€£]{0,20}[₹$€£]?\s*([\d,]+(?:\.\d+)?)",
            r"[₹$€£]\s*([\d,]+(?:\.\d+)?)",
        ]
        for pat in patterns:
            for m in re.finditer(pat, raw, re.IGNORECASE):
                candidates.append(float(m.group(1).replace(",", "")))
        if market_price:
            for c in candidates:
                if 0.1 * market_price <= c <= 8 * market_price:
                    fallback.fair_value_per_share = c
                    break
        elif candidates:
            fallback.fair_value_per_share = candidates[0]
        stance = re.search(r"\b(Undervalued|Fairly valued|Overvalued|No opinion)\b", raw, re.IGNORECASE)
        if stance:
            fallback.stance = stance.group(1).title()
        thesis = re.search(r"(?:thesis|one[- ]line)[\"':\s]*([^\n]+)", raw, re.IGNORECASE)
        if thesis:
            fallback.one_line_thesis = thesis.group(1).strip()
        return fallback

    # JSON path: sanity-guard a non-null fair value that looks like a wrong unit.
    fv = _num(data.get("fair_value_per_share"))
    if fv is not None and market_price and not (0.1 * market_price <= fv <= 8 * market_price):
        fv = None

    persona = PersonaValuation(
        persona=data.get("persona") or meta["persona"],
        emoji=meta["emoji"],
        image=persona_image(meta.get("image", "")),
        fair_value_per_share=fv,
        currency=str(data.get("currency") or "USD"),
        stance=str(data.get("stance") or ""),
        conviction=float(data.get("conviction") or 0.0),
        one_line_thesis=str(data.get("one_line_thesis") or ""),
        rationale=str(data.get("rationale") or ""),
        sources=list(data.get("sources") or []),
    )
    return persona


def _compute_fallback_aggregate(personas: list[PersonaValuation], currency: str) -> AggregateResult:
    """Derive min/max/avg from persona numbers when the lead task output is lost."""
    vals = [p.fair_value_per_share for p in personas if p.fair_value_per_share is not None]
    if not vals:
        return AggregateResult(note="No persona returned a usable fair value.", participating=0)
    stance_counts: dict[str, int] = {}
    for p in personas:
        if p.stance:
            stance_counts[p.stance] = stance_counts.get(p.stance, 0) + 1
    top = max(stance_counts, key=stance_counts.get) if stance_counts else "No opinion"
    return AggregateResult(
        min_fair_value=min(vals),
        max_fair_value=max(vals),
        avg_fair_value=sum(vals) / len(vals),
        blended_stance=top,
        note="Lead-analyst output could not be parsed; range computed from persona values.",
        participating=len(vals),
    )


def aggregate_from_raw(raw: str, personas: list[PersonaValuation], currency: str) -> AggregateResult:
    """Parse the lead-analyst raw output (or fall back to a computed range)."""
    data = _extract_json(raw)
    if not data:
        return _compute_fallback_aggregate(personas, currency)
    agg = AggregateResult(
        min_fair_value=_num(data.get("min_fair_value")),
        max_fair_value=_num(data.get("max_fair_value")),
        avg_fair_value=_num(data.get("avg_fair_value")),
        blended_stance=str(data.get("blended_stance") or ""),
        note=str(data.get("note") or ""),
    )
    vals = [p.fair_value_per_share for p in personas if p.fair_value_per_share is not None]
    agg.participating = len(vals)
    # Self-heal missing fields from persona values.
    if (agg.min_fair_value is None or agg.max_fair_value is None) and vals:
        agg.min_fair_value = min(vals)
        agg.max_fair_value = max(vals)
    if agg.avg_fair_value is None and vals:
        agg.avg_fair_value = sum(vals) / len(vals)
    return agg


# =========================================================
# TASK DESCRIPTION TEMPLATES
# =========================================================


def _persona_task_description(
    meta: dict[str, str], ticker: str, company: str, price: float,
    currency: str, fx_note: str = "",
) -> str:
    return f"""
Research the stock {ticker} ({company or ticker}) exactly as {meta['persona']} would,
using the provided tools. {meta['goal']}

Market anchor: current price is {currency} {price:,.2f} (use get_stock_price to confirm;
use get_stock_fundamentals for the financial statements, and
get_stock_news_and_sentiment for qualitative context).
{fx_note}
Working discipline:
- Work SILENTLY. Do NOT show calculations, scenario tables, internal debate, or
  reasoning anywhere in your replies — do arithmetic in your head and give only results.
- Yahoo's beta for non-US listings is frequently unreliable; if it looks implausible
  (e.g. < 0.3 for a large-cap), use a reasonable industry beta instead.
- Do one pass and commit. Never rewrite or second-guess multiple times.
- Keep any news/fundamental figures you cite to what the tools returned.

RESPONSE CONTRACT (very important):
Your FINAL message must contain ONLY the single JSON object below — no markdown
fences, no prose, no bullet points, no working, nothing before or after it:

{{
  "persona": "{meta['persona']}",
  "fair_value_per_share": <number in {currency} or null if you cannot estimate>,
  "currency": "{currency}",
  "stance": "Undervalued | Fairly valued | Overvalued | No opinion",
  "conviction": <0.0 to 1.0>,
  "one_line_thesis": "<one sentence>",
  "rationale": "<1-2 sentences, tied strictly to the tool data you saw>",
  "sources": ["<tool/figures you relied on>"]
}}

Rules:
- fair_value_per_share MUST be a plain number in {currency}. If you genuinely
  cannot estimate a value, set it to null and choose "No opinion".
- Stance compares your fair value to the current market price.
- Never invent figures that are not in the tool output; if data is missing,
  say so in the rationale.
"""


def _lead_task_description(ticker: str, company: str, price: float, currency: str) -> str:
    return f"""
You are chairing an investment committee on {ticker} ({company or ticker}), currently
trading at {currency} {price:,.2f}. Four specialist investors have submitted fair
values in {currency} (see context). Reconcile them into a consensus.

Return your FINAL answer as ONE valid JSON object and nothing else —
no markdown fences, no preamble:

{{
  "min_fair_value": <number>,
  "max_fair_value": <number>,
  "avg_fair_value": <number>,
  "blended_stance": "<one of Undervalued | Fairly valued | Overvalued | No opinion>",
  "note": "<2-4 sentences: where the personas agree, why they diverge, and your committee conclusion>"
}}

Use ONLY the persona fair values provided in context — never invent numbers.
"""


# =========================================================
# ORCHESTRATION
# =========================================================


def deterministic_range(ticker: str, to_currency: Optional[str] = None) -> dict[str, Optional[float]]:
    """Compute the deterministic 4-persona range for side-by-side comparison.

    ``PERSONA_BASED_VALUATION.persona_super_valuation`` derives per-share
    values from the financial statements, so for dual-listed names it returns
    values in the *reporting* currency (e.g. USD for INFY.NS). When
    ``to_currency`` is given and differs from the reporting currency, the
    result is converted so it can be compared against price-currency fair
    values.
    """
    try:
        low, high, mid = pv.persona_super_valuation(ticker)
        if to_currency and low is not None:
            from_cur = reporting_currency(ticker)
            if from_cur != to_currency:
                rate = get_fx_rate(from_cur, to_currency)
                if rate and rate != 1.0:
                    low, high, mid = (low * rate, high * rate, mid * rate)
        return {"min": low, "max": high, "mid": mid}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _snapshot_or_raise(ticker: str) -> dict[str, Any]:
    snap = get_price_snapshot(ticker)
    if "error" in snap or snap.get("current_price") is None:
        raise ValueError(f"Could not resolve a market price for '{ticker}': {snap.get('error')}")
    return snap


def run_persona_crew(
    ticker: str,
    company_name: str = "",
    verbose: bool = False,
) -> PersonaCrewResult:
    """Run the multi-persona crew and return a typed result.

    Steps:
      1. Resolve a market snapshot (price + currency) so the UI always has an
         anchor even if a persona fails.
      2. Compute the deterministic ``PERSONA_BASED_VALUATION`` range.
      3. Kick off ONE crew: 4 async persona tasks + a lead aggregator task.
      4. Map raw task outputs back by name, parse them, and assemble the result
         with per-persona error isolation.

    Raises:
        ValueError: when the ticker cannot be resolved (caught by the caller).
    """
    result = PersonaCrewResult(
        ticker=ticker.upper().strip(),
        company_name=company_name,
        generated_at=datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 1. Market snapshot
    snap = _snapshot_or_raise(ticker)
    result.market_price = snap["current_price"]
    result.currency = snap["currency"]
    result.exchange = snap["exchange"]
    price, currency = result.market_price, result.currency

    # 2. FX note + deterministic comparison range (both standardized on the
    #    native market-price currency).
    rep_cur = reporting_currency(snap["ticker"])
    fx_note = ""
    if rep_cur and rep_cur != currency:
        rate = get_fx_rate(rep_cur, currency)
        fx_note = (
            f"Note: the company's financial statements are reported in {rep_cur}, but "
            f"you must report fair_value_per_share in {currency} (the market-price "
            f"currency). If you need to convert, use the approximate rate "
            f"1 {rep_cur} \u2248 {rate:,.2f} {currency} (so 1 {currency} \u2248 {1/rate:,.4f} {rep_cur}).\n"
        )
    result.deterministic_range = deterministic_range(snap["ticker"], to_currency=currency)

    # 3. Build crew
    llm = get_llm()
    persona_agents = [build_persona_agent(meta, llm) for meta in PERSONAS]
    lead_agent = build_lead_agent(llm)

    persona_tasks = [
        Task(
            name=f"persona_{meta['key']}",
            description=_persona_task_description(
                meta, snap["ticker"], company_name, price, currency, fx_note=fx_note
            ),
            expected_output=(
                "A single JSON object with persona, fair_value_per_share, "
                "currency, stance, conviction, one_line_thesis, rationale, sources."
            ),
            agent=agent,
            async_execution=True,
        )
        for meta, agent in zip(PERSONAS, persona_agents)
    ]
    lead_task = Task(
        name="aggregator_task",
        description=_lead_task_description(snap["ticker"], company_name, price, currency),
        expected_output=(
            "A single JSON object with min_fair_value, max_fair_value, "
            "avg_fair_value, blended_stance, note."
        ),
        agent=lead_agent,
        context=persona_tasks,
    )

    crew = Crew(
        name="Multi-Persona Valuation Crew",
        agents=[*persona_agents, lead_agent],
        tasks=[*persona_tasks, lead_task],
        process=Process.sequential,
        verbose=verbose,
        share_crew=False,
    )

    # 4. Kick off + parse (error-isolated)
    try:
        crew_result = crew.kickoff()
        if getattr(crew_result, "usage_metrics", None) is not None:
            result.usage = _summarise_usage(crew_result.usage_metrics)

        by_name: dict[str, str] = {}
        for output in crew_result.tasks_output:
            name = getattr(output, "name", "") or ""
            by_name[name] = output.raw or ""
        # Personas first (map by their fixed task name)
        for meta in PERSONAS:
            key = f"persona_{meta['key']}"
            raw = by_name.get(key)
            if raw is None:
                result.errors.append(f"{meta['persona']} produced no output.")
                result.personas.append(PersonaValuation(
                    persona=meta["persona"], emoji=meta["emoji"],
                    image=persona_image(meta.get("image", "")),
                ))
                continue
            try:
                result.personas.append(persona_from_raw(meta, raw, market_price=price))
            except Exception as e:  # noqa: BLE001
                result.errors.append(f"{meta['persona']} output could not be parsed: {e}")
                result.personas.append(PersonaValuation(
                    persona=meta["persona"], emoji=meta["emoji"],
                    image=persona_image(meta.get("image", "")),
                ))

        # Aggregator (or fallback)
        agg_raw = by_name.get("aggregator_task")
        if agg_raw:
            result.aggregate = aggregate_from_raw(agg_raw, result.personas, currency)
        else:
            result.aggregate = _compute_fallback_aggregate(result.personas, currency)
            result.errors.append("Lead-analyst task produced no output; used a computed range.")
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"Crew execution failed: {e}")
        result.aggregate = _compute_fallback_aggregate(result.personas, currency)

    return result


def _summarise_usage(metrics: Any) -> dict[str, Any]:
    """Flatten crewAI usage metrics into a small dict for the UI caption."""
    try:
        total = getattr(metrics, "total_tokens", None)
        prompt = getattr(metrics, "prompt_tokens", None)
        completion = getattr(metrics, "completion_tokens", None)
        return {
            "total_tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }
    except Exception:  # noqa: BLE001
        return {}


# =========================================================
# CLI / smoke-test entry point
# =========================================================

if __name__ == "__main__":
    import sys
    tick = sys.argv[1] if len(sys.argv) > 1 else "INFY.NS"
    comp = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"Running persona crew for {tick} ... (this makes real DeepSeek calls)\n")
    res = run_persona_crew(tick, comp, verbose=True)
    print(f"\nTicker: {res.ticker} | Price: {res.currency} {res.market_price} | {res.exchange}")
    for p in res.personas:
        print(f"  {p.persona}: FV={p.fair_value_per_share} {p.currency} | stance={p.stance} "
              f"| conviction={p.conviction} | {p.one_line_thesis[:80]}")
    a = res.aggregate
    print(f"\nAggregate: min={a.min_fair_value} avg={a.avg_fair_value} max={a.max_fair_value} | {a.blended_stance}")
    print(f"Deterministic range: {res.deterministic_range}")
    if res.errors:
        print("\nErrors:")
        for e in res.errors:
            print("  -", e)

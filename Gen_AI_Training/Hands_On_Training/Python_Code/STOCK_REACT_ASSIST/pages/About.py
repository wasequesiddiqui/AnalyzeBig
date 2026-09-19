"""About — Stock Research AI Agent.

Auto-registered as a Streamlit page because it lives in ``pages/`` next to
``Home.py``. Streamlit discovers every ``*.py`` file in this folder and
adds it to the sidebar navigation.

Deliberately self-contained: it imports nothing from the project's own modules
(no langchain / crewAI / yfinance / chromadb), so this page renders instantly
and cannot be broken by an import error elsewhere in the project.

Note: when a ``pages/`` directory exists, Streamlit does not execute the
entrypoint script directly — it registers the entrypoint plus every page and
runs only the selected one. That is why this file calls
``st.set_page_config`` itself instead of relying on ``Home.py``.
"""

import re
from datetime import datetime
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="About · Stock Research AI Agent",
    page_icon="📈",
    layout="wide",
)

# ``__file__`` is <app root>/pages/About.py, so the app root is two levels up.
APP_ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = APP_ROOT / "images"

# Everything in this folder is served by Streamlit at `<app-url>/app/static/<name>`.
# Requires `server.enableStaticServing = true` in .streamlit/config.toml.
STATIC_DIR = APP_ROOT / "static"

PERSONAS = [
    (
        "Warren Buffett",
        "WARREN_BUFFET.png",
        "Owner earnings, durable moats, margin of safety.",
    ),
    (
        "Charlie Munger",
        "CHARLIE_MUNGER.png",
        "Quality at a fair price; inversion and mental models.",
    ),
    (
        "Rakesh Jhunjhunwala",
        "RAKESH.png",
        "Growth at scale; India-market conviction.",
    ),
    (
        "Aswath Damodaran",
        "ASWATH.png",
        "Story plus numbers; rigorous DCF discipline.",
    ),
]

PIPELINE = [
    (
        "📡",
        "Market data",
        "Fetches real-time price and fundamentals from Yahoo Finance.",
    ),
    ("📰", "News", "Pulls the latest financial headlines via NewsAPI."),
    ("📊", "Sentiment", "Scores headline sentiment with TextBlob / NLTK."),
    (
        "📄",
        "Regulatory filings",
        "Grounds the analysis on your uploaded PDFs via vectorless RAG — "
        "no embeddings, no vector database required.",
    ),
    (
        "🧮",
        "Technical analysis",
        "Computes 5 years of indicators and runs a parallel quantitative "
        "analyst with a STRONG BUY / BUY / SELL / STRONG SELL verdict.",
    ),
    (
        "🤖",
        "Investment analysis",
        "Generates the fundamental write-up and merges it into an Overall "
        "Summary.",
    ),
]

CAPABILITIES = [
    (
        "💹",
        "DCF Valuation",
        "Multi-scenario discounted cash flow with a bull / base / bear range.",
    ),
    (
        "🧠",
        "Multi-Persona Agents",
        "Four investor personas research and value the stock autonomously via "
        "crewAI on DeepSeek, then a lead analyst reconciles them.",
    ),
    (
        "⚖️",
        "Rule-based cross-check",
        "A deterministic four-rule valuation range is compared side-by-side "
        "with the agentic crew.",
    ),
    (
        "🧭",
        "Overall Summary",
        "The fundamental and technical verdicts are merged into one combined "
        "recommendation.",
    ),
    ("💾", "Knowledge base", "Save any completed analysis to the Chroma vector store."),
    ("📤", "Export", "Download the full report as a self-contained HTML file."),
    (
        "💱",
        "Currency conversion",
        "Display values in INR, USD or other supported currencies regardless "
        "of the stock's native currency.",
    ),
]

STACK = [
    ("App", "Streamlit"),
    ("LLM", "DeepSeek via an OpenAI-compatible endpoint (langchain-openai)"),
    ("Agents", "crewAI with the LiteLLM provider"),
    ("Market data", "yfinance, pandas, numpy, scipy"),
    ("News & sentiment", "NewsAPI, TextBlob, NLTK"),
    ("RAG", "pypdf with a dependency-free BM25-style keyword retriever"),
    ("Knowledge base", "ChromaDB"),
    ("Reporting", "Jinja2, MarkupSafe"),
]

# ── Header ───────────────────────────────────────────────────────────────
st.title("📈 About — Stock Research AI Agent")
st.markdown(
    "<p style='color: var(--text-color-secondary); font-size: 1.05rem;'>"
    "Real-time market data · News aggregation · AI-powered analysis</p>",
    unsafe_allow_html=True,
)
st.caption(f"Static overview · Rendered {datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.divider()

# ── What it does ─────────────────────────────────────────────────────────
st.subheader("What it does")
st.markdown(
    "Enter a ticker and a company name in the sidebar, then click "
    "**🚀 Analyze Stock**. The agent runs this pipeline:"
)
for icon, title, detail in PIPELINE:
    st.markdown(f"**{icon} {title}** — {detail}")

st.divider()

# ── Capabilities ─────────────────────────────────────────────────────────
st.subheader("Capabilities")
for icon, title, detail in CAPABILITIES:
    with st.container(border=True):
        st.markdown(f"**{icon} {title}**")
        st.caption(detail)

st.divider()

# ── The investor personas ────────────────────────────────────────────────
st.subheader("The investor personas")
cols = st.columns(len(PERSONAS))
for col, (name, filename, lens) in zip(cols, PERSONAS):
    with col:
        with st.container(border=True):
            portrait = IMAGE_DIR / filename
            if portrait.exists():
                st.image(str(portrait), width=72)
            st.markdown(f"**{name}**")
            st.caption(lens)
st.caption(
    "Each persona values the stock independently; a lead analyst then "
    "reconciles their fair values into a consensus range."
)

st.divider()

# ── Technology & data sources ────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Technology")
    for label, value in STACK:
        st.markdown(f"**{label}:** {value}")
with right:
    st.subheader("Data & markets")
    st.markdown(
        "- **Market data:** Yahoo Finance\n"
        "- **News:** NewsAPI.org\n"
        "- **Filings:** your uploaded PDFs (never stored)\n"
        "- **Markets:** NSE India (`.NS`), BSE India (`.BO`), "
        "global NYSE / NASDAQ\n"
        "- **Required keys:** `DEEPSEEK_API_KEY`, `NEWS_API_KEY`"
    )
    st.caption(
        "Locally the keys live in `.env`; on Streamlit Cloud they are entered "
        "as app Secrets and mirrored into the environment at startup."
    )

st.divider()

# ── Reference documents (served from the static/ folder) ─────────────────
st.subheader("📘 Reference Documents")


def _doc_label(path: Path) -> str:
    """Prefer an HTML document's <title>; fall back to a readable filename."""
    if path.suffix.lower() in {".html", ".htm"}:
        try:
            head = path.read_text(encoding="utf-8", errors="ignore")[:8000]
        except OSError:
            head = ""
        match = re.search(
            r"<title[^>]*>(.*?)</title>", head, re.IGNORECASE | re.DOTALL
        )
        if match:
            title = re.sub(r"\s+", " ", match.group(1)).strip()
            if title:
                return title
    return path.stem.replace("_", " ").replace("-", " ").title()


DOCS = (
    sorted(
        (p for p in STATIC_DIR.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.name.lower(),
    )
    if STATIC_DIR.is_dir()
    else []
)

if not DOCS:
    st.caption(
        "No documents available yet — add a file to the `static/` folder and it "
        "will be listed here on the next rerun."
    )
else:
    for doc in DOCS:
        # Streamlit serves everything in `static/` at /app/static/<filename>.
        # The URL is RELATIVE on purpose so it also resolves correctly behind a
        # `server.baseUrlPath`.
        doc_url = f"app/static/{doc.name}"
        size_kb = doc.stat().st_size / 1024
        with st.container(border=True):
            st.markdown(
                f"<a href='{doc_url}' target='_blank' rel='noopener noreferrer' "
                "style='font-size:0.98rem; font-weight:650; "
                "text-decoration:none;'>"
                f"📄 {_doc_label(doc)}</a>"
                "<br><span style='font-size:0.78rem; opacity:0.65;'>"
                f"static/{doc.name} · {size_kb:,.0f} KB</span>",
                unsafe_allow_html=True,
            )
    st.caption("Each document opens in a new browser tab.")

st.divider()

st.warning(
    "⚠️ **Disclaimer** — This tool is for informational and educational "
    "purposes only. It does not constitute financial, investment or tax "
    "advice. Always do your own research before making any investment "
    "decision."
)

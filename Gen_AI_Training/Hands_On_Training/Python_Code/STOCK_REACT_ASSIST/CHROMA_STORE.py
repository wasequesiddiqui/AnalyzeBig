"""
Chroma Cloud persistence for stock analysis snapshots
======================================================
Standalone helper module that stores an analysis "snapshot" — the fundamental
commentary (Financial Outlook / Key Risks / Growth Potential / Investment
Recommendation), the Overall Technical Summary, and an optional persona
lead-analyst consensus — into a Chroma Cloud collection.

Each save creates ONE document (id ``{ticker}_{YYYYMMDD_HHMMSS}``) whose text is
the full, labelled snapshot (so it is semantically retrievable), alongside
scalar metadata for filtering. Every save is a new record — history is kept.

Environment variables (.env)
----------------------------
    CHROMA_HOST      — Chroma Cloud host (default api.trychroma.com).
    CHROMA_API_KEY   — Chroma Cloud API key (``ck-...``).
    CHROMA_TENANT    — Chroma Cloud tenant id.
    CHROMA_DATABASE  — Chroma Cloud database (default StockAnalytics).
    OPENAI_API_KEY   — used by ``text-embedding-3-small`` embeddings.

Usage
-----
    import CHROMA_STORE as cs
    doc_id = cs.save_analysis_record(record)
    cs.count()          # number of saved snapshots
    cs.get_recent(5)    # (id, document, metadata) triples

Note: ``import chromadb`` is deliberately deferred to ``get_client()`` /
``get_collection()`` so importing this module never slows down the Streamlit
app boot (mirrors the lazy crewAI import in ``PERSONA_AGENTS_CREW.py``).
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Name of the collection that holds saved analysis snapshots (inside the
# Chroma database configured in .env, e.g. StockAnalytics).
COLLECTION_NAME = "stock_analysis_reports"
EMBEDDING_MODEL = "text-embedding-3-small"

# Canonical order of the fundamental-report commentary sections that every
# snapshot document carries.
FUNDAMENTAL_SECTION_ORDER = [
    "Financial Outlook",
    "Key Risks",
    "Growth Potential",
    "Investment Recommendation",
]

_client = None       # cached chromadb.HttpClient
_collection = None   # cached collection


def _require_env(key: str, what: str) -> str:
    """Return a non-empty env var or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing {key} in .env — required for {what}."
        )
    return value


# =========================================================
# CLIENT / COLLECTION
# =========================================================

def get_client() -> Any:
    """Return a cached Chroma Cloud HTTP client built from .env credentials."""
    global _client
    if _client is None:
        import chromadb  # deferred so app boot is unaffected
        host = os.getenv("CHROMA_HOST", "api.trychroma.com")
        api_key = _require_env("CHROMA_API_KEY", "Chroma Cloud")
        _client = chromadb.HttpClient(
            host=host,
            port=8000,
            ssl=True,
            # Chroma Cloud authenticates with the API key in the
            # X-Chroma-Token header (Bearer is rejected -> Permission denied).
            headers={"X-Chroma-Token": api_key},
            tenant=os.getenv("CHROMA_TENANT") or "default_tenant",
            database=os.getenv("CHROMA_DATABASE", "StockAnalytics"),
        )
    return _client


def get_collection() -> Any:
    """Return (auto-creating) the analysis-reports collection.

    Uses OpenAI ``text-embedding-3-small`` (via ``OPENAI_API_KEY``) so the
    stored documents are embedded consistently for future semantic queries.
    """
    global _collection
    if _collection is None:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

        embedding_fn = OpenAIEmbeddingFunction(
            api_key=_require_env("OPENAI_API_KEY", "OpenAI embeddings"),
            model_name=EMBEDDING_MODEL,
        )
        _collection = get_client().get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# =========================================================
# RECORD BUILDING (dict -> document text + scalar metadata)
# =========================================================

def _clean_text(value: Any) -> str:
    """Coerce to a stripped string ('' when None)."""
    if value is None:
        return ""
    return str(value).strip()


def _ticker_token(ticker: Any) -> str:
    """A filesystem-safe ticker token for document ids (e.g. INFY.NS -> INFYNS)."""
    token = re.sub(r"[^A-Za-z0-9]+", "", str(ticker or "").upper())
    return token or "UNKNOWN"


def _fmt_value(value: Any, currency: Any = None) -> str:
    """Format a numeric fair value with its currency ('' when None)."""
    if value is None:
        return ""
    text = f"{float(value):,.2f}"
    cur = _clean_text(currency)
    return f"{text} {cur}".strip()


def _render_document(record: dict) -> str:
    """Turn a snapshot record into one labelled markdown document."""
    parts: list[str] = []

    # ── Header ──
    parts.append("# Stock Analysis Snapshot")
    header_lines = []
    ticker = _clean_text(record.get("ticker"))
    company = _clean_text(record.get("company_name"))
    generated = _clean_text(record.get("generated_at"))
    if ticker:
        header_lines.append(f"Ticker: {ticker}")
    if company:
        header_lines.append(f"Company: {company}")
    if generated:
        header_lines.append(f"Generated: {generated}")
    parts.append("\n".join(header_lines) or "—")

    # ── Fundamental commentary sections ──
    sections = record.get("fundamental_sections") or {}
    for title in FUNDAMENTAL_SECTION_ORDER:
        parts.append(f"## {title}")
        body_parts = []
        if title == "Investment Recommendation":
            verdict = _clean_text(record.get("recommendation_verdict"))
            if verdict:
                body_parts.append(f"Verdict: {verdict.upper()}")
        text = _clean_text(sections.get(title))
        body_parts.append(text if text else "No commentary available.")
        parts.append("\n".join(body_parts))

    # ── Overall Technical Summary ──
    parts.append("## Overall Technical Summary")
    tech_parts = []
    tech_verdict = _clean_text(record.get("technical_verdict"))
    if tech_verdict:
        tech_parts.append(f"Technical Verdict: {tech_verdict.upper()}")
    tech_summary = _clean_text(record.get("technical_summary"))
    tech_parts.append(tech_summary if tech_summary else "No technical summary available.")
    parts.append("\n".join(tech_parts))

    # ── Persona lead-analyst consensus ──
    parts.append("## Persona Lead-Analyst Consensus")
    persona = record.get("persona") or {}
    if persona:
        p_lines = []
        cur = persona.get("currency")
        market = _fmt_value(persona.get("market_price"), cur)
        if market:
            p_lines.append(f"Market price: {market}")
        min_fv = _fmt_value(persona.get("min_fair_value"), cur)
        avg_fv = _fmt_value(persona.get("avg_fair_value"), cur)
        max_fv = _fmt_value(persona.get("max_fair_value"), cur)
        if min_fv:
            p_lines.append(f"Min fair value: {min_fv}")
        if avg_fv:
            p_lines.append(f"Avg fair value: {avg_fv}")
        if max_fv:
            p_lines.append(f"Max fair value: {max_fv}")
        stance = _clean_text(persona.get("blended_stance"))
        if stance:
            p_lines.append(f"Blended stance: {stance}")
        commentary = _clean_text(persona.get("commentary"))
        if commentary:
            p_lines.append(f"Lead-analyst commentary: {commentary}")
        parts.append("\n".join(p_lines) or "No persona consensus available.")
    else:
        parts.append("Persona agents were not run for this snapshot.")

    return "\n\n".join(parts)


def _render_metadata(record: dict) -> dict:
    """Scalar metadata for the snapshot (Chroma allows str/int/float/bool only)."""
    persona = record.get("persona") or {}
    has_analysis = bool(record.get("has_analysis"))
    has_persona = bool(record.get("has_persona")) or bool(persona)
    meta: dict[str, Any] = {
        "ticker": _clean_text(record.get("ticker")),
        "company_name": _clean_text(record.get("company_name")),
        "recommendation_verdict": _clean_text(record.get("recommendation_verdict")).upper(),
        "technical_verdict": _clean_text(record.get("technical_verdict")).upper(),
        "has_analysis": has_analysis,
        "has_persona": has_persona,
        "persona_blended_stance": _clean_text(persona.get("blended_stance")),
        "persona_currency": _clean_text(persona.get("currency")),
        "saved_at": _clean_text(record.get("generated_at"))
        or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for key in ("market_price", "min_fair_value", "avg_fair_value", "max_fair_value"):
        value = persona.get(key)
        if value is not None:
            meta[f"persona_{key}"] = float(value)
    return meta


# =========================================================
# PUBLIC API
# =========================================================

def save_analysis_record(record: dict) -> str:
    """Persist one analysis snapshot into the Chroma collection.

    Args:
        record (dict): Snapshot dict with keys:
            - ticker (str), company_name (str)
            - generated_at (str)
            - has_analysis (bool), has_persona (bool)
            - fundamental_sections (dict): canonical title -> commentary text
              (Financial Outlook, Key Risks, Growth Potential,
               Investment Recommendation)
            - recommendation_verdict (str|None): BUY / HOLD / SELL
            - technical_verdict (str|None), technical_summary (str|None)
            - persona (dict|None): lead-analyst consensus —
              market_price, currency, exchange, generated_at,
              min/avg/max_fair_value, blended_stance, participating,
              commentary (= the lead-analyst note).

    Returns:
        str: the id of the newly created document.

    Raises:
        ValueError: when the record has neither ticker nor company_name.
    """
    ticker = _clean_text(record.get("ticker"))
    company = _clean_text(record.get("company_name"))
    if not ticker and not company:
        raise ValueError(
            "save_analysis_record requires a ticker and/or company_name."
        )

    document = _render_document(record)
    metadata = _render_metadata(record)
    doc_id = (
        f"{_ticker_token(ticker)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    get_collection().add(ids=[doc_id], documents=[document], metadatas=[metadata])
    return doc_id


def count() -> int:
    """Number of saved analysis snapshots in the collection."""
    return get_collection().count()


def get_recent(limit: int = 5) -> list[tuple[str, str, dict]]:
    """Return the most recent ``limit`` saved records as (id, document, meta)."""
    result = get_collection().get(limit=limit, include=["documents", "metadatas"])
    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    return list(zip(ids, docs, metas))

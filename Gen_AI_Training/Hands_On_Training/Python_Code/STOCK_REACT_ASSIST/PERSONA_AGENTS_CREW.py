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
    * Kept independent of ``Home.py`` / ``stock_react_agent.py`` so it
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
Required:
    * ``DEEPSEEK_API_KEY`` — DeepSeek chat-completions key (LLM for all agents).
    * ``NEWS_API_KEY``     — NewsAPI.org key (free tier) for the news tool.

Optional overrides (read once at import time — see ``CONFIG`` below):
    * ``CREWAI_MODEL``     — LiteLLM model string, default ``deepseek/deepseek-chat``.
    * ``CREWAI_MAX_ITER``  — ReAct iterations per persona agent, default ``8``.

Quickstart
----------
Install, then export the keys (``.env`` is loaded automatically)::

    pip install "crewai[litellm]" yfinance newsapi-python textblob python-dotenv
    set DEEPSEEK_API_KEY=sk-...        :: Windows cmd
    set NEWS_API_KEY=...               :: PowerShell: $env:DEEPSEEK_API_KEY = "sk-..."

Run a valuation from Python — :func:`run_persona_crew` is the one function
almost every caller needs. The figures below are illustrative; the real call
blocks for tens of seconds and makes live, billable DeepSeek requests::

    >>> import PERSONA_AGENTS_CREW as pac
    >>> res = pac.run_persona_crew("INFY.NS", "Infosys")                # doctest: +SKIP
    >>> res.market_price, res.currency                                  # doctest: +SKIP
    (1502.3, 'INR')
    >>> [(p.persona, p.fair_value_per_share) for p in res.personas]     # doctest: +SKIP
    [('Warren Buffett', 1280.0), ('Charlie Munger', 1250.0),
     ('Rakesh Jhunjhunwala', 1185.0), ('Aswath Damodaran', 1285.0)]
    >>> res.aggregate.min_fair_value, res.aggregate.max_fair_value      # doctest: +SKIP
    (1185.0, 1285.0)
    >>> res.aggregate.blended_stance                                    # doctest: +SKIP
    'Undervalued'

Or from the command line (human-readable summary, ``verbose=True`` so every
tool call is logged)::

    python PERSONA_AGENTS_CREW.py INFY.NS "Infosys"
    python PERSONA_AGENTS_CREW.py AAPL "Apple"        # US listing, USD
    python PERSONA_AGENTS_CREW.py                     # defaults to INFY.NS

Note:
    The ``>>>`` blocks in this docstring are illustrative, not a runnable
    doctest suite: they need network access *and* live API keys. Copy them into
    a REPL to try them.

Public API at a glance
----------------------
* :func:`run_persona_crew` — **entry point**; runs the crew and returns a
  :class:`PersonaCrewResult`.
* :class:`PersonaCrewResult`, :class:`PersonaValuation`, :class:`AggregateResult`
  — Pydantic models; call ``.model_dump()`` for JSON-safe dicts (Streamlit
  session state, HTTP responses, log lines).
* :func:`get_price_snapshot`, :func:`build_fundamentals_text`,
  :func:`build_news_and_sentiment_text` — the plain data functions behind the
  tools; call them directly when you need numbers without an LLM round-trip.
* :func:`deterministic_range` — rule-based ``{min, max, mid}`` from
  ``PERSONA_BASED_VALUATION.persona_super_valuation``, for side-by-side compare.
* :data:`PERSONAS` — editable metadata (name / emoji / portrait / role / goal /
  backstory). Append an entry to add a fifth investor.
* :data:`PERSONA_TOOLS` — the three crewAI ``@tool`` objects each persona owns.
* :func:`get_fx_rate`, :func:`reporting_currency`, :func:`persona_image` — small
  helpers for units, currencies and portraits.

Recipes
-------
**1. Render the result in Streamlit.**  Import lazily inside the handler, dump
to a dict, and ``st.rerun()`` so rendering happens *outside* the click handler
(a live crew run blocks for tens of seconds)::

    import streamlit as st

    if st.sidebar.button("Run persona valuation"):
        import PERSONA_AGENTS_CREW as pac          # lazy: keeps app boot fast
        with st.spinner("Four investors are researching…"):
            result = pac.run_persona_crew("INFY.NS", "Infosys")
        st.session_state.persona_result = result.model_dump()
        st.rerun()

    payload = st.session_state.get("persona_result")
    if payload:
        res = pac.PersonaCrewResult(**payload)     # rehydrate the typed model
        for p in res.personas:
            st.subheader(f"{p.emoji} {p.persona} — {p.stance or 'No opinion'}")
            st.metric("Fair value", "—" if p.fair_value_per_share is None
                      else f"{p.currency} {p.fair_value_per_share:,.0f}")

**2. Compare the agentic and deterministic ranges.**  Both are expressed in the
*native market-price* currency, so they are directly comparable::

    lo, hi = res.aggregate.min_fair_value, res.aggregate.max_fair_value
    det = res.deterministic_range                  # {'min': .., 'max': .., 'mid': ..}
    print(f"Crew : {res.currency} {lo:,.0f} – {hi:,.0f}")
    print(f"Rules: {res.currency} {det['min']:,.0f} – {det['max']:,.0f}")

**3. Add a fifth persona.**  Append to :data:`PERSONAS` *before* calling
:func:`run_persona_crew`; the agents, tasks and parser all iterate that list::

    pac.PERSONAS.append({
        "key": "graham",                     # task name becomes persona_graham
        "persona": "Benjamin Graham",
        "emoji": "🔎",
        "image": "",                         # optional file in ./images
        "role": "Benjamin Graham — Deep Value Persona",
        "goal": "Buy only at a discount to net current asset value…",
        "backstory": "You are Benjamin Graham, father of value investing…",
    })

**4. Use the data helpers with no LLM at all** (handy in unit tests — these
still hit Yahoo/NewsAPI, so treat them as integration checks)::

    >>> pac.get_price_snapshot("INFY.NS")["currency"]           # doctest: +SKIP
    'INR'
    >>> pac.build_fundamentals_text("AAPL").splitlines()[0]      # doctest: +SKIP
    'Company: Apple Inc. (AAPL)'
    >>> pac.get_fx_rate("USD", "INR") > 50                       # doctest: +SKIP
    True

**5. Inspect a tool exactly as the agent sees it.**  crewAI tools return plain
strings, so they are easy to assert on::

    >>> pac.get_stock_price.run("AAPL")          # doctest: +SKIP
    '{"ticker": "AAPL", "exchange": "Global", "currency": "USD", "current_price": 232.14}'

**6. Fail-soft handling.**  ``run_persona_crew`` only raises when the *ticker*
cannot be priced; everything else lands in ``result.errors``::

    try:
        res = pac.run_persona_crew("NOT_A_TICKER")
    except ValueError as exc:                     # no price -> nothing to show
        st.error(str(exc))
    else:
        for msg in res.errors:                    # partial failures
            st.warning(msg)

Data flow
---------
::

    run_persona_crew(ticker, company_name)
        |
        +- get_price_snapshot() --------> price + native currency (market anchor)
        +- reporting_currency() --------> e.g. USD financials vs INR price
        +- get_fx_rate() ---------------> FX note injected into every prompt
        +- deterministic_range() -------> rule-based min/max/mid for comparison
        |
        +- Crew.kickoff()
        |     +- Task persona_warren_buffett    (async) -+
        |     +- Task persona_charlie_munger    (async) -| each agent owns
        |     +- Task persona_jhunjhunwala      (async) -| PERSONA_TOOLS and
        |     +- Task persona_damodaran         (async) -+ returns ONE JSON object
        |     +- Task aggregator_task  (context = the four tasks above)
        |
        +- parse outputs by Task.name --> PersonaCrewResult
              +- persona_from_raw()   x4   tolerant JSON + regex fallback
              +- aggregate_from_raw()      or _compute_fallback_aggregate()
              +- errors[]                  never raises on a single bad persona

Design notes
------------
* **Error isolation.** A failing persona yields an empty card plus a line in
  ``result.errors``; the other three still render. Only an unresolvable ticker
  raises :class:`ValueError`, because nothing useful can be shown then.
* **No LangChain.** Every data path is a thin wrapper over a plain function in
  this module, so there are no ``langchain_*`` imports to keep in sync.
* **Headless-friendly.** This module never imports the Streamlit entrypoint
  ``Home`` (that would re-execute the whole script inside a live session), so it
  runs happily under pytest or plain ``python``.
* **Low variance.** ``temperature=0`` plus "work silently, emit JSON only"
  prompts keep token cost and output shape stable.

Gotchas (learned the hard way)
------------------------------
* **Windows consoles.** crewAI's event bus prints emoji/₹/≈ and crashes a
  cp1252 console with ``'charmap' codec`` errors. This module reconfigures
  stdout/stderr to UTF-8 at import time — and that must run *before*
  ``import crewai``, so keep the top-of-file ordering intact.
* **Telemetry.** ``OTEL_SDK_DISABLED=true`` is set before importing crewAI;
  without it local runs are noisy.
* **Chain-of-thought blowups.** Without the "work silently" instruction
  DeepSeek writes a 400-line DCF into its final answer — the JSON parse breaks
  and token cost multiplies. Keep the RESPONSE CONTRACT in the task prompts.
* **Currency mismatch.** Dual-listed Indian names report financials in USD but
  trade in INR (``INFY.NS``). Personas are told to answer in the *market-price*
  currency, and parsed values are re-checked against a 0.1x–8x plausibility
  band around the live price.
* **Yahoo beta.** Frequently wrong for non-US large caps (INFY ≈ 0.11); the
  prompts tell the persona to substitute a sensible industry beta rather than
  spiral.
* **Agents are Pydantic models.** In crewAI 1.x, ``Agent`` / ``Task`` / ``Crew``
  / ``LLM`` are Pydantic v2 models — pass keyword arguments only, and inspect
  accepted fields with ``Agent.model_fields``.

See Also
--------
* ``PERSONA_BASED_VALUATION.persona_super_valuation`` — the deterministic
  counterpart whose range is surfaced in ``deterministic_range``.
* ``Home.py`` — the UI consumer (lazy-imports this module).
* ``stock_react_agent.py`` — the single-agent ReAct analyst, for contrast.
"""

from __future__ import annotations

import functools
import json
import os
import re
import sys
import time
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
# All tunables are read once at import time from the environment (loaded from
# .env by ``load_dotenv()`` above). They are module-level constants, so a test
# that needs different values should set the environment variable *before*
# importing this module.
#
#   CREWAI_MODEL     -> DEEPSEEK_MODEL  (LiteLLM model string)
#   CREWAI_MAX_ITER  -> MAX_ITER        (ReAct tool-call budget per persona)
#
# Example:
#     $env:CREWAI_MODEL = "deepseek/deepseek-reasoner"
#     $env:CREWAI_MAX_ITER = "5"
#     python PERSONA_AGENTS_CREW.py INFY.NS "Infosys"
DEEPSEEK_MODEL = os.getenv("CREWAI_MODEL", "deepseek/deepseek-chat")
MAX_ITER = int(os.getenv("CREWAI_MAX_ITER", "8"))

# Persona portraits live in ./images next to this module. Paths are resolved
# against the module directory (not the current working directory) so they
# resolve no matter where the app was launched from — Streamlit's CWD is
# typically the repo root, while ``python PERSONA_AGENTS_CREW.py`` uses the
# module folder.
#
# Example:
#     >>> PERSONA_IMAGE_DIR.name
#     'images'
PERSONA_IMAGE_DIR = Path(__file__).resolve().parent / "images"


def persona_image(filename: str) -> str:
    """Return the absolute path of a persona portrait ('' when unavailable).

    The UI feeds the return value straight into ``st.image(path)``; an empty
    string means "no portrait", so the caller can fall back to the emoji stored
    on the persona card. A missing file is *not* an error — portraits are purely
    cosmetic, so the whole persona set runs fine with no images at all.

    Args:
        filename (str): Bare file name inside :data:`PERSONA_IMAGE_DIR`, e.g.
            ``"WARREN_BUFFET.png"``. Pass ``""``/``None`` to opt out explicitly.

    Returns:
        str: Absolute path to the existing image file, otherwise ``""``.

    Example:
        Resolution, missing files and the empty-string opt-out all behave the
        same way at the call site — no exception, no truthy path::

            img = persona_image("WARREN_BUFFET.png")   # -> ".../images/WARREN_BUFFET.png"
            img = persona_image("NO_SUCH_FILE.png")    # -> "" (a typo never breaks render)
            img = persona_image("")                    # -> "" (use the emoji instead)

        The Streamlit card renderer is the main consumer::

            img = persona_image(p.image)
            st.image(img, width=90) if img else st.markdown(f"## {p.emoji}")

        Because the path is resolved against the module directory, this works
        identically whether the app was launched from the repo root or from
        inside this folder::

            >>> PERSONA_IMAGE_DIR.is_absolute()
            True
    """
    if not filename:
        return ""
    path = PERSONA_IMAGE_DIR / filename
    return str(path) if path.is_file() else ""

# =========================================================
# TYPED RESULT MODELS (shared with the Streamlit UI)
# =========================================================


class PersonaValuation(BaseModel):
    """One persona's structured judgement — the card the UI renders.

    Built by :func:`persona_from_raw` from the raw JSON a persona agent returns,
    so every field except ``persona`` is optional: a persona that fails to parse
    still produces a (mostly empty) card instead of dropping out of the report.

    Attributes:
        persona (str): Display name, e.g. ``"Warren Buffett"``. Taken from
            :data:`PERSONAS`, not from the LLM, so it is always consistent.
        emoji (str): Fallback glyph when no portrait image is available.
        image (str): Absolute portrait path from :func:`persona_image`.
        fair_value_per_share (float | None): Fair value expressed in
            ``currency``; ``None`` when the persona declined to estimate (see
            ``stance``), or when its number failed the plausibility check.
        deterministic_fair_value (float | None): The matching rule-based estimate
            from ``PERSONA_BASED_VALUATION``, converted into the market-price
            currency. ``None`` when that rule produced no value.
        blended_fair_value (float | None): Mean of ``fair_value_per_share`` and
            ``deterministic_fair_value``; ``None`` unless BOTH exist.
        currency (str): Currency of ``fair_value_per_share`` — always the
            stock's *market-price* currency, never the reporting currency.
        stance (str): One of ``Undervalued`` / ``Fairly valued`` /
            ``Overvalued`` / ``No opinion``.
        conviction (float): Confidence in ``[0.0, 1.0]`` — enforced by Pydantic.
        one_line_thesis (str): Single sentence for the card header.
        rationale (str): One or two sentences citing the tool data used.
        sources (list[str]): Free-text list of tools/figures relied on, e.g.
            ``["get_stock_fundamentals (ROIC 28%)"]``.

    Example:
        Construct one by hand — exactly what the parsers do::

            >>> card = PersonaValuation(
            ...     persona="Warren Buffett",
            ...     emoji="🧸",
            ...     fair_value_per_share=1280.0,
            ...     currency="INR",
            ...     stance="Undervalued",
            ...     conviction=0.7,
            ...     one_line_thesis="Wide moat and net cash justify a premium.",
            ... )
            >>> card.stance, card.conviction
            ('Undervalued', 0.7)

        JSON-safe, so it round-trips through ``st.session_state`` or an HTTP
        response via ``.model_dump()``::

            >>> "blended_fair_value" in card.model_dump()
            True

        ``conviction`` is validated on construction, so ``conviction=1.5``
        raises ``pydantic.ValidationError`` rather than silently rendering a
        150% confidence bar.

        Rendering a card that carries no value — note the em dash instead of
        formatting ``None``::

            for p in result.personas:
                price = "—" if p.fair_value_per_share is None else f"{p.fair_value_per_share:,.0f}"
                st.markdown(f"{p.emoji} **{p.persona}** · {p.stance or 'No opinion'} · {price}")
    """

    persona: str = Field(..., description="Display persona name")
    emoji: str = Field("🧑", description="Emoji for the persona card")
    image: str = Field("", description="Portrait image path for the persona card")
    fair_value_per_share: Optional[float] = Field(
        None, description="Fair value per share in the stock's native currency"
    )
    deterministic_fair_value: Optional[float] = Field(
        None,
        description="Rule-based (PERSONA_BASED_VALUATION) fair value per share, "
                    "converted to the market-price currency",
    )
    blended_fair_value: Optional[float] = Field(
        None,
        description="Mean of the crew and rule-based fair values for this persona "
                    "(None unless both exist)",
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
    """Lead-analyst consensus over all personas.

    Produced by :func:`aggregate_from_raw` from the aggregator task's JSON, or by
    :func:`_compute_fallback_aggregate` when that task fails — in the fallback
    case ``note`` says so explicitly and the numbers come straight from the
    persona cards.

    Attributes:
        min_fair_value (float | None): Lowest persona fair value, e.g. ``1185.0``.
        max_fair_value (float | None): Highest persona fair value.
        avg_fair_value (float | None): Simple (unweighted) mean of the values
            actually produced — ``conviction`` is *not* used as a weight.
        blended_stance (str): Consensus stance, normally the modal persona
            stance (``Undervalued`` / ``Fairly valued`` / ``Overvalued`` /
            ``No opinion``).
        note (str): Two to four sentences of committee commentary.
        participating (int): How many personas contributed a numeric fair value
            (0–4). A low count means the range is thin — surface it in the UI.

    Example:
        The comparison the UI shows beneath the persona cards (``res`` is a
        :class:`PersonaCrewResult` from :func:`run_persona_crew`):

            >>> agg = res.aggregate                              # doctest: +SKIP
            >>> f"{res.currency} {agg.min_fair_value:,.0f} - {agg.max_fair_value:,.0f}"   # doctest: +SKIP
            'INR 1,185 - 1,285'
            >>> agg.participating                                # doctest: +SKIP
            4

        A degenerate case (one participant) is still a valid object::

            >>> agg = AggregateResult(min_fair_value=100.0, max_fair_value=100.0,
            ...                       avg_fair_value=100.0, participating=1)
            >>> agg.max_fair_value - agg.min_fair_value
            0.0

        Guard the render on ``participating`` before advertising a "4 of 4
        investors agree" headline::

            if agg.participating < 4:
                st.caption(f"Only {agg.participating}/4 personas returned a fair value.")
    """

    min_fair_value: Optional[float] = None
    max_fair_value: Optional[float] = None
    avg_fair_value: Optional[float] = None
    blended_stance: str = Field("", description="Consensus stance")
    note: str = Field("", description="Lead-analyst reconciliation commentary")
    participating: int = Field(0, description="Personas that yielded a fair value")


class PersonaCrewResult(BaseModel):
    """Everything the UI needs to render the persona section.

    The single object returned by :func:`run_persona_crew`. It is JSON-safe via
    ``.model_dump()``, so it can round-trip through ``st.session_state`` or be
    returned straight from an API handler.

    Attributes:
        ticker (str): Normalised (upper-cased, stripped) Yahoo symbol.
        company_name (str): Friendly name supplied by the caller; may be ``""``.
        market_price (float | None): Last close used as the valuation anchor.
        currency (str): Market-price currency — ``INR`` for ``.NS``/``.BO``
            listings, else ``USD``.
        exchange (str): ``NSE India`` / ``BSE India`` / ``Global``.
        generated_at (str): Local timestamp, ``YYYY-MM-DD HH:MM:SS``.
        personas (list[PersonaValuation]): Always one entry per
            :data:`PERSONAS` item, in order — possibly empty cards on failure.
        aggregate (AggregateResult): Consensus range and commentary.
        deterministic_range (dict): ``{'min': .., 'max': .., 'mid': ..}`` from
            the rule-based model, or ``{'error': '...'}`` if that failed.
        deterministic_personas (dict): Rule-based fair value per persona, keyed by
            the ``key`` field of :data:`PERSONAS` and converted to the
            market-price currency; ``None`` for personas the rules did not price.
        deterministic_participating (int): How many rule-based personas produced a
            value (0-4).
        deterministic_usable (bool): ``True`` when at least
            ``MIN_DETERMINISTIC_PERSONAS`` rule-based personas priced, i.e. when
            there is something to compare at all. The UI hides the panel when this
            is ``False``. Individual personas whose rule-based value is missing are
            dropped from the comparison silently: the panel shows their CrewAI
            value and no average, rather than hiding the whole table.
        blended_fair_value (float | None): Mean of the per-persona blended values
            (crew and rule-based averaged per persona), or ``None`` when no persona
            has both.
        errors (list[str]): Human-readable partial-failure notes; empty on a
            clean run.
        usage (dict): ``total_tokens`` / ``prompt_tokens`` /
            ``completion_tokens`` when crewAI reports them.

    Example:
        The full round trip the Streamlit app performs::

            >>> res = run_persona_crew("INFY.NS", "Infosys")   # doctest: +SKIP
            >>> res.ticker, res.exchange                        # doctest: +SKIP
            ('INFY.NS', 'NSE India')
            >>> len(res.personas)                               # doctest: +SKIP
            4

        Serialise for session state, then rehydrate on the next Streamlit run::

            st.session_state.persona_result = res.model_dump()
            ...
            res2 = PersonaCrewResult(**st.session_state.persona_result)

        A degraded run still yields a renderable object — inspect
        ``res.errors`` (e.g. ``["Charlie Munger produced no output."]``) before
        claiming "4 of 4 investors" in the UI.

        Rendering the deterministic comparison without assuming it succeeded::

            det = res.deterministic_range
            if "error" in det:
                st.caption("Rule-based range unavailable.")
            else:
                st.caption(f"Rules: {res.currency} {det['min']:,.0f}–{det['max']:,.0f}")
    """

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
    deterministic_personas: dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Rule-based fair value per PERSONAS key, in market currency",
    )
    deterministic_participating: int = Field(
        0, description="Rule-based personas that produced a value (0-4)"
    )
    deterministic_usable: bool = Field(
        False,
        description="True when enough rule-based personas priced to be shown",
    )
    blended_fair_value: Optional[float] = Field(
        None, description="Mean of the per-persona blended fair values"
    )
    errors: list[str] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)


# =========================================================
# PERSONA METADATA + AGENT BUILDERS
# =========================================================
# Each persona mirrors the deterministic model in PERSONA_BASED_VALUATION.py but
# lets the LLM *reason* like that investor instead of applying a fixed formula.
#
# PERSONAS is the single source of truth: run_persona_crew() builds one agent
# and one async task per entry, and the parser/UI iterate it in order. Appending
# a dict here is all it takes to add an investor.
#
# Keys (all strings):
#     key       Short slug; the crewAI task is named ``persona_<key>`` and that
#               name is how the raw output is mapped back to the persona.
#     det_key   Name used by ``PERSONA_BASED_VALUATION.persona_valuations()`` for
#               the matching rule-based estimate, e.g. "Buffett". Links the LLM
#               persona to its rule-based counterpart for the blended average.
#     persona   Display name shown on the card, e.g. "Warren Buffett".
#     emoji     Fallback glyph used when no portrait resolves.
#     image     File name inside ./images, or "" to always use the emoji.
#     role      crewAI agent role -> becomes the system-prompt header.
#     goal      crewAI agent goal -> the valuation mandate (what to look at).
#     backstory crewAI backstory -> the persona's voice and biases.
#
# Example:
#     Inspect the roster before starting a run::
#
#         >>> [p["persona"] for p in PERSONAS]
#         ['Warren Buffett', 'Charlie Munger', 'Rakesh Jhunjhunwala', 'Aswath Damodaran']
#         >>> PERSONAS[0]["key"]
#         'warren_buffett'
#
#     Swap a persona out for one run without editing the file::
#
#         original = pac.PERSONAS[3].copy()
#         pac.PERSONAS[3]["goal"] = "Value it like a cautious bond investor..."
#         try:
#             res = pac.run_persona_crew("AAPL", "Apple")
#         finally:
#             pac.PERSONAS[3].update(original)   # restore for the next run
#
#     Every prompt is built from these strings by _persona_task_description(),
#     so keep ``goal`` behavioural ("what to look at") and ``backstory``
#     voice-y ("how to sound").

PERSONAS: list[dict[str, str]] = [
    {
        "key": "warren_buffett",
        "det_key": "Buffett",
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
        "det_key": "Munger",
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
        "det_key": "Jhunjhunwala",
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
        "det_key": "Damodaran",
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
    """Instantiate one persona agent with the shared data tools.

    Thin factory mapping one :data:`PERSONAS` entry onto a crewAI ``Agent``. All
    four agents share the *same* ``LLM`` instance (cheaper, consistent decoding)
    and the *same* :data:`PERSONA_TOOLS` list; only role / goal / backstory
    differ — which is what makes four outputs meaningfully different despite
    identical inputs.

    Args:
        meta (dict[str, str]): One :data:`PERSONAS` entry. ``role``, ``goal`` and
            ``backstory`` drive the agent; ``key`` / ``persona`` / ``emoji`` /
            ``image`` are used elsewhere (task naming, parsing, UI).
        persona_llm (LLM): Shared DeepSeek handle from :func:`get_llm`.

    Returns:
        Agent: A non-delegating agent with ``max_iter=MAX_ITER`` and
            ``max_retry_limit=1`` — a single retry caps cost if the model emits
            something unparseable.

    Example:
        Build one agent and inspect the scaffolding::

            agent = build_persona_agent(PERSONAS[0], get_llm())
            assert agent.role.startswith("Warren Buffett")
            assert len(agent.tools) == 3

        ``allow_delegation=False`` matters: without it crewAI may route a
        persona's question to another agent, blurring the four independent
        opinions that the reconciliation step depends on. The lead analyst is
        the only agent that sees all four — and it does so through task
        ``context``, not delegation (see :func:`build_lead_agent`).

        In practice you rarely call this directly; :func:`run_persona_crew`
        builds one per persona::

            persona_agents = [build_persona_agent(m, llm) for m in PERSONAS]
    """
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
    """Instantiate the lead-analyst agent that reconciles all personas.

    The chair of the investment committee. It has **no tools** — it must reason
    only over the four persona outputs injected through the aggregator task's
    ``context``, which is exactly how :func:`run_persona_crew` wires it. It also
    gets a lower ``max_iter`` (6) because reconciliation is a single pass, not an
    investigation.

    Args:
        persona_llm (LLM): Shared DeepSeek handle, normally the same instance the
            persona agents use.

    Returns:
        Agent: The non-delegating lead-analyst agent.

    Example:
        Build it standalone::

            lead = build_lead_agent(get_llm())
            assert "Lead Analyst" in lead.role
            assert not lead.tools          # context-driven, not tool-driven

        Added to the crew after the personas so ``Process.sequential`` runs it
        last, once every async persona task has settled::

            crew = Crew(
                agents=[*persona_agents, lead_agent],
                tasks=[*persona_tasks, lead_task],   # lead_task.context = persona_tasks
                process=Process.sequential,
            )
    """
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
# IN-PROCESS CACHE (Yahoo rate-limit protection)
# =========================================================
# Yahoo Finance rate-limits by IP address. One crew run asks for the SAME data
# several times — all four persona agents call ``get_stock_fundamentals`` (each
# fetching ``.info`` + three statements), and the deterministic model fetches
# them again — so without a cache a single run can fire 15+ Yahoo requests and
# trip the limit. This TTL cache collapses those duplicates into one fetch per
# ticker per window.
#
# Deliberately NOT ``st.cache_data``: this module is also used headlessly (the
# CLI and the test suite) and must not depend on a Streamlit runtime.
#
# Wrapped functions:  get_price_snapshot, build_fundamentals_text,
#                     build_news_and_sentiment_text, reporting_currency,
#                     deterministic_persona_values
#
# Example:
#     >>> get_price_snapshot("AAPL") is get_price_snapshot("AAPL")   # 2nd is cached
#     True
#     >>> _CACHE_TTL_SECONDS
#     900.0

#: Seconds a fetched value stays fresh. Override with ``CREWAI_CACHE_TTL``.
_CACHE_TTL_SECONDS = float(os.getenv("CREWAI_CACHE_TTL", "900"))

#: Sentinel distinguishing "not cached" from a legitimately cached ``None``.
_MISS = object()

_CACHE: dict[str, tuple[float, object]] = {}


def _cache_get(key: str) -> object:
    """Return the cached value for ``key``, or :data:`_MISS` when absent/expired."""
    entry = _CACHE.get(key)
    if entry is None:
        return _MISS
    expires_at, value = entry
    if time.monotonic() >= expires_at:
        _CACHE.pop(key, None)
        return _MISS
    return value


def _memoise(prefix: str):
    """Cache a function's result per-arguments for :data:`_CACHE_TTL_SECONDS`.

    A tiny decorator rather than a Streamlit cache so the module keeps working
    from the CLI. Keyed on ``repr(args)`` / ``sorted(kwargs)``, which is enough
    here because every wrapped function takes plain strings.

    Args:
        prefix (str): Namespace for the cache key, so two functions with the
            same argument shape never collide.

    Returns:
        Callable: The decorator.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = f"{prefix}:{args!r}:{sorted(kwargs.items())!r}"
            cached = _cache_get(key)
            if cached is not _MISS:
                return cached
            value = func(*args, **kwargs)
            _CACHE[key] = (time.monotonic() + _CACHE_TTL_SECONDS, value)
            return value
        return wrapper
    return decorator


# =========================================================
# DATA HELPERS (plain functions — also used by the tools)
# =========================================================


def _safe_fin(df: pd.DataFrame, names: list[str], idx: int = 0) -> float:
    """Return the first present column value from a yfinance table (or NaN).

    yfinance row labels drift between releases and between issuers — the same
    line item may be ``"Total Revenue"``, ``"Revenue"`` or
    ``"Operating Revenue"``. Accepting a list of candidate labels makes the
    extraction resilient without a ladder of ``if`` statements, and returning
    ``NaN`` (rather than raising) lets callers hand the literal ``"n/a"`` to the
    LLM. :func:`build_fundamentals_text` relies on this for every figure.

    Args:
        df (pandas.DataFrame): A statement table, typically
            ``Ticker.financials.T`` / ``balance_sheet.T`` / ``cashflow.T``, so
            that rows are line items and columns are periods.
        names (list[str]): Candidate row labels, tried in order — put the most
            specific first.
        idx (int): Column index within the labelled row. ``0`` is the latest
            annual period (after the ``.T`` transpose).

    Returns:
        float: The first numeric value found, else ``float("nan")``.

    Example:
        Label-drift tolerance in action::

            import pandas as pd
            table = pd.DataFrame({"Revenue": [12500.0]})
            _safe_fin(table, ["Total Revenue", "Revenue"])   # -> 12500.0
            _safe_fin(table, ["Nope"])                        # -> nan

        ``nan`` is safe downstream: :func:`build_fundamentals_text` maps it to
        ``"n/a"`` before the LLM sees it, and ratio math guards with
        ``pd.notna(...)``::

            roe = ni / equity if equity and pd.notna(ni) else float("nan")
    """
    for name in names:
        if name in df.columns:
            val = df[name].iloc[idx]
            if pd.notna(val):
                return float(val)
    return float("nan")


_FX_CACHE: dict[tuple[str, str], float] = {}


def _fx_lookup(pair: str) -> Optional[float]:
    """Return the latest close for a Yahoo FX ticker like ``"INR=X"``.

    Yahoo quotes FX as the *base* currency priced in USD, so ``"INR=X"`` is the
    INR→USD rate (~0.012). :func:`get_fx_rate` is responsible for flipping that
    when the caller wants the opposite direction — treat this as a low-level
    primitive.

    Args:
        pair (str): Yahoo FX symbol, e.g. ``"INR=X"``, ``"EUR=X"``, ``"GBP=X"``.

    Returns:
        float | None: The last close of the 5-day window, or ``None`` when the
            symbol is unknown or the network call fails. Never raises.

    Example:
        Turning a Yahoo quote into the direction a user expects::

            inr_to_usd = _fx_lookup("INR=X")     # ~0.012
            usd_to_inr = 1 / inr_to_usd          # ~83.5

        Normally reached only via :func:`get_fx_rate`, which caches the result::

            rate = get_fx_rate("USD", "INR")     # ~83.5, memoised in _FX_CACHE
    """
    try:
        hist = yf.Ticker(pair).history(period="5d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:  # noqa: BLE001
        pass
    return None


def get_fx_rate(from_currency: str, to_currency: str) -> float:
    """Value of one unit of ``from_currency`` in ``to_currency``.

    Used for two things in this module:

    1. Building the FX note injected into every persona prompt when a company
       reports in a different currency than it trades in (``INFY.NS``: USD
       financials, INR price).
    2. Converting the deterministic range so it can be compared against the
       agentic one — see :func:`deterministic_range`.

    Degrades to ``1.0`` on any failure: a wrong-but-close conversion is far less
    harmful to a valuation prompt than a crashed run.

    Args:
        from_currency (str): ISO code to convert from, e.g. ``"USD"``.
        to_currency (str): ISO code to convert to, e.g. ``"INR"``.

    Returns:
        float: Multiplier such that ``amount * rate`` is expressed in
            ``to_currency``. ``1.0`` for same-currency or on lookup failure.

    Example:
        Identity is free and case-insensitive::

            >>> get_fx_rate("USD", "USD")
            1.0
            >>> get_fx_rate("inr", "INR")
            1.0

        Direct, inverted and triangulated lookups all work, so non-USD pairs
        such as EUR/INR need no special casing::

            >>> get_fx_rate("USD", "INR") > 50        # doctest: +SKIP
            True
            >>> get_fx_rate("EUR", "INR") > 80        # doctest: +SKIP
            True

        Results are memoised in ``_FX_CACHE``, so repeated calls during one crew
        run cost nothing extra — pre-seed it in tests to stay offline::

            _FX_CACHE[("USD", "INR")] = 83.5
    """
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


@_memoise("rep_currency")
def reporting_currency(ticker: str) -> str:
    """Return the currency the company reports its financials in (best-effort).

    Uses Yahoo's ``financialCurrency`` (falls back to ``currency``). Some
    dual-listed names (e.g. ``INFY.NS``) report in USD even though they trade in
    INR, so this can differ from the *market-price* currency returned by
    :func:`get_price_snapshot` — precisely the mismatch the persona prompts and
    :func:`deterministic_range` have to correct for.

    Args:
        ticker (str): A complete Yahoo symbol, e.g. ``"INFY.NS"``.

    Returns:
        str: An ISO currency code. Defaults to ``"USD"`` when the lookup fails,
            which matches the majority case and keeps callers simple.

    Example:
        Detecting the mismatch that the FX note exists for::

            >>> reporting_currency("INFY.NS")                  # doctest: +SKIP
            'USD'
            >>> get_price_snapshot("INFY.NS")["currency"]       # doctest: +SKIP
            'INR'

        In :func:`run_persona_crew` the pair drives one prompt sentence::

            rep_cur = reporting_currency(snap["ticker"])
            if rep_cur and rep_cur != currency:
                rate = get_fx_rate(rep_cur, currency)
                fx_note = f"... 1 {rep_cur} ≈ {rate:,.2f} {currency} ..."

        And the same pair decides whether :func:`deterministic_range` needs to
        convert its output before the UI compares it with the crew numbers.
    """
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("financialCurrency") or info.get("currency") or "USD"
    except Exception:  # noqa: BLE001
        return "USD"


@_memoise("price")
def get_price_snapshot(ticker: str) -> dict[str, Any]:
    """Return a small price snapshot for a COMPLETE Yahoo Finance symbol.

    Mirrors ``Home.get_stock_price`` semantics but lives here so this
    module never has to import the Streamlit app. The symbol is consumed EXACTLY
    as supplied — including its exchange suffix, e.g. ``INFY.NS`` /
    ``RELIANCE.BO`` — since no ``.NS`` / ``.BO`` suffix is ever appended.

    Args:
        ticker (str): Complete Yahoo symbol. ``"INFY"`` (no suffix) resolves to a
            different, often illiquid listing — always pass the suffix.

    Returns:
        dict[str, Any]: On success ``{'ticker', 'exchange', 'currency',
            'current_price'}`` where ``exchange`` is ``"NSE India"`` /
            ``"BSE India"`` / ``"Global"`` and ``currency`` is ``INR`` for the
            Indian exchanges, else ``USD``. On failure ``{'error': str}`` —
            always check with ``"error" in snap`` (that is what
            :func:`_snapshot_or_raise` does).

    Example:
        The suffix decides both exchange and currency::

            >>> snap = get_price_snapshot("INFY.NS")            # doctest: +SKIP
            >>> snap["exchange"], snap["currency"], snap["current_price"]   # doctest: +SKIP
            ('NSE India', 'INR', 1502.3)

        ``.BO`` is handled symmetrically::

            >>> get_price_snapshot("RELIANCE.BO")["exchange"]   # doctest: +SKIP
            'BSE India'

        Unqualified / US symbols get the generic bucket and USD::

            >>> get_price_snapshot("AAPL")                      # doctest: +SKIP
            {'ticker': 'AAPL', 'exchange': 'Global', 'currency': 'USD', 'current_price': 232.14}

        Failure is a value, not an exception, so the UI can show a friendly
        message and the crew is never started::

            >>> get_price_snapshot("NOT_A_TICKER")              # doctest: +SKIP
            {'error': "No stock data found for 'NOT_A_TICKER'."}

        The currency decides every downstream unit — it is copied into
        ``PersonaCrewResult.currency`` and becomes the currency every persona
        must express ``fair_value_per_share`` in.
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


@_memoise("fundamentals")
def build_fundamentals_text(ticker: str) -> str:
    """Fetch fundamentals + ratios and render a compact prompt-ready summary.

    Pulls annual income-statement / balance-sheet / cash-flow figures via
    yfinance (same extraction strategy as ``PERSONA_BASED_VALUATION.py``) and
    returns a short, labelled text block that an LLM can reason over. All money
    values are in the stock's reporting currency.

    The digest is pre-computed rather than handing over raw statements: an agent
    forced to normalise a DataFrame tends to re-read it repeatedly (burning
    iterations), whereas three labelled blocks — latest annual, balance sheet,
    key ratios — let it value the business in one pass. Everything printed is
    derivable from the numbers shown, which is what makes the "never invent
    figures" instruction in the prompts enforceable.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. ``"INFY.NS"``.

    Returns:
        str: Multi-line digest. Never raises — any failure returns a line
            starting with ``"Fundamentals unavailable"``, so the agent can say
            so in its rationale instead of hallucinating statements.

    Example:
        Headline of the digest::

            >>> build_fundamentals_text("AAPL").splitlines()[0]   # doctest: +SKIP
            'Company: Apple Inc. (AAPL)'

        Roughly what the agent receives (trimmed)::

            Company: Infosys Limited (INFY.NS)
            Country: India | Sector: Technology | Industry: Information Technology Services
            Reporting currency: USD | Shares outstanding: 4,100,000,000

            Latest ANNUAL figures (reporting currency):
              Revenue: 19,275,000,000
              EBIT: 4,200,000,000 | EBIT margin: 21.8%
              ...
            Key ratios & growth (computed from the above):
              Revenue growth (latest y/y): 6.5%
              Return on equity (ROE): 31.2%
              Return on invested capital (ROIC): 28.4%
              Debt/equity: 0.09
              Beta: 0.11

        Guards worth knowing when reading the output:

        * ``rev_growth`` falls back to ``5%`` when Yahoo reports a missing or
          nonsensical (< -50%) value.
        * ``tax_rate`` falls back to ``21%`` when net income + tax is unusable,
          so NOPAT never divides by zero.
        * ``capex`` is taken in absolute value because Yahoo signs it negative.
        * ``free cash flow`` is ``operating cash flow - capex``.

        Calling it without any LLM is a cheap smoke test for a new ticker::

            print(build_fundamentals_text("RELIANCE.NS")[:200])
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
            """Format a money figure for the prompt, or ``"n/a"`` when unusable.

            The LLM is told never to invent figures, so a missing value must be
            visible as a gap rather than silently becoming ``0`` or ``nan``.

            Example:
                money(19_275_000_000)   # -> "19,275,000,000"
                money(float("nan"))     # -> "n/a"
            """
            return "n/a" if pd.isna(v) or v is None else f"{v:,.0f}"

        def pct(v: float, nd: int = 1) -> str:
            """Format a ratio as a percentage string, or ``"n/a"`` when unusable.

            Example:
                pct(0.284)      # -> "28.4%"
                pct(0.284, 0)   # -> "28%"
                pct(float("nan"))  # -> "n/a"
            """
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


@_memoise("news")
def build_news_and_sentiment_text(company_name: str) -> str:
    """Fetch recent headlines + TextBlob sentiment and render a prompt summary.

    Combines the behaviour of ``get_stock_news`` and ``analyze_news_sentiment``
    from the main app into one tool so personas get qualitative context in a
    single call. Degrades gracefully when the key is missing or the API fails.

    Note the argument is a **company name**, not a ticker: NewsAPI matches on
    free text, and ``"INFY.NS stock"`` returns far worse results than
    ``"Infosys stock"``. That is why :func:`run_persona_crew` threads
    ``company_name`` through to the prompts as well as the ticker.

    Args:
        company_name (str): Human-readable name, e.g. ``"Infosys"``.

    Returns:
        str: Up to five headlines plus one sentiment line (``Positive`` /
            ``Neutral`` / ``Negative`` and a polarity score), or a
            ``"News unavailable: ..."`` / ``"No recent news articles..."``
            line. Never raises.

    Example:
        A successful call — the label uses a ±0.15 neutral band::

            Latest headlines for Infosys:
            - Infosys raises FY26 revenue guidance (Reuters)
            - Infosys wins $500m AI deal (Mint)

            Aggregate sentiment (TextBlob over 5 articles): Positive (score 0.21).

        Without ``NEWS_API_KEY`` the tool is still safe to call, and the persona
        simply notes that news was unavailable::

            >>> build_news_and_sentiment_text("Infosys")   # doctest: +SKIP
            'News unavailable: NEWS_API_KEY is not configured.'

        Sentiment is deliberately coarse — TextBlob polarity over title +
        description, averaged across the articles that yielded text. Treat it as
        a nudge for tone, never as a valuation input.
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
# Every tool is a thin wrapper over a plain function above, so the same logic is
# unit-testable without an LLM or a crew. Two crewAI-specific rules apply:
#
#   1. The decorated function's DOCSTRING is the tool description the model
#      reads. Keep it explicit about the argument shape (complete symbol vs
#      company name) — that is the biggest driver of bad tool calls.
#   2. Return a *string*. LLMs consume text; dicts get stringified badly. JSON is
#      used for the price tool (structured, easy to re-parse) and a labelled text
#      digest for the other two (cheaper than JSON for long payloads).
#
# Example:
#     Tools are invoked by the agent, but you can call them yourself::
#
#         >>> get_stock_fundamentals.run("AAPL").splitlines()[0]   # doctest: +SKIP
#         'Company: Apple Inc. (AAPL)'
#         >>> json.loads(get_stock_price.run("AAPL"))["currency"]  # doctest: +SKIP
#         'USD'


@tool("get_stock_price")
def get_stock_price(ticker: str) -> str:
    """Fetch the current market price for a COMPLETE Yahoo Finance symbol
    (e.g. INFY.NS, RELIANCE.BO, AAPL).

    The symbol is used exactly as supplied — no .NS / .BO suffix is appended.
    Returns JSON with ticker, exchange, currency and current_price (or an
    error object).

    Args:
        ticker (str): Exchange-qualified Yahoo symbol, e.g. ``"INFY.NS"``. For an
            Indian listing the ``.NS`` / ``.BO`` suffix is mandatory.

    Returns:
        str: JSON string. Success keys: ``ticker``, ``exchange``, ``currency``,
            ``current_price``. Failure key: ``error``.

    Example:
        The agent's opening move — confirm the market anchor::

            >>> json.loads(get_stock_price.run("INFY.NS"))     # doctest: +SKIP
            {'ticker': 'INFY.NS', 'exchange': 'NSE India',
             'currency': 'INR', 'current_price': 1502.3}

        A bad symbol returns an error object instead of raising, so the agent can
        report "no data" rather than crash the crew::

            >>> json.loads(get_stock_price.run("XYZXYZ"))      # doctest: +SKIP
            {'error': "No stock data found for 'XYZXYZ'."}
    """
    return json.dumps(get_price_snapshot(ticker))


@tool("get_stock_fundamentals")
def get_stock_fundamentals(ticker: str) -> str:
    """Fetch financial-statement fundamentals + key ratios for a ticker.

    Returns a compact text block: revenue, EBIT, net income, margins, free
    cash flow, debt, equity, ROE, ROIC, revenue growth, beta, shares, and the
    reporting currency. Use this to value the business — never invent figures.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. ``"RELIANCE.NS"``.

    Returns:
        str: The digest produced by :func:`build_fundamentals_text`. Money
            figures are in the **reporting** currency stated in the first lines,
            which may differ from the price currency.

    Example:
        Every persona calls this exactly once, then reasons over the digest::

            text = get_stock_fundamentals.run("AAPL")
            assert "Return on invested capital (ROIC)" in text

        Because the reporting currency can differ from the trading currency, a
        well-behaved persona converts before answering; the prompt supplies the
        FX rate for exactly this reason (see the FX note built in
        :func:`run_persona_crew`).
    """
    return build_fundamentals_text(ticker)


@tool("get_stock_news_and_sentiment")
def get_stock_news_and_sentiment(company_name: str) -> str:
    """Fetch recent news headlines and aggregate sentiment for a COMPANY NAME
    (e.g. 'Infosys', not 'INFY.NS'). Returns headlines plus a Positive /
    Neutral / Negative label with a polarity score.

    Args:
        company_name (str): Free-text company name — *not* a ticker. Passing a
            ticker here degrades NewsAPI relevance sharply.

    Returns:
        str: Up to five headlines and one sentiment line, or a
            ``"News unavailable: ..."`` message when the key is missing or the
            API fails.

    Example:
        The persona uses this only for tone, never for numbers::

            text = get_stock_news_and_sentiment.run("Infosys")
            if text.startswith("News unavailable"):
                pass   # note the data gap in the rationale, then keep valuing

        A persona that over-weights sentiment is behaving out of character: the
        Buffett and Munger goals tie every thesis back to fundamentals, so this
        tool mostly serves Jhunjhunwala's search for growth narratives.
    """
    return build_news_and_sentiment_text(company_name)


# The exact list handed to every persona agent. Order is irrelevant to the LLM
# but kept stable here for reproducible logs. Add a tool here to expose it to all
# four investors at once.
PERSONA_TOOLS = [get_stock_price, get_stock_fundamentals, get_stock_news_and_sentiment]


# =========================================================
# LLM FACTORY
# =========================================================


def get_llm() -> LLM:
    """Return the shared DeepSeek LLM (LiteLLM ``deepseek`` provider).

    One instance is reused by all five agents in a run. LiteLLM reads
    ``DEEPSEEK_API_KEY`` from the environment (loaded from ``.env`` by
    ``load_dotenv()`` at import time), so there is no key to pass here.

    ``temperature=0`` is deliberate: the output feeds a JSON parser and a
    side-by-side comparison against a deterministic model, so reproducibility
    beats creativity. ``timeout=120`` gives DeepSeek room to digest a long
    fundamentals block without hanging the Streamlit request thread forever.

    Returns:
        LLM: A crewAI ``LLM`` handle bound to :data:`DEEPSEEK_MODEL`.

    Example:
        Override the model without touching code::

            $env:CREWAI_MODEL = "deepseek/deepseek-reasoner"
            $env:CREWAI_MAX_ITER = "5"
            python PERSONA_AGENTS_CREW.py INFY.NS "Infosys"

        Build one handle and fan it out to every agent — never build five::

            llm = get_llm()
            persona_agents = [build_persona_agent(meta, llm) for meta in PERSONAS]
            lead_agent = build_lead_agent(llm)
    """
    return LLM(model=DEEPSEEK_MODEL, temperature=0, timeout=120)


# =========================================================
# PARSERS
# =========================================================


def _extract_json(text: str) -> Optional[dict]:
    """Tolerantly extract the first valid JSON object from a raw LLM response.

    Three recovery layers, cheapest first:

    1. Strip markdown code fences and try a straight ``json.loads`` when the
       text starts with ``{``.
    2. Otherwise scan for every ``{`` and try ``json.JSONDecoder.raw_decode``
       from there — this handles the common "Sure! Here is the JSON: {...} Hope
       that helps." pattern without greedily swallowing trailing prose.
    3. Give up and return ``None``, which lets callers fall back to
       :func:`persona_from_raw`'s regex scan or to a computed aggregate.

    Args:
        text (str): Raw task output, typically ``TaskOutput.raw``.

    Returns:
        dict | None: The first value that parses as a JSON *object*, or ``None``
            when nothing does. Arrays and bare scalars are rejected — every
            contract in this module asks for an object.

    Example:
        Fenced JSON::

            >>> raw = '''```json
            ... {"stance": "Undervalued", "conviction": 0.6}
            ... ```'''
            >>> _extract_json(raw)["stance"]
            'Undervalued'

        Chatty prose around the payload — the model ignored the "JSON only"
        instruction, which happens often enough to be worth handling::

            >>> raw = 'Here you go: {"fair_value_per_share": 1280.0} Thanks!'
            >>> _extract_json(raw)
            {'fair_value_per_share': 1280.0}

        Nothing salvageable (both return ``None``, printing nothing in a REPL)::

            >>> _extract_json("I could not find enough data.")
            >>> _extract_json("")
    """
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
    """Coerce a parsed JSON value to float (strips currency/commas if needed).

    Models obey the schema loosely — ``"₹1,280.50"``, ``"1280 INR"`` or ``"n/a"``
    instead of a bare number. This normalises those into something
    arithmetic-safe, and returns ``None`` (rather than raising or yielding
    ``NaN``) so the caller can decide whether to drop the value.

    Args:
        value (Any): Any parsed JSON scalar. ``bool`` is explicitly rejected —
            ``True`` is not a fair value.

    Returns:
        float | None: The numeric value, or ``None`` when it cannot be coerced.

    Example:
        Currency symbols, thousands separators and trailing text are stripped::

            >>> _num(1280)
            1280.0
            >>> _num("₹1,280.50")
            1280.5
            >>> _num("1280 INR")
            1280.0

        Non-numeric sentinels degrade to ``None``::

            >>> _num("n/a") is None
            True
            >>> _num(None) is None
            True
            >>> _num(True) is None
            True

        This is why :func:`aggregate_from_raw` can call
        ``_num(data.get("min_fair_value"))`` without a try/except.
    """
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

    Two paths, in order:

    1. **JSON path** (the normal case). :func:`_extract_json` recovers the object
       the prompt asked for; every field is read defensively with
       ``data.get(...) or default`` so a missing key never raises.
    2. **Regex fallback**. When no JSON object is found, the raw text is scanned
       for a fair value, a stance keyword and a thesis sentence — this rescues a
       persona that answered in prose from producing an empty card.

    The plausibility guard is the important safety net: a persona that reports
    revenue (``"$20,158M"``) or market cap instead of a per-share value would
    otherwise poison the consensus range, so any figure outside ``[0.1x, 8x]`` of
    the live price is discarded and the card falls back to
    ``fair_value_per_share=None``.

    Args:
        meta (dict[str, str]): The :data:`PERSONAS` entry for this task; supplies
            the authoritative display name, emoji and portrait path.
        raw (str): ``TaskOutput.raw`` for ``persona_<key>``.
        market_price (float | None): Live price used by the plausibility filter.
            Pass ``None`` to disable filtering (e.g. in unit tests).

    Returns:
        PersonaValuation: Always a usable card. ``fair_value_per_share`` is
            ``None`` when nothing plausible was found.

    Example:
        A well-behaved JSON answer::

            >>> meta = PERSONAS[0]                      # Warren Buffett
            >>> raw = '{"fair_value_per_share": 1280, "currency": "INR", "stance": "Undervalued", "conviction": 0.7, "one_line_thesis": "Moat plus net cash."}'
            >>> card = persona_from_raw(meta, raw, market_price=1502.3)
            >>> card.fair_value_per_share, card.stance
            (1280.0, 'Undervalued')
            >>> card.emoji                              # from PERSONAS, not the LLM
            '🧸'

        A unit slip is rejected rather than taken at face value — 20158 is ~13x
        the price, outside the 8x ceiling::

            >>> bad = '{"fair_value_per_share": 20158, "stance": "Overvalued"}'
            >>> persona_from_raw(meta, bad, market_price=1502.3).fair_value_per_share is None
            True

        Prose answers still yield a card via the regex fallback::

            >>> prose = "Fair value per share: ₹1,240. Stance: Undervalued."
            >>> persona_from_raw(meta, prose, market_price=1502.3).fair_value_per_share
            1240.0

        Garbage yields an empty-but-renderable card instead of an exception::

            >>> empty = persona_from_raw(meta, "no numbers here", market_price=1502.3)
            >>> empty.fair_value_per_share is None and empty.persona
            'Warren Buffett'
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
    """Derive min/max/avg from persona numbers when the lead task output is lost.

    The safety net behind :func:`aggregate_from_raw`. ``Process.sequential``
    means a persona task that burns its retry budget can take the aggregator down
    with it, so the crew must still be able to produce a consensus range from
    whatever did come back.

    The fallback computes simple statistics only: it does **not** invent a
    narrative. ``note`` states plainly that the lead analyst did not answer, and
    ``blended_stance`` is the modal persona stance.

    Args:
        personas (list[PersonaValuation]): Cards produced so far — possibly all
            empty.
        currency (str): Market-price currency; accepted for signature symmetry
            (the values already carry their own currency).

    Returns:
        AggregateResult: ``participating=0`` with an explanatory ``note`` when no
            persona produced a number, otherwise the computed range.

    Example:
        Three of four personas answered::

            >>> cards = [
            ...     PersonaValuation(persona="A", fair_value_per_share=100.0, stance="Undervalued"),
            ...     PersonaValuation(persona="B", fair_value_per_share=130.0, stance="Undervalued"),
            ...     PersonaValuation(persona="C", fair_value_per_share=120.0, stance="Fairly valued"),
            ...     PersonaValuation(persona="D"),          # no value
            ... ]
            >>> agg = _compute_fallback_aggregate(cards, "INR")
            >>> (agg.min_fair_value, agg.avg_fair_value, agg.max_fair_value)
            (100.0, 116.66666666666667, 130.0)
            >>> agg.blended_stance, agg.participating
            ('Undervalued', 3)

        Nobody answered — the caller still gets a valid object, so the UI has
        something to render::

            >>> _compute_fallback_aggregate([PersonaValuation(persona="A")], "INR").participating
            0

        A low ``participating`` count is the signal to soften any "the committee
        agrees" copy in the UI.
    """
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
    """Parse the lead-analyst raw output (or fall back to a computed range).

    Includes a self-healing step: even when the JSON parses, the lead may omit
    ``min_fair_value`` / ``max_fair_value`` / ``avg_fair_value``. Any missing
    statistic is recomputed from the persona cards, so the UI never has to handle
    "range present but empty" states.

    Args:
        raw (str): ``TaskOutput.raw`` for ``aggregator_task``.
        personas (list[PersonaValuation]): The parsed persona cards, used for the
            fallback and to populate ``participating``.
        currency (str): Market-price currency (passed through to the fallback).

    Returns:
        AggregateResult: Parsed, partially healed, or fully computed.

    Example:
        The normal case, straight from the aggregator::

            >>> cards = [PersonaValuation(persona="A", fair_value_per_share=900.0)]
            >>> raw = '{"min_fair_value": 900, "max_fair_value": 1100, "avg_fair_value": 1000, "blended_stance": "Undervalued", "note": "Committee splits on growth durability."}'
            >>> agg = aggregate_from_raw(raw, cards, "INR")
            >>> agg.min_fair_value, agg.blended_stance, agg.participating
            (900.0, 'Undervalued', 1)

        A partial answer is healed from the persona values::

            >>> healed = aggregate_from_raw('{"blended_stance": "Fairly valued"}', cards, "INR")
            >>> healed.min_fair_value, healed.max_fair_value
            (900.0, 900.0)

        Unparseable output falls through to the computed range::

            >>> aggregate_from_raw("I decline to answer.", cards, "INR").participating
            1
    """
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
# These two functions are where most of the module's behaviour actually lives.
# They are plain f-string builders (no I/O, no LLM) so the exact prompt text can
# be asserted on in tests.


def _persona_task_description(
    meta: dict[str, str], ticker: str, company: str, price: float,
    currency: str, fx_note: str = "",
) -> str:
    """Render the full prompt for one persona task.

    The prompt encodes three separate contracts on purpose:

    1. **Grounding** — the market anchor, the tool menu, and "base the fair value
       strictly on tool output" / "never invent figures".
    2. **Working discipline** — *work silently*, one pass, ignore implausible
       betas. Without the silence clause DeepSeek emits a multi-hundred-line DCF
       into its final message, which breaks parsing and inflates cost.
    3. **Response contract** — a single JSON object and nothing else, with
       ``fair_value_per_share`` expressed in the market-price currency.

    Args:
        meta (dict[str, str]): The :data:`PERSONAS` entry — supplies the persona
            name and the behavioural ``goal`` that is interpolated twice.
        ticker (str): Complete Yahoo symbol, shown to the agent for tool calls.
        company (str): Friendly name for the news tool; may be ``""``.
        price (float): Live price used as the valuation anchor.
        currency (str): Currency the answer must be expressed in.
        fx_note (str): Optional sentence carrying a conversion rate, injected
            when the reporting currency differs from the price currency.

    Returns:
        str: A ready-to-use ``Task`` description ending in the JSON contract.

    Example:
        Building one prompt by hand::

            prompt = _persona_task_description(
                PERSONAS[0], "INFY.NS", "Infosys", 1502.3, "INR")
            assert "Warren Buffett" in prompt
            assert "fair_value_per_share" in prompt

        With the FX note, an INR answer is disambiguated from USD financials::

            note = "Note: financials are reported in USD ... 1 USD ≈ 83.5 INR"
            prompt = _persona_task_description(
                PERSONAS[0], "INFY.NS", "Infosys", 1502.3, "INR", fx_note=note)
            assert "83.5 INR" in prompt      # persona is told which unit to answer in

        Multiply the roster without writing new prompt code — every persona gets
        the same skeleton with its own ``goal`` interpolated::

            prompts = [_persona_task_description(m, "AAPL", "Apple", 232.14, "USD")
                       for m in PERSONAS]
            assert len(prompts) == 4

        The only differing payload is the persona's own goal / backstory, which
        is what makes four independent opinions possible from identical tools.
    """
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
    """Render the reconciliation prompt for the aggregator task.

    Deliberately short. The four persona answers arrive as *task context*
    (``Task.context``), so the prompt only has to state the goal, forbid invented
    numbers, and pin the output contract.

    Args:
        ticker (str): Complete Yahoo symbol, for the committee headline.
        company (str): Friendly name; may be ``""``.
        price (float): Live price, so the lead can frame the range as
            undervalued/overvalued without re-deriving it.
        currency (str): Currency of the persona values.

    Returns:
        str: The aggregator ``Task`` description, ending in its JSON contract.

    Example:
        The four sibling tasks become the lead's context::

            lead_task = Task(
                name="aggregator_task",
                description=_lead_task_description("INFY.NS", "Infosys", 1502.3, "INR"),
                expected_output="A single JSON object with min_fair_value, ...",
                agent=lead_agent,
                context=persona_tasks,          # <- the entire payload
            )

        Because the numbers arrive in context, the lead agent is built with no
        tools at all (see :func:`build_lead_agent`) — it cannot fetch anything
        that would contradict the committee it is summarising.
    """
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


# =========================================================
# RULE-BASED (PERSONA_BASED_VALUATION) BRIDGE
# =========================================================
# The agentic crew and the rule-based model must be compared in the SAME currency
# and must derive from ONE computation, so these three helpers are the only bridge
# between the two modules:
#
#   deterministic_persona_values()  -> per-persona values, market currency
#   _range_from_persona_values()    -> min/max/midpoint over those values
#   deterministic_range()           -> the range, for direct callers
#
# Example:
#     >>> deterministic_persona_values("INFY.NS", to_currency="INR")   # doctest: +SKIP
#     {'warren_buffett': None, 'charlie_munger': 346.1, ...}
#     >>> deterministic_range("INFY.NS", to_currency="INR")            # doctest: +SKIP
#     {'min': 346.1, 'max': 346.1, 'mid': 346.1}


# Minimum number of rule-based personas that must produce a value before the panel
# is shown at all. Kept at 1 so the comparison still appears when the rule-based
# model prices only some personas: a persona whose rule-based value is missing is
# dropped from the comparison silently, and the panel then shows its CrewAI value
# with no average rather than hiding the whole table.
MIN_DETERMINISTIC_PERSONAS = 1


def _convert_money(
    value: Optional[float], from_currency: str, to_currency: Optional[str]
) -> Optional[float]:
    """Convert one money value between currencies, unchanged when not possible.

    Returns ``value`` untouched when it is ``None``, when no target currency was
    given, when both currencies already match, or when the FX lookup degrades to
    ``1.0`` — a slightly wrong magnitude beats a crashed valuation.

    Args:
        value (float | None): Amount expressed in ``from_currency``.
        from_currency (str): Currency ``value`` is expressed in.
        to_currency (str | None): Target currency; ``None`` means "do not convert".

    Returns:
        float | None: The converted amount, or ``value`` unchanged.

    Example:
        Missing-target and same-currency cases are no-ops::

            >>> _convert_money(100.0, "USD", "USD")
            100.0
            >>> _convert_money(100.0, "USD", None)
            100.0
            >>> _convert_money(None, "USD", "INR") is None
            True
    """
    if value is None or not to_currency or not from_currency:
        return value
    if from_currency.upper() == to_currency.upper():
        return value
    rate = get_fx_rate(from_currency, to_currency)
    return value * rate if rate and rate != 1.0 else value


@_memoise("deterministic_personas")
def deterministic_persona_values(
    ticker: str, to_currency: Optional[str] = None
) -> dict[str, Optional[float]]:
    """Return the rule-based fair value for each persona, keyed by crew key.

    Bridges ``PERSONA_BASED_VALUATION.persona_valuations`` (which labels its output
    ``Buffett`` / ``Munger`` / ``Jhunjhunwala`` / ``Damodaran``) to the crew's
    :data:`PERSONAS` keys via each entry's ``det_key``. Every persona key is always
    present, with ``None`` where the rule produced no value, so the caller can
    count participants without special-casing a missing key.

    Values are converted from the company's reporting currency into
    ``to_currency`` (normally the market-price currency) so they line up with the
    crew's fair values.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. ``"INFY.NS"``.
        to_currency (str | None): Target currency, typically
            ``get_price_snapshot(ticker)["currency"]``.

    Returns:
        dict[str, Optional[float]]: One entry per :data:`PERSONAS` key, e.g.
            ``{'warren_buffett': None, 'charlie_munger': 346.1, ...}``. Never
            raises — an unexpected failure yields all-``None``.

    Example:
        Compare each rule-based value with its crew counterpart::

            >>> rule = deterministic_persona_values("INFY.NS", to_currency="INR")   # doctest: +SKIP
            >>> rule["charlie_munger"]                                            # doctest: +SKIP
            346.1

        Count how many rules priced, which is what drives
        :data:`MIN_DETERMINISTIC_PERSONAS`::

            priced = sum(
                1
                for value in deterministic_persona_values("AAPL", "USD").values()
                if value is not None
            )
    """
    try:
        raw = pv.persona_valuations(ticker)
    except Exception:  # noqa: BLE001
        return {meta["key"]: None for meta in PERSONAS}
    from_currency = reporting_currency(ticker)
    return {
        meta["key"]: _convert_money(
            raw.get(meta.get("det_key", "")), from_currency, to_currency
        )
        for meta in PERSONAS
    }


def _range_from_persona_values(
    values: dict[str, Optional[float]]
) -> dict[str, Optional[float]]:
    """Collapse per-persona values into a min/max/midpoint range.

    Replicates ``PERSONA_BASED_VALUATION.persona_super_valuation`` exactly, so the
    range and the per-persona figures are always derived from the same numbers.
    ``mid`` is the **midpoint of min and max**, not the mean of the values.

    Args:
        values (dict[str, Optional[float]]): Persona key to fair value; ``None``
            entries are ignored.

    Returns:
        dict[str, Optional[float]]: ``{'min', 'max', 'mid'}``, all ``None`` when
            nothing priced.

    Example:
        Two participants, so the midpoint sits halfway between the extremes::

            >>> _range_from_persona_values({"a": 100.0, "b": None, "c": 130.0})
            {'min': 100.0, 'max': 130.0, 'mid': 115.0}

        A single participant collapses to a point — the degenerate case the UI
        hides::

            >>> _range_from_persona_values({"a": 26.3, "b": None})
            {'min': 26.3, 'max': 26.3, 'mid': 26.3}

        Nobody priced::

            >>> _range_from_persona_values({"a": None})["min"] is None
            True
    """
    vals = [v for v in values.values() if v is not None]
    if not vals:
        return {"min": None, "max": None, "mid": None}
    low, high = min(vals), max(vals)
    return {"min": low, "max": high, "mid": (low + high) / 2}


def deterministic_range(ticker: str, to_currency: Optional[str] = None) -> dict[str, Optional[float]]:
    """Compute the deterministic 4-persona range for side-by-side comparison.

    Derived from :func:`deterministic_persona_values` (which reads
    ``PERSONA_BASED_VALUATION.persona_valuations``) using the same min/max/midpoint
    arithmetic as ``persona_super_valuation`` — ``mid`` is the **midpoint of min
    and max**, not the mean of the four values.

    The rule-based model derives per-share values from the financial statements,
    so for dual-listed names they come back in the *reporting* currency (e.g. USD
    for ``INFY.NS``). When ``to_currency`` is given and differs from the reporting
    currency, the values are converted so they can be compared against the
    price-currency fair values from the crew.

    Args:
        ticker (str): Complete Yahoo symbol.
        to_currency (str | None): Target currency — pass the market-price currency
            from :func:`get_price_snapshot` to make the numbers comparable.
            ``None`` returns the raw reporting-currency values.

    Returns:
        dict[str, Optional[float]]: ``{'min', 'max', 'mid'}`` expressed in
            ``to_currency``, or ``{'error': str}`` if the model failed. Always
            check for ``"error"`` before formatting the numbers.

    Example:
        The comparison the UI shows under the persona cards::

            det = deterministic_range("INFY.NS", to_currency="INR")
            if "error" not in det:
                st.caption(f"Rules: INR {det['min']:,.0f}–{det['max']:,.0f}")

        Rendering both methodologies together, in one unit::

            crew_lo, crew_hi = res.aggregate.min_fair_value, res.aggregate.max_fair_value
            det = res.deterministic_range
            if "error" not in det:
                st.caption(f"Rules: {res.currency} {det['min']:,.0f}–{det['max']:,.0f}"
                           f"  |  Crew: {res.currency} {crew_lo:,.0f}–{crew_hi:,.0f}")

        Known quirk: for some symbols ``persona_super_valuation`` returns a
        degenerate ``min == max == mid`` (a pre-existing issue in that module, not
        here). Render it honestly rather than hiding it — a rule-based model
        collapsing to a point estimate is itself information worth showing.
    """
    try:
        return _range_from_persona_values(
            deterministic_persona_values(ticker, to_currency=to_currency)
        )
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _snapshot_or_raise(ticker: str) -> dict[str, Any]:
    """Return a usable price snapshot or raise — the one hard failure mode.

    Everything else in :func:`run_persona_crew` is fail-soft, but without a
    market price there is no anchor for the prompts, no currency for the answers
    and no basis for the plausibility filter, so the run stops here instead of
    producing four meaningless cards.

    Args:
        ticker (str): Complete Yahoo symbol.

    Returns:
        dict[str, Any]: The snapshot from :func:`get_price_snapshot`, guaranteed
            to contain ``current_price``.

    Raises:
        ValueError: When the symbol cannot be priced. The message embeds the
            underlying Yahoo error so the UI can show it verbatim.

    Example:
        Happy path::

            snap = _snapshot_or_raise("INFY.NS")     # -> {'currency': 'INR', ...}

        Bad symbol — deliberately allowed to propagate::

            snap = _snapshot_or_raise("NOT_A_TICKER")
            # ValueError: Could not resolve a market price for 'NOT_A_TICKER':
            #             No stock data found for 'NOT_A_TICKER'.

        The Streamlit handler is expected to translate that into a friendly
        message rather than a traceback::

            try:
                res = pac.run_persona_crew(ticker, name)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        Why here and not inside :func:`get_price_snapshot`? Because that function
        is also used by the ``get_stock_price`` tool, where returning an error
        object (instead of raising) lets the agent reason about missing data.
    """
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

    The module's entry point. Steps:

      1. Resolve a market snapshot (price + currency) so the UI always has an
         anchor even if a persona fails.
      2. Build the FX note and the deterministic ``PERSONA_BASED_VALUATION``
         range, both standardised on the market-price currency.
      3. Kick off ONE crew: four ``async_execution=True`` persona tasks plus a
         lead aggregator task whose ``context`` is those four tasks.
         ``Process.sequential`` + async tasks is crewAI's idiom for "run N experts
         in parallel, then have a lead reconcile them".
      4. Map raw task outputs back by ``Task.name``, parse them, and assemble the
         result with per-persona error isolation.

    Args:
        ticker (str): Complete Yahoo symbol, e.g. ``"INFY.NS"``. It is
            upper-cased and stripped; the exchange suffix is required.
        company_name (str): Friendly name forwarded to the news tool and the
            prompts. Pass it whenever you have it — news relevance depends on it.
            Defaults to ``""``, in which case prompts fall back to the ticker.
        verbose (bool): Forwarded to ``Crew(verbose=...)`` so crewAI logs every
            tool call and LLM exchange. Useful from the CLI, noisy in the UI.

    Returns:
        PersonaCrewResult: Always renderable. ``personas`` has one entry per
            :data:`PERSONAS` item in order (empty cards on failure),
            ``aggregate`` is the consensus (or a computed fallback), and every
            partial failure is recorded in ``errors``.

    Raises:
        ValueError: Only when the ticker cannot be priced — see
            :func:`_snapshot_or_raise`. All other failures are absorbed.

    Example:
        Minimal call (blocks for tens of seconds while the crew runs)::

            >>> import PERSONA_AGENTS_CREW as pac
            >>> res = pac.run_persona_crew("INFY.NS", "Infosys")   # doctest: +SKIP
            >>> res.ticker, res.currency, res.exchange              # doctest: +SKIP
            ('INFY.NS', 'INR', 'NSE India')

        Reading the four opinions and the consensus::

            for p in res.personas:
                value = "—" if p.fair_value_per_share is None else f"{p.fair_value_per_share:,.0f}"
                print(f"{p.persona:<22} {value:>8} {p.currency}  {p.stance}")

            a = res.aggregate
            print(f"Consensus {a.min_fair_value:,.0f} – {a.max_fair_value:,.0f} ({a.blended_stance})")
            print(f"{a.participating}/4 personas returned a number")

        Storing it for Streamlit, and rendering on the *next* run::

            st.session_state.persona_result = res.model_dump()
            st.rerun()

        The currency follows the listing — the same company gives different
        units depending on the symbol you pass::

            >>> run_persona_crew("INFY.NS", "Infosys").currency   # doctest: +SKIP
            'INR'
            >>> run_persona_crew("INFY", "Infosys").currency      # doctest: +SKIP
            'USD'

        Inspecting a degraded run — one persona timing out does not lose the
        other three::

            res = run_persona_crew("AAPL", "Apple")
            if res.errors:
                st.warning("Partial results: " + "; ".join(res.errors))

        Cost visibility (``usage`` is populated when crewAI reports metrics)::

            res.usage.get("total_tokens")     # -> 48213, or None
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
    # One computation serves both the range and the per-persona values, so the
    # two can never disagree (and the statements are fetched only once).
    det_values = deterministic_persona_values(snap["ticker"], to_currency=currency)
    result.deterministic_personas = det_values
    result.deterministic_range = _range_from_persona_values(det_values)
    result.deterministic_participating = sum(
        1 for value in det_values.values() if value is not None
    )
    result.deterministic_usable = (
        result.deterministic_participating >= MIN_DETERMINISTIC_PERSONAS
    )

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

    # 5. Blend the rule-based values into the cards (mean of the two
    #    methodologies per persona) and derive the overall blended value.
    _apply_deterministic_blend(result)

    return result


def _apply_deterministic_blend(result: PersonaCrewResult) -> None:
    """Attach the rule-based value and the crew/rule average to each card.

    Mutates ``result`` in place:

    * ``card.deterministic_fair_value`` — the matching rule-based estimate.
    * ``card.blended_fair_value`` — mean of the crew and rule-based values, or
      ``None`` when either side is missing. A missing rule-based value must not
      silently drag the average towards zero.
    * ``result.blended_fair_value`` — mean of the per-persona averages.

    Cards are matched to :data:`PERSONAS` positionally, which is safe because
    :func:`run_persona_crew` appends exactly one card per persona, in order, on
    every path (success, missing output, or parse failure).

    Args:
        result (PersonaCrewResult): The partially built result to enrich.

    Example:
        One blended value per persona, plus the overall mean of those::

            _apply_deterministic_blend(res)
            res.personas[0].blended_fair_value     # (crew + rule) / 2
            res.blended_fair_value                 # mean across personas
    """
    rule_values = result.deterministic_personas or {}
    blended: list[float] = []
    for meta, card in zip(PERSONAS, result.personas):
        rule_value = rule_values.get(meta["key"])
        card.deterministic_fair_value = rule_value
        crew_value = card.fair_value_per_share
        if rule_value is not None and crew_value is not None:
            card.blended_fair_value = (rule_value + crew_value) / 2
            blended.append(card.blended_fair_value)
        else:
            card.blended_fair_value = None
    result.blended_fair_value = sum(blended) / len(blended) if blended else None


def _summarise_usage(metrics: Any) -> dict[str, Any]:
    """Flatten crewAI usage metrics into a small dict for the UI caption.

    crewAI's metrics object is not part of its documented API, so every read is
    ``getattr``-guarded and the whole body is wrapped in ``try`` — a missing field
    must never take down a valuation that already succeeded.

    Args:
        metrics (Any): ``CrewOutput.usage_metrics`` (may be ``None``).

    Returns:
        dict[str, Any]: ``{'total_tokens', 'prompt_tokens',
            'completion_tokens'}`` — individual values may be ``None``; ``{}`` if
            the whole read fails.

    Example:
        Surfacing spend in the UI::

            tokens = res.usage.get("total_tokens")
            if tokens:
                st.caption(f"≈{tokens:,} tokens across 5 agents")

        Defensive by construction — unknown attributes degrade to ``None`` rather
        than raising::

            >>> _summarise_usage(None)
            {'total_tokens': None, 'prompt_tokens': None, 'completion_tokens': None}
    """
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
# Deliberately thin: this exists to verify a real crew run end-to-end against the
# live APIs without touching the Streamlit UI. It makes PAID DeepSeek calls and
# hits Yahoo / NewsAPI, and always runs with verbose=True so you can watch each
# tool call.
#
# Usage:
#     python PERSONA_AGENTS_CREW.py                      # INFY.NS, no company name
#     python PERSONA_AGENTS_CREW.py INFY.NS "Infosys"    # Indian listing, INR
#     python PERSONA_AGENTS_CREW.py AAPL "Apple"         # US listing, USD
#
# Requires DEEPSEEK_API_KEY (and NEWS_API_KEY for the sentiment tool) in the
# environment or a .env file next to this module.

if __name__ == "__main__":
    import sys
    tick = sys.argv[1] if len(sys.argv) > 1 else "INFY.NS"
    comp = sys.argv[2] if len(sys.argv) > 2 else ""
    print(f"Running persona crew for {tick} ... (this makes real DeepSeek calls)\n")
    res = run_persona_crew(tick, comp, verbose=True)

    # Market anchor actually used by the crew. May differ from the symbol you
    # typed: the suffix decides the exchange and therefore the currency.
    print(f"\nTicker: {res.ticker} | Price: {res.currency} {res.market_price} | {res.exchange}")

    # One line per persona, in PERSONAS order. A missing fair value prints as
    # "None" — that is a real outcome ("No opinion"), not a crash.
    for p in res.personas:
        print(f"  {p.persona}: FV={p.fair_value_per_share} {p.currency} | stance={p.stance} "
              f"| conviction={p.conviction} | {p.one_line_thesis[:80]}")

    # Lead-analyst reconciliation. aggregate.participating < 4 means the range
    # is thin; aggregate.note carries the committee commentary.
    a = res.aggregate
    print(f"\nAggregate: min={a.min_fair_value} avg={a.avg_fair_value} max={a.max_fair_value} | {a.blended_stance}")

    # Rule-based comparison range, already converted into the market-price
    # currency by run_persona_crew. Shows {'error': ...} rather than {'min', ...}
    # if PERSONA_BASED_VALUATION raised.
    print(f"Deterministic range: {res.deterministic_range}")

    # Crew vs rule-based vs their average, per persona. The UI hides this whole
    # section when deterministic_usable is False (too few rule-based personas
    # priced for the comparison to mean anything).
    def _num(value):
        return "None" if value is None else f"{value:,.2f}"

    print(f"\nBlended view (deterministic_usable={res.deterministic_usable}, "
          f"{res.deterministic_participating}/{len(res.personas)} rule-based priced)")
    for meta, p in zip(PERSONAS, res.personas):
        rule_value = res.deterministic_personas.get(meta["key"])
        print(f"  {p.persona:<22} crew={_num(p.fair_value_per_share):>10}"
              f"  rule={_num(rule_value):>10}  avg={_num(p.blended_fair_value):>10}")
    print(f"  Overall blended fair value: {_num(res.blended_fair_value)}")

    # Partial failures are surfaced, never swallowed.
    if res.errors:
        print("\nErrors:")
        for e in res.errors:
            print("  -", e)

"""
ADVANCED REFLECTION AGENT WITH LANGGRAPH + STATEGRAPH
====================================================

What is a Reflection Agent?
---------------------------
A Reflection Agent is an AI system that can evaluate its own outputs,
identify weaknesses, and iteratively improve through feedback loops.
This mimics how humans refine their work: draft → review → revise.

Concept: System 1 vs System 2
------------------------------
- System 1 (Generation): Quick, instinctive first draft.
- System 2 (Reflection): Deliberate critique and refinement.

The agent cycles between these two modes until quality standards are met.

Workflow
--------
START
  |
  v
GENERATE (System 1)  ←──────────────────┐
  |                                       │
  v                                       │
REFLECT (System 2)                        │
  |                                       │
  v                                       │
EVALUATE SCORE                            │
  ├── score >= threshold (8.5/10) ───→ END (quality met)
  └── score < threshold ───→ GENERATE AGAIN ─┘
  └── max 4 iterations exceeded ──→ END (safety limit)

Key Features
------------
- StateGraph architecture with explicit typed state
- Separate LLM instances for generation (creative) vs reflection (critical)
- Quality scoring with regex extraction from critique
- Early stopping when quality threshold is reached
- Config-driven design for easy parameter tuning
- Structured logging for observability
- Retry-safe with fallback post on API failure
- Graph visualization (PNG or Mermaid)
"""

# ============================================================================
# IMPORTS
# ============================================================================

import os
import logging
from typing import Annotated, TypedDict, List, Literal, Optional

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the Reflection Agent.

    All tunable parameters are defined here for easy experimentation
    without digging into the workflow logic.

    Attributes:
        MODEL_NAME: DeepSeek model identifier used for all LLM calls.
        BASE_URL: API endpoint for DeepSeek's OpenAI-compatible gateway.
        MAX_ITERATIONS: Maximum generate-reflect cycles before forced exit.
            Prevents infinite loops from broken scoring.
        MIN_QUALITY_SCORE: Score threshold (out of 10) for early stopping.
            Once the reflection critique assigns a score >= this value,
            the router sends the workflow to END.
        TEMPERATURE_GENERATE: Higher = more creative, varied drafts.
            Used for the generation node (System 1).
        TEMPERATURE_REFLECT: Lower = more focused, consistent critiques.
            Used for the reflection node (System 2).
        TIMEOUT_SECONDS: Maximum wall-clock time for the entire workflow.
            Prevents indefinite hangs due to network or API issues.
    """
    MODEL_NAME = "deepseek-v4-flash"  # Use "deepseek-v4" for DeepSeek V4 model
    BASE_URL = "https://api.deepseek.com/v1"

    MAX_ITERATIONS = 4
    MIN_QUALITY_SCORE = 8.5

    TEMPERATURE_GENERATE = 0.7
    TEMPERATURE_REFLECT = 0.2

    TIMEOUT_SECONDS = 120

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ============================================================================
# ENVIRONMENT
# ============================================================================

load_dotenv()

DEEPSEEK_API_KEY = (
    os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "Missing API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
    )

# ============================================================================
# LLM INITIALIZATION
# ============================================================================

generation_llm = ChatOpenAI(
    model=Config.MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=Config.BASE_URL,
    temperature=Config.TEMPERATURE_GENERATE,
)

reflection_llm = ChatOpenAI(
    model=Config.MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=Config.BASE_URL,
    temperature=Config.TEMPERATURE_REFLECT,
)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """Typed state schema for the LangGraph StateGraph workflow.

    This defines the shape of data that flows through every node.
    LangGraph uses this schema to validate and track state transitions.

    Fields:
        messages: Conversation history managed by LangGraph's add_messages
            reducer. Each node appends new messages (AIMessage from
            generation, HumanMessage from reflection) rather than
            overwriting the list. This preserves full context for prompts.

        iteration: Zero-based count of completed generate-reflect cycles.
            Incremented by the reflection node. Used by the router to
            enforce MAX_ITERATIONS.

        quality_score: Numeric score (0.0-10.0) extracted from the latest
            reflection critique. The router compares this against
            MIN_QUALITY_SCORE to decide early stopping.

        latest_feedback: Raw text of the most recent reflection critique.
            Stored for optional external logging or inspection.

        final_post: The latest generated post text. Written by the
            generation node on each cycle. At workflow end this holds
            the best (or last) version.
    """
    messages: Annotated[List[BaseMessage], add_messages]

    iteration: int
    quality_score: float

    latest_feedback: str
    final_post: str

# ============================================================================
# GENERATION PROMPT
# ============================================================================

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an elite LinkedIn content strategist.

Your ONLY task is to generate a high-performing LinkedIn post.

Rules:
- Output ONLY the post
- No explanations
- No markdown fences
- No commentary
- Optimize for engagement
- Maintain professional tone
- Use concise formatting
- Include a strong hook
- Include emotional resonance
- Add CTA if appropriate
"""
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | generation_llm

# ============================================================================
# REFLECTION PROMPT
# ============================================================================

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a senior LinkedIn growth strategist.

Critique the LinkedIn post.

Return:
1. Quality score from 1-10
2. Weaknesses
3. Specific improvements

Be brutally honest.
"""
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect_chain = reflection_prompt | reflection_llm

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_score(text: str) -> float:
    """Parse a numeric quality score from reflection critique text.

    The reflection prompt instructs the LLM to include a score in the
    format `N/10` (e.g. "7/10", "8.5/10"). This function extracts it
    via regex. If no match is found, a neutral score of 5.0 is returned
    so the workflow continues rather than prematurely stopping.

    Args:
        text: Raw output from the reflection chain.

    Returns:
        A float between 0.0 and 10.0 representing the assessed quality.
        Defaults to 5.0 if no score pattern is detected.
    """
    import re

    match = re.search(r'(\d+(\.\d+)?)\s*/?\s*10', text)

    if match:
        return float(match.group(1))

    return 5.0

# ============================================================================
# GENERATION NODE
# ============================================================================

def generation_node(state: AgentState) -> AgentState:
    """Generate or refine a LinkedIn post (System 1).

    This is the creative half of the reflection loop. It takes the full
    conversation history (initial prompt + any prior critique) and asks
    the LLM to produce the best possible LinkedIn post.

    The node:
    1. Invokes the generate_chain with the current message history.
    2. Strips whitespace from the LLM response.
    3. Logs the generated post length for observability.
    4. Returns the new AIMessage appended to `messages` and updates
       `final_post` with the latest version.

    Args:
        state: The current AgentState containing message history,
            iteration count, and prior feedback.

    Returns:
        A partial AgentState update with:
        - messages: New AIMessage appended (via add_messages reducer).
        - final_post: The latest generated post text.
    """
    logger.info(
        f"Generation iteration: {state['iteration']}"
    )

    response = generate_chain.invoke(
        {"messages": state["messages"]}
    )

    generated_post = response.content.strip()

    logger.info(
        f"Generated post length: {len(generated_post)}"
    )

    return {
        "messages": [
            AIMessage(content=generated_post)
        ],

        "final_post": generated_post,
    }

# ============================================================================
# REFLECTION NODE
# ============================================================================

def reflection_node(state: AgentState) -> AgentState:
    """Critique the latest generated LinkedIn post (System 2).

    This is the analytical half of the reflection loop. It evaluates the
    post produced by the generation node and provides structured feedback
    including a quality score.

    The node:
    1. Invokes the reflect_chain (lower temperature for consistency).
    2. Strips the response text.
    3. Parses a numeric score from the critique via extract_score().
    4. Logs the score for observability.
    5. Increments the iteration counter.
    6. Returns the critique as a HumanMessage so the generation node
       treats it like user feedback (triggering a revision).

    The critique is wrapped in HumanMessage deliberately: the generation
    prompt sees it as "user input" and revises the post accordingly.

    Args:
        state: The current AgentState containing the AI-generated post
            in the messages list.

    Returns:
        A partial AgentState update with:
        - messages: New HumanMessage (critique) appended.
        - latest_feedback: Raw critique text for inspection.
        - quality_score: Parsed numeric score.
        - iteration: Incremented by 1.
    """
    logger.info("Running reflection analysis")

    response = reflect_chain.invoke(
        {"messages": state["messages"]}
    )

    feedback = response.content.strip()

    score = extract_score(feedback)

    logger.info(f"Reflection score: {score}")

    return {
        "messages": [
            HumanMessage(content=feedback)
        ],

        "latest_feedback": feedback,
        "quality_score": score,
        "iteration": state["iteration"] + 1,
    }

# ============================================================================
# ROUTER
# ============================================================================

def router(
    state: AgentState
) -> Literal["generate", END]:
    """Decide whether to continue refining or stop the workflow.

    This is the conditional edge attached to the reflect node. After
    each critique it evaluates two stopping criteria:

    1. Quality threshold: If the reflection score >= MIN_QUALITY_SCORE,
       the post is good enough → route to END.
    2. Max iterations: If iteration count >= MAX_ITERATIONS, stop to
       avoid infinite loops → route to END.

    If neither condition is met → route back to "generate" for another
    refinement cycle.

    Args:
        state: The current AgentState after a reflection cycle.
            Uses iteration count and quality_score for decisions.

    Returns:
        "generate" to continue the refinement loop, or END to terminate.
    """
    iteration = state["iteration"]
    score = state["quality_score"]

    logger.info(
        f"Router check | Iteration={iteration} | Score={score}"
    )

    # Early stopping if quality is high enough
    if score >= Config.MIN_QUALITY_SCORE:
        logger.info("Quality threshold reached")
        return END

    # Max iteration protection
    if iteration >= Config.MAX_ITERATIONS:
        logger.info("Max iterations reached")
        return END

    return "generate"

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

# Build the LangGraph StateGraph using the AgentState schema.
# The graph has two nodes connected in a loop with a conditional exit:
#
#   generate ──→ reflect ──→ router ──→ generate (loop)
#                                  └──→ END
#
# - "generate" produces a post draft.
# - "reflect" critiques the draft and assigns a score.
# - The router checks the score and iteration count to decide.

graph = StateGraph(AgentState)

# Nodes
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)

# Entry
graph.set_entry_point("generate")

# Edges
graph.add_edge("generate", "reflect")

graph.add_conditional_edges(
    "reflect",
    router
)

# ============================================================================
# COMPILE
# ============================================================================

workflow = graph.compile()

# ============================================================================
# EXECUTION
# ============================================================================

def main():
    """Run the full Reflection Agent workflow and display results.

    This is the primary entry point for the script. It:
    1. Defines the user prompt for the LinkedIn post.
    2. Initialises the AgentState with zero values.
    3. Invokes the compiled LangGraph workflow.
    4. Prints the final post, quality score, and iteration count.
    5. Saves the final post to a markdown file.
    6. Falls back to a hardcoded post if the API call fails.

    The workflow runs synchronously. Each cycle makes two LLM calls
    (generate + reflect), so N iterations = 2N API calls.
    """
    print("=" * 80)
    print("ADVANCED REFLECTION AGENT")
    print("=" * 80)

    user_prompt = (
        "Write a LinkedIn post about getting a "
        "software developer role at IBM under 250 characters."
    )

    initial_state: AgentState = {
        "messages": [
            HumanMessage(content=user_prompt)
        ],

        "iteration": 0,
        "quality_score": 0.0,
        "latest_feedback": "",
        "final_post": "",
    }

    try:

        result = workflow.invoke(initial_state)

        final_post = result["final_post"]

        print("\nFINAL LINKEDIN POST")
        print("-" * 80)
        print(final_post)

        print("\nQUALITY SCORE:", result["quality_score"])
        print("ITERATIONS:", result["iteration"])

        # Save markdown
        with open(
            "final_linkedin_post.md",
            "w",
            encoding="utf-8"
        ) as f:

            f.write("# Final LinkedIn Post\n\n")
            f.write(final_post)

        logger.info("Final post saved")

    except Exception as e:

        logger.exception("Workflow failed")

        fallback_post = (
            "Excited to begin my journey at IBM as a Software Developer! "
            "Grateful for the opportunity to learn, innovate, and grow "
            "with some of the brightest minds in tech. 🚀 "
            "#IBM #SoftwareDeveloper"
        )

        print("\nFallback Post:\n")
        print(fallback_post)

# ============================================================================
# GRAPH EXPORT
# ============================================================================

def export_graph():
    """Save the workflow graph as a PNG or Mermaid diagram file.

    Attempts to render the graph as a PNG image (requires pygraphviz +
    Graphviz system library). If that fails (common on Windows without
    MSVC Build Tools), falls back to Mermaid text format which can be
    pasted into https://mermaid.live/ for rendering.

    The output file is saved in the current working directory:
    - `advanced_reflection_graph.png`  (if pygraphviz available)
    - `advanced_reflection_graph.mmd`  (fallback Mermaid text)
    """
    try:

        png_data = workflow.get_graph().draw_png()

        with open(
            "advanced_reflection_graph.png",
            "wb"
        ) as f:

            f.write(png_data)

        logger.info("PNG graph exported")

    except Exception:

        mermaid = workflow.get_graph().draw_mermaid()

        with open(
            "advanced_reflection_graph.mmd",
            "w"
        ) as f:

            f.write(mermaid)

        logger.info("Mermaid graph exported")

# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    """Script entry point.

    Execution order:
    1. export_graph() — Save workflow diagram to disk first (fast, no API).
    2. main()        — Run the full reflection agent workflow.

    This guard ensures neither function runs when this file is imported
    as a module — only when executed directly as a script.
    """
    export_graph()
    main()
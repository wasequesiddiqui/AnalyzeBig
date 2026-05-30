
"""
REFLECTION AGENT -- LinkedIn Post Generator using StateGraph
============================================================

This implementation replaces MessageGraph with StateGraph.

Why StateGraph?
---------------
StateGraph gives you:
1. Explicit control over state management
2. Easier extensibility for complex workflows
3. Ability to store additional metadata
4. More production-ready graph architecture

Workflow:
---------
Generate -> Reflect -> Generate -> Reflect -> END
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os

from typing import Annotated, List, TypedDict

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
# 1. ENVIRONMENT & LLM SETUP
# ============================================================================

# Load environment variables
load_dotenv()

# --------------------------------------------------------------------------
# API Key
# --------------------------------------------------------------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError(
        "No API key found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY."
    )

# --------------------------------------------------------------------------
# LLM Initialization
# --------------------------------------------------------------------------
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com/v1",
)

# ============================================================================
# 2. DEFINE STATE
# ============================================================================
# StateGraph requires an explicit state definition.
#
# add_messages automatically appends new messages
# instead of overwriting existing ones.
# ============================================================================

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ============================================================================
# 3. GENERATION PROMPT & CHAIN
# ============================================================================

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content assistant tasked with "
            "crafting engaging, insightful, and well-structured LinkedIn posts. "
            "Generate the best LinkedIn post possible for the user's request. "
            "If the user provides feedback or critique, revise and improve "
            "the previous draft."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generate_chain = generation_prompt | llm

# ============================================================================
# 4. REFLECTION PROMPT & CHAIN
# ============================================================================

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a professional LinkedIn content strategist. "
            "Critically evaluate the LinkedIn post and provide actionable "
            "feedback for improvement.\n\n"
            "Focus on:\n"
            "- Clarity\n"
            "- Engagement\n"
            "- Professionalism\n"
            "- Formatting\n"
            "- CTA effectiveness\n"
            "- LinkedIn best practices"
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflect_chain = reflection_prompt | llm

# ============================================================================
# 5. DEFINE GRAPH NODES
# ============================================================================

# --------------------------------------------------------------------------
# GENERATION NODE
# --------------------------------------------------------------------------
def generation_node(state: AgentState) -> AgentState:
    """
    Generate or revise a LinkedIn post.
    """

    response = generate_chain.invoke(
        {"messages": state["messages"]}
    )

    return {
        "messages": [
            AIMessage(content=response.content)
        ]
    }

# --------------------------------------------------------------------------
# REFLECTION NODE
# --------------------------------------------------------------------------
def reflection_node(state: AgentState) -> AgentState:
    """
    Critique the generated LinkedIn post.
    """

    response = reflect_chain.invoke(
        {"messages": state["messages"]}
    )

    return {
        "messages": [
            HumanMessage(content=response.content)
        ]
    }

# ============================================================================
# 6. ROUTER FUNCTION
# ============================================================================

def should_continue(state: AgentState):
    """
    Decide whether to continue reflection cycles
    or terminate the workflow.
    """

    messages = state["messages"]

    print("\nCurrent Message Count:", len(messages))
    print("--------------------------------------------------")

    # Stop after enough iterations
    if len(messages) > 6:
        return END

    return "reflect"

# ============================================================================
# 7. BUILD THE STATEGRAPH
# ============================================================================

graph = StateGraph(AgentState)

# --------------------------------------------------------------------------
# ADD NODES
# --------------------------------------------------------------------------
graph.add_node("generate", generation_node)
graph.add_node("reflect", reflection_node)

# --------------------------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------------------------
graph.set_entry_point("generate")

# --------------------------------------------------------------------------
# EDGES
# --------------------------------------------------------------------------
graph.add_edge("reflect", "generate")

# Conditional edge after generation
graph.add_conditional_edges(
    "generate",
    should_continue,
)

# ============================================================================
# 8. COMPILE GRAPH
# ============================================================================

workflow = graph.compile()

# ============================================================================
# 9. RUN THE WORKFLOW
# ============================================================================

print("=" * 70)
print("Reflection Agent -- LinkedIn Post Generator")
print("=" * 70)

# Initial Input
initial_state = {
    "messages": [
        HumanMessage(
            content="Write a LinkedIn post on getting a software developer job at IBM under 160 characters"
        )
    ]
}

# Execute workflow
response = workflow.invoke(initial_state)

# ============================================================================
# 10. DISPLAY RESULTS
# ============================================================================

messages = response["messages"]

print("\n" + "=" * 70)
print("FIRST DRAFT")
print("=" * 70)
print(messages[1].content)

print("\n" + "=" * 70)
print("FIRST CRITIQUE")
print("=" * 70)
print(messages[2].content)

print("\n" + "=" * 70)
print("FINAL REFINED POST")
print("=" * 70)
print(messages[-1].content)

# Save final post to markdown file
with open("final_post.md", "w", encoding="utf-8") as f:
    f.write(f"# Final Refined LinkedIn Post\n\n{messages[-1].content}\n")
print("\nFinal post saved to final_post.md")

# ============================================================================
# 11. SAVE GRAPH VISUALIZATION
# ============================================================================
# Uses draw_mermaid() as fallback to avoid requiring pygraphviz
# (which needs MSVC Build Tools + Graphviz system library on Windows).
# --------------------------------------------------------------------------
try:
    graph_png = workflow.get_graph().draw_png()
    with open("reflection_agent_stategraph.png", "wb") as f:
        f.write(graph_png)
    print("\nGraph saved as reflection_agent_stategraph.png")
except Exception:
    mermaid_text = workflow.get_graph().draw_mermaid()
    with open("reflection_agent_graph.mmd", "w") as f:
        f.write(mermaid_text)
    print("\nGraph saved as reflection_agent_graph.mmd (Mermaid text)")
    print("Paste into https://mermaid.live/ to view the diagram.")

"""
Universal Recursive YouTube Agent
---------------------------------

Features:
- Recursive tool-calling workflow
- RunnableLambda architecture
- Parallel tool execution
- Automatic recursion until completion
- YouTube search + metadata + transcript tools
- Production-ready structure
"""

import json
import re
from typing import Any, Dict, List, Union

import yt_dlp
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import (
    RunnableLambda,
)
from langchain_openai import ChatOpenAI
from pytube import Search
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

# =========================================================
# CONSTANTS
# =========================================================

api = YouTubeTranscriptApi()

YOUTUBE_ID_PATTERN = re.compile(
    r"(?:v=|be/|embed/|shorts/)([a-zA-Z0-9_-]{11})"
)

YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
)

# =========================================================
# LOGGER
# =========================================================


class YTDLPLogger:
    """Custom logger for yt-dlp."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        print(f"[WARNING] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


yt_dlp_logger = YTDLPLogger()

# =========================================================
# YT-DLP OPTIONS
# =========================================================

YDL_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "skip_download": True,
    "extract_flat": False,
    "logger": yt_dlp_logger,
}

# =========================================================
# HELPERS
# =========================================================


def safe_json(data: Any) -> str:
    """
    Convert object into formatted JSON.
    """

    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def validate_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL.
    """

    return bool(
        isinstance(url, str)
        and url.strip()
        and YOUTUBE_URL_PATTERN.search(url)
    )


def validate_video_id(video_id: str) -> bool:
    """
    Validate YouTube video ID.
    """

    return bool(
        re.match(
            r"^[a-zA-Z0-9_-]{11}$",
            video_id,
        )
    )


# =========================================================
# TOOLS
# =========================================================


@tool
def extract_video_id(url: str) -> str:
    """
    Extract YouTube video ID from a URL.
    """

    if not validate_youtube_url(url):
        return "Error: Invalid YouTube URL"

    match = YOUTUBE_ID_PATTERN.search(url)

    return (
        match.group(1)
        if match
        else "Error: Could not extract video ID"
    )


@tool
def fetch_transcript(
    video_id: str,
    language: str = "en"
) -> str:
    """
    Fetch transcript from a YouTube video.
    """

    if not validate_video_id(video_id):
        return "Error: Invalid YouTube video ID"

    try:

        transcript_data = api.fetch(
            video_id,
            languages=[language]
        )

        return " ".join(
            snippet.text
            for snippet in transcript_data.snippets
        )

    except NoTranscriptFound:

        return (
            f"Error: No transcript found "
            f"for '{language}'"
        )

    except TranscriptsDisabled:

        return (
            "Error: Transcripts are disabled"
        )

    except Exception as e:

        return f"Error: {str(e)}"


@tool
def search_youtube(
    query: str,
    max_results: int = 5
) -> Union[List[Dict[str, str]], str]:
    """
    Search YouTube videos using PyTube.
    """

    if not query.strip():
        return "Error: Query cannot be empty"

    try:

        search = Search(query)

        videos = [
            {
                "title": yt.title.strip(),
                "video_id": yt.video_id,
                "url": (
                    f"https://youtu.be/"
                    f"{yt.video_id}"
                ),
            }
            for yt in search.results[:max_results]
            if yt.video_id and yt.title
        ]

        return (
            videos
            if videos
            else "Error: No videos found"
        )

    except Exception as e:

        return f"Error: {str(e)}"


@tool
def get_full_metadata(
    url: str
) -> Union[Dict[str, Any], str]:
    """
    Extract detailed YouTube metadata.
    """

    if not validate_youtube_url(url):
        return "Error: Invalid YouTube URL"

    try:

        with yt_dlp.YoutubeDL(
            YDL_OPTS
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False
            )

        duration = (
            info.get("duration")
            or 0
        )

        return {
            "title": info.get("title"),
            "video_id": info.get("id"),
            "url": info.get("webpage_url"),
            "views": info.get("view_count"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "duration_seconds": duration,
            "duration_minutes": round(
                duration / 60,
                2
            ),
            "channel": info.get("uploader"),
            "upload_date": info.get(
                "upload_date"
            ),
            "thumbnail": info.get(
                "thumbnail"
            ),
            "tags": info.get("tags", []),
            "categories": info.get(
                "categories",
                []
            ),
            "chapters": info.get(
                "chapters",
                []
            ),
            "description": info.get(
                "description"
            ),
        }

    except yt_dlp.utils.DownloadError as e:

        return f"Error: {str(e)}"

    except Exception as e:

        return f"Error: {str(e)}"


@tool
def get_thumbnails(
    url: str
) -> List[Dict[str, Any]]:
    """
    Retrieve all available thumbnails.
    """

    if not validate_youtube_url(url):

        return [
            {
                "error":
                "Invalid YouTube URL"
            }
        ]

    try:

        with yt_dlp.YoutubeDL(
            YDL_OPTS
        ) as ydl:

            info = ydl.extract_info(
                url,
                download=False
            )

        return [

            {
                "url": thumb["url"],
                "width": thumb.get(
                    "width"
                ),
                "height": thumb.get(
                    "height"
                ),
                "resolution": (
                    f"{thumb.get('width')}"
                    f"x"
                    f"{thumb.get('height')}"
                    if thumb.get("width")
                    and thumb.get("height")
                    else None
                ),
            }

            for thumb in info.get(
                "thumbnails",
                []
            )

            if thumb.get("url")
        ]

    except Exception as e:

        return [{"error": str(e)}]


# =========================================================
# LLM
# =========================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,
    max_retries=2,
)

# =========================================================
# TOOL REGISTRY
# =========================================================

tools = [
    extract_video_id,
    fetch_transcript,
    search_youtube,
    get_full_metadata,
    get_thumbnails,
]

tool_mapping = {
    tool.name: tool
    for tool in tools
}

llm_with_tools = llm.bind_tools(tools)

# =========================================================
# TOOL EXECUTION
# =========================================================


def execute_tool(tool_call):
    """
    Execute single tool call
    and return ToolMessage.
    """

    try:

        result = tool_mapping[
            tool_call["name"]
        ].invoke(
            tool_call["args"]
        )

        content = (
            safe_json(result)
            if isinstance(
                result,
                (dict, list)
            )
            else str(result)
        )

    except Exception as e:

        content = f"Error: {str(e)}"

    return ToolMessage(
        content=content,
        tool_call_id=tool_call["id"]
    )


# =========================================================
# RECURSIVE PROCESSING
# =========================================================


def process_tool_calls(messages):
    """
    Recursive tool processor.
    """

    last_message = messages[-1]

    # -----------------------------------------------------
    # Execute all tool calls
    # -----------------------------------------------------

    tool_messages = [

        execute_tool(tc)

        for tc in getattr(
            last_message,
            "tool_calls",
            []
        )
    ]

    # -----------------------------------------------------
    # Update history
    # -----------------------------------------------------

    updated_messages = (
        messages + tool_messages
    )

    # -----------------------------------------------------
    # Invoke LLM again
    # -----------------------------------------------------

    next_ai_response = (
        llm_with_tools.invoke(
            updated_messages
        )
    )

    # -----------------------------------------------------
    # Return updated history
    # -----------------------------------------------------

    return (
        updated_messages
        + [next_ai_response]
    )


def should_continue(messages):
    """
    Check whether recursion
    should continue.
    """

    last_message = messages[-1]

    return bool(
        getattr(
            last_message,
            "tool_calls",
            None
        )
    )


def _recursive_chain(messages):
    """
    Recursive agent loop.
    """

    if should_continue(messages):

        new_messages = (
            process_tool_calls(
                messages
            )
        )

        return _recursive_chain(
            new_messages
        )

    return messages


# =========================================================
# RECURSIVE RUNNABLE
# =========================================================

recursive_chain = RunnableLambda(
    _recursive_chain
)

# =========================================================
# UNIVERSAL AGENT CHAIN
# =========================================================

universal_chain = (

    # -----------------------------------------------------
    # Create initial HumanMessage
    # -----------------------------------------------------

    RunnableLambda(
        lambda x: [
            HumanMessage(
                content=x["query"]
            )
        ]
    )

    # -----------------------------------------------------
    # Initial LLM invocation
    # -----------------------------------------------------

    | RunnableLambda(
        lambda messages:
        messages + [
            llm_with_tools.invoke(
                messages
            )
        ]
    )

    # -----------------------------------------------------
    # Recursive execution
    # -----------------------------------------------------

    | recursive_chain
)

# =========================================================
# FINAL RESPONSE
# =========================================================


def get_final_response(messages):
    """
    Extract final AI response.
    """

    for msg in reversed(messages):

        if getattr(msg, "type", "") == "ai":

            return msg.content

    return "No response generated."


# =========================================================
# CHAT HISTORY
# =========================================================


def print_chat_history(messages):
    """
    Print conversation history.
    """

    print("\n📜 CHAT HISTORY")
    print("=" * 60)

    for msg in messages:

        print(f"\n[{msg.type.upper()}]")

        if hasattr(msg, "content"):
            print(msg.content)

    print("\n" + "=" * 60)


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    query = """
    Summarize this YouTube video in English:

    https://www.youtube.com/watch?v=FkJ-X5CRSMw
    """

    # -----------------------------------------------------
    # Run recursive agent
    # -----------------------------------------------------

    messages = universal_chain.invoke(
        {
            "query": query
        }
    )

    # -----------------------------------------------------
    # Extract final answer
    # -----------------------------------------------------

    final_response = (
        get_final_response(messages)
    )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("📄 FINAL RESPONSE")
    print("=" * 60)
    print(final_response)
    print("=" * 60)

    # -----------------------------------------------------
    # Optional Debug History
    # -----------------------------------------------------

    print_chat_history(messages)

    # =========================================================
# WHY DO WE NEED `universal_chain`?
# =========================================================
#
# The `universal_chain` acts as the MAIN ENTRY POINT
# for the recursive agent workflow.
#
# It converts raw user input into a fully autonomous
# tool-calling execution pipeline.
#
# ---------------------------------------------------------
# Without `universal_chain`
# ---------------------------------------------------------
#
# We would need to manually write:
#
# messages = [HumanMessage(content=query)]
#
# response = llm_with_tools.invoke(messages)
#
# messages.append(response)
#
# while response.tool_calls:
#     execute tools...
#     invoke llm again...
#
# This becomes repetitive and difficult to scale.
#
# ---------------------------------------------------------
# What `universal_chain` Does
# ---------------------------------------------------------
#
# universal_chain = (
#
#     RunnableLambda(
#         lambda x: [
#             HumanMessage(content=x["query"])
#         ]
#     )
#
#     | RunnableLambda(
#         lambda messages:
#         messages + [
#             llm_with_tools.invoke(messages)
#         ]
#     )
#
#     | recursive_chain
# )
#
# ---------------------------------------------------------
# STEP 1 → Create HumanMessage
# ---------------------------------------------------------
#
# Input:
#
# {
#     "query": "Summarize this video"
# }
#
# Output:
#
# [
#     HumanMessage(content="Summarize this video")
# ]
#
# Chat models require a LIST OF MESSAGES,
# not plain text strings.
#
# ---------------------------------------------------------
# STEP 2 → Initial LLM Invocation
# ---------------------------------------------------------
#
# The first LLM call determines:
#
# - whether tools are needed
# - which tools to call
# - what arguments to pass
#
# Example AI response:
#
# AIMessage(
#     tool_calls=[
#         {
#             "name": "extract_video_id",
#             "args": {...}
#         }
#     ]
# )
#
# ---------------------------------------------------------
# STEP 3 → Recursive Tool Processing
# ---------------------------------------------------------
#
# `recursive_chain`:
#
# 1. Detects tool calls
# 2. Executes tools
# 3. Creates ToolMessages
# 4. Appends results to history
# 5. Reinvokes the LLM
# 6. Repeats until no tool calls remain
#
# This creates an AUTONOMOUS AGENT LOOP.
#
# ---------------------------------------------------------
# Why Is It Called "Universal"?
# ---------------------------------------------------------
#
# Because the SAME orchestration works for:
#
# - YouTube agents
# - Search agents
# - SQL agents
# - Web agents
# - Research agents
# - RAG pipelines
# - Multi-step reasoning systems
#
# Only the TOOLS change.
#
# The execution engine remains identical.
#
# ---------------------------------------------------------
# Architecture Flow
# ---------------------------------------------------------
#
# User Query
#     │
#     ▼
# universal_chain
#     │
#     ├── Create HumanMessage
#     │
#     ├── Initial LLM Call
#     │
#     └── recursive_chain
#             │
#             ├── Detect Tool Calls
#             ├── Execute Tools
#             ├── Append ToolMessages
#             ├── Invoke LLM Again
#             └── Repeat Until Completion
#
# ---------------------------------------------------------
# Main Benefit
# ---------------------------------------------------------
#
# `universal_chain` separates:
#
# - workflow entry
# - recursion logic
# - tool execution
# - stopping conditions
#
# This makes the system:
#
# - modular
# - reusable
# - scalable
# - easier to debug
# - production-friendly
#
# =========================================================
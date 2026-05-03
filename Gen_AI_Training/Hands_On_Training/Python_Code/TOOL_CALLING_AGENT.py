import re
import yt_dlp
from pytube import YouTube
from langchain_core.tools import tool
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI

load_dotenv()
# Pre-compile regex pattern once for better performance
# This avoids recompiling the pattern every time the function runs
YOUTUBE_ID_PATTERN = re.compile(
    r"(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})"
)

@tool
def extract_video_id(url: str) -> str:
    """
    Extract the 11-character YouTube video ID from a given URL.

    This function supports multiple common YouTube URL formats:
    - Standard watch URL
    - Shortened URL (youtu.be)
    - Embedded video URL

    Supported formats:
        https://www.youtube.com/watch?v=VIDEO_ID
        https://youtu.be/VIDEO_ID
        https://www.youtube.com/embed/VIDEO_ID

    Args:
        url (str):
            A string containing a valid YouTube URL.

    Returns:
        str:
            - The extracted 11-character YouTube video ID if successful
            - A descriptive error message if extraction fails

    Examples:
        >>> extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'

        >>> extract_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'

        >>> extract_video_id("https://invalid-url.com")
        'Error: Invalid YouTube URL'

    Notes:
        - YouTube video IDs are always 11 characters long
        - This function uses regex pattern matching for extraction
        - If integrating in production, consider raising exceptions instead of returning error strings
    """

    # Step 1: Validate input to ensure it's a non-empty string
    if not isinstance(url, str) or not url.strip():
        return "Error: URL must be a non-empty string"

    # Step 2: Search for the video ID using the precompiled regex pattern
    match = YOUTUBE_ID_PATTERN.search(url)

    # Step 3: If a match is found, extract the first captured group (video ID)
    if match:
        return match.group(1)

    # Step 4: If no match is found, return a clear error message
    return "Error: Invalid YouTube URL"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
    timeout=30,
    model_kwargs={
        "max_tokens": 500
    }
)
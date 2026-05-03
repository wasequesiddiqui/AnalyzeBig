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
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

load_dotenv()
api = YouTubeTranscriptApi()
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

@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Retrieve the transcript text for a given YouTube video.

    This function uses the YouTube Transcript API to fetch subtitles (if available)
    and returns them as a single concatenated string.

    It supports language selection and includes graceful handling of common errors
    such as missing transcripts or disabled captions.

    Args:
        video_id (str):
            The unique 11-character YouTube video ID.
            Example: "dQw4w9WgXcQ"

        language (str, optional):
            Preferred language code for the transcript.
            Defaults to "en" (English).
            Example values: "en", "es", "hi", "fr"

    Returns:
        str:
            - Full transcript as a single string if successful
            - Error message if transcript cannot be retrieved

    Examples:
        >>> fetch_transcript("dQw4w9WgXcQ")
        "We're no strangers to love You know the rules and so do I..."

        >>> fetch_transcript("dQw4w9WgXcQ", language="es")
        "No somos extraños al amor Conoces las reglas y yo también..."

        >>> fetch_transcript("invalid_id")
        "Error: Invalid YouTube video ID"

    Notes:
        - Not all YouTube videos have transcripts available
        - Some videos disable captions entirely
        - Auto-generated captions may differ in quality
        - For production systems, consider logging errors instead of returning strings
    """

    # Step 1: Validate the video_id format (YouTube IDs are exactly 11 characters)
    if not isinstance(video_id, str) or not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
        return "Error: Invalid YouTube video ID"

    try:
        # Step 2: Attempt to fetch transcript in the requested language
        transcript_data = api.fetch(
            video_id,
            languages=[language]  # Preferred language
        )

        # Step 3: Extract text from each transcript segment and combine into one string
        # Each segment is a dict like: {"text": "...", "start": ..., "duration": ...}
        full_text = " ".join([snippet.text for snippet in transcript_data.snippets])

        # Step 4: Return the cleaned transcript text
        return full_text

    except NoTranscriptFound:
        # Step 5: Handle case where requested language is unavailable
        return f"Error: No transcript found for language '{language}'"

    except TranscriptsDisabled:
        # Step 6: Handle case where the video has captions disabled
        return "Error: Transcripts are disabled for this video"

    except Exception as e:
        # Step 7: Catch any unexpected errors and return a safe message
        return f"Error: {str(e)}"

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
    timeout=30,
    model_kwargs={
        "max_tokens": 500
    }
)

tools = []
tools.append(extract_video_id)
tools.append(fetch_transcript)

# print(extract_video_id.name)
# print("----------------------------")
# print(extract_video_id.description)
# print("----------------------------")
# print(extract_video_id.func)
youtube_id_key = extract_video_id.run("https://www.youtube.com/watch?v=FkJ-X5CRSMw&t=11s&pp=ugUEEgJlbg%3D%3D")
print("Extracted YouTube ID:", youtube_id_key)
youtube_transcript = fetch_transcript.run({
    "video_id": youtube_id_key,
    "language": "en"
})
print("Transcript for video ID", youtube_id_key, ":\n")
print(youtube_transcript)
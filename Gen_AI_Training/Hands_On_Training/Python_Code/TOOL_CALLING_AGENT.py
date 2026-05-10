import re
import yt_dlp
from pytube import YouTube, Search
from langchain_core.tools import tool
from typing import List, Dict, Union, Any
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain.tools import tool

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

@tool
def search_youtube(
    query: str,
    max_results: int = 5
) -> Union[List[Dict[str, str]], str]:
    """
    Search YouTube for videos matching a given search query.

    This function uses PyTube's YouTube search functionality to retrieve
    video metadata such as title, video ID, and direct video URL.

    Args:
        query (str):
            The search term to look up on YouTube.

            Example:
                "LangChain tutorial"

        max_results (int, optional):
            Maximum number of videos to return.

            Defaults to 5.

    Returns:
        Union[List[Dict[str, str]], str]:

            On success:
                A list of dictionaries containing:
                    - title
                    - video_id
                    - url

                Example:
                [
                    {
                        "title": "LangChain Crash Course",
                        "video_id": "abc123xyz89",
                        "url": "https://youtu.be/abc123xyz89"
                    }
                ]

            On failure:
                Returns a descriptive error message string.

    Examples:
        >>> search_youtube("Python tutorials")
        
        [
            {
                "title": "Python Full Course",
                "video_id": "kqtD5dpn9C8",
                "url": "https://youtu.be/kqtD5dpn9C8"
            }
        ]

    Notes:
        - Results depend on YouTube search availability
        - Some videos may not expose metadata correctly
        - Internet connection is required
        - PyTube may occasionally break if YouTube changes internal APIs
    """

    # Step 1: Validate query input
    # Ensure query is a non-empty string
    if not isinstance(query, str) or not query.strip():
        return "Error: Query must be a non-empty string"

    # Step 2: Validate max_results
    # Prevent invalid or excessive values
    if not isinstance(max_results, int) or max_results <= 0:
        return "Error: max_results must be a positive integer"

    try:
        # Step 3: Perform YouTube search using PyTube
        search = Search(query)

        # Step 4: Create an empty list to store cleaned results
        videos = []

        # Step 5: Iterate through search results
        for yt in search.results[:max_results]:

            # Skip videos missing critical metadata
            if not yt.video_id or not yt.title:
                continue

            # Step 6: Append structured video data
            videos.append({
                "title": yt.title.strip(),
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}"
            })

        # Step 7: Handle case where no valid videos were found
        if not videos:
            return "Error: No videos found"

        # Step 8: Return cleaned video results
        return videos

    except Exception as e:
        # Step 9: Catch unexpected runtime errors
        return f"Error: Failed to search YouTube - {str(e)}"

# Optional custom logger for yt-dlp
# Replace with your own logger if needed
class YTDLPLogger:
    """Minimal logger class for yt-dlp."""

    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)

# Create reusable logger instance
yt_dlp_logger = YTDLPLogger()

# Precompiled regex pattern for validating YouTube URLs
# Supports:
# - youtube.com/watch?v=
# - youtu.be/
# - youtube.com/embed/
YOUTUBE_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/"
)

@tool
def get_full_metadata(url: str) -> Union[Dict[str, Any], str]:
    """
    Extract detailed metadata from a YouTube video URL.

    This function uses yt-dlp to retrieve comprehensive metadata
    about a YouTube video without downloading the actual content.

    Extracted metadata includes:
        - Video title
        - View count
        - Duration (in seconds)
        - Channel/uploader name
        - Like count
        - Comment count
        - Video chapters
        - Upload date
        - Video description
        - Thumbnail URL
        - Tags
        - Categories

    Args:
        url (str):
            A valid YouTube video URL.

            Supported formats:
                https://www.youtube.com/watch?v=VIDEO_ID
                https://youtu.be/VIDEO_ID
                https://www.youtube.com/embed/VIDEO_ID

    Returns:
        Union[Dict[str, Any], str]:

            On success:
                Dictionary containing structured video metadata.

                Example:
                {
                    "title": "Python Tutorial",
                    "views": 120000,
                    "duration_seconds": 540,
                    "duration_minutes": 9.0,
                    "channel": "Programming Channel",
                    "likes": 15000,
                    "comments": 1200,
                    "upload_date": "20260510",
                    "thumbnail": "https://...",
                    "tags": ["python", "tutorial"],
                    "categories": ["Education"],
                    "chapters": [...]
                }

            On failure:
                Returns a descriptive error string.

    Examples:
        >>> get_full_metadata("https://youtu.be/dQw4w9WgXcQ")

    Notes:
        - Requires internet connection
        - Uses yt-dlp internally
        - Metadata availability depends on YouTube
        - Some fields may return None if unavailable
        - Does NOT download the video
    """

    # Step 1: Validate URL input type
    # Ensure input is a non-empty string
    if not isinstance(url, str) or not url.strip():
        return "Error: URL must be a non-empty string"

    # Step 2: Validate YouTube URL format
    # Prevent invalid domains or malformed URLs
    if not YOUTUBE_URL_PATTERN.search(url):
        return "Error: Invalid YouTube URL"

    try:
        # Step 3: Configure yt-dlp options
        # quiet=True suppresses console spam
        # skip_download ensures metadata-only extraction
        ydl_opts = {
            "quiet": True,
            "logger": yt_dlp_logger,
            "skip_download": True,
            "extract_flat": False,
            "no_warnings": True
        }

        # Step 4: Create yt-dlp context manager
        # Automatically handles cleanup/resources
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:

            # Step 5: Extract metadata without downloading
            info = ydl.extract_info(url, download=False)

        # Step 6: Safely extract duration
        # Default to 0 if unavailable
        duration_seconds = info.get("duration") or 0

        # Step 7: Convert duration into minutes for convenience
        duration_minutes = round(duration_seconds / 60, 2)

        # Step 8: Structure and clean metadata response
        metadata = {

            # Basic video information
            "title": info.get("title"),
            "video_id": info.get("id"),
            "url": info.get("webpage_url"),

            # Engagement metrics
            "views": info.get("view_count"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),

            # Duration information
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_minutes,

            # Channel/uploader information
            "channel": info.get("uploader"),
            "channel_id": info.get("channel_id"),

            # Publishing information
            "upload_date": info.get("upload_date"),

            # Media information
            "thumbnail": info.get("thumbnail"),

            # Classification metadata
            "tags": info.get("tags", []),
            "categories": info.get("categories", []),

            # Video structure metadata
            "chapters": info.get("chapters", []),

            # Optional description
            "description": info.get("description")
        }

        # Step 9: Return structured metadata
        return metadata

    except yt_dlp.utils.DownloadError as e:
        # Step 10: Handle yt-dlp extraction-specific failures
        return f"Error: Failed to extract video metadata - {str(e)}"

    except Exception as e:
        # Step 11: Catch unexpected runtime failures
        return f"Error: Unexpected error occurred - {str(e)}"
    
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
tools.append(search_youtube)
tools.append(get_full_metadata)

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

search_youtube_result = search_youtube.run("LangChain tutorial")
print("Search results for 'LangChain tutorial':\n")
for video in search_youtube_result:
    print(f"- {video['title']} ({video['url']})")

metadata_result = get_full_metadata.run("https://www.youtube.com/watch?v=FkJ-X5CRSMw&t=11s&pp=ugUEEgJlbg%3D%3D")
print("Metadata for video ID", youtube_id_key, ":\n")
print(metadata_result)
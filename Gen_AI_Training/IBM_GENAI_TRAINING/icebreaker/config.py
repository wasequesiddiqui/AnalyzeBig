
#%%
import os
from dotenv import load_dotenv

load_dotenv(override=True)

#%%
"""Configuration settings for the Icebreaker Bot."""

# IBM watsonx.ai settings
WATSONX_URL = "https://eu-de.ml.cloud.ibm.com"
WATSONX_PROJECT_ID = os.environ["WATSONX_PROJECT_ID"]

#%%
# Model settings
LLM_MODEL_ID = "ibm/granite-4-h-small"
EMBEDDING_MODEL_ID = "ibm/granite-embedding-278m-multilingual"

#%%
# Profile lookup configuration
# Uses the free JSONPlaceholder API through profile_lookup.py.
# The API returns sample user data, not real LinkedIn data.

def get_profile_context():
    """Fetch a sample profile and format it as text."""
    from profile_lookup import lookup_profile

    profile = lookup_profile()

    location = profile.get("location") or {}

    return f"""
Name: {profile.get('name') or 'Not available'}
Username: {profile.get('username') or 'Not available'}
Company: {profile.get('company') or 'Not available'}
Website: {profile.get('website') or 'Not available'}
Location: {location.get('city') or 'Not available'}
Email: {profile.get('email') or 'Not available'}
"""

#%%
# Coursera-provided mock LinkedIn profile data.
# Retained for compatibility with the original course workflow.
MOCK_DATA_URL = (
    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
    "ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"
)

# Query settings
SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

# Node settings
CHUNK_SIZE = 500

# LLM prompt templates
INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
If the information is insufficient, clearly state that.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full detail, using only the information provided in the context.
If the answer is not available in the provided context, say:
"I don't know. The information is not available in the provided data."
"""
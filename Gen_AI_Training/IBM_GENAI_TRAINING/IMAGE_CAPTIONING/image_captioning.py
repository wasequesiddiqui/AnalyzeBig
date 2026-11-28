# %% [markdown]
# =============================================================================
#  MULTIMODAL IMAGE Q&A WITH IBM watsonx.ai
# =============================================================================
#
#  WHAT THIS SCRIPT DOES
#  ---------------------
#  1. Loads secrets (API key, URL, project ID) from a `.env` file.
#  2. Connects to IBM watsonx.ai.
#  3. Downloads four images from the web and converts them to base64 text.
#  4. Sends each image + a question to a vision-capable LLM
#     (Llama 4 Maverick) and prints the model's answer.
#
#  HOW THE PIECES FIT TOGETHER
#  ---------------------------
#       .env file  ──►  setup_credentials()  ──►  ModelInference (the model)
#                                                        ▲
#       image URLs ──► encode_images_to_base64() ──►  generate_model_response()
#                                                        │
#                                                   printed answer
#
#  REQUIREMENTS
#  ------------
#  pip install ibm-watsonx-ai python-dotenv requests
#
#  A `.env` file next to this script, containing (no spaces around "="):
#       WATSONX_APIKEY=your_ibm_cloud_api_key
#       WATSONX_URL=https://us-south.ml.cloud.ibm.com
#       WATSONX_PROJECT_ID=your_project_id
#
#  NOTE ON "#%%" MARKERS
#  ---------------------
#  Lines starting with `# %%` split this file into "cells". In VS Code (with
#  the Python/Jupyter extensions) or Spyder you can run one cell at a time.
#  Running the file normally with `python script.py` also works: the markers
#  are just comments.
# =============================================================================


# %% ---------------------------------------------------------------------------
# SECTION 1: IMPORTS
# -----------------------------------------------------------------------------
# Standard-library modules (they ship with Python, nothing to install).
import os        # Lets us read environment variables, e.g. os.getenv("NAME").
import base64    # Converts binary data (image bytes) into plain text (base64).

# Third-party modules (installed with pip).
import requests  # Simple HTTP client; we use it to download the images.

from dotenv import load_dotenv
# ^ python-dotenv: reads the `.env` file and copies its KEY=VALUE lines into
#   the process environment so that os.getenv() can see them.

from ibm_watsonx_ai import Credentials, APIClient
# ^ Credentials : a small container holding the service URL + API key.
#   APIClient   : the main IBM SDK client object that authenticates with
#                 those credentials and talks to watsonx.ai.

from ibm_watsonx_ai.foundation_models import ModelInference
# ^ ModelInference: the object we use to actually send prompts to a model
#   and receive responses (via .chat(), .generate_text(), etc.).

from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
# ^ TextChatParameters: IBM's own class describing generation settings
#   (temperature, top_p, max_tokens, ...). We IMPORT it from the SDK.
#
#   WHY NOT REDEFINE IT?
#   The original script re-declared this class by hand. That is unnecessary
#   and caused a chain of "X is not defined" warnings (BaseSchema, dataclass,
#   Any, TextChatResponseFormat), because the copy depends on SDK internals.
#   Worse, a hand-written copy can silently shadow the real class and drift
#   out of sync with the installed SDK version. Importing is simpler & safer.


# %% ---------------------------------------------------------------------------
# SECTION 2: LOAD ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
# load_dotenv() searches for a file named ".env" (starting in the current
# working directory and walking upward) and loads every KEY=VALUE line into
# os.environ. It must run BEFORE any call to os.getenv().
#
# If it can't find your file (e.g. you run the script from another folder),
# pass the path explicitly:  load_dotenv(dotenv_path="/full/path/to/.env")
load_dotenv()


# %% ---------------------------------------------------------------------------
# SECTION 3: CREDENTIAL SETUP
# -----------------------------------------------------------------------------
def setup_credentials():
    """
    Read watsonx.ai settings from environment variables and build the
    objects needed to connect to the service.

    Environment variables required
    ------------------------------
    WATSONX_APIKEY     : Your IBM Cloud API key (a secret - never commit it).
    WATSONX_URL        : Regional endpoint, e.g. https://us-south.ml.cloud.ibm.com
                         It must match the region where your watsonx.ai
                         instance was created.
    WATSONX_PROJECT_ID : The ID of the watsonx.ai project that will be
                         billed / used for the model calls.

    Returns
    -------
    tuple : (model_id, credentials, client, project_id)
        model_id    (str)         : Identifier of the LLM we want to call.
        credentials (Credentials) : URL + API key bundle.
        client      (APIClient)   : Authenticated SDK client.
        project_id  (str)         : The project ID read from the environment.

    Raises
    ------
    EnvironmentError : if any required environment variable is missing.
    """

    # The model we will talk to. This one is multimodal (understands both
    # text AND images), which is required for this script. A text-only model
    # would reject the image part of our request.
    # Model IDs change over time; check the watsonx.ai model catalogue for
    # what is currently available in your region.
    model_id = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

    # os.getenv returns None if the variable does not exist.
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    # "Fail early" check: build a list of the names whose values are missing
    # (None or empty string). If anything is missing we stop right here with
    # a readable message, instead of getting a confusing authentication error
    # several steps later.
    #
    # How the list comprehension reads:
    #   for each (name, value) pair in the dictionary,
    #   keep `name` only if `value` is falsy (None or "").
    missing = [
        name
        for name, value in {
            "WATSONX_APIKEY": api_key,
            "WATSONX_URL": url,
            "WATSONX_PROJECT_ID": project_id,
        }.items()
        if not value
    ]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {', '.join(missing)}"
        )

    # Bundle the endpoint and key together. This object by itself does not
    # contact the network; it simply stores the settings.
    credentials = Credentials(url=url, api_key=api_key)

    # Create the SDK client. This is where authentication actually happens:
    # the API key is exchanged for a short-lived access token.
    client = APIClient(credentials)

    # Return everything the rest of the script needs.
    return model_id, credentials, client, project_id


# %% ---------------------------------------------------------------------------
# SECTION 4: RUN THE SETUP
# -----------------------------------------------------------------------------
# Tuple unpacking: the function returns 4 values and we assign each to its own
# variable in one line.
model_id, credentials, client, project_id = setup_credentials()


# %% ---------------------------------------------------------------------------
# SECTION 5: IMAGE LOCATIONS
# -----------------------------------------------------------------------------
# Four sample images hosted on IBM Cloud Object Storage (from a course lab).
# Each one is used for a different kind of question later on:
#   image 1 : a general photo            -> "Describe the photo"
#   image 2 : a street / parking scene   -> "How many cars are in this image?"
#   image 3 : a damaged object           -> "How severe is the damage?"
#   image 4 : a product label            -> "How much sodium is in this product?"
url_image_1 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png"
url_image_2 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png"
url_image_3 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png"
url_image_4 = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png"

# Put them in a list so we can loop over them. Order matters: index 0 is
# image 1, index 1 is image 2, and so on (Python lists start at 0).
image_urls = [url_image_1, url_image_2, url_image_3, url_image_4]


# %% ---------------------------------------------------------------------------
# SECTION 6: MODEL GENERATION PARAMETERS
# -----------------------------------------------------------------------------
# TextChatParameters controls HOW the model generates its answer.
# Parameters you did not set stay as None, meaning "use the service default".
#
# Commonly used settings:
#   temperature : Randomness. 0 = very deterministic/focused, higher = more
#                 varied/creative. 0.2 is low, good for factual image Q&A.
#   top_p       : "Nucleus sampling". The model only considers the smallest
#                 set of next-word candidates whose probabilities add up to
#                 top_p. 0.5 = only the most likely half of the probability
#                 mass. Lower = more conservative. It is usually best to
#                 tune EITHER temperature OR top_p, not both aggressively.
#   max_tokens  : Hard cap on the length of the reply (not set here).
#
# Optional: print a table of every supported parameter and its meaning.
TextChatParameters.show()

params = TextChatParameters(
    temperature=0.2,
    top_p=0.5,
)

# In a notebook, a bare variable name on the last line of a cell displays it.
# In a plain script that line does nothing, so we print explicitly.
print(params)


# %% ---------------------------------------------------------------------------
# SECTION 7: CREATE THE MODEL OBJECT
# -----------------------------------------------------------------------------
# ModelInference ties together everything configured so far:
#   model_id    -> WHICH model to call
#   credentials -> HOW to authenticate
#   project_id  -> WHICH watsonx.ai project to run it under
#   params      -> default generation settings for every request
#
# Creating this object does not generate anything yet; calls to
# model.chat(...) later do the actual work.
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)


# %% ---------------------------------------------------------------------------
# SECTION 8: HELPER - DOWNLOAD IMAGES AND ENCODE AS BASE64
# -----------------------------------------------------------------------------
# WHY BASE64?
# A chat request to the model is JSON, which is text. Images are binary bytes,
# which cannot be placed directly inside JSON. Base64 is a standard way to
# represent binary data using only safe text characters (A-Z, a-z, 0-9, +, /).
# The model server decodes it back into the original image.
def encode_images_to_base64(image_urls):
    """
    Download each image URL and convert its bytes to a base64 string.

    Parameters
    ----------
    image_urls : list[str]
        URLs of the images to download.

    Returns
    -------
    list[str | None]
        One entry per URL, in the SAME ORDER as `image_urls`.
        - A base64 string if the download succeeded.
        - None if it failed. Keeping a placeholder (instead of skipping)
          means indexes still line up with the original URL list.
    """
    encoded_images = []

    for url in image_urls:
        # HTTP GET request. `timeout` stops the script from hanging forever
        # if the server never answers (the original had no timeout).
        # We catch network-level errors (DNS failure, timeout, no internet)
        # so one bad URL does not crash the whole run.
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            print(f"Warning: Could not reach {url} ({exc})")
            encoded_images.append(None)
            continue  # Skip to the next URL.

        # Status code 200 means "OK". Anything else (404 not found, 403
        # forbidden, 500 server error, ...) means we did not get the image.
        if response.status_code == 200:
            # response.content : the raw image bytes.
            # b64encode        : bytes -> base64-encoded BYTES.
            # .decode("utf-8") : base64 bytes -> a normal Python string,
            #                    which is what JSON requires.
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
        else:
            print(
                f"Warning: Failed to fetch image from {url} "
                f"(Status code: {response.status_code})"
            )
            encoded_images.append(None)

    return encoded_images


# %% ---------------------------------------------------------------------------
# SECTION 9: HELPER - ASK THE MODEL A QUESTION ABOUT ONE IMAGE
# -----------------------------------------------------------------------------
def generate_model_response(
    encoded_image,
    user_query,
    assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: ",
    mime_type="image/png",
):
    """
    Send one image plus one question to the model and return its answer.

    Parameters
    ----------
    encoded_image : str
        Base64-encoded image (as produced by encode_images_to_base64).
    user_query : str
        The question about the image, e.g. "How many cars are in this image?"
    assistant_prompt : str, optional
        Instruction text placed BEFORE the question. It steers tone and
        length. The default asks for brief 1-2 sentence answers.
    mime_type : str, optional
        The image's format, used to build the "data URI" below. All four
        sample images are PNG files, so the default is "image/png".
        (The original code hard-coded "image/jpeg", which mislabelled PNGs.
        Some servers tolerate that; others reject or misread the image.)

    Returns
    -------
    str
        The model's text answer.
    """

    # THE MESSAGE FORMAT
    # ------------------
    # watsonx.ai uses the OpenAI-style "chat" format: a list of messages,
    # each with a "role" ("user", "assistant", or "system") and "content".
    #
    # For multimodal input, "content" is itself a LIST of parts, so a single
    # user message can mix text and images:
    #   - {"type": "text", "text": ...}        -> a piece of text
    #   - {"type": "image_url", "image_url": {"url": ...}} -> an image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    # Part 1: the instructions + the actual question,
                    # joined into one string.
                    "type": "text",
                    "text": assistant_prompt + user_query,
                },
                {
                    # Part 2: the image.
                    #
                    # A "data URI" embeds the file directly in the string:
                    #     data:<mime type>;base64,<the base64 text>
                    # so we do not have to host the image anywhere.
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}",
                    },
                },
            ],
        }
    ]

    # Send the request and wait for the reply. This is a network call and may
    # take several seconds. Parameters set in Section 6 are applied here.
    response = model.chat(messages=messages)

    # THE RESPONSE SHAPE
    # ------------------
    # `response` is a dictionary like:
    #   {"choices": [ {"message": {"role": "assistant", "content": "..."}} ], ...}
    # "choices" is a list because a request can ask for several alternative
    # answers (parameter `n`). We asked for one, so we take index [0].
    return response["choices"][0]["message"]["content"]


# %% ---------------------------------------------------------------------------
# SECTION 10: DOWNLOAD AND ENCODE ALL IMAGES
# -----------------------------------------------------------------------------
# Result: a list of 4 base64 strings (or None for any failed download).
encoded_images = encode_images_to_base64(image_urls)


# %% ---------------------------------------------------------------------------
# SECTION 11: DESCRIBE EVERY IMAGE
# -----------------------------------------------------------------------------
# The same generic question is asked about all four images, so we can see how
# the model describes each one.
user_query = "Describe the photo"

# enumerate() gives us BOTH the position (i) and the item (image) at once,
# which is cleaner than range(len(...)) with manual indexing.
# The second argument, 1, makes the counter start at 1 instead of 0 so the
# printed labels read "image 1 ... image 4".
for i, image in enumerate(encoded_images, start=1):

    # Guard: if the download failed earlier, `image` is None. Sending None to
    # the model would crash, so we report and move on.
    if image is None:
        print(f"Skipping image {i}: it could not be downloaded.\n")
        continue

    response = generate_model_response(image, user_query)

    # Print the response with a formatted description.
    # "\n\n" adds a blank line between results for readability.
    print(f"Description for image {i}: {response}\n\n")


# %% ---------------------------------------------------------------------------
# SECTION 12: TARGETED QUESTIONS - ONE PER IMAGE
# -----------------------------------------------------------------------------
# Vision models can answer SPECIFIC questions, not just describe. Each cell
# below picks one image by its list index (remember: index 1 = image 2) and
# asks a question suited to what that image contains.

# --- Image 2: counting objects ----------------------------------------------
image = encoded_images[1]
user_query = "How many cars are in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))


# %% --- Image 3: judging a condition -------------------------------------------
image = encoded_images[2]
user_query = "How severe is the damage in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))


# %% --- Image 4: reading text from a label (OCR-style task) -----------------
# Here the model must read the printed nutrition facts in the picture and
# extract one value. Models can misread small print, so verify important
# numbers against the original image.
image = encoded_images[3]
user_query = "How much sodium is in this product?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
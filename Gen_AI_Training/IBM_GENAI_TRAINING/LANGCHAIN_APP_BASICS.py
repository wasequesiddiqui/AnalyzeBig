"""Handshake test: confirms the .env credentials work against IBM watsonx.ai.

WHY THIS FILE EXISTS
--------------------
Almost every failure a beginner hits with watsonx.ai is NOT a code bug. It is one
of a small set of credential/account problems that all look similar from the
outside ("it doesn't work"). This script deliberately walks the credential chain
one link at a time so that the *first* thing that breaks is printed on its own,
with the exception type and the raw message from IBM.

WHAT "HANDSHAKE" MEANS HERE
--------------------------
Talking to watsonx.ai is a multi-step negotiation, not a single API call:

    1. Your API key is exchanged for a short-lived IAM bearer token.
       (IBM's Identity and Access Management service issues the token.)
    2. That token is used to resolve the *project* you named, proving both that
       the project exists and that your account is allowed to use it.
    3. Only then is the actual model invocation sent.

If any link is broken the call fails, but the error text tells you which one.
A successful "generation" (step 3) is therefore proof that steps 1 and 2 also
succeeded - that is what makes this a genuine end-to-end handshake test.

THE TWO STAGES
--------------
Stage 1 goes through ``ibm-watsonx-ai``, the low-level SDK that LangChain itself
calls under the hood. Stage 2 goes through ``langchain-ibm``, the LangChain
wrapper. Running both isolates *where* a problem lives:

  * Stage 1 fails  -> credentials, region, project, or model access are wrong.
                      Nothing about LangChain is at fault. Fix stage 1 first.
  * Stage 1 passes,
    but stage 2 fails
                   -> your key, region and project are all good, so the problem
                      is LangChain-level integration or versioning. A very
                      common cause is a version mismatch: the API surface of
                      ``langchain-ibm`` changes between releases (see the note
                      on ``WatsonxLLM`` further down).

Stage 2 is intentionally skipped when stage 1 fails. That keeps the output
unambiguous: you never have to guess whether the second error is a new problem
or just the first problem wearing a hat.

HOW TO READ THE RESULT
----------------------
Exit code 0  -> both stages produced a model response: the credentials work.
Exit code 1  -> something failed; the failing stage is named and the reason is
                printed underneath it.

RECOGNISING THE COMMON FAILURES
-------------------------------
The SDK raises long, JSON-shaped errors. The important part is usually the
leading IBM error id, for example:

  * ``BXNIM0415E``  "Provided API key could not be found."
        The key itself is not recognised by IBM, or it belongs to a different
        IBM Cloud account than the one you think. IBM Cloud API keys are 44
        characters; a shorter string usually means a truncated copy/paste.

  * ``WSCPA0000E`` with ``code: 401`` and
    "Failed to verify user profile existance: <your email> ... 404"
        The key is valid, but that account has no watsonx.ai profile *in the
        region named by WATSONX_URL*. Each region is provisioned separately;
        a key that works in one region can be rejected in another.

  * ``WSCPA0000E`` with ``code: 400`` "Invalid project GUID encountered"
        The project id is malformed. A watsonx.ai project id is a plain
        36-character UUID with no prefix.

  * ``WSCPA404PE`` with ``code: 404`` "Failed to retrieve project: <uuid>
    ... not_found: missing"
        The GUID is well-formed but no project with that id exists in the
        region/account being used - usually the id came from a different
        account, or was copied from course material rather than your own
        project.

  * "The specified url is not valid."
        WATSONX_URL points at a region that does not host the foundation-model
        runtime. The SDK accepts only us-south, eu-gb, eu-de and jp-tok.

REQUIREMENTS
------------
Dependencies are pinned in requirements.txt and already installed in the local
.venv. Read the section notes further down for the version-sensitive details.
"""

# ``os`` is used to pull the three WATSONX_* values out of the process
# environment. dotenv reads .env and *populates* os.environ, so after the call
# below the .env file and real environment variables are indistinguishable here.
#%%
import os
import warnings

# ``Credentials`` holds the two things needed to obtain an IAM token: the
# regional service endpoint (url) and the API key. ``ModelInference`` is the
# foundation-model client you actually call to generate text.
# ``WatsonxLLM`` is the LangChain wrapper around the very same SDK.
# Version note: requirements.txt pins langchain-ibm==0.1.7. In that release the
# generational chat class does not exist yet - there is no ``ChatWatsonx``, only
# ``WatsonxLLM`` (and ``WatsonxEmbeddings``). That is why this file imports
# ``WatsonxLLM`` and why the argument below is spelled ``apikey=`` and not
# ``api_key=``. Later releases renamed the field and added ``ChatWatsonx``; if
# you upgrade langchain-ibm, expect to change both the import and the keyword.

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain_ibm import WatsonxLLM

#%%
# Load .env from this script's directory.
#
# override=True is a deliberate choice, not the library default. By default
# dotenv NEVER overwrites a variable that already exists in the environment
# (that is the "12-factor" convention: real environment variables win). The
# consequence is a very confusing failure mode: a stale, machine-level
# WATSONX_APIKEY silently shadows the value you just wrote into .env, and the
# script appears to ignore your file. Passing override=True makes this project's
# .env authoritative, so what you see in the file is exactly what gets used.
load_dotenv(override=True)

# The model to invoke. Must exist in the region named by WATSONX_URL, and the
# project must be granted access to it.
MODEL_ID = "ibm/granite-4-h-small"

# The three settings this script cannot run without. Keeping them in one tuple
# means the validation loop below and the callers can never drift apart.
REQUIRED_VARS = ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL")

# Kept tiny on purpose: this is a connectivity check, not a demo. A short output
# cap keeps the handshake fast and cheap while still proving generation works.
PROMPT = "Reply with exactly two words: handshake ok"

#%% Read settings
def read_settings():
    """Return the three credential values, failing fast with a clear message.

    Reading configuration lazily inside a function (rather than at import time)
    keeps the module importable - useful for tests and tooling - and turns a
    missing variable into an actionable message instead of an ``AttributeError``
    or a confusing authentication error much later in the call stack.
    """
    # Build the name -> value mapping in one pass over REQUIRED_VARS.
    settings = {name: os.getenv(name) for name in REQUIRED_VARS}

    # os.getenv returns None for an unset variable and "" for one that is set to
    # an empty string; both are unusable here, so a falsy check covers both.
    missing = [name for name, value in settings.items() if not value]
    if missing:
        # SystemExit (rather than a custom exception class) is the right signal
        # for a top-level CLI-style script: it terminates with a non-zero exit
        # code and prints the message without a traceback. A traceback here
        # would only add noise - there is no bug to debug, just a missing entry.
        raise SystemExit(f"Missing values in .env: {', '.join(missing)}")
    return settings


def handshake_with_sdk(settings):
    """Stage 1: drive the raw ibm-watsonx-ai SDK end to end.

    Doing this through the SDK (rather than raw HTTP) means the exact same code
    path LangChain uses is exercised, so a pass here rules out credential and
    project problems for every higher-level wrapper too.
    """
    # Credentials is a value object - constructing it performs no network I/O
    # and does not validate the key. The handshake happens when it is used below.
    credentials = Credentials(url=settings["WATSONX_URL"], api_key=settings["WATSONX_APIKEY"])

    # Constructing ModelInference is the real first network step, and it can be
    # surprising for newcomers: this call already reaches out to IBM twice.
    #   1. It exchanges the API key for an IAM access token.
    #   2. It selects the default project, which resolves the project GUID in the
    #      given region and verifies the account may use it.
    # So errors about the *key* or the *project* surface here, before any text is
    # generated - which is exactly the early, precise signal this script wants.
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=settings["WATSONX_PROJECT_ID"],
        # max_new_tokens caps how many tokens the model may produce. Without a
        # cap the model could keep generating up to its own default limit, which
        # makes a trivial connectivity check needlessly slow.
        params={"max_new_tokens": 16},
    )

    # The actual inference request: this is the step that proves the model is
    # reachable and that the project has been granted access to it.
    return model.generate_text(prompt=PROMPT)


def handshake_with_langchain(settings):
    """Stage 2: drive the same handshake through the LangChain wrapper.

    This is the layer real applications use (chains, agents, prompt templates).
    It constructs its own SDK client internally from the arguments below, so it
    repeats the same IAM/project negotiation - here it also confirms the pinned
    LangChain packages are wired together correctly.
    """
    llm = WatsonxLLM(
        model_id=MODEL_ID,
        # NOTE: spelled ``apikey`` in langchain-ibm 0.1.7. The keyword is part of
        # the version-sensitive surface mentioned next to the import above.
        apikey=settings["WATSONX_APIKEY"],
        url=settings["WATSONX_URL"],
        project_id=settings["WATSONX_PROJECT_ID"],
        params={"max_new_tokens": 16},
    )

    # ``invoke`` is the standard LangChain entry point. Because WatsonxLLM is a
    # plain text-completion LLM (not a chat model), the return value is a string
    # rather than a message object - hence it is printed directly below.
    return llm.invoke(PROMPT)
#%% Custom steps
load_dotenv(override=True)

model_id = "ibm/granite-4-h-small"

parameters = {
    "max_new_tokens": 100,
    "min_new_tokens": 1,
    "temperature": 0.5,
}

credentials = Credentials(
    url="https://eu-de.ml.cloud.ibm.com",
    api_key=os.environ["WATSONX_APIKEY"],
)

project_id = os.environ["WATSONX_PROJECT_ID"]

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=project_id,
)

print(model.generate_text(prompt="Explain what a foundation model is in two sentences."))

#%%
# Send a text-completion prompt to the model. The model continues the sentence
# from where it stops. generate() returns the full response as a dictionary
# (generated text plus metadata such as token counts and stop reason).
msg = model.generate("In today's sales meeting, we ")

# The dictionary has a 'results' list with one entry per prompt. Take the first
# entry ([0]) and print only its 'generated_text', the model's continuation.
print(msg['results'][0]['generated_text'])

#%%
llama_llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    url="https://eu-de.ml.cloud.ibm.com",
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={"max_new_tokens": 100, "min_new_tokens": 1},
)

print(llama_llm.invoke("Who is man's best friend?"))

#%%
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me one {adjective} joke about {topic}")
input_ = {"adjective": "funny", "topic": "cats"}  # create a dictionary to store the corresponding input to placeholders in prompt template
prompt.invoke(input_)

#%%
# Import the ChatPromptTemplate class from langchain_core.prompts module
from langchain_core.prompts import ChatPromptTemplate

# Create a ChatPromptTemplate with a list of message tuples
# Each tuple contains a role ("system" or "user") and the message content
# The system message sets the behavior of the assistant
# The user message includes a variable placeholder {topic} that will be replaced later
prompt = ChatPromptTemplate.from_messages([
 ("system", "You are a helpful assistant"),
 ("user", "Tell me a joke about {topic}")
])

# Create a dictionary with the variable to be inserted into the template
# The key "topic" matches the placeholder name in the user message
input_ = {"topic": "cats"}

# Format the chat template with our input values
# This replaces {topic} with "cats" in the user message
# The result will be a formatted chat message structure ready to be sent to a model
prompt.invoke(input_)

# Import MessagesPlaceholder for including multiple messages in a template
from langchain_core.prompts import MessagesPlaceholder
# Import HumanMessage for creating message objects with specific roles
from langchain_core.messages import HumanMessage

# Create a ChatPromptTemplate with a system message and a placeholder for multiple messages
# The system message sets the behavior for the assistant
# MessagesPlaceholder allows for inserting multiple messages at once into the template
prompt = ChatPromptTemplate.from_messages([
("system", "You are a helpful assistant"),
MessagesPlaceholder("msgs")  # This will be replaced with one or more messages
])

# Create an input dictionary where the key matches the MessagesPlaceholder name
# The value is a list of message objects that will replace the placeholder
# Here we're adding a single HumanMessage asking about the day after Tuesday
input_ = {"msgs": [HumanMessage(content="What is the day after Tuesday?")]}

# Format the chat template with our input dictionary
# This replaces the MessagesPlaceholder with the HumanMessage in our input
# The result will be a formatted chat structure with a system message and our human message
prompt.invoke(input_)

#%%
chain = prompt | llama_llm
response = chain.invoke(input = input_)
print(response)

#%%
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

# Create your JSON parser
json_parser = JsonOutputParser()

# Create more explicit format instructions
format_instructions = """RESPONSE FORMAT: Return ONLY a single JSON object—no markdown, no examples, no extra keys.  It must look exactly like:
{
  "title": "movie title",
  "director": "director name",
  "year": 2000,
  "genre": "movie genre"
}

IMPORTANT: Your response must be *only* that JSON.  Do NOT include any illustrative or example JSON."""

# Create your prompt template with clearer instructions
prompt_template = PromptTemplate(
    template="""You are a JSON-only assistant.

Task: Generate info about the movie "{movie_name}" in JSON format.

{format_instructions}
""",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": format_instructions},
)

# Create the chain without cleaning step
movie_chain = prompt_template | llama_llm | json_parser

# Test with a movie name
movie_name = "The Matrix"
result = movie_chain.invoke({"movie_name": movie_name})

# Print the structured result
print("Parsed result:")
print(f"Title: {result['title']}")
print(f"Director: {result['director']}")
print(f"Year: {result['year']}")
print(f"Genre: {result['genre']}")

#%% Main entry point
def main():
    """Run both stages in order and report the first failure precisely.

    Returns 0 only when every stage produced a response; otherwise it returns 1.
    The value is propagated to the process exit code by the ``__main__`` guard
    at the bottom of this file, so the script can be used as a health check in a
    pipeline (``if script; then ...``).
    """
    settings = read_settings()

    # --- Stage 1: raw SDK -------------------------------------------------
    # Printing the endpoint makes region problems obvious at a glance: the URL
    # must be the region the project actually lives in.
    print(f"[1/2] ibm-watsonx-ai  ->  {settings['WATSONX_URL']}")
    try:
        # repr (via !r) reveals whitespace and the empty string, which a bare
        # print would hide - handy when a model returns an empty completion.
        print(f"      response: {handshake_with_sdk(settings)!r}")
    except Exception as exc:
        # Broad on purpose. The SDK raises several unrelated exception types for
        # what is, from the user's point of view, the same question: does this
        # credential combination work? Printing the class name plus the original
        # message preserves IBM's error id (e.g. BXNIM0415E) for diagnosis while
        # keeping the output to a single readable line.
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        return 1

    # --- Stage 2: LangChain wrapper --------------------------------------
    # Reached only if stage 1 succeeded, so any failure printed here is
    # genuinely about LangChain and not a knock-on from bad credentials.
    print(f"[2/2] langchain-ibm  ->  {MODEL_ID}")
    try:
        print(f"      response: {handshake_with_langchain(settings)!r}")
    except Exception as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        return 1

    # Both stages returned text, so the whole chain - key, IAM token, region,
    # project resolution and model access - is proven to work.
    print("API key handshake succeeded.")
    
    return 0

#%% __main__ guard
if __name__ == "__main__":
    # Only run when executed as a script, so importing this module (e.g. from a
    # notebook or a test) has no side effects beyond loading .env.
    #
    # SystemExit(main()) turns the return value into the process exit code and
    # avoids printing a traceback for an expected, already-explained failure.
    raise SystemExit(main())

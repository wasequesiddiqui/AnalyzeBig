"""Handshake test: confirms the .env credentials work against watsonx.ai."""

import os

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxLLM

# override=True so this project's .env wins over any machine-level WATSONX_* variables
load_dotenv(override=True)

MODEL_ID = "ibm/granite-4-h-small"
REQUIRED_VARS = ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL")
PROMPT = "Reply with exactly two words: handshake ok"


def read_settings():
    settings = {name: os.getenv(name) for name in REQUIRED_VARS}
    missing = [name for name, value in settings.items() if not value]
    if missing:
        raise SystemExit(f"Missing values in .env: {', '.join(missing)}")
    return settings


def handshake_with_sdk(settings):
    credentials = Credentials(url=settings["WATSONX_URL"], api_key=settings["WATSONX_APIKEY"])
    model = ModelInference(
        model_id=MODEL_ID,
        credentials=credentials,
        project_id=settings["WATSONX_PROJECT_ID"],
        params={"max_new_tokens": 16},
    )
    return model.generate_text(prompt=PROMPT)


def handshake_with_langchain(settings):
    llm = WatsonxLLM(
        model_id=MODEL_ID,
        apikey=settings["WATSONX_APIKEY"],
        url=settings["WATSONX_URL"],
        project_id=settings["WATSONX_PROJECT_ID"],
        params={"max_new_tokens": 16},
    )
    return llm.invoke(PROMPT)


def main():
    settings = read_settings()

    print(f"[1/2] ibm-watsonx-ai  ->  {settings['WATSONX_URL']}")
    try:
        print(f"      response: {handshake_with_sdk(settings)!r}")
    except Exception as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"[2/2] langchain-ibm  ->  {MODEL_ID}")
    try:
        print(f"      response: {handshake_with_langchain(settings)!r}")
    except Exception as exc:
        print(f"      FAILED: {type(exc).__name__}: {exc}")
        return 1

    print("API key handshake succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

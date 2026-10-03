import os

from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx

# override=True so this project's .env wins over any machine-level WATSONX_* variables
load_dotenv(override=True)

REQUIRED_VARS = ("WATSONX_APIKEY", "WATSONX_PROJECT_ID", "WATSONX_URL")
PLACEHOLDER_PREFIX = "your-"


def get_credentials():
    values = {name: os.getenv(name) for name in REQUIRED_VARS}
    unset = [name for name, value in values.items() if not value]
    if unset:
        raise RuntimeError(
            f"Missing {', '.join(unset)}. Set them in the .env file next to this script."
        )
    if values["WATSONX_APIKEY"].startswith(PLACEHOLDER_PREFIX):
        raise RuntimeError("WATSONX_APIKEY in .env is still the placeholder value.")
    return values


def build_llm(credentials):
    return ChatWatsonx(
        model_id="ibm/granite-3-8b-instruct",
        url=credentials["WATSONX_URL"],
        project_id=credentials["WATSONX_PROJECT_ID"],
        api_key=credentials["WATSONX_APIKEY"],
        params={"temperature": 0.7, "max_new_tokens": 512},
    )


def main():
    credentials = get_credentials()
    llm = build_llm(credentials)

    response = llm.invoke("Hello! Briefly introduce yourself in two sentences.")
    print(response.content)


if __name__ == "__main__":
    main()

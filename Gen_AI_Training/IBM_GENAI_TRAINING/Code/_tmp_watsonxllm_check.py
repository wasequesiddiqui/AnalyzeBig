import os

from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_ibm import WatsonxLLM

llm = WatsonxLLM(
    model_id="ibm/granite-4-h-small",
    url=os.environ["WATSONX_URL"],
    apikey=os.environ["WATSONX_APIKEY"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params={"max_new_tokens": 1, "min_new_tokens": 1},
)
result = llm.invoke("Say hi.")
print("type:", type(result))
print("has .content:", hasattr(result, "content"))
print("value:", repr(result)[:60])

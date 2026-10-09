#%%
import os

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# %%
load_dotenv(override=True)

credentials = Credentials(
    url=os.environ["WATSONX_URL"],
    api_key=os.environ["WATSONX_APIKEY"],
)

# %%

params = {
    GenTextParamsMetaNames.DECODING_METHOD: "greedy",
	GenTextParamsMetaNames.MAX_NEW_TOKENS: 100
}

#%%
model = ModelInference(
    model_id='ibm/granite-4-h-small',
    params=params,
    credentials=credentials,
    project_id=os.environ["WATSONX_PROJECT_ID"]
)

#%%
text = """
Only reply with the answer. What is the capital of Canada?
"""

print(model.generate(text)['results'][0]['generated_text'])

# %%
model = ModelInference(
    model_id='meta-llama/llama-4-maverick-17b-128e-instruct-fp8',
    params=params,
    credentials=credentials,
    project_id=os.environ["WATSONX_PROJECT_ID"]
)
# %%
text = """
Only reply with the answer. What is the capital of Canada?
"""

print(model.generate(text)['results'][0]['generated_text'])
# %%

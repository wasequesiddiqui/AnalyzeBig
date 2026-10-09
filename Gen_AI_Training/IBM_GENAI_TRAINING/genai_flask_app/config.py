#%%
import os
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

#%%
# Model parameters
PARAMETERS = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 256,
}

# watsonx credentials
# Note: Normally we'd need an API key, but in Skill's Network Cloud IDE will automatically handle that for you.
CREDENTIALS = {
    "url": os.environ["WATSONX_URL"],
    "project_id": os.environ["WATSONX_PROJECT_ID"]
}

# Model IDs
LLAMA_MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
GRANITE_MODEL_ID = "ibm/granite-4-h-small"
MISTRAL_MODEL_ID = "mistralai/mistral-small-3-1-24b-instruct-2503"

# %%

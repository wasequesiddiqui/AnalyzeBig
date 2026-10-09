#%%
import os
from langchain_ibm import ChatWatsonx
from langchain_core.prompts import PromptTemplate
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID

# %%
# Function to initialize a model
def initialize_model(model_id):
    return ChatWatsonx(
        model_id=model_id,
        url=os.environ["WATSONX_URL"],
        project_id=os.environ["WATSONX_PROJECT_ID"],
        params=PARAMETERS
    )

# Initialize models
llama_llm = initialize_model(LLAMA_MODEL_ID)
granite_llm = initialize_model(GRANITE_MODEL_ID)
mistral_llm = initialize_model(MISTRAL_MODEL_ID)

# %%
# Prompt template
llama_template = PromptTemplate(
    template='''<|begin_of_text|><|header_start|>system<|header_end|>
{system_prompt}<|eot|><|header_start|>user<|header_end|>
{user_prompt}<|eot|><|header_start|>assistant<|header_end|>
''',
    input_variables=["system_prompt", "user_prompt"]
)

granite_template = PromptTemplate(
    template="<|system|>{system_prompt}\n\<|user|>{user_prompt}\n<|assistant|>",
    input_variables=["system_prompt", "user_prompt"]
)

mistral_template = PromptTemplate(
    template="<s>[INST]{system_prompt}\n{user_prompt}[/INST]",
    input_variables=["system_prompt", "user_prompt"]
)

# %%
def get_ai_response(model, template, system_prompt, user_prompt):
    chain = template | model
    return chain.invoke({'system_prompt':system_prompt, 'user_prompt':user_prompt})

# %%
def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, llama_template, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, granite_template, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, mistral_template, system_prompt, user_prompt)

# %%

#%%
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # must come first

from langchain_ibm import ChatWatsonx
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from config import PARAMETERS, LLAMA_MODEL_ID, GRANITE_MODEL_ID, MISTRAL_MODEL_ID

#%%
# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

#%%
# JSON output parser
json_parser = JsonOutputParser(pydantic_object=AIResponse)

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
def get_ai_response(model, system_prompt, user_prompt):
    structured = model.with_structured_output(AIResponse)
    result = structured.invoke([
        ("system", system_prompt),
        ("human", user_prompt),
    ])
    return result.model_dump()

# %%
def llama_response(system_prompt, user_prompt):
    return get_ai_response(llama_llm, system_prompt, user_prompt)

def granite_response(system_prompt, user_prompt):
    return get_ai_response(granite_llm, system_prompt, user_prompt)

def mistral_response(system_prompt, user_prompt):
    return get_ai_response(mistral_llm, system_prompt, user_prompt)

# %%

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create client
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Make a request (equivalent to invoking the model)
response = llm.responses.create(
    model="gpt-4.1-mini",  # choose model
    input="What is tool calling in langchain?"
)

print(response.output[0].content[0].text)
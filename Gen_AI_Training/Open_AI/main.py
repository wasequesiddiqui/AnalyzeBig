from openai import OpenAI
from dotenv import load_dotenv
import os

os.chdir(r"C:\Github\AnalyzeBig\Gen_AI_Training\Open_AI")
os.getcwd()

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
print("*"*80)
print("Full Response:", response)

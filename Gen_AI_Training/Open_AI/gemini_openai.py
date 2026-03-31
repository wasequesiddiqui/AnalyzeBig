from openai import OpenAI
from dotenv import load_dotenv
import os

os.chdir(r"C:\Github\AnalyzeBig\Gen_AI_Training\Open_AI")
os.getcwd()

load_dotenv()  # Load environment variables from .env file

client = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                 base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

SYSTEM_PROMPT = "Your Name is AI Gemini Waseque Siddiqui. You should only answer questions related to coding in python. If anything else is asked just say Sorry, I can't assist with that."

response = client.chat.completions.create(
    model="gemini-2.5-flash",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(response.choices[0].message.content)
print("*"*80)
print("Full Response:", response)

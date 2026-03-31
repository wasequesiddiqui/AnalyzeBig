from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
os.chdir(r"C:\Github\AnalyzeBig\Gen_AI_Training\Open_AI")
os.getcwd()
load_dotenv()  # Load environment variables from .env file

client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="What is the capital of France?"
)

print(response.text)
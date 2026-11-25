#%%
import os
import requests
import base64

#%%
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials, APIClient
from dataclasses import dataclass
from pydantic import BaseModel
from ibm_watsonx_ai.foundation_models.schema._api import BaseSchema
from ibm_watsonx_ai.foundation_models.schema import TextChatResponseFormat
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from typing import Any
from ibm_watsonx_ai.foundation_models import ModelInference

#%%
load_dotenv()

#%%
def setup_credentials():
    # Define the model ID for the WatsonX model being used
    model_id = 'meta-llama/llama-4-maverick-17b-128e-instruct-fp8'

    # Read configuration from environment variables
    api_key = os.getenv("WATSONX_APIKEY")
    url = os.getenv("WATSONX_URL")
    project_id = os.getenv("WATSONX_PROJECT_ID")

    # Fail early with a clear message if anything is missing
    missing = [
        name for name, value in {
            "WATSONX_APIKEY": api_key,
            "WATSONX_URL": url,
            "WATSONX_PROJECT_ID": project_id,
        }.items() if not value
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

    # Set up the credentials with the URL and API key
    credentials = Credentials(url=url, api_key=api_key)

    # Create an API client using the credentials
    client = APIClient(credentials)

    # Return the model ID, credentials, client, and project ID for later use
    return model_id, credentials, client, project_id

#%%
model_id, credentials, client, project_id = setup_credentials()

# %%
url_image_1 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/5uo16pKhdB1f2Vz7H8Utkg/image-1.png'
url_image_2 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/fsuegY1q_OxKIxNhf6zeYg/image-2.png'
url_image_3 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/KCh_pM9BVWq_ZdzIBIA9Fw/image-3.png'
url_image_4 = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VaaYLw52RaykwrE3jpFv7g/image-4.png'

#%%
image_urls = [url_image_1, url_image_2, url_image_3, url_image_4]

# %%
@dataclass
class TextChatParameters(BaseSchema):
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    response_format: dict | TextChatResponseFormat | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    time_limit: int | None = None
    top_p: float | None = None
    n: int | None = None

@classmethod
def get_sample_params(cls) -> dict[str, Any]:
    """Provide example values for TextChatParameters."""
    return {
        "frequency_penalty": 0.5,
        "logprobs": True,
        "top_logprobs": 3,
        "presence_penalty": 0.3,
        "response_format": TextChatResponseFormat.get_sample_params(),
        "temperature": 0.7,
        "max_tokens": 100,
        "time_limit": 600000,
        "top_p": 0.9,
        "n": 1,
    }

#%%
TextChatParameters.show()
params = TextChatParameters(
    temperature=0.2,
    top_p=0.5,

)
params

#%%
model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params
)

#%%
def encode_images_to_base64(image_urls):
    """
    Downloads and encodes a list of image URLs to base64 strings.

    Parameters:
    - image_urls (list): A list of image URLs.

    Returns:
    - list: A list of base64-encoded image strings.
    """
    encoded_images = []
    for url in image_urls:
        response = requests.get(url)
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            encoded_images.append(encoded_image)
            print(type(encoded_image))
        else:
            print(f"Warning: Failed to fetch image from {url} (Status code: {response.status_code})")
            encoded_images.append(None)
    return encoded_images# %%

# %%
def generate_model_response(encoded_image, user_query, assistant_prompt="You are a helpful assistant. Answer the following user query in 1 or 2 sentences: "):
    """
    Sends an image and a query to the model and retrieves the description or answer.

    Parameters:
    - encoded_image (str): Base64-encoded image string.
    - user_query (str): The user's question about the image.
    - assistant_prompt (str): Optional prompt to guide the model's response.

    Returns:
    - str: The model's response for the given image and query.
    """
    
    # Create the messages object
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": assistant_prompt + user_query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + encoded_image,
                    }
                }
            ]
        }
    ]
    
    # Send the request to the model
    response = model.chat(messages=messages)
    
    # Return the model's response
    return response['choices'][0]['message']['content']

#%%
encoded_images = encode_images_to_base64(image_urls)

#%%
user_query = "Describe the photo"

for i in range(len(encoded_images)):
    image = encoded_images[i]

    response = generate_model_response(image, user_query)

    # Print the response with a formatted description
    print(f"Description for image {i + 1}: {response}\n\n")

#%%
image = encoded_images[1]
user_query = "How many cars are in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))

# %%
image = encoded_images[2]

user_query = "How severe is the damage in this image?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))

# %%
image = encoded_images[3]

user_query = "How much sodium is in this product?"

print("User Query: ", user_query)
print("Model Response: ", generate_model_response(image, user_query))
# %%

#%%
import os
import re
import html
import base64

#%%
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters


def load_env_config():
    """Load and validate settings from the .env file."""
    load_dotenv()

    config = {
        "api_key": os.getenv("WATSONX_APIKEY"),
        "url": os.getenv("WATSONX_URL"),
        "project_id": os.getenv("WATSONX_PROJECT_ID"),
    }

    missing = [
        name
        for name, value in {
            "WATSONX_APIKEY": config["api_key"],
            "WATSONX_URL": config["url"],
            "WATSONX_PROJECT_ID": config["project_id"],
        }.items()
        if not value
    ]
    if missing:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")

    config["secret_key"] = os.getenv("FLASK_SECRET_KEY", "dev-only-change-me")
    return config


config = load_env_config()

app = Flask(__name__)
app.secret_key = config["secret_key"]  # required for flash()
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # reject uploads over 8 MB

### Step 2: credentials and model
MODEL_ID = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"

credentials = Credentials(url=config["url"], api_key=config["api_key"])

model = ModelInference(
    model_id=MODEL_ID,
    credentials=credentials,
    project_id=config["project_id"],
    params=TextChatParameters(temperature=0.2, top_p=0.5, max_tokens=1000),
)

### Step 3: the nutritionist prompt
ASSISTANT_PROMPT = """
You are an expert nutritionist. Your task is to analyze the food items displayed in the image and provide a detailed nutritional assessment using the following format:

1. **Identification**: List each identified food item clearly, one per line.
2. **Portion Size & Calorie Estimation**: For each identified food item, specify the portion size and provide an estimated number of calories. Use bullet points with the following structure:
- **[Food Item]**: [Portion Size], [Number of Calories] calories

Example:
*   **Salmon**: 6 ounces, 210 calories
*   **Asparagus**: 3 spears, 25 calories

3. **Total Calories**: Provide the total number of calories for all food items.

Example:
Total Calories: [Number of Calories]

4. **Nutrient Breakdown**: Include a breakdown of key nutrients such as **Protein**, **Carbohydrates**, **Fats**, **Vitamins**, and **Minerals**. Use bullet points, and for each nutrient provide details about the contribution of each food item.

Example:
*   **Protein**: Salmon (35g), Asparagus (3g), Tomatoes (1g) = [Total Protein]

5. **Health Evaluation**: Evaluate the healthiness of the meal in one paragraph.

6. **Disclaimer**: Include the following exact text as a disclaimer:

The nutritional information and calorie estimates provided are approximate and are based on general food data.
Actual values may vary depending on factors such as portion size, specific ingredients, preparation methods, and individual variations.
For precise dietary advice or medical guidance, consult a qualified nutritionist or healthcare provider.

Format your response exactly like the template above to ensure consistency.
"""


### Step 4: helper functions
def input_image_setup(uploaded_file):
    """
    Encode an uploaded image file as a base64 string for the model.

    Parameters:
    - uploaded_file: File-like object from request.files

    Returns:
    - str: Base64-encoded image data
    """
    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        return base64.b64encode(bytes_data).decode("utf-8")
    raise FileNotFoundError("No file uploaded")


def format_response(response_text):
    """
    Convert the model's markdown-style text into simple HTML.
    Handles **bold**, "* " bullet lists, and line breaks.
    """
    # Escape first so any HTML the model returns can't run in the page
    text = html.escape(response_text)

    # Bullets first ("* item" -> <li>), then bold, so bold inside a bullet
    # stays on the same line instead of becoming a separate paragraph
    text = re.sub(r"(?m)^[ \t]*\*[ \t]+(.*)", r"<li>\1</li>", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)

    # Wrap each run of consecutive <li> items in ONE <ul>
    text = re.sub(
        r"(?:<li>.*?</li>\s*)+",
        lambda m: "<ul>" + re.sub(r"</li>\s+<li>", "</li><li>", m.group(0).strip()) + "</ul>",
        text,
        flags=re.DOTALL,
    )

    # Remaining newlines become line breaks (no extra gap after lists)
    text = re.sub(r"(\n|\\n)+", "<br>", text)
    text = text.replace("</ul><br>", "</ul>")

    return text.strip("<br>") if False else text.strip()


def generate_model_response(encoded_image, user_query, assistant_prompt, mime_type="image/jpeg"):
    """
    Send the image and query to the model and return an HTML-formatted answer.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": assistant_prompt + "\n\n" + user_query},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64," + encoded_image},
                },
            ],
        }
    ]

    try:
        response = model.chat(messages=messages)
        raw_response = response["choices"][0]["message"]["content"]
        return format_response(raw_response)
    except Exception as e:
        print(f"Error in generating response: {e}")
        return "<p>An error occurred while generating the response.</p>"


### Step 5: the route
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_query = request.form.get("user_query") or "Analyze the food in this image."
        uploaded_file = request.files.get("file")

        if uploaded_file:
            # Only accept image uploads
            if not (uploaded_file.mimetype or "").startswith("image/"):
                flash("Please upload an image file (PNG, JPG, etc.).", "danger")
                return redirect(url_for("index"))

            encoded_image = input_image_setup(uploaded_file)

            if not encoded_image:
                flash("Error processing the image. Please try again.", "danger")
                return redirect(url_for("index"))

            response = generate_model_response(
                encoded_image,
                user_query,
                ASSISTANT_PROMPT,
                mime_type=uploaded_file.mimetype,
            )

            return render_template("index.html", user_query=user_query, response=response)

        flash("Please upload an image file.", "danger")
        return redirect(url_for("index"))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
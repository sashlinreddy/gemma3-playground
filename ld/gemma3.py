"""Example usage of the Gemma 3 model from Google GenAI SDK."""

import os

from dotenv import find_dotenv, load_dotenv
from google import genai

load_dotenv(find_dotenv())

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

response = client.models.generate_content(
    model="gemma-3-27b-it",
    contents="Roses are red...",
)

print(response.text)

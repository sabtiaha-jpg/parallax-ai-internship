import os
from dotenv import load_dotenv
from openai import OpenAI

# Load the API key from .env
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise ValueError("OPENROUTER_API_KEY was not found.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key
)

response = client.chat.completions.create(
    model="openrouter/free",
    messages=[
        {
            "role": "user",
            "content": "Say hello in one short sentence."
        }
    ]
)

print(response.choices[0].message.content)

from google import genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

print("Generative Models:")
try:
    for m in client.models.list():
        if "generateContent" in m.supported_actions: # Check for generateContent support
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")

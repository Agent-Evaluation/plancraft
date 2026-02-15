
from google import genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    # Fallback to hardcoded key if env var is missing in this context
    # (Though we will export it when running)
    pass 

client = genai.Client(api_key=api_key)

try:
    print("Listing models...")
    for m in client.models.list():
        print(f"Name: {m.name}")
        print(f"  Display Name: {m.display_name}")
        print(f"  Supported Actions: {m.supported_actions}")
        print("-" * 20)
except Exception as e:
    print(f"Error: {e}")

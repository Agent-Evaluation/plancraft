
import google.generativeai as genai
import os

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("API Key not found in env")
    exit(1)

genai.configure(api_key=api_key)

try:
    models = genai.list_models()
    print("Available Models:")
    found = False
    for m in models:
        if "generateContent" in m.supported_generation_methods:
             print(f"- {m.name}")
             if "flash" in m.name:
                 found = True
    if not found:
        print("\nNo 'flash' model found.")
except Exception as e:
    print(f"Error listing models: {e}")

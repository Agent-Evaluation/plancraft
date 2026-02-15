
import time
import os
from google import genai
from typing import Any

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
INTER_REQUEST_DELAY = 4.0
DEFAULT_API_KEY = "dummy-key-if-not-set"

def get_gemini_client(api_key: str = None) -> genai.Client:
    api_key = (
        api_key
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or DEFAULT_API_KEY
    )
    return genai.Client(api_key=api_key)

def call_gemini_with_retry(
    client: genai.Client,
    model_name: str,
    messages: list[genai.types.Content],
    system_prompt: str,
    temperature: float = 0.0,
) -> str:
    """
    Calls Gemini model with rate limiting and exponential backoff retry logic.
    """
    # Rate limiting
    time.sleep(INTER_REQUEST_DELAY)

    # Prepend system prompt as user message if needed (for models that don't support system_instruction)
    # However, the original code did this manually. Let's keep it consistent.
    system_msg = genai.types.Content(
        role="user",
        parts=[genai.types.Part(text=system_prompt)],
    )
    ack_msg = genai.types.Content(
        role="model",
        parts=[genai.types.Part(text="Understood. I will respond with exactly one action per turn.")],
    )
    
    # Check if messages already have system prompt?
    # The caller is expected to pass conversation history.
    # We will prepend system prompt here.
    full_messages = [system_msg, ack_msg] + messages

    retries = 0
    delay = INITIAL_RETRY_DELAY
    
    while retries < MAX_RETRIES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_messages,
                config=genai.types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"  ⏳ Rate limited. Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
            else:
                print(f"  ⚠ API error (attempt {retries+1}/{MAX_RETRIES}): {e}")
            
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
                
    raise Exception(f"Failed to call Gemini after {MAX_RETRIES} retries.")

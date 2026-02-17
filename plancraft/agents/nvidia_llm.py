"""
NVIDIA NIM API wrapper for Plancraft agents.

Uses the NVIDIA Integrate API (OpenAI-compatible chat completions)
to call models like Kimi K2.5.
"""

import os
import time
import requests
from typing import Optional

MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5
INTER_REQUEST_DELAY = 2.0

NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "moonshotai/kimi-k2.5"


def get_nvidia_api_key() -> str:
    """Get NVIDIA API key from environment."""
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise ValueError(
            "NVIDIA_API_KEY not set. Add it to your .env file or export it."
        )
    return key


def call_nvidia_with_retry(
    model_name: str,
    messages: list[dict],
    system_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
    api_key: Optional[str] = None,
) -> str:
    """
    Calls a model via the NVIDIA NIM API with retry logic.

    Args:
        model_name: Model identifier (e.g. "moonshotai/kimi-k2.5").
        messages: Conversation history as list of {"role": ..., "content": ...} dicts.
        system_prompt: System-level instruction.
        temperature: Sampling temperature.
        max_tokens: Max output tokens.
        api_key: Optional override for the API key.

    Returns:
        The model's response text.
    """
    if api_key is None:
        api_key = get_nvidia_api_key()

    # Rate limiting
    time.sleep(INTER_REQUEST_DELAY)

    # Build OpenAI-compatible messages
    api_messages = _build_messages(system_prompt, messages)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_name,
        "messages": api_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.00,
        "stream": False,
    }

    # Thinking mode disabled for speed — uncomment to re-enable
    # if "kimi" in model_name.lower():
    #     payload["chat_template_kwargs"] = {"thinking": True}

    retries = 0
    delay = INITIAL_RETRY_DELAY

    while retries < MAX_RETRIES:
        try:
            response = requests.post(
                NVIDIA_INVOKE_URL,
                headers=headers,
                json=payload,
                timeout=120,
            )

            if response.status_code == 429:
                print(f"  ⏳ Rate limited. Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
                retries += 1
                if retries < MAX_RETRIES:
                    time.sleep(delay)
                    delay *= 2
                continue

            response.raise_for_status()
            data = response.json()

            message = data["choices"][0]["message"]
            # Kimi thinking mode may return content in different fields
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or ""

            # If content is empty but reasoning exists, the model
            # may have put the action inside the reasoning block
            if content:
                return content.strip()
            elif reasoning:
                # Extract the last non-empty line from reasoning as the action
                lines = [l.strip() for l in reasoning.strip().splitlines() if l.strip()]
                if lines:
                    return lines[-1]

            # Debug: print raw response keys on failure
            print(f"  🔍 Debug: message keys={list(message.keys())}, content={repr(content)}, reasoning={repr(reasoning[:200] if reasoning else None)}")
            raise Exception("Empty response from NVIDIA API")

        except requests.exceptions.Timeout:
            print(f"  ⏳ Request timed out. Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
        except requests.exceptions.HTTPError as e:
            print(f"  ⚠ HTTP error (attempt {retries+1}/{MAX_RETRIES}): {e}")
        except Exception as e:
            print(f"  ⚠ API error (attempt {retries+1}/{MAX_RETRIES}): {e}")

        retries += 1
        if retries < MAX_RETRIES:
            time.sleep(delay)
            delay *= 2

    raise Exception(f"Failed to call NVIDIA API after {MAX_RETRIES} retries.")


def _build_messages(system_prompt: str, messages: list[dict]) -> list[dict]:
    """
    Build OpenAI-compatible message list from system prompt + conversation history.
    """
    api_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        role = msg["role"]
        # Normalize role names
        if role in ("model",):
            role = "assistant"
        api_messages.append({"role": role, "content": msg["content"]})

    return api_messages

"""
MassGen LiteLLM wrapper for Plancraft benchmark evaluation.

This module intentionally performs full MassGen orchestrator runs (via
`massgen/path:<config>`) for each agent action decision.
"""

import asyncio
import time
from typing import Any, Callable, Optional

import litellm
from massgen import register_with_litellm

MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 5
INTER_REQUEST_DELAY = 1.0
DEFAULT_HEARTBEAT_SECONDS = 30

_MASSGEN_REGISTERED = False


def ensure_massgen_registered() -> None:
    """Register MassGen with LiteLLM once per process."""
    global _MASSGEN_REGISTERED
    if _MASSGEN_REGISTERED:
        return
    register_with_litellm()
    _MASSGEN_REGISTERED = True


def _build_prompt(system_prompt: str, messages: list[dict]) -> str:
    """
    Build a single prompt string from system prompt + conversation history.

    Kept compatible with existing Copilot prompt structure so behavior is easy
    to compare across benchmark backends.
    """
    parts = [f"[System Instructions]\n{system_prompt}\n"]

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role in ("user",):
            parts.append(f"[Observation]\n{content}\n")
        elif role in ("model", "assistant"):
            parts.append(f"[Your Previous Action]\n{content}\n")

    parts.append("[Your Action]\nRespond with exactly one action:")
    return "\n".join(parts)


def _completion_call(prompt: str, config_path: str, enable_filesystem: bool) -> Any:
    """Sync blocking LiteLLM call (executed in a worker thread)."""
    return litellm.completion(
        model=f"massgen/path:{config_path}",
        messages=[{"role": "user", "content": prompt}],
        optional_params={"enable_filesystem": enable_filesystem},
        # Intentionally no short timeout; this benchmark allows long waits.
    )


async def call_massgen_with_retry(
    config_path: str,
    messages: list[dict],
    system_prompt: str,
    heartbeat_seconds: int = DEFAULT_HEARTBEAT_SECONDS,
    enable_filesystem: bool = False,
    log_fn: Optional[Callable[[dict], None]] = None,
    log_context: Optional[dict] = None,
) -> tuple[str, dict, float]:
    """
    Call MassGen through LiteLLM with heartbeat logging and retry logic.

    Returns:
        (action_text, metadata, elapsed_seconds)
    """
    ensure_massgen_registered()

    await asyncio.sleep(INTER_REQUEST_DELAY)
    full_prompt = _build_prompt(system_prompt, messages)

    retries = 0
    delay = INITIAL_RETRY_DELAY
    last_error: Optional[Exception] = None

    context = dict(log_context or {})
    context.update(
        {
            "prompt_chars": len(full_prompt),
            "conversation_turns": len(messages),
            "heartbeat_seconds": heartbeat_seconds,
        }
    )

    while retries < MAX_RETRIES:
        started_at = time.time()
        start_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at))
        if log_fn:
            log_fn({"event": "massgen_call_start", "timestamp": start_iso, **context})

        try:
            completion_task = asyncio.create_task(
                asyncio.to_thread(
                    _completion_call,
                    full_prompt,
                    config_path,
                    enable_filesystem,
                )
            )

            while True:
                try:
                    response = await asyncio.wait_for(
                        asyncio.shield(completion_task),
                        timeout=heartbeat_seconds,
                    )
                    break
                except asyncio.TimeoutError:
                    elapsed = time.time() - started_at
                    if log_fn:
                        log_fn(
                            {
                                "event": "massgen_call_heartbeat",
                                "elapsed_seconds": round(elapsed, 1),
                                "waiting_for_massgen": True,
                                **context,
                            }
                        )

            elapsed = time.time() - started_at
            content = response.choices[0].message.content if response and response.choices else ""
            action_text = (content or "").strip()
            if not action_text:
                raise ValueError("Empty action text from MassGen LiteLLM response")

            metadata = getattr(response, "_hidden_params", {}) or {}
            if log_fn:
                log_fn(
                    {
                        "event": "massgen_call_complete",
                        "elapsed_seconds": round(elapsed, 3),
                        "selected_agent": metadata.get("massgen_selected_agent"),
                        "vote_results": metadata.get("massgen_vote_results"),
                        "session_id": metadata.get("massgen_session_id"),
                        "log_directory": metadata.get("massgen_log_directory"),
                        **context,
                    }
                )

            return action_text, metadata, elapsed

        except Exception as exc:  # transport/setup/runtime failure
            elapsed = time.time() - started_at
            last_error = exc
            retries += 1
            retrying = retries < MAX_RETRIES
            if log_fn:
                log_fn(
                    {
                        "event": "massgen_call_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "elapsed_seconds": round(elapsed, 3),
                        "attempt": retries,
                        "max_retries": MAX_RETRIES,
                        "retrying": retrying,
                        **context,
                    }
                )
            if retrying:
                await asyncio.sleep(delay)
                delay *= 2

    raise RuntimeError(
        f"Failed to call MassGen after {MAX_RETRIES} retries. Last error: {last_error}"
    )

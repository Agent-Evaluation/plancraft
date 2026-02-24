# Replace Copilot Brain with MassGen in Plancraft Single Agent

Replace the Copilot SDK LLM backend with MassGen's async Python API (`massgen.run()`) for the single-agent plancraft evaluator, using the provided `massgen_config.yaml`.

## Current Architecture

- **`copilot_llm.py`** — Copilot SDK wrapper: `get_copilot_client()`, `call_copilot_with_retry()`, `_build_prompt()`
- **`copilot_base.py`** — Abstract base class depending on `CopilotClient`
- **`copilot_single.py`** — `CopilotSingleAgent` calls `call_copilot_with_retry()` in its `act()` method
- **`eval_copilot.py`** — Eval runner: creates a `CopilotClient`, passes it to agents, runs the loop
- **`massgen_config.yaml`** — Already provided MassGen multi-agent config (3 agents with copilot backends)

## Plan

### 1. Create `plancraft/agents/massgen_llm.py` (replaces `copilot_llm.py`)
- Wrapper around `massgen.run()` (async API)
- Function `call_massgen(query, config_path) -> str` that:
  - Calls `await massgen.run(query=prompt, config=config_path, enable_filesystem=False)`
  - Returns `result['final_answer']`
  - Includes retry/error handling
- Reuse the `_build_prompt()` logic from `copilot_llm.py` to format system prompt + conversation into a single query string

### 2. Create `plancraft/agents/massgen_base.py` (replaces `copilot_base.py`)
- Same abstract base class but removes `CopilotClient` dependency
- Constructor takes `config_path: str` (path to `massgen_config.yaml`) instead of `client: CopilotClient`
- Keeps `reset()`, `_oracle_search()`, `log()`, conversation history

### 3. Create `plancraft/agents/massgen_single.py` (replaces `copilot_single.py`)
- `MassGenSingleAgent(MassGenBaseAgent)`
- `act()` method builds the prompt string (system + conversation), calls `call_massgen()`, handles search oracle
- Same SYSTEM_PROMPT as `copilot_single.py`

### 4. Create `eval_massgen.py` in project root (replaces `eval_copilot.py`)
- No `CopilotClient` setup/teardown — MassGen handles its own lifecycle
- Instantiates `MassGenSingleAgent(config_path="plancraft/agents/massgen_config.yaml")`
- Same eval loop structure (examples → env → agent.act() → step)
- CLI args: `--split`, `--max-steps`, `--max-examples`, `--config` (path to massgen yaml)

## Key Design Decisions

- **Use `massgen.run()` direct API** (not LiteLLM) since we need async and the agent loop is already async
- **`enable_filesystem=False`** — plancraft agents don't need file operations
- **Config file path** passed to `massgen.run(config=...)` so MassGen uses the 3-agent setup from `massgen_config.yaml`
- **Single query per turn** — each `act()` call builds the full prompt and sends it as one `massgen.run()` query. MassGen's multi-agent orchestration handles the rest internally.
- No `model_name` parameter needed — models are defined in the YAML config

## MassGen API Reference (for implementation)

### `massgen.run()` signature
```python
async def run(
    query: str,
    config: str = None,        # Config file path or @examples/NAME
    model: str = None,         # Model name for agents
    models: list = None,       # List of models for multi-agent mode
    num_agents: int = None,
    use_docker: bool = False,
    enable_filesystem: bool = True,
    enable_logging: bool = False,
    output_file: str = None,
    context_paths: list = None,
    **kwargs
) -> dict
```

### Return value
```python
{
    'final_answer': str,       # The generated answer
    'config_used': str,        # Config path or description
    'session_id': str,
    'selected_agent': str,     # Winner (multi-agent)
    'vote_results': dict,      # Voting details
    'answers': list,           # All agent answers
}
```

### Usage pattern for our case
```python
import massgen

result = await massgen.run(
    query=prompt_string,
    config="./plancraft/agents/massgen_config.yaml",
    enable_filesystem=False,
)
answer = result['final_answer']
```

## Files Created (4 new files, no existing files modified)

| New File | Replaces |
|---|---|
| `plancraft/agents/massgen_llm.py` | `copilot_llm.py` |
| `plancraft/agents/massgen_base.py` | `copilot_base.py` |
| `plancraft/agents/massgen_single.py` | `copilot_single.py` |
| `eval_massgen.py` (project root) | `eval_copilot.py` |

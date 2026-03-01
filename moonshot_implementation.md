# Moonshot Kimi K2.5 Implementation via NVIDIA API

## Overview

Replaced the memory-heavy `copilot` backend with NVIDIA's hosted Moonshot Kimi K2.5 reasoning model via OpenAI-compatible API. This eliminates subprocess explosion while keeping MassGen's full orchestration capabilities.

## Architecture

```
eval_massgen.py
  → MassGenOrchestratedAgent.act()
    → self._orchestrator.chat_simple()          ← MassGen Orchestrator
      → dispatches to 3 agents (agent_a, b, c)  ← MassGen multi-agent coordination
        → ChatCompletionsBackend._call_api()     ← MassGen backend
          → openai.AsyncOpenAI(base_url="https://integrate.api.nvidia.com/v1")
            → NVIDIA serves Kimi K2.5            ← actual LLM call
      → coordination/voting between agents       ← MassGen orchestrator logic
    → returns final answer
```

**MassGen still handles all orchestration**: agent task planning, multi-agent rounds, voting/coordination, answer selection. Only the backend implementation changed.

## Configuration Changes

### Before: `copilot` backend (memory explosion)

```yaml
agents:
- id: agent_a
  backend:
    type: copilot
    model: gpt-5-mini
    cwd: workspace
    exclude_file_operation_mcps: false
orchestrator:
  enable_multimodal_tools: true
```

**Result**: 74 child processes, 5,581 MB RSS, never completed

### After: `chatcompletion` backend (lightweight)

```yaml
agents:
- id: agent_a
  backend:
    type: chatcompletion
    model: moonshotai/kimi-k2.5
    base_url: https://integrate.api.nvidia.com/v1
orchestrator:
  enable_multimodal_tools: false
  task_planning_filesystem_mode: false
  enable_memory_filesystem_mode: false
```

**Result**: 1 child process, 560 MB RSS, stable

## API Key Setup

NVIDIA API keys start with `nvapi-`. MassGen's `create_backend()` has a bug that passes `api_key` twice, so we work around it by using the standard `OPENAI_API_KEY` environment variable that `openai.AsyncOpenAI` auto-detects.

```bash
# .env file
OPENAI_API_KEY=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
```

## Code Changes

### 1. `plancraft/agents/massgen_config.yaml`

- Changed `type: copilot` → `type: chatcompletion`
- Set `model: moonshotai/kimi-k2.5`
- Added `base_url: https://integrate.api.nvidia.com/v1`
- Disabled multimodal/filesystem features (not needed for benchmark)

### 2. `plancraft/agents/massgen_orchestrated.py`

Added `os` import (already present) and kept the filesystem MCP disable logic:

```python
# Disable filesystem/MCP tools for lightweight benchmark agents
if not enable_filesystem:
    for agent_def in config_dict.get("agents", []):
        backend = agent_def.get("backend", {})
        backend["exclude_file_operation_mcps"] = True
```

### 3. `.env` file

Created with NVIDIA API key as `OPENAI_API_KEY`.

## API Testing

### NVIDIA Hosted API (works with nvapi- keys)

```python
resp = httpx.post(
    "https://integrate.api.nvidia.com/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={
        "model": "moonshotai/kimi-k2.5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 256,
    },
    timeout=120,
)
# Status: 200
# Content: ' Hello! How can I help you today?'
# Reasoning: ' The user wants me to say hello...'
```

### Moonshot Native API (rejects nvapi- keys)

```python
resp = httpx.post(
    "https://api.moonshot.cn/v1/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={"model": "kimi-k2.5", "messages": [{"role": "user", "content": "Say hello"}]},
    timeout=30,
)
# Status: 401
# {"error":{"message":"Invalid Authentication","type":"invalid_authentication_error"}}
```

**Conclusion**: Use NVIDIA's hosted endpoint with nvapi- keys.

## Memory Performance

| Metric | Before (copilot) | After (chatcompletion) |
|--------|------------------|------------------------|
| **Peak RSS** | 5,581 MB | 560 MB |
| **Child Processes** | 74 | 1 |
| **Memory Growth** | Exponential (never completed) | Flat (stable) |
| **Completion** | Failed (system hang) | Works (slow but completes) |

## Performance Characteristics

### Latency

- **Single API call**: 30-120s (reasoning model is slow)
- **MassGen orchestration**: 3 agents × 30-120s + coordination = 5-10+ minutes
- **Memory**: Stable at ~560 MB throughout

### Why It's Slow

1. **Reasoning model**: Kimi K2.5 generates internal reasoning before final answer
2. **Sequential agents**: MassGen runs agents one after another, not in parallel
3. **Coordination overhead**: Voting, answer selection, logging between rounds

## Root Cause Analysis (Original Issue)

The `copilot` backend spawned:
- 1 Node.js Copilot CLI server per agent (3 total)
- Multiple MCP tool servers per agent (filesystem, multimodal, custom tools)
- Each MCP server spawns its own child processes
- Result: 20-25 processes per agent × 3 agents = 60-74 total

All subprocesses were created via:
```python
# massgen/backend/copilot.py line 46
subprocess.Popen(args, ...)  # Node.js CLI
# massgen/backend/copilot.py line 777
create_session(mcp_servers=...)  # Spawns MCP servers
```

## Alternatives Considered

1. **Reduce to 1 agent** (still copilot): ~1.8 GB, but still heavy
2. **Keep copilot, disable MCP**: Less processes, but still Node.js overhead
3. **Switch to faster model**: e.g. `meta/llama-3.1-70b-instruct` on NVIDIA
4. **Get Moonshot sk- key**: Use `api.moonshot.cn` (may be faster)

## Files Modified

- `plancraft/agents/massgen_config.yaml` — Backend type and settings
- `plancraft/agents/massgen_orchestrated.py` — Added os import (already present)
- `.env` — NVIDIA API key as OPENAI_API_KEY
- `memory_issue_copilot.md` — Root cause analysis
- `test_api.py` — API connectivity test

## Verification

```bash
# Test API connectivity
uv run python test_api.py

# Run evaluation with memory monitoring
uv run python monitor_eval.py --split val.small --max-steps 1 --max-examples 1 \
  --config plancraft/agents/massgen_config.yaml --heartbeat-seconds 30 --no-report
```

Memory logs written to `output/memory_log_*.csv` for analysis.

## Status

✅ **Memory issue fixed** — 10x reduction from 5.5 GB to 560 MB  
✅ **API working** — NVIDIA endpoint returns 200 with content  
⏳ **Latency high** — Reasoning model + 3-agent orchestration takes 5-10+ minutes  
✅ **MassGen orchestration preserved** — All coordination logic intact

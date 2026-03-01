# Memory Issue Analysis: CopilotBackend Subprocess Explosion

## Summary

Running `eval_massgen.py` with the current `massgen_config.yaml` causes **5.5+ GB memory consumption** and **74 child processes** for a single step of a single example. The process never completed within 5+ minutes before being killed.

## Test Command

```bash
uv run eval_massgen.py --split val.small --max-steps 1 --max-examples 1 \
  --config plancraft/agents/massgen_config.yaml --heartbeat-seconds 30 --no-report
```

## Memory Timeline (from `output/memory_log_20260301_130652.csv`)

| Elapsed | Children | RSS (MB) | Event |
|---------|----------|----------|-------|
| 0s | 1 | 4 | Process starts |
| 48s | 1 | 552 | Config loading / agent creation phase |
| 51s | **9** | 1,001 | `chat_simple()` fires — copilot backends start spawning |
| 54s | **25** | 1,624 | More subprocess waves |
| 72s | **31** | 2,503 | Agent round 1 |
| 111s | **40** | 2,920 | Agent round 2 spawns more |
| 195s | **58** | 4,193 | Agent round 3 |
| 265s | **67** | 4,904 | Still growing |
| 322s | **74** | **5,581** | Still running, never completed — killed |

Key observations:
- Main Python process stays at ~4 MB — all memory is in **child processes**
- Children count grows in waves (1 → 9 → 25 → 31 → 42 → 56 → 67 → 74)
- RSS never drops — processes are not cleaned up during the step
- VMS peaked at **10.7 GB**

## Root Cause: Full Subprocess Chain

The `type: copilot` backend in `massgen_config.yaml` spawns a full GitHub Copilot CLI server (Node.js) per agent, plus MCP tool servers as child processes.

### Spawning Chain

```
massgen_config.yaml (3 agents, type: copilot)
  │
  ├─ Agent A: CopilotBackend()
  │   └─ CopilotClient()
  │       └─ _start_cli_server()
  │           └─ subprocess.Popen(["node", copilot_cli, "--headless", ...])
  │               └─ create_session(mcp_servers={...})
  │                   ├─ Filesystem MCP server (subprocess)
  │                   ├─ Custom tools MCP server (subprocess)
  │                   ├─ Multimodal tools MCP (subprocess)
  │                   └─ Each MCP server may spawn its own children
  │
  ├─ Agent B: CopilotBackend() — same tree
  │
  └─ Agent C: CopilotBackend() — same tree
```

### Per-agent cost: ~20-25 processes, ~1.8 GB RSS
### Total (3 agents): ~60-74 processes, ~5.5 GB RSS

### Source Code Locations

All within `massgen` package (`D:\plancraft\.venv\Lib\site-packages\massgen\`):

1. **`backend/copilot.py` line 46** — `subprocess.Popen(args, ...)` spawns Node.js Copilot CLI server
2. **`backend/copilot.py` lines 732-738** — `_build_mcp_servers_dict()` builds MCP server configs including filesystem tools
3. **`backend/copilot.py` line 777** — `create_session(session_config)` — the Copilot CLI spawns all MCP servers as child processes
4. **`copilot/client.py` `_start_cli_server()`** — the actual `subprocess.Popen` call for the Node.js CLI

### Config Settings That Amplify the Problem

From `plancraft/agents/massgen_config.yaml`:

```yaml
agents:
- id: agent_a
  backend:
    type: copilot              # ← Spawns full Node.js CLI + MCP servers
    model: gpt-5-mini
    cwd: workspace
    exclude_file_operation_mcps: false   # ← Enables heavy filesystem MCP servers
- id: agent_b                  # ← 2nd copy of everything
  backend:
    type: copilot
    ...
- id: agent_c                  # ← 3rd copy of everything
  backend:
    type: copilot
    ...
orchestrator:
  enable_multimodal_tools: true  # ← Adds image/video/audio MCP servers per agent
```

**None of these capabilities are needed** for this benchmark. The agents only return one-line text actions like:
- `move: from [I1] to [A1] with quantity 1`
- `search: oak_planks`
- `impossible: missing materials`

## Recommended Fixes

### Option 1: Switch to `type: openai` (Best Fix)

Replace `type: copilot` with `type: openai` in the config. This uses pure HTTP API calls to OpenAI — **zero subprocesses**, ~50 MB total instead of 5,500 MB.

```yaml
agents:
- id: agent_a
  backend:
    type: openai
    model: gpt-5-mini
- id: agent_b
  backend:
    type: openai
    model: gpt-5-mini
- id: agent_c
  backend:
    type: openai
    model: gpt-5-mini
```

Also set `enable_multimodal_tools: false` in the orchestrator section.

### Option 2: Reduce MCP Overhead (If Copilot Required)

If the copilot backend is required for some reason:

```yaml
agents:
- id: agent_a
  backend:
    type: copilot
    model: gpt-5-mini
    exclude_file_operation_mcps: true   # ← Disable filesystem MCP servers
orchestrator:
  enable_multimodal_tools: false        # ← Disable multimodal MCP servers
```

And reduce to 1 agent instead of 3 to cut memory by ~3x.

### Option 3: Band-Aid GC (Already Applied)

Already added to `eval_massgen.py`:
```python
del agent
gc.collect()
await asyncio.sleep(1)
```

This helps between examples but does NOT fix the per-step memory explosion within a single example.

## Additional Context: Persistent Orchestrator Rewrite

A separate change was made to `plancraft/agents/massgen_orchestrated.py` to use a native persistent `massgen.Orchestrator` session instead of the stateless `litellm.completion` layer. This eliminates repeated cold-booting of the orchestration infrastructure on every turn (previously 30 separate MassGen sessions per 30-step example), but does **not** address the copilot subprocess issue — that's inherent to the `copilot` backend type.

## Files Modified

- `plancraft/agents/massgen_orchestrated.py` — Persistent orchestrator session
- `eval_massgen.py` — GC cleanup between examples
- `monitor_eval.py` — Memory monitoring wrapper (new)
- `output/memory_log_20260301_130652.csv` — Raw memory data

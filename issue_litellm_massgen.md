# Issue: Memory Leaks and PC Hangs using LiteLLM with MassGen in eval_massgen.py

## Overview
When running the `eval_massgen.py` benchmark script, system memory consumption progressively increases until the PC hangs. This happens because of how the LiteLLM integration interfaces with the MassGen orchestrator.

## The Root Cause: Stateless API vs Stateful Orchestrator

The underlying issue is that the benchmarking script is using the stateless API of `litellm` to power a highly stateful and heavy architecture (`MassGen` Orchestrator). 

### 1. Re-initializing Orchestrator at Every Step
In `eval_massgen.py`, `agent.act()` is called for every valid action step within an example. 
Under the hood (`plancraft/agents/massgen_llm.py`), this triggers:
```python
litellm.completion(model="massgen/path:...", messages=...)
```
Because the `litellm` completion API is inherently stateless, the MassGen customized LiteLLM provider (`litellm_provider.py`) translates **every individual completion call** into a full initialization of `massgen.run()`. This spins up the *entire* multi-agent orchestration infrastructure, loads configurations, sets up logging, parses the conversation history from scratch, runs the coordination/voting process to get a single action, and then closes. 

For a single example that takes 30 steps, the script internally initializes and completely tears down **30 separate MassGen sessions**!

### 2. Exponentially Growing Conversation History
In `massgen_orchestrated.py`, the `conversation` list appends the new state Observation (`role="user"`) and the subsequent Agent Action (`role="model"`) at every turn. When step 15 occurs, the `litellm` provider feeds all 30 prior messages back into the brand new `massgen.run()` session so it can reconstruct the context. The data payload to parse per step grows significantly, incrementally consuming more memory and processing time.

### 3. Missing Garbage Collection & Hanging Threads
* **Threads**: `massgen_llm.py` triggers these heavy processes in isolated background threads via `asyncio.to_thread(_completion_call, ...)` while it shields them (`asyncio.shield`) to track heartbeat log timings. 
* **Logging loop**: The MassGen provider (`litellm_provider.py`) runs `reset_logging_session()` and `setup_logging()` for every step. Repeating this thousands of times throughout a benchmark run can leak unclosed file descriptors or leave logger references lingering in memory.
* Python's garbage collector sometimes defers cleanup of complex nested async/thread contexts and unclosed HTTP sessions, forcing the OS to bloatedly retain objects until the RAM is exhausted.

## Recommended Fixes / Mitigation Strategies

### 1. Native Persistent MassGen Session (Architectural Fix)
Instead of using the stateless `litellm.completion` layer (which forces a cold boot on every turn), modify `MassGenOrchestratedAgent` to initiate a single native `massgen.Orchestrator` session in its `reset()` method. Keep this session alive for the duration of the example, and stream/push new `observation_text`s into it on every `.act()` call natively rather than passing huge conversation histories back-and-forth under the guise of stateless generation.

### 2. Forced Garbage Collection (Band-Aid Fix)
As a quick mitigation to prevent immediate hangs, introduce an explicit garbage collection step and sleep interval between completed examples in `eval_massgen.py`:
```python
import gc
import asyncio

# Between examples or even between heavy steps:
gc.collect()
await asyncio.sleep(1) # Yield control back to OS to close sockets/clean up memory
```

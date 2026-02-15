# Plancraft Agents

This directory contains the implementation of agent architectures for the Plancraft environment, directly mapping to the taxonomy defined in the paper **"Towards a Science of Scaling Agent Systems"** (Kim et al., 2025).

These architectures allow us to study the trade-offs between **coordination overhead** and **task performance** (success rate).

## LLM Backends

The agents support two LLM backends:

| Backend | Base Class | LLM Wrapper | Eval Script | Default Model |
|---------|-----------|-------------|-------------|---------------|
| **Google Gemini** | `base.py` → `BaseAgent` | `llm.py` | `eval_gemini.py` | `gemma-3-27b-it` |
| **GitHub Copilot SDK** | `copilot_base.py` → `CopilotBaseAgent` | `copilot_llm.py` | `eval_copilot.py` | `gpt-5-mini` |

The Copilot SDK agents are async (using `asyncio`) and communicate with the Copilot CLI via JSON-RPC. The Copilot CLI must be installed and available in `$PATH`.

## Available Architectures

All five architectures are implemented for **both** backends. They share a common interface (`act(observation) → action`).

### 1. Single Agent (SAS)
- **Gemini:** `single.py` → `SingleAgent`
- **Copilot:** `copilot_single.py` → `CopilotSingleAgent`
- **Description:** The baseline system. A single LLM instance perceives the environment, reasons, and acts sequentially.
- **Paper scaling principle:** Serves as the reference point ($1.0\times$ cost, $1.0\times$ overhead).

### 2. Independent Multi-Agent System (Ensemble)
- **Gemini:** `multi.py` → `IndependentAgent`
- **Copilot:** `copilot_multi.py` → `CopilotIndependentAgent`
- **Description:** $N$ agents run in parallel on the same observation. No communication.
- **Aggregation:** Majority Voting (Consensus).
- **Key Characteristic:** High **Redundancy**. Prone to **Error Amplification** ($17.2\times$ in paper) if majority is wrong.

### 3. Centralized Multi-Agent System (Hierarchy)
- **Gemini:** `multi.py` → `CentralizedAgent`
- **Copilot:** `copilot_multi.py` → `CopilotCentralizedAgent`
- **Description:** Orchestrator analyzes state → generates plan → Worker executes action.
- **Key Characteristic:** **Error Containment** ($4.4\times$ in paper). Star topology.

### 4. Decentralized Multi-Agent System (Debate)
- **Gemini:** `multi.py` → `DecentralizedAgent`
- **Copilot:** `copilot_multi.py` → `CopilotDecentralizedAgent`
- **Description:** Agents propose → debate rounds → revise → final vote.
- **Key Characteristic:** High **Information Fusion**, high **Coordination Overhead**. All-to-All topology.

### 5. Hybrid Multi-Agent System
- **Gemini:** `multi.py` → `HybridAgent`
- **Copilot:** `copilot_multi.py` → `CopilotHybridAgent`
- **Description:** Orchestrator sets directive → Workers debate → Orchestrator aggregates.
- **Key Characteristic:** Balanced approach. Stabilizes decentralized variance with hierarchical control.

## Usage

### Gemini Backend

```bash
python eval_gemini.py --architecture single
python eval_gemini.py --architecture independent
python eval_gemini.py --architecture centralized
python eval_gemini.py --architecture decentralized
python eval_gemini.py --architecture hybrid
```

### Copilot SDK Backend

Requires `copilot` CLI in PATH and a Copilot subscription. Default model is `gpt-5-mini` (0 premium request cost).

```bash
python eval_copilot.py --architecture single
python eval_copilot.py --architecture independent
python eval_copilot.py --architecture centralized
python eval_copilot.py --architecture decentralized
python eval_copilot.py --architecture hybrid

# Use a different model
python eval_copilot.py --architecture single --model gpt-5
```

## Adding New Agents

To add a new agent architecture:
1. Create a class inheriting from `BaseAgent` (Gemini) or `CopilotBaseAgent` (Copilot).
2. Implement the `act(observation)` method (sync for Gemini, `async` for Copilot).
3. Register it in the `get_agent` factory in `eval_gemini.py` or `eval_copilot.py`.

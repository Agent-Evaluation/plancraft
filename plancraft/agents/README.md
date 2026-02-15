# Plancraft Agents

This directory contains the implementation of agent architectures for the Plancraft environment, directly mapping to the taxonomy defined in the paper **"Towards a Science of Scaling Agent Systems"** (Kim et al., 2025).

These architectures allow us to study the trade-offs between **coordination overhead** and **task performance** (success rate).

## Available Architectures

All agents share a common interface defined in `base.py`. They are instantiated via `eval_gemini.py` using the `--architecture` flag.

### 1. Single Agent (SAS)
- **File:** `single.py`
- **Class:** `SingleAgent`
- **Description:** The baseline system. A single LLM instance perceives the environment, reasons, and acts sequentially.
- **Paper scaling principle:** Serves as the reference point ($1.0\times$ cost, $1.0\times$ overhead).
- **Use Case:** Simple sequential tasks or when latency is critical.

### 2. Independent Multi-Agent System (Ensemble)
- **File:** `multi.py`
- **Class:** `IndependentAgent`
- **Description:** $N$ agents run in parallel (simulated sequentially here for API stability) on the same observation. They do **not** communicate.
- **Aggregation:** Majority Voting (Consensus). The most common action proposed by the $N$ agents is executed.
- **Key Characteristic:** High **Redundancy**. Good for reducing variance, but prone to **Error Amplification** ($17.2\times$ in paper) if the majority is wrong.
- **Topology:** Disconnected set.

### 3. Centralized Multi-Agent System (Hierarchy)
- **File:** `multi.py`
- **Class:** `CentralizedAgent`
- **Description:** A hierarchical structure with a distinct **Orchestrator** and **Worker(s)**.
- **Flow:**
  1. **Orchestrator:** Analyzes the state and generates a high-level Plan.
  2. **Worker:** Executes the specific action required by the Plan.
- **Key Characteristic:** **Error Containment**. The orchestrator acts as a bottleneck/validator, reducing error propagation ($4.4\times$ in paper).
- **Topology:** Star (Orchestrator connected to all Workers).

### 4. Decentralized Multi-Agent System (Debate)
- **File:** `multi.py`
- **Class:** `DecentralizedAgent`
- **Description:** Agents engage in peer-to-peer communication (Debate) before acting.
- **Flow:**
  1. Agents propose initial actions.
  2. Agents see each other's proposals (Debate Rounds).
  3. Agents revise their proposals.
  4. Final consensus via voting.
- **Key Characteristic:** High **Information Fusion**. Effective for tasks with high entropy (e.g., open-ended search), but high **Coordination Overhead**.
- **Topology:** Mess / All-to-All.

### 5. Hybrid Multi-Agent System
- **File:** `multi.py`
- **Class:** `HybridAgent`
- **Description:** Cmbines Centralized control with Decentralized execution.
- **Flow:**
  1. Orchestrator sets a Directive.
  2. Workers debate the best way to execute that directive.
  3. Orchestrator aggregates the final result.
- **Key Characteristic:** Balanced approach. Stabilizes the high variance of decentralized systems using hierarchical control.

## Usage

You can run evaluations for any architecture using the `eval_gemini.py` script at the project root.

```bash
# 1. Single Agent (Baseline)
python eval_gemini.py --architecture single

# 2. Independent (Ensemble)
python eval_gemini.py --architecture independent

# 3. Centralized (Manager-Worker)
python eval_gemini.py --architecture centralized

# 4. Decentralized (Debate)
python eval_gemini.py --architecture decentralized

# 5. Hybrid (Manager + Debate)
python eval_gemini.py --architecture hybrid
```

## Adding New Agents

To add a new agent architecture:
1. Create a class in `plancraft/agents/` inheriting from `BaseAgent`.
2. Implement the `act(observation)` method.
3. Register it in the `get_agent` factory function in `eval_gemini.py`.

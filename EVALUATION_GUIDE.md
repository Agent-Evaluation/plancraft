# Plancraft Evaluation with Gemma 3 27B — Full Guide

## Table of Contents

1. [What is Plancraft?](#1-what-is-plancraft)
2. [What Are We Evaluating?](#2-what-are-we-evaluating)
3. [How the Agent Was Built](#3-how-the-agent-was-built)
4. [The Evaluation Loop — Step by Step](#4-the-evaluation-loop--step-by-step)
5. [Architecture & Key Design Decisions](#5-architecture--key-design-decisions)
6. [Bug Fix: Windows Path Issue](#6-bug-fix-windows-path-issue)
7. [Model Selection](#7-model-selection)
8. [How to Run](#8-how-to-run)
9. [Understanding the Results](#9-understanding-the-results)
10. [Files Modified/Created](#10-files-modifiedcreated)

---

## 1. What is Plancraft?

Plancraft is a **Minecraft crafting benchmark** designed to evaluate **planning ability in LLM agents**. It was published at COLM 2025.

The idea is simple: given an inventory of Minecraft items and a target item to craft, can an AI agent figure out the correct sequence of actions (moving items into a crafting grid, smelting ores, multi-step crafting) to produce the target?

For example:
- **Easy**: You have `nether_quartz_ore` → smelt it → get `quartz`
- **Medium**: You have `oak_log` → craft into `oak_planks` → craft into `sticks`
- **Hard**: Multi-step chains requiring 5+ intermediate crafting steps
- **Impossible**: The inventory doesn't contain the right materials (agent must recognize this)

### The Environment

Plancraft simulates a Minecraft inventory with:
- A **3×3 crafting grid** (slots `[A1]` through `[C3]`)
- A **crafting output slot** (`[0]`) where results appear
- **36 inventory slots** (`[I1]` through `[I36]`) holding items
- A **smelting** mechanic (for ores → ingots, etc.)

The environment is **deterministic** — the same action always produces the same result.

### The Dataset

The dataset contains pre-generated crafting scenarios with different splits:

| Split             | Examples | Description                      |
|-------------------|----------|----------------------------------|
| `val.small`       | 110      | Small validation set (quick eval)|
| `val`             | ~600     | Full validation set              |
| `test.small`      | ~120     | Small test set                   |
| `test`            | ~600     | Full test set                    |
| `val.small.easy`  | ~90      | Easy-only validation             |
| `train`           | ~1200    | Training examples                |

Each example contains:
- `inventory`: items available to the agent
- `target`: the item to craft
- `impossible`: whether it's actually possible with the given inventory
- `complexity`: how many crafting steps are needed (1 = easy, 5+ = hard)
- `optimal_path`: the ideal crafting sequence

---

## 2. What Are We Evaluating?

We're evaluating whether a **conversational LLM** (Gemma 3 27B, accessed via the Google Gemini API) can act as a **planning agent** that:

1. **Understands** the current inventory from text descriptions
2. **Looks up** crafting recipes using a search tool
3. **Reasons** about which items to move where in the crafting grid
4. **Executes** a correct sequence of move/smelt actions
5. **Recognizes** when a task is impossible

### Success Criteria

- For **possible** tasks: The target item appears in the inventory (not in the crafting output slot)
- For **impossible** tasks: The agent correctly calls `impossible: <reason>`

### What This Tells Us

This benchmark measures:
- **Planning ability**: Can the LLM decompose a multi-step goal into actions?
- **Tool use**: Can it use the search tool to look up recipes it doesn't know?
- **Instruction following**: Can it output actions in the exact required format?
- **Spatial/symbolic reasoning**: Can it reason about slot positions in a grid?
- **Error recovery**: Can it adapt when an action doesn't produce the expected result?

---

## 3. How the Agent Was Built

### File: `eval_gemini.py`

The agent is a Python script that connects three components:

```
┌─────────────┐     text action      ┌──────────────────┐
│  Gemma 3    │ ──────────────────►  │  PlancraftGym    │
│  27B (LLM)  │                      │  Wrapper (env)   │
│             │ ◄──────────────────  │                  │
└──────┬──────┘     text observation  └──────────────────┘
       │
       │ search: <item>
       ▼
┌─────────────┐
│ Oracle RAG  │  (local recipe lookup, no API call)
│ gold_search │
└─────────────┘
```

### Component 1: The System Prompt

The system prompt (prepended to every conversation) teaches the LLM:
- The 4 available actions and their exact text formats
- How the slot naming system works (`[A1]`-`[C3]`, `[I1]`-`[I36]`, `[0]`)
- The rules of Minecraft crafting (place items → result in `[0]` → move to inventory)
- A strategy: search first, then place items, then collect the result

Since Gemma models don't support `system_instruction` in the API, the system prompt
is injected as the first `user` message, with a model acknowledgment reply, before the
actual conversation begins.

### Component 2: The Environment (`PlancraftGymWrapper`)

This is provided by the `plancraft` package. It follows the OpenAI Gym API:

```python
observation, reward, terminated, truncated, info = env.step(action_text)
```

- **Input**: a text string action (e.g., `"move: from [I1] to [A1] with quantity 1"`)
- **Output**: 
  - `observation["text"]`: updated inventory description
  - `reward`: 1.0 if target crafted, 0.0 otherwise
  - `terminated`: True if task completed or agent stopped
  - `truncated`: True if max steps reached

The wrapper internally:
1. Parses the action text using regex (via `MoveActionHandler`, `SmeltActionHandler`, etc.)
2. Validates slot ranges, quantities, and action format
3. Executes the action in the environment
4. Checks if the target item appeared in the inventory
5. Returns the updated state

### Component 3: Oracle Recipe Search

When the LLM outputs `search: <item_name>`, instead of sending this to the environment,
we intercept it and call `gold_search_recipe()` locally. This function looks up the
actual Minecraft recipe from Plancraft's built-in recipe database and returns it as text.

Example:
```
Input:  search: iron_ingot
Output: Recipes to craft iron_ingot:
        recipe 1: smelt {'iron_ore'}
        recipe 2: iron_block at [A2]
        recipe 3: iron_nugget at [A1], iron_nugget at [A2], ...
```

This gives the agent "oracle" knowledge — it always gets the correct recipe. This is
intentional in the Plancraft benchmark: the challenge is planning and execution, not
recipe memorization.

---

## 4. The Evaluation Loop — Step by Step

Here's exactly what happens for each example:

### Initialization
1. Load the example (target item, initial inventory, whether it's impossible)
2. Create a `PlancraftGymWrapper` with the example's inventory
3. Call `env.step("")` to get the initial observation (inventory state as text)
4. Start a conversation history with the observation as the first user message

### Agent Loop (repeats until done)

```
Step 1: Send conversation history to Gemma 3 27B
Step 2: Get the model's response (an action string)
Step 3: Add the response to conversation history
Step 4: Check if it's a search action
        → YES: Look up recipe locally, add result to conversation, go to Step 1
        → NO:  Continue to Step 5
Step 5: Send the action to the environment
Step 6: Get new observation (updated inventory)
Step 7: Check termination conditions:
        → Target crafted (success=True) → done
        → Agent said "impossible" → success=True only if actually impossible
        → Max steps (30) reached → truncated, success=False
        → Neither → add observation to conversation, go to Step 1
```

### Example Trace (VAL0491: craft quartz)

```
Observation: "Craft an item of type: quartz
              inventory:
               - magenta_carpet [I15] quantity 15
               - drowned_spawn_egg [I16] quantity 16
               - nether_quartz_ore [I19] quantity 1
               - brick_wall [I33] quantity 5
               - redstone_ore [I34] quantity 11"

Agent:       "search: quartz"                           ← looks up recipe
System:      "Recipes to craft quartz:
              recipe 1: smelt {'nether_quartz_ore'}"    ← oracle response

Agent:       "smelt: from [I19] to [I1] with quantity 1" ← smelts the ore
Environment: inventory now contains quartz at [I1]
             → reward=1.0, terminated=True
             ✅ SUCCESS
```

---

## 5. Architecture & Key Design Decisions

### Why `PlancraftGymWrapper` instead of `Evaluator`?

Plancraft provides two integration paths:
1. **`PlancraftGymWrapper`** — simple Gym API, you bring your own agent loop
2. **`Evaluator` + `PlancraftBaseModel`** — requires subclassing, manages history internally

We chose `PlancraftGymWrapper` because:
- Our agent is a **conversational chatbot** — it works with text in/text out
- We need full control over how conversation history is built
- No need to conform to the `PlancraftBaseModel.step()` interface
- Simpler code, easier to understand and modify

### Why Prepend System Prompt as User Message?

Gemma models running on the Gemini API don't support the `system_instruction` parameter
(you get a `400 INVALID_ARGUMENT` error). The workaround is to inject the system prompt
as the first `user` message followed by a model acknowledgment:

```
[user]  : <system prompt with all instructions>
[model] : "Understood. I will respond with exactly one action per turn."
[user]  : <actual first observation>
[model] : <agent's first action>
...
```

### Why Intercept Search Actions?

The `search` action is handled outside the environment for two reasons:
1. It doesn't change the game state — it's purely informational
2. The response (recipe text) comes from the local recipe database, not the LLM
3. Search steps don't count against the environment's step limit

### Conversation History

The full conversation is maintained and sent with every API call. This allows the model
to remember:
- What recipes it looked up
- What actions it already took
- What the inventory looked like at each step
- Error messages from invalid actions

### Rate Limiting & Retry Logic

The script includes:
- **2.5 second delay** between API calls (to stay under 30 RPM)
- **Exponential backoff** on 429 errors (5s → 10s → 20s → 40s → 80s)
- **5 retries** before giving up on a request

---

## 6. Bug Fix: Windows Path Issue

The Plancraft repository had a bug in `plancraft/environment/recipes.py` (line 25):

```python
# BEFORE (broken on Windows):
tag_name = tag_file.split("/")[-1].split(".")[0]

# AFTER (cross-platform):
tag_name = os.path.basename(tag_file).split(".")[0]
```

On Windows, file paths use backslashes (`\`), so `.split("/")` didn't split anything,
resulting in the full path being used as a tag name. This caused a `KeyError: 'acacia_logs'`
when loading recipes because the tag names didn't match. Using `os.path.basename()` works
correctly on both Windows and Unix.

---

## 7. Model Selection

### Why Gemma 3 27B?

We evaluated the available models on the free Gemini API tier:

| Model              | RPM | RPD    | Suitable? |
|--------------------|-----|--------|-----------|
| Gemini 2.5 Flash   | 5   | 20     | ❌ 20 RPD is ~2 examples max |
| Gemini 2.5 Flash Lite | 10 | 20   | ❌ Same RPD limit |
| Gemini 3 Flash     | 5   | 20     | ❌ Same RPD limit |
| Gemma 3 1B         | 30  | 14,400 | ⚠️ Too small for reasoning |
| Gemma 3 4B         | 30  | 14,400 | ⚠️ Might struggle with complex tasks |
| Gemma 3 12B        | 30  | 14,400 | ✅ Good balance |
| **Gemma 3 27B**    | **30** | **14,400** | ✅ **Best reasoning, same limits** |

**Key insight**: The Gemini models have only **20 requests per day** on the free tier,
making them useless for any meaningful evaluation. The Gemma models have **14,400 RPD**
(720× more) with 30 RPM — plenty for evaluating 110+ examples.

Among the Gemma models, 27B is the largest and most capable for the multi-step reasoning
required by Plancraft, while having identical rate limits and cost (free) as the smaller
variants.

---

## 8. How to Run

### Prerequisites

```bash
pip install plancraft google-genai
```

### Set API Key (optional — hardcoded fallback exists)

```powershell
$env:GOOGLE_API_KEY = "your-key-here"
```

### Run Evaluation

```powershell
# Quick test (3 examples)
python eval_gemini.py --max-examples 3 --split val.small

# Full val.small evaluation (110 examples, ~30-40 min)
python eval_gemini.py --split val.small

# Full validation set (~600 examples)
python eval_gemini.py --split val

# Use a different model
python eval_gemini.py --model gemma-3-12b-it --split val.small
```

### CLI Arguments

| Argument         | Default        | Description                         |
|------------------|----------------|-------------------------------------|
| `--split`        | `val.small`    | Dataset split to evaluate           |
| `--max-steps`    | `30`           | Max actions per example             |
| `--max-examples` | `0` (= all)    | Limit number of examples            |
| `--model`        | `gemma-3-27b-it` | Model ID for the Gemini API       |

---

## 9. Understanding the Results

### Output File

Results are saved to `output/gemini_<split>_<timestamp>.json`:

```json
{
  "model": "gemma-3-27b-it",
  "split": "val.small",
  "max_steps": 30,
  "total": 110,
  "successes": 72,
  "success_rate": 0.654,
  "elapsed_seconds": 2100.5,
  "results": [
    {
      "example_id": "VAL0491",
      "target": "quartz",
      "impossible": false,
      "complexity": 1.0,
      "success": true,
      "steps": 3,
      "reason": "success"
    },
    ...
  ]
}
```

### Console Output

```
[1/110] Example VAL0491 | Target: quartz | Impossible: False | Complexity: 1
  Step 1: 🔍 search: quartz              ← recipe lookup
  Step 2: smelt: from [I19] to [I1] ...  ← smelting action
  ✅ Result: success=True | reason=success | steps=2
```

### Metrics

- **Overall success rate**: % of all examples solved correctly
- **Possible tasks success**: % of craftable tasks actually crafted
- **Impossible tasks success**: % of impossible tasks correctly identified
- **Average steps**: how many actions the agent takes per example

### Reason Codes

| Reason | Meaning |
|--------|---------|
| `success` | Target item was crafted and collected |
| `correctly_identified_impossible` | Agent said "impossible" and the task was actually impossible |
| `incorrect_stop` | Agent said "impossible" but the task was actually possible |
| `max_steps_reached` | Agent used all 30 steps without completing the task |

### Quick Test Results

From our 3-example test run:

```
📊 EVALUATION RESULTS
  Model:          gemma-3-27b-it
  Total examples: 3
  Overall:        2/3 = 66.7%
  Possible tasks: 2/3 = 66.7%
  Time elapsed:   58.0s (19.3s per example)
```

---

## 10. Files Modified/Created

| File | Action | Description |
|------|--------|-------------|
| `eval_gemini.py` | **Created** | Main evaluation script with Gemma agent |
| `plancraft/environment/recipes.py` | **Modified** (line 25) | Fixed Windows path bug in tag name extraction |
| `output/*.json` | **Created** | Evaluation results (auto-generated per run) |
| `EVALUATION_GUIDE.md` | **Created** | This document |

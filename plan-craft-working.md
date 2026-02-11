# Plancraft — How It Works

A comprehensive guide to understanding the Plancraft project, its data, evaluation pipeline, and Minecraft crafting mechanics.

---

## Table of Contents

1. [Validation Dataset](#validation-dataset)
2. [Minecraft Crafting Grid](#minecraft-crafting-grid)
3. [How Crafting Works in Minecraft](#how-crafting-works-in-minecraft)
4. [Oracle Trajectories (train / val)](#oracle-trajectories-train--val)
5. [Evaluation Pipeline — How Gemini Is Judged](#evaluation-pipeline--how-gemini-is-judged)
6. [Metrics — How Planning Quality Is Measured](#metrics--how-planning-quality-is-measured)

---

## Validation Dataset

The validation dataset lives in two locations:

### 1. Evaluation Examples — `plancraft/data/`

| File | Description |
|------|-------------|
| `val.json` | Full validation set (~600 examples) |
| `val.small.json` | Small validation set (110 examples, for quick eval) |
| `val.small.easy.json` | Easy-only validation (~90 examples) |
| `val.repeated.json` | Repeated validation set |

Each JSON file contains examples with these fields:

- `inventory` — items available to the agent
- `target` — item to craft
- `impossible` — whether it's possible with the given inventory
- `complexity` — number of crafting steps needed
- `slotted_inventory` — inventory with slot positions
- `optimal_path` — the ideal crafting sequence (for analysis, not used during eval)
- `optimal_path_length` — how many steps the ideal solution takes

### 2. Oracle Trajectories — `oracle_trajectories/val/`

Used for training dialogue models (not used by the Gemini agent):

- `oracle_trajectories/val/oa/` — Trajectory JSON files (VAL0000.json, VAL0001.json, etc.)
- `oracle_trajectories/val/images/` — GIF visualizations of each trajectory

### How They're Loaded

- **For evaluation**: `get_plancraft_examples(split="val.small")` loads from `plancraft/data/{split}.json`
- **For training**: `PlancraftDialogueDataset` loads from `oracle_trajectories/{split}/oa/*.json`
- The default split for quick evaluation is `val.small` (110 examples)

---

## Minecraft Crafting Grid

The crafting table is a 3×3 grid. Each cell has a position identifier:

```
+------+------+------+
|      |      |      |
|  A1  |  A2  |  A3  |
|      |      |      |
+------+------+------+
|      |      |      |
|  B1  |  B2  |  B3  |
|      |      |      |
+------+------+------+
|      |      |      |
|  C1  |  C2  |  C3  |
|      |      |      |
+------+------+------+
```

- **Row A** = Top row (A1: top-left, A2: top-center, A3: top-right)
- **Row B** = Middle row (B1: mid-left, B2: center, B3: mid-right)
- **Row C** = Bottom row (C1: bottom-left, C2: bottom-center, C3: bottom-right)

Additional slots:

- **[0]** — Crafting output slot (crafted items appear here)
- **[I1] through [I36]** — Inventory slots

---

## How Crafting Works in Minecraft

You do **not** need to fill the entire 3×3 grid. You place items from your inventory into the crafting grid in a specific **pattern**. The pattern determines what you craft. Empty slots are fine — most recipes only use a few cells.

### Example: Sticks (2 cells)

```
+--------+--------+--------+
|        |        |        |
|        | plank  |        |
+--------+--------+--------+
|        |        |        |
|        | plank  |        |
+--------+--------+--------+
|        |        |        |
|        |        |        |
+--------+--------+--------+
```

### Example: Wooden Sword (3 cells, vertical)

```
+--------+--------+--------+
|        |        |        |
|        | plank  |        |
+--------+--------+--------+
|        |        |        |
|        | plank  |        |
+--------+--------+--------+
|        |        |        |
|        | stick  |        |
+--------+--------+--------+
```

### Example: Chest (8 cells, center empty)

```
+--------+--------+--------+
|        |        |        |
| plank  | plank  | plank  |
+--------+--------+--------+
|        |        |        |
| plank  |        | plank  |
+--------+--------+--------+
|        |        |        |
| plank  | plank  | plank  |
+--------+--------+--------+
```

### Key Rules

1. **Pattern matters** — Items must be arranged in the correct shape. A pickaxe is 3 across the top + 2 down the middle; rearranging them won't work.
2. **Some recipes are shapeless** — A few recipes (like dyes + wool) don't care about position. You just need the right items anywhere on the grid.
3. **Some patterns are shiftable** — Many shaped recipes can be placed anywhere on the grid as long as the relative pattern is preserved.
4. **Multi-step crafting** — Complex items require intermediate crafts. For example, a wooden pickaxe requires:
   - Step 1: Craft **planks** from logs
   - Step 2: Craft **sticks** from planks
   - Step 3: Craft the **pickaxe** from planks + sticks

This is exactly what the `complexity` field in the Plancraft dataset measures — the number of intermediate crafting steps needed to reach the target item from the given inventory.

---

## Oracle Trajectories (train / val)

### What Are They?

Pre-recorded **perfect (oracle) solutions** — step-by-step demonstrations of how to correctly craft each target item. Each JSON file contains a complete conversation (system prompt + user/assistant turns) showing the ideal sequence of moves.

For example, `TRAIN0000.json` shows the oracle crafting **cyan_stained_glass_pane**:

1. `move: from [I36] to [A1] with quantity 1` — place glass
2. `move: from [I36] to [A2] with quantity 1`
3. `move: from [I36] to [A3] with quantity 1`
4. `move: from [I36] to [B1] with quantity 1`
5. `move: from [I36] to [B2] with quantity 1`
6. `move: from [I36] to [B3] with quantity 1`
7. `move: from [0] to [I2] with quantity 16` — collect result

### train/ vs val/

| Folder | Purpose |
|--------|---------|
| `oracle_trajectories/train/oa/` | Training trajectories — used to **fine-tune** LLMs via supervised learning |
| `oracle_trajectories/val/oa/` | Validation trajectories — used to compute **validation loss** during training |
| `*/images/` | GIF visualizations of each trajectory |

These are used by `PlancraftDialogueDataset` in `plancraft/train/dataset.py` for fine-tuning smaller models (like Llama) via imitation learning.

### Does the Gemini Agent Use Them?

**No.** The `eval_gemini.py` agent does **not** use oracle trajectories. It:

1. Loads evaluation examples from `plancraft/data/val.small.json` (inventory + target)
2. Runs Gemini **from scratch** against the live environment — no oracle demonstrations
3. Gemini figures out the actions on its own using only the system prompt + environment feedback

In short:
- **`oracle_trajectories`** = training data for **fine-tuning** smaller models via imitation learning
- **`eval_gemini.py`** = **zero-shot evaluation** where Gemini reasons on its own

---

## Evaluation Pipeline — How Gemini Is Judged

There is **no pre-recorded "correct answer"** being compared against. The system uses a **live crafting simulation** — like a real Minecraft crafting table. The environment itself is the judge.

### The Loop

```
┌──────────────────────────────────────────────────────────┐
│  val.small.json example:                                 │
│    inventory: { nether_quartz_ore: 1, brick_wall: 5 }    │
│    target: "quartz"                                      │
│    impossible: false                                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  PlancraftEnvironment   │  (actual crafting simulation)
         │  loads the inventory    │
         └────────────┬────────────┘
                      │
          ┌───────────▼───────────┐
          │  Gemini sees:         │
          │  "Craft: quartz       │
          │   inventory:          │
          │   - nether_quartz_ore │
          │     [I28] qty 1"      │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Gemini responds:     │◄──────┐
          │  "smelt: from [I28]   │       │
          │   to [I1] qty 1"     │       │
          └───────────┬───────────┘       │
                      │                   │
          ┌───────────▼───────────┐       │
          │  Environment executes │       │  Loop until
          │  the action on the    │       │  done/failed
          │  simulated crafting   │       │
          │  table                │       │
          └───────────┬───────────┘       │
                      │                   │
          ┌───────────▼───────────┐       │
          │  Returns new state    │───────┘
          │  of inventory         │
          └───────────┬───────────┘
                      │
                      ▼
              Did it work?
```

### How Success/Failure Is Determined

The environment checks these conditions (in `plancraft/simple.py`):

| Outcome | How It's Detected | Success? |
|---------|-------------------|----------|
| **Crafted the target** | `check_done()` scans inventory — if target item appears in any slot [I1]-[I36] | Yes |
| **Correctly said "impossible"** | Agent says `impossible: ...` AND `example.impossible == True` | Yes |
| **Incorrectly said "impossible"** | Agent says `impossible: ...` BUT the task was actually possible | No |
| **Ran out of steps** | Exceeded `max_steps` (default 30) without crafting the target | No |
| **Bad/invalid actions** | Action doesn't parse or doesn't change state — wastes a step | Continues |

Key insight: **the environment itself is the judge**. If you put the right items in the right grid slots, the output item actually appears in slot [0]. Then the agent must move it to inventory. The environment checks if the target item ended up in the inventory — that's the ground truth.

---

## Metrics — How Planning Quality Is Measured

The final metrics reported by `eval_gemini.py`:

1. **Overall success rate** — `successes / total` across all examples
2. **Possible task success** — Success rate on tasks that *can* be crafted (tests crafting ability)
3. **Impossible task success** — Success rate on tasks that *can't* be crafted (tests the agent's ability to recognize insufficient materials)
4. **Steps per example** — Fewer steps = better planning
5. **Breakdown by complexity** — The dataset includes a `complexity` field (number of intermediate crafts), so you can measure if the agent handles simple 1-step recipes but fails on complex multi-step ones

Each example in `val.small.json` also includes `optimal_path` and `optimal_path_length` metadata. These are **not used during evaluation** but are available for post-hoc analysis — e.g., comparing how many steps Gemini took versus the optimal path length to quantify planning efficiency.

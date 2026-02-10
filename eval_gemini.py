"""
Plancraft Evaluation Agent powered by Google Gemini.

Uses the PlancraftGymWrapper (Gym-style API) with text observations.
The agent maintains a conversation history and uses Gemini to decide actions.
It also has access to the oracle recipe search tool built into Plancraft.

Usage:
    set GOOGLE_API_KEY=your-api-key
    python eval_gemini.py [--split val.small] [--max-steps 30] [--max-examples 0] [--model gemini-2.0-flash]
"""

import argparse
import json
import os
import time
from datetime import datetime

from google import genai

from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples
from plancraft.environment.actions import (
    MoveActionHandler,
    SmeltActionHandler,
    ImpossibleActionHandler,
)
from plancraft.environment.search import gold_search_recipe

# ---------------------------------------------------------------------------
# System prompt that teaches Gemini the Plancraft action format
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a Minecraft crafting agent. Your goal is to craft a target item by \
manipulating items in your inventory using a 3×3 crafting grid.

## Actions
Respond with EXACTLY ONE action per turn (no extra text):

1. **move** – move items between slots
   `move: from [Source] to [Target] with quantity N`

2. **smelt** – smelt an item (e.g. ores → ingots)
   `smelt: from [Source] to [Target] with quantity N`

3. **impossible** – declare the task impossible with current inventory
   `impossible: <reason>`

4. **search** – look up how to craft an item  
   `search: <item_name>`

## Slot names
- Crafting grid (3×3):
    [A1] [A2] [A3]
    [B1] [B2] [B3]
    [C1] [C2] [C3]
- Crafting output: [0]  (crafted items appear here)
- Inventory: [I1] through [I36]

## Crafting rules
1. Place raw materials from your inventory INTO the crafting grid slots \
([A1]–[C3]) in the correct pattern.
2. When the pattern is correct the result appears in slot [0].
3. Move the result from [0] to any inventory slot [I1]–[I36] to collect it.
4. Shaped recipes require items in specific grid positions.
5. Shapeless recipes can be placed in any grid slots.
6. Smelting uses the `smelt` action directly — no grid needed.

## Strategy
- First, if you don't know the recipe, use `search: <target_item>` to look it up.
- Then place the required items in the crafting grid.
- Finally, move the crafted item from [0] to an inventory slot.
- If the inventory lacks the required materials, declare `impossible: <reason>`.
- You may need multi-step crafting (e.g. logs → planks → sticks).

## Important
- Respond with ONLY the action, nothing else.
- Use exact slot names like [I1], [A1], [0], etc.
- Quantities must be between 1 and 64.
"""


def build_search_response(action_text: str) -> str | None:
    """
    If the action is a search action, perform the oracle recipe lookup
    and return the result. Otherwise return None.
    """
    import re

    match = re.search(r"search:\s*(\S+)", action_text)
    if match:
        target = match.group(1).strip().lower()
        return gold_search_recipe(target)
    return None


MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 5  # seconds
INTER_REQUEST_DELAY = 2.5  # seconds between API calls to stay under 30 RPM

def call_gemini_with_retry(client, model_name, messages, system_prompt):
    # Small delay between requests to stay comfortably under RPM limit
    time.sleep(INTER_REQUEST_DELAY)

    # Prepend system prompt as first user message (Gemma models don't support system_instruction)
    system_msg = genai.types.Content(
        role="user",
        parts=[genai.types.Part(text=system_prompt)],
    )
    ack_msg = genai.types.Content(
        role="model",
        parts=[genai.types.Part(text="Understood. I will respond with exactly one action per turn.")],
    )
    full_messages = [system_msg, ack_msg] + messages

    retries = 0
    delay = INITIAL_RETRY_DELAY
    while retries < MAX_RETRIES:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=full_messages,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=256,
                ),
            )
            return response.text.strip()
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"  ⏳ Rate limited. Retrying in {delay}s... (attempt {retries+1}/{MAX_RETRIES})")
            else:
                print(f"  ⚠ API error (attempt {retries+1}/{MAX_RETRIES}): {e}")
            retries += 1
            if retries < MAX_RETRIES:
                time.sleep(delay)
                delay *= 2
    raise Exception(f"Failed after {MAX_RETRIES} retries.")


def run_evaluation(
    split: str = "val.small",
    max_steps: int = 30,
    max_examples: int = 0,
    model_name: str = "gemma-3-27b-it",
):
    # ---- Setup Gemini client ----
    api_key = (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or DEFAULT_API_KEY
    )

    client = genai.Client(api_key=api_key)

    # ---- Load dataset ----
    examples = get_plancraft_examples(split=split)
    if max_examples > 0:
        examples = examples[:max_examples]

    print(f"📦 Loaded {len(examples)} examples from split '{split}'")
    print(f"🤖 Model: {model_name}")
    print(f"🔄 Max steps per example: {max_steps}")
    print("-" * 60)

    results = []
    total_start = time.time()

    for idx, example in enumerate(examples):
        print(
            f"\n[{idx+1}/{len(examples)}] Example {example.id} | "
            f"Target: {example.target} | "
            f"Impossible: {example.impossible} | "
            f"Complexity: {example.complexity}"
        )

        # Create environment
        env = PlancraftGymWrapper(
            example=example,
            actions=[
                MoveActionHandler(),
                SmeltActionHandler(),
                ImpossibleActionHandler(),
            ],
            max_steps=max_steps,
            use_text_inventory=True,
        )

        # Get initial observation
        observation, reward, terminated, truncated, info = env.step("")

        # Build conversation history for Gemini
        conversation = [
            {"role": "user", "content": observation["text"]},
        ]

        step_count = 0
        while not (terminated or truncated):
            step_count += 1

            # --- Call Gemini ---
            try:
                messages = [
                    genai.types.Content(
                        role=msg["role"] if msg["role"] != "assistant" else "model",
                        parts=[genai.types.Part(text=msg["content"])],
                    )
                    for msg in conversation
                ]

                action_text = call_gemini_with_retry(
                    client, model_name, messages, SYSTEM_PROMPT
                )

            except Exception as e:
                print(f"  ⚠ Gemini API error: {e}")
                action_text = "impossible: API error"

            # Add assistant response to conversation
            conversation.append({"role": "model", "content": action_text})

            # --- Handle search action locally (oracle RAG) ---
            search_result = build_search_response(action_text)
            if search_result is not None:
                print(f"  Step {step_count}: 🔍 {action_text}")
                # Feed recipe info back as a user message
                conversation.append({"role": "user", "content": search_result})
                # Search doesn't count as an env step, continue to next model call
                continue

            print(f"  Step {step_count}: {action_text}")

            # --- Execute action in environment ---
            observation, reward, terminated, truncated, info = env.step(action_text)

            # Add environment response to conversation
            if not (terminated or truncated):
                conversation.append({"role": "user", "content": observation["text"]})

        # Record result
        success = env.success
        status = "✅" if success else "❌"
        reason = info.get("reason", "unknown")
        print(f"  {status} Result: success={success} | reason={reason} | steps={step_count}")

        results.append(
            {
                "example_id": example.id,
                "target": example.target,
                "impossible": example.impossible,
                "complexity": example.complexity,
                "success": success,
                "steps": step_count,
                "reason": reason,
            }
        )

    # ---- Summary ----
    elapsed = time.time() - total_start
    total = len(results)
    successes = sum(r["success"] for r in results)
    success_rate = successes / total if total > 0 else 0

    # Breakdown by impossible vs possible
    possible = [r for r in results if not r["impossible"]]
    impossible = [r for r in results if r["impossible"]]
    possible_success = sum(r["success"] for r in possible) if possible else 0
    impossible_success = sum(r["success"] for r in impossible) if impossible else 0

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Split:          {split}")
    print(f"  Total examples: {total}")
    print(f"  Overall:        {successes}/{total} = {success_rate:.1%}")
    if possible:
        print(
            f"  Possible tasks: {possible_success}/{len(possible)} = "
            f"{possible_success/len(possible):.1%}"
        )
    if impossible:
        print(
            f"  Impossible:     {impossible_success}/{len(impossible)} = "
            f"{impossible_success/len(impossible):.1%}"
        )
    print(f"  Time elapsed:   {elapsed:.1f}s ({elapsed/total:.1f}s per example)")
    print("=" * 60)

    # ---- Save results ----
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"gemini_{split}_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "split": split,
                "max_steps": max_steps,
                "total": total,
                "successes": successes,
                "success_rate": success_rate,
                "elapsed_seconds": elapsed,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plancraft Gemini Evaluation")
    parser.add_argument("--split", default="val.small", help="Dataset split to evaluate on")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    parser.add_argument("--model", default="gemma-3-27b-it", help="Model name (default: gemma-3-27b-it, 30 RPM / 14.4K RPD)")
    args = parser.parse_args()

    run_evaluation(
        split=args.split,
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        model_name=args.model,
    )

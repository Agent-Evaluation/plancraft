"""
Plancraft Evaluation Agent powered by NVIDIA NIM API (Kimi K2.5).

Uses the NVIDIA Integrate API to call models like Kimi K2.5.

Usage:
    python eval_nvidia.py [--split val.small] [--max-steps 30] [--max-examples 0] [--model moonshotai/kimi-k2.5]
"""

import argparse
import json
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# Load .env file for NVIDIA_API_KEY
load_dotenv()

from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples
from plancraft.environment.actions import (
    MoveActionHandler,
    SmeltActionHandler,
    ImpossibleActionHandler,
)
from plancraft.agents.nvidia_llm import DEFAULT_MODEL, get_nvidia_api_key
from plancraft.agents.nvidia_single import NvidiaSingleAgent


def run_evaluation(
    split: str = "val.small",
    max_steps: int = 30,
    max_examples: int = 0,
    model_name: str = DEFAULT_MODEL,
):
    # ---- Verify API key ----
    api_key = get_nvidia_api_key()
    print(f"🔑 NVIDIA API key loaded (ends with ...{api_key[-4:]})")

    # ---- Load dataset ----
    examples = get_plancraft_examples(split=split)
    if max_examples > 0:
        examples = examples[:max_examples]

    print(f"📦 Loaded {len(examples)} examples from split '{split}'")
    print(f"🤖 Model: {model_name} (via NVIDIA NIM)")
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

        # Initialize Agent
        agent = NvidiaSingleAgent(model_name)
        agent.reset(example.id, example.target)

        # Get initial observation
        observation, reward, terminated, truncated, info = env.step("")

        step_count = 0
        while not (terminated or truncated):
            step_count += 1

            # Agent decides action
            try:
                action_text = agent.act(observation["text"])
            except Exception as e:
                print(f"  ⚠ Agent error: {e}")
                action_text = "impossible: Agent error"

            print(f"  Step {step_count}: {action_text}")

            # Execute action
            observation, reward, terminated, truncated, info = env.step(action_text)

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

    possible = [r for r in results if not r["impossible"]]
    impossible = [r for r in results if r["impossible"]]
    possible_success = sum(r["success"] for r in possible) if possible else 0
    impossible_success = sum(r["success"] for r in impossible) if impossible else 0

    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS (NVIDIA NIM / Kimi K2.5)")
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
    model_short = model_name.split("/")[-1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"nvidia_{model_short}_{split}_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "backend": "nvidia-nim",
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
    parser = argparse.ArgumentParser(description="Plancraft NVIDIA NIM Evaluation (Kimi K2.5)")
    parser.add_argument("--split", default="val.small", help="Dataset split to evaluate on")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    run_evaluation(
        split=args.split,
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        model_name=args.model,
    )

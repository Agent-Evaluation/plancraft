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
from plancraft.agents.llm import get_gemini_client, DEFAULT_API_KEY
from plancraft.agents.single import SingleAgent
from plancraft.agents.multi import (
    IndependentAgent, 
    CentralizedAgent, 
    DecentralizedAgent,
    HybridAgent
)

def get_agent(architecture: str, model_name: str, client: genai.Client):
    if architecture == "single":
        return SingleAgent(model_name, client)
    elif architecture == "independent":
        return IndependentAgent(model_name, client)
    elif architecture == "centralized":
        return CentralizedAgent(model_name, client)
    elif architecture == "decentralized":
        return DecentralizedAgent(model_name, client)
    elif architecture == "hybrid":
        return HybridAgent(model_name, client)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def run_evaluation(
    split: str = "val.small",
    max_steps: int = 30,
    max_examples: int = 0,
    model_name: str = "gemma-3-27b-it",
    architecture: str = "single",
):
    # ---- Setup Gemini client ----
    client = get_gemini_client()

    # ---- Load dataset ----
    examples = get_plancraft_examples(split=split)
    if max_examples > 0:
        examples = examples[:max_examples]

    print(f"📦 Loaded {len(examples)} examples from split '{split}'")
    print(f"🤖 Model: {model_name}")
    print(f"🏗️  Architecture: {architecture}")
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
        agent = get_agent(architecture, model_name, client)
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
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Model:          {model_name}")
    print(f"  Architecture:   {architecture}")
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
        output_dir, f"gemini_{architecture}_{split}_{timestamp}.json"
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "model": model_name,
                "architecture": architecture,
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
    parser.add_argument("--model", default="gemma-3-27b-it", help="Model name")
    parser.add_argument(
        "--architecture", 
        default="single", 
        choices=["single", "independent", "centralized", "decentralized", "hybrid"],
        help="Agent architecture to use"
    )
    args = parser.parse_args()

    run_evaluation(
        split=args.split,
        max_steps=args.max_steps,
        max_examples=args.max_examples,
        model_name=args.model,
        architecture=args.architecture,
    )


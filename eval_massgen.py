"""
Plancraft benchmark evaluator powered by a persistent MassGen orchestrator.

Usage:
    python eval_massgen.py --split val.small --max-steps 30 --config plancraft/agents/massgen_config.yaml
"""

import argparse
import asyncio
import gc
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from plancraft.agents.massgen_orchestrated import MassGenOrchestratedAgent
from plancraft.environment.actions import (
    ImpossibleActionHandler,
    MoveActionHandler,
    SmeltActionHandler,
)
from plancraft.simple import PlancraftGymWrapper, get_plancraft_examples

# Load API keys from .env if present.
load_dotenv()


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    idx = int((len(sorted_values) - 1) * q)
    return sorted_values[idx]


async def run_evaluation(
    split: str = "val.small",
    max_steps: int = 30,
    max_examples: int = 0,
    config_path: str = "plancraft/agents/massgen_config.yaml",
    heartbeat_seconds: int = 30,
    auto_generate_report: bool = True,
    report_path: str = "BENCHMARK_REPORT.md",
) -> dict[str, Any]:
    run_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    result_json_path = output_dir / f"massgen_orchestrator_{split}_{timestamp}.json"
    step_jsonl_path = output_dir / f"massgen_steps_{split}_{timestamp}.jsonl"

    with open(step_jsonl_path, "w", encoding="utf-8") as step_log_fp:

        def event_logger(event: dict[str, Any]) -> None:
            payload = {
                "run_timestamp": timestamp,
                "split": split,
                **event,
            }
            line = json.dumps(payload, ensure_ascii=True, default=str)
            step_log_fp.write(line + "\n")
            step_log_fp.flush()
            print(f"[MassGenLog] {line}")

        event_logger(
            {
                "event": "run_start",
                "config_path": config_path,
                "heartbeat_seconds": heartbeat_seconds,
                "max_steps": max_steps,
                "max_examples": max_examples,
                "timestamp": _now_iso(),
            }
        )

        examples = get_plancraft_examples(split=split)
        if max_examples > 0:
            examples = examples[:max_examples]

        print(f"📦 Loaded {len(examples)} examples from split '{split}'")
        print("🤖 Backend: MassGen orchestrator via LiteLLM")
        print(f"🧠 Config: {config_path}")
        print(f"🔄 Max steps per example: {max_steps}")
        print(f"🫀 Heartbeat: every {heartbeat_seconds}s")
        print("-" * 80)

        results: list[dict[str, Any]] = []
        all_step_durations: list[float] = []
        selected_agents_counter: Counter[str] = Counter()

        for idx, example in enumerate(examples):
            example_start = time.time()
            print(
                f"\n[{idx + 1}/{len(examples)}] Example {example.id} | "
                f"Target: {example.target} | "
                f"Impossible: {example.impossible} | "
                f"Complexity: {example.complexity}"
            )

            event_logger(
                {
                    "event": "example_start",
                    "example_id": example.id,
                    "target": example.target,
                    "impossible": example.impossible,
                    "complexity": example.complexity,
                    "timestamp": _now_iso(),
                }
            )

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

            agent = MassGenOrchestratedAgent(
                config_path=config_path,
                heartbeat_seconds=heartbeat_seconds,
                event_logger=event_logger,
            )
            agent.reset(example.id, example.target)

            observation, reward, terminated, truncated, info = env.step("")
            step_count = 0
            step_durations: list[float] = []
            selected_agents: list[str] = []

            while not (terminated or truncated):
                step_count += 1
                step_start = time.time()

                event_logger(
                    {
                        "event": "env_step_start",
                        "example_id": example.id,
                        "target": example.target,
                        "env_step": step_count,
                        "timestamp": _now_iso(),
                    }
                )

                try:
                    action_text = await agent.act(observation["text"])
                except Exception as exc:
                    print(f"  ⚠ Agent error: {exc}")
                    action_text = "impossible: Agent error"
                    event_logger(
                        {
                            "event": "env_step_agent_error",
                            "example_id": example.id,
                            "target": example.target,
                            "env_step": step_count,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "timestamp": _now_iso(),
                        }
                    )

                print(f"  Step {step_count}: {action_text}")
                observation, reward, terminated, truncated, info = env.step(action_text)
                step_elapsed = time.time() - step_start
                step_durations.append(step_elapsed)
                all_step_durations.append(step_elapsed)

                metadata = agent.last_massgen_metadata or {}
                selected_agent = metadata.get("massgen_selected_agent")
                if selected_agent:
                    selected_agents.append(selected_agent)
                    selected_agents_counter[selected_agent] += 1

                event_logger(
                    {
                        "event": "env_step_complete",
                        "example_id": example.id,
                        "target": example.target,
                        "env_step": step_count,
                        "action_text": action_text,
                        "step_elapsed_seconds": round(step_elapsed, 3),
                        "massgen_call_elapsed_seconds": round(agent.last_call_elapsed_seconds, 3),
                        "selected_agent": selected_agent,
                        "vote_results": metadata.get("massgen_vote_results"),
                        "session_id": metadata.get("massgen_session_id"),
                        "log_directory": metadata.get("massgen_log_directory"),
                        "timestamp": _now_iso(),
                    }
                )

            success = env.success
            status = "✅" if success else "❌"
            reason = info.get("reason", "unknown")
            example_elapsed = time.time() - example_start
            print(
                f"  {status} Result: success={success} | reason={reason} | "
                f"steps={step_count} | elapsed={example_elapsed:.1f}s"
            )

            event_logger(
                {
                    "event": "example_complete",
                    "example_id": example.id,
                    "target": example.target,
                    "success": success,
                    "reason": reason,
                    "steps": step_count,
                    "elapsed_seconds": round(example_elapsed, 3),
                    "timestamp": _now_iso(),
                }
            )

            results.append(
                {
                    "example_id": example.id,
                    "target": example.target,
                    "impossible": example.impossible,
                    "complexity": example.complexity,
                    "success": success,
                    "steps": step_count,
                    "reason": reason,
                    "example_elapsed_seconds": round(example_elapsed, 3),
                    "step_durations_seconds": [round(x, 3) for x in step_durations],
                    "selected_agents": selected_agents,
                }
            )

            # Reclaim memory from the previous orchestrator session
            del agent
            gc.collect()
            await asyncio.sleep(1)

        elapsed = time.time() - run_start
        total = len(results)
        successes = sum(r["success"] for r in results)
        success_rate = (successes / total) if total else 0.0

        possible = [r for r in results if not r["impossible"]]
        impossible = [r for r in results if r["impossible"]]
        possible_success = sum(r["success"] for r in possible) if possible else 0
        impossible_success = sum(r["success"] for r in impossible) if impossible else 0

        avg_seconds_per_example = (elapsed / total) if total else 0.0

        print("\n" + "=" * 80)
        print("📊 EVALUATION RESULTS (MassGen Orchestrator via LiteLLM)")
        print("=" * 80)
        print(f"  Config:           {config_path}")
        print(f"  Split:            {split}")
        print(f"  Total examples:   {total}")
        print(f"  Overall:          {successes}/{total} = {success_rate:.1%}")
        if possible:
            print(
                f"  Possible tasks:   {possible_success}/{len(possible)} = "
                f"{possible_success / len(possible):.1%}"
            )
        if impossible:
            print(
                f"  Impossible tasks: {impossible_success}/{len(impossible)} = "
                f"{impossible_success / len(impossible):.1%}"
            )
        print(f"  Time elapsed:     {elapsed:.1f}s ({avg_seconds_per_example:.1f}s per example)")
        print(f"  Step latency p50: {_quantile(all_step_durations, 0.5):.1f}s")
        print(f"  Step latency p95: {_quantile(all_step_durations, 0.95):.1f}s")
        print(f"  Selected agents:  {dict(selected_agents_counter)}")
        print("=" * 80)

        payload = {
            "backend": "massgen-litellm-orchestrator",
            "config": config_path,
            "split": split,
            "max_steps": max_steps,
            "heartbeat_seconds": heartbeat_seconds,
            "generated_at": _now_iso(),
            "total": total,
            "successes": successes,
            "success_rate": success_rate,
            "elapsed_seconds": elapsed,
            "avg_seconds_per_example": avg_seconds_per_example,
            "step_latency_p50": _quantile(all_step_durations, 0.5),
            "step_latency_p95": _quantile(all_step_durations, 0.95),
            "selected_agent_counts": dict(selected_agents_counter),
            "step_log_jsonl": str(step_jsonl_path),
            "results": results,
        }

        with open(result_json_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)

        print(f"\n💾 Results saved to {result_json_path}")
        print(f"🧾 Step logs saved to {step_jsonl_path}")

        report_generated = False
        if auto_generate_report:
            try:
                from scripts.generate_benchmark_report import generate_benchmark_report

                generate_benchmark_report(
                    input_json_path=str(result_json_path),
                    step_jsonl_path=str(step_jsonl_path),
                    output_path=report_path,
                )
                report_generated = True
                print(f"📄 Benchmark report generated: {report_path}")
            except Exception as exc:
                print(f"⚠ Failed to auto-generate benchmark report: {exc}")
                event_logger(
                    {
                        "event": "report_generation_error",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "timestamp": _now_iso(),
                    }
                )

        event_logger(
            {
                "event": "run_complete",
                "results_json": str(result_json_path),
                "steps_jsonl": str(step_jsonl_path),
                "report_generated": report_generated,
                "report_path": report_path,
                "timestamp": _now_iso(),
            }
        )

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plancraft MassGen Orchestrator Benchmark Evaluation"
    )
    parser.add_argument("--split", default="val.small", help="Dataset split to evaluate on")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Max examples to evaluate (0 = all)",
    )
    parser.add_argument(
        "--config",
        default="plancraft/agents/massgen_config.yaml",
        help="MassGen YAML config path",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=int,
        default=30,
        help="How often to log heartbeat while waiting for MassGen response",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable auto-generation of BENCHMARK_REPORT.md",
    )
    parser.add_argument(
        "--report-path",
        default="BENCHMARK_REPORT.md",
        help="Output path for markdown benchmark report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        run_evaluation(
            split=args.split,
            max_steps=args.max_steps,
            max_examples=args.max_examples,
            config_path=args.config,
            heartbeat_seconds=args.heartbeat_seconds,
            auto_generate_report=not args.no_report,
            report_path=args.report_path,
        )
    )

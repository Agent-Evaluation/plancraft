"""
Generate a markdown benchmark report from MassGen evaluation outputs.

Inputs:
- Evaluation JSON from eval_massgen.py
- Optional step-level JSONL logs

Output:
- BENCHMARK_REPORT.md (or custom path)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


def _bucket_name(complexity: int | None) -> str:
    c = complexity or 0
    if c <= 2:
        return "Low (1-2)"
    if c <= 8:
        return "Medium (3-8)"
    return "High (9+)"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    idx = int((len(s) - 1) * q)
    return s[idx]


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _extract_step_latencies(step_jsonl_path: str | None) -> list[float]:
    if not step_jsonl_path:
        return []
    path = Path(step_jsonl_path)
    if not path.exists():
        return []

    latencies: list[float] = []
    with open(path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") != "env_step_complete":
                continue
            value = row.get("massgen_call_elapsed_seconds")
            if isinstance(value, (int, float)):
                latencies.append(float(value))
    return latencies


def generate_benchmark_report(
    input_json_path: str,
    output_path: str = "BENCHMARK_REPORT.md",
    step_jsonl_path: str | None = None,
) -> None:
    with open(input_json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    results: list[dict[str, Any]] = data.get("results", [])
    total = len(results)
    successes = sum(1 for r in results if r.get("success"))
    success_rate = _safe_pct(successes, total)

    avg_time_per_example = float(data.get("avg_seconds_per_example", 0.0))

    buckets = {
        "Low (1-2)": {"total": 0, "success": 0},
        "Medium (3-8)": {"total": 0, "success": 0},
        "High (9+)": {"total": 0, "success": 0},
    }
    failure_reasons: Counter[str] = Counter()
    selected_agents: Counter[str] = Counter(data.get("selected_agent_counts", {}))

    for row in results:
        bucket = _bucket_name(row.get("complexity"))
        buckets[bucket]["total"] += 1
        if row.get("success"):
            buckets[bucket]["success"] += 1
        else:
            failure_reasons[row.get("reason", "unknown")] += 1

        for agent_id in row.get("selected_agents", []):
            selected_agents[agent_id] += 1

    latencies = _extract_step_latencies(step_jsonl_path or data.get("step_log_jsonl"))
    if not latencies:
        for row in results:
            latencies.extend(row.get("step_durations_seconds", []))

    latency_p50 = _quantile(latencies, 0.50)
    latency_p95 = _quantile(latencies, 0.95)

    output = Path(output_path)
    with open(output, "w", encoding="utf-8") as fp:
        fp.write("# Research Benchmark: MassGen Orchestrator Evaluation Report\n")
        fp.write(f"**Config**: {data.get('config', 'unknown')}\n")
        fp.write(f"**Backend**: {data.get('backend', 'massgen-litellm-orchestrator')}\n")
        fp.write(f"**Dataset**: Plancraft ({data.get('split', 'unknown')})\n")
        fp.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}\n\n")

        fp.write("## 1. Executive Summary\n")
        fp.write("| Metric | Value |\n")
        fp.write("|--------|-------|\n")
        fp.write(f"| Total Examples | {total} |\n")
        fp.write(f"| Solved | {successes} |\n")
        fp.write(f"| Success Rate | {success_rate:.1%} |\n")
        fp.write(f"| Avg Time / Example | {avg_time_per_example:.1f}s |\n")
        fp.write(f"| Step Latency p50 | {latency_p50:.1f}s |\n")
        fp.write(f"| Step Latency p95 | {latency_p95:.1f}s |\n\n")

        fp.write("## 2. Complexity Breakdown\n")
        fp.write("| Complexity | Total | Success | Rate |\n")
        fp.write("|------------|-------|---------|------|\n")
        for key in ["Low (1-2)", "Medium (3-8)", "High (9+)"]:
            b = buckets[key]
            fp.write(
                f"| {key} | {b['total']} | {b['success']} | "
                f"{_safe_pct(b['success'], b['total']):.1%} |\n"
            )
        fp.write("\n")

        fp.write("## 3. Top Failure Modes\n")
        if not failure_reasons:
            fp.write("All examples succeeded; no failure reasons to report.\n\n")
        else:
            fp.write("| Failure Reason | Count |\n")
            fp.write("|----------------|-------|\n")
            for reason, count in failure_reasons.most_common(10):
                fp.write(f"| `{reason}` | {count} |\n")
            fp.write("\n")

        fp.write("## 4. Runtime / Latency Summary\n")
        fp.write(f"- Total elapsed runtime: {float(data.get('elapsed_seconds', 0.0)):.1f}s\n")
        fp.write(f"- Average time per example: {avg_time_per_example:.1f}s\n")
        fp.write(f"- Step latency p50: {latency_p50:.1f}s\n")
        fp.write(f"- Step latency p95: {latency_p95:.1f}s\n\n")

        fp.write("## 5. MassGen Orchestration Metadata\n")
        if not selected_agents:
            fp.write("No selected-agent metadata captured.\n")
        else:
            fp.write("| Selected Agent | Count |\n")
            fp.write("|----------------|-------|\n")
            for agent_id, count in selected_agents.most_common():
                fp.write(f"| {agent_id} | {count} |\n")

        fp.write("\n")
        fp.write("## 6. Artifact Paths\n")
        fp.write(f"- Results JSON: `{input_json_path}`\n")
        fp.write(
            f"- Step JSONL: `{step_jsonl_path or data.get('step_log_jsonl', 'not_available')}`\n"
        )
        fp.write(f"- Report: `{output_path}`\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate BENCHMARK_REPORT.md from MassGen eval output")
    parser.add_argument("--input", required=True, help="Path to eval_massgen JSON output")
    parser.add_argument("--steps", default=None, help="Optional step JSONL path")
    parser.add_argument("--output", default="BENCHMARK_REPORT.md", help="Output markdown report path")
    args = parser.parse_args()

    generate_benchmark_report(
        input_json_path=args.input,
        output_path=args.output,
        step_jsonl_path=args.steps,
    )
    print(f"Report generated: {args.output}")

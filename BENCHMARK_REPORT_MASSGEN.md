# Research Benchmark: MassGen Orchestrator Evaluation Report
**Config**: plancraft/agents/massgen_config.yaml
**Backend**: massgen-litellm-orchestrator
**Dataset**: Plancraft (val.small)
**Date**: 2026-02-27

## 1. Executive Summary
| Metric | Value |
|--------|-------|
| Total Examples | 110 |
| Solved | 20 |
| Success Rate | 18.2% |
| Avg Time / Example | 0.2s |
| Step Latency p50 | 0.0s |
| Step Latency p95 | 0.0s |

## 2. Complexity Breakdown
| Complexity | Total | Success | Rate |
|------------|-------|---------|------|
| Low (1-2) | 38 | 20 | 52.6% |
| Medium (3-8) | 30 | 0 | 0.0% |
| High (9+) | 42 | 0 | 0.0% |

## 3. Top Failure Modes
| Failure Reason | Count |
|----------------|-------|
| `incorrect_stop` | 90 |

## 4. Runtime / Latency Summary
- Total elapsed runtime: 22.7s
- Average time per example: 0.2s
- Step latency p50: 0.0s
- Step latency p95: 0.0s

## 5. MassGen Orchestration Metadata
No selected-agent metadata captured.

## 6. Artifact Paths
- Results JSON: `output\massgen_orchestrator_val.small_20260227_154156.json`
- Step JSONL: `output\massgen_steps_val.small_20260227_154156.jsonl`
- Report: `BENCHMARK_REPORT.md`

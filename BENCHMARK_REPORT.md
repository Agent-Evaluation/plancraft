# Research Benchmark: Single Agent Copilot Evaluation

**Model**: gpt-5-mini
**Dataset**: Plancraft (val.small, N=110)
**Date**: 2026-02-16

## 1. Executive Summary
- **Overall Success Rate**: 80.0%
- **Average Steps (Successes)**: 5.9
- **Total Examples Solved**: 88/110

## 2. Performance by Complexity
The agent demonstrates strong performance on low-complexity tasks but shows degradation as task complexity increases.

| Complexity Level | Total | Success | Success Rate | Avg Steps |
|------------------|-------|---------|--------------|-----------|
| Low (1-2) | 38 | 38 | 100.0% | 1.5 |
| Medium (3-8) | 30 | 28 | 93.3% | 6.4 |
| High (9+) | 42 | 22 | 52.4% | 12.9 |

## 3. Failure Analysis
Total Failures: 22

| Failure Reason | Count |
|----------------|-------|
| `incorrect_stop` | 13 |
| `max_steps_reached` | 9 |

### Analysis of Failures
- **`incorrect_stop`**: The agent hallucinated that the task was impossible or claimed success prematurely. This is common in high-complexity tasks where the agent fails to plan deep dependency trees (e.g., needing to craft intermediate tools).
- **`max_steps_reached`**: The agent got stuck in a loop or inefficiently wandered through the crafting graph, exceeding the 30-step limit. This typically happens with deep recipe chains (Complexity 9+).

## 4. Conclusion & Recommendations
The **Single Agent** architecture using `gpt-5-mini` provides a strong baseline with **80.0% accuracy** on the `val.small` set.

### Strengths
- **Efficiency**: Solves simple tasks (Complexity 1-2) with near-optimal step counts.
- **Tool Usage**: Correctly uses the `search` tool to discover recipes.

### Weaknesses
- **Long-Horizon Planning**: Struggles with high-complexity items (Complexity > 9) leading to timeouts or incorrect stops.
- **False Negatives**: Occasionally incorrectly labels feasible tasks as `impossible`.

### Future Work
- **Multi-Agent Comparison**: A Multi-Agent System (MAS) could tackle high-complexity tasks by parallelizing sub-goals.
- **Planning Decomposition**: Separating planning from execution could reduce `max_steps_reached` failures.

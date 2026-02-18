# Research Benchmark: Copilot Evaluation Report
**Model**: gpt-5-mini
**Dataset**: Plancraft (val.small)
**Date**: 2026-02-18

## 1. Executive Summary
| Architecture | Success Rate | Avg Time/Ex | Status |
|--------------|--------------|-------------|--------|
| Single       | 82.7% | 227.9s | ✅ Complete |
| Independent  | 77.3% | 626.2s | ✅ Complete |
| Centralized  | TBD          | TBD         | 🔄 Running... |
| Decentralized| TBD          | TBD         | ⏳ Pending |
| Hybrid       | TBD          | TBD         | ⏳ Pending |

## 2. Single Agent Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate |
|------------|-------|---------|------|
| Low (1-2) | 18 | 18 | 100.0% |
| Medium (3-8) | 30 | 26 | 86.7% |
| High (9+) | 62 | 47 | 75.8% |

### Top Failure Modes
- `max_steps_reached`: 10
- `incorrect_stop`: 9

## 3. Independent Agent Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate |
|------------|-------|---------|------|
| Low (1-2) | 18 | 17 | 94.4% |
| Medium (3-8) | 30 | 26 | 86.7% |
| High (9+) | 62 | 42 | 67.7% |

### Top Failure Modes
- `incorrect_stop`: 14
- `max_steps_reached`: 11
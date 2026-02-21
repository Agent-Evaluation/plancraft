# Research Benchmark: Copilot Evaluation Report
**Model**: gpt-5-mini
**Dataset**: Plancraft (val.small)
**Date**: 2026-02-18

## 1. Executive Summary
| Architecture | Success Rate | Avg Time/Ex | Status |
|--------------|--------------|-------------|--------|
| Single       | 82.7% | 227.9s | ✅ Complete |
| Independent  | 77.3% | 626.2s | ✅ Complete |
| Centralized  | 66.4% | 409.5s | ✅ Complete |
| Decentralized| TBD          | TBD         | 🔄 Running... |
| Hybrid       | 52.7% | 678.7s | ✅ Complete |

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

## 4. Centralized Agent Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate |
|------------|-------|---------|------|
| Low (1-2) | 18 | 18 | 100.0% |
| Medium (3-8) | 30 | 25 | 83.3% |
| High (9+) | 42 | 10 | 23.8% |

### Top Failure Modes
- `incorrect_stop`: 24
- `max_steps_reached`: 13

## 5. Hybrid Agent Analysis
### Complexity Breakdown
| Complexity | Total | Success | Rate |
|------------|-------|---------|------|
| Low (1-2) | 18 | 12 | 66.7% |
| Medium (3-8) | 30 | 17 | 56.7% |
| High (9+) | 42 | 9 | 21.4% |

### Top Failure Modes
- `incorrect_stop`: 32
- `max_steps_reached`: 20
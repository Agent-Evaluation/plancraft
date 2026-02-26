# MassGen Eval Run Guide

This file documents how to run the MassGen orchestrator benchmark evaluator for Plancraft using the current implementation in `eval_massgen.py`.

## 1) Environment setup (MassGen docs style)

MassGen requires Python 3.11+.

```powershell
python --version
```

Install and use `uv` (Windows):

```powershell
pip install uv
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install massgen litellm python-dotenv
```

Install project dependencies:

```powershell
uv pip install -e .
```

## 2) Quick smoke test

```powershell
.\.venv\Scripts\python.exe -c "import massgen, litellm, dotenv; print('imports_ok')"
```

## 3) Run evaluator

Minimal smoke run (1 example, 1 step):

```powershell
.\.venv\Scripts\python.exe eval_massgen.py --split val.small --max-steps 1 --max-examples 1 --config plancraft/agents/massgen_config.yaml --heartbeat-seconds 30 --no-report
```

Normal run with report generation:

```powershell
.\.venv\Scripts\python.exe eval_massgen.py --split val.small --max-steps 30 --config plancraft/agents/massgen_config.yaml --heartbeat-seconds 30 --report-path BENCHMARK_REPORT.md
```

## 4) What gets generated

- Results JSON: `output/massgen_orchestrator_<split>_<timestamp>.json`
- Step JSONL logs: `output/massgen_steps_<split>_<timestamp>.jsonl`
- Benchmark report (unless `--no-report`): `BENCHMARK_REPORT.md`

## 5) Optional: Generate report manually

```powershell
.\.venv\Scripts\python.exe scripts/generate_benchmark_report.py --input output/<your_results>.json --steps output/<your_steps>.jsonl --output BENCHMARK_REPORT.md
```

## 6) Notes on logging behavior

- Heartbeat logs are emitted every `--heartbeat-seconds` while waiting on MassGen.
- Oracle search injections are logged when model returns `search: ...`.
- Step-level metadata (`selected_agent`, vote results, session/log paths) is recorded in JSONL when available.

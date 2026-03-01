"""
Memory-monitoring wrapper for eval_massgen.py.

Spawns eval_massgen.py as a subprocess and polls its memory (RSS, VMS, and
children) every few seconds, writing timestamped rows to a CSV log file.

Usage:
    uv run monitor_eval.py [-- eval_massgen args ...]
"""

import csv
import os
import sys
import time
import subprocess
import psutil
from datetime import datetime
from pathlib import Path


POLL_INTERVAL_SECONDS = 3
LOG_DIR = Path("output")


def mb(bytes_val: int) -> float:
    return round(bytes_val / (1024 * 1024), 2)


def monitor(proc: subprocess.Popen, log_path: Path) -> None:
    try:
        ps = psutil.Process(proc.pid)
    except psutil.NoSuchProcess:
        print(f"[monitor] Process {proc.pid} not found")
        return

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "elapsed_s",
            "main_rss_mb",
            "main_vms_mb",
            "children_count",
            "children_rss_mb",
            "total_rss_mb",
            "total_vms_mb",
            "note",
        ])

        start = time.time()
        peak_total_rss = 0.0

        while proc.poll() is None:
            elapsed = round(time.time() - start, 1)
            ts = datetime.now().isoformat(timespec="seconds")
            note = ""

            try:
                mem = ps.memory_info()
                main_rss = mb(mem.rss)
                main_vms = mb(mem.vms)

                children = ps.children(recursive=True)
                children_rss = sum(
                    mb(c.memory_info().rss) for c in children
                    if c.is_running()
                )
                children_vms = sum(
                    mb(c.memory_info().vms) for c in children
                    if c.is_running()
                )
                total_rss = round(main_rss + children_rss, 2)
                total_vms = round(main_vms + children_vms, 2)

                if total_rss > peak_total_rss:
                    peak_total_rss = total_rss
                    note = "new_peak"

                writer.writerow([
                    ts, elapsed,
                    main_rss, main_vms,
                    len(children), children_rss,
                    total_rss, total_vms,
                    note,
                ])
                f.flush()

                print(
                    f"[mem {ts}] {elapsed:>7.1f}s | "
                    f"main={main_rss:>8.1f} MB | "
                    f"children({len(children)})={children_rss:>8.1f} MB | "
                    f"TOTAL={total_rss:>8.1f} MB | "
                    f"peak={peak_total_rss:>8.1f} MB"
                    f"{'  << NEW PEAK' if note else ''}"
                )

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                note = "process_gone"
                writer.writerow([ts, elapsed, 0, 0, 0, 0, 0, 0, note])
                f.flush()

            time.sleep(POLL_INTERVAL_SECONDS)

    print(f"\n[monitor] Process exited with code {proc.returncode}")
    print(f"[monitor] Peak total RSS: {peak_total_rss:.1f} MB")
    print(f"[monitor] Memory log: {log_path}")


def main() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"memory_log_{timestamp}.csv"

    # Build the eval command
    eval_args = sys.argv[1:] if len(sys.argv) > 1 else []
    cmd = [sys.executable, "eval_massgen.py"] + eval_args

    print(f"[monitor] Command: {' '.join(cmd)}")
    print(f"[monitor] Memory log: {log_path}")
    print(f"[monitor] Poll interval: {POLL_INTERVAL_SECONDS}s")
    print("-" * 80)

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        cwd=os.getcwd(),
    )

    try:
        monitor(proc, log_path)
    except KeyboardInterrupt:
        print("\n[monitor] Interrupted, terminating subprocess...")
        proc.terminate()
        proc.wait(timeout=10)

    sys.exit(proc.returncode or 0)


if __name__ == "__main__":
    main()

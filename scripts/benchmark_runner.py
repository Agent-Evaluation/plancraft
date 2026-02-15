
import subprocess
import time
import argparse
import os
import json
from datetime import datetime

# Architectures to benchmark
ARCHITECTURES = ["single", "independent", "centralized", "decentralized", "hybrid"]

BACKEND_SCRIPTS = {
    "gemini": "eval_gemini.py",
    "copilot": "eval_copilot.py",
}

def run_benchmark(limit: int, model: str, max_steps: int, output_dir: str, backend: str):
    """
    Runs the benchmark for all architectures sequentially.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = {}
    
    eval_script = BACKEND_SCRIPTS[backend]
    print(f"🔧 Backend: {backend} ({eval_script})")
    print(f"🤖 Model: {model}")
    
    for arch in ARCHITECTURES:
        print(f"\n🚀 Starting benchmark for: {arch}")
        start_time = time.time()
        
        # Command to run the eval script
        cmd = [
            "python3", eval_script,
            "--split", "val.small.easy",
            "--max-examples", str(limit),
            "--max-steps", str(max_steps),
            "--model", model,
            "--architecture", arch
        ]
        
        try:
            # Run the command and capture output
            result = subprocess.run(cmd, check=True)
            
            # Find the output file
            prefix = "copilot" if backend == "copilot" else "gemini"
            output_files = sorted(
                [f for f in os.listdir("output") if f.endswith(".json") and arch in f and prefix in f],
                key=lambda x: os.path.getmtime(os.path.join("output", x)),
                reverse=True
            )
            
            if output_files:
                last_file = os.path.join("output", output_files[0])
                with open(last_file) as f:
                    data = json.load(f)
                
                report[arch] = {
                    "success_rate": data["success_rate"],
                    "avg_steps": sum(r["steps"] for r in data["results"]) / len(data["results"]),
                    "file": last_file
                }
                print(f"✅ {arch} completed. Success Rate: {data['success_rate']:.1%}")
            else:
                print(f"⚠ Could not find output file for {arch}")
                report[arch] = "Output not found"

        except subprocess.CalledProcessError as e:
            print(f"❌ {arch} failed: {e}")
            report[arch] = "Failed"
            
        # Rate limiting / Cool-down between architectures
        print("Waiting 10s cooldown...")
        time.sleep(10)

    # Save summary report
    summary_file = os.path.join(output_dir, f"benchmark_{backend}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(summary_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n🏁 Benchmark complete! Summary saved to {summary_file}")
    return summary_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20, help="Examples per agent")
    parser.add_argument("--model", default="gpt-5-mini", help="Model to use")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per example")
    parser.add_argument("--output-dir", default="benchmark_results", help="Directory for summary")
    parser.add_argument(
        "--backend",
        default="copilot",
        choices=["gemini", "copilot"],
        help="LLM backend to use (default: copilot)"
    )
    
    args = parser.parse_args()
    
    run_benchmark(args.limit, args.model, args.max_steps, args.output_dir, args.backend)


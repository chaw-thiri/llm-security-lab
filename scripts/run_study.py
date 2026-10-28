"""Run predeclared training comparisons sequentially, then fixed-test and Ollama evaluation.

Pause: create runs/PAUSE_STUDY. Training saves at its next optimizer boundary
after validation; the runner exits before beginning another stage. Resume with
this script's --resume flag. Checkpoints/data/results remain local and ignored.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from promptshield.common import read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    (ROOT / "runs").mkdir(exist_ok=True)
    status_path = ROOT / "runs/study-status.json"
    if status_path.exists() and not args.resume:
        raise SystemExit("Study status already exists. Inspect it, then use --resume to continue.")
    if args.resume:
        (ROOT / "runs/PAUSE_STUDY").unlink(missing_ok=True)
    # One runner at a time, independently of the per-training-run lock.
    lock = ROOT / "runs/.study.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state = {"status": "running", "phase": "initializing", "pid": os.getpid(), "completed": [],
             "started_at_utc": datetime.now(timezone.utc).isoformat()}
    def update(phase, status="running"):
        state.update(phase=phase, status=status, updated_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(status_path, state)
        print(json.dumps(state), flush=True)
    def command(phase, *arguments):
        update(phase)
        subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)
        state["completed"].append(phase)
    try:
        for name in ("full", "frozen", "mean"):
            if (ROOT / "runs/PAUSE_STUDY").exists():
                update("between_stages", "paused")
                return
            run = ROOT / "runs" / name
            prior = read_json(run / "status.json") if (run / "status.json").exists() else {}
            if prior.get("status") == "complete":
                state["completed"].append("train_" + name)
                continue
            arguments = ["-m", "promptshield.train", "--config", f"configs/{name}.json"]
            if (run / "config.json").exists():
                arguments.append("--resume")
            command("train_" + name, *arguments)
            if read_json(run / "status.json")["status"] == "paused":
                update("train_" + name, "paused")
                return
        for name in ("full", "frozen", "mean"):
            if (ROOT / "runs/PAUSE_STUDY").exists():
                update("before_evaluation", "paused")
                return
            report_path = ROOT / f"results/{name}-test/metrics.json"
            if report_path.exists():
                existing = read_json(report_path)
                if existing["checkpoint_weights_sha256"] != sha256(ROOT / f"runs/{name}/best/model.safetensors"):
                    raise ValueError("Existing test report uses different weights; review before repeating test evaluation.")
                state["completed"].append("test_" + name)
            else:
                command("test_" + name, "-m", "promptshield.evaluate", "--checkpoint", f"runs/{name}/best",
                        "--output", f"results/{name}-test", "--batch-size", "4")
        command("security_demo", "-m", "promptshield.security_eval", "--checkpoint", "runs/full/best",
                "--output", "results/security")
        summary = {"status": "complete", "comparisons": {},
                   "data_manifest_sha256": sha256(ROOT / "data/splits/piguard-v1/manifest.json"),
                   "security": read_json(ROOT / "results/security/security-metrics.json"),
                   "limitations": ["Single seed; case/whitespace deduplication does not rule out template leakage.",
                                   "Sources/styles are shared across the stratified split.",
                                   "Frozen head uses a higher learning rate: comparison is between declared training recipes, not only freezing.",
                                   "20 synthetic security cases are not evidence of production readiness."]}
        for name in ("full", "frozen", "mean"):
            summary["comparisons"][name] = {"config": read_json(ROOT / f"configs/{name}.json"),
                "checkpoint": read_json(ROOT / f"runs/{name}/best/detector.json"),
                "held_out": read_json(ROOT / f"results/{name}-test/metrics.json")}
        write_json(ROOT / "docs/experiment-results.json", summary)
        update("finished", "complete")
    except KeyboardInterrupt:
        update(state["phase"], "interrupted")
        raise
    except Exception as exc:
        state["error"] = type(exc).__name__ + ": " + str(exc)
        update(state["phase"], "failed")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

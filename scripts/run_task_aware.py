"""Run the preregistered matched task-aware and document-only training arms."""
from pathlib import Path
import subprocess
import sys

from promptshield.common import read_json, write_json


def main():
    root = Path("results/task-aware-v1")
    root.mkdir(parents=True, exist_ok=True)
    runs = [(arm, seed) for seed in (42, 43, 44) for arm in ("document", "task-document")]
    for index, (arm, seed) in enumerate(runs, 1):
        run = Path(f"runs/task-aware-{arm}-{seed}")
        status = run / "status.json"
        if status.exists() and read_json(status).get("status") == "complete":
            print("SKIP complete", arm, seed, flush=True)
            continue
        config = f"configs/task-aware-{arm}-{seed}.json"
        command = [sys.executable, "-m", "promptshield.train", "--config", config]
        if (run / "config.json").exists():
            command.append("--resume")
        write_json(root / "study-status.json", {"status": "training", "arm": arm,
                                                  "seed": seed, "run_index": index,
                                                  "total_runs": len(runs)})
        print("START", index, "/", len(runs), arm, seed, flush=True)
        subprocess.run(command, check=True)
        if not status.exists() or read_json(status).get("status") != "complete":
            raise RuntimeError(f"Run did not finish: {arm} seed {seed}; inspect {run} and resume later.")
    write_json(root / "study-status.json", {"status": "training_complete",
                                              "runs": [f"{arm}-{seed}" for arm, seed in runs]})
    print("ALL TASK-AWARE RUNS COMPLETE", flush=True)


if __name__ == "__main__":
    main()

"""Boundary tests for paired training, calibration and access to exposed evaluation."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from promptshield.common import read_json, sha256, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pooling_study as study


class PoolingStudyTests(unittest.TestCase):
    def test_every_pair_differs_only_in_pooling_and_output(self):
        plan = read_json(ROOT / study.PROTOCOL)
        base = read_json(ROOT / plan["base_config"])
        configs = study.run_configs(plan, base)
        self.assertEqual(list(configs), [f"{arm}-{seed}" for seed in (42, 43, 44) for arm in ("attention", "cls")])
        for seed in plan["seeds"]:
            attention, control = (configs[f"{a}-{seed}"] for a in ("attention", "cls"))
            self.assertEqual({k for k in control if control[k] != attention[k]}, {"pooling", "run_dir"})
            self.assertEqual({k for k in base if base[k] != control[k]} - {"seed"}, {"run_dir"})
            self.assertEqual(control["seed"], seed)

    def test_all_policies_locked_before_first_replay_and_pause_never_opens_it(self):
        plan = read_json(ROOT / study.PROTOCOL)
        base = read_json(ROOT / plan["base_config"])
        start = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            self.addCleanup(os.chdir, start)
            try:
                base.update(data_dir="development-fixture")
                write_jsonl(Path(base["data_dir"]) / "calibration.jsonl", [{"unused": True}])
                trained, opened = [], []

                def fake_train(path, resume):
                    config = read_json(path)
                    run = Path(config["run_dir"])
                    self.assertFalse(resume)
                    trained.append(run.name)
                    write_json(run / "config.json", config)
                    write_json(run / "status.json", {"status": "complete", "updates": 512})
                    (run / "best").mkdir()
                    (run / "best/model.safetensors").write_bytes(run.name.encode())

                def calibrate(config, _, rows):
                    best = Path(config["run_dir"]) / "best"
                    return {"checkpoint": str(best), "weights_sha256": sha256(best / "model.safetensors"), "threshold": .25}

                def replay(policy, cohorts, device):
                    frozen = read_json(study.OUT / "frozen-policies.json")["policies"]
                    self.assertEqual(set(frozen), set(trained))
                    self.assertEqual(len(frozen), 6)
                    opened.append(policy["checkpoint"])
                    counts = dict(attacks_detected=1, false_alarms=1, correct_benign_answers=1, exact_hijacks=1)
                    return {c: {"detector": counts, "by_attack_position": {p: counts for p in ("start", "middle", "end")}}
                            for c in cohorts}

                with patch.object(study, "preflight", return_value=(plan, base, {})), \
                        patch.object(study, "train", side_effect=fake_train), \
                        patch.object(study, "calibrate", side_effect=calibrate), \
                        patch.object(study, "exposed_replay", side_effect=replay):
                    study.RUNS.mkdir(parents=True)
                    pause = study.RUNS / "PAUSE_STUDY"
                    pause.touch()
                    study.run()
                    self.assertEqual((trained, opened), ([], []))
                    pause.unlink()
                    study.run()
                    self.assertEqual((len(trained), len(opened)), (6, 6))
                    self.assertEqual(read_json(study.OUT / "status.json")["phase"], "complete")
                    study.run()  # A rerun must not overwrite or remeasure completed arms.
                    self.assertEqual((len(trained), len(opened)), (6, 6))
            finally:
                os.chdir(start)

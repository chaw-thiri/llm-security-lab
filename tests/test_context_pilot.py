"""Check quotation counterexamples and separation of the synthetic pilot."""
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location("pilot",Path(__file__).resolve().parents[1]/"scripts/prepare_context_pilot.py")
pilot=importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)
class ContextPilotTests(unittest.TestCase):
    def test_both_labels_with_and_without_quotes(self):
        for split in ("train","validation","test"):
            rows=pilot.examples(split)
            counts={(label,q):sum(r["label"]==label and r["quoted"]==q for r in rows) for label in (0,1) for q in (False,True)}
            self.assertEqual(len(set(counts.values())),1)
            self.assertGreater(min(counts.values()),0)
    def test_split_payloads_and_examples_are_disjoint(self):
        names=("train","validation","test")
        for i,a in enumerate(names):
            for b in names[i+1:]:
                self.assertFalse(set(pilot.PAYLOADS[a]) & set(pilot.PAYLOADS[b]))
                self.assertFalse({r["prompt"] for r in pilot.examples(a)} & {r["prompt"] for r in pilot.examples(b)})

class ContinuationConfigTests(unittest.TestCase):
    def test_continuation_requires_checkpoint_and_hash_together(self):
        import json
        import tempfile
        from promptshield.train import load_config
        root=Path(__file__).resolve().parents[1]
        base=json.loads((root/"configs/full.json").read_text())
        for key,value in (("initial_checkpoint","example"),("initial_weights_sha256","abc")):
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/"config.json"
                path.write_text(json.dumps(dict(base,**{key:value})))
                with self.assertRaisesRegex(ValueError,"requires both"):
                    load_config(path)

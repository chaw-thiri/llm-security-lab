"""Baseline preservation and pinned-acquisition tests; no network, GPU or real checkpoints required."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from promptshield.acquire import acquire, blob_sha
from promptshield.data import TextDataset

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_baseline.py"
spec = importlib.util.spec_from_file_location("baseline_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class BaselineToolsTests(unittest.TestCase):
    def test_hash_verification_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "weights.bin").write_bytes(b"original")
            manifest = {"artifacts": [{"path": "weights.bin", "bytes": 8,
                        "sha256": hashlib.sha256(b"original").hexdigest()}]}
            self.assertEqual(verifier.verify(root, manifest), [])
            (root / "weights.bin").write_bytes(b"modified")
            self.assertIn("SHA256 mismatch", verifier.verify(root, manifest)[0])

    def test_missing_artifact_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = {"artifacts": [{"path": "missing", "bytes": 1, "sha256": "unused"}]}
            self.assertIn("Missing", verifier.verify(Path(directory), manifest)[0])

    def test_verifier_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            verifier.verify(Path("."), {"artifacts": [{"path": "../outside"}]})

    def test_only_prompt_text_reaches_tokenizer(self):
        class RecordingTokenizer:
            def __call__(self, texts, **kwargs):
                self.texts = texts
                return {"input_ids": [[2, 3] for _ in texts], "attention_mask": [[1, 1] for _ in texts]}
        tokenizer = RecordingTokenizer()
        dataset = TextDataset([{"prompt": "document", "source": "do-not-tokenize", "label": 1}], tokenizer, 16)
        self.assertEqual(tokenizer.texts, ["document"])
        self.assertEqual(set(dataset[0]), {"input_ids", "attention_mask", "labels"})

    def test_pinned_existing_data_requires_no_head_lookup_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = {"train.json": b"[]", "valid.json": b"[ ]"}
            blobs = {name: blob_sha(content) for name, content in data.items()}
            pinned = {"repository_revision": "a" * 40, "files": {}}
            for name, content in data.items():
                (root / name).write_bytes(content)
                pinned["files"][name] = {"git_blob_sha": blobs[name], "sha256": hashlib.sha256(content).hexdigest()}
            manifest = root / "baseline.json"
            manifest.write_text(json.dumps({"data_provenance": pinned}))
            with patch("promptshield.acquire.EXPECTED_BLOBS", blobs), patch("requests.Session.get", side_effect=AssertionError("Network forbidden in this test")):
                actual = acquire(root, with_backbone=False, baseline=manifest)
            self.assertEqual(actual["repository_revision"], "a" * 40)
            self.assertIn("/" + "a" * 40 + "/", actual["files"]["train.json"]["url"])


if __name__ == "__main__":
    unittest.main()

"""Offline tests: leakage boundaries, real transformer gradients, checkpoint reload and fail-closed gating."""
import tempfile
import copy
from pathlib import Path
import unittest

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast

from promptshield.application import process_document
from promptshield.predict import Detector
from promptshield.classifier import InjectionClassifier
from promptshield.data import clean_rows, split_rows
from promptshield.scoring import metrics, select_threshold
from promptshield.security_eval import canonical, synthetic_cases
from promptshield.train import optimizer_for, schedule, rng_state, restore_rng, predict_loader, train
from promptshield.common import write_json, write_jsonl, sha256, read_json


def tiny_model(frozen=False, pooling="cls"):
    config = DistilBertConfig(vocab_size=32, dim=16, hidden_dim=32, n_layers=1, n_heads=2,
                            dropout=0, attention_dropout=0, max_position_embeddings=32)
    return InjectionClassifier(DistilBertModel(config), pooling=pooling, dropout=0, frozen=frozen)


class DataTests(unittest.TestCase):
    def test_dedup_conflicts_and_reserved_overlap(self):
        def row(text, label=0): return {"prompt": text, "label": label, "source": "fixture"}
        rows = [row("Hello"), row(" HELLO "), row("conflict"), row("Conflict", 1), row(" "), row("reserved"), row("attack", 1)]
        clean, report = clean_rows(rows, [row("RESERVED")])
        self.assertEqual({r["prompt"] for r in clean}, {" HELLO ", "attack"})
        self.assertEqual(report["conflicting_rows"], 2)
        self.assertEqual(report["blank_rows"], 1)
        self.assertEqual(report["reserved_overlap_groups"], 1)

    def test_reproducible_disjoint_split_and_both_classes(self):
        rows = [{"id": str(i), "prompt": str(i), "label": i % 2, "source": "fixture"} for i in range(100)]
        first, second = split_rows(rows), split_rows(list(reversed(rows)))
        self.assertEqual(first, second)
        ids = [set(r["id"] for r in group) for group in first.values()]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(sum(map(len, first.values())), 100)
        for group in first.values(): self.assertEqual({r["label"] for r in group}, {0, 1})

    def test_invalid_labels_are_not_silently_coerced(self):
        with self.assertRaises(ValueError):
            clean_rows([{"prompt": "text", "label": "1", "source": "fixture"}])


class MetricTests(unittest.TestCase):
    def test_threshold_enforces_budget_and_handles_ties(self):
        labels, scores = [0, 0, 1, 1], [0.2, 0.8, 0.8, 0.9]
        threshold = select_threshold(labels, scores, 0)
        result = metrics(labels, scores, threshold)
        self.assertEqual(result["benign_fpr"], 0)
        self.assertEqual(result["recall"], 0.5)

    def test_no_alert_policy_when_all_scores_equal_one(self):
        threshold = select_threshold([0, 1], [1, 1], 0)
        self.assertGreater(threshold, 1)
        self.assertEqual(metrics([0, 1], [1, 1], threshold)["recall"], 0)

    def test_majority_accuracy_is_not_attack_recall(self):
        result = metrics([0, 0, 0, 0, 1], [0, 0, 0, 0, 0])
        self.assertEqual(result["accuracy"], 0.8)
        self.assertEqual(result["recall"], 0)


class ModelTests(unittest.TestCase):
    def test_encoder_and_head_receive_gradients(self):
        model = tiny_model()
        logits = model(torch.tensor([[2, 5, 3], [2, 7, 3]]), torch.ones((2, 3), dtype=torch.long))
        self.assertEqual(tuple(logits.shape), (2, 2))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is not None for p in model.head.parameters()))

    def test_frozen_encoder_unchanged_after_update(self):
        model = tiny_model(frozen=True)
        before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
        model.train()
        self.assertFalse(model.encoder.training)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=0.01)
        logits = model(torch.tensor([[2, 5, 3]]), torch.ones((1, 3), dtype=torch.long))
        torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
        optimizer.step()
        for key, value in model.encoder.state_dict().items(): self.assertTrue(torch.equal(value, before[key]))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))

    def test_mean_pooling_does_not_include_padding(self):
        model = tiny_model(pooling="mean").eval()
        with torch.no_grad():
            a = model(torch.tensor([[2, 5, 3]]), torch.tensor([[1, 1, 1]]))
            b = model(torch.tensor([[2, 5, 3, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_safe_checkpoint_roundtrip_offline(self):
        model = tiny_model().eval()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordPiece(
            {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4}, unk_token="[UNK]")),
            unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
        ids, mask = torch.tensor([[2, 4, 3]]), torch.ones((1, 3), dtype=torch.long)
        with tempfile.TemporaryDirectory() as directory:
            model.save(directory, tokenizer, {"threshold": 0.6, "max_length": 32, "diagnostic_run": True})
            loaded, _, metadata = InjectionClassifier.load(directory)
            with torch.no_grad(): torch.testing.assert_close(model(ids, mask), loaded(ids, mask))
            self.assertEqual(metadata["threshold"], 0.6)

    def test_inference_special_tokens_truncation_and_overflow_windows(self):
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        backend = Tokenizer(WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
        with tempfile.TemporaryDirectory() as directory:
            tiny_model().save(directory, tokenizer, {"threshold": 1.01, "max_length": 6, "diagnostic_run": True})
            detector = Detector(directory, "cpu")
            short = detector.score("a a")
            long = detector.score(" ".join(["a"] * 20))
            chunked = detector.score(" ".join(["a"] * 20), chunk=True)
            self.assertFalse(short["truncated"])
            self.assertTrue(long["truncated"])
            self.assertFalse(chunked["truncated"])
            self.assertGreater(chunked["chunks"], 1)

    def test_accumulation_tail_matches_whole_batch_update(self):
        torch.manual_seed(7)
        whole = tiny_model()
        micro = copy.deepcopy(whole)
        ids = torch.tensor([[2, i, 3] for i in range(5, 10)])
        mask = torch.ones_like(ids)
        labels = torch.tensor([0, 1, 0, 1, 1])
        opt_a = torch.optim.AdamW(whole.parameters(), lr=1e-3)
        opt_b = torch.optim.AdamW(micro.parameters(), lr=1e-3)
        torch.nn.functional.cross_entropy(whole(ids, mask), labels).backward()
        for offset in range(0, 5, 2):
            losses = torch.nn.functional.cross_entropy(micro(ids[offset:offset+2], mask[offset:offset+2]), labels[offset:offset+2], reduction="none")
            (losses.sum() / 5).backward()
        torch.nn.utils.clip_grad_norm_(whole.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(micro.parameters(), 1)
        opt_a.step(); opt_b.step()
        for a, b in zip(whole.parameters(), micro.parameters()):
            torch.testing.assert_close(a, b, atol=2e-6, rtol=1e-4)

    def test_weighted_validation_loss_is_independent_of_batch_partition(self):
        class FixedModel(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return torch.tensor([[1., 0.], [0., 2.], [2., 0.]])[input_ids[:, 0]]
        model = FixedModel()
        ids = torch.tensor([[0], [1], [2]])
        labels = torch.tensor([0, 1, 1])
        weights = torch.tensor([1., 3.])
        full = [{"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}]
        pieces = [{"input_ids": ids[i:i+1], "attention_mask": torch.ones_like(ids[i:i+1]), "labels": labels[i:i+1]} for i in range(3)]
        a = predict_loader(model, full, torch.device("cpu"), weights)[2]
        b = predict_loader(model, pieces, torch.device("cpu"), weights)[2]
        self.assertAlmostEqual(a, b, places=6)

    def test_paused_training_resume_matches_uninterrupted_weights(self):
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone = root / "backbone"
            torch.manual_seed(3)
            tiny_model().encoder.save_pretrained(backbone)
            backend = Tokenizer(WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "benign": 4, "attack": 5}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
            tokenizer.save_pretrained(backbone)
            splits = root / "splits"
            rows = [{"id": str(i), "prompt": "attack" if i % 2 else "benign", "label": i % 2, "source": "fixture"} for i in range(8)]
            manifest = {"splits": {}}
            for name, group in [("train", rows), ("validation", rows[:4])]:
                path = splits / (name + ".jsonl")
                write_jsonl(path, group)
                manifest["splits"][name] = {"sha256": sha256(path)}
            write_json(splits / "manifest.json", manifest)
            config = {"backbone": str(backbone), "data_dir": str(splits), "run_dir": str(root / "uninterrupted"),
                "seed": 7, "epochs": 1, "batch_size": 2, "accumulation_steps": 2,
                "encoder_lr": 0.001, "head_lr": 0.001, "weight_decay": 0.01, "warmup_ratio": 0,
                "max_length": 16, "pooling": "cls", "dropout": 0.2, "frozen": False,
                "weighted_loss": False, "max_fpr": 0.5, "device": "cpu"}
            config_path = root / "config.json"
            write_json(config_path, config)
            train(config_path)
            expected, _, _ = InjectionClassifier.load(root / "uninterrupted/last")
            config["run_dir"] = str(root / "interrupted")
            write_json(config_path, config)
            (root / "interrupted").mkdir()
            (root / "interrupted/PAUSE").touch()
            train(config_path)
            self.assertEqual(read_json(root / "interrupted/status.json")["status"], "paused")
            self.assertFalse((root / "interrupted/PAUSE").exists())
            train(config_path, resume=True)
            actual, _, _ = InjectionClassifier.load(root / "interrupted/last")
            self.assertEqual(read_json(root / "interrupted/status.json")["updates"], 2)
            for a, b in zip(expected.parameters(), actual.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_paired_ranking_resume_matches_uninterrupted_weights(self):
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backbone = root / "backbone"
            torch.manual_seed(3)
            tiny_model().encoder.save_pretrained(backbone)
            backend = Tokenizer(WordPiece({"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "benign": 4, "attack": 5}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
            tokenizer.save_pretrained(backbone)
            splits = root / "splits"
            rows = [{"id": str(i), "prompt": "attack" if i % 2 else "benign", "label": i % 2, "source": "fixture", "pair_id": str(i // 2)} for i in range(8)]
            manifest = {"splits": {}}
            for name, group in [("train", rows), ("validation", rows[:4])]:
                path = splits / (name + ".jsonl")
                write_jsonl(path, group)
                manifest["splits"][name] = {"sha256": sha256(path)}
            write_json(splits / "manifest.json", manifest)
            config = {"backbone": str(backbone), "data_dir": str(splits), "run_dir": str(root / "uninterrupted"),
                "seed": 7, "epochs": 1, "batch_size": 2, "accumulation_steps": 2,
                "encoder_lr": 0.001, "head_lr": 0.001, "weight_decay": 0.01, "warmup_ratio": 0,
                "max_length": 16, "pooling": "cls", "dropout": 0.2, "frozen": False,
                "weighted_loss": False, "max_fpr": 0.5, "device": "cpu", "paired_batches": True, "ranking_weight": 0.5, "ranking_margin": 1.0}
            config_path = root / "config.json"
            write_json(config_path, config)
            train(config_path)
            expected, _, _ = InjectionClassifier.load(root / "uninterrupted/last")
            config["run_dir"] = str(root / "interrupted")
            write_json(config_path, config)
            (root / "interrupted").mkdir()
            (root / "interrupted/PAUSE").touch()
            train(config_path)
            self.assertEqual(read_json(root / "interrupted/status.json")["status"], "paused")
            self.assertFalse((root / "interrupted/PAUSE").exists())
            train(config_path, resume=True)
            actual, _, _ = InjectionClassifier.load(root / "interrupted/last")
            self.assertEqual(read_json(root / "interrupted/status.json")["updates"], 2)
            for a, b in zip(expected.parameters(), actual.parameters()):
                torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)

    def test_optimizer_scheduler_and_rng_state_resume(self):
        torch.manual_seed(42)
        model = tiny_model()
        config = {"encoder_lr": 1e-3, "head_lr": 1e-3, "weight_decay": 0.01}
        optimizer = optimizer_for(model, config)
        scheduler = schedule(optimizer, 10, 0.1)
        loss = model(torch.tensor([[2, 4, 3]]), torch.ones((1, 3), dtype=torch.long)).sum()
        loss.backward(); optimizer.step(); scheduler.step()
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "rng": rng_state(torch.device("cpu"))}
        with tempfile.TemporaryDirectory() as directory:
            torch.save(state, Path(directory)/"state.pt")
            restored = torch.load(Path(directory)/"state.pt", weights_only=True)
        new_optimizer = optimizer_for(model, config)
        new_scheduler = schedule(new_optimizer, 10, 0.1)
        new_optimizer.load_state_dict(restored["optimizer"])
        new_scheduler.load_state_dict(restored["scheduler"])
        self.assertEqual(new_scheduler.last_epoch, scheduler.last_epoch)
        self.assertEqual(new_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])
        restore_rng(restored["rng"], torch.device("cpu")); a = torch.rand(4)
        restore_rng(restored["rng"], torch.device("cpu")); b = torch.rand(4)
        self.assertTrue(torch.equal(a, b))


class ApplicationTests(unittest.TestCase):
    def test_blocked_or_uninspected_text_never_reaches_generator(self):
        class Client:
            def generate(self, *args): raise AssertionError("Generator must not be called")
        class Detector:
            def __init__(self, blocked, truncated): self.blocked, self.truncated = blocked, truncated
            def score(self, text): return {"blocked": self.blocked, "truncated": self.truncated}
        for blocked, truncated in [(True, False), (False, True)]:
            result = process_document(Detector(blocked, truncated), Client(), "summarize", "text")
            self.assertTrue(result["blocked"])
            self.assertIsNone(result["response"])

    def test_allowed_document_and_task_are_forwarded_unchanged(self):
        class Detector:
            def score(self, text): return {"blocked": False, "truncated": False}
        class Client:
            def generate(self, task, document): return task + ":" + document
        self.assertEqual(process_document(Detector(), Client(), "task", "document")["response"], "task:document")

    def test_controlled_cases_and_exact_scoring(self):
        cases = synthetic_cases()
        self.assertEqual(len(cases), 20)
        self.assertEqual(sum(c["label"] for c in cases), 10)
        self.assertNotEqual(canonical("I will not output HIJACK-900"), canonical("HIJACK-900"))


if __name__ == "__main__":
    unittest.main()

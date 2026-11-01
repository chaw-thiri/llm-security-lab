import unittest
import torch
from promptshield.consistency import ConsistencyBatchSampler, consistency_sum


class ConsistencyTests(unittest.TestCase):
    def test_sampler_preserves_cross_domain_quartets_and_replay(self):
        rows = [dict(context_group=str(g), domain=d, label=y) for g in range(3)
                for d in ("technical", "general") for y in (0, 1)]+[dict(label=i % 2) for i in range(8)]
        batches = list(ConsistencyBatchSampler(rows, 4, 42))
        self.assertEqual(sorted(i for b in batches for i in b), list(range(20)))
        for b in batches:
            ids = {rows[i].get("context_group") for i in b}
            self.assertEqual(len(ids), 1)
        self.assertEqual(batches, list(ConsistencyBatchSampler(rows, 4, 42)))
        self.assertNotEqual(batches, list(ConsistencyBatchSampler(rows, 4, 43)))

    def test_invalid_group_cannot_mix_roles_or_domains(self):
        rows = [dict(context_group="a", domain="same", label=i % 2) for i in range(4)]
        with self.assertRaises(ValueError):
            ConsistencyBatchSampler(rows, 4, 42)

    def test_gradient_equalizes_margins_without_moving_pair_mean(self):
        logits = torch.tensor([[0., -5.], [0., -1.], [0., 1.], [0., 5.]], requires_grad=True)
        y = torch.tensor([0, 0, 1, 1]); ids = torch.ones(4, dtype=torch.long)
        loss = consistency_sum(logits, y, ids)/4
        self.assertEqual(loss.item(), 16.)
        loss.backward()
        updated = logits.detach()-.01*logits.grad
        old, new = logits[:, 1]-logits[:, 0], updated[:, 1]-updated[:, 0]
        for label in (0, 1):
            mask = y == label
            self.assertLess(new[mask].diff().abs().item(), old[mask].diff().abs().item())
            torch.testing.assert_close(old[mask].mean(), new[mask].mean())

    def test_accumulation_equals_whole_batch_and_replay_is_zero(self):
        logits = torch.tensor([[0., 1.], [0., 2.], [0., 3.], [0., 4.]]*2, requires_grad=True)
        y = torch.tensor([0, 1, 0, 1]*2); ids = torch.tensor([1]*4+[2]*4)
        consistency_sum(logits, y, ids).div(8).backward(); expected = logits.grad.clone()
        logits.grad = None
        for start in (0, 4):
            consistency_sum(logits[start:start+4], y[start:start+4], ids[start:start+4]).div(8).backward()
        torch.testing.assert_close(logits.grad, expected)
        self.assertEqual(consistency_sum(logits, y, torch.zeros(8, dtype=torch.long)).item(), 0)

    def test_incomplete_quartet_is_rejected(self):
        with self.assertRaises(ValueError):
            consistency_sum(torch.zeros(3, 2), torch.tensor([0, 0, 1]), torch.ones(3, dtype=torch.long))

    def test_training_resume_preserves_weights_and_loss_components(self):
        import tempfile
        from pathlib import Path
        from tokenizers import Tokenizer
        from tokenizers.models import WordPiece
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import DistilBertConfig, DistilBertModel, PreTrainedTokenizerFast
        from promptshield.classifier import InjectionClassifier
        from promptshield.common import write_json, write_jsonl, sha256, read_json, read_jsonl
        from promptshield.train import train
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); backbone = root/"backbone"; splits = root/"splits"
            torch.manual_seed(3)
            encoder = DistilBertModel(DistilBertConfig(vocab_size=16, dim=16, hidden_dim=32,
                n_layers=1, n_heads=2, dropout=.1, attention_dropout=.1, max_position_embeddings=32))
            encoder.save_pretrained(backbone)
            backend = Tokenizer(WordPiece({"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"benign":4,"attack":5,"other":6}, unk_token="[UNK]"))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]",2),("[SEP]",3)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]")
            tokenizer.save_pretrained(backbone)
            rows = [dict(id=str(i), prompt=("attack" if i % 2 else "benign")+(" other" if i % 4 >= 2 else ""),
                         label=i % 2, context_group=str(i//4), domain="technical" if i % 4 < 2 else "general") for i in range(16)]
            manifest = {"splits":{}}
            for name, examples in (("train",rows),("validation",rows[:4])):
                path = splits/f"{name}.jsonl"; write_jsonl(path,examples)
                manifest["splits"][name] = {"sha256":sha256(path)}
            write_json(splits/"manifest.json",manifest)
            config = dict(backbone=str(backbone), data_dir=str(splits), run_dir=str(root/"whole"), seed=7,
                epochs=1, batch_size=4, accumulation_steps=2, encoder_lr=.001, head_lr=.001, weight_decay=.01,
                warmup_ratio=0, max_length=16, pooling="cls", dropout=.2, frozen=False, weighted_loss=False,
                max_fpr=.5, device="cpu", consistency_batches=True, consistency_weight=.05)
            path = root/"config.json"; write_json(path,config); train(path)
            expected,_,_ = InjectionClassifier.load(root/"whole/last")
            config["run_dir"] = str(root/"resumed"); write_json(path,config)
            (root/"resumed").mkdir(); (root/"resumed/PAUSE").touch(); train(path)
            self.assertEqual(read_json(root/"resumed/status.json")["status"],"paused")
            train(path,resume=True)
            actual,_,_ = InjectionClassifier.load(root/"resumed/last")
            for a,b in zip(expected.parameters(),actual.parameters()):
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            a,b = read_jsonl(root/"whole/history.jsonl")[-1],read_jsonl(root/"resumed/history.jsonl")[-1]
            for key in ("train_loss","train_ce_loss","train_consistency_loss"):
                self.assertAlmostEqual(a[key],b[key],places=6)


if __name__ == "__main__":
    unittest.main()

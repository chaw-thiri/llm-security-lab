"""DistilBERT encoder plus an explicit, inspectable classification head."""
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from promptshield.common import read_json, write_json


class InjectionClassifier(nn.Module):
    def __init__(self, encoder, pooling="cls", dropout=0.2, frozen=False):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be cls or mean")
        self.encoder = encoder
        self.pooling, self.dropout, self.frozen = pooling, dropout, frozen
        width = encoder.config.dim
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 2))
        if frozen:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.pooling == "cls":
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

    @classmethod
    def from_backbone(cls, path, **kwargs):
        encoder = DistilBertModel.from_pretrained(path, local_files_only=True, attn_implementation="sdpa")
        return cls(encoder, **kwargs)

    def save(self, path, tokenizer, metadata):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.config.save_pretrained(path / "encoder_config")
        tokenizer.save_pretrained(path)
        # Own checkpoints have independent tensors; no remote custom code or pickle needed for inference.
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        temporary = path / "model.safetensors.tmp"
        save_file(state, str(temporary))
        temporary.replace(path / "model.safetensors")
        write_json(path / "detector.json", dict(metadata, pooling=self.pooling, dropout=self.dropout,
                                               frozen=self.frozen, format_version=1))

    @classmethod
    def load(cls, path, device="cpu"):
        path = Path(path)
        metadata = read_json(path / "detector.json")
        config = DistilBertConfig.from_pretrained(path / "encoder_config", local_files_only=True)
        config._attn_implementation = "sdpa"
        model = cls(DistilBertModel(config), pooling=metadata["pooling"], dropout=metadata["dropout"],
                    frozen=metadata["frozen"])
        model.load_state_dict(load_file(str(path / "model.safetensors")), strict=True)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        return model, tokenizer, metadata

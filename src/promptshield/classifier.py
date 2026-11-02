"""DistilBERT encoder plus an explicit, inspectable classification head."""
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, DistilBertConfig, DistilBertModel

from promptshield.common import read_json, write_json


class ResidualAttentionPool(nn.Module):
    """Add a learned summary of non-padding states to the final first-token state."""

    def __init__(self, width):
        super().__init__()
        # No extra RNG draws: the paired runs have the same initial logits and dropout stream.
        self.query = nn.Parameter(torch.zeros(width))
        self.projection = nn.Parameter(torch.zeros(width, width))

    def forward(self, hidden, attention_mask):
        valid = attention_mask.bool()
        logits = hidden @ self.query
        weights = logits.masked_fill(~valid, torch.finfo(logits.dtype).min).softmax(dim=1)
        weights = weights.masked_fill(~valid, 0)
        summary = torch.bmm(weights.unsqueeze(1), hidden).squeeze(1)
        return hidden[:, 0] + nn.functional.linear(summary, self.projection)


class InjectionClassifier(nn.Module):
    def __init__(self, encoder, pooling="cls", dropout=0.2, frozen=False):
        super().__init__()
        if pooling not in ("cls", "mean", "cls_attention"):
            raise ValueError("pooling must be cls, mean or cls_attention")
        self.encoder = encoder
        self.pooling, self.dropout, self.frozen = pooling, dropout, frozen
        width = encoder.config.dim
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, 2))
        self.attention_pool = ResidualAttentionPool(width) if pooling == "cls_attention" else None
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
        elif self.attention_pool is not None:
            pooled = self.attention_pool(hidden, attention_mask)
        else:
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

    def initialize_from(self, source):
        """Copy shared weights strictly; a new attention branch starts at zero."""
        if self.pooling == source.pooling:
            self.load_state_dict(source.state_dict(), strict=True)
        elif self.pooling == "cls_attention" and source.pooling == "cls":
            self.encoder.load_state_dict(source.encoder.state_dict(), strict=True)
            self.head.load_state_dict(source.head.state_dict(), strict=True)
            with torch.no_grad():
                self.attention_pool.query.zero_()
                self.attention_pool.projection.zero_()
        else:
            raise ValueError(f"Unsupported continuation from {source.pooling} to {self.pooling}")

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

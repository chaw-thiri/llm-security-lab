"""Matched 2-context x 2-wording x 2-role batches and margin consistency."""
from collections import defaultdict
import math
import random
import torch
from promptshield.consistency import ConsistencyDataset


class WordingDataset(ConsistencyDataset):
    def __getitem__(self, index):
        return dict(super().__getitem__(index),
                    grid_context_ids=self.rows[index].get("context_index", -1),
                    grid_wording_ids=self.rows[index].get("wording_index", -1))


class WordingBatchSampler:
    def __init__(self, rows, batch_size, seed):
        if batch_size < 8 or batch_size % 8 or len(rows) % 8:
            raise ValueError("Wording batches and row counts must be multiples of eight")
        self.size, self.batch_size, self.seed = len(rows), batch_size, seed
        groups, self.singles = defaultdict(list), []
        for i, r in enumerate(rows):
            if r.get("context_group"):
                groups[r["context_group"]].append(i)
            else:
                self.singles.append(i)
        self.units = []
        expected = {(c, w, y) for c in (0, 1) for w in (0, 1) for y in (0, 1)}
        for indices in groups.values():
            members = [rows[i] for i in indices]
            if len(indices) != 8 or {(r["context_index"], r["wording_index"], r["label"]) for r in members} != expected:
                raise ValueError("Each wording group must contain the complete context/wording/role grid")
            domains = [{r["domain"] for r in members if r["context_index"] == c} for c in (0, 1)]
            if any(len(d) != 1 for d in domains) or domains[0] == domains[1]:
                raise ValueError("Each context must have one distinct domain")
            self.units.append(indices)
        if len(self.singles) % 8:
            raise ValueError("Replay must form complete eight-example units")

    def __len__(self):
        return math.ceil(self.size/self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed)
        singles = list(self.singles)
        rng.shuffle(singles)
        units = list(self.units)+[singles[i:i+8] for i in range(0, len(singles), 8)]
        rng.shuffle(units)
        for start in range(0, len(units), self.batch_size//8):
            yield [i for u in units[start:start+self.batch_size//8] for i in u]


def wording_sum(logits, labels, groups, contexts, wordings, axes):
    """Sample-summed mean edge penalty; context/control and treatment share scale.

    Four context edges (weight 2) or eight context+wording edges (weight 1)
    per eight examples. Opposite roles are never aligned; replay contributes zero.
    """
    if axes not in ("context", "context_wording"):
        raise ValueError("Unknown consistency axes")
    result = logits.sum()*0
    margins = logits[:, 1]-logits[:, 0]
    for group in groups.unique().tolist():
        if group == 0:
            continue
        mask = groups == group
        if int(mask.sum().item()) != 8:
            raise ValueError("Complete eight-example grids are required")
        cells = {}
        for c in (0, 1):
            for w in (0, 1):
                for y in (0, 1):
                    cell = margins[mask & (contexts == c) & (wordings == w) & (labels == y)]
                    if cell.numel() != 1:
                        raise ValueError("Missing or repeated grid cell")
                    cells[c,w,y] = cell[0]
        for y in (0, 1):
            for w in (0, 1):
                result = result+(2 if axes == "context" else 1)*(cells[0,w,y]-cells[1,w,y]).square()
            if axes == "context_wording":
                for c in (0, 1):
                    result = result+(cells[c,0,y]-cells[c,1,y]).square()
    return result

"""Cross-context consistency of native binary-classification logit margins."""
from collections import Counter, defaultdict
import math
import random

import torch
from promptshield.data import TextDataset


class ConsistencyDataset(TextDataset):
    def __init__(self, rows, tokenizer, max_length):
        super().__init__(rows, tokenizer, max_length)
        ids = {key: i+1 for i, key in enumerate(sorted({r["context_group"] for r in rows if r.get("context_group")}))}
        self.consistency_ids = [ids.get(r.get("context_group"), 0) for r in rows]

    def __getitem__(self, index):
        return dict(super().__getitem__(index), consistency_ids=self.consistency_ids[index])


class ConsistencyBatchSampler:
    """Keep two roles crossed with two source contexts in one four-example unit."""
    def __init__(self, rows, batch_size, seed):
        if batch_size < 4 or batch_size % 4 or len(rows) % 4:
            raise ValueError("Consistency batches and row counts must be multiples of four")
        self.batch_size, self.seed, self.size = batch_size, seed, len(rows)
        groups, self.singles = defaultdict(list), []
        for i, row in enumerate(rows):
            if row.get("context_group"):
                groups[row["context_group"]].append(i)
            else:
                self.singles.append(i)
        self.units = []
        for indices in groups.values():
            members = [rows[i] for i in indices]
            domains = {r["domain"] for r in members}
            if (len(members) != 4 or Counter(r["label"] for r in members) != {0: 2, 1: 2}
                    or len(domains) != 2 or any({r["domain"] for r in members if r["label"] == label} != domains for label in (0, 1))):
                raise ValueError("Each context group must cross two labels with two distinct domains")
            self.units.append(indices)
        if len(self.singles) % 4:
            raise ValueError("Replay rows must form complete four-example units")

    def __len__(self):
        return math.ceil(self.size/self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed)
        singles = list(self.singles)
        rng.shuffle(singles)
        units = list(self.units)+[singles[i:i+4] for i in range(0, len(singles), 4)]
        rng.shuffle(units)
        per_batch = self.batch_size//4
        for start in range(0, len(units), per_batch):
            yield [i for unit in units[start:start+per_batch] for i in unit]


def consistency_sum(logits, labels, group_ids):
    """Sample-summed squared margin differences for equal-role, different-context pairs.

    A quartet contributes 2*(delta_benign**2 + delta_attack**2). Dividing by
    four examples gives the mean of the two pair penalties. Replay contributes
    zero; classification CE anchors the labels and prevents constant-score collapse.
    """
    result = logits.sum()*0
    margins = logits[:, 1]-logits[:, 0]
    for group in group_ids.unique().tolist():
        if group == 0:
            continue
        mask = group_ids == group
        if int(mask.sum().item()) != 4:
            raise ValueError("A complete context quartet is required in each microbatch")
        for label in (0, 1):
            selected = margins[mask & (labels == label)]
            if len(selected) != 2:
                raise ValueError("Each role needs two contexts")
            result = result+2*(selected[0]-selected[1]).square()
    return result

"""Keep labeled counterfactual pairs together; add a log-odds ranking objective."""
import math
import random
from collections import defaultdict
import torch
from torch.nn import functional as F
from promptshield.data import TextDataset


class PairedDataset(TextDataset):
    def __init__(self, rows, tokenizer, max_length):
        super().__init__(rows, tokenizer, max_length)
        ids = {value: i + 1 for i, value in enumerate(sorted({r['pair_id'] for r in rows if r.get('pair_id')}))}
        self.pair_ids = [ids.get(r.get('pair_id'), 0) for r in rows]

    def __getitem__(self, index):
        return dict(super().__getitem__(index), pair_ids=self.pair_ids[index])


class PairBatchSampler:
    """Two-example units keep each pair intact and retain every original example once."""
    def __init__(self, rows, batch_size, seed):
        if batch_size < 2 or batch_size % 2 or len(rows) % 2:
            raise ValueError('Paired sampling requires an even batch size and row count.')
        self.batch_size, self.seed = batch_size, seed
        groups, singles = defaultdict(list), []
        for i, row in enumerate(rows):
            if row.get('pair_id'):
                groups[row['pair_id']].append(i)
            else:
                singles.append(i)
        self.units = []
        for indices in groups.values():
            if len(indices) != 2 or {rows[i]['label'] for i in indices} != {0, 1}:
                raise ValueError('Each pair must contain exactly one benign and one attack example.')
            self.units.append(indices)
        self.singles = singles
        if len(singles) % 2:
            raise ValueError('Unpaired examples must also have even count.')
        self.size = len(rows)

    def __len__(self):
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed)
        singles = list(self.singles)
        rng.shuffle(singles)
        units = [list(unit) for unit in self.units] + [singles[i:i+2] for i in range(0, len(singles), 2)]
        rng.shuffle(units)
        for offset in range(0, len(units), self.batch_size // 2):
            yield [i for unit in units[offset:offset+self.batch_size//2] for i in unit]


def ranking_sum(logits, labels, pair_ids, margin=1.0):
    """Return 2*sum hinge losses, normalized by total examples by the caller.

    Thus the objective is mean CE + weight*(paired-example fraction)*mean pair hinge.
    Using logit differences avoids saturated softmax probabilities.
    """
    result = logits.sum() * 0
    odds = logits[:, 1] - logits[:, 0]
    for pair_id in pair_ids.unique().tolist():
        if pair_id == 0:
            continue
        mask = pair_ids == pair_id
        attack, benign = mask & (labels == 1), mask & (labels == 0)
        if attack.sum().item() != 1 or benign.sum().item() != 1:
            raise ValueError('Ranking requires a complete benign/attack pair in each batch.')
        result = result + 2 * F.relu(margin - (odds[attack][0] - odds[benign][0]))
    return result

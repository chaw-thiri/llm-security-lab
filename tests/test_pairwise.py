import unittest
import torch
from promptshield.pairwise import PairBatchSampler, ranking_sum

class PairwiseTests(unittest.TestCase):
    def test_sampler_keeps_pairs_and_covers_every_example(self):
        rows=[{'label':i%2,'pair_id':str(i//2)} for i in range(6)]+[{'label':i%2} for i in range(4)]
        batches=list(PairBatchSampler(rows,4,42))
        self.assertEqual(sorted(i for batch in batches for i in batch),list(range(10)))
        for batch in batches:
            for i in batch:
                if 'pair_id' in rows[i]:
                    self.assertEqual(sum(rows[j].get('pair_id')==rows[i]['pair_id'] for j in batch),2)
        self.assertEqual(batches,list(PairBatchSampler(rows,4,42)))
        self.assertNotEqual(batches,list(PairBatchSampler(rows,4,43)))

    def test_invalid_pair_fails(self):
        with self.assertRaises(ValueError):
            PairBatchSampler([{'label':0,'pair_id':'x'},{'label':0,'pair_id':'x'}],2,0)

    def test_gradient_pushes_attack_above_benign(self):
        logits=torch.zeros(2,2,requires_grad=True)
        ranking_sum(logits,torch.tensor([0,1]),torch.tensor([1,1])).backward()
        updated=logits.detach()-.1*logits.grad
        odds=updated[:,1]-updated[:,0]
        self.assertGreater(odds[1],odds[0])

    def test_accumulated_pair_loss_matches_whole_batch(self):
        logits=torch.tensor([[0.,1.],[1.,0.],[0.,2.],[0.,1.]],requires_grad=True)
        labels=torch.tensor([0,1,0,1]);pairs=torch.tensor([1,1,2,2])
        ranking_sum(logits,labels,pairs).div(4).backward();expected=logits.grad.clone()
        logits.grad=None
        for start in (0,2):ranking_sum(logits[start:start+2],labels[start:start+2],pairs[start:start+2]).div(4).backward()
        torch.testing.assert_close(logits.grad,expected)

    def test_satisfied_margin_and_unpaired_rows_have_zero_penalty(self):
        logits=torch.tensor([[0.,0.],[0.,2.],[0.,9.]],requires_grad=True)
        loss=ranking_sum(logits,torch.tensor([0,1,0]),torch.tensor([1,1,0]))
        self.assertEqual(loss.item(),0)
        loss.backward();self.assertEqual(logits.grad.abs().sum().item(),0)

    def test_incomplete_batch_pair_fails(self):
        with self.assertRaises(ValueError):ranking_sum(torch.zeros(1,2),torch.tensor([1]),torch.tensor([1]))

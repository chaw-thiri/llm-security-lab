"""Calibration must honor each benign domain and preserve tied-score semantics."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from promptshield.scoring import select_grouped_threshold
from prepare_grounded_upgrade import pair


class GroupedCalibrationTests(unittest.TestCase):
    def test_small_domain_cannot_be_hidden_by_pooled_fpr(self):
        y=[1,1,0]+[0]*20
        p=[.9,.4,.8]+[.1]*20
        groups=["small","large","small"]+["large"]*20
        threshold=select_grouped_threshold(y,p,groups,.05)
        self.assertGreater(threshold,.8)

    def test_equal_scores_are_not_split(self):
        threshold=select_grouped_threshold([1,0],[.9,.9],["same","same"],0)
        self.assertGreater(threshold,1)

    def test_missing_benign_domain_and_invalid_probabilities_fail(self):
        with self.assertRaises(ValueError):
            select_grouped_threshold([1,0],[.9,.1],["attack-only","benign"],.05)
        with self.assertRaises(ValueError):
            select_grouped_threshold([1,0],[float("nan"),.1],["same","same"],.05)

    def test_matched_pair_retains_real_context_under_both_labels(self):
        context=dict(document="Original source paragraph.",source_id="source",cohort="security")
        rows=pair(context,7,"train")
        self.assertEqual({r["label"] for r in rows},{0,1})
        self.assertEqual(rows[0]["pair_id"],rows[1]["pair_id"])
        self.assertTrue(all(context["document"] in r["prompt"] for r in rows))
        self.assertTrue(all(r["human_reviewed"] is False for r in rows))

    def test_quote_variation_is_not_fixed_by_domain(self):
        for offset,domain in enumerate(("technical","security","general")):
            rows=[pair(dict(document="Source material.",source_id="s",cohort=domain),i,"train")[1]
                  for i in range(offset,300,3)]
            self.assertEqual({r["quoted"] for r in rows},{False,True})


if __name__=="__main__":
    unittest.main()

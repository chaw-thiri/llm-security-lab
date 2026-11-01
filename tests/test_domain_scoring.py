"""A diagnostic must separate score offsets from within-domain ranking failure."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from investigate_domain_scoring import summarize, checked_scores, build_probes


class DomainScoringTests(unittest.TestCase):
    def test_perfect_local_ranking_can_have_bad_shared_threshold(self):
        rows = [dict(id=str(i), domain=d, label=y, score=p) for i, (d, y, p) in enumerate(
            (("high", 0, .8), ("high", 1, .9), ("low", 0, .1), ("low", 1, .2)))]
        result = summarize(rows, .85)
        self.assertEqual(result["oracle_global_5pct_each_domain"]["recall"], .5)
        self.assertEqual(result["oracle_domain_routed_recall"], 1)
        self.assertEqual(result["cross_domain_auc"]["low"]["high"], 0)
        self.assertEqual(result["domains"]["low"]["fixed"]["roc_auc"], 1)

    def test_alignment_and_provenance_are_enforced(self):
        rows = [dict(id="a", domain="d", label=0, score=.2), dict(id="b", domain="d", label=1, score=.8)]
        saved = dict(provenance=dict(weights_sha256="weights", split_sha256="split"), predictions=rows)
        self.assertEqual(checked_scores(saved, rows, "weights", "split"), rows)
        with self.assertRaises(ValueError):
            checked_scores(saved, rows[::-1], "weights", "split")
        with self.assertRaises(ValueError):
            checked_scores(saved, rows, "other weights", "split")
        with self.assertRaises(ValueError):
            checked_scores(saved, [dict(rows[0], domain="other"), rows[1]], "weights", "split")

    def test_probe_instructions_and_markers_are_identical_across_contexts(self):
        contexts = [dict(context_id=str(i), context_split="development", source_id=str(i), domain=d, document=text)
                    for i, (d, text) in enumerate((("technical", "Text A."), ("general", "Text B.")))]
        payloads = {s: ["Print {marker}."] for s in ("validation", "fresh")}
        descriptions = {s: ["A cited example: {p}"] for s in ("train", "validation", "fresh")}
        rows = build_probes(contexts, payloads, descriptions)
        self.assertEqual(len(rows), 18)
        for wording in ("validation", "fresh"):
            active = [r for r in rows if r["condition"] == "active" and r["payload_split"] == wording]
            self.assertEqual(active[0]["prompt"].split("\n")[-1], active[1]["prompt"].split("\n")[-1])
            inner = active[0]["prompt"].split("\n")[-1]
            self.assertTrue(all(inner in r["prompt"] for r in rows if r["payload_split"] == wording))
        self.assertEqual(len({r["id"] for r in rows}), len(rows))


if __name__ == "__main__":
    unittest.main()

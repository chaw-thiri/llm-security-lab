"""Attainable score frontiers must respect ties and per-source false alarms."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_detectors import frontier, best_recall, least_fpr


class FrontierTests(unittest.TestCase):
    def row(self, key, label, score, source="a"):
        return dict(id=key, label=label, score=score, source=source)

    def test_tied_scores_cannot_be_split(self):
        points = frontier([self.row("attack", 1, .9), self.row("benign", 0, .9)])
        self.assertEqual(len(points), 2)
        self.assertEqual(best_recall(points, 0)["recall"], 0)
        self.assertEqual(least_fpr(points, 1)["worst_source_fpr"], 1)

    def test_per_source_budget_does_not_hide_small_source(self):
        rows = [self.row("p1", 1, .9), self.row("p2", 1, .4),
                self.row("n1", 0, .8), self.row("n2", 0, .3, "b"), self.row("n3", 0, .2, "b")]
        points = frontier(rows)
        self.assertEqual(best_recall(points, .5)["recall"], .5)
        self.assertEqual(best_recall(points, .5, "fpr")["recall"], 1)
        self.assertEqual(least_fpr(points, 1)["source_fpr"], {"a": 1, "b": 0})

    def test_points_match_greater_equal_prediction_rule(self):
        rows = [self.row("p", 1, .7), self.row("n", 0, .4)]
        for point in frontier(rows):
            self.assertEqual(point["tp"], sum(r["label"] == 1 and r["score"] >= point["threshold"] for r in rows))
            self.assertEqual(point["fp"], sum(r["label"] == 0 and r["score"] >= point["threshold"] for r in rows))

    def test_invalid_scores_and_duplicate_ids_rejected(self):
        for rows in ([self.row("p", 1, float("nan")), self.row("n", 0, .5)],
                     [self.row("same", 1, .9), self.row("same", 0, .5)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                frontier(rows)


if __name__ == "__main__":
    unittest.main()

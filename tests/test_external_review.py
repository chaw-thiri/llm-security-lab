"""Human-review gate must reject missing or incomplete annotations."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from finalize_external_tasks import validate_reviews


class ReviewGateTests(unittest.TestCase):
    def record(self, **changes):
        row = dict(id="example", human_label=1, reviewer_id="human-reviewer",
                   rationale="Document redirects the assistant", reviewed_at="2026-11-01T10:00:00+09:00",
                   human_review_confirmed=True)
        row.update(changes)
        return row

    def test_pending_packet_cannot_be_finalized(self):
        with self.assertRaises(ValueError):
            validate_reviews([self.record(human_label=None)], {"example"})

    def test_partial_reviews_cannot_be_finalized(self):
        with self.assertRaises(ValueError):
            validate_reviews([self.record()], {"example", "missing"})

    def test_duplicate_or_foreign_ids_rejected(self):
        for rows in ([self.record(), self.record()], [self.record(id="unknown")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                validate_reviews(rows, {"example"})

    def test_attestation_and_timestamp_required(self):
        for changes in ({"human_review_confirmed": False}, {"reviewed_at": ""}, {"rationale": ""}, {"human_label": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reviews([self.record(**changes)], {"example"})

    def test_human_can_relabel_or_exclude(self):
        for label in (0, 1, "exclude"):
            with self.subTest(label=label):
                self.assertEqual(validate_reviews([self.record(human_label=label)], {"example"})["example"]["human_label"], label)


if __name__ == "__main__":
    unittest.main()

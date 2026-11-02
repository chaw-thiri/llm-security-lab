# External task/document corpus — preparation report

**Assessment:** 10,359 candidates are assembled; zero have completed independent human review.

## Design

The adapter used official Microsoft BIPIA data pinned to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`, retaining email QA, table QA and code-debugging contexts. Related contexts stayed together; cross-split/exposed contexts were quarantined, challenge families reserved and overlap screened. Upstream payload review does not constitute review of locally assembled task/document pairs.

## Results

| Partition | Candidate rows |
|---|---|
| train | 7353 |
| validation | 1548 |
| challenge | 1458 |

## Decision and limitations

Keep export blocked until complete human annotations pass validation. The 60-case training/development starter is a reviewer calibration packet; its challenge remains unopened.

The mixture is table-heavy and attack-heavy; repeated variants are not independent documents. Longer inputs remain complete and flagged. Review attestations cannot establish that a person actually performed the review.

**Records:** [Measured record](../records/external-tasks-v2-manifest.json). [Source provenance](../records/bipia-source.json) · [review procedure](../../contribution.md#data-review). Detailed rationale and chronology: [research diary](../../research_diary.md).

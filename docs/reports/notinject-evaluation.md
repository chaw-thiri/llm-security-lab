# NotInject — external benign-probe report

**Assessment:** Contextual training reduced false alarms modestly; approximately one third of benign probes remained blocked.

## Design

The evaluation used pinned NotInject revision `9fdfe4368fc43c592e601f8fc6217de7c777f9e7`. Excluding 48 exact normalized overlaps with upstream PIGuard validation from 339 published probes left 291. Existing validation-selected thresholds and the common numerical baseline threshold were compared; none was fitted on NotInject.

## Results

| Model | FPR at own threshold | FPR at baseline threshold |
|---|---|---|
| Baseline | 40.9% | 40.9% |
| Contextual CE (3-seed mean) | 34.5% | 31.3% |
| Contextual ranking (3-seed mean) | 35.5% | 31.0% |

## Decision and limitations

Retain this set as an inspected false-alarm regression benchmark; do not tune on its outcomes.

All cases are benign trigger-word probes. Attack recall, complete-document utility and production prevalence are unmeasured; exact-match screening does not exclude semantic overlap.

**Records:** [Measured record](../records/notinject-evaluation.json). Detailed rationale and chronology: [research diary](../../research_diary.md).

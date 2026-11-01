# Final wording-consistency attempt — supervisor report

**Assessment:** The calibration-selected recipe met the fixed overall target in every seed.

## Design

Two matched full first-token DistilBERT recipes continued the predeclared prior consistency seed-42 weights (66,955,010 trainable parameters). Both used 4,096 balanced rows, including 1,536 original replay rows and 320 eight-example grids, two epochs, batch eight, accumulation two and 512 updates at 5e-6 learning rates. CE plus 0.05-weighted mean context-edge margin consistency was compared with equally weighted context-and-wording consistency. Seeds 42/43/44 ran unconditionally.

Checkpoint AP used 1,024 rows; disjoint 2,048-row calibration selected shared thresholds at <=10% benign FPR per development domain. Three-seed calibration recall selected the recipe before final scoring. The frozen final challenge contains 1,764 constructions (900 benign, 864 attacks) on 18 new official PostgreSQL, MITRE CWE and EIA pages, with held-out outer styles and payload wording. A pass requires **over 80% overall recall and <=10% overall benign FPR in every seed**.

## Results

| Recipe | Recall | Overall FPR | Worst-domain FPR | ROC-AUC | PIGuard F1 |
|---|---|---|---|---|---|
| Initializer (own threshold) | 50.9% | 17.8% | 30.3% | 0.777 | 96.2% |
| context-42 | 97.8% | 9.4% | 14.7% | 0.993 | 96.4% |
| wording-42 | 97.5% | 6.2% | 11.3% | 0.995 | 96.6% |
| context-43 | 96.3% | 6.6% | 11.7% | 0.992 | 96.4% |
| wording-43 | 95.4% | 4.7% | 10.0% | 0.992 | 96.6% |
| context-44 | 97.7% | 10.1% | 13.7% | 0.990 | 96.4% |
| wording-44 | 97.9% | 8.3% | 12.3% | 0.993 | 96.4% |

Calibration selected **wording**. Selected-family final domain means: technical: 96.5% recall / 3.2% FPR; security: 94.2% recall / 4.8% FPR; general: 100.0% recall / 11.2% FPR. On the inspected previous challenge: **95.4% recall / 19.7% FPR**.

## Decision and limitations

Freeze this empirical milestone. Older-challenge and general-domain false alarms still exceed 10%; independently review those errors and test task-conditioned verification with unprotected/detector-only application controls before serving promotion. Preserve baseline weights and serving defaults.

Construction labels received an assistant audit, not independent human review. Variants share 18 pages, overlapping passages and templates; seeds share initialization. All active wrappers explicitly request current execution. These short answer-hijack cases do not establish adaptive-attack or full-document safety. Only the within-study matched comparison isolates the wording loss; the initializer retains its own original threshold and older studies used different data. Preflight v1 was rejected before training; corrected v2 was frozen without outcome-based filtering.

**Records:** [Protocol](wording-final-protocol.json) · [Data](wording-final-data.json) · [Results](wording-final-results.json) · [Cross-seed/error audit](wording-final-summary.json) · [Semantic audit](wording-final-semantic-audit.json) · [Execution archive](provenance/wording-final-execution.json). Local joined errors: `results/wording-final-v2/per-example-errors.jsonl`. Detailed rationale and chronology: [research diary](../research_diary.md).

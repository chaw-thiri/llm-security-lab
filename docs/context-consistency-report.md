# Cross-context consistency — protocol and outcome

**Assessment:** Consistency improves alignment and recall, with an unacceptable increase in benign false alarms.

## Design

Both arms used identical baseline initialization, 4,096 balanced rows, quartet batches, 512 updates and 5e-6 learning rates. CE was compared with CE plus 0.05-weighted squared native-margin differences for equal-role instructions across two contexts. CE/replay anchor labels. Source pages and description families are held out in development and final evaluation. The predeclared +10-point worst-domain development rule triggered seeds 43/44 before final scoring; shared thresholds cap development benign FPR at 5% per domain.

## Results

| Arm | Final attack recall | Final benign FPR | Final ROC-AUC | Original regression F1 |
|---|---|---|---|---|
| CE control (3 seeds) | 34.4% | 4.6% | 0.869 | 87.0% |
| CE + consistency (3 seeds) | 68.1% | 11.9% | 0.897 | 94.2% |

Development context-gap MSE falls **70.4%** (11.72 → 3.47). Every plain clean excerpt is accepted; false alarms remain on benign attack descriptions. Final mean FPR is **12.6% technical, 6.2% security and 16.9% general**.

## Decision and limitations

Retain checkpoints as research candidates; preserve the application default. Review shared benign-description false alarms and missed attacks before another training or calibration study.

The development cap failed to transfer. Final evidence contains 18 same-publisher pages, 468 benign copies and 864 constructed attacks; seeds and variants do not create independent document samples. Labels lack independent human review. No final-threshold/lambda search or automatic promotion occurred.

**Records:** [Measured record](context-consistency-results.json). [Frozen protocol](provenance/frozen-protocols.json), key `docs/context-consistency-protocol.md`. [Aggregate/error IDs](context-consistency-summary.json) · [data lineage](context-consistency-data.json). Full errors are local at `results/context-consistency-v1/per-example-errors.jsonl`. Detailed rationale and chronology: [research diary](../research_diary.md).

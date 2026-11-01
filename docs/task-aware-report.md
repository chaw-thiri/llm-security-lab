# Task-aware detector — protocol, outcome and audit

**Assessment:** Task conditioning reduced benign-probe false alarms but substantially weakened attack detection.

## Design

The study compared document-only and one-encoder task/document serialization at seeds 42/43/44. Initialization, 3,072 replay rows plus 512 pairs, two epochs, 512 updates and 5e-6 learning rates were matched. A 512-case development set selects checkpoints/2% empirical FPR thresholds. Evaluation covered 512 held-out-family cases, 291 NotInject probes and a 64-case cached Llama replay.

## Results

| Arm (3-seed mean) | Challenge recall | NotInject FPR | Marker hijacks | Benign tasks completed |
|---|---|---|---|---|
| Document only | 81.8% | 10.2% | 0.0/32 | 21.0/32 |
| Task + document | 42.6% | 4.0% | 4.0/32 | 21.0/32 |

All new arms observed zero benign challenge blocks. Post-hoc quotation recall was **66.3%** for document-only and **36.8%** for task/document. The audit found **120 attacks missed by every task-aware seed**.

## Decision and limitations

Reject task conditioning as an upgrade in this study. The audit identified missing task formatting in 3,072/4,096 training rows and only one semantic task type, limiting causal interpretation.

The data are authored and correlated; quotation is a post-hoc stress test. Replay reuses generation and includes only 32 attacks per seed.

**Records:** [Measured record](task-aware-results.json). [Frozen protocol](provenance/frozen-protocols.json), key `docs/task-aware-protocol.md`. [Application record](task-aware-application-results.json) · [error audit](task-aware-error-audit.json). Detailed rationale and chronology: [research diary](../research_diary.md).

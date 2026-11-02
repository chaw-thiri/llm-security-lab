# Contextual training v2 — protocol and outcome

**Assessment:** Paired training improved the authored challenge; ranking loss added little average benefit.

## Design

The study compared original-data control, paired CE and paired CE-plus-ranking across seeds 42/43/44. Initialization, 4,096 rows, two epochs, 512 updates and 5e-6 learning rates were matched. Treatments use 512 contextual pairs plus 3,072 replay rows. Training/development/challenge wording families differ; original validation selects checkpoints and thresholds. One actual Llama response per application case was cached and replayed for identical forwarded inputs.

## Results

| Method (3-seed mean) | Original F1 | Attacks caught | Benign blocked | Benign tasks completed |
|---|---|---|---|---|
| Control | 95.7% | 62.7/64 | 55.7/64 | 7.0/64 |
| Cross-entropy | 95.9% | 64.0/64 | 20.7/64 | 39.3/64 |
| CE + ranking | 96.0% | 64.0/64 | 19.7/64 | 40.3/64 |

No gated replay produced an exact marker hijack; unprotected Llama produced **1/64**. Baseline benign completion was **5/64**.

## Decision and limitations

Retain the baseline; use CE as the simpler contextual reference for subsequent investigation.

The 128 challenge cases share an author and construction style; seeds share baseline weights. This is not independent generation per gate or evidence against adaptive attacks.

**Records:** [Measured record](../records/context-v2-results.json). [Frozen protocol](../provenance/frozen-protocols.json), key `docs/context-v2-protocol.md`. [Data review](../records/context-v2-data-review.json). Detailed rationale and chronology: [research diary](../../research_diary.md).

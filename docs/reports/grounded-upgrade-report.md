# Grounded continuation — protocol and outcome

**Assessment:** Ranking improved, but shared development thresholds failed to transfer.

## Design

Balanced control and grounded arms continued baseline weights with 4,096 rows, two epochs/512 updates and learning rates 5e-6. Grounded uses 1,536 replay rows plus 1,280 contextual pairs. The 30 inspected pages were partitioned by cohort into 6/2/2 train/development/regression pages; 1,024 development cases calibrated a shared threshold at ≤5% benign FPR per domain. The +10-point development recall rule qualified both arms for seeds 43/44 before final scoring.

## Results

| Model (continuations: 3-seed mean) | Final attack recall | Final benign FPR | Final ROC-AUC |
|---|---|---|---|
| Baseline, saved threshold | 98.1% | 73.6% | 0.725 |
| Baseline, recalibrated | 7.6% | 2.8% | 0.725 |
| Original-data control | 8.2% | 2.3% | 0.729 |
| Grounded continuation | 13.6% | 0.5% | 0.916 |

## Decision and limitations

Do not promote. Investigate score offsets and within-domain separation rather than extending training blindly.

Final evidence has 18 new same-publisher pages, 72 benign cases and 432 authored attacks. Labels are construction-based. Older public-document application replay is exposed regression, not fresh utility evidence. The final pages later became development material for the consistency study.

**Records:** [Measured record](../records/grounded-upgrade-results.json). [Frozen protocol](../provenance/frozen-protocols.json), key `docs/grounded-upgrade-protocol.md`. [Data/source lineage](../records/grounded-upgrade-data.json). Detailed rationale and chronology: [research diary](../../research_diary.md).

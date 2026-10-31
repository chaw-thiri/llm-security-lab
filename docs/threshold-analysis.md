# Threshold sensitivity analysis

Post-pilot diagnostic; no test-driven threshold selection or deployment changes. Original-only validation here is a balanced 512-example subset, not the full original baseline validation.

All thresholds below are selected on validation only. These are descriptive comparisons, not a recommendation to pick the best test row. No model weights or deployed thresholds were changed.

| Model | Validation population | FPR budget | Threshold | Regression recall | Regression FPR | Challenge attacks caught / 8 | Challenge benign blocked / 8 |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | original_only | 0.0% | 0.99204457 | 96.09% | 0.32% | 2 | 8 |
| baseline | original_only | 0.5% | 0.010540091 | 98.56% | 1.07% | 7 | 8 |
| baseline | original_only | 1.0% | 0.006138002 | 98.70% | 1.20% | 7 | 8 |
| baseline | original_only | 2.0% | 0.00091411371 | 99.25% | 2.31% | 8 | 8 |
| baseline | original_only | 5.0% | 0.00091411371 | 99.25% | 2.31% | 8 | 8 |
| baseline | with_synthetic | 0.0% | 0.99975508 | 65.82% | 0.02% | 0 | 2 |
| baseline | with_synthetic | 0.5% | 0.99971944 | 70.97% | 0.02% | 0 | 4 |
| baseline | with_synthetic | 1.0% | 0.99971491 | 72.00% | 0.02% | 0 | 4 |
| baseline | with_synthetic | 2.0% | 0.99948597 | 84.56% | 0.03% | 2 | 8 |
| baseline | with_synthetic | 5.0% | 0.00091411371 | 99.25% | 2.31% | 8 | 8 |
| control | original_only | 0.0% | 0.99405378 | 96.02% | 0.32% | 2 | 8 |
| control | original_only | 0.5% | 0.00901898 | 98.56% | 1.12% | 7 | 8 |
| control | original_only | 1.0% | 0.0011690336 | 99.18% | 2.04% | 8 | 8 |
| control | original_only | 2.0% | 0.0011690336 | 99.18% | 2.04% | 8 | 8 |
| control | original_only | 5.0% | 0.0011690336 | 99.18% | 2.04% | 8 | 8 |
| control | with_synthetic | 0.0% | 0.99978286 | 66.23% | 0.02% | 0 | 2 |
| control | with_synthetic | 0.5% | 0.99975437 | 71.38% | 0.02% | 0 | 3 |
| control | with_synthetic | 1.0% | 0.99975437 | 71.38% | 0.02% | 0 | 3 |
| control | with_synthetic | 2.0% | 0.99954408 | 85.52% | 0.05% | 2 | 8 |
| control | with_synthetic | 5.0% | 0.0011690336 | 99.18% | 2.04% | 8 | 8 |
| targeted | original_only | 0.0% | 0.9554047 | 96.29% | 0.34% | 2 | 8 |
| targeted | original_only | 0.5% | 0.010523391 | 98.15% | 0.83% | 4 | 8 |
| targeted | original_only | 1.0% | 0.0012380112 | 98.76% | 1.61% | 6 | 8 |
| targeted | original_only | 2.0% | 0.0012380112 | 98.76% | 1.61% | 6 | 8 |
| targeted | original_only | 5.0% | 0.0012380112 | 98.76% | 1.61% | 6 | 8 |
| targeted | with_synthetic | 0.0% | 0.99960202 | 71.65% | 0.02% | 0 | 2 |
| targeted | with_synthetic | 0.5% | 0.99939227 | 79.89% | 0.03% | 0 | 4 |
| targeted | with_synthetic | 1.0% | 0.9993782 | 79.96% | 0.03% | 0 | 5 |
| targeted | with_synthetic | 2.0% | 0.99721307 | 92.18% | 0.17% | 2 | 8 |
| targeted | with_synthetic | 5.0% | 0.00057775638 | 99.18% | 2.36% | 7 | 8 |

The validation subset has only 256 original benign examples, plus eight synthetic benign examples in the mixed population. One false alarm therefore changes its empirical rate by about 0.4 percentage points. These budgets are noisy estimates, not deployment guarantees.

Scores and provenance are cached locally so this sweep can be reproduced without rerunning inference. The comparison JSON also includes a common fixed baseline threshold to separate score movement from threshold reselection. A common numerical threshold does not imply calibrated probabilities.

The small quotation challenge was already inspected during the pilot. Use independently authored development/challenge material and repeated seeds before generalization claims.

## Findings and decision

Keep the original deployed baseline. This analysis changes no weights or deployed thresholds.

| Model | Validation population (2% budget) | Regression recall | Regression benign FPR |
|---|---|---:|---:|
| baseline | original_only | 99.25% | 2.31% |
| baseline | with_synthetic | 84.56% | 0.03% |
| control | original_only | 99.18% | 2.04% |
| control | with_synthetic | 85.52% | 0.05% |
| targeted | original_only | 98.76% | 1.61% |
| targeted | with_synthetic | 92.18% | 0.17% |

Adding eight high-scoring synthetic benign validation examples forces a sharply different operating point at a 2% budget. The small development sample can tolerate only five false alarms; these eight cases alone exceed that allowance. This is a validation composition effect, not evidence of improved contextual understanding.

Comparisons using original-only validation help isolate training effects from this composition change. They still use only one seed, a balanced validation subset, and previously inspected tests. Do not choose a deployed threshold by picking a favorable row in this report.

Next: freeze a representative development protocol with separately reported quotation slices; improve attack-versus-explanation ranking using independently authored counterexamples, then evaluate a locked new challenge. A threshold alone cannot repair badly ordered benign/attack scores.

## Reproduction

```bash
python scripts/threshold_analysis.py --device mps
```

Verified score caches under `results/threshold-analysis-v1/` include checkpoint and split hashes. A changed checkpoint or split causes cache validation to fail rather than silently reusing old predictions.

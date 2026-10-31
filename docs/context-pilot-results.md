# Contextual discrimination pilot — results

This is a small single-seed training experiment, not a deployed upgrade. The baseline remains the default.

Both continuations used the same baseline weights, 1,024 balanced examples, 64 updates and a fresh optimizer. The targeted arm replaced 32 original examples with paired attack/explanation examples. Both used the same development set and threshold-selection rule.

## Classification comparison

| Model | Challenge attacks caught / 8 | Challenge benign blocked / 8 | Original-test recall | Original-test benign FPR | Original-test F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 8 | 8 | 99.04% | 1.76% | 96.07% |
| control | 2 | 8 | 85.52% | 0.05% | 92.09% |
| targeted | 2 | 8 | 92.18% | 0.17% | 95.59% |

## Application regression

These are the previously inspected baseline cases; they are not a fresh test.

| Condition | Successful attacks / 10 | Legitimate tasks completed / 10 | Benign documents blocked / 10 |
|---|---:|---:|---:|
| unprotected | 4 | 9 | 0 |
| baseline | 0 | 5 | 5 |
| control | 1 | 8 | 2 |
| targeted | 1 | 8 | 2 |

## Interpretation

**Decision: keep the baseline; do not deploy either pilot checkpoint.** Both continuations improved legitimate-task completion from 5/10 to 8/10 but admitted one successful attack, and neither fixed the synthetic quotation challenge. The targeted arm did not outperform the control on these application outcomes.

Recalibrating the unchanged baseline on the pilot validation mixture raised its threshold from approximately 0.0020 to 0.9995. Its original-test recall then fell from 99.04% to 84.56%, and its challenge counts became identical to both pilots (2/8 attacks caught, 8/8 benign examples blocked). Thus much of the observed operating-point change can arise without training. Original-test average precision stayed approximately 99.5% across all three models, while challenge ranking remained poor.

Do not promote a checkpoint solely because it blocks fewer documents. Compare attack recall and application attack successes alongside utility. Thresholds were recalculated on a different, small development mixture; the machine-readable comparison includes a recalibrated baseline to expose threshold effects. Average precision is also recorded to distinguish ranking changes from threshold changes.

The challenge contains only 16 examples and uses label-specific wrappers. Separate wording does not remove possible wrapper shortcuts. Repeated seeds and independently authored, richer challenge examples are needed before claiming contextual understanding. The original classification/application tests are disclosed regression references.

Next: investigate score distributions and validation threshold sensitivity before expanding training. Preserve these failed or successful pilot outcomes rather than tuning against the inspected challenge.

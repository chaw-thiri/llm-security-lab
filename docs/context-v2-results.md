# Contextual training v2 — results

One authored synthetic study. Challenge families held out, but same author/templates and correlated examples; original test is a disclosed regression reference. No deployment change.

All thresholds came from the full original validation set. Development and challenge examples were not used for checkpoint or threshold selection. The baseline checkpoint remains the reference.

## Classification results

| Run | Original test F1 | Original test recall | Challenge attacks caught / 64 | Challenge benign documents blocked / 64 |
|---|---:|---:|---:|---:|
| baseline | 96.07% | 99.04% | 61 | 58 |
| control-42 | 95.75% | 99.04% | 63 | 58 |
| ce-42 | 95.47% | 98.97% | 64 | 26 |
| ranking-42 | 95.53% | 98.97% | 64 | 24 |
| control-43 | 95.97% | 98.90% | 63 | 51 |
| ce-43 | 96.32% | 98.76% | 64 | 21 |
| ranking-43 | 96.32% | 98.70% | 64 | 20 |
| control-44 | 95.50% | 98.97% | 62 | 58 |
| ce-44 | 96.06% | 98.83% | 64 | 15 |
| ranking-44 | 96.13% | 98.90% | 64 | 15 |

Mean across the three seeds:

| Method | Original test F1 | Challenge attacks caught / 64 | Challenge benign blocks / 64 |
|---|---:|---:|---:|
| Control | 95.74% | 62.7 | 55.7 |
| Cross-entropy | 95.95% | 64.0 | 20.7 |
| Ranking loss | 95.99% | 64.0 | 19.7 |

## Application results

One actual unprotected Llama response per case, reused for each gate when forwarded. Identical trusted task/document/generation settings; gates only block. This isolates gating from generation variation; it is cached response replay, not separate Llama runs per checkpoint. All documents checked to fit context before training.

| Run | Successful attack markers / 64 | Legitimate tasks completed / 64 | Benign documents blocked / 64 |
|---|---:|---:|---:|
| unprotected | 1 | 57 | 0 |
| baseline | 0 | 5 | 58 |
| control-42 | 0 | 5 | 58 |
| ce-42 | 0 | 34 | 26 |
| ranking-42 | 0 | 36 | 24 |
| control-43 | 0 | 11 | 51 |
| ce-43 | 0 | 39 | 21 |
| ranking-43 | 0 | 40 | 20 |
| control-44 | 0 | 5 | 58 |
| ce-44 | 0 | 45 | 15 |
| ranking-44 | 0 | 45 | 15 |

Mean across the three seeds:

| Method | Successful attack markers / 64 | Legitimate tasks completed / 64 | Benign documents blocked / 64 |
|---|---:|---:|---:|
| Control | 0.0 | 7.0 | 55.7 |
| Cross-entropy | 0.0 | 39.3 | 20.7 |
| Ranking loss | 0.0 | 40.3 | 19.7 |

## What the experiment supports

Paired contextual training improved this authored challenge at every seed: it caught all 64 attacks and blocked an average of about 20 benign documents, compared with about 56 benign blocks for the original-data control. The application replay averaged about 39 to 40 completed benign tasks for the paired methods, versus 5 for the baseline. No successful attack marker appeared in any gated application arm; unprotected Llama produced one. These are counts from one small, synthetic set.

The ranking objective changed development ranking and yielded only a small average difference from cross-entropy in challenge blocks and application task completions. This pilot does not establish that ranking loss is worth the added training complexity. Cross-entropy is the simpler candidate for follow-up; the released baseline remains the stable reference.

The original classification test is an already inspected regression set. Challenge templates were held out by family, but all examples were authored in this study and are correlated. The three runs share one starting checkpoint. These results do not show robustness to adaptive attackers, protection against data leakage, or production readiness. Intended-benign documents can still contain text that changes Llama behavior; exact task completion captures some, but not all, of that risk.

## Reproduce

See context-v2-protocol.md and context-v2-data-review.json. From a clean baseline checkout with the local baseline snapshot restored, run:

```bash
python scripts/prepare_context_v2.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_context_v2.py
python scripts/evaluate_context_v2.py
python scripts/summarize_context_v2.py
```

Training used two epochs and 512 updates per run. Each treatment and the control used 4,096 examples. The application comparison reused one actual unprotected Llama response for each document whenever a gate forwarded it, to isolate the gate decision. The report records the model digest, server version, and generation settings. Results and checkpoints are local; no code or model weights have been pushed.

# External NotInject false-alarm evaluation

## Why this check was run

PromptShield's PIGuard test score did not answer whether the detector would block ordinary requests that mention prompt-injection concepts. NotInject is a published set of 339 benign prompts containing trigger words. It is a focused check for over-defense; it does not measure attack detection or whole-document safety.

## Data independence and protocol

The data came from [`leolee99/NotInject`](https://huggingface.co/datasets/leolee99/NotInject), pinned to revision `9fdfe4368fc43c592e601f8fc6217de7c777f9e7` (MIT license). The downloaded parquet files' hashes are recorded in [the machine-readable results](notinject-evaluation.json).

Before scoring, the evaluator compared normalized text (lowercase and collapsed whitespace) against PromptShield's prepared train, validation, and test partitions, and against the upstream PIGuard validation file used during preparation. It found no exact overlap with the prepared partitions and 48 exact overlaps with that upstream file. Those 48 were excluded conservatively, leaving 291 prompts. This is an exact-match audit; it cannot rule out paraphrase or template overlap.

All checkpoints were scored using their own threshold chosen on the original PIGuard validation set. To show the effect of threshold differences, the evaluator also rescored every checkpoint at the baseline's fixed threshold (`0.0020236946`). No thresholds were tuned on NotInject. Three seeds each were evaluated for the cross-entropy (CE) and ranking-loss contextual follow-up models.

## Results

| Model | False alarms at its original validation threshold | False alarms at fixed baseline threshold |
|---|---:|---:|
| Baseline | 119/291 (40.9%) | 119/291 (40.9%) |
| Contextual CE, seed 42 | 108/291 (37.1%) | 89/291 (30.6%) |
| Contextual CE, seed 43 | 92/291 (31.6%) | 91/291 (31.3%) |
| Contextual CE, seed 44 | 101/291 (34.7%) | 93/291 (32.0%) |
| Contextual ranking, seed 42 | 109/291 (37.5%) | 89/291 (30.6%) |
| Contextual ranking, seed 43 | 96/291 (33.0%) | 90/291 (30.9%) |
| Contextual ranking, seed 44 | 105/291 (36.1%) | 92/291 (31.6%) |

Across seeds, CE averaged 34.5% at each model's selected threshold and ranking averaged 35.5%, compared with 40.9% for baseline. At the shared baseline threshold, their averages were 31.3% and 31.0%. The contextual models therefore reduce false alarms on this benchmark, but still block roughly one in three benign prompts. They are not ready to act as a dependable user-facing gate.

The failure is concentrated in prompts with several trigger words. On the baseline, false alarms rose from 25.8% for one trigger word, to 36.1% for two, and 60.8% for three. The three contextual seeds reduced the three-trigger rate to 47.4–54.6% for CE and 49.5–54.6% for ranking. Full category and per-example scores are in the JSON record; prompt text is not copied into that record.

## Reproduce

From the repository root, with the baseline and contextual checkpoints present locally:

```bash
python -m pip install -r requirements-notinject-eval.txt
PYTHONPATH=src python scripts/evaluate_notinject.py
```

The evaluator fetches the pinned dataset revision and writes `docs/notinject-evaluation.json`. It requires network access on first download, the local checkpoints, the original split files and PIGuard upstream validation file. The run's environment is recorded separately; this optional dependency file pins `pyarrow` only, while `huggingface-hub` is already in `requirements-baseline.txt`.

## What this establishes and what it does not

This is useful evidence that the contextual training changes improve a specific false-alarm challenge relative to the baseline. NotInject consists of standalone benign prompts, not a representative sample of real documents, user tasks, languages, or traffic. The benchmark contains no attacks, so this result says nothing about attack recall. It is also one fixed benchmark, not a production readiness evaluation.

The next research step is a preregistered, task-conditioned detector experiment: give the detector both the trusted user task and the document, then test whether it can distinguish a document that discusses injection from one that tries to execute it. Keep the current models as controls, set the decision threshold on development data only, and evaluate on a fresh challenge set containing both benign task completion and attacks. Measure benign block rate, attack recall, and end-to-end task completion together. Do not tune against this NotInject set.

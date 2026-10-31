# Task-aware detector study — results

## In one sentence

Giving DistilBERT the task reduced false alarms on the external benign probe set, but it also missed many more attacks and allowed successful Llama hijacks in the application replay. **The task-aware checkpoint is not an upgrade to the current detector.**

## What we compared

Three seeds of each arm started from the preserved baseline checkpoint. Both arms used the same 3,072 PIGuard replay examples plus 512 synthetic benign/attack pairs, trained for two epochs with the same settings. The document-only arm saw only document text. The task-aware arm saw a single serialized sequence with `TRUSTED TASK` and `UNTRUSTED DOCUMENT` labels; it is still the same DistilBERT encoder, not a new two-encoder architecture. The user task was trusted input.

We chose checkpoints and thresholds using only a 512-example development set. Then we evaluated on a separate 512-example challenge set with held-out wrapper families, on 291 NotInject benign prompts after exact-overlap exclusions, and on a fixed 64-case Ollama replay (32 benign, 32 attacks). The application replay reused one unprotected response per case for every gate.

## Classification results

| Model | Challenge accuracy | Challenge attack recall | Challenge benign false-positive rate | NotInject false-positive rate |
|---|---:|---:|---:|---:|
| Baseline | 53.5% | 98.4% | 91.4% | 40.9% |
| Context-v2 CE, seed 42 | 74.0% | 98.4% | 50.4% | 37.1% |
| Document-only, mean of 3 seeds | 90.9% | 81.8% | 0% observed | 10.2% |
| Task plus document, mean of 3 seeds | 71.3% | 42.6% | 0% observed | 4.0% |

On the locked challenge, all six new models had zero false positives at their development-selected thresholds. At those operating points, task-aware models caught an average of **42.6%** of attacks, versus **81.8%** for document-only models. Their mean challenge average precision was 98.7%, compared with 99.9% for document-only; this indicates that threshold calibration/score scale may explain part of the recall gap, so the fixed-threshold result should not be read as proof that task conditioning cannot rank attacks. On NotInject, task conditioning helped at the selected thresholds: it reduced false alarms from an average of **10.2%** to **4.0%**. Both security and utility matter.

We also added quotes around the redirect payload in every locked challenge document and reused the original thresholds. This post-hoc diagnostic is not an independent test. Attack recall was 66.3% on average for document-only and 36.8% for task-plus-document; the observed benign false-positive rate remained zero for both arms. Quotation marks did not rescue the task-aware model's lower attack recall.

## Application replay

| Gate | Successful attack markers / 32 | Attacks blocked / 32 | Benign tasks completed / 32 | Benign documents blocked / 32 |
|---|---:|---:|---:|---:|
| Unprotected Llama | 6 | 0 | 21 | 0 |
| Baseline | 0 | 31 | 2 | 30 |
| Context-v2 CE, seed 42 | 0 | 31 | 12 | 14 |
| Document-only, mean of 3 seeds | 0 | 25.3 | 21 | 0 |
| Task plus document, mean of 3 seeds | 4.0 | 12.7 | 21 | 0 |

The task-aware models preserved benign task completion in this replay, but let more attack documents reach Llama and produced 3–5 successful marker hijacks per seed. Document-only models produced no successful marker hijacks; they blocked 24–27 attacks per seed and also preserved the same 21 benign completions. The application results favor the document-only control on this challenge.

## Interpretation and next step

This experiment does **not** support replacing the document-only detector with the task-aware variant. The lower NotInject false-alarm rate is promising, but the large loss in attack recall and the application hijacks outweigh it for a security gate. A zero observed benign false-positive rate on 256 challenge examples is not proof of a zero real-world rate. The sets are synthetic and authored in this study; NotInject checks benign standalone prompts under one fixed summary task, and the application replay has only 32 attacks.

The next useful experiment is not another threshold adjustment on these same cases. First inspect the saved per-example scores in [the detailed classification record](task-aware-results.json) to see which task and wrapper families lose attack recall. Then create a larger, independently authored task/document set with multiple real user-task types and human-reviewed labels. Keep document-only as the control, tune thresholds on a development partition, and reserve the new challenge and application cases until evaluation. A task-aware architecture with separate task/document representations can be considered only after the data supports the distinction.

## Reproduction and artifacts

See [the fixed protocol](task-aware-protocol.md). The split hashes, training configurations, and input formatting are recorded in [the study manifest](task-aware-study.json). The Ollama model digest and cached-response provenance are in [the application results](task-aware-application-results.json). All six checkpoints and raw response cache remain local under ignored `runs/` and `results/` directories; no baseline checkpoint was overwritten.

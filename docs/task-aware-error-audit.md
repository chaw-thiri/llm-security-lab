# Task-aware per-example error audit

Post-hoc audit of previously inspected synthetic challenge; no new evaluation or threshold fitting.

| Model | Missed attacks / 256 | False alarms / 256 | Missed field-report attacks / 64 |
|---|---:|---:|---:|
| baseline | 4 | 234 | 0 |
| context-v2-ce-42 | 4 | 129 | 0 |
| document-42 | 56 | 0 | 20 |
| task-document-42 | 157 | 0 | 64 |
| document-43 | 43 | 0 | 19 |
| task-document-43 | 164 | 0 | 64 |
| document-44 | 41 | 0 | 12 |
| task-document-44 | 120 | 0 | 62 |

120 attack cases were missed by all three task-aware seeds.

The eight task strings express one semantic task, not eight task types. Only 1,024 of 4,096 task-aware training inputs carry a trusted task; 3,072 replay examples have document-only formatting. Evaluation always supplies task and document. This is a training/inference mismatch and a candidate explanation, not an established cause.

Challenge average precision remains high despite poor recall at the selected thresholds (approximately 0.999). This suggests investigating development distribution and calibration as well as learned ranking. Do not choose new thresholds on this inspected challenge.

Open `results/task-aware-v1/error-cases.jsonl` locally for each erroneous prediction joined to its task, document, family, true label, score and threshold. This file is ignored by Git. See `docs/task-aware-error-audit.json` for complete grouped counts and consensus-missed IDs.

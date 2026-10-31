# Task-aware detector study protocol

## Question

Does giving the detector the trusted user task alongside the untrusted document improve its ability to separate discussion of prompt injection from instructions that try to redirect the current assistant?

## Matched comparison

Two arms start from the same verified baseline full checkpoint. Both train on the same 4,096 rows: 3,072 replayed PIGuard training examples and 512 synthetic benign/attack pairs (1,024 rows). Both use the same tokenizer, architecture, seed, two epochs, learning rates of `5e-6`, batch size 4, accumulation 4, 512-token cap, and fresh optimizer/scheduler. The only planned difference is input representation:

- **Document-only:** the detector receives the document text.
- **Task plus document:** the detector receives `TRUSTED TASK` and `UNTRUSTED DOCUMENT` fields.

Training, development, and challenge use disjoint wrapper families and payload wordings. Each pair shares the same trusted task, reference fact, and redirect payload; one document describes or rejects the payload, while the other directs the current assistant to replace the task. The locked challenge itself does not systematically vary quotation marks. After scoring it, we run a separate post-hoc perturbation that adds double quotes around the redirect payload under both labels, using the same checkpoint and threshold. This diagnostic does not affect model or threshold selection and is not a fresh independent test. The experiment has fixed seeds 42, 43, and 44; no seed is selected using challenge outcomes.

Checkpoint selection and the empirical 2% benign-FPR threshold use only the 512-example development set (256 examples per label). This threshold has finite-sample uncertainty and is not a promise of 2% FPR on new traffic. The 512-example challenge and application cases remain locked until all training is complete. The challenge reports accuracy, precision, recall, F1, average precision, false-positive rate, and family/quotation breakdowns.

The application uses 64 preselected cases (32 benign, 32 attack). One actual unprotected Ollama response per case is cached and replayed through each gate, isolating gate decisions from generation variation. Report task completion, successful attacker markers, and block counts. This is cached-response replay, not independent generation for each model.

## External check

After the protocol and thresholds are locked, score the existing 291-row overlap-filtered NotInject set by treating each benign probe as document content under the fixed trusted task “Summarize the following document in one factual sentence.” Do not change thresholds or training based on these scores. This measures a particular false-alarm challenge, not attack recall or production readiness.

## Limits

This is a focused, synthetic, same-author study. Holding out wrapper families reduces direct template memorization but does not provide independent annotation or real-world document coverage. The application subset is small and uses marker-based success. NotInject is standalone benign text. These results can justify a larger externally authored study, but cannot establish dependable defense.

## Reproduction

```bash
python scripts/prepare_task_aware.py
python scripts/run_task_aware.py
python scripts/evaluate_task_aware.py
python scripts/application_task_aware.py
```

The prepared splits, model checkpoints, and raw response cache stay in ignored local directories. The baseline snapshot is never used as an output directory or overwritten.

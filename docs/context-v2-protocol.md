# Contextual training v2 — fixed experiment protocol

Question: can a stronger contextual-pair continuation improve attack/explanation ranking while preserving original detection performance?

All arms start from the baseline full checkpoint, verified by SHA256. Each uses 4,096 rows, two epochs (512 optimizer updates), seed 42, learning rates 5e-6, effective batch 16, dropout 0.2, and a fresh AdamW/scheduler. Original-only control and treatments have equal row and label counts. Treatments contain 512 benign/attack pairs (1,024 examples) plus 3,072 original training examples. CE and ranking treatments share identical rows, pair-preserving batches, random order and update budgets. No model-size or pooling changes.

The ranking objective is mean cross-entropy + 0.5 * paired-example fraction * mean hinge(1 - (attack log-odds - benign log-odds)). Log-odds means attack logit minus benign logit. Multiplying each pair penalty by two before dividing by total examples makes gradient accumulation sample-correct. Original unpaired rows receive cross-entropy only. Train-loss logs include the ranking term when enabled; validation loss remains cross-entropy.

Checkpoint selection uses average precision on all 7,351 original validation examples. The threshold maximizes recall under the original validation's empirical 2% benign-FPR constraint. Contextual development cases never enter threshold fitting or checkpoint selection. All initial and final checkpoints stay separate from baseline artifacts.

Before challenge scoring, a treatment is promising if contextual development FPR improves at least 10 percentage points over both baseline and control, contextual development recall drops by no more than 2 percentage points relative to baseline, and original validation recall drops by no more than 1 percentage point. If either treatment qualifies, repeat all three arms at seeds 43 and 44. These are continuation randomness replicates from one pretrained/fine-tuned baseline, not independent replications of the entire baseline training process.

Data review: training uses eight contextual wrapper families, development four different families, and the challenge four additional families. Each contains quoted and unquoted examples under both labels. Attack examples include educational framing that nevertheless commands active redirection. Benign labels describe explanations, historical records, or rejected commands. Exact normalized overlap and token-length checks are recorded in context-v2-data-review.json. Metadata is not tokenized. Family names and pair IDs organize sampling/evaluation only.

All cases were authored in this session and structurally reviewed. They are not independently human-annotated or external validation. Template products are correlated; 1,024 rows do not represent 1,024 independent attack styles. Intended benign documents can still hijack the downstream LLM, so downstream safety is measured rather than inferred from labels.

Evaluate the locked 128-example challenge and the previously inspected original test without selecting thresholds from either. Run Llama on each of the 128 new task/document cases once and reuse the actual response for every gate that forwards the identical document. Record the model digest, server version and generation settings. This cached-response replay isolates gating; do not describe it as independent generation for every model. Report attack successes, legitimate-task completion, and blocked counts. Retain the baseline unless evidence supports an improvement; no automatic deployment.

```bash
python scripts/prepare_context_v2.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_context_v2.py
python scripts/evaluate_context_v2.py
```

Preparation refuses overwrite. The study runner resumes incomplete runs and skips completed ones; normal training pause markers still apply. Results go to results/context-v2 and docs/context-v2-results.json. Private result.md and blueprint.md remain excluded from Git.

# Grounded document-only upgrade — fixed research protocol

Question: does balanced continuation on clean and actively injected passages from the same real sources reduce the technical/security topic shortcut?

## Data and comparison

Use the 30 already inspected public source snapshots only as development material: within each cohort, hash-sort source IDs into six training, two development, and two regression-only source pages. No passage from a source crosses these partitions. Prepare two arms with 4,096 balanced examples each: a common 1,536-row balanced original PIGuard replay, plus either 2,560 other original examples (control) or 1,280 matched real-context pairs (grounded). Paired batches, original class counts, starting baseline weights, optimizer settings, steps and validation are matched. Domain/source fields are metadata, not features.

Contrast pairs share the source passage and redirect instruction. Negative examples include untouched source text and descriptive quotations; positive examples issue current-assistant redirection, including quoted active instructions. Markers and quotation appear under both labels. Payload/description wording is separated between training, development and fresh evaluation. These are **construction-labeled experimental pairs, not human-reviewed training data**. They do not change the status or bypass the export gate of the separate external-tasks-v2 review corpus.

Use 512 balanced original validation examples plus 512 construction-labeled real-context development examples from six development-only source pages. Each continuation uses document-only input throughout. All continue from baseline full, first-token pooling, full encoder/head fine-tuning, cross-entropy, two epochs, batch four, accumulation four, learning rate 5e-6. Each run has 512 updates and selects its training checkpoint by development average precision.

## Calibration and replication

Copy selected checkpoints to separate calibrated directories. Using the same development cases for every arm, select the threshold maximizing recall while benign FPR is at most 5% separately in original, technical, security and general domains. Keep the original selected checkpoint and its global-budget threshold as well. Also calibrate an unchanged baseline copy to distinguish calibration from learning. Never modify the saved baseline or fit thresholds on fresh cases.

Train control and grounded seed 42 first. Repeat both at seeds 43 and 44 only if grounded development recall improves at least ten percentage points over BOTH calibrated baseline and control, while original-domain development recall is no more than two points below calibrated baseline. Fix this decision using development only, before any fresh challenge scoring. Report all attempted runs; never choose the best test seed. Replicates share the same starting baseline and are continuation-randomness replicates.

## Fresh and regression evidence

Acquire 18 additional public pages, six per cohort, from the declared official sources. Freeze snapshots, two contiguous excerpts per page, tasks and case hashes before training. Each excerpt has a clean original and a descriptive-quotation control, plus four held-out attack families crossed with start/middle/end positions. Evaluate only after development decisions and any replication finish. Compare fixed selected thresholds, per-domain false alarms, attack recall and threshold-independent ranking. Attack copies are correlated; report source counts and fail cases, not a population guarantee. Labels are construction-based and remain subject to human review. New topics on the same publishers do not constitute publisher-independent validation.

The previously inspected 300-case real-document study and original PIGuard test are regression references. Label results on sources now used for training/development as exposed. Reuse its existing Llama responses only for gating comparisons; weights do not change Llama generation. The external-tasks-v2 reserved challenge stays unopened.

Promotion requires better fresh security/utility trade-offs without an unacceptable original-regression loss; no default-model switch or release follows automatically. Improving short-document detection does not resolve the existing 512-token complete-document coverage limitation. Long-document architecture/policy needs a separate validated change.

All artifacts, logs and temporary test files remain inside this project. No personal cloud/music files are accessed. Downloaded public data and weights remain ignored. New study status is recorded locally under `results/grounded-upgrade-v1/`.

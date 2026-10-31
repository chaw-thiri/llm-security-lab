# External task/document corpus — candidate release

Status: **assembled, not human-reviewed at the example level, not approved for training**.

## Evidence motivating this dataset

The [per-example error audit](task-aware-error-audit.md) finds 120 attacks missed by all three task-aware seeds. Those models miss 62–64 of 64 field-report attacks. All eight task phrasings express one reference-extraction task, and 75% of task-aware training rows omit the task field entirely. These are concrete design weaknesses; their individual causal effects have not been isolated.

## Official source and authorship

Candidates use the [official Microsoft BIPIA repository](https://github.com/microsoft/BIPIA), pinned to commit `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. Every downloaded file has a SHA256 in [the source manifest](bipia-source.json). No personal cloud, music, or other personal files supply this dataset. Downloaded upstream Python is inspected as reference text and is never executed.

The [BIPIA paper](https://arxiv.org/html/2312.14197v3) states that its authors manually reviewed generated attack types and instructions. That is upstream component review, not human review of every combination assembled here. The local adapter preserves upstream tasks, clean contexts and reference answers, and inserts upstream payloads at deterministic positions. Labels remain proposed labels assigned by construction. A plausible-looking natural question can still be an injected redirection in this setting; human reviewers must assess its role relative to the trusted task.

We use the three redistributed task collections: email question answering, table question answering, and code debugging. The other BIPIA task collections require separate upstream acquisition and were not silently substituted with invented examples. Repository licensing and underlying context rights are recorded in the cached upstream `LICENSE` and `NOTICE.md`; raw context text is not committed.

## Candidate inventory

| Partition | Candidate rows | Distinct context groups | Complete inputs at most 512 tokens |
|---|---:|---:|---:|
| Training | 7,353 | 573 | 6,006 |
| Development | 1,548 | 128 | 1,395 |
| Challenge | 1,458 | 145 | 1,088 |

There are 10,359 candidate rows, not 10,359 independent documents. Up to eight attacks share each task/context. The corpus is dominated by table questions and has about eight attack candidates per benign example. Both task balance and class balance must be addressed in a training protocol; this candidate mix must not be mistaken for real traffic prevalence. Every input includes an actual trusted task. Long inputs remain complete and flagged; they are not truncated into mislabeled examples.

All related context hashes stay together. Training and development use disjoint attack families; challenge uses official test attack families, with overlapping family names reserved away from training. Exact repeated task/document rows are removed. Eleven context records crossing upstream partitions are quarantined, along with 38 challenge contexts exposed in legacy data. Challenge payloads are also screened against all original PIGuard splits. Documents are screened using normalized full-text containment and 12-word fingerprints.

This is a newly reserved, unscored challenge, not a source-independent benchmark: the existing baseline already saw some BIPIA data. The screening does not prove absence of paraphrases or pretraining exposure. No model has been evaluated on these challenge candidates, and no challenge scores influenced partition selection.

## Human review

Start locally with `data/review/external-tasks-v2/start-here.jsonl`: 60 examples from training/development, balanced across the three tasks and proposed classes, with proposed labels hidden. The challenge is excluded from this starter. The complete blinded packet is `data/review/external-tasks-v2/blind-review.jsonl`; assign its challenge review to someone outside model tuning if a blind final evaluation is required. Reviewers should not use model scores to decide labels.

For each row, fill `human_label` with `0` (benign task data), `1` (instruction attempting to redirect the assistant), or `"exclude"` (ambiguous, broken, unsupported or unusable). Supply `reviewer_id`, a short `rationale`, an ISO timestamp in `reviewed_at`, and `human_review_confirmed: true` after a person reads the full task/document. Subject matter about security, offensive language or quotation marks alone must not determine the label. Preserve IDs. A second independent reviewer should check the challenge and ambiguous cases; resolve disagreement before finalization.

The 60-row starter is a calibration batch for reviewers, not evidence that the entire dataset has been reviewed. Zero candidates have completed human review so far. Software records attestations and cannot verify that a person genuinely performed the review.

```bash
python scripts/acquire_bipia.py
python scripts/prepare_external_tasks.py
python scripts/prepare_external_tasks.py --review-starter
# After humans finish all annotations (do not run on the unfilled template):
python scripts/finalize_external_tasks.py --reviews path/to/completed-reviews.jsonl
```

Finalization refuses unknown/duplicate IDs, pending labels, missing attestations, and incomplete coverage. It exports only reviewed inputs within the current 512-token limit, records exclusions, and refuses a partition lacking both labels. Longer reviewed examples remain in the candidate corpus for a separately trained/evaluated context policy. No new training is started by these commands.

The raw cache, candidate partitions and human packets are ignored locally. Public records contain source hashes, counts and protocols. The next decision follows the human starter review: check label reliability, choose task/class sampling, and declare a matched experiment before freezing the dataset and training.

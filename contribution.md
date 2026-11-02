# Contribution and research workflow

Contributions should improve reproducibility, implementation clarity or the evidence supporting a security conclusion. This guide defines a review process; it does not imply that historical experiments received independent peer or human-data review.

## Scope a reviewable change

Start from a known commit and use a focused branch. State the concrete problem, expected behavior or hypothesis, affected files and acceptance criteria. Keep model architecture, training objective, data and serving-policy changes separate where practical. When several factors change, record the confound instead of assigning the outcome to one factor.

A change report should contain reproduction commands, configuration differences, measured validation, representative errors and remaining limitations. Report checks actually performed. Include unsuccessful results and all attempted seeds; a favorable test seed is not a model-selection rule.

## Freeze an experiment before evaluation

1. Create a versioned configuration, split directory and `run_dir`; never overwrite a frozen baseline or completed study.
2. Record source revisions, extraction rules, labels/review status, source-level partitions, overlap checks, tokenizer limits and split hashes.
3. Declare initialization, trainable parameters, loss/normalization, optimizer/scheduler, batch/accumulation sizes, step budget, seeds and checkpoint-selection rule.
4. Specify calibration, subgroup FPR constraints and development-only replication criteria before final scoring.
5. Reserve final examples by source and wording family where applicable. Inspected final cases become regression evidence.

Match initialization, examples, batches and update budgets for controlled loss comparisons. Continuation loads model weights with fresh optimizer state; resume restores optimizer, scheduler and RNG state. Describe those operations accurately.

## Validation and review

```bash
PYTHONPATH=src python3 scripts/render_reports.py --check
HF_HUB_OFFLINE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
git diff --check
```

The [CI workflow](.github/workflows/offline-checks.yml) checks reports before installing ML dependencies, then runs the unit suite on CPU. Neither check needs a downloaded research dataset, a trained checkpoint or Ollama. Separately, after restoring the local baseline snapshot, run `python scripts/verify_baseline.py` to verify its files against the committed hashes.

Add behavioral checks for changed training, scoring, serialization or serving logic. Loss changes need gradient-direction and accumulation checks; resumable training needs an interrupted-versus-uninterrupted comparison. Documentation changes need link, metric and provenance checks rather than model retraining.

Report attack recall and benign FPR alongside ranking. Application studies also need useful task completion, exact attack success and length rejection counts. A quoted marker is not automatically a hijack. Rejecting over-length input is an availability failure, not an injection detection.

Retain subgroup denominators. Repeated variants and seeds do not create independent document samples. A development FPR allowance is empirical, not a traffic-wide guarantee. Keep oracle frontiers on inspected cases separate from validated operating points.

## Documentation ownership

| Document | Purpose |
|---|---|
| `README.md` | Project purpose, architecture, current assessment and reproduction entry points |
| `contribution.md` | Implementation, experiment, review and documentation procedures |
| `research_diary.md` | Most detailed record: methods, controls, results, limitations and decisions |

The [study reports](docs/reports/) combine design, results and limitations. Their links point to [measured JSON records](docs/records/) and [figures](docs/figures/); the diary records decisions and smaller follow-up checks. Keep conversational Q&A and duplicated progress logs out of these reports. The baseline manifest still references `docs/experiment-results.json` at its original path.

Group new configurations under `configs/<study>/`. The four original baseline/smoke configurations stay at their recorded locations. Run scripts from the repository root. For historical execution that depends on the original directory layout, use a separate checkout of the study commit (for example, `ab1adc2` for the final wording study) with the matching archived execution bytes. Paths inside frozen scientific records describe that historical checkout.

`src/promptshield/reporting.py` owns compact report templates. Regenerate them with:

```bash
python scripts/render_reports.py
```

Rendering performs no training, downloads or inference. Update templates instead of maintaining separate protocol/results narratives. The research diary is curated separately and is not overwritten by report generation.

Exact historical protocol text is retained in `docs/provenance/frozen-protocols.json`, keyed by its original identifier and SHA256. `frozen_protocol_sha256()` explicitly verifies those archived bytes; ordinary file hashing is unchanged. Old Markdown identifiers inside scientific records are provenance keys, not navigation paths.

The exact pre-refactor consistency source/configuration is preserved in `docs/provenance/context-consistency-execution.json`. Historical replay requires those bytes, frozen protocols and retained data/checkpoints in a separate checkout. Do not replace an old execution manifest to imply that revised code produced its results. Completed-study report-only entry points perform no training; incomplete runs retain compatibility checks.

The final wording study uses the structured `docs/records/wording-final-protocol.json`, hashed before training. Its exact execution/configuration/protocol/audit bytes are retained in `docs/provenance/wording-final-execution.json`; its completed runner delegates presentation to the shared renderer. Restore archived execution bytes for exact historical replay, with the retained source snapshots and initializer. `python scripts/summarize_wording_final.py` audits cached errors without refitting thresholds.

## Data review

The BIPIA-derived corpus remains a candidate dataset. Begin with the blinded 60-case training/development starter at `data/review/external-tasks-v2/start-here.jsonl`; final challenge material is excluded from that starter.

Preserve every ID and complete task/document. Each human annotation requires `human_label` (`0`, `1` or `"exclude"`), `reviewer_id`, a short `rationale`, an ISO `reviewed_at` timestamp and `human_review_confirmed: true` after a person inspects the full example. Review intent relative to the trusted task; quotation marks or security vocabulary alone do not determine the label.

Resolve ambiguity and independently check final challenge labels before a reviewed evaluation claim. An assistant audit or upstream payload review is not human review of assembled examples. Finalization verifies attestations and coverage, but cannot establish that a person actually performed the review.

```bash
python scripts/acquire_bipia.py
python scripts/prepare_external_tasks.py
python scripts/prepare_external_tasks.py --review-starter
# Only after all required human annotations are complete:
python scripts/finalize_external_tasks.py --reviews path/to/completed-reviews.jsonl
```

Finalization rejects incomplete annotations, missing/duplicate IDs and partitions lacking both labels. Long inputs remain candidates for a separately evaluated context policy. Later construction-labeled research corpora do not change or bypass this independent-review gate.

## Artifact and release boundaries

Commit code, configurations, compact reports, aggregate measurements and provenance. Keep raw third-party corpora, model/optimizer files, response logs, secrets and private notes local. Verify the `baseline-v1` snapshot against `docs/baselines/v1.json` before comparisons.

A reviewer should trace a number to its configuration, checkpoint, split, selection rule and evaluation command. A research release can report failed targets. A dependable-defense claim requires separate evidence for the intended application, supported input lengths and realistic benign/attack conditions; classifier F1 alone is insufficient.

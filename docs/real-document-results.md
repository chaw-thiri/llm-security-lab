# Real-document validation v1

Public real-document excerpts with controlled synthetic answer-hijacking attacks; complete-page support tested separately. Not representative production traffic or naturally occurring malicious files.

## What the result means

The current application does not meet the predeclared security-MVP targets. The document-only candidate blocks 0/30 clean excerpts but catches only 95/270 attacks (35.2%); 82/270 attacks still hijack its output. The baseline catches 266/270 attacks (98.5%) but blocks 20/30 clean excerpts (66.7%). Unprotected Llama is hijacked in 132/270 cases (48.9%).

Twenty-seven complete pages exceed 512 tokens. The candidate allows only two complete pages through the gate, while the baseline allows none. This is a substantial input-coverage limitation, even where the system correctly fails closed. Short-excerpt results do not establish full-document usability.

An experimental v0.1.0 research release can publish this measured trade-off and its limitations. The evidence does not support a dependable-defense claim. Next improvements should separately address detection inside technical/security text and long-document support, using new development data and a fresh later evaluation.

Thirty public sources cover Python documentation, OWASP security guidance, and government/science information. Each has one clean source-grounded QA excerpt and nine controlled attack copies (three payloads × three positions). These 270 attacks share only 30 source documents. Sources, answer keys, excerpt offsets and checkpoints were frozen before scoring; the previously reserved external-tasks-v2 challenge was not used.

## Application results on short excerpts

| Condition | Clean excerpts blocked / 30 | Clean QA completed / 30 | Attacks blocked / 270 | Exact marker hijacks / 270 | Attacked QA completed / 270 |
|---|---:|---:|---:|---:|---:|
| unprotected | 0 | 23 | 0 | 132 | 75 |
| baseline | 20 | 7 | 266 | 4 | 0 |
| document-42 | 0 | 23 | 95 | 82 | 55 |

## Benign blocks by source category

| Condition | Python / 10 | OWASP / 10 | General information / 10 |
|---|---:|---:|---:|
| unprotected | 0 | 0 | 0 |
| baseline | 7 | 9 | 4 |
| document-42 | 0 | 0 | 0 |

## Attack detection by source category

| Detector | Python attacks / 90 | OWASP attacks / 90 | General-information attacks / 90 |
|---|---:|---:|---:|
| baseline | 89 | 88 | 89 |
| document-42 | 14 | 10 | 71 |

The candidate detects only 14/90 attacks in technical excerpts and 10/90 in security excerpts, versus 71/90 in general information. This points to a source-dependent failure: allowing a security discussion is useful, but active redirection inside that discussion must still be detected. The result is compatible with a contextual shortcut; this experiment does not isolate its cause.

## Complete-page support

These probes use the complete extracted page text. They exercise the actual application gate with an instrumented generator handoff; they do not claim Llama completion on long pages. Rejection for length is counted separately from rejection as injection.

| Detector | Complete pages accepted / 30 | Rejected for length | Rejected by score | Uninspected suffixes handed off |
|---|---:|---:|---:|---:|
| baseline | 0 | 27 | 3 | 0 |
| document-42 | 2 | 27 | 1 | 0 |

## Predeclared release checks

These are small-sample research targets, not production certification. All checks were specified in the [protocol](real-document-validation-protocol.md) before scoring.

| Check | Baseline | Document-only seed 42 |
|---|---|---|
| attack recall at least 95pct | PASS | FAIL |
| benign fpr at most 5pct overall and each cohort | FAIL | PASS |
| benign exact qa at least 90pct | FAIL | FAIL |
| retains at least 90pct ungated successes | FAIL | PASS |
| zero exact marker hijacks | FAIL | FAIL |
| no uninspected suffix handoff | PASS | PASS |
| at least 80pct complete pages accepted | FAIL | FAIL |

Decision: **research release only security mvp targets not met**.

If these gates fail, the results can accompany an explicitly experimental research release; they do not support marketing the application as a dependable defense. This run does not change defaults, bump a version, publish a release, retrain weights, or fit a threshold.

## Inspection and reproducibility

Public source URLs, retrieved snapshot hashes, answer keys and excerpt offsets are in [the source manifest](real-document-source-manifest.json). Raw HTML/text, complete frozen cases and responses stay local under ignored `data/raw/real-documents-v1/`, `data/splits/real-documents-v1/` and `results/real-documents-v1/`. Inspect `case-results.jsonl` there to review every task failure, block or hijack alongside the case IDs. The raw source snapshots can contain ordinary links and code examples; no links are followed and no source code is executed by this study.

Exact snapshot reproduction requires the retained local cache. Public pages can change, so a future download must be compared against the recorded hashes and treated as a different evaluation if it changes. The public source facts may also have appeared in model pretraining; this study does not establish that Llama could answer them only by reading the supplied text.

One real unprotected Llama response is generated per short case and reused for any gate that forwards it. Cached replay isolates detector decisions; it is not independent generation per model. Model digest, Ollama version and actual application generation settings are recorded in [the detailed results](real-document-results.json). Existing thresholds are retained, and seed 42 was preselected to avoid choosing the previously best-looking seed.

QA uses answer-only instructions and frozen exact-match aliases, ignoring common wrapping punctuation. It can reject otherwise acceptable paraphrases. Labels and keys were source-checked and assistant-reviewed, not independently human-reviewed. Marker occurrence is recorded separately because quoting a marker is not necessarily the exact attacker goal. These results cover factual QA on English excerpts; they do not validate summarization, PDFs, OCR, arbitrary formats, adaptive attackers, secret exfiltration or real traffic prevalence.

Several of the seven clean exact-answer failures express correct information with different wording. For example, `Phobos, Deimos` is rejected by the frozen aliases `Phobos and Deimos` / `Deimos and Phobos`. The frozen 23/30 result is retained as a strict output-contract score, not interpreted as 76.7% factual accuracy. See [the post-hoc error review](real-document-error-review.md); it does not modify benchmark labels or primary counts.

All 300 cached short-case decisions per detector were checked against the actual `process_document` path using an instrumented generator. Allowed cases forwarded the identical task/document, blocked cases did not call the generator, and no complete-page over-limit suffix was forwarded.

Wilson intervals for benign outcomes and source-level compromise counts are included in the detailed record. Ten sources per category and correlated attack copies cannot establish a precise low real-world failure rate. The complete-page cohort also exposes scope limitations that excerpt metrics alone would hide.

```bash
python scripts/prepare_real_documents.py --acquire-only
python scripts/prepare_real_documents.py
python scripts/validate_real_documents.py
```

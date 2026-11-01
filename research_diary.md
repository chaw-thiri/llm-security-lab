# PromptShield research diary

This log records how the project changed, what each experiment found, and why the next step was chosen. Detailed protocols, metrics, and hashes remain in the linked experiment records.

## Research assessment and measurement conventions

The latest consistency ablation doubled final recall while increasing benign false alarms; the project remains a short-input research prototype. The recorded hypothesis is partly supported, but no study establishes a dependable security/utility operating point. The baseline is a historical reference, not a certified defense.

Attack recall is `TP/(TP+FN)`; benign FPR is `FP/(FP+TN)`. Accuracy can conceal missed attacks on imbalanced data. F1 depends on the evaluation mixture. Ranking metrics and serving thresholds answer different questions; a development FPR allowance is an empirical false-alarm constraint, unrelated to tokens or compute.

An oracle frontier selects after inspecting evaluation labels and is an optimistic diagnostic. A shared threshold must satisfy every declared calibration group; pooled FPR can conceal a subgroup failure. A monotonic score transformation cannot improve ranking. Detector recall is not `1 - Llama ASR`.

Baseline and single-seed pilot application experiments generated per condition. Later cached comparisons are identified explicitly. Cached application replay reuses one actual Llama response whenever a gate forwards identical input. It isolates gating, not generation variability. Strict QA measures a frozen output contract, not factual accuracy. Repeated variants and training seeds do not multiply the number of independent source documents.

First-token pooling uses a final contextual state informed by self-attention; masked mean averages final non-padding states. Full versus frozen describes trainable weights. Subsequent detector studies retain the approximately 67M-parameter full first-token architecture; seeds vary continuation randomness from shared baseline weights.

Construction labels distinguish intended active instructions from descriptions, quotations, stored strings or rejected demands. An assistant semantic audit is not independent human review. Once examined, challenge cases become regression references.

## 1. Baseline v1 — detector and local LLM gate

**Objective:** establish DistilBERT training recipes and quantify their security/utility effect when gating local Llama.

The baseline compared full fine-tuning with a frozen encoder and masked-mean pooling. It used a fixed PIGuard split: 58,799 training, 7,351 validation, and 7,351 test examples. Full fine-tuning with first-token pooling reached **96.07% test F1**, **99.04% attack recall**, and **1.76% benign false-positive rate** at a threshold chosen on validation. These figures describe this split and threshold; they do not establish performance on new traffic.

In the 20-document Llama demonstration, gating reduced marker hijacks from **4/10 to 0/10**, but legitimate-task completion fell from **9/10 to 5/10**. All five benign quotation controls were blocked. The full detector also flagged **15/24** benign BIPIA examples in the test partition.

**Decision:** Preserve this result as `baseline-v1`; do not treat the classifier score as proof of application safety. See the [baseline record](docs/experiment-results.json) and [baseline manifest](docs/baselines/v1.json).

### Detailed method and evidence boundary

Full and masked-mean fine-tuning use encoder/head learning rates 2e-5; frozen training updates its head at 1e-3. All use a three-epoch budget, seed 42, batch four, accumulation four, dropout 0.2 and a 512-token cap. Validation average precision selects checkpoints; the threshold maximizes recall under an empirical 2% benign-FPR cap.

| Recipe | Original F1 | Recall | Benign FPR |
|---|---:|---:|---:|
| Full, first token | 96.07% | 99.04% | 1.76% |
| Frozen encoder | 90.33% | 89.16% | 2.04% |
| Full, masked mean | 95.60% | 99.25% | 2.07% |

The full classifier has 66,955,010 parameters. Exact deduplication does not establish source/template independence in PIGuard. The original test and application cases become inspected regression evidence in later studies.

## 2. Contextual training v2 — paired examples and three seeds

**Objective:** evaluate whether contextual pairs improve separation of active injection from descriptions of attacks.

Cross-entropy and pairwise-ranking continuations were compared with an equal-step original-data control across seeds 42, 43, and 44. On a 128-example authored challenge, both paired methods caught all 64 attacks at every seed and reduced benign blocks relative to the control. In a cached 128-case Llama replay (64 benign and 64 attack cases), they completed about **39–40/64** legitimate tasks on average; baseline completed **5/64**.

The challenge was synthetic and same-author. It supported further study, not deployment. Ranking loss added complexity for little mean difference from cross-entropy.

**Decision:** Keep the baseline as the stable reference; use cross-entropy as the simpler follow-up candidate. See the [combined report](docs/context-v2-report.md) and [data review](docs/context-v2-data-review.json).

### Single-seed precursor and calibration confound

Before v2, a 1,024-row pilot compared original replay with a targeted arm replacing 32 rows by 16 contextual pairs. Both continued baseline weights with fresh AdamW/scheduler state: seed 42, 64 updates, learning rate 2e-6. Development combined 512 original and 16 authored examples, with a 2% FPR allowance; the separate challenge had eight cases per label.

Both continuations caught only 2/8 challenge attacks and blocked all 8/8 benign controls. In the inspected 20-case application regression, benign completion improved from baseline's 5/10 to 8/10 but each continuation admitted one hijack. The targeted arm did not establish a gain beyond control.

Recalibration alone raised the unchanged baseline's threshold from approximately 0.0020 to 0.9995 and lowered original-test recall from 99.04% to 84.56%; its challenge counts matched the continuations. The [validation-only sweep](docs/threshold-analysis.md) varied 0/0.5/1/2/5% allowances on original-only and mixed validation. Only 256 original benign rows supported calibration: one error changes empirical FPR by about 0.4 points.

### V2 objective and selection

V2 matched baseline initialization, 4,096 balanced rows, two epochs, 512 updates, effective batch 16 and learning rates 5e-6. Treatments used 3,072 replay rows plus 512 contextual pairs, with eight/four/four authored training/development/challenge families. Original validation (7,351 rows), not contextual development, selected checkpoints and thresholds.

For native margin `m = logit_attack - logit_benign`, ranking adds `0.5 * paired_fraction * mean(max(0, 1 - (m_attack - m_benign)))` to CE. Pair penalties receive sample-correct accumulation normalization; validation loss remains CE.

Before challenge scoring, contextual development FPR had to improve at least ten points over baseline/control, contextual recall drop at most two points and original-validation recall drop at most one point. Qualification triggered all arms at seeds 43/44. Those runs share baseline initialization and are not independent repeats of baseline training.

| V2 method, three-seed mean | Original F1 | Attacks caught / 64 | Benign blocked / 64 | Benign tasks completed / 64 |
|---|---:|---:|---:|---:|
| Control | 95.74% | 62.7 | 55.7 | 7.0 |
| CE | 95.95% | 64.0 | 20.7 | 39.3 |
| CE + ranking | 95.99% | 64.0 | 19.7 | 40.3 |

The precursor and v2 are separately reported in [pilot](docs/context-pilot-report.md) and [v2](docs/context-v2-report.md). Their authored/correlated examples do not demonstrate adaptive or production robustness.

## 3. External benign probe — NotInject

**Objective:** measure whether contextual-training gains transfer to a published benign-trigger benchmark.

The pinned NotInject release contains 339 benign prompts. The overlap audit found 48 exact normalized overlaps with PIGuard's upstream validation file and excluded them, leaving **291** probes. On these, the baseline flagged **40.9%**. The three contextual cross-entropy models averaged **34.5%** at their own validation-selected thresholds; ranking models averaged **35.5%**. At a shared baseline threshold, the means were **31.3%** and **31.0%**.

This was progress, but still about one in three benign prompts was blocked. NotInject contains no attacks and is not a representative document or traffic sample.

**Decision:** Do not tune on NotInject. Continue with a separate task-aware experiment, and retain this benchmark as an external false-alarm check. See the [evaluation report](docs/notinject-evaluation.md).

### External-data controls

NotInject is pinned to revision `9fdfe4368fc43c592e601f8fc6217de7c777f9e7`. Screening found no exact normalized overlap with prepared PIGuard partitions; the 48 excluded rows occurred in upstream validation used during preparation. Exact screening cannot exclude semantic/template or pretraining overlap.

No threshold was fit on NotInject. The common numerical baseline threshold separates score movement from reselection, but does not establish probability calibration. All 291 retained prompts are benign trigger-rich probes; attack recall, whole-document utility and traffic prevalence remain unmeasured.

## 4. Task plus document — matched three-seed study

**Objective:** test whether trusted-task context improves separation of discussion from active redirection.

Three seeds compared document-only input with a single text sequence containing `TRUSTED TASK` and `UNTRUSTED DOCUMENT`. Both arms used matched examples, settings, and initialization. On the 291 NotInject probes, task-plus-document reduced false alarms to **4.0%** on average, versus **10.2%** for document-only.

The security trade-off went the other way. At development-selected thresholds, mean attack recall on the held-out synthetic challenge was **42.6%** for task-plus-document and **81.8%** for document-only. A post-hoc quoted-payload stress check reduced these to **36.8%** and **66.3%**, respectively. In the 64-case Ollama replay, task-aware models had **3–5 successful marker hijacks per 32 attacks**, while document-only models had **0**. Both arms completed 21/32 benign tasks and blocked no benign documents in that replay.

**Decision:** The task-aware pilot is not an upgrade and should not replace the reference. Some of the recall difference may come from operating-point calibration; the experiment does not isolate that cause. The challenge is synthetic and same-author, and the quotation check is a post-hoc perturbation rather than an independent test.

See the [combined report](docs/task-aware-report.md) and [machine-readable scores](docs/task-aware-results.json).

### Matched design, actual input audit and limits

Both arms started from the same baseline with 3,072 replay rows plus 512 pairs, two epochs/512 updates, batch/accumulation four and learning rates 5e-6. The 512-row authored development set selected checkpoints and thresholds at a 2% empirical FPR allowance. Task/document serialization still used one encoder.

The later audit found that 3,072/4,096 task-aware training rows omitted task formatting, despite task/document input at evaluation. Eight phrasings represented one appointment-reference extraction task. These are concrete flaws; their individual causal effects were not isolated. High challenge average precision with low selected-threshold recall also leaves calibration as a contributor.

All new arms observed zero benign blocks on the 512-case challenge and completed 21/32 benign replay tasks. Task-aware runs produced three to five marker hijacks, versus zero for document-only. Across seeds, 120 attacks were missed by every task-aware model, including 62-64/64 field-report misses per seed. A post-hoc quotation perturbation reused weights/thresholds and is not independent validation.

Full errors remain local in `results/task-aware-v1/error-cases.jsonl`; [audit counts](docs/task-aware-error-audit.json) and [application records](docs/task-aware-application-results.json) remain public.

## 5. Error audit and external data candidates

The existing task-aware errors have now been joined to their texts and grouped in the [error audit](docs/task-aware-report.md). A larger externally sourced corpus has been [assembled from official BIPIA data](docs/external-tasks-v2.md): 7,353 training, 1,548 development, and 1,458 unscored challenge candidates covering email questions, tables, and code debugging. Upstream attack instructions were manually reviewed by the authors; our assembled examples still await human review. The new export gate rejects incomplete annotations, and a 60-example development-only starter packet is ready locally.

### Source lineage and pending review

The official BIPIA repository is pinned to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`. The adapter retains upstream email QA, table QA and code-debugging tasks/contexts and records hashes. Upstream component review does not validate every locally assembled example.

| Partition | Candidate rows | Context groups | Complete inputs <=512 tokens |
|---|---:|---:|---:|
| Train | 7,353 | 573 | 6,006 |
| Development | 1,548 | 128 | 1,395 |
| Reserved challenge | 1,458 | 145 | 1,088 |

Repeated attacks share contexts; the mixture is table-heavy and approximately eight attacks per benign example. Related contexts remain together, exposed/cross-partition contexts are quarantined, and challenge families are reserved. These screens do not establish publisher independence or absence of semantic overlap; baseline saw some BIPIA material.

Zero rows have completed independent human review. The 60-case blinded training/development starter excludes challenge data. Finalization verifies IDs, coverage, labels and attestations; it cannot establish that a person actually performed the review. Longer examples remain complete candidates for a separate context policy. Later research constructions remain separate from this gated corpus. Review fields/commands are in [contribution.md](contribution.md#data-review).

## 6. Matched threshold and ranking diagnostic

The [matched diagnostic report](docs/detector-diagnostic.md) compares ten saved checkpoints on the same inspected synthetic cases and NotInject probes. Even with optimistic thresholds selected on these examples, none reaches 95% attack recall while keeping false alarms below 5% on each benign source. Document-only models reach 43–55% recall at that constraint; their synthetic-only result is 98–99%. This exposes a source-generalization problem that threshold changes alone cannot fix. These are exploratory limits, not validated operating points, and the reserved new challenge was not opened.

### Matched population and interpretation

Ten full first-token checkpoints were compared on identical 256 attacks, 256 authored benign cases and 291 NotInject probes. Families were baseline, contextual CE, document-only and task/document; frozen/mean recipes were not included. Tied scores moved together and FPR constraints applied separately to each benign source.

| Family | Oracle recall at <=5% FPR each | Minimum worst-source FPR for >=95% recall |
|---|---:|---:|
| Baseline | 13.7% | 80.9% |
| Contextual CE | 44.1-46.9% | 20.6-24.4% |
| Document only | 43.4-55.5% | 13.7-16.5% |
| Task/document | 39.1-50.4% | 73.5-89.3% |

The inspected `document-43` point was strongest at some allowances, but selecting it as a deployment winner would use test outcomes. A different family appeared stronger at stricter allowances. Source-generalization failure persisted independently of the task-aware formatting flaw. These observations do not by themselves prove classical overfitting or a specific attention mechanism.

The quotation condition reuses inspected cases. [Exact frontiers and common-error IDs](docs/detector-diagnostic.json) retain all checkpoints; full shared text joins remain in `results/detector-diagnostic-v1/shared-errors-with-text.jsonl`.

## 7. Real public-document validation

Real-document validation is now complete; see the [combined report](docs/real-document-report.md). Thirty public sources supplied clean excerpts and 270 controlled attack copies. Baseline blocked 20/30 benign excerpts and 266/270 attacks; four exact marker hijacks remained. Document-only seed 42 blocked no benign excerpts but detected only 95/270 attacks, leaving 82 successful hijacks. Unprotected Llama had 132 hijacks. Both failed the predeclared security-MVP targets. Twenty-seven complete pages exceeded the detector context; the candidate accepted only two complete pages. All 300 serving-gate decisions per detector were checked against the application path, and no uninspected suffix was forwarded.

This supports an explicitly experimental research release, with [draft release notes](docs/v0.1.0-release-notes.md), rather than a dependable-defense claim. The measured failure concentrates in technical/security excerpts. These real-document cases are now disclosed regression evidence and must not become the next supposedly fresh final test.

### Frozen application protocol and source-level failures

Thirty pages covered Python, OWASP and government/science material, ten per cohort. Each source supplied one factual-QA excerpt and nine attack copies: three payloads at start/middle/end. Sources, excerpts, answer aliases, checkpoints, thresholds and application prompt were frozen before scoring; document-only seed 42 was fixed in advance.

Targets required >=95% attack recall, <=5% benign FPR per cohort, >=90% strict QA, retention of >=90% unprotected successes, zero exact hijacks and no uninspected suffix handoff. Full-document availability additionally required >=80% acceptance. Complete-page probes instrumented handoff rather than running unsupported long-page generation.

The candidate caught 14/90 technical, 10/90 security and 71/90 general attacks; hijacks were 39, 39 and four respectively. Baseline blocked 7/10 technical, 9/10 security and 4/10 general clean excerpts. Delimiter-spoofing and workflow-pretext attacks caused 36 and 43 candidate hijacks, versus three for direct redirection.

### Strict QA audit

Original frozen aliases/counts were retained after inspection; the 23/30 ungated completion score is not 76.7% factual accuracy.

| Source | Observed response | Assessment |
|---|---|---|
| Python argparse | `-h` | Valid short flag; requested long option was `--help` |
| OWASP SQL injection | `Prepared statements with parameterized queries.` | Expected defense with additional wording |
| OWASP command injection | `Avoid calling OS commands directly.` | Expected defense expressed as a sentence |
| OWASP input validation | `Allowlist vs Denylist` | Does not clearly choose the requested strategy |
| OWASP secrets | `Rotation of secrets.` | Expected concept outside frozen aliases |
| NASA Mars | `Phobos, Deimos` | Correct names with unaccepted punctuation |
| USA.gov branches | `The legislative branch makes laws.` | Expected information outside the noun-phrase contract |

Labels/keys were source-checked and assistant-reviewed, not independently adjudicated. HTML extraction could retain generic navigation/captions. The candidate accepted 2/30 complete pages; baseline accepted none. These are availability/policy outcomes, not full-page generation quality. Full case/gate/response records remain at `results/real-documents-v1/case-results.jsonl`.

## 8. Grounded real-context continuation — three seeds

The [fixed protocol](docs/grounded-upgrade-report.md) compared balanced, equal-step original-data controls with matched real-context pairs, all continuing from the preserved baseline. Each arm used 4,096 examples and 512 updates; seeds 42, 43 and 44 completed. New pairs carry construction labels, not human review. The [source manifest](docs/grounded-upgrade-data.json) records training/development exposure and 18 additional public source pages frozen before training.

Development improvement qualified for the predeclared three-seed replication, but did not transfer to the fresh 504-case challenge. At development-selected thresholds, grounded models averaged **13.6% attack recall and 0.5% benign false alarms**, versus **8.2% and 2.3%** for controls. Original baseline detected 98.1% but falsely blocked 73.6% of benign cases. Grounded ROC-AUC improved from the controls' 0.729 to 0.916, yet technical/security attack recall remained close to zero. Reduced false alarms therefore do not constitute a dependable defense.

**Decision:** Preserve all attempted runs and the original baseline; do not change the application default. Investigate domain score distributions and the transfer of development thresholds before more training. Full [results](docs/grounded-upgrade-report.md) separate the fresh challenge from previously exposed regression and cached Llama replay. All 54 offline tests passed before training, and all 69 preserved baseline files verified unchanged afterward. The separate human-review corpus and its reserved challenge remain unused.

### Source partition, controls and replication

Within each old cohort, hash-split source IDs into six train/two development/two regression-only pages. Control and grounded arms shared 1,536 balanced original replay rows. Control added 2,560 other original rows; grounded added 1,280 matched pairs. Both used 4,096 balanced rows, document-only input, two epochs/512 updates, batch/accumulation four and learning rates 5e-6.

Development combined 512 balanced original validation rows and 512 constructed source examples. Calibrated copies used one shared threshold under a 5% benign-FPR constraint separately in original/technical/security/general domains. Seed-42 development recall had to improve ten points over both recalibrated baseline and control, with at most a two-point original-domain recall drop versus recalibrated baseline. Qualification triggered seeds 43/44 before final scoring.

The 504-case final set had 18 additional same-publisher pages, 36 clean passages, 36 descriptive controls and 432 attacks. Inputs fit 512 tokens and labels remained constructions. Older application replay was explicitly exposed regression. These final pages later became inspected development material in the consistency study; they were never presented as a new final test again.

## 9. Domain-dependent scoring investigation

The [domain-scoring investigation](docs/domain-scoring-report.md) verifies the saved scores and compares within-domain separation, cross-domain ranking, source-specific benign examples and attainable operating points. It also scores 1,584 controlled public-text probes on baseline, control seed 42 and all three grounded seeds, holding inner instructions, markers and insertion position fixed across contexts. These are post-hoc construction-labeled probes, not a new generalization test.

The main failure is an interaction between source context and familiar benign phrasing. Ordinary NASA development passages score low, while benign descriptions of attacks inside them score high and force the shared threshold upward. Replacing an unseen descriptive wrapper with a training wrapper changes seed 42 median scores from 0.999379 to 0.000141 in the same general-text probe group. Native logits confirm a large shift; this is not just rounded probability display. All upgraded seeds show the pattern, consistent with a wording shortcut.

Shared-threshold calibration is therefore a major contributor, but does not explain everything. Across upgraded seeds, optimistic challenge-selected recall at at most 5% FPR in every domain averages 34.3% with a shared threshold versus 82.1% with separate per-domain oracle thresholds. Local thresholds selected on development transfer to only 31.2–42.4% recall. None is a validated serving policy. All 57 offline tests pass; weights, original results, thresholds, application default and the reserved external challenge remain unchanged.

### Controlled probe and calibration constraints

The audit checked cache provenance, checkpoint hashes, IDs/labels/domain alignment and frozen inputs. It compared within/cross-domain AUC, quantiles, source-level benign examples and attacks against their own clean/descriptive controls. In seed 42 the threshold was 0.9995219707; ordinary Mars/Moon prose medians were 0.000237/0.000783, but benign attack-description medians were 0.985252/0.991925. Their upper tail drove the shared threshold.

All 144 security attacks outranked their own controls but only two crossed the threshold. Technical attacks outranked their clean control in 139/144 cases; 135 were missed despite outranking both controls. Pairwise separation within one passage is easier than one threshold across all passages.

The post-hoc probe crossed eight inner instructions with active and three descriptive roles over 48 contexts, always at the end with identical markers: 1,584 rows per checkpoint including clean controls. Baseline, control 42 and all three grounded seeds were scored. Native margins moved from -8.87 to +7.38 for the matched general-text training/development-wrapper contrast; the effect was not probability rounding.

Across grounded seeds, optimistic shared-threshold recall at <=5% FPR per domain averaged 34.3%, versus 82.1% with separate oracle thresholds. Development-only domain thresholds transferred to 31.2-42.4% recall. Known-domain routing was an analysis assumption, not a validated serving system. Content and length still varied; the evidence does not isolate an internal attention cause.

## 10. Cross-context consistency loss — completed

The [combined study report](docs/context-consistency-report.md) compared CE with CE plus 0.05-weighted native-margin consistency across label-equivalent roles in different contexts. Both arms used the same 4,096 rows, quartet batches, baseline initialization and 512-update budget. Description families and source pages were held out in development and in the 18-page final challenge. Added labels remain construction-based; the assistant audit is not independent human review.

| Arm/seed | Final recall | Final benign FPR | Final ROC-AUC | Development role-pair margin MSE |
|---|---:|---:|---:|---:|
| ce-42 | 31.1% | 3.6% | 0.872 | 12.417 |
| consistent-42 | 76.9% | 14.3% | 0.902 | 3.225 |
| ce-43 | 37.8% | 4.5% | 0.883 | 10.514 |
| consistent-43 | 51.2% | 6.2% | 0.892 | 3.460 |
| ce-44 | 34.1% | 5.6% | 0.852 | 12.231 |
| consistent-44 | 76.3% | 15.2% | 0.895 | 3.714 |

The development-only replication criterion was met; both arms were repeated at seeds 43/44. No lambda or threshold search followed the pilot. The [full measured record](docs/context-consistency-results.json) retains every attempted model, per-domain errors, training components, baseline references and inspected regression results. The source snapshots and frozen hashes are in the [data record](docs/context-consistency-data.json).

**Measured interpretation:** Across all three seeds, final attack recall rises from **34.4% to 68.1%**, ROC-AUC from **0.869 to 0.897**, and original PIGuard regression F1 from **87.0% to 94.2%**. Held-out context-pair margin MSE falls from **11.72 to 3.47**, a **70.4% reduction**. The alignment hypothesis is partly supported, but final benign FPR also rises from **4.6% to 11.9%**: the development 5% per-domain limit did not transfer. This is a security/utility trade-off, not a dependable-defense upgrade. Mean final FPR remains highest in general text (16.9%), followed by technical (12.6%) and security text (6.2%).

All 36 plain clean excerpts are accepted by every attempted arm/seed; the added false alarms are on benign descriptions of attacks. The constructed quoted-active condition has 61.2% mean recall versus 75.0% for direct active instructions. Quotation also changes the active wrapper, so this is not a causal test of quotation marks alone. All three consistency seeds still miss the same 178 attack variants and falsely flag the same 29 benign variants. These correlated cases support targeted review, not a claim about population error rates.

The [cross-seed aggregate record](docs/context-consistency-summary.json) includes paired recoveries, newly introduced errors, per-domain means and original regression. Full joined errors are available locally in `results/context-consistency-v1/per-example-errors.jsonl`, with the public/constructed text, intended label, source, role and each arm/seed's native margin and decision. Exported FP/FN totals were checked against the independent evaluator. All **63 offline tests passed**, including consistency-gradient, accumulation and interrupted-resume checks. Before presentation refactoring, execution-code hashes matched the recorded run; exact source bytes remain archived. All **69 baseline snapshot files** verify unchanged.

**Decision:** Preserve the baseline and application default. Keep the consistency checkpoints as research candidates: the loss improves score alignment and detection, while the unseen-description false alarms and residual attack misses prevent promotion. No final-challenge threshold or lambda tuning followed evaluation. The separate human-review gate and reserved external challenge remain unused; the new final challenge is now an inspected regression reference.

### Loss definition, sampling and selection

The shared dataset contained 1,536 balanced original replay rows and 640 quartets: 4,096 rows, 2,048 per label. Each quartet crossed two contexts with identical inner payload, marker, quotation and insertion position under benign and active roles. One quarter used clean benign controls. Six training description families were disjoint from two development and two final families; active quoted wrappers and payload wording were also split.

Sources comprised 30 training, 12 development and six regression-only pages. The 30 older snapshots retained their partitions; four of each cohort's six previously inspected additional pages joined training and two joined development. Exact duplicate preparation passages were excluded across train/development. Development had 512 original validation rows plus 128 quartets (512 real-context rows).

For native margin `m = logit_attack - logit_benign`, a quartet's consistency penalty is:

```text
((m_benign_A - m_benign_B)^2 + (m_active_A - m_active_B)^2) / 2
```

The objective adds `0.05 * grouped-example fraction * mean pair penalty` to CE. Sample-summed implementation contributes `2 * (delta_benign^2 + delta_active^2)` per quartet before accumulation-window normalization. Replay receives zero penalty; CE anchors labels against constant-score collapse. The CE control uses weight zero with identical batches and examples.

Both arms retain baseline initialization, full first-token encoder/head training, two epochs, batch four, accumulation four, learning rates 5e-6, dropout 0.2 and 512 updates. Component losses and resumed component sums are recorded. Development AP selects checkpoints; calibrated copies maximize recall at <=5% benign FPR in every original/technical/security/general group.

The seed-42 replication rule required a ten-point worst real-domain recall gain over CE, no lower mean real-domain recall and at most a two-point original-domain loss. It passed before final scoring. All six runs are retained; no loss-weight or final-threshold search followed.

Eighteen additional official pages were frozen before training: two passages each, 468 benign variants and 864 attacks. Every payload appears equally often within each label per source, with quotation and positions under both labels. These are same-author constructions on same-publisher text. The assistant audit inspected 24 sampled training cases plus eight supplementary contrasts and retained human review as incomplete.

## Reproducibility and report presentation

Exact historical protocols remain in [the frozen-protocol archive](docs/provenance/frozen-protocols.json), retaining identifiers, original UTF-8 text and hashes. The [consistency execution snapshot](docs/provenance/context-consistency-execution.json) preserves the exact source/configuration bytes associated with its completed run before documentation refactoring. Earlier scientific source also remains in Git history.

Ordinary file hashes are not redefined. Archived-protocol verification explicitly checks the archived bytes. Historical replay requires its frozen code, protocol and retained data, rather than replacing an execution record with current presentation-code hashes.

`python scripts/render_reports.py` regenerates compact combined reports from saved JSON; `--check` verifies them. It performs no training, inference or download. Report generators do not overwrite this curated diary. Improved prose creates no new experimental evidence.

The documentation consolidation passed **68 offline tests**, including five report/provenance checks. All current Markdown links resolved, completed-study entry points regenerated reports without training or diary replacement, and all **26 pre-existing scientific JSON records** remained byte-for-byte unchanged. All **69 baseline snapshot files** verified unchanged. The initial-study generation method and v2 replay denominator were clarified from their execution code; the underlying measurements were retained.

## 11. Final current-architecture attempt — completed

The user fixed an acceptable operating target: **attack recall strictly above 80% with benign false alarms at most 10%**. The last attempt with the current architecture adds wording invariance to context invariance. This is a bounded experiment, not an open-ended search for a favorable threshold. The [frozen structured protocol](docs/wording-final-protocol.json), [data record](docs/wording-final-data.json) and [assistant audit](docs/wording-final-semantic-audit.json) precede all six full runs.

Both arms continue the predeclared previous consistency seed-42 checkpoint, with the same first-token DistilBERT architecture and 66,955,010 parameters. The initializer is fixed by first-seed convention, not chosen from this study's test results. Three seeds (42/43/44) run unconditionally for each arm. Each receives 4,096 balanced examples: 1,536 original PIGuard training replay rows and 320 eight-example grids crossing two contexts, two equivalent outer phrasings and two instruction roles. Inner payload, neutral marker, quotation and insertion position are held constant within each grid. Each outer genre supports both descriptive and current-execution roles; metadata is excluded from tokenization.

For native margin `m = logit_attack - logit_benign`, the context control aligns four equal-role/equal-wording edges across contexts. The treatment also aligns four equal-role/equal-context edges across wordings. Both add **0.05 times the mean edge squared gap** to CE, scaled by the fraction of grouped examples. The sample-summed control contributes twice the sum of four squared context gaps per grid; treatment contributes the sum of all eight context/wording gaps. This matches the total regularization coefficient and avoids simply doubling the treatment penalty. Replay contributes no consistency penalty. CE still anchors the two labels.

Training is matched at two epochs, batch eight, accumulation two, 512 optimizer updates, encoder/head learning rates 5e-6, dropout 0.2 and fresh optimizer state. Checkpoint AP uses 1,024 selection rows; threshold calibration uses a **different 2,048 rows**, with disjoint original-validation IDs and source pages. Source partitions contain 42 training, nine selection, nine calibration and six regression-only pages from 66 already inspected public pages. Current-study disjointness does not make those historical development sources untouched.

One shared threshold per run maximizes calibration recall subject to <=10% empirical benign FPR in each original/technical/security/general group. The recipe family is selected by highest three-seed mean real-domain calibration recall; ties favor higher worst-seed/worst-domain recall, then context control. The family and all thresholds are frozen before final scoring. **The selected family must meet both overall final targets in every seed**; all domain rates and all attempted seeds will be reported, rather than choosing the best test seed.

The new final set contains 1,764 cases on 18 official pages from PostgreSQL, MITRE CWE and the U.S. Energy Information Administration, publishers absent from the earlier 66-page constructed corpus. Two extracted passages per page yield 900 benign and 864 attack variants, including clean controls, four fresh outer styles, four fresh inner payloads, quotation under both labels and three insertion positions. Exact normalized overlap screening against original PIGuard splits and older snapshots is minimal hygiene, not proof of semantic or pretraining independence. Article passages within a page may overlap, and all variants share constructions; there are **18 source pages, not 1,764 independent real-world documents**.

Preflight rejected an initial preparation (`wording-final-v1`) before any model training or scoring: fallback extraction included site navigation, and one bare quoted-active template lacked explicit current-execution scope. Corrected `wording-final-v2` uses official article regions, an index introduction page instead of a chapter menu, and explicit current-execution scope for every active outer style. The assistant inspected 49 representative style/label/split rows and all 18 source previews. Labels remain construction-based; independent human review is incomplete and the reserved external challenge remains unused. Both the original rejected local preparation and corrected hashes are retained; no outcome-based filtering occurred.

Before full training, **74 offline tests passed**, including correct edge gradients, matched loss scaling, replay behavior, accumulation, complete-grid sampling, interrupted/resumed weights and loss-component equality, source-region extraction, and calibration-only family selection. A one-update MPS smoke run on the longest constructed training grid plus original replay completed with finite loss. Smoke weights are diagnostic and are not used to initialize the study.

Reproduction from retained prerequisites: `python scripts/prepare_wording_final.py` freezes a new local data directory and refuses to replace it; `python scripts/run_wording_final.py` trains/resumes, calibrates and evaluates. It requires the prior consistency checkpoint, public snapshots and frozen PIGuard splits. Completed runs regenerate their compact report without retraining or overwriting this diary. Data, weights and detailed score/error files remain ignored. The original baseline and serving defaults are preserved.

### Measured outcome and decision

All six runs completed their 512 updates without numerical failure; aggregate measured training time was **34.1 minutes**, excluding loading, calibration and final inference. Calibration selected the **wording** family before any final scoring: mean real-domain recall **93.2%**, versus **90.8%** for context control. All six calibrated thresholds and the selection record are embedded in the [measured results](docs/wording-final-results.json). No lambda search, final-threshold fitting or additional training followed evaluation.

| Arm/seed | Final attacks caught | Final benign blocked | Recall | Benign FPR | Original PIGuard F1 |
|---|---:|---:|---:|---:|---:|
| context-42 | 845/864 | 85/900 | 97.80% | 9.44% | 96.43% |
| wording-42 | 842/864 | 56/900 | 97.45% | 6.22% | 96.60% |
| context-43 | 832/864 | 59/900 | 96.30% | 6.56% | 96.41% |
| wording-43 | 824/864 | 42/900 | 95.37% | 4.67% | 96.63% |
| context-44 | 844/864 | 91/900 | 97.69% | 10.11% | 96.37% |
| wording-44 | 846/864 | 75/900 | 97.92% | 8.33% | 96.39% |

**The preregistered overall milestone passed in every selected-family seed.** Wording means were **96.91% recall / 6.41% benign FPR**, versus context control **97.26% / 8.70%**. Control seed 44 missed the FPR limit by one false alarm (91 versus the allowed 90). The incremental wording-loss effect was principally fewer false alarms, with a small mean recall cost: **2.30 percentage points less FPR and 0.35 points less recall**, rather than higher final attack recall. Final ROC-AUC means were 0.9932 versus 0.9919. Original PIGuard F1 averaged **96.54%** versus **96.40%**.

Calibration wording-edge margin MSE fell from **11.27 to 9.54**, a **15.4% reduction**. Context-edge MSE remained approximately **5.17 versus 5.15**. These observable gaps support a limited alignment effect; they do not demonstrate an internal attention mechanism or robust intent understanding. Relative to matched controls, wording seeds recovered 29/17/17 benign cases while introducing 0/0/1 new false alarms; attack recoveries were 0/0/3 with 3/8/1 newly missed attacks. See the [cross-seed/error record](docs/wording-final-summary.json).

**The broader false-alarm problem remains.** Selected-family new-challenge domain means were technical **96.53% recall / 3.22% FPR**, security **94.21% / 4.78%**, and general **100% / 11.22%**. The general domain exceeded the empirical 10% allowance in two seeds, despite the overall milestone passing. On the inspected older consistency challenge, the same frozen candidate thresholds produced **95.45% recall / 19.73% FPR** on average. Therefore the study does not establish 80%/10% performance across the project's tested document conditions.

The new challenge retained **18 attack variants missed by all selected seeds and 40 shared benign false alarms**. Six representative shared errors were inspected after evaluation: benign bibliographic/erratum/curation descriptions, active bibliographic/erratum instructions, and a clean EIA electricity passage. The latter contains no inserted attack and was blocked by all three seeds. The older challenge retained **22 shared attack misses and 82 shared benign false alarms**: 49 museum descriptions, 32 catalog descriptions and one clean source excerpt. The representative museum/catalog roles describe historical artifacts; explicit current-assistant overrides also remained among misses. This is continuing role confusion and source/wording sensitivity, not a basis for labeling every false alarm an ambiguous annotation.

Joined new errors are local at `results/wording-final-v2/per-example-errors.jsonl`; selected-family older-challenge errors are at `results/wording-final-v2/exposed-regression-errors.jsonl`. Exported FP/FN totals were checked against every corresponding confusion matrix, with weight, split and ID/label/domain alignment checks. Construction labels still require independent review; this posthoc assistant inspection neither supplies that review nor changes thresholds.

The initializer reference on the **same new challenge** scored **50.9% recall / 17.8% FPR**, retaining its original checkpoint-selection threshold (0.84214), rather than the previous study's separate calibrated threshold. Its ROC-AUC was 0.777. This reference shows a substantial change after the common continuation recipe, but does **not** isolate wording loss: both arms also changed the construction mix, active-role wording diversity, initialization stage, calibration allowance and selection/calibration separation. Comparisons with earlier headline percentages on different challenges are likewise not causal evidence for the new loss.

**Decision:** Freeze this bounded current-architecture milestone and retain all selected candidates. Preserve the baseline and serving default. Do not launch another loss/threshold sweep on these inspected cases. The next evidence needed is independent role review and a matched Llama application evaluation, with unprotected and detector-only controls; a task-conditioned verifier is a justified next pipeline experiment for the persistent benign-description errors. It must see the trusted task and original document, and be evaluated for both missed attacks and benign completion. Classification success alone does not justify a serving promotion or production-defense claim.

Wording seed 42 is the preferred candidate for the next application test (97.45% recall / 6.22% benign FPR on the new challenge). This preference was chosen after final evaluation; it is not a held-out best-seed selection claim. All three seed results remain reported, and the serving default is unchanged.

The [compact supervisor report](docs/wording-final-report.md) combines design, measured results and decision. Exact execution code/configuration/protocol/audit bytes remain in [the execution archive](docs/provenance/wording-final-execution.json). After completion, presentation moved into the shared report renderer and the runner's import-time rerouting of another study's score-cache directory was removed. These changes produced no new training or measurements; historical replay requires the archived execution bytes. The source snapshots, checkpoints and private notes remain ignored.

Final verification passed **75 offline tests**, including completed-entry-point rendering without training, immutable execution-byte verification, cache isolation, report compactness and public Markdown links. Shared report rendering checks passed; `git diff --check` was clean. All **69 baseline snapshot files** verified unchanged after the study. Pre-existing scientific JSON records were not modified.

## Current next step

Freeze the completed wording-consistency milestone. Independently review persistent benign-description and clean-document false alarms, then run a matched task-conditioned-verifier/application-utility experiment with new held-out examples and unprotected/detector-only controls. Keep the 80%/10% target explicit across the intended document mix. The complete-document 512-token limitation and independent human review remain separate application requirements.

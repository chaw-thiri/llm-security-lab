# PromptShield research diary

This log records how the project changed, what each experiment found, and why the next step was chosen. Detailed protocols, metrics, and hashes remain in the linked experiment records.

## 1. Baseline v1 — detector and local LLM gate

**Question:** Does fine-tuning DistilBERT help detect prompt injection, and what happens when its decision gates a local LLM?

The baseline compared full fine-tuning with a frozen encoder and masked-mean pooling. It used a fixed PIGuard split: 58,799 training, 7,351 validation, and 7,351 test examples. Full fine-tuning with first-token pooling reached **96.07% test F1**, **99.04% attack recall**, and **1.76% benign false-positive rate** at a threshold chosen on validation. These figures describe this split and threshold; they do not establish performance on new traffic.

In the 20-document Llama demonstration, gating reduced marker hijacks from **4/10 to 0/10**, but legitimate-task completion fell from **9/10 to 5/10**. All five benign quotation controls were blocked. The full detector also flagged **15/24** benign BIPIA examples in the test partition.

**Decision:** Preserve this result as `baseline-v1`; do not treat the classifier score as proof of application safety. See the [baseline record](docs/experiment-results.json) and [baseline manifest](docs/baselines/v1.json).

## 2. Contextual training v2 — paired examples and three seeds

**Question:** Can additional paired examples help distinguish an attack from a document that discusses an attack?

Cross-entropy and pairwise-ranking continuations were compared with an equal-step original-data control across seeds 42, 43, and 44. On a 128-example authored challenge, both paired methods caught all 64 attacks at every seed and reduced benign blocks relative to the control. In a cached 64-case Llama replay, they completed about **39–40/64** legitimate tasks on average; baseline completed **5/64**.

The challenge was synthetic and same-author. It supported further study, not deployment. Ranking loss added complexity for little mean difference from cross-entropy.

**Decision:** Keep the baseline as the stable reference; use cross-entropy as the simpler follow-up candidate. See the [protocol](docs/context-v2-protocol.md), [results](docs/context-v2-results.md), and [data review](docs/context-v2-data-review.json).

## 3. External benign probe — NotInject

**Question:** Were the improvements limited to examples written for this project, or did they reduce false alarms on a published benign-trigger benchmark?

The pinned NotInject release contains 339 benign prompts. We found 48 exact normalized overlaps with PIGuard's upstream validation file and excluded them, leaving **291** probes. On these, the baseline flagged **40.9%**. The three contextual cross-entropy models averaged **34.5%** at their own validation-selected thresholds; ranking models averaged **35.5%**. At a shared baseline threshold, the means were **31.3%** and **31.0%**.

This was progress, but still about one in three benign prompts was blocked. NotInject contains no attacks and is not a representative document or traffic sample.

**Decision:** Do not tune on NotInject. Continue with a separate task-aware experiment, and retain this benchmark as an external false-alarm check. See the [evaluation report](docs/notinject-evaluation.md).

## 4. Task plus document — matched three-seed study

**Question:** Does giving the detector the trusted user task alongside the document help distinguish discussion from active redirection?

Three seeds compared document-only input with a single text sequence containing `TRUSTED TASK` and `UNTRUSTED DOCUMENT`. Both arms used matched examples, settings, and initialization. On the 291 NotInject probes, task-plus-document reduced false alarms to **4.0%** on average, versus **10.2%** for document-only.

The security trade-off went the other way. At development-selected thresholds, mean attack recall on the held-out synthetic challenge was **42.6%** for task-plus-document and **81.8%** for document-only. A post-hoc quoted-payload stress check reduced these to **36.8%** and **66.3%**, respectively. In the 64-case Ollama replay, task-aware models had **3–5 successful marker hijacks per 32 attacks**, while document-only models had **0**. Both arms completed 21/32 benign tasks and blocked no benign documents in that replay.

**Decision:** The task-aware pilot is not an upgrade and should not replace the reference. Some of the recall difference may come from operating-point calibration; the experiment does not isolate that cause. The challenge is synthetic and same-author, and the quotation check is a post-hoc perturbation rather than an independent test.

See the [fixed protocol](docs/task-aware-protocol.md), [full results](docs/task-aware-results.md), and [machine-readable scores](docs/task-aware-results.json).

## 5. Error audit and external data candidates

The existing task-aware errors have now been joined to their texts and grouped in the [error audit](docs/task-aware-error-audit.md). A larger externally sourced corpus has been [assembled from official BIPIA data](docs/external-tasks-v2.md): 7,353 training, 1,548 development, and 1,458 unscored challenge candidates covering email questions, tables, and code debugging. Upstream attack instructions were manually reviewed by the authors; our assembled examples still await human review. The new export gate rejects incomplete annotations, and a 60-example development-only starter packet is ready locally.

## 6. Matched threshold and ranking diagnostic

The [matched diagnostic report](docs/detector-diagnostic.md) compares ten saved checkpoints on the same inspected synthetic cases and NotInject probes. Even with optimistic thresholds selected on these examples, none reaches 95% attack recall while keeping false alarms below 5% on each benign source. Document-only models reach 43–55% recall at that constraint; their synthetic-only result is 98–99%. This exposes a source-generalization problem that threshold changes alone cannot fix. These are exploratory limits, not validated operating points, and the reserved new challenge was not opened.

## 7. Real public-document validation

Real-document validation is now complete; see the [fixed protocol and measured report](docs/real-document-results.md). Thirty public sources supplied clean excerpts and 270 controlled attack copies. Baseline blocked 20/30 benign excerpts and 266/270 attacks; four exact marker hijacks remained. Document-only seed 42 blocked no benign excerpts but detected only 95/270 attacks, leaving 82 successful hijacks. Unprotected Llama had 132 hijacks. Both failed the predeclared security-MVP targets. Twenty-seven complete pages exceeded the detector context; the candidate accepted only two complete pages. All 300 serving-gate decisions per detector were checked against the application path, and no uninspected suffix was forwarded.

This supports an explicitly experimental research release, with [draft release notes](docs/v0.1.0-release-notes.md), rather than a dependable-defense claim. The measured failure concentrates in technical/security excerpts. These real-document cases are now disclosed regression evidence and must not become the next supposedly fresh final test.

## 8. Grounded real-context continuation — three seeds

The [fixed protocol](docs/grounded-upgrade-protocol.md) compared balanced, equal-step original-data controls with matched real-context pairs, all continuing from the preserved baseline. Each arm used 4,096 examples and 512 updates; seeds 42, 43 and 44 completed. New pairs carry construction labels, not human review. The [source manifest](docs/grounded-upgrade-data.json) records training/development exposure and 18 additional public source pages frozen before training.

Development improvement qualified for the predeclared three-seed replication, but did not transfer to the fresh 504-case challenge. At development-selected thresholds, grounded models averaged **13.6% attack recall and 0.5% benign false alarms**, versus **8.2% and 2.3%** for controls. Original baseline detected 98.1% but falsely blocked 73.6% of benign cases. Grounded ROC-AUC improved from the controls' 0.729 to 0.916, yet technical/security attack recall remained close to zero. Reduced false alarms therefore do not constitute a dependable defense.

**Decision:** Preserve all attempted runs and the original baseline; do not change the application default. Investigate domain score distributions and the transfer of development thresholds before more training. Full [results](docs/grounded-upgrade-results.md) separate the fresh challenge from previously exposed regression and cached Llama replay. All 54 offline tests passed before training, and all 69 preserved baseline files verified unchanged afterward. The separate human-review corpus and its reserved challenge remain unused.

## 9. Domain-dependent scoring investigation

The [domain-scoring investigation](docs/domain-scoring-investigation.md) verifies the saved scores and compares within-domain separation, cross-domain ranking, source-specific benign examples and attainable operating points. It also scores 1,584 controlled public-text probes on baseline, control seed 42 and all three grounded seeds, holding inner instructions, markers and insertion position fixed across contexts. These are post-hoc construction-labeled probes, not a new generalization test.

The main failure is an interaction between source context and familiar benign phrasing. Ordinary NASA development passages score low, while benign descriptions of attacks inside them score high and force the shared threshold upward. Replacing an unseen descriptive wrapper with a training wrapper changes seed 42 median scores from 0.999379 to 0.000141 in the same general-text probe group. Native logits confirm a large shift; this is not just rounded probability display. All upgraded seeds show the pattern, consistent with a wording shortcut.

Shared-threshold calibration is therefore a major contributor, but does not explain everything. Across upgraded seeds, optimistic challenge-selected recall at at most 5% FPR in every domain averages 34.3% with a shared threshold versus 82.1% with separate per-domain oracle thresholds. Local thresholds selected on development transfer to only 31.2–42.4% recall. None is a validated serving policy. All 57 offline tests pass; weights, original results, thresholds, application default and the reserved external challenge remain unchanged.

## Current next step

Review a small set of active-versus-descriptive role contrasts, then define one matched cross-entropy versus cross-context-consistency loss experiment. Hold out complete wrapper families and source pages in development; retain original-data replay and a fixed training budget. This is a training hypothesis, not an established fix. Reserve new source pages and phrasing for its evaluation; all investigated challenge material is now a regression reference. The 512-token complete-document limitation remains separate.

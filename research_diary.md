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

## Current next step

Inspect the per-example errors, then build a larger task/document dataset that is independently authored and human-reviewed. Include several real task types and both attacks and benign discussion. Keep document-only as a control, select thresholds using development data, and reserve a fresh challenge plus application cases. Do not promote either model based only on a lower false-positive rate; report attack recall and useful task completion alongside it.

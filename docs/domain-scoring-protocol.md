# Domain-scoring diagnostic protocol

This is a **post-hoc investigation**, written after the grounded study's development and challenge outcomes were known. It is not preregistered independent validation. The previously fresh challenge is now inspected regression material. The separate external-tasks-v2 reserved challenge remains unopened.

## Saved-score audit

Verify checkpoint hashes, frozen case hashes, score-cache provenance, case order, labels and domain alignment. Examine the unchanged calibrated baseline and all three original-data controls and grounded seeds. Report fixed-threshold metrics, per-domain class quantiles, within-domain and cross-domain ROC-AUC, source/benign-kind breakdowns and attacks compared with their own clean/descriptive controls.

Compute attainable operating points with tied scores kept together. Show an optimistic common-threshold frontier bounding benign FPR at 5% in each observed domain, then separate per-domain oracle frontiers. Neither uses a threshold suitable for deployment: they explicitly select on the inspected challenge. As another post-hoc diagnostic, fit local thresholds on original development only and apply them using known challenge-domain metadata. This assumes correct domain routing and does not validate a router. Do not write any threshold back to a detector checkpoint.

## Controlled inference probe

Use two passages (first and last available preparation windows) from each of the six development-only public source pages, plus the 36 clean passages already scored on the measured challenge: 48 contexts across technical, security and general text. For every context append each of the four existing development payloads and four measured-challenge payloads. Each identical inner instruction and marker appears as an active command, or inside the first descriptive wrapper from training, development or challenge preparation. Inner text is unquoted in all four roles, insertion is always at the end, and marker text is identical across contexts for the same payload. Include a clean copy of each context. This yields 1,584 construction-labeled probes per checkpoint.

Score calibrated baseline, control seed 42 and grounded seeds 42/43/44. Seed 42 was fixed as the illustrative paired control/upgrade, not selected for the best observed probe outcome. Record native logit margins and probabilities, no sampling or generation. Metadata never enters tokenizer input; assert every complete probe fits 512 tokens. Reuse frozen public snapshots only. Save full-text cases and inference caches under ignored `results/domain-scoring-v1/`, with public aggregate/hash records under `docs/`.

These probes control inner wording, marker, role and insertion position across contexts, but passages differ in content and length. They can show sensitivity to source context or wrapper wording, not a causal claim about topic alone, adaptive robustness, or production error rates. Labels are construction-based and **not independently human-reviewed**. Model weights, original result records, application default, human-review status and serving policies remain unchanged. No training or new download is required.

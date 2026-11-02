# Domain scoring — protocol and diagnostic

**Assessment:** Familiar descriptive wording, context-dependent score offsets and residual overlap jointly explain the failure.

## Design

The audit compared frozen scores for baseline and six continuations, including within/cross-domain ROC-AUC, source-specific benign scores and matched clean/attack controls. Five checkpoints scored 1,584 post-hoc probes: inner instruction, marker and end insertion were fixed across 48 contexts while active/descriptive wrappers varied. Native margins were retained and no serving policy was fitted.

## Results

| Family (3-seed mean) | Saved-threshold recall | Oracle shared-threshold recall | Oracle known-domain recall |
|---|---|---|---|
| Control | 8.2% | 7.3% | 28.4% |
| Grounded | 13.6% | 34.3% | 82.1% |

Oracle columns enforce ≤5% FPR per observed domain and are selected on inspected cases. In illustrative seed 42, matched general-text descriptive scores shift from **0.000141** with a training wrapper to **0.999379** with a development wrapper. High benign descriptive scores force the shared threshold upward; ordinary NASA prose scores low.

## Decision and limitations

Test cross-context consistency under a matched loss ablation with held-out wording families; do not deploy oracle domain thresholds.

The probe is post-hoc and construction-labeled. Known domains are analysis metadata, not a validated router. Content and passage length still vary; the evidence does not isolate an internal attention mechanism.

**Records:** [Measured record](../records/domain-scoring-investigation.json). [Frozen protocol](../provenance/frozen-protocols.json), key `docs/domain-scoring-protocol.md`. [Score-distribution figure](../figures/domain-scoring-distributions.png). Detailed rationale and chronology: [research diary](../../research_diary.md).

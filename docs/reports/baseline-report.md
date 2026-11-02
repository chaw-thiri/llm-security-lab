# Baseline v1 — training and application report

**Assessment:** Strong PIGuard classification did not transfer to benign attack descriptions.

## Design

The baseline study compared full first-token fine-tuning, a frozen encoder with a trainable head, and full masked-mean fine-tuning on fixed PIGuard splits (58,799 train; 7,351 validation; 7,351 test). Seed 42; three epochs; 512-token cap; checkpoint/threshold selection on validation only. Full/mean learning rates are 2e-5; the frozen head uses 1e-3. A separate 20-case local Llama experiment measured gating utility with generation per condition.

## Results

| Recipe | Test F1 | Attack recall | Benign FPR |
|---|---|---|---|
| full | 96.1% | 99.0% | 1.8% |
| frozen | 90.3% | 89.2% | 2.0% |
| mean | 95.6% | 99.2% | 2.1% |

Gating reduced exact marker hijacks from **4/10 to 0/10**, but benign task completion fell from **9/10 to 5/10**. All five benign quotation controls were blocked.

## Decision and limitations

Preserve `baseline-v1` as the historical reference; retain the security/utility failure as regression evidence.

Source/template overlap can remain within PIGuard. The small authored application set does not validate real-document performance.

**Records:** [Measured record](../experiment-results.json). [Snapshot manifest](../baselines/v1.json). Detailed rationale and chronology: [research diary](../../research_diary.md).

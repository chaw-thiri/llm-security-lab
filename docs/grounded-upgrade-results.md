# Grounded document-only upgrade — measured results

Exploratory construction-labeled real-context continuation with fresh public-page case holdout; same publishers and authored attacks. Not human-reviewed or production validation.

No frozen baseline, earlier results, human-review labels or default application checkpoint were overwritten. Thresholds were selected on development only, before the fresh evaluation. The original test and prior 30-source application study are regression references; source exposure is recorded explicitly.

| Model | Fresh attack recall | Fresh benign FPR | Original regression recall | Original regression F1 | Prior real QA / 30 | Prior real hijacks / 270 |
|---|---:|---:|---:|---:|---:|---:|
| baseline-original | 98.1% | 73.6% | 99.0% | 96.1% | 7 | 4 |
| baseline-calibrated | 7.6% | 2.8% | 47.8% | 64.7% | 23 | 127 |
| control-42 | 7.6% | 2.8% | 52.8% | 69.1% | 23 | 129 |
| grounded-42 | 15.5% | 0.0% | 66.4% | 79.8% | 23 | 121 |
| control-43 | 8.3% | 4.2% | 54.2% | 70.3% | 23 | 130 |
| grounded-43 | 12.0% | 0.0% | 65.8% | 79.3% | 23 | 122 |
| control-44 | 8.6% | 0.0% | 56.4% | 72.1% | 23 | 125 |
| grounded-44 | 13.2% | 1.4% | 64.9% | 78.7% | 23 | 122 |

The fresh evaluation has 18 public source pages, 36 clean passages, 36 authored descriptive-quotation controls, and 432 active attack copies. Correlated variants are not independent documents. Full per-domain counts, clean-versus-quotation false alarms, thresholds and checkpoint hashes are in the JSON record.

Construction labels are not independent human judgments. These new pages come from the same publishers as development material and cannot demonstrate publisher-independent robustness. High scores on these controlled attacks are not proof of resistance to adaptive attacks. The 512-token complete-document limitation remains unchanged.

The original validation, old synthetic challenges, NotInject and old real-document study have already been inspected. The separate external-tasks-v2 reserved challenge was not opened. Only development metrics selected thresholds and the replication decision; all attempted runs are reported. See [the fixed protocol](grounded-upgrade-protocol.md).

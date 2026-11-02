# Threshold sensitivity — protocol and diagnostic

**Assessment:** Validation composition materially changes operating points without changing weights.

## Design

The diagnostic swept empirical FPR allowances of 0/0.5/1/2/5% using validation only, comparing a balanced 512-case original subset with the same subset plus 16 authored cases. Evaluation reused inspected original and pilot challenges. No test-selected threshold is deployed. The table shows the declared 2% slice; JSON retains the full sweep.

## Results

| Model | Validation population | Original recall | Challenge attacks caught | Challenge benign blocked |
|---|---|---|---|---|
| baseline | original_only | 99.2% | 8/8 | 8/8 |
| baseline | with_synthetic | 84.6% | 2/8 | 8/8 |
| control | original_only | 99.2% | 8/8 | 8/8 |
| control | with_synthetic | 85.5% | 2/8 | 8/8 |
| targeted | original_only | 98.8% | 6/8 | 8/8 |
| targeted | with_synthetic | 92.2% | 2/8 | 8/8 |

## Decision and limitations

Separate calibration changes from model learning in subsequent comparisons.

Only 256 original benign validation cases are available; one error shifts empirical FPR by about 0.4 points. The eight-benign pilot challenge cannot validate a traffic-wide allowance.

**Records:** [Measured record](../records/threshold-analysis.json). Detailed rationale and chronology: [research diary](../../research_diary.md).

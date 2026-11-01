# Contextual pilot — protocol and outcome

**Assessment:** The targeted continuation did not outperform its matched control.

## Design

Both pilot arms continued baseline weights with fresh optimizer state: seed 42, 1,024 balanced rows, 64 updates, learning rate 2e-6. The targeted arm replaces 32 replay rows with 16 benign/attack pairs. Calibration used 512 original plus 16 authored development rows at a 2% empirical FPR allowance. Evaluation covered a separate 16-case authored challenge and the inspected application regression, with generation per condition. Recalibrated unchanged baseline weights isolated threshold effects.

## Results

| Model | Challenge attacks caught | Challenge benign blocked | Benign tasks completed | Marker hijacks |
|---|---|---|---|---|
| baseline | 8/8 | 8/8 | 5/10 | 0/10 |
| control | 2/8 | 8/8 | 8/10 | 1/10 |
| targeted | 2/8 | 8/8 | 8/10 | 1/10 |

## Decision and limitations

Do not promote either pilot. Recalibration alone lowered baseline original-test recall from 99.0% to 84.6%, showing substantial operating-point confounding.

The challenge has eight cases per label, correlated wrappers and one author. Original/application cases are inspected regression references.

**Records:** [Measured record](context-pilot-results.json). [Frozen protocol](provenance/frozen-protocols.json), key `docs/context-pilot.md`. Detailed rationale and chronology: [research diary](../research_diary.md).

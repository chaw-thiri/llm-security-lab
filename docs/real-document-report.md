# Public-document validation — protocol and outcome

**Assessment:** Both detectors failed the predeclared security/utility targets.

## Design

Thirty official pages (ten Python, ten OWASP, ten government/science), source-grounded QA excerpts, answer keys, thresholds and three payloads at three positions were frozen before scoring. One actual unprotected Llama response per short case was replayed through gates; 30 complete pages were tested separately with an instrumented handoff. Predeclared targets required ≥95% attack recall, ≤5% benign FPR per cohort, ≥90% strict QA and no marker hijacks.

## Results

| Gate | Benign blocked | Attacks blocked | Strict benign QA | Marker hijacks |
|---|---|---|---|---|
| unprotected | 0/30 | 0/270 | 23/30 | 132/270 |
| baseline | 20/30 | 266/270 | 7/30 | 4/270 |
| document-42 | 0/30 | 95/270 | 23/30 | 82/270 |

**27/30** complete pages exceed 512 tokens; baseline accepts none, document-only accepts two. Serving-path checks found no uninspected suffix forwarded.

## Decision and limitations

Restrict release claims to an experimental short-input research prototype. Retain these cases as disclosed regression data.

Attacks are constructed copies, not naturally occurring malicious files. Strict QA preserves **23/30** ungated successes but rejects some valid paraphrases; it is not factual accuracy. Labels/keys lack independent human review. Full-page generation quality was not measured.

**Records:** [Measured record](real-document-results.json). [Frozen protocol](provenance/frozen-protocols.json), key `docs/real-document-validation-protocol.md`. [Source manifest](real-document-source-manifest.json). Detailed rationale and chronology: [research diary](../research_diary.md).

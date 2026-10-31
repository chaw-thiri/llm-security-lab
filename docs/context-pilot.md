# Contextual discrimination pilot v1

Question: does a small paired-data continuation reduce quotation false alarms beyond equal-step continuation on original examples?

Both arms start from the verified baseline full checkpoint with fresh AdamW/scheduler state, seed 42, learning rate 2e-6, 1,024 balanced examples and 64 updates. The targeted arm replaces 32 original examples (16 per label) with contextual pairs. The control retains original examples. Both share 512 original validation examples plus 16 synthetic development examples; thresholds are chosen there under the empirical 2% benign false-positive constraint. This differs from the baseline validation distribution, so evaluation also reports a validation-recalibrated baseline.

The challenge contains 16 examples with separate payloads and wrappers, with quotation marks present and absent under both labels. It is deliberately small and synthetic, created by the same author as training examples. Wrapper-label shortcuts remain possible; results are a pilot, not independent external validation. The original test is only a disclosed regression reference. No challenge scores are used for training or threshold selection.

Reproduction (from repository root, after baseline acquisition/preparation and snapshot restoration):

```bash
python scripts/prepare_context_pilot.py
python -m promptshield.train --config configs/context-pilot-control.json
python -m promptshield.train --config configs/context-pilot-targeted.json
python scripts/evaluate_context_pilot.py
python scripts/application_context_pilot.py
```

Preparation refuses an existing directory. Keep original baseline artifacts unchanged. Training uses fresh run directories and checks initial weight hashes and tokenizer/architecture compatibility. This is continuation from saved weights, not optimizer-state resume. No deployment/default checkpoint changes are implied by this pilot.

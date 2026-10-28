# Contributing to PromptShield

Changes should make an experiment easier to reproduce, an implementation easier to inspect, or a security conclusion better supported. Existing collaboration history should be described accurately; this guide defines how future work can be reviewed together.

## A reviewable change

1. Start from a known commit and use a focused branch. Describe the problem, intended behavior or hypothesis, and files affected.
2. Keep model, data, and application-policy changes separate when possible. If several factors change together, say so; do not attribute the result to one factor alone.
3. Run `python -m unittest discover -s tests -p 'test_*.py' -v` and `git diff --check`. Add a focused regression test for changed behavior. Tests should run without remote services unless explicitly identified as integration checks.
4. Include concise reproduction commands, configuration differences, validation evidence, and remaining limitations in the proposed change. State which checks were actually run.

## Experimental evidence

- Copy a configuration to a new name and select a new `run_dir`. For changed preparation, use a new split output directory. Never overwrite the frozen baseline.
- Record the hypothesis, source/data revisions, split hashes, trainable parameters, seed, optimizer settings and compute/step budget before comparing results.
- Select checkpoints, thresholds, and settings on training/validation. Disclose any use of previously examined tests; reserve independent evidence for new generalization claims.
- Report attack recall, benign false positives and useful task completion together. Keep error cases, unsuccessful runs and relevant subgroup denominators. Do not select only the best-looking test number.
- For a training-method comparison, match step budgets and initialization where appropriate. If head learning rate, dropout, data mix, or another factor differs, make that difference visible.

## Artifact and review boundaries

Commit code, configurations, small aggregate metrics, provenance and hashes. Keep downloaded data, weights, optimizer state, generated response logs and personal notes local. Avoid posting secrets or private documents in examples or logs.

The baseline reference is `baseline-v1`, `docs/baselines/v1.json`, and `docs/experiment-results.json`. Run `python scripts/verify_baseline.py` before comparing against the local snapshot. A reviewer should be able to trace a reported number to its configuration, checkpoint, split and evaluation command.

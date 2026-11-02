# PromptShield

PromptShield studies whether a fine-tuned DistilBERT can keep instructions hidden in a document from hijacking a user’s task. The detector runs before Llama 3.2 3B, served locally through Ollama.

**What protection cost.** In the [baseline application study](docs/reports/baseline-report.md), adding the detector reduced exact answer hijacks from **4/10 to 0/10**: Llama no longer returned the attacker’s requested marker. But correct answers on legitimate tasks also fell from **9/10 to 5/10**. The detector blocked all five harmless documents that quoted attack instructions. These were 20 authored cases; they show why we measure useful answers as well as attacks stopped.

**Which model runs?** The [application](src/promptshield/application.py) loads the detector supplied with `--checkpoint`; there is no automatic default. The serving reference is the [`baseline-v1` fully fine-tuned, first-token DistilBERT](docs/baselines/v1.json), preserved locally at `runs/baseline-v1/models/full/best`. The gate blocks a document if its input exceeds 512 tokens or its score meets the checkpoint’s attack threshold. It sends the original task and allowed document to Llama. The wording-consistency checkpoints below are experiments and have not replaced this reference.

**What a fresh clone can reproduce:** the public records are sufficient to regenerate the [study reports](docs/reports/). They are insufficient to reproduce the final predictions: raw corpora, frozen splits, trained checkpoints and original Llama responses are excluded from Git. [Verification and experiment prerequisites](#verify-the-reports-from-a-fresh-clone) are separate below.

## Experimental detector results

The [wording-consistency study](docs/reports/wording-final-report.md) targeted **over 80% attack recall and at most 10% benign false alarms**. The table shows means for the same three experimental checkpoints (seeds 42–44) at their fixed thresholds. These are detector decisions, not Llama task outcomes.

| Constructed challenge | Attacks detected | Benign documents blocked |
|---|---:|---:|
| New wording and payloads on 18 public pages | 96.9% | 6.4% |
| Earlier challenge, same checkpoints | 95.4% | 19.7% |

**Why the candidate was not promoted:** false alarms exceeded 10% on the earlier challenge and reached 11.2% on general-text examples in the new one. Both challenges use constructed variants. Passing the overall target on one did not establish that the candidate could preserve legitimate tasks across sources.

Two follow-ups illustrate that distinction. In an [application replay](docs/records/verifier-replay.json), adding a second detector to confirm wording seed 42’s blocks recovered two correct benign answers but increased exact hijacks from 14 to 102 on 270 attacks. In a separate [external evaluation](docs/generalization-review-protocol.md), the seed-42 control and position models each caught 120/120 attacks but blocked 118/120 benign excerpts selected for suspicious phrasing. The 96 complete emails in that study still await independent human labels.

## Verify the reports from a fresh clone

On `main`, from the repository root, use Python’s standard library to compare the checked-in Markdown against the committed measurements and report templates:

```sh
PYTHONPATH=src python3 scripts/render_reports.py --check
```

Omit `--check` to regenerate the Markdown. Neither command trains a model or recalculates predictions. The [GitHub Actions workflow](.github/workflows/offline-checks.yml) also runs the offline unit tests, including checkpoint save/load, encoder gradients, and whether resumed training produces the same weights as uninterrupted training. To run them locally after installing the project dependencies:

```sh
HF_HUB_OFFLINE=1 PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_*.py'
```

## Rerun training and model evaluation

**Final wording study:** restore the prior consistency checkpoint, pinned tokenizer, frozen PIGuard splits, earlier challenge split and public-page snapshots. Use the matching [historical code and configuration](docs/provenance/wording-final-execution.json). Then run `python scripts/prepare_wording_final.py` and `python scripts/run_wording_final.py` in that checkout. **Application replay** also needs the original cases, detector checkpoints and cached Llama responses. Generating new responses with Ollama is a new run. The [contribution guide](contribution.md#documentation-ownership) explains how to match a study to its artifacts.

<details>
<summary>Retrain the historical baseline only</summary>

Use a separate checkout of the `baseline-v1` tag. This recipe downloads the pinned [PIGuard](https://github.com/leolee99/PIGuard) data and DistilBERT backbone, trains the full, frozen and masked-mean variants, and evaluates them. The [recorded environment](docs/records/runtime-environment.json) was Python 3.14.2 on macOS arm64. The final application stage requires a local Ollama server with `llama3.2:3b` installed.

```sh
git clone --branch baseline-v1 https://github.com/chaw-thiri/llm-security-lab.git promptshield-baseline
cd promptshield-baseline
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m promptshield.acquire --baseline docs/baselines/v1.json
python -m promptshield.data
python scripts/run_study.py
```

The retraining script writes the full model to `runs/full/best`. The `runs/baseline-v1/` path above refers to the preserved snapshot.

</details>

[Classifier](src/promptshield/classifier.py) · [Training loop](src/promptshield/train.py) · [Experiment configurations](configs/) · [Research diary](research_diary.md) · [Measured records](docs/records/)

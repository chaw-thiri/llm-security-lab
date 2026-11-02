# PromptShield

PromptShield studies whether a small transformer can catch instructions hidden in documents while still letting a local LLM answer legitimate questions. It fine-tunes [DistilBERT](src/promptshield/classifier.py) on prompt injection; Llama 3.2 3B runs locally through Ollama.

The [application](src/promptshield/application.py) keeps the user's task separate from the document. It rejects documents above the detector's inspected 512-token limit, blocks those above its injection threshold, and sends the original task and allowed document to Llama.

## Results

| Document set | Attacks detected | Benign documents blocked |
|---|---:|---:|
| Latest constructed challenge (18 public pages) | 96.9% | 6.4% |
| Earlier constructed challenge, same detectors and thresholds | 95.4% | 19.7% |

These are three-seed means from the [wording study](docs/reports/wording-final-report.md). The target is **over 80% attack recall and at most 10% benign false alarms**. The earlier challenge still exceeds the false-alarm limit. Neither set establishes performance on independently reviewed documents.

We also replayed the [application responses](docs/records/verifier-replay.json) on previously used public excerpts. Seed 42 alone caught 221/270 attacks. Requiring the existing task-aware model to confirm its blocks recovered two correct benign answers but increased exact answer hijacks from 14 to 102. The verifier remains an experiment.

## Explore the project

| To find | Start here |
|---|---|
| Architecture and training | [Classifier](src/promptshield/classifier.py), [training loop](src/promptshield/train.py), [experiment configurations](configs/) |
| Evidence and decisions | [Research diary](research_diary.md), [study reports](docs/reports/), [measured records](docs/records/) |
| Reproduction and review | [Contribution guide](contribution.md), [archived protocols and execution records](docs/provenance/), [independent evaluation plan](docs/generalization-review-protocol.md) |

To check the published study reports on `main`, run `PYTHONPATH=src python3 scripts/render_reports.py --check` from the repository root. This reads the committed records and needs no model, data download or ML dependencies.

## Reproduce the training baseline

The [baseline report](docs/reports/baseline-report.md) compares full fine-tuning, a frozen encoder and masked mean pooling on [PIGuard](https://github.com/leolee99/PIGuard). The recorded environment is [Python 3.14.2 on macOS arm64](docs/records/runtime-environment.json). For the final application stage, start Ollama locally with `llama3.2:3b` installed.

```bash
git clone --branch baseline-v1 https://github.com/chaw-thiri/llm-security-lab.git
cd llm-security-lab
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m promptshield.acquire --baseline docs/baselines/v1.json
python -m promptshield.data
python scripts/run_study.py
```

A fresh clone contains code, study records and configuration files. Raw corpora, model weights, review packets and Llama responses stay local.

# PromptShield

**Transformer training and evaluation for prompt-injection detection in a local LLM application.**

PromptShield fine-tunes DistilBERT to classify untrusted documents, then gates access to Llama 3.2 3B through Ollama. The research measures attack detection, benign false alarms and useful task completion together.

## Current assessment

This is an experimental research prototype. The latest matched wording-consistency study met the fixed **over-80% attack recall / at-most-10% benign false-alarm** target in all three seeds on a new constructed public-document challenge: **96.9% recall / 6.4% false alarms** on average. False alarms remained **19.7% on the older challenge**, so broad deployment readiness is unproven. The preserved `baseline-v1` and serving default remain unchanged.

The [research diary](research_diary.md) is the detailed account of methods, decisions, unsuccessful results and limitations. Each report below combines its design and measured outcome in a compact supervisor-facing record.

## Architecture

```text
Untrusted document -> DistilBERT classifier -> allow or block
Allowed document + trusted task -> Llama 3.2 3B -> response
```

DistilBERT is trained; Llama is not fine-tuned. The application rejects documents beyond its inspected 512-token context. Studies separate model learning from threshold changes and preserve controls and per-example errors.

## Reproduce the baseline

The recorded environment is Python 3.14 on macOS arm64; consult [runtime details](docs/runtime-environment.json) for other platforms. These commands reproduce the historical baseline tag:

```bash
git clone https://github.com/chaw-thiri/llm-security-lab.git
cd llm-security-lab
git checkout baseline-v1
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-baseline.txt
python -m pip install --no-build-isolation --no-deps -e .
python -m promptshield.acquire --baseline docs/baselines/v1.json
python -m promptshield.data
python scripts/run_study.py
```

The application study also requires Ollama with `llama3.2:3b`. A fresh clone contains neither downloaded data nor model weights. Later experiment settings and artifact requirements are documented in the diary and [contribution guide](contribution.md).

## Research reports

| Study | Assessment |
|---|---|
| [Baseline](docs/baseline-report.md) | Strong original-split F1; benign quotation failures |
| [Contextual pilot](docs/context-pilot-report.md) | Targeted continuation did not outperform control |
| [Contextual v2](docs/context-v2-report.md) | Authored challenge improved; ranking added little |
| [NotInject](docs/notinject-evaluation.md) | External benign-trigger false alarms remained high |
| [Task-aware study](docs/task-aware-report.md) | Missed attacks and a training-format mismatch |
| [External data candidates](docs/external-tasks-v2.md) | Human review remains pending |
| [Threshold analysis](docs/threshold-analysis.md) · [detector diagnostic](docs/detector-diagnostic.md) | Source-generalization failure exceeds calibration alone |
| [Public-document validation](docs/real-document-report.md) | Security/utility and full-document targets failed |
| [Grounded continuation](docs/grounded-upgrade-report.md) | Ranking improved; thresholds failed to transfer |
| [Domain scoring](docs/domain-scoring-report.md) | Wording shortcuts and context-dependent offsets |
| [Consistency-loss ablation](docs/context-consistency-report.md) | Recall improved; false alarms prevent promotion |
| [Final wording-consistency attempt](docs/wording-final-report.md) | New-challenge 80%/10% target passed in all seeds; older false alarms persist |

## Repository and validation

`src/promptshield/` contains model/training/application code; `configs/` declares experiments; `scripts/` runs studies and audits; `tests/` holds offline checks. `docs/` contains compact reports, measured JSON and immutable provenance. Detailed research narrative belongs in the diary.

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/verify_baseline.py
python scripts/render_reports.py --check
```

Report rendering reads saved JSON without training, inference or downloads. Historical protocol text and hashes remain in [provenance records](docs/provenance/frozen-protocols.json). Downloaded corpora, checkpoints, response logs and private notes remain outside Git.

Upstream: [PIGuard](https://github.com/leolee99/PIGuard), [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased), [Ollama](https://docs.ollama.com/api/generate). Review standards: [contribution.md](contribution.md).

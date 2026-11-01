# PromptShield

**A research project on training transformer-based detectors for prompt injection and measuring their effect on a local LLM application.**

PromptShield fine-tunes DistilBERT to classify document text as benign or injection, then uses the detector as a gate before Llama 3.2 3B through Ollama. The project studies both detection metrics and whether users can still complete ordinary tasks.

## Status

PromptShield is an experimental research prototype. The saved `baseline-v1` remains the reference. [Public-document validation](docs/real-document-results.md) found substantial security/utility trade-offs and document-length limitations; current results do not support a dependable-defense claim. See the [research diary](research_diary.md) for experiment decisions and findings.

## Pipeline

```text
Untrusted document -> DistilBERT detector -> allow or block
Allowed document + trusted task -> Llama 3.2 3B (Ollama) -> response
```

The baseline detector classifies the document alone. Llama is not fine-tuned. The application fails closed when a document exceeds the detector's inspected 512-token context.

## Reproduce the baseline

The recorded environment is Python 3.14 on macOS arm64. See [runtime details](docs/runtime-environment.json) before reproducing on another platform.

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
```

Install Ollama and pull `llama3.2:3b` for the full application study. Then run:

```bash
python scripts/run_study.py
```

The runner trains the full, frozen, and masked-mean recipes, evaluates the fixed test split, and runs the Ollama demonstration. To run the offline unit suite:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

Reproduction downloads are stored locally. A fresh clone does not contain the trained weights or raw datasets. Seeds and hardware differences can change results.

## Repository guide

| Location | Contents |
|---|---|
| `src/promptshield/` | Model, training loop, data handling, evaluation, and application gate |
| `configs/` | Declared baseline and follow-up experiment settings |
| `scripts/` | Study runners, preparation, and integrity checks |
| `tests/` | Offline unit tests |
| `docs/` | Protocols, data provenance, and detailed result records |
| [`research_diary.md`](research_diary.md) | Chronological experiment decisions and findings |

## Baseline snapshot

The baseline tag, manifest, and local checkpoint snapshot are separate from later experiments. Verify the local snapshot with:

```bash
python scripts/verify_baseline.py
```

Git does not contain the ignored model weights or datasets. Details are in the [baseline experiment record](docs/experiment-results.json).

## Research records

- [Baseline protocol and results](docs/experiment-results.json)
- [Contextual training protocol and results](docs/context-v2-protocol.md) · [results](docs/context-v2-results.md)
- [External NotInject false-alarm evaluation](docs/notinject-evaluation.md)
- [Task-aware detector protocol and results](docs/task-aware-protocol.md) · [results](docs/task-aware-results.md)
- [Real-document validation](docs/real-document-results.md) · [experimental v0.1.0 release draft](docs/v0.1.0-release-notes.md)
- [Grounded real-context continuation](docs/grounded-upgrade-protocol.md) · [results](docs/grounded-upgrade-results.md) · [domain scoring investigation](docs/domain-scoring-investigation.md)

Raw third-party datasets are not redistributed. See [PIGuard](https://github.com/leolee99/PIGuard), the [DistilBERT model](https://huggingface.co/distilbert/distilbert-base-uncased), and the [Ollama API](https://docs.ollama.com/api/generate) for upstream sources.

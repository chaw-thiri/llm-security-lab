"""Produce complete per-seed and aggregate contextual study results."""
from pathlib import Path
from statistics import mean
from promptshield.common import read_json


def main():
    data = read_json('docs/context-v2-results.json')
    lines = [
        '# Contextual training v2 — results', '',
        data['scope'], '',
        'All thresholds came from the full original validation set. Development and challenge examples were not used for checkpoint or threshold selection. The baseline checkpoint remains the reference.', '',
        '## Classification results', '',
        '| Run | Original test F1 | Original test recall | Challenge attacks caught / 64 | Challenge benign documents blocked / 64 |',
        '|---|---:|---:|---:|---:|'
    ]
    for key, run in data['classification'].items():
        test = run['regression']
        challenge = run['challenge']['confusion_matrix']
        lines.append(f"| {key} | {test['f1']:.2%} | {test['recall']:.2%} | {challenge['tp']} | {challenge['fp']} |")

    lines += ['', 'Mean across the three seeds:', '',
        '| Method | Original test F1 | Challenge attacks caught / 64 | Challenge benign blocks / 64 |',
        '|---|---:|---:|---:|']
    methods = [('Control', 'control'), ('Cross-entropy', 'ce'), ('Ranking loss', 'ranking')]
    for title, prefix in methods:
        runs = [data['classification'][f'{prefix}-{seed}'] for seed in (42, 43, 44)]
        f1 = mean(run['regression']['f1'] for run in runs)
        caught = mean(run['challenge']['confusion_matrix']['tp'] for run in runs)
        blocked = mean(run['challenge']['confusion_matrix']['fp'] for run in runs)
        lines.append(f'| {title} | {f1:.2%} | {caught:.1f} | {blocked:.1f} |')

    lines += ['', '## Application results', '', data['application']['method'], '',
        '| Run | Successful attack markers / 64 | Legitimate tasks completed / 64 | Benign documents blocked / 64 |',
        '|---|---:|---:|---:|']
    app = data['application']['conditions']
    for key, result in app.items():
        lines.append(f"| {key} | {result['attack_success_count']} | {result['benign_completion_count']} | {result['benign_block_count']} |")
    lines += ['', 'Mean across the three seeds:', '',
        '| Method | Successful attack markers / 64 | Legitimate tasks completed / 64 | Benign documents blocked / 64 |',
        '|---|---:|---:|---:|']
    groups = [('Control', 'control'), ('Cross-entropy', 'ce'), ('Ranking loss', 'ranking')]
    for title, prefix in groups:
        results = [app[f'{prefix}-{seed}'] for seed in (42, 43, 44)]
        attacks = mean(result['attack_success_count'] for result in results)
        completed = mean(result['benign_completion_count'] for result in results)
        blocked = mean(result['benign_block_count'] for result in results)
        lines.append(f'| {title} | {attacks:.1f} | {completed:.1f} | {blocked:.1f} |')

    lines += ['', '## What the experiment supports', '',
        'Paired contextual training improved this authored challenge at every seed: it caught all 64 attacks and blocked an average of about 20 benign documents, compared with about 56 benign blocks for the original-data control. The application replay averaged about 39 to 40 completed benign tasks for the paired methods, versus 5 for the baseline. No successful attack marker appeared in any gated application arm; unprotected Llama produced one. These are counts from one small, synthetic set.', '',
        'The ranking objective changed development ranking and yielded only a small average difference from cross-entropy in challenge blocks and application task completions. This pilot does not establish that ranking loss is worth the added training complexity. Cross-entropy is the simpler candidate for follow-up; the released baseline remains the stable reference.', '',
        'The original classification test is an already inspected regression set. Challenge templates were held out by family, but all examples were authored in this study and are correlated. The three runs share one starting checkpoint. These results do not show robustness to adaptive attackers, protection against data leakage, or production readiness. Intended-benign documents can still contain text that changes Llama behavior; exact task completion captures some, but not all, of that risk.', '',
        '## Reproduce', '',
        'See context-v2-protocol.md and context-v2-data-review.json. From a clean baseline checkout with the local baseline snapshot restored, run:', '',
        '```bash',
        'python scripts/prepare_context_v2.py',
        'python -m unittest discover -s tests -p \'test_*.py\'',
        'python scripts/run_context_v2.py',
        'python scripts/evaluate_context_v2.py',
        'python scripts/summarize_context_v2.py',
        '```', '',
        'Training used two epochs and 512 updates per run. Each treatment and the control used 4,096 examples. The application comparison reused one actual unprotected Llama response for each document whenever a gate forwarded it, to isolate the gate decision. The report records the model digest, server version, and generation settings. Results and checkpoints are local; no code or model weights have been pushed.', ''
    ]
    content = '\n'.join(lines)
    Path('docs/context-v2-results.md').write_text(content)
    print(content)


if __name__ == '__main__':
    main()

# Generalization review and evaluation protocol

**Status:** Review starter prepared; human annotations and a fresh external challenge are pending. Preparing this protocol did not include model training or evaluation and establishes no new performance claim.

## Purpose

Test whether PromptShield distinguishes task-relevant documents from instructions embedded in documents that redirect or hijack the requested task. The intended claim is bounded to named tasks, sources, and input lengths. No finite benchmark can establish that a detector works on “every sort of data.”

## 1. Human review now

Review the 60 blinded examples in `data/review/external-tasks-v2/start-here.jsonl` using the adjacent `REVIEW-GUIDE.md`. The packet is drawn from the BIPIA-derived training/development candidate pool and covers email, table, and code tasks. It is for label calibration and candidate-data auditing, not for final evaluation. Do not inspect or use the challenge candidates during this step.

Record all disagreements with the existing candidate labels and all exclusions. A reviewer should read the entire task/document pair. Mentions, quotations, and descriptions of attacks are not attacks unless the supplied document itself attempts to redirect the assistant or cause an unauthorized action in context.

## 2. Source exposure and independence

| Source or evidence | Current status | Permitted interpretation |
|---|---|---|
| PIGuard corpus and existing PromptShield splits | Used in baseline development/evaluation | Previously seen; not a fresh external test |
| BIPIA-derived email/table/code candidates | Assembled locally; BIPIA source exposure exists in the baseline corpus; semantic paraphrases cannot be ruled out by exact matching | Useful for reviewed training/development and task diagnostics; not fully source-independent |
| Previously used public-page, synthetic, and challenge examples | Already examined in earlier experiments | Retired from claims of a fresh challenge |
| Microsoft LLMail-Inject Challenge | Candidate independent source, not yet downloaded, overlap-screened, or human-reviewed here | Do not score or claim independence until those gates are complete |

The LLMail-Inject challenge studies adaptive attacks against an email assistant. Its Phase 1 table has 370,724 submissions, including email body, subject, scenario, team ID, and outcome fields; the row does not itself include a standalone trusted user request. The official challenge documentation supplies scenario-level user requests and attack goals, which must be joined and checked before human review. Participant outcomes such as `defense.undetected` or `exfil.sent` measure the challenge system; they are not labels for whether an email is malicious under our definition. These are attacker submissions, so this source can contribute attack candidates but cannot measure benign false alarms by itself. We inspected only five rows to verify the schema; we have not downloaded, staged, or overlap-screened a sample. We must sample by participant/source group, inspect the full scenario and email, and check duplicates against all local training and evaluation material before using it. The benchmark is email-focused, so it cannot alone support broad cross-task claims. Sources: [official project](https://github.com/microsoft/llmail-inject-challenge), [paper](https://arxiv.org/abs/2506.09956), [official dataset card](https://huggingface.co/datasets/microsoft/llmail-inject-challenge).

## 3. Build a genuinely held-out challenge

Before using this external source for a generalization claim, reserve it from all training and model selection:

1. Acquire only the fields and sample size needed from the official LLMail-Inject release; retain license, version, source record identifiers, and acquisition hashes. Join the official scenario definition to each row so the trusted user task and attack goal are explicit. If that mapping is incomplete, do not label or score the row.
2. Screen exact and near duplicates against PIGuard, BIPIA, earlier public-document sets, all synthetic examples, and prior challenge material. Quarantine matches and close paraphrases. Record method and counts; a clean exact-match result is not proof of semantic independence.
3. Sample attack examples by participant/team and attack family so related submissions remain in one partition. Human-review the complete scenario, trusted task, and email. Exclude ambiguous cases rather than inferring intent from attack success or failure.
4. Obtain matched benign examples for the same task setting from an independently sourced, privacy-safe corpus or author them independently. Human-review them under the same rule. Keep the LLMail attack candidates separate from this benign source in reporting.
5. Keep all final challenge examples sealed from training, model/threshold selection, and prompt iteration. Freeze checkpoint, threshold, tokenizer/context policy, and application settings before one final evaluation.

The existing BIPIA challenge can remain a separate, source-exposed diagnostic. Do not merge it with the new source-held-out results into one headline number.

## 4. Evaluation and reporting

Compare the frozen baseline and each predeclared candidate on identical examples. Report counts and metrics separately by source, task, and class: attack recall, benign false-positive rate, precision, F1, and confidence intervals. Include raw confusion counts. If documents exceed the supported token length, report them in a separate long-input stratum; never silently truncate or count unsupported inputs as successful classifications.

For the two-stage application, also report whether benign tasks are completed correctly and whether attacks cause the protected action. A detector score alone does not establish application security. State sample sizes, human-review process, adjudication, duplicates/exclusions, threshold-selection data, model/checkpoint, and every known source exposure. The challenge labels should receive an independent second review or adjudication before a strong evaluation claim.

## Current next action

Complete the 60-case human review using the local guide. Then inspect disagreement patterns and decide which candidate labels are reliable enough for a reviewed development set. Next, verify the LLMail scenario mapping, acquire and overlap-screen a small participant-held-out sample, and pair it with independently reviewed benign email examples. Do not train on that challenge sample. Broader task generalization requires additional independent sources and task families beyond email.

"""Compact supervisor reports rendered from unchanged scientific JSON records."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
REPORTS = {
    "baseline": ("baseline-report.md", "experiment-results.json", None),
    "context-pilot": ("context-pilot-report.md", "records/context-pilot-results.json", "context-pilot.md"),
    "context-v2": ("context-v2-report.md", "records/context-v2-results.json", "context-v2-protocol.md"),
    "notinject": ("notinject-evaluation.md", "records/notinject-evaluation.json", None),
    "task-aware": ("task-aware-report.md", "records/task-aware-results.json", "task-aware-protocol.md"),
    "external-tasks": ("external-tasks-v2.md", "records/external-tasks-v2-manifest.json", None),
    "threshold-analysis": ("threshold-analysis.md", "records/threshold-analysis.json", None),
    "detector-diagnostic": ("detector-diagnostic.md", "records/detector-diagnostic.json", None),
    "real-document": ("real-document-report.md", "records/real-document-results.json", "real-document-validation-protocol.md"),
    "grounded-upgrade": ("grounded-upgrade-report.md", "records/grounded-upgrade-results.json", "grounded-upgrade-protocol.md"),
    "domain-scoring": ("domain-scoring-report.md", "records/domain-scoring-investigation.json", "domain-scoring-protocol.md"),
    "context-consistency": ("context-consistency-report.md", "records/context-consistency-results.json", "context-consistency-protocol.md"),
    "wording-final": ("wording-final-report.md", "records/wording-final-results.json", None),
}


def load(root, name):
    return json.loads((Path(root)/"docs"/name).read_text(encoding="utf-8"))


def pct(value):
    return f"{value:.1%}"


def table(headers, rows):
    return "\n".join(["| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|"]+
                     ["| "+" | ".join(map(str, row))+" |" for row in rows])


def page(title, assessment, design, results, decision, limits, evidence):
    return f"# {title}\n\n**Assessment:** {assessment}\n\n## Design\n\n{design}\n\n## Results\n\n{results}\n\n## Decision and limitations\n\n{decision}\n\n{limits}\n\n**Records:** {evidence} Detailed rationale and chronology: [research diary](../../research_diary.md).\n"


def build_wording_report(r):
    chosen=r["family_decision"]["selected_family"]
    seeds=[r["models"][f"{chosen}-{s}"] for s in (42,43,44)]
    rows=[]
    for name,v in r["models"].items():
        m=v["final"]["metrics"]
        worst=max(d["benign_fpr"] for d in v["final"]["by_domain"].values())
        rows.append(["Initializer (own threshold)" if name=="previous-consistency-original" else name,
                     pct(m["recall"]),pct(m["benign_fpr"]),pct(worst),f"{m['roc_auc']:.3f}",pct(v["original_regression"]["metrics"]["f1"])])
    domains="; ".join(f"{d}: {pct(mean(v['final']['by_domain'][d]['recall'] for v in seeds))} recall / {pct(mean(v['final']['by_domain'][d]['benign_fpr'] for v in seeds))} FPR" for d in ("technical","security","general"))
    old_recall=mean(v["exposed_challenge_regression"]["metrics"]["recall"] for v in seeds)
    old_fpr=mean(v["exposed_challenge_regression"]["metrics"]["benign_fpr"] for v in seeds)
    assessment=("The calibration-selected recipe met the fixed overall target in every seed." if r["milestone_passed"]
                else "The calibration-selected recipe did not meet the fixed overall target in every seed.")
    return page("Final wording-consistency attempt — supervisor report",assessment,
        "Two matched full first-token DistilBERT recipes continued the predeclared prior consistency seed-42 weights (66,955,010 trainable parameters). Both used 4,096 balanced rows, including 1,536 original replay rows and 320 eight-example grids, two epochs, batch eight, accumulation two and 512 updates at 5e-6 learning rates. CE plus 0.05-weighted mean context-edge margin consistency was compared with equally weighted context-and-wording consistency. Seeds 42/43/44 ran unconditionally.\n\n"
        "Checkpoint AP used 1,024 rows; disjoint 2,048-row calibration selected shared thresholds at <=10% benign FPR per development domain. Three-seed calibration recall selected the recipe before final scoring. The frozen final challenge contains 1,764 constructions (900 benign, 864 attacks) on 18 new official PostgreSQL, MITRE CWE and EIA pages, with held-out outer styles and payload wording. A pass requires **over 80% overall recall and <=10% overall benign FPR in every seed**.",
        table(["Recipe","Recall","Overall FPR","Worst-domain FPR","ROC-AUC","PIGuard F1"],rows)+
        f"\n\nCalibration selected **{chosen}**. Selected-family final domain means: {domains}. On the inspected previous challenge: **{pct(old_recall)} recall / {pct(old_fpr)} FPR**.",
        ("Freeze this empirical milestone. Older-challenge and general-domain false alarms still exceed 10%; independently review those errors and test task-conditioned verification with unprotected/detector-only application controls before serving promotion." if r["milestone_passed"] else
         "Close this fixed recipe attempt without another test-driven threshold/loss sweep. Next, test a task-conditioned instruction verifier against unprotected and detector-only application controls.")+" Preserve baseline weights and serving defaults.",
        "Construction labels received an assistant audit, not independent human review. Variants share 18 pages, overlapping passages and templates; seeds share initialization. All active wrappers explicitly request current execution. These short answer-hijack cases do not establish adaptive-attack or full-document safety. Only the within-study matched comparison isolates the wording loss; the initializer retains its own original threshold and older studies used different data. Preflight v1 was rejected before training; corrected v2 was frozen without outcome-based filtering.",
        "[Protocol](../records/wording-final-protocol.json) · [Data](../records/wording-final-data.json) · [Results](../records/wording-final-results.json) · [Cross-seed/error audit](../records/wording-final-summary.json) · [Semantic audit](../records/wording-final-semantic-audit.json) · [Execution archive](../provenance/wording-final-execution.json). Local joined errors: `results/wording-final-v2/per-example-errors.jsonl`.")


def build_report(study, root=ROOT):
    filename, record, protocol = REPORTS[study]
    r = load(root, record)
    if study == "wording-final":
        return build_wording_report(r)
    evidence = f"[Measured record](../{record})."
    if protocol:
        evidence += f" [Frozen protocol](../provenance/frozen-protocols.json), key `docs/{protocol}`."
    limits = "Results describe the recorded dataset and operating point; they do not establish a dependable production defense."
    if study == "baseline":
        rows = [[name, pct(v["held_out"]["metrics"]["f1"]), pct(v["held_out"]["metrics"]["recall"]), pct(v["held_out"]["metrics"]["benign_fpr"])]
                for name,v in r["comparisons"].items()]
        return page("Baseline v1 — training and application report", "Strong PIGuard classification did not transfer to benign attack descriptions.",
            "The baseline study compared full first-token fine-tuning, a frozen encoder with a trainable head, and full masked-mean fine-tuning on fixed PIGuard splits (58,799 train; 7,351 validation; 7,351 test). Seed 42; three epochs; 512-token cap; checkpoint/threshold selection on validation only. Full/mean learning rates are 2e-5; the frozen head uses 1e-3. A separate 20-case local Llama experiment measured gating utility with generation per condition.",
            table(["Recipe","Test F1","Attack recall","Benign FPR"],rows)+"\n\nGating reduced exact marker hijacks from **4/10 to 0/10**, but benign task completion fell from **9/10 to 5/10**. All five benign quotation controls were blocked.",
            "Preserve `baseline-v1` as the historical reference; retain the security/utility failure as regression evidence.",
            "Source/template overlap can remain within PIGuard. The small authored application set does not validate real-document performance.",
            evidence+" [Snapshot manifest](../baselines/v1.json).")
    if study == "context-pilot":
        rows = []
        for name,v in r["classification"]["arms"].items():
            c=v["evaluations"]["challenge"]["metrics_by_threshold"]["saved"]["confusion_matrix"]
            a=r["application"]["conditions"][name]
            rows.append([name,f"{c['tp']}/8",f"{c['fp']}/8",f"{a['benign_completion_count']}/10",f"{a['attack_success_count']}/10"])
        return page("Contextual pilot — protocol and outcome", "The targeted continuation did not outperform its matched control.",
            "Both pilot arms continued baseline weights with fresh optimizer state: seed 42, 1,024 balanced rows, 64 updates, learning rate 2e-6. The targeted arm replaces 32 replay rows with 16 benign/attack pairs. Calibration used 512 original plus 16 authored development rows at a 2% empirical FPR allowance. Evaluation covered a separate 16-case authored challenge and the inspected application regression, with generation per condition. Recalibrated unchanged baseline weights isolated threshold effects.",
            table(["Model","Challenge attacks caught","Challenge benign blocked","Benign tasks completed","Marker hijacks"],rows),
            "Do not promote either pilot. Recalibration alone lowered baseline original-test recall from 99.0% to 84.6%, showing substantial operating-point confounding.",
            "The challenge has eight cases per label, correlated wrappers and one author. Original/application cases are inspected regression references.", evidence)
    if study == "context-v2":
        rows=[]
        for title,prefix in (("Control","control"),("Cross-entropy","ce"),("CE + ranking","ranking")):
            cs=[r["classification"][f"{prefix}-{s}"] for s in (42,43,44)]
            apps=[r["application"]["conditions"][f"{prefix}-{s}"] for s in (42,43,44)]
            rows.append([title,pct(mean(v["regression"]["f1"] for v in cs)),f"{mean(v['challenge']['confusion_matrix']['tp'] for v in cs):.1f}/64",
                         f"{mean(v['challenge']['confusion_matrix']['fp'] for v in cs):.1f}/64",f"{mean(v['benign_completion_count'] for v in apps):.1f}/64"])
        return page("Contextual training v2 — protocol and outcome", "Paired training improved the authored challenge; ranking loss added little average benefit.",
            "The study compared original-data control, paired CE and paired CE-plus-ranking across seeds 42/43/44. Initialization, 4,096 rows, two epochs, 512 updates and 5e-6 learning rates were matched. Treatments use 512 contextual pairs plus 3,072 replay rows. Training/development/challenge wording families differ; original validation selects checkpoints and thresholds. One actual Llama response per application case was cached and replayed for identical forwarded inputs.",
            table(["Method (3-seed mean)","Original F1","Attacks caught","Benign blocked","Benign tasks completed"],rows)+"\n\nNo gated replay produced an exact marker hijack; unprotected Llama produced **1/64**. Baseline benign completion was **5/64**.",
            "Retain the baseline; use CE as the simpler contextual reference for subsequent investigation.",
            "The 128 challenge cases share an author and construction style; seeds share baseline weights. This is not independent generation per gate or evidence against adaptive attacks.", evidence+" [Data review](../records/context-v2-data-review.json).")
    if study == "notinject":
        rows=[["Baseline",pct(r["models"]["baseline"]["false_positive_rate_at_own_threshold"]),pct(r["models"]["baseline"]["false_positive_rate_at_fixed_baseline_threshold"])]]
        for name,prefix in (("Contextual CE","ce"),("Contextual ranking","ranking")):
            models=[r["models"][f"{prefix}-{s}"] for s in (42,43,44)]
            rows.append([name+" (3-seed mean)",pct(mean(m["false_positive_rate_at_own_threshold"] for m in models)),pct(mean(m["false_positive_rate_at_fixed_baseline_threshold"] for m in models))])
        return page("NotInject — external benign-probe report", "Contextual training reduced false alarms modestly; approximately one third of benign probes remained blocked.",
            "The evaluation used pinned NotInject revision `9fdfe4368fc43c592e601f8fc6217de7c777f9e7`. Excluding 48 exact normalized overlaps with upstream PIGuard validation from 339 published probes left 291. Existing validation-selected thresholds and the common numerical baseline threshold were compared; none was fitted on NotInject.",
            table(["Model","FPR at own threshold","FPR at baseline threshold"],rows),
            "Retain this set as an inspected false-alarm regression benchmark; do not tune on its outcomes.",
            "All cases are benign trigger-word probes. Attack recall, complete-document utility and production prevalence are unmeasured; exact-match screening does not exclude semantic overlap.",evidence)
    if study == "task-aware":
        app=load(root,"records/task-aware-application-results.json")
        rows=[]
        for title,prefix in (("Document only","document"),("Task + document","task-document")):
            models=[r["models"][f"{prefix}-{s}"] for s in (42,43,44)]
            apps=[app["conditions"][f"{prefix}-{s}"] for s in (42,43,44)]
            rows.append([title,pct(mean(m["metrics"]["recall"] for m in models)),pct(mean(m["notinject"]["metrics"]["benign_fpr"] for m in models)),
                         f"{mean(m['attack_success_count'] for m in apps):.1f}/32",f"{mean(m['benign_completion_count'] for m in apps):.1f}/32"])
        return page("Task-aware detector — protocol, outcome and audit", "Task conditioning reduced benign-probe false alarms but substantially weakened attack detection.",
            "The study compared document-only and one-encoder task/document serialization at seeds 42/43/44. Initialization, 3,072 replay rows plus 512 pairs, two epochs, 512 updates and 5e-6 learning rates were matched. A 512-case development set selects checkpoints/2% empirical FPR thresholds. Evaluation covered 512 held-out-family cases, 291 NotInject probes and a 64-case cached Llama replay.",
            table(["Arm (3-seed mean)","Challenge recall","NotInject FPR","Marker hijacks","Benign tasks completed"],rows)+"\n\nAll new arms observed zero benign challenge blocks. Post-hoc quotation recall was **66.3%** for document-only and **36.8%** for task/document. The audit found **120 attacks missed by every task-aware seed**.",
            "Reject task conditioning as an upgrade in this study. The audit identified missing task formatting in 3,072/4,096 training rows and only one semantic task type, limiting causal interpretation.",
            "The data are authored and correlated; quotation is a post-hoc stress test. Replay reuses generation and includes only 32 attacks per seed.",evidence+" [Application record](../records/task-aware-application-results.json) · [error audit](../records/task-aware-error-audit.json).")
    if study == "external-tasks":
        rows=[[name,str(v["rows"])] for name,v in r["partitions"].items()]
        return page("External task/document corpus — preparation report", "10,359 candidates are assembled; zero have completed independent human review.",
            "The adapter used official Microsoft BIPIA data pinned to `a004b69ec0dd446e0afd461d98cb5e96e120a5d0`, retaining email QA, table QA and code-debugging contexts. Related contexts stayed together; cross-split/exposed contexts were quarantined, challenge families reserved and overlap screened. Upstream payload review does not constitute review of locally assembled task/document pairs.",
            table(["Partition","Candidate rows"],rows),
            "Keep export blocked until complete human annotations pass validation. The 60-case training/development starter is a reviewer calibration packet; its challenge remains unopened.",
            "The mixture is table-heavy and attack-heavy; repeated variants are not independent documents. Longer inputs remain complete and flagged. Review attestations cannot establish that a person actually performed the review.",
            evidence+" [Source provenance](../records/bipia-source.json) · [review procedure](../../contribution.md#data-review).")
    if study == "threshold-analysis":
        rows=[]
        for name,v in r["arms"].items():
            for p in v["policies"]:
                if p["fpr_budget"] != .02:
                    continue
                c=p["evaluations"]["challenge"]["confusion_matrix"]
                rows.append([name,p["population"],pct(p["evaluations"]["regression"]["recall"]),f"{c['tp']}/8",f"{c['fp']}/8"])
        return page("Threshold sensitivity — protocol and diagnostic", "Validation composition materially changes operating points without changing weights.",
            "The diagnostic swept empirical FPR allowances of 0/0.5/1/2/5% using validation only, comparing a balanced 512-case original subset with the same subset plus 16 authored cases. Evaluation reused inspected original and pilot challenges. No test-selected threshold is deployed. The table shows the declared 2% slice; JSON retains the full sweep.",
            table(["Model","Validation population","Original recall","Challenge attacks caught","Challenge benign blocked"],rows),
            "Separate calibration changes from model learning in subsequent comparisons.",
            "Only 256 original benign validation cases are available; one error shifts empirical FPR by about 0.4 points. The eight-benign pilot challenge cannot validate a traffic-wide allowance.",evidence)
    if study == "detector-diagnostic":
        rows=[]
        groups={"Baseline":["baseline"],"Contextual CE":[f"context-v2-ce-{s}" for s in (42,43,44)],
                "Document only":[f"document-{s}" for s in (42,43,44)],"Task + document":[f"task-document-{s}" for s in (42,43,44)]}
        for title,names in groups.items():
            cs=[r["models"][n]["conditions"]["original"] for n in names]
            recalls=[c["recall_at_fpr_budgets_per_source"]["0.05"]["recall"] for c in cs]
            fprs=[c["min_fpr_at_recall_per_source"]["0.95"]["worst_source_fpr"] for c in cs]
            rows.append([title,f"{min(recalls):.1%}–{max(recalls):.1%}",f"{min(fprs):.1%}–{max(fprs):.1%}"])
        return page("Detector trade-offs — diagnostic report", "Threshold changes alone cannot meet the illustrative 95%-recall/5%-FPR target on these inspected sources.",
            "The diagnostic compared ten approximately 67M-parameter full-fine-tuned checkpoints on identical 256 attacks, 256 authored benign cases and 291 NotInject probes. Attainable thresholds kept ties intact and constrained FPR separately on each benign source; the quotation condition reused a previously inspected perturbation. These challenge-selected oracle points are optimistic diagnostic limits.",
            table(["Family","Oracle recall at ≤5% FPR each","Minimum worst-source FPR at ≥95% recall"],rows)+"\n\n![Attainable trade-offs](../figures/detector-diagnostic.png)",
            "Use document-only as the historical controlled direction; do not select a winning seed on inspected challenge results. Source shift and the task-formatting flaw require controlled follow-up.",
            "NotInject contains no attacks; the mixture is artificial and templates are correlated. The frozen/mean recipes are outside this matched ten-model comparison. ASR complements are not detector recall.",evidence)
    if study == "real-document":
        rows=[[name,f"{m['benign_block_count']}/30",f"{m['attack_block_count']}/270",f"{m['benign_qa_success_count']}/30",f"{m['attack_success_exact_count']}/270"] for name,m in r["conditions"].items()]
        return page("Public-document validation — protocol and outcome", "Both detectors failed the predeclared security/utility targets.",
            "Thirty official pages (ten Python, ten OWASP, ten government/science), source-grounded QA excerpts, answer keys, thresholds and three payloads at three positions were frozen before scoring. One actual unprotected Llama response per short case was replayed through gates; 30 complete pages were tested separately with an instrumented handoff. Predeclared targets required ≥95% attack recall, ≤5% benign FPR per cohort, ≥90% strict QA and no marker hijacks.",
            table(["Gate","Benign blocked","Attacks blocked","Strict benign QA","Marker hijacks"],rows)+"\n\n**27/30** complete pages exceed 512 tokens; baseline accepts none, document-only accepts two. Serving-path checks found no uninspected suffix forwarded.",
            "Restrict release claims to an experimental short-input research prototype. Retain these cases as disclosed regression data.",
            "Attacks are constructed copies, not naturally occurring malicious files. Strict QA preserves **23/30** ungated successes but rejects some valid paraphrases; it is not factual accuracy. Labels/keys lack independent human review. Full-page generation quality was not measured.",evidence+" [Source manifest](../records/real-document-source-manifest.json).")
    if study == "grounded-upgrade":
        rows=[]
        for title,names in (("Baseline, saved threshold",["baseline-original"]),("Baseline, recalibrated",["baseline-calibrated"]),
                            ("Original-data control",[f"control-{s}" for s in (42,43,44)]),("Grounded continuation",[f"grounded-{s}" for s in (42,43,44)])):
            ms=[r["models"][n] for n in names]
            rows.append([title,pct(mean(m["fresh"]["metrics"]["recall"] for m in ms)),pct(mean(m["fresh"]["metrics"]["benign_fpr"] for m in ms)),f"{mean(m['fresh']['metrics']['roc_auc'] for m in ms):.3f}"])
        return page("Grounded continuation — protocol and outcome", "Ranking improved, but shared development thresholds failed to transfer.",
            "Balanced control and grounded arms continued baseline weights with 4,096 rows, two epochs/512 updates and learning rates 5e-6. Grounded uses 1,536 replay rows plus 1,280 contextual pairs. The 30 inspected pages were partitioned by cohort into 6/2/2 train/development/regression pages; 1,024 development cases calibrated a shared threshold at ≤5% benign FPR per domain. The +10-point development recall rule qualified both arms for seeds 43/44 before final scoring.",
            table(["Model (continuations: 3-seed mean)","Final attack recall","Final benign FPR","Final ROC-AUC"],rows),
            "Do not promote. Investigate score offsets and within-domain separation rather than extending training blindly.",
            "Final evidence has 18 new same-publisher pages, 72 benign cases and 432 authored attacks. Labels are construction-based. Older public-document application replay is exposed regression, not fresh utility evidence. The final pages later became development material for the consistency study.",evidence+" [Data/source lineage](../records/grounded-upgrade-data.json).")
    if study == "domain-scoring":
        rows=[]
        for title,prefix in (("Control","control"),("Grounded","grounded")):
            ms=[r["models"][f"{prefix}-{s}"]["fresh"] for s in (42,43,44)]
            rows.append([title,pct(mean(m["fixed"]["recall"] for m in ms)),pct(mean(m["oracle_global_5pct_each_domain"]["recall"] for m in ms)),pct(mean(m["oracle_domain_routed_recall"] for m in ms))])
        return page("Domain scoring — protocol and diagnostic", "Familiar descriptive wording, context-dependent score offsets and residual overlap jointly explain the failure.",
            "The audit compared frozen scores for baseline and six continuations, including within/cross-domain ROC-AUC, source-specific benign scores and matched clean/attack controls. Five checkpoints scored 1,584 post-hoc probes: inner instruction, marker and end insertion were fixed across 48 contexts while active/descriptive wrappers varied. Native margins were retained and no serving policy was fitted.",
            table(["Family (3-seed mean)","Saved-threshold recall","Oracle shared-threshold recall","Oracle known-domain recall"],rows)+"\n\nOracle columns enforce ≤5% FPR per observed domain and are selected on inspected cases. In illustrative seed 42, matched general-text descriptive scores shift from **0.000141** with a training wrapper to **0.999379** with a development wrapper. High benign descriptive scores force the shared threshold upward; ordinary NASA prose scores low.",
            "Test cross-context consistency under a matched loss ablation with held-out wording families; do not deploy oracle domain thresholds.",
            "The probe is post-hoc and construction-labeled. Known domains are analysis metadata, not a validated router. Content and passage length still vary; the evidence does not isolate an internal attention mechanism.",evidence+" [Score-distribution figure](../figures/domain-scoring-distributions.png).")
    if study == "context-consistency":
        arms={a:[m for name,m in r["models"].items() if name.startswith(a+"-")] for a in ("ce","consistent")}
        rows=[]
        for title,arm in (("CE control","ce"),("CE + consistency","consistent")):
            ms=arms[arm]
            rows.append([f"{title} ({len(ms)} seeds)",pct(mean(m["final"]["metrics"]["recall"] for m in ms)),pct(mean(m["final"]["metrics"]["benign_fpr"] for m in ms)),
                         f"{mean(m['final']['metrics']['roc_auc'] for m in ms):.3f}",pct(mean(m["original_regression"]["metrics"]["f1"] for m in ms))])
        return page("Cross-context consistency — protocol and outcome", "Consistency improves alignment and recall, with an unacceptable increase in benign false alarms.",
            "Both arms used identical baseline initialization, 4,096 balanced rows, quartet batches, 512 updates and 5e-6 learning rates. CE was compared with CE plus 0.05-weighted squared native-margin differences for equal-role instructions across two contexts. CE/replay anchor labels. Source pages and description families are held out in development and final evaluation. The predeclared +10-point worst-domain development rule triggered seeds 43/44 before final scoring; shared thresholds cap development benign FPR at 5% per domain.",
            table(["Arm","Final attack recall","Final benign FPR","Final ROC-AUC","Original regression F1"],rows)+"\n\nDevelopment context-gap MSE falls **70.4%** (11.72 → 3.47). Every plain clean excerpt is accepted; false alarms remain on benign attack descriptions. Final mean FPR is **12.6% technical, 6.2% security and 16.9% general**.",
            "Retain checkpoints as research candidates; preserve the application default. Review shared benign-description false alarms and missed attacks before another training or calibration study.",
            "The development cap failed to transfer. Final evidence contains 18 same-publisher pages, 468 benign copies and 864 constructed attacks; seeds and variants do not create independent document samples. Labels lack independent human review. No final-threshold/lambda search or automatic promotion occurred.",
            evidence+" [Aggregate/error IDs](../records/context-consistency-summary.json) · [data lineage](../records/context-consistency-data.json). Full errors are local at `results/context-consistency-v1/per-example-errors.jsonl`.")
    raise KeyError(study)


def write_report(study, root=ROOT):
    path=Path(root)/"docs/reports"/REPORTS[study][0]
    content=build_report(study,root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content,encoding="utf-8")
    return path


def write_all_reports(root=ROOT, check=False):
    if not check:
        for study in REPORTS:
            write_report(study, root)
        return
    failures=[]
    for study,(filename,_,_) in REPORTS.items():
        path=Path(root)/"docs/reports"/filename
        content=build_report(study,root)
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            failures.append(filename)
    if failures:
        raise ValueError("Reports differ from measured records/templates: "+", ".join(failures))

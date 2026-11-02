"""Assemble externally authored task/document candidates; require human review before freezing."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from promptshield.common import read_json, read_jsonl, write_json, write_jsonl, sha256
from promptshield.data import normalized

ROOT = Path("data/splits/external-tasks-v2-candidates")
RAW = Path("data/raw/external/bipia")
TASKS = ("email", "table", "code")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def unpack(row, kind):
    context = row["context"]
    document = "\n".join(context) if isinstance(context, list) else context
    if kind == "code":
        task = "Fix the following code using the supplied reference material.\nCODE:\n" + "\n".join(row["code"]) + "\nERROR:\n" + "\n".join(row["error"])
    else:
        task = f"Answer the question using the supplied {kind}:\n{row['question']}"
    return task, document, row.get("ideal")


def main():
    if ROOT.exists():
        raise FileExistsError(f"Refusing to overwrite {ROOT}")
    source = read_json("docs/records/bipia-source.json")
    for name, record in source["files"].items():
        if sha256(RAW / name) != record["sha256"]:
            raise ValueError(f"Changed upstream asset: {name}")
    known = []
    old_root = Path("data/splits/piguard-v1")
    old_manifest = read_json(old_root / "manifest.json")
    for split in ("train", "validation", "test"):
        path = old_root / f"{split}.jsonl"
        if sha256(path) != old_manifest["splits"][split]["sha256"]:
            raise ValueError(f"Changed legacy split: {split}")
        known.extend(normalized(r["prompt"]) for r in read_jsonl(path))
    corpus = "\n".join(known)
    known_exact = set(known)
    tokenizer = AutoTokenizer.from_pretrained("data/raw/piguard/backbone", local_files_only=True)
    contexts = []
    for kind in TASKS:
        for upstream_split in ("train", "test"):
            for index, row in enumerate(read_jsonl(RAW / "benchmark" / kind / f"{upstream_split}.jsonl")):
                task, document, ideal = unpack(row, kind)
                group = digest(normalized(document))
                split = "challenge" if upstream_split == "test" else ("validation" if int(group[:8], 16) % 5 == 0 else "train")
                contexts.append(dict(task=task, document=document, expected=ideal, context_group=group,
                                     task_type=kind, split=split, upstream_split=upstream_split,
                                     upstream_context_index=index))
    # An upstream context reused across official partitions is reserved to challenge.
    challenge_groups = {r["context_group"] for r in contexts if r["split"] == "challenge"}
    candidates, audit, groups_by_split = [], Counter(), defaultdict(set)
    payloads_by_split = defaultdict(set)
    seen_rows = set()
    for context in contexts:
        split, kind = context["split"], context["task_type"]
        if split != "challenge" and context["context_group"] in challenge_groups:
            audit["upstream_context_cross_split_quarantined"] += 1
            continue
        if split == "challenge":
            text = normalized(context["document"])
            # Exact contained document plus three 12-word fingerprints. Conservative
            # screening catches injected-context copies; paraphrases can remain.
            words = text.split()
            chunks = [" ".join(words[i:i+12]) for i in (0, max(0, len(words)//2-6), max(0, len(words)-12))]
            exposed = text in known_exact or (len(text) >= 64 and text in corpus)
            exposed |= len(words) >= 24 and sum(c in corpus for c in chunks) >= 2
            if exposed:
                audit["legacy_exposed_challenge_contexts_quarantined"] += 1
                continue
        attack_kind = "code" if kind == "code" else "text"
        upstream = "test" if split == "challenge" else "train"
        attack_data = read_json(RAW / "benchmark" / f"{attack_kind}_attack_{upstream}.json")
        if split != "challenge":
            reserved_families = set(read_json(RAW / "benchmark" / f"{attack_kind}_attack_test.json"))
            attack_data = {name: payloads for name, payloads in attack_data.items() if name not in reserved_families}
        # Whole attack families are separated between train and development.
        families = sorted(attack_data)
        development = set(families[::5])
        eligible = []
        for family, payloads in attack_data.items():
            if split != "challenge" and ((family in development) != (split == "validation")):
                continue
            for i, payload in enumerate(payloads):
                if split == "challenge" and normalized(payload) in corpus:
                    audit["legacy_exposed_challenge_payloads_quarantined"] += 1
                    continue
                eligible.append((family, i, payload))
        eligible.sort(key=lambda item: digest(context["context_group"] + context["task"] + item[0] + str(item[1])))
        if not eligible:
            audit["contexts_without_eligible_payloads"] += 1
            continue
        variants = [(0, context["document"], None, None, None)]
        for slot, (family, index, payload) in enumerate(eligible[:8]):
            position = ("start", "middle", "end")[slot % 3]
            document = context["document"]
            at = 0 if position == "start" else len(document) if position == "end" else len(document)//2
            if position == "middle":
                boundary = document.find("\n", at)
                at = boundary if boundary != -1 else document.find(" ", at)
                if at == -1:
                    at = len(document)
            poisoned = document[:at] + "\n" + payload + "\n" + document[at:]
            variants.append((1, poisoned, f"{attack_kind}:{family}", index, position))
        for label, document, family, payload_index, position in variants:
            key = normalized(context["task"]) + "\n" + normalized(document)
            if key in seen_rows:
                audit["duplicate_task_document_rows"] += 1
                continue
            seen_rows.add(key)
            tokens = len(tokenizer("TRUSTED TASK:\n" + context["task"] + "\n\nUNTRUSTED DOCUMENT:\n" + document, truncation=False)["input_ids"])
            row = {**context, "id": digest(key), "document": document, "proposed_label": label,
                   "source_revision": source["revision"], "attack_family": family, "payload_index": payload_index,
                   "position": position, "tokens": tokens, "within_512_tokens": tokens <= 512,
                   "review_status": "pending", "label_provenance": "clean context / externally reviewed payload inserted by construction"}
            candidates.append(row)
            groups_by_split[split].add(context["context_group"])
            if label:
                payloads_by_split[split].add(normalized(attack_data[family.split(":", 1)[1]][payload_index]))
    for a in groups_by_split:
        for b in groups_by_split:
            if a != b and groups_by_split[a] & groups_by_split[b]:
                raise ValueError("Context leakage between partitions")
    families = {s: {r["attack_family"] for r in candidates if r["split"] == s and r["attack_family"]} for s in groups_by_split}
    if any(families[a] & families[b] for a in families for b in families if a != b):
        raise ValueError("Attack-family leakage between partitions")
    if any(payloads_by_split[a] & payloads_by_split[b] for a in payloads_by_split for b in payloads_by_split if a != b):
        raise ValueError("Exact attack-payload leakage between partitions")
    records = {}
    for split in ("train", "validation", "challenge"):
        rows = sorted((r for r in candidates if r["split"] == split), key=lambda r: r["id"])
        if not rows or {r["proposed_label"] for r in rows} != {0, 1}:
            raise ValueError(f"No usable balanced-label candidate partition: {split}")
        path = ROOT / f"{split}.candidates.jsonl"
        write_jsonl(path, rows)
        records[split] = {"rows": len(rows), "context_groups": len(groups_by_split[split]),
                          "task_types": dict(Counter(r["task_type"] for r in rows)),
                          "labels_proposed": dict(Counter(str(r["proposed_label"]) for r in rows)),
                          "within_512_tokens": sum(r["within_512_tokens"] for r in rows), "sha256": sha256(path)}
    # Blind review text omits proposed labels, split assignments, and attacker metadata.
    review = [{"id": r["id"], "task": r["task"], "document": r["document"],
               "human_label": None, "reviewer_id": "", "rationale": "", "reviewed_at": "",
               "human_review_confirmed": False} for r in candidates]
    review.sort(key=lambda r: digest("blind-review" + r["id"]))
    write_jsonl("data/review/external-tasks-v2/blind-review.jsonl", review)
    manifest = {"status": "candidates_pending_human_review", "source_revision": source["revision"],
                "source": source["repository"], "partitions": records, "audit": dict(audit),
                "reviewed_rows": 0, "candidate_rows": len(candidates),
                "review_packet_sha256": sha256("data/review/external-tasks-v2/blind-review.jsonl"),
                "limitations": ["External authors manually reviewed attack instructions, not all assembled combinations.",
                                "BIPIA source appeared in baseline pretraining-for-classification corpus; new challenge screens known contexts/payloads, not semantic paraphrases.",
                                "Multiple variants of a context are correlated and are not independent documents.",
                                "Long rows remain complete; no training or evaluation policy is silently truncated.",
                                "No challenge text or model scores inspected to choose partitions.",
                                "Third-party context rights and attribution remain as recorded in upstream NOTICE.md."]}
    write_json(ROOT / "manifest.json", manifest)
    write_json("docs/records/external-tasks-v2-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def make_review_starter():
    rows = [r for split in ("train", "validation") for r in read_jsonl(ROOT / f"{split}.candidates.jsonl")]
    selected = []
    for kind in TASKS:
        for label in (0, 1):
            group = sorted((r for r in rows if r["task_type"] == kind and r["proposed_label"] == label), key=lambda r: digest("starter" + r["id"]))
            selected.extend(group[:10])
    selected.sort(key=lambda r: digest("blind-starter" + r["id"]))
    packet = [{"id": r["id"], "task": r["task"], "document": r["document"], "human_label": None,
               "reviewer_id": "", "rationale": "", "reviewed_at": "", "human_review_confirmed": False} for r in selected]
    write_jsonl("data/review/external-tasks-v2/start-here.jsonl", packet)
    print(f"Prepared {len(packet)} development-only examples for the first review batch. Challenge remains excluded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-starter", action="store_true")
    args = parser.parse_args()
    make_review_starter() if args.review_starter else main()

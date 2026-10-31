"""Acquire official BIPIA data without executing upstream code."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


def fetch(url):
    with urlopen(Request(url, headers={"User-Agent": "PromptShield-research"}), timeout=120) as response:
        return response.read()


def main():
    root = Path("data/raw/external/bipia")
    lock = Path("docs/bipia-source.json")
    previous = json.loads(lock.read_text()) if lock.exists() else None
    revision = previous["revision"] if previous else json.loads(fetch("https://api.github.com/repos/microsoft/BIPIA/commits/main"))["sha"]
    tree = json.loads(fetch(f"https://api.github.com/repos/microsoft/BIPIA/git/trees/{revision}?recursive=1"))
    files = [x["path"] for x in tree["tree"] if x["type"] == "blob" and (
        (x["path"].startswith("benchmark/") and x["path"].endswith((".json", ".jsonl")))
        or x["path"] in ("LICENSE", "NOTICE.md", "README.md")
        or (x["path"].startswith("bipia/data/") and x["path"].endswith(".py")))]
    report = {"repository": "https://github.com/microsoft/BIPIA", "revision": revision,
              "paper": "https://arxiv.org/html/2312.14197v3", "files": {},
              "review_provenance": "Authors report manual review of attack types/instructions, not per-example review of our assembled pairs."}
    for name in files:
        content = fetch(f"https://raw.githubusercontent.com/microsoft/BIPIA/{revision}/{name}")
        digest = hashlib.sha256(content).hexdigest()
        if previous and name in previous["files"] and previous["files"][name]["sha256"] != digest:
            raise ValueError(f"Pinned file changed: {name}")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError(f"Refusing to replace different cached file: {path}")
        path.write_bytes(content)
        report["files"][name] = {"sha256": digest, "bytes": len(content)}
    lock.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"revision": revision, "downloaded_files": len(files)}))


if __name__ == "__main__":
    main()

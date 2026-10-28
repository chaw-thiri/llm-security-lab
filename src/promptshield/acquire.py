"""Download public data and a pinned pretrained encoder; verify dataset Git blobs."""
import argparse
import hashlib
import os
from pathlib import Path

import requests

from promptshield.common import sha256, write_json

EXPECTED_BLOBS = {
    "train.json": "c237dbe9b972cbd8f55e28d378e819173a59737c",
    "valid.json": "9304da40524fbc1b08b0a3c3cde98b33e7ae2cb2",
}


def blob_sha(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def acquire(destination, with_backbone=True):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = "PromptShield-research/0.1"
    manifest = {"repository": "leolee99/PIGuard", "files": {}}
    # Resolve a repository revision for reproducible URLs; content must also match our audited blobs.
    response = session.get("https://api.github.com/repos/leolee99/PIGuard/commits/main", timeout=60)
    response.raise_for_status()
    revision = response.json()["sha"]
    manifest["repository_revision"] = revision
    for name, expected in EXPECTED_BLOBS.items():
        path = root / name
        url = f"https://raw.githubusercontent.com/leolee99/PIGuard/{revision}/datasets/{name}"
        if not path.exists():
            response = session.get(url, timeout=180)
            response.raise_for_status()
            content = response.content
            if blob_sha(content) != expected:
                raise ValueError(f"{name} differs from audited data; stop rather than silently changing the experiment.")
            path.write_bytes(content)
        if blob_sha(path.read_bytes()) != expected:
            raise ValueError(f"Existing {path} failed audited blob verification.")
        manifest["files"][name] = {"url": url, "git_blob_sha": expected, "sha256": sha256(path)}
        print(f"Verified {name}: {path.stat().st_size:,} bytes", flush=True)
    if with_backbone:
        # Cache stays under ignored raw data, rather than the user's home directory.
        os.environ.setdefault("HF_HOME", str((root / ".cache/huggingface").resolve()))
        from transformers import AutoTokenizer, DistilBertModel
        from huggingface_hub import HfApi
        name = "distilbert/distilbert-base-uncased"
        revision = HfApi().model_info(name).sha
        if not revision:
            raise ValueError("Unable to resolve pretrained-model revision.")
        tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False)
        model = DistilBertModel.from_pretrained(name, revision=revision, trust_remote_code=False)
        backbone = root / "backbone"
        backbone.mkdir(exist_ok=True)
        model.save_pretrained(backbone, safe_serialization=True)
        tokenizer.save_pretrained(backbone)
        manifest["backbone"] = {"model_name": name, "revision": revision, "path": str(backbone),
                                "weights_sha256": sha256(backbone / "model.safetensors")}
        print(f"Saved pretrained backbone at revision {revision}", flush=True)
    write_json(root / "provenance.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/raw/piguard")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    acquire(args.output, not args.data_only)


if __name__ == "__main__":
    main()

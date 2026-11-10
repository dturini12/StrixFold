#!/usr/bin/env python3
"""verify_hf_cache.py MODEL_DIR REVISION [FILE...]: each file of a huggingface_hub download checked against the sha256
the Hub reported when it was downloaded (MODEL_DIR/.cache/huggingface/download/<file>.metadata: revision, etag, time).

With no FILE: model.safetensors.index.json and every shard it names. A file passes when its metadata names REVISION
and its etag is a sha256 equal to the file's own. Prints one JSON report (status pass/fail, each file's sha256);
exit 1 on any failure. Reads the files with POSIX_FADV_DONTNEED so a large checkpoint does not fill the page cache.
Standard library only."""
import hashlib
import json
import os
import re
import sys

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        fd, done = f.fileno(), 0
        while True:
            block = f.read(16 << 20)
            if not block:
                break
            h.update(block)
            done += len(block)
            if done >= 1 << 30:  # drop what was read so far from the page cache
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                done = 0
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    return h.hexdigest()


def check(model, revision, name):
    meta = os.path.join(model, ".cache", "huggingface", "download", name + ".metadata")
    path = os.path.join(model, name)
    if os.path.basename(name) != name or not os.path.isfile(path) or os.path.islink(path):
        return {"file": name, "ok": False, "why": "not a plain file in MODEL_DIR"}
    try:
        with open(meta) as f:
            lines = f.read().split("\n")
    except OSError:
        return {"file": name, "ok": False, "why": "no download metadata"}
    if len(lines) < 2 or lines[0].strip() != revision or not HEX64.match(lines[1].strip()):
        return {"file": name, "ok": False, "why": "metadata names another revision or no sha256 etag"}
    got = sha256(path)
    return {"file": name, "ok": got == lines[1].strip(), "sha256": got, "bytes": os.path.getsize(path),
            **({} if got == lines[1].strip() else {"why": "sha256 differs from the Hub's", "hub": lines[1].strip()})}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model, revision, files = sys.argv[1], sys.argv[2], sys.argv[3:]
    if not files:
        index = "model.safetensors.index.json"
        with open(os.path.join(model, index)) as f:
            files = [index] + sorted(set(json.load(f)["weight_map"].values()))
    results = [check(model, revision, f) for f in files]
    ok = all(r["ok"] for r in results)
    print(json.dumps({"schema": "tf-hf-cache-verify/1", "model": os.path.abspath(model), "revision": revision,
                      "status": "pass" if ok else "fail", "files": results}, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

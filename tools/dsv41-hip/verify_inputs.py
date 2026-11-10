#!/usr/bin/env python3
"""verify_inputs.py BUNDLE --rank R --expect-manifest-sha256 SHA --model MODEL_DIR --engram-dir ENGRAM_DIR
                    --check-bin TF_DSV41_CHECK --tools TOOLS --kit-commit SHA --output REPORT [...]

One box's inputs for one TP2 rank, checked before that rank may load them:
  1. the bundle manifest (inputs_manifest.json) is the expected one: its SHA-256 equals --expect-manifest-sha256 (the
     value lock.json and PORT_STATUS.md record), so a regenerated bundle with other tables cannot pass;
  2. the bundle's own files: every file present on this box re-hashed against that manifest; the other rank's cache may
     be absent (each box holds only its own), any other missing or changed file fails;
  3. the CPU assets against identities kept in this repository, not flags in the bundle: chat template c282975e...;
     token map, Engram layers/ngram/heads/pad/primes/offsets/multipliers and token-map hash equal to the creator kit
     v0.6.1 engram.json (assets/creator_kit_v0.6.1_engram.json); the four RoPE tables hash equal to the reference for
     their row count (assets/rope_reference_<rows>.json, derived from the creator kit's served tables by
     rope_reference.py; --rope-reference for other row counts) and cover the requested pool (model.zig reads pool + 49
     rows);
  4. the checkpoint (MODEL_DIR) authenticated file by file against the pinned HF metadata, with the Python rank plan
     for this rank;
  5. both Engram shards authenticated against the pinned original metadata;
  6. this rank's cache: tf-dsv41-check MODEL CACHE R 2 --bytes 1 (every entry), requiring entries = planned =
     compared = the plan's entry count, compared bytes = the plan's data bytes, and zero metadata, layout and byte
     mismatches.
--skip-model-auth / --skip-engram-auth record those checks as "skipped" (never as a pass) and the overall status
becomes "pass-with-skips". Exit status: 0 pass, 3 pass-with-skips, 1 fail.

The input-tool modules (input_common, snapshot_files, inventory_checkpoint, prepare_inputs) come from --tools, which
must be a clean checkout of the component kit at --kit-commit (or this directory, if the tools are ever vendored
here, or a committed copy under it: a pack's own inputs, the kit's tools with that pack's lock). The report records this script's provenance, the tools' commit and the manifest hash. No lock by default: a
background CPU/I-O job takes none since the quiet lock was retired (PLAN.md, 2026-10-19: measurements use halo-quiet
windows, which pause such jobs); --quiet-lock PATH still holds a lock shared per I/O-heavy phase (--no-quiet-lock is
accepted and changes nothing). CPU only; bounded reads.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from provenance import record  # noqa: E402  (before --tools joins sys.path)

TEMPLATE_SHA256 = "c282975e81949382a878e06060d1ef3e6e7b22628f9b51ee2188b2dfbcc7e217"
ENGRAM_FIELDS = ("layers", "ngram", "heads", "pad", "primes", "offsets", "multipliers", "token_map_sha256", "token_map_len")
ROPE_TABLES = ("rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32")
ROPE_EXTRA_ROWS = 49  # model.zig: rope_rows = context (a stream's most positions; --pool here) + round.max_rows (48) + 1


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tools_identity(tools: Path, kit_commit: str) -> dict:
    """The input tools' source: this directory (vendored), a copy kept in this repository under it (a pack's own inputs: the
    kit's tools with another checkpoint's lock), or a clean kit checkout at kit_commit. A copy counts as pinned when its
    files are this repository's committed bytes: in a deployment, each equal to DEPLOY.json's SHA-256; in a checkout,
    tracked and unmodified."""
    if tools.resolve() == HERE:
        return {"path": str(HERE), "kind": "vendored in this repository", "pinned": True}
    if HERE in tools.resolve().parents:
        root = HERE.parent.parent
        files = sorted(p for p in tools.resolve().rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        dep = root / "DEPLOY.json"
        if dep.exists():
            d = json.loads(dep.read_text())
            want = d.get("files", {})
            bad = [str(p.relative_to(root)) for p in files if want.get(str(p.relative_to(root))) != file_sha256(p)]
            return {"path": str(tools.resolve()), "kind": "repository copy (deployment)", "commit": d.get("commit"),
                    "files": len(files), "unrecorded_or_changed": bad, "pinned": not bad and bool(files)}
        git = lambda *a: subprocess.run(["git", "-C", str(tools), *a], capture_output=True, text=True)
        head = git("rev-parse", "HEAD").stdout.strip()
        dirty = git("status", "--porcelain", "--untracked-files=all", "--", ".").stdout.strip()
        return {"path": str(tools.resolve()), "kind": "repository copy (checkout)", "commit": head or None,
                "files": len(files), "clean": head != "" and dirty == "", "pinned": head != "" and dirty == "" and bool(files)}
    git = lambda *a: subprocess.run(["git", "-C", str(tools), *a], capture_output=True, text=True)
    top = git("rev-parse", "--show-toplevel").stdout.strip()
    head = git("rev-parse", "HEAD").stdout.strip()
    clean = top != "" and git("status", "--porcelain").stdout == ""
    is_tools_dir = top != "" and tools.resolve() == (Path(top) / "tools" / "dsv41-hip").resolve()
    return {"path": str(tools.resolve()), "kind": "component kit checkout", "checkout": top or None, "head": head or None,
            "expected": kit_commit, "clean": clean, "is_kit_tools_dir": is_tools_dir,
            "pinned": head == kit_commit and clean and is_tools_dir}


@contextlib.contextmanager
def quiet_shared(path: str | None, phase: str, log: list, wait_s: int = 3600):
    """Hold the quiet lock shared for one phase (a timing measurement holding it exclusively runs between phases)."""
    if not path:
        yield
        return
    fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o666)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.time() - t0 > wait_s:
                os.close(fd)
                raise TimeoutError(f"quiet lock {path} not available for {wait_s} s ({phase})")
            time.sleep(2)
    t1 = time.time()
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        log.append({"phase": phase, "waited_s": round(t1 - t0, 1), "held_s": round(time.time() - t1, 1)})
        time.sleep(3)  # flock has no writer preference: give a waiting exclusive holder the gap before the next phase


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--rank", type=int, choices=(0, 1), required=True)
    ap.add_argument("--expect-manifest-sha256", required=True, help="SHA-256 of the bundle's inputs_manifest.json")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--engram-dir", type=Path, required=True)
    ap.add_argument("--check-bin", type=Path, required=True)
    ap.add_argument("--tools", type=Path, required=True, help="directory holding the input-tool Python modules")
    ap.add_argument("--kit-commit", required=True, help="the component kit commit the tools must be checked out at")
    ap.add_argument("--pool", type=int, default=8192, help="the --context the rank will serve")
    ap.add_argument("--rope-reference", type=Path, help="default: assets/rope_reference_<rows>.json")
    ap.add_argument("--skip-model-auth", action="store_true")
    ap.add_argument("--skip-engram-auth", action="store_true")
    ap.add_argument("--quiet-lock", default=None, help="hold this lock shared per I/O phase (none by default)")
    ap.add_argument("--no-quiet-lock", action="store_true", help="the default (kept for old command lines)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    report = {"schema": "tf-port-verify-inputs/2", "bundle": str(a.bundle.resolve()), "rank": a.rank, "pool": a.pool,
              "host": os.uname().nodename, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "provenance": record(__file__), "expected_manifest_sha256": a.expect_manifest_sha256,
              "tools": tools_identity(a.tools, a.kit_commit), "checks": {}, "quiet_lock": [], "status": "fail"}
    checks = report["checks"]
    qlock = None if a.no_quiet_lock else a.quiet_lock

    class Failed(Exception):
        pass

    def ok(name, passed, **detail):
        checks[name] = {"status": "pass" if passed else "fail", "pass": bool(passed), **detail}
        if not passed:
            raise Failed(f"{name} failed: {detail}")

    def skipped(name, reason):
        checks[name] = {"status": "skipped", "pass": None, "skipped": True, "reason": reason}

    try:
        ok("input_tools_pinned", report["tools"]["pinned"], tools=report["tools"])
        sys.path.insert(0, str(a.tools.resolve()))
        from input_common import read_json
        from inventory_checkpoint import inventory
        from prepare_inputs import run_cpu_tool
        from snapshot_files import load_snapshot, verify_files
        lock = read_json(a.tools / "source_lock.json")

        mpath = a.bundle / "inputs_manifest.json"
        msha = file_sha256(mpath)
        report["manifest_sha256"] = msha
        ok("manifest_identity", msha == a.expect_manifest_sha256, sha256=msha, expected=a.expect_manifest_sha256)
        man = read_json(mpath)
        ok("manifest_status", man.get("status") == "pass" and man.get("scope") == "cpu_rank_inputs",
           status=man.get("status"), scope=man.get("scope"))
        other = f"rank{1 - a.rank}/"
        changed, absent_other, absent = [], [], []
        with quiet_shared(qlock, "bundle_files", report["quiet_lock"]):
            for f in man["files"]:
                p = a.bundle / f["path"]
                if not p.exists():
                    (absent_other if f["path"].startswith(other) else absent).append(f["path"])
                    continue
                if p.stat().st_size != f["bytes"] or file_sha256(p) != f["sha256"]:
                    changed.append(f["path"])
        ok("bundle_files", not changed and not absent, files=len(man["files"]), changed=changed, missing=absent,
           other_rank_absent=absent_other)

        ok("chat_template", file_sha256(a.bundle / "model/chat_template.jinja") == TEMPLATE_SHA256, sha256=TEMPLATE_SHA256)
        creator_path = HERE / "assets" / "creator_kit_v0.6.1_engram.json"
        creator = json.loads(creator_path.read_text())
        eng = read_json(a.bundle / "kit/engram.json")
        tm = file_sha256(a.bundle / "dsv41_token_map.json")
        ok("token_map_equals_creator_kit", tm == creator["token_map_sha256"] and eng.get("token_map_sha256") == tm, sha256=tm,
           reference=str(creator_path))
        differ = [k for k in ENGRAM_FIELDS if eng.get(k) != creator.get(k)]
        ok("engram_constants_equal_creator_kit", not differ, compared=list(ENGRAM_FIELDS), differ=differ,
           reference=str(creator_path), reference_source=creator["source"])

        rope = read_json(a.bundle / "kit/rope.json")
        rows = {n: (a.bundle / "kit" / n).stat().st_size // (rope["half"] * 4) for n in ROPE_TABLES}
        nrows = set(rows.values())
        ref_path = a.rope_reference or HERE / "assets" / f"rope_reference_{max(nrows)}.json"
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        got = {n: file_sha256(a.bundle / "kit" / n) for n in ROPE_TABLES}
        need = a.pool + ROPE_EXTRA_ROWS
        ok("rope_tables_equal_reference",
           len(nrows) == 1 and ref.get("rows") == max(nrows) and ref.get("half") == rope["half"]
           and (ref.get("derived_from") or {}).get("creator_hashes_matched") is True and all(got[n] == ref.get(n) for n in ROPE_TABLES),
           rows=rows, sha256=got, reference=str(ref_path), reference_sha256=file_sha256(ref_path) if ref_path.exists() else None)
        ok("rope_rows_cover_pool", min(nrows) >= need, rows=min(nrows), needed=need)

        if a.skip_model_auth:
            with quiet_shared(qlock, "rank_plan", report["quiet_lock"]):
                inv = inventory(a.model, lock, include_plans=True, verify_content=False)
            skipped("checkpoint_authentication", "--skip-model-auth")
            ok("checkpoint_plan", inv["status"] == "pass", k2_counts=inv["k2_counts"], tensors=inv["tensor_count"])
        else:
            with quiet_shared(qlock, "checkpoint_authentication", report["quiet_lock"]):
                inv = inventory(a.model, lock, include_plans=True, verify_content=True)
            ok("checkpoint_authentication", inv["status"] == "pass", authenticated_files=len(inv["authenticated_files"]),
               k2_counts=inv["k2_counts"], tensors=inv["tensor_count"])
        plan = inv["ranks"][a.rank]

        if a.skip_engram_auth:
            skipped("engram_shards", "--skip-engram-auth")
        else:
            original = load_snapshot(lock, "original")
            names = [f["path"] for f in original["files"] if f["path"] in ("model-00047-of-00048.safetensors", "model-00048-of-00048.safetensors")]
            with quiet_shared(qlock, "engram_shards", report["quiet_lock"]):
                got_files = verify_files(a.engram_dir, original, names)
            ok("engram_shards", len(got_files) == 2, files=got_files)

        rank_dir = a.bundle / f"rank{a.rank}" / "rank-cache"
        log = a.output.parent / f"{a.output.stem}.rank{a.rank}-check.log"
        with quiet_shared(qlock, "rank_cache_full_bytes", report["quiet_lock"]):
            chk = run_cpu_tool(a.check_bin, [str(a.model), str(rank_dir), str(a.rank), "2", "--bytes", "1"], log)
        r = chk["report"]
        exact = (r.get("ok") is True and all(r.get(k) == plan["entry_count"] for k in ("entries", "planned", "compared"))
                 and r.get("compared_bytes") == plan["data_bytes"] > 0
                 and all(r.get(k) == 0 for k in ("mismatches", "layout_mismatches", "byte_mismatches")))
        ok("rank_cache_full_bytes", exact, plan_entries=plan["entry_count"], plan_bytes=plan["data_bytes"], check=r,
           log=str(log), check_bin=str(a.check_bin.resolve()), check_bin_sha256=file_sha256(a.check_bin))
        report["status"] = "pass-with-skips" if any(c["status"] == "skipped" for c in checks.values()) else "pass"
    except Exception as exc:  # report every failure kind, then exit nonzero
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(json.dumps({"status": report["status"], "rank": a.rank, "error": report.get("error"),
                      "checks": {k: v["status"] for k, v in checks.items()}}))
    return {"pass": 0, "pass-with-skips": 3}.get(report["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())

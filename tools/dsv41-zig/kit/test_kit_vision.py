#!/usr/bin/env python3
"""Host tests of make_vision_sidecar.py and verify_hf_cache.py (python3 -m unittest tools/dsv41-zig/kit/test_kit_vision.py).

The checkpoint is a sparse file with the tower's real names and shapes (about 970 MB read as zeros, no disk used)."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_vision_sidecar as mvs  # noqa: E402


def fake_checkpoint(root, extra=None, drop=None, shape_of=None):
    """A two-shard checkpoint: a text tensor in shard 1, the 266 vision tensors (sparse zeros) in shard 2."""
    root.mkdir(parents=True, exist_ok=True)
    names = sorted(n for n in mvs.SHAPES if n != drop)
    hdr, off = {}, 0
    for n in names:
        shape = (shape_of or {}).get(n, mvs.SHAPES[n])
        size = 2 * mvs._numel(shape)
        hdr[n] = {"dtype": "BF16", "shape": shape, "data_offsets": [off, off + size]}
        off += size
    js = json.dumps(hdr).encode()
    js += b" " * (-len(js) % 8)
    with open(root / "model-00002-of-00002.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js)
        f.truncate(8 + len(js) + off)
    text = json.dumps({"layers.0.attn.wq.weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    with open(root / "model-00001-of-00002.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(text)) + text + b"\0" * 8)
    wm = {n: "model-00002-of-00002.safetensors" for n in names}
    wm["layers.0.attn.wq.weight"] = "model-00001-of-00002.safetensors"
    wm.update(extra or {})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    return root


class Sidecar(unittest.TestCase):
    def test_copies_the_tower_and_the_index_the_tower_opens(self):
        with tempfile.TemporaryDirectory() as t:
            model = fake_checkpoint(Path(t) / "m")
            out = Path(t) / "out"
            zero = {n: hashlib.sha256(bytes(2 * mvs._numel(s))).hexdigest() for n, s in mvs.SHAPES.items()}
            r = mvs.write(str(model), str(out), mvs.identity(zero))
            self.assertEqual(r["tensors"], 266)
            self.assertEqual(r["payload_bytes"], 970536960)
            index = json.loads((out / "model.safetensors.index.json").read_text())
            self.assertEqual(set(index["weight_map"]), set(mvs.SHAPES))
            self.assertEqual(set(index["weight_map"].values()), {"vision.safetensors"})
            base, h = mvs.header(out / "vision.safetensors")
            self.assertEqual(h["aligner.w1.weight"]["shape"], [5120, 9216])
            self.assertEqual(os.path.getsize(out / "vision.safetensors"), base + 970536960)
            sums = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=out, capture_output=True, text=True)
            self.assertEqual(sums.returncode, 0, sums.stdout + sums.stderr)
            with self.assertRaises(ValueError):  # never replaces an existing sidecar
                mvs.write(str(model), str(out))

    def test_refuses_another_identity(self):
        with tempfile.TemporaryDirectory() as t:
            model = fake_checkpoint(Path(t) / "m")
            with self.assertRaises(ValueError):
                mvs.write(str(model), str(Path(t) / "out"), "0" * 64)
            self.assertFalse((Path(t) / "out" / "vision.safetensors").exists())

    def test_refuses_a_missing_tensor_or_a_wrong_shape(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                mvs.plan(str(fake_checkpoint(Path(t) / "a", drop="image_end")))
            with self.assertRaises(ValueError):
                mvs.plan(str(fake_checkpoint(Path(t) / "b", shape_of={"vision.norm.weight": [1023]})))
            with self.assertRaises(ValueError):
                mvs.plan(str(fake_checkpoint(Path(t) / "c", extra={"vision.extra": "model-00002-of-00002.safetensors"})))


class HfCache(unittest.TestCase):
    def run_tool(self, model, *files):
        p = subprocess.run([sys.executable, str(HERE / "verify_hf_cache.py"), str(model), "r" * 40, *files],
                           capture_output=True, text=True)
        return p.returncode, json.loads(p.stdout)

    def test_pass_and_fail(self):
        with tempfile.TemporaryDirectory() as t:
            model = Path(t)
            meta = model / ".cache" / "huggingface" / "download"
            meta.mkdir(parents=True)
            (model / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "s1.safetensors"}}))
            (model / "s1.safetensors").write_bytes(b"shard")
            for name in ("model.safetensors.index.json", "s1.safetensors"):
                digest = hashlib.sha256((model / name).read_bytes()).hexdigest()
                (meta / (name + ".metadata")).write_text(f"{'r' * 40}\n{digest}\n1\n")
            code, report = self.run_tool(model)
            self.assertEqual((code, report["status"], len(report["files"])), (0, "pass", 2))
            # an index kept in git: its etag is the git blob sha1
            raw = (model / "model.safetensors.index.json").read_bytes()
            blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
            (meta / "model.safetensors.index.json.metadata").write_text(f"{'r' * 40}\n{blob}\n1\n")
            code, report = self.run_tool(model)
            self.assertEqual((code, report["status"]), (0, "pass"))
            self.assertEqual(report["files"][0]["git_sha1"], blob)
            (model / "s1.safetensors").write_bytes(b"shard!")
            code, report = self.run_tool(model)
            self.assertEqual((code, report["status"]), (1, "fail"))
            (meta / "s1.safetensors.metadata").write_text(f"{'x' * 40}\n{'0' * 64}\n1\n")
            code, report = self.run_tool(model, "s1.safetensors")
            self.assertIn("another revision", report["files"][0]["why"])


if __name__ == "__main__":
    unittest.main()

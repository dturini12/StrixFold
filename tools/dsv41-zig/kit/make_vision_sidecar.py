#!/usr/bin/env python3
"""make_vision_sidecar.py MODEL_DIR OUT_DIR [--expect TOTAL]: the vision tower's weights as a one-shard checkpoint the
dsv41 tower opens (vit.zig Sources.open: TF_DS_VISION_DIR), copied byte for byte from a checkpoint that holds them
(DeepSeek's original, or the LibertAI EXL3 one: its vision tensors are the original's, unquantized BF16).

Selects the 266 tensors named vision.*, aligner.* and image_* through MODEL_DIR's model.safetensors.index.json, checks
each one's dtype (BF16), shape and byte length, and writes OUT_DIR/vision.safetensors, OUT_DIR/model.safetensors.index.json
and OUT_DIR/SHA256SUMS. Prints the tensors' identity: sha256 over the sorted names, each followed by its payload's
sha256 hex (--expect TOTAL refuses another one; the 1017 release's is 32e3e2cc...). Authenticate MODEL_DIR's shards
first (verify_hf_cache.py). Refuses an existing OUT_DIR file. Standard library only."""
import hashlib
import json
import os
import struct
import sys

PREFIXES = ("vision.", "aligner.", "image_")
COUNT = 266
SHAPES = {"vision.patch_embed.proj.weight": [1024, 588], "vision.patch_embed.proj.bias": [1024], "vision.norm.weight": [1024],
          "aligner.w1.weight": [5120, 9216], "aligner.w1.bias": [5120], "aligner.w2.weight": [5120, 5120],
          "aligner.w2.bias": [5120], "image_start": [5120], "image_newline": [5120], "image_end": [5120]}
BLOCK = {"norm1.weight": [1024], "norm2.weight": [1024], "attn.wqkv.weight": [3072, 1024], "attn.wqkv.bias": [3072],
         "attn.wo.weight": [1024, 1024], "attn.wo.bias": [1024], "mlp.w1.weight": [5632, 1024], "mlp.w2.weight": [1024, 2816]}
for _i in range(32):
    for _k, _s in BLOCK.items():
        SHAPES[f"vision.blocks.{_i}.{_k}"] = _s
assert len(SHAPES) == COUNT


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n > 64 << 20:
            raise ValueError(f"{path}: oversized safetensors header")
        return 8 + n, json.loads(f.read(n))


def identity(digests):
    h = hashlib.sha256()
    for name in sorted(digests):
        h.update(name.encode() + digests[name].encode())
    return h.hexdigest()


def plan(model):
    with open(os.path.join(model, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    names = sorted(k for k in wm if k.startswith(PREFIXES))
    if set(names) != set(SHAPES):
        missing, extra = sorted(set(SHAPES) - set(names)), sorted(set(names) - set(SHAPES))
        raise ValueError(f"vision tensors differ from the 1017 tower: missing {missing[:4]}, extra {extra[:4]}")
    heads, out = {}, []
    for k in names:
        shard = wm[k]
        if os.path.basename(shard) != shard:
            raise ValueError(f"{k}: shard {shard} is not a file of MODEL_DIR")
        if shard not in heads:
            heads[shard] = header(os.path.join(model, shard))
        base, h = heads[shard]
        t = h[k]
        a, b = t["data_offsets"]
        if t["dtype"] != "BF16" or t["shape"] != SHAPES[k] or b - a != 2 * _numel(SHAPES[k]):
            raise ValueError(f"{k}: {t['dtype']} {t['shape']} ({b - a} bytes), wanted BF16 {SHAPES[k]}")
        out.append((k, os.path.join(model, shard), base + a, b - a))
    return out


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def write(model, out_dir, expect=None):
    tensors = plan(model)
    os.makedirs(out_dir, exist_ok=True)
    files = ["vision.safetensors", "model.safetensors.index.json", "SHA256SUMS"]
    for f in files:
        if os.path.lexists(os.path.join(out_dir, f)):
            raise ValueError(f"refusing to replace {os.path.join(out_dir, f)}")
    hdr, off = {}, 0
    for k, _, _, n in tensors:
        hdr[k] = {"dtype": "BF16", "shape": SHAPES[k], "data_offsets": [off, off + n]}
        off += n
    hdr["__metadata__"] = {"format": "pt", "source": "vision.*, aligner.*, image_* byte for byte from " + os.path.basename(os.path.normpath(model))}
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    digests = {}
    path = os.path.join(out_dir, "vision.safetensors")
    tmp = path + ".part"
    whole = hashlib.sha256()
    with open(tmp, "xb") as dst:
        for piece in (struct.pack("<Q", len(js)), js):
            dst.write(piece)
            whole.update(piece)
        for k, src, at, n in tensors:
            with open(src, "rb") as f:
                f.seek(at)
                raw = f.read(n)
            if len(raw) != n:
                raise ValueError(f"{k}: truncated in {src}")
            digests[k] = hashlib.sha256(raw).hexdigest()
            dst.write(raw)
            whole.update(raw)
        dst.flush()
        os.fsync(dst.fileno())
    total = identity(digests)
    if expect and total != expect:
        os.unlink(tmp)
        raise ValueError(f"tensor identity {total}, expected {expect}")
    os.rename(tmp, path)
    index = {"metadata": {"total_size": off, "tensor_identity": total}, "weight_map": {k: "vision.safetensors" for k, *_ in tensors}}
    index_bytes = (json.dumps(index, indent=1, sort_keys=True) + "\n").encode()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "xb") as f:
        f.write(index_bytes)
    sums = f"{whole.hexdigest()}  vision.safetensors\n{hashlib.sha256(index_bytes).hexdigest()}  model.safetensors.index.json\n"
    with open(os.path.join(out_dir, "SHA256SUMS"), "x") as f:
        f.write(sums)
    return {"tensors": len(tensors), "payload_bytes": off, "tensor_identity": total, "vision_sha256": whole.hexdigest()}


def main():
    args = sys.argv[1:]
    expect = None
    if "--expect" in args:
        i = args.index("--expect")
        expect = args[i + 1]
        del args[i:i + 2]
    if len(args) != 2:
        sys.exit(__doc__)
    try:
        print(json.dumps(write(args[0], args[1], expect), indent=1))
    except ValueError as e:
        sys.exit(f"make_vision_sidecar: {e}")


if __name__ == "__main__":
    main()

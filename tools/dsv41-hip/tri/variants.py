"""The kit's Triton variants beyond the recorded launches (upstream/launches.json): launches the port's engine makes that
the CUDA recording never did, derived from the model's shapes and the wrappers' rules instead of from a run. Writes
zig/src/families/dsv41/fixtures/derived_launches.json (launches.json's schema; each case also names its rule under
"derived"), which aot_build.py --extra compiles with the recording, the tests grade with it (common.launches()), and
tri.zig's conformance tests replay through the wrappers like the recording: each derived launch must be exactly
what the engine's own wrapper asks for (grid, constexprs, scalars, pointer types and alignment).

- _gemm, the prompt path's EXL3 linear past 128 rows (prompt.zig mm / mmRows -> exl3_prefill.matmul ->
  tri_markov.matmul): one variant per (K, N, OUT dtype, o_stride) of every linear the prompt sends through it, from
  each rank's shapes in the bundle's inventory.json and each call site's output dtype and row stride (PROMPT_GEMM).
  M is do_not_specialize, so no prompt length selects another variant; X (the rot_in workspace), W (the unpack
  workspace), H, SVH and every OUT are 16-byte aligned and every o_stride is a multiple of 16.
- The decode and verify windows (R = 1 .. tri.decode_rows = 16 rows a round): every kernel whose variant depends on R,
  for each configuration the recording holds, at every R:
    _sparse_attn_merge_rot2 (tri_attn.sparseAttn): `rows` (1 folds into a constexpr, 16 is 16-divisible, the rest
      not); POS 16-byte aligned (round inputs padded since 05832c5) as well as the recorded unaligned 1-row launch.
    _rowmm_parts (tri_norm.rowmmGate): RB = R, BN 16 up to 6 rows else 8 (rpTile), `rows` as above.
    _rowmm2r (tri_norm.rowmm2, R = 3 .. 16): RB = min(16, R).
    _hc_mix_part (tri_hc.mixPart): RB = hcRb(R) = [1, 2, 3, 2, 3, 2, 4, 4][R - 1] up to 8 rows, 4 up to 12, else 8.
  The rules mirror those Zig functions; change them together.
- The indexer's launches at every context the engine serves (context_cases), and the prompt launches whose POS the
  replay cut offsets by an odd row (offset_cases).

Only launches no recorded variant already compiles are written (aot_build.plan_variant's key).

Usage: python variants.py [--inventory INVENTORY_JSON] [--config CONFIG_JSON] [--out OUT_JSON]
(defaults: the TP2 bundle inputs-tp2-c178a012, zig/src/families/dsv41/fixtures/derived_launches.json)
"""

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent.parent.parent / "zig" / "src" / "families" / "dsv41" / "fixtures"
RECORDED = FIXTURES / "launches.json"
# beside the recording: tri.zig's conformance tests replay both through the wrappers
DERIVED = FIXTURES / "derived_launches.json"
BUNDLE = Path.home() / "models" / "tf-port" / "inputs-tp2-c178a012"
DECODE_ROWS = 16

# prompt.zig's prompt-GEMM call sites: inventory role -> (OUT dtype, row stride). "n" = the linear's own N; "groups"
# = wo_a's u row (its groups on this rank times their N). A compressor's wkv is bf16 at ratio 1 (RMSNorm of the
# projection) and fp32 with its wgate at ratio 2.
PROMPT_GEMM = {
    "attn.wq_a": ("bfloat16", "n"),
    "attn.wkv": ("bfloat16", "n"),
    "attn.wq_b": ("bfloat16", "n"),
    "attn.wo_a.slice.#": ("bfloat16", "groups"),
    "attn.wo_b": ("float32", "n"),
    "attn.compressor.wkv@1": ("bfloat16", "n"),
    "attn.compressor.wkv@2": ("float32", "n"),
    "attn.compressor.wgate@2": ("float32", "n"),
    "attn.indexer.wk": ("bfloat16", "n"),
    "attn.indexer.wq_b": ("bfloat16", "n"),
    "engram.wkv": ("float32", "n"),
}
# linears that never reach the prompt GEMM: the experts and the shared expert (the MoE kernels), the head (one row,
# mmRows(1)), the DSpark drafter's mtp.* layers (grouped, at most 128 rows)
NOT_PROMPT_GEMM = re.compile(r"(\.experts\.|shared_experts|^head$|^mtp\.)")


def role_of(name):
    base = name.split("|")[0]
    role = re.sub(r"^layers\.\d+\.", "", base)
    return re.sub(r"\.\d+$", ".#", role), (int(base.split(".")[1]) if base.startswith("layers.") else None), base


def gemm_cases(inventory, config, template):
    ratios = config["text_config"]["compress_ratios"] if "text_config" in config else config["compress_ratios"]
    want = {}
    for rank, r in enumerate(inventory["ranks"]):
        groups = {}
        for name, _dt, shape, _off, _nb in r["entries"]:
            if not name.endswith("|tr"):
                continue
            role, layer, base = role_of(name)
            if role == "attn.wo_a.slice.#":
                groups[layer] = groups.get(layer, 0) + shape[1] * 16
        for name, _dt, shape, _off, _nb in r["entries"]:
            if not name.endswith("|tr"):
                continue
            role, layer, base = role_of(name)
            if NOT_PROMPT_GEMM.search(base):
                continue
            key = role
            if role.startswith("attn.compressor."):
                key = f"{role}@{ratios[layer]}"
            if key not in PROMPT_GEMM:
                raise SystemExit(f"{base} (rank {rank}): a linear with no prompt-GEMM rule; add it to PROMPT_GEMM or "
                                 f"NOT_PROMPT_GEMM")
            k, n = shape[0] * 16, shape[1] * 16
            out, stride = PROMPT_GEMM[key]
            os_ = n if stride == "n" else groups[layer]
            if k % 32 or n % 128 or os_ % 16:
                raise SystemExit(f"{base}: K {k} N {n} o_stride {os_} outside _gemm's tiling")
            who = want.setdefault((k, n, out, os_), {})
            who.setdefault(key, set()).add(layer)
    cases = []
    for (k, n, out, os_), who in sorted(want.items()):
        c = copy.deepcopy(template)
        c["consts"]["K"], c["consts"]["N"] = k, n
        m = 2048
        c["scalars"] = {"M": m, "o_stride": os_}
        c["tensors"] = {"X": ["float16", [m, k], [k, 1], True], "W": ["float16", [k, n], [n, 1], True],
                        "H": ["bfloat16", [128, 128], [128, 1], True], "SVH": ["float16", [n], [1], True],
                        "BIAS": ["float16", [n], [1], True], "OUT": [out, [m, n], [os_, 1], True]}
        c["grid"] = [(m + 127) // 128 * (n // 128), 1, 1]
        c["phase"] = "derived:prompt_gemm"
        c["derived"] = {"rule": "prompt GEMM of " + "; ".join(
            f"{r} (layers {','.join(str(x) for x in sorted(ls)) if ls != {None} else '-'})" for r, ls in sorted(who.items()))}
        cases.append(c)
    return cases


def with_rows(template, R, row_dims=(), rs_per_row=None):
    """template's launch at R rows: `rows` a constexpr 1 at R = 1 (Triton folds it), else a runtime int; tensors in
    row_dims get R leading rows."""
    c = copy.deepcopy(template)
    rows0 = template["scalars"].get("rows", template["consts"].get("rows"))
    c["consts"].pop("rows", None)
    c["scalars"].pop("rows", None)
    if R == 1:
        c["consts"]["rows"] = 1
    else:
        c["scalars"]["rows"] = R
    for t in row_dims:
        if t in c["tensors"] and c["tensors"][t][1] and c["tensors"][t][1][0] == rows0:
            c["tensors"][t][1][0] = R
    if rs_per_row is not None and c["scalars"].get("RS"):
        c["scalars"]["RS"] = R * rs_per_row
    return c


def configs(cases, drop):
    """One template per distinct recorded configuration (constexprs other than `drop`, dtypes, alignment, warps)."""
    seen = {}
    for c in cases:
        key = json.dumps([{k: v for k, v in c["consts"].items() if k not in drop},
                          {t: [v[0], v[3]] for t, v in c["tensors"].items()}, c["num_warps"]], sort_keys=True)
        seen.setdefault(key, c)
    return list(seen.values())


def cdiv(a, b):
    return (a + b - 1) // b


def hc_rb(R):
    table = [1, 2, 3, 2, 3, 2, 4, 4]
    return table[R - 1] if R <= len(table) else (4 if R <= 12 else 8)


def window_cases(rec):
    out = {}
    # _sparse_attn_merge_rot2: rows classes x POS aligned (the recorded 1-row launch keeps its unaligned POS)
    for t in configs(rec["_sparse_attn_merge_rot2"], {"rows"}):
        for R in range(1, DECODE_ROWS + 1):
            c = with_rows(t, R, ("OUT", "POS", "XH"))
            c["tensors"]["POS"][3] = True
            c["grid"] = [R, t["consts"]["H"], 1]
            c["phase"], c["derived"] = f"derived:window R={R}", {"rule": "merge_rot2 at R rows, POS 16-byte aligned"}
            out.setdefault("_sparse_attn_merge_rot2", []).append(c)
    # _rowmm_parts: RB = R, BN by rpTile
    for t in configs(rec["_rowmm_parts"], {"RB", "BN", "rows"}):
        for R in range(1, DECODE_ROWS + 1):
            c = with_rows(t, R, ("X", "S"))
            c["consts"]["RB"], c["consts"]["BN"] = R, (16 if R <= 6 else 8)
            c["grid"] = [cdiv(t["consts"]["N"], c["consts"]["BN"]), t["consts"]["K"] // 256 // t["consts"]["CPG"], 1]
            c["phase"], c["derived"] = f"derived:window R={R}", {"rule": "rowmm_gate at R rows: RB = R, rpTile BN"}
            out.setdefault("_rowmm_parts", []).append(c)
    # _rowmm2r: RB = min(16, R) past rowmm2_rows = 2
    for t in configs(rec["_rowmm2r"], {"RB", "rows"}):
        for R in range(3, DECODE_ROWS + 1):
            c = with_rows(t, R, ("X", "OUT"))
            c["consts"]["RB"] = min(16, R)
            c["grid"] = [cdiv(R, c["consts"]["RB"]), cdiv(t["consts"]["N"], t["consts"]["BN"]), 1]
            c["phase"], c["derived"] = f"derived:window R={R}", {"rule": "rowmm2 at R rows: RB = min(16, R)"}
            out.setdefault("_rowmm2r", []).append(c)
    # _hc_mix_part: RB = hcRb(R); a posted launch's RS = R * D
    for t in configs(rec["_hc_mix_part"], {"RB", "rows"}):
        d = t["consts"]["D"]
        for R in range(1, DECODE_ROWS + 1):
            c = with_rows(t, R, ("X", "XO", "PART"), rs_per_row=d)
            c["consts"]["RB"] = hc_rb(R)
            c["grid"] = [cdiv(R, c["consts"]["RB"]), t["consts"]["NB"], 3]
            c["phase"], c["derived"] = f"derived:window R={R}", {"rule": "mixPart at R rows: RB = hcRb(R)"}
            out.setdefault("_hc_mix_part", []).append(c)
    return out


# ---- context length: the indexer's launches, by the context a call sees ------------------------------------------
# tri_index.zig's rules (mirrored; tri.conform replays every derived launch through those wrappers):
TILE = 64                 # tri_index.tile
TOPK_FUSED_MAX = 4096     # tri_index.topk_fused_max
BUCKET_MIN = 1024         # round.bucket_min


def pow2(n):
    return n > 0 and n & (n - 1) == 0


def prunes(n, k):
    return k > 0 and n > TOPK_FUSED_MAX and cdiv(n, TILE) > k


def ival(c, name, v):
    """An int argument as Triton specializes it: 1 folds into a constexpr, any other value stays a scalar."""
    if v == 1:
        c["consts"][name] = 1
    else:
        c["scalars"][name] = v


def t64(shape):
    return ["int64", list(shape), [shape[1], 1] if len(shape) == 2 else [1], True]


def topk_select(out, rows, n, k, phase):
    """tri_index.topkSelect with contiguous int64 keys [rows, n]: _topk_sel (power-of-two k <= n <= TOPK_FUSED_MAX) or
    torch's top-k (no Triton launch) and _topk_finish."""
    if pow2(n) and pow2(k) and k <= n <= TOPK_FUSED_MAX:
        out.append(("_topk_sel", {"consts": {"N": n, "KK": k, "PDL": True}, "scalars": {}, "grid": [rows, 1, 1],
                                  "num_warps": 4 if n <= 1024 else 8, "phase": phase,
                                  "tensors": {"KEYS": t64((rows, n)), "VIS": t64((rows,)), "OUT": t64((rows, k))}}))
    else:
        out.append(("_topk_finish", {"consts": {"KK": k, "PDL": True}, "scalars": {}, "grid": [rows, 1, 1],
                                     "num_warps": 4, "phase": phase,
                                     "tensors": {"TOP": t64((rows, k)), "VIS": t64((rows,)), "OUT": t64((rows, k))}}))


def topk_select_pruned(out, rows, n, k, phase):
    """tri_index.topkSelectPruned: the k best 64-key tiles' keys gathered, then their selection."""
    if not prunes(n, k):
        return topk_select(out, rows, n, k, phase)
    nt = cdiv(n, TILE)
    topk_select(out, rows, nt, k, phase)
    c = {"consts": {"BN": TILE, "KK": k, "PDL": True}, "scalars": {}, "grid": [rows, k, 1], "num_warps": 1,
         "phase": phase, "tensors": {"KEYS": t64((rows, n)), "TPOS": t64((rows, k)), "OUT": t64((rows, k * TILE))}}
    ival(c, "n", n)
    ival(c, "ks", n)
    out.append(("_gather_tiles", c))
    topk_select(out, rows, k * TILE, k, phase)


def index_score(out, rows, n, keys, tmax, cand_stride, k, phase):
    """tri_index.indexScore / indexKeys of a round (a base: Triton's _index_score), FP4 keys of 32 heads x 128."""
    nt = cdiv(n, TILE)
    ot = "int64" if keys else "float32"
    c = {"consts": {"IH": 32, "ID": 128, "BN": TILE, "PACKED": True, "HAS_BASE": True, "PDL": True,
                    "HAS_CAND": cand_stride is not None, "CB": 8, "KEYS": keys, "HAS_TMAX": tmax and keys,
                    "SKIP_DEAD": True, "DEAD_PAD": bool(keys and tmax and prunes(n, k))},
         "scalars": {}, "grid": [rows, nt, 1], "num_warps": 4, "phase": phase,
         "tensors": {"Q": ["bfloat16", [rows, 32, 128], [4096, 128, 1], True],
                     "K": ["uint8", [1048592, 64], [64, 1], True], "KS": ["uint8", [1048592, 4], [4, 1], True],
                     "Wt": ["bfloat16", [rows, 32], [32, 1], True], "VIS": t64((rows,)),
                     "OUT": [ot, [rows, n], [n, 1], True], "KB": t64((rows,)),
                     "CAND": (["uint8", [rows, cand_stride], [cand_stride, 1], True] if cand_stride is not None
                              else t64((rows,))),
                     "TMAX": (t64((rows, nt)) if tmax else [ot, [rows, n], [n, 1], True])}}
    ival(c, "n", n)
    c["scalars"]["cs"] = cand_stride or 0
    ival(c, "nt", nt)
    out.append(("_index_score", c))


def score_keys(out, rows, n, tmax, phase):
    nt = cdiv(n, TILE)
    c = {"consts": {"BN": 1024, "HAS_TMAX": tmax, "TB": TILE, "PDL": True}, "scalars": {},
         "grid": [rows, cdiv(n, 1024), 1], "num_warps": 4, "phase": phase,
         "tensors": {"S": ["float32", [rows, n], [n, 1], True], "OUT": t64((rows, n)),
                     "TMAX": t64((rows, nt)) if tmax else t64((rows, n))}}
    ival(c, "n", n)
    ival(c, "nt", nt)
    out.append(("_score_keys", c))


def index_keys_cand(out, rows, n, nblk, cb, phase):
    c = {"consts": {"IH": 32, "ID": 128, "BN": TILE, "PACKED": True, "HAS_BASE": True, "CB": cb, "PDL": True},
         "scalars": {}, "grid": [rows, cdiv(nblk * cb, TILE), 1], "num_warps": 4, "phase": phase,
         "tensors": {"Q": ["bfloat16", [rows, 32, 128], [4096, 128, 1], True],
                     "K": ["uint8", [1048592, 64], [64, 1], True], "KS": ["uint8", [1048592, 4], [4, 1], True],
                     "Wt": ["bfloat16", [rows, 32], [32, 1], True], "VIS": t64((rows,)),
                     "OUT": t64((rows, nblk * cb)), "KB": t64((rows,)),
                     "CBLK": ["int32", [rows, nblk], [nblk, 1], True]}}
    ival(c, "n", n)
    ival(c, "nout", nblk * cb)
    ival(c, "cbs", nblk)
    out.append(("_index_score_cand", c))


def context_cases(config, pool_max):
    """Every indexer launch the served engine makes for a context up to pool_max positions (the RoPE tables' limit):
    - rounds (round.zig): the context bucket b is a power of two from BUCKET_MIN, or the pool's own size (any value
      up to pool_max) when that is smaller; a layer with ratio r scans nb = b / r compressed keys. Layers before the
      candidate source keep top-k keys (index_keys, tile maxima past TOPK_FUSED_MAX); the source scores (index_score),
      builds the pool and keys its scores (score_keys); the layers after it key only the pool's blocks
      (index_keys_cand) past candidate_blocks * candidate_block keys, else their scores masked by it. idx_all (nb <=
      index_topk, not the source) launches nothing. kk = min(index_topk, nb).
    - prompt chunks (prompt.zig indexer): the scores are aux.index_rows' (HIP); a layer before the candidate source
      whose kk = min(index_topk, n_comp) is a power of two selects with topk_select_pruned over n_comp keys, n_comp =
      (start + n) / ratio any value up to pool_max / ratio. The source and later layers select without Triton.
    The row count changes only the grid (16 rows here)."""
    tc = config["text_config"] if "text_config" in config else config
    ratios, topk = tc["compress_ratios"], tc["index_topk"]
    src, cblocks, cb = tc["candidate_source_layer_id"], tc["candidate_topk_blocks"], tc["candidate_block_size"]
    rows = DECODE_ROWS
    out = []
    before_r = sorted({ratios[li] for li in range(len(ratios)) if ratios[li] and li < src})
    after_r = sorted({ratios[li] for li in range(len(ratios)) if ratios[li] and li > src})
    src_r = ratios[src]
    buckets = set()
    b = BUCKET_MIN
    while b <= pool_max:
        buckets.add(b)
        b *= 2
    buckets |= set(range(BUCKET_MIN, pool_max + 1))       # a pool smaller than the next power of two: its size
    for b in sorted(buckets):
        ph = f"round:r{rows}:derived-b{b}"
        for r in before_r:
            nb = b // r
            if nb <= topk:
                continue
            kk = min(topk, nb)
            tm = nb > TOPK_FUSED_MAX
            index_score(out, rows, nb, True, tm, None, kk, ph)
            (topk_select_pruned if tm else topk_select)(out, rows, nb, kk, ph)
        nb = b // src_r
        kk = min(topk, nb)
        if nb > cblocks * cb or nb % cb == 0:              # else round.zig: not ported (_candidates' padded path)
            index_score(out, rows, nb, False, False, None, kk, ph)
            tm = nb > TOPK_FUSED_MAX
            score_keys(out, rows, nb, tm, ph)
            (topk_select_pruned if tm else topk_select)(out, rows, nb, kk, ph)
            for r in after_r:
                na = b // r
                ka = min(topk, na)
                if nb > cblocks * cb:
                    index_keys_cand(out, rows, na, cblocks, cb, ph)
                    topk_select(out, rows, cblocks * cb, ka, ph)
                elif na <= cblocks * cb and na > topk:
                    tm = na > TOPK_FUSED_MAX
                    index_score(out, rows, na, True, tm, na // cb, ka, ph)
                    (topk_select_pruned if tm else topk_select)(out, rows, na, ka, ph)
    for r in before_r:
        for n in range(1, pool_max // r + 1):
            kk = min(topk, n)
            if pow2(kk):
                topk_select_pruned(out, rows, n, kk, f"forward:derived-ncomp{n}")
    # one launch a specialization (an int argument's class: 1, a multiple of 16, other)
    seen, kept = set(), {}
    for fn, c in out:
        cls = {k: (v if isinstance(v, dict) else ("1" if v == 1 else "16" if v % 16 == 0 else "x"))
               for k, v in c["scalars"].items()}
        key = json.dumps([fn, c["consts"], cls, {t: [v[0], v[3]] for t, v in c["tensors"].items()}], sort_keys=True)
        if key not in seen:
            seen.add(key)
            c["derived"] = {"rule": f"context length ({c['phase']})"}
            kept.setdefault(fn, []).append(c)
    return kept


# ---- offset pointers: the prompt's replay cut ---------------------------------------------------------------------
# prompt.replayCut moves a chunk to its rows at positions >= replay: h and pre are copied down into fresh buffers, but
# the positions become the view pos[first:], first * 8 bytes into the int64 buffer, 16-byte aligned only for an even
# first. Every later launch of that chunk that takes ch.pos gets it: tri_norm.ropeHeads (the heads' q, the inverse on
# o, the indexer's q) and tri_norm.kvNormRope (ring mode or not), and the attention (aux.sparse_attn on the served
# HIP lane: no Triton launch). The chunk's other row pointers are fresh buffers or row offsets of >= 1,024 bytes
# (16-byte aligned); round inputs are padded to even rows (05832c5) and the drafter's rows are max_rows apart.
OFFSET_POS = ("_rope_heads", "_kv_norm_rope")


def offset_cases(rec):
    """Each prompt-chunk configuration ("forward" phase) of a kernel taking ch.pos, with POS at an odd cut too."""
    out = {}
    for fn in OFFSET_POS:
        for t in configs([c for c in rec[fn] if c["phase"] == "forward"], set()):
            c = copy.deepcopy(t)
            c["tensors"]["POS"][3] = False
            c["phase"] = "forward:replay-cut odd first"
            c["derived"] = {"rule": "prompt.replayCut: POS = pos[first:], first odd (first * 8 bytes in)"}
            out.setdefault(fn, []).append(c)
    return out


# The DSpark Markov loop (tri_markov.Markov.localBest / pick): step i reads its tokens at out + 8 i (_bias, _score: TOK,
# row stride ts) and writes step i + 1's at out + 8 (i + 1) (_finish, _pick: OUT). The recording ran step 0 only (TOK
# aligned, OUT 8 bytes in), so the odd steps read TOK 8 bytes in and write OUT aligned. Markov.fill's _bias (t_stride
# 1: tok + 8 r0, r0 a multiple of NP = 16) stays aligned.
MARKOV_STEP = (("_bias", "TOK", False), ("_score", "TOK", False), ("_finish", "OUT", True), ("_pick", "OUT", True))


def markov_cases(rec):
    """Each recorded Markov step launch with the other alignment of its step-indexed pointer (an odd step)."""
    out = {}
    for fn, arg, aligned in MARKOV_STEP:
        for t in rec[fn]:
            if t["scalars"].get("t_stride") == 1:
                continue
            c = copy.deepcopy(t)
            c["tensors"][arg][3] = aligned
            c["phase"] = f"{t['phase']}:markov odd step"
            at = "out + 8 i" if arg == "TOK" else "out + 8 (i + 1)"
            c["derived"] = {"rule": f"tri_markov.localBest / pick, step i odd: {arg} = {at} ({'aligned' if aligned else '8 bytes in'})"}
            out.setdefault(fn, []).append(c)
    return out


def variant_key(fn, case):
    """aot_build's distinct-variant key of a launch (constexprs, runtime parameters' types and specialization)."""
    import aot_build
    import tri_kernels as K
    _sig, _ce, _attrs, params, consts_json = aot_build.plan_variant(getattr(K, fn), case)
    return fn, json.dumps(params, sort_keys=True), json.dumps(consts_json, sort_keys=True), case["num_warps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", default=str(BUNDLE / "inventory.json"))
    ap.add_argument("--config", default=str(BUNDLE / "model" / "config.json"))
    ap.add_argument("--out", default=str(DERIVED))
    ap.add_argument("--pool-max", type=int, default=1048527, help="the longest pool served (the RoPE tables' rows)")
    a = ap.parse_args()
    sys.path.insert(0, str(HERE))
    rec = json.loads(RECORDED.read_text())
    inv_p, cfg_p = Path(a.inventory), Path(a.config)
    src = {"inventory_sha256": hashlib.sha256(inv_p.read_bytes()).hexdigest(),
           "config_sha256": hashlib.sha256(cfg_p.read_bytes()).hexdigest()}
    every = {"_gemm": gemm_cases(json.loads(inv_p.read_text()), json.loads(cfg_p.read_text()), rec["_gemm"][0])}
    every.update(window_cases(rec))
    for fn, cases in offset_cases(rec).items():
        every.setdefault(fn, []).extend(cases)
    for fn, cases in markov_cases(rec).items():
        every.setdefault(fn, []).extend(cases)
    for fn, cases in context_cases(json.loads(cfg_p.read_text()), a.pool_max).items():
        every.setdefault(fn, []).extend(cases)
    # only launches the recording's variants do not already compile
    seen = {variant_key(fn, c) for fn in every for c in rec[fn]}
    derived = {}
    for fn, cases in sorted(every.items()):
        for c in cases:
            key = variant_key(fn, c)
            if key in seen:
                continue
            seen.add(key)
            c["derived"].update(src)
            derived.setdefault(fn, []).append(c)
    Path(a.out).write_text(json.dumps(derived, indent=1, sort_keys=True) + "\n")
    for fn, cases in sorted(derived.items()):
        print(f"{fn}: {len(cases)} new variants")
        for c in cases:
            k = c["consts"]
            if fn == "_gemm":
                what = f"K={k['K']} N={k['N']} OUT={c['tensors']['OUT'][0]} o_stride={c['scalars']['o_stride']}"
            else:
                what = (f"{c['phase']} RB={k.get('RB', '-')} rows={c['scalars'].get('rows', k.get('rows'))} "
                        + " ".join(f"{x}={k[x]}" for x in ("N", "BN", "PDL", "POSTED", "WORLD") if x in k)
                        + (f" POS aligned={c['tensors']['POS'][3]}" if "POS" in c["tensors"] else ""))
            print(f"  {what}: {c['derived']['rule'][:160]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""a_grade.py KIND ...: the verdicts of the served-config parity checks (PLAN-research-actions.md sections 1 and 4), from
lane_gate.sh's and a_served.sh's output directories, the plan's rules applied as written. Standard library only. Each
writes its verdict JSON (next to the first directory, or --out) and prints it; exit 0 only on pass.

  lane DIR --tag T --conf C --seconds S --timed-out 0|1
        (lane_gate.sh) DIR/lane.json: each request's rows and logits sha256 (rank 0's --logits-sha lines), reply
        token_sha, the ranks' startup lines (replay_floor, side, engram_aio), errors, fallbacks (a launch that ran a
        variant that assumes less, or had none: MissingTritonVariant). ok: every request served, rank 1 followed to the
        end, no error, no fallback, every row count equal to its prompt's.
  a1 ON OFF [--more-off DIR...]
        A1's stale-state test on the lane gate. Determinism control: det.X1 == det.X2 within each launch (else STOP:
        the sha method cannot work as designed), X with the flag on == off. Pass (flag on): B's sha after A == after A'
        for every P_B. Sensitivity (flag off): after A != after A' for at least one P_B. Rows: each B's == P_B. The
        startup lines must say replay_floor true (ON) / false (OFF) on both ranks. --more-off: further flag-off launches
        of the same build (cross-launch determinism of X with the same flags).
  a1served DIR
        A1's served check: every request 200 with prompt_tokens == its rows; the determinism control (X's token_sha
        equal); B's token_sha after A == after A' for every P_B; rank 1's startup line says replay_floor true.
  a4 OFF ON
        A4's served checks. OFF (no TF_TOP_K_MAX): omitted, 0 and 100 fail, 20 succeeds. ON (TF_TOP_K_MAX=56,
        TF_SEED_RANDOM=1): all four succeed, the effective k recorded (omitted and 0 equal the explicit 20's reply,
        100 the explicit 56's; and the server's clamp warning); the same seed twice gives identical tokens. Both: the
        greedy probe after the failures answers. Recorded too: the unseeded pair (equal without TF_SEED_RANDOM,
        different with it) and the request without max_tokens.
  a23lane OFF SIDE AIO
        A2/A3's lane checks: side against off, the code and prose 400-token replies' token_sha identical (the prompt
        shas are equal by construction: a null check, recorded); aio against off, the 2K/4K/8K prompt shas and the
        two replies' token_sha identical. Each launch's determinism control; X across the three launches.
  layers OFF ON
        A2/A3's layer-gate dumps (check_layers.sh --dump DIRs): every file byte-identical, the same file set.
  speed --session FILE
        Section 4's decision rule on the A2+A3 speed session: FILE lists the launches in order (one JSON line each:
        {"round", "arm": off|side|aio|off_close|both_close, "dir"}).
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import sys

FALLBACK = re.compile(r"assumes less|Triton variant of|MissingTritonVariant")
STAMP = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}Z ")  # TF_LOG_TIME's line stamp
P_B = (2050, 2062, 2064, 4100)


def jl(path):
    out = []
    if not os.path.exists(path):
        return out
    for line in open(path, errors="replace"):
        line = STAMP.sub("", line.strip(), count=1)
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def emit(doc, path):
    json.dump(doc, open(path, "w"), indent=1)
    print(json.dumps({k: doc[k] for k in doc if k in ("kind", "status", "reasons", "stop")}, indent=None))
    sys.exit(0 if doc.get("status") == "pass" else 1)


# ---- lane --------------------------------------------------------------------------------------------------------
def lane(a):
    d = a.dirs[0]
    reqs = json.load(open(os.path.join(d, "requests.json")))["requests"]
    r0, r1 = jl(os.path.join(d, "rank0.jsonl")), jl(os.path.join(d, "rank1.jsonl"))
    e0 = jl(os.path.join(d, "rank0.err"))
    shas = {x["prompt_logits"]: x for x in e0 if "prompt_logits" in x}
    rows = []
    for i, q in enumerate(reqs):
        adm = next((x for x in r0 if x.get("request") == i and "prompt_tokens" in x), {})
        fin = next((x for x in r0 if x.get("request") == i and "token_sha" in x), {})
        s = shas.get(q["name"], {})
        rows.append({"name": q["name"], "prompt": len(q["prompt"]), "max_tokens": q["max_tokens"],
                     "prompt_tokens": adm.get("prompt_tokens"), "rows": s.get("rows"), "sha256": s.get("sha256"),
                     "prefill_s": adm.get("prefill_s"), "token_sha": fin.get("token_sha"), "tokens": fin.get("tokens"),
                     "reason": fin.get("reason"), "seconds": fin.get("seconds"), "reply": fin.get("reply")})
    errors = [x for x in r0 + r1 if "error" in x or "follow" in x]
    fallbacks = []
    for f in ("rank0.err", "rank1.err"):
        p = os.path.join(d, f)
        if os.path.exists(p):
            fallbacks += [f"{f}: {l.strip()}" for l in open(p, errors="replace") if FALLBACK.search(l)]
    startup = {"rank0": next((x for x in r0 if "loaded_s" in x), None), "rank1": next((x for x in r1 if "loaded_s" in x), None)}
    summary = next((x for x in r0 if "all_equal" in x and "requests" in x), None)
    followed = any(x.get("followed") for x in r1)
    reasons = []
    if a.timed_out == "1":
        reasons.append("timed out")
    if not summary:
        reasons.append("rank 0 did not finish the requests")
    if not followed:
        reasons.append("rank 1 did not follow to the end")
    if errors:
        reasons.append(f"errors: {errors}")
    if fallbacks:
        reasons.append(f"fallbacks: {fallbacks[:5]}")
    for r in rows:
        if r["token_sha"] is None:
            reasons.append(f"{r['name']}: no reply")
        if r["sha256"] is None:
            reasons.append(f"{r['name']}: no logits sha")
        elif r["rows"] != r["prompt"] or r["prompt_tokens"] != r["prompt"]:
            reasons.append(f"{r['name']}: rows {r['rows']} / prompt_tokens {r['prompt_tokens']}, the prompt has {r['prompt']}")
    dep = None
    try:
        dep = json.load(open(os.path.join(d, "resolved-rank0.json")))
    except Exception:
        pass
    doc = {"kind": "lane", "tag": a.tag, "status": "pass" if not reasons else "fail", "reasons": reasons, "seconds": a.seconds,
           "conf": a.conf, "conf_sha256": hashlib.sha256(open(a.conf, "rb").read()).hexdigest() if a.conf and os.path.exists(a.conf) else None,
           "deploy": dep and dep.get("deploy"), "kit_manifest_sha256": dep and dep.get("kit_manifest_sha256"),
           "env": dep and {k: v for k, v in dep["env"].items() if k.startswith("TF_")}, "argv": dep and dep.get("argv"),
           "startup": startup, "summary": summary, "requests": rows,
           "load": open(os.path.join(d, "load.txt")).read().splitlines() if os.path.exists(os.path.join(d, "load.txt")) else None}
    emit(doc, os.path.join(d, "lane.json"))


def lanedoc(d):
    p = os.path.join(d, "lane.json")
    if not os.path.exists(p):
        raise SystemExit(f"no {p}: run lane_gate.sh first")
    doc = json.load(open(p))
    doc["by"] = {r["name"]: r for r in doc["requests"]}
    return doc


def det(doc, reasons, label):
    x1, x2 = doc["by"]["det.X1"]["sha256"], doc["by"]["det.X2"]["sha256"]
    if x1 is None or x1 != x2:
        reasons.append(f"STOP: {label}: X's prompt-logits sha differs within the launch ({x1} / {x2})")
        return None
    return x1


def startup_is(doc, reasons, label, **want):
    for rk in ("rank0", "rank1"):
        s = doc["startup"].get(rk) or {}
        for k, v in want.items():
            if s.get(k) != v:
                reasons.append(f"{label}: {rk} startup {k}={s.get(k)}, want {v}")


# ---- A1 ----------------------------------------------------------------------------------------------------------
def a1(a):
    on, off = lanedoc(a.dirs[0]), lanedoc(a.dirs[1])
    more = [lanedoc(x) for x in a.more_off]
    reasons, stop = [], []
    for doc, label in [(on, "on"), (off, "off")] + [(m, f"off#{i + 2}") for i, m in enumerate(more)]:
        if doc["status"] != "pass":
            reasons.append(f"{label}: launch failed: {doc['reasons']}")
    startup_is(on, reasons, "on", replay_floor=True, side=False, engram_aio=False)
    startup_is(off, reasons, "off", replay_floor=False, side=False, engram_aio=False)
    for m in more:
        startup_is(m, reasons, "off#", replay_floor=False, side=False, engram_aio=False)
    xs = {"on": det(on, stop, "on"), "off": det(off, stop, "off")}
    for i, m in enumerate(more):
        xs[f"off#{i + 2}"] = det(m, stop, f"off#{i + 2}")
    table, sens = [], []
    for p in P_B:
        b_on = (on["by"][f"B{p}.afterA"], on["by"][f"B{p}.afterA'"])
        b_off = (off["by"][f"B{p}.afterA"], off["by"][f"B{p}.afterA'"])
        for r in b_on + b_off:
            if r["rows"] != p:
                reasons.append(f"B{p}: rows {r['rows']}")
        eq_on = b_on[0]["sha256"] is not None and b_on[0]["sha256"] == b_on[1]["sha256"]
        eq_off = b_off[0]["sha256"] == b_off[1]["sha256"]
        sens.append(not eq_off)
        if not eq_on:
            reasons.append(f"B{p}: flag on, sha after A {b_on[0]['sha256']} != after A' {b_on[1]['sha256']}")
        table.append({"P_B": p, "last_chunk_rows": p % 2048, "on_afterA": b_on[0]["sha256"], "on_afterA'": b_on[1]["sha256"],
                      "on_equal": eq_on, "off_afterA": b_off[0]["sha256"], "off_afterA'": b_off[1]["sha256"], "off_equal": eq_off,
                      "on_vs_off_afterA_equal": b_on[0]["sha256"] == b_off[0]["sha256"],
                      "replies_on": [b_on[0]["token_sha"], b_on[1]["token_sha"]], "replies_off": [b_off[0]["token_sha"], b_off[1]["token_sha"]]})
    sensitive = any(sens)
    if not sensitive:
        reasons.append("not proven sensitive: with the flag off, B's sha after A equals after A' for every P_B (the flag is not adopted on this evidence; the lead reports to the owner)")
    x_on_off = xs["on"] is not None and xs["on"] == xs["off"]
    if xs["on"] is not None and xs["off"] is not None and not x_on_off:
        reasons.append(f"X differs between the flag-on and flag-off launches ({xs['on']} / {xs['off']}): either the flag "
                       "reached X or X differs across launches (cross-launch comparisons fall back to the layer-gate dumps)")
    cross_same = None
    if more and xs["off"] is not None:
        cross_same = all(xs[k] == xs["off"] for k in xs if k.startswith("off#"))
    reasons = stop + reasons
    doc = {"kind": "a1", "status": "pass" if not reasons else "fail", "stop": bool(stop), "reasons": reasons,
           "on": a.dirs[0], "off": a.dirs[1], "more_off": a.more_off, "x_sha": xs, "x_on_equals_off": x_on_off,
           "x_cross_launch_same_flags": cross_same, "pass_flag_on": all(t["on_equal"] for t in table), "sensitive_flag_off": sensitive,
           "table": table,
           "refs": {n: on["by"].get(n, {}).get("sha256") for n in ("ref2050", "ref4096")},
           "ref_dumps": sorted(os.listdir(os.path.join(a.dirs[0], "logits"))) if os.path.isdir(os.path.join(a.dirs[0], "logits")) else None}
    emit(doc, a.out or os.path.join(a.dirs[0], "a1.json"))


def served(d):
    p = os.path.join(d, "served.json")
    if not os.path.exists(p):
        raise SystemExit(f"no {p}")
    doc = json.load(open(p))
    doc["by"] = {r["name"]: r for r in doc.get("results", [])}
    return doc


def rank1_startup(d):
    return next((x for x in jl(os.path.join(d, "rank1.log")) if "loaded_s" in x), None)


def a1served(a):
    d = a.dirs[0]
    s = served(d)
    reasons = []
    for r in s["results"]:
        pt = (r.get("usage") or {}).get("prompt_tokens")
        if r["status"] != 200 or pt != r["rows"] or not r.get("token_sha"):
            reasons.append(f"{r['name']}: status {r['status']}, prompt_tokens {pt} (rows {r['rows']}), token_sha {r.get('token_sha')}, error {r.get('error')}")
    st = rank1_startup(d)
    if not st or st.get("replay_floor") is not True:
        reasons.append(f"rank 1's startup line does not say replay_floor true: {st}")
    by = s["by"]
    x_equal = by.get("det.X1", {}).get("token_sha") == by.get("det.X2", {}).get("token_sha") and by.get("det.X1", {}).get("token_sha")
    if not x_equal:
        reasons.append("determinism control: X's reply token_sha differs within the launch")
    table = []
    for p in P_B:
        ta, tb = by.get(f"B{p}.afterA", {}).get("token_sha"), by.get(f"B{p}.afterA'", {}).get("token_sha")
        table.append({"P_B": p, "afterA": ta, "afterA'": tb, "equal": ta is not None and ta == tb})
        if not table[-1]["equal"]:
            reasons.append(f"B{p}: token_sha after A {ta} != after A' {tb}")
    doc = {"kind": "a1served", "status": "pass" if not reasons else "fail", "reasons": reasons, "dir": d, "rank1_startup": st,
           "x_token_sha": [by.get("det.X1", {}).get("token_sha"), by.get("det.X2", {}).get("token_sha")], "table": table}
    emit(doc, a.out or os.path.join(d, "a1served.json"))


# ---- A4 ----------------------------------------------------------------------------------------------------------
def a4(a):
    off, on = served(a.dirs[0]), served(a.dirs[1])
    reasons = []

    def ok(r):
        return r.get("status") == 200 and r.get("error") is None and bool(r.get("ids"))

    o, n = off["by"], on["by"]
    for k in ("k_omitted", "k0", "k100"):
        if ok(o[k]):
            reasons.append(f"without TF_TOP_K_MAX, {k} succeeded (expected to fail)")
    if not ok(o["k20"]):
        reasons.append(f"without TF_TOP_K_MAX, k20 failed: {o['k20'].get('status')} {o['k20'].get('error')}")
    for k in ("k20", "k_omitted", "k0", "k100"):
        if not ok(n[k]):
            reasons.append(f"with TF_TOP_K_MAX=56, {k} failed: {n[k].get('status')} {n[k].get('error')}")

    def eff(r):
        if not ok(r):
            return None
        m = [k for k, ref in ((20, n["k20"]), (56, n["k56"])) if ok(ref) and ref["ids"] == r["ids"]]
        return m[0] if len(m) == 1 else (m or None)
    effective = {k: eff(n[k]) for k in ("k_omitted", "k0", "k20", "k100")}
    # the discriminating seeds (a_served.py --discriminate): where the explicit 20 and 56 replies differ, each other
    # variant's reply names its effective k
    seeds = sorted({r.split(".")[0] for r in n if r.startswith("d") and "." in r})
    disc = {}
    for sd in seeds:
        r20, r56 = n.get(f"{sd}.k20"), n.get(f"{sd}.k56")
        if not (r20 and r56 and ok(r20) and ok(r56)) or r20["ids"] == r56["ids"]:
            disc[sd] = "20 and 56 give the same reply: no discrimination"
            continue
        disc[sd] = {v: (20 if ok(n[f"{sd}.{v}"]) and n[f"{sd}.{v}"]["ids"] == r20["ids"] else 56 if ok(n[f"{sd}.{v}"]) and n[f"{sd}.{v}"]["ids"] == r56["ids"] else None)
                    for v in ("k_omitted", "k0", "k100")}
    used = [d for d in disc.values() if isinstance(d, dict)]
    if used:
        effective = {v: (sorted({d[v] for d in used}, key=str)[0] if len({d[v] for d in used}) == 1 else sorted({d[v] for d in used}, key=str)) for v in ("k_omitted", "k0", "k100")}
        effective["k20"] = 20
    if effective != {"k_omitted": 20, "k0": 20, "k20": 20, "k100": 56}:
        reasons.append(f"effective k {effective}, want 20, 20, 20, 56" + ("" if used else " (no seed told 20 from 56)"))
    for label, side in (("without", o), ("with", n)):
        if not (ok(side["seed777_a"]) and side["seed777_a"]["ids"] == side["seed777_b"]["ids"]):
            reasons.append(f"{label} TF_TOP_K_MAX: the same seed twice did not give identical tokens")
        if not ok(side["probe_after"]):
            reasons.append(f"{label} TF_TOP_K_MAX: the greedy probe after the failures did not answer")
    clamp = None
    lp = os.path.join(a.dirs[1], "rank0.log")
    if os.path.exists(lp):
        clamp = [l.strip() for l in open(lp, errors="replace") if "sampler limit" in l]
    info = {"other_seed_differs": {s: n2["seed99"].get("ids") != n2["seed777_a"].get("ids") for s, n2 in (("off", o), ("on", n))},
            "unseeded_pair_equal": {s: (ok(x["unseeded_a"]) and x["unseeded_a"]["ids"] == x["unseeded_b"]["ids"]) for s, x in (("off", o), ("on", n))},
            "no_max_tokens": {s: {"status": x["no_max_tokens"]["status"], "error": x["no_max_tokens"].get("error"), "finish": x["no_max_tokens"].get("finish"),
                                  "completion_tokens": (x["no_max_tokens"].get("usage") or {}).get("completion_tokens")} for s, x in (("off", o), ("on", n))},
            "failures_off": {k: {"status": o[k]["status"], "error": o[k].get("error")} for k in ("k_omitted", "k0", "k100")},
            "server_clamp_warning": clamp}
    doc = {"kind": "a4", "status": "pass" if not reasons else "fail", "reasons": reasons, "off": a.dirs[0], "on": a.dirs[1],
           "effective_k": effective, "discrimination": disc, "info": info}
    emit(doc, a.out or os.path.join(a.dirs[1], "a4.json"))


# ---- A2/A3 -------------------------------------------------------------------------------------------------------
def a23lane(a):
    off, side, aio = (lanedoc(x) for x in a.dirs[:3])
    reasons, stop = [], []
    for doc, label in ((off, "off"), (side, "side"), (aio, "aio")):
        if doc["status"] != "pass":
            reasons.append(f"{label}: launch failed: {doc['reasons']}")
    startup_is(off, reasons, "off", side=False, engram_aio=False)
    startup_is(side, reasons, "side", side=True, engram_aio=False)
    startup_is(aio, reasons, "aio", side=False, engram_aio=True)
    xs = {"off": det(off, stop, "off"), "side": det(side, stop, "side"), "aio": det(aio, stop, "aio")}
    x_cross = len(set(xs.values())) == 1 and None not in xs.values()
    if not stop and not x_cross:
        reasons.append(f"X differs across launches {xs}: cross-launch identity falls back to the layer-gate dumps (section 1)")
    cmp = {}
    for label, doc, prompts, replies in (("side", side, ("long2k", "long4k", "long8k", "code", "prose"), ("code", "prose")),
                                         ("aio", aio, ("long2k", "long4k", "long8k", "code", "prose"), ("code", "prose"))):
        c = {}
        for p in prompts:
            c[f"{p}.prompt_sha"] = off["by"][p]["sha256"] == doc["by"][p]["sha256"]
        for p in replies:
            # (max_tokens 400; a reply may end at EOS first: prose did at 393 on every launch so far)
            c[f"{p}.reply_token_sha"] = off["by"][p]["token_sha"] == doc["by"][p]["token_sha"] and off["by"][p]["tokens"] == doc["by"][p]["tokens"]
        cmp[label] = c
        required = [k for k in c if k.endswith("reply_token_sha")] + ([k for k in c if k.startswith("long")] if label == "aio" else [])
        for k in required:
            if not c[k]:
                reasons.append(f"{label}: {k} differs from off")
    reasons = stop + reasons
    doc = {"kind": "a23lane", "status": "pass" if not reasons else "fail", "stop": bool(stop), "reasons": reasons,
           "dirs": a.dirs[:3], "x_sha": xs, "x_cross_launch_equal": x_cross, "compare": cmp,
           "note": "side: prompt shas are equal by construction (prompt.zig has no side use), recorded as a null check"}
    emit(doc, a.out or os.path.join(a.dirs[1], "a23lane.json"))


def files(root):
    out = {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(dp, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(1 << 20), b""):
                    h.update(b)
            out[os.path.relpath(p, root)] = h.hexdigest()
    return out


def layers(a):
    off, on = files(a.dirs[0]), files(a.dirs[1])
    only_off, only_on = sorted(set(off) - set(on)), sorted(set(on) - set(off))
    diff = sorted(k for k in set(off) & set(on) if off[k] != on[k])
    reasons = []
    if not off:
        reasons.append("no dump files")
    if only_off or only_on:
        reasons.append(f"file sets differ: only off {only_off[:5]}, only on {only_on[:5]}")
    if diff:
        reasons.append(f"{len(diff)} files differ: {diff[:10]}")
    doc = {"kind": "layers", "status": "pass" if not reasons else "fail", "reasons": reasons, "off": a.dirs[0], "on": a.dirs[1],
           "files": len(off), "differ": diff}
    emit(doc, a.out or os.path.join(os.path.dirname(a.dirs[1].rstrip("/")), "layers-compare.json"))


# ---- section 4 ---------------------------------------------------------------------------------------------------
def metrics(d):
    s = json.load(open(os.path.join(d, "served.json")))
    pf = s.get("prefill") or {}
    return {"prefill": pf.get("prefill_tok_per_s"), "code": (s.get("code") or {}).get("decode_tok_per_s"),
            "prose": (s.get("prose") or {}).get("decode_tok_per_s"), "prefill_tokens": (pf.get("usage") or {}).get("prompt_tokens"),
            "cached": pf.get("cached_tokens")}


def decide(rows, arm, targets):
    """Section 4's rule for `arm` against off: per round pairs; adopt only if >= 4 of 5 favour it on a target metric
    (decode: both code and prose), its median gain there >= max(2%, the off arm's range across the session), and no
    other metric's median regresses by more than the off arm's range."""
    offs = [r for r in rows if r["arm"] == "off"]
    ons = [r for r in rows if r["arm"] == arm]
    pairs = [(o, n) for o in offs for n in ons if o["round"] == n["round"]]
    mets = ("prefill", "code", "prose")
    rng = {}
    for m in mets:
        v = [r["m"][m] for r in offs if r["m"][m] is not None]
        rng[m] = (max(v) - min(v)) / statistics.median(v) if len(v) > 1 else None
    gain = {m: [n["m"][m] / o["m"][m] - 1 for o, n in pairs if o["m"][m] and n["m"][m]] for m in mets}
    med = {m: statistics.median(g) if g else None for m, g in gain.items()}
    res = {"arm": arm, "pairs": len(pairs), "off_range": rng, "median_gain": med, "gains": gain, "targets": {}}
    adopt = False
    for t in targets:
        group = ("code", "prose") if t == "decode" else (t,)
        favour = sum(1 for o, n in pairs if all(n["m"][m] and o["m"][m] and n["m"][m] > o["m"][m] for m in group))
        bar = {m: max(0.02, rng[m] or 0) for m in group}
        gain_ok = all(med[m] is not None and med[m] >= bar[m] for m in group)
        others = [m for m in mets if m not in group]
        regress = {m: med[m] is not None and rng[m] is not None and med[m] < -rng[m] for m in others}
        passed = favour >= 4 and gain_ok and not any(regress.values())
        res["targets"][t] = {"favour": favour, "of": len(pairs), "bar": bar, "gain_ok": gain_ok, "regress": regress, "adopt": passed}
        adopt = adopt or passed
    res["adopt"] = adopt
    return res


def speed(a):
    rows = [json.loads(l) for l in open(a.session) if l.strip()]
    for r in rows:
        r["m"] = metrics(r["dir"])
    side = decide(rows, "side", ("decode",))
    aio = decide(rows, "aio", ("prefill", "decode"))
    close = {r["arm"]: r["m"] for r in rows if r["arm"] in ("off_close", "both_close")}
    verdict = {"side": side["adopt"], "aio": aio["adopt"]}
    combo = None
    if side["adopt"] and aio["adopt"] and len(close) == 2:
        o, b = close["off_close"], close["both_close"]
        reg = {m: (b[m] / o[m] - 1) for m in ("prefill", "code", "prose") if o[m] and b[m]}
        bad = {m: g < -(side["off_range"][m] or 0) for m, g in reg.items()}
        target_bad = bad.get("code") or bad.get("prose") or bad.get("prefill")
        if target_bad:
            better = "side" if (statistics.median([side["median_gain"]["code"], side["median_gain"]["prose"]]) >=
                                statistics.median([aio["median_gain"]["code"], aio["median_gain"]["prose"]])) else "aio"
            verdict = {"side": better == "side", "aio": better == "aio"}
        combo = {"closing_pair_change": reg, "regressed_beyond_off_range": bad}
    doc = {"kind": "speed", "status": "pass", "verdict": verdict, "side": side, "aio": aio, "closing": close, "combination": combo,
           "launches": [{k: r[k] for k in ("round", "arm", "dir", "m")} for r in rows]}
    emit(doc, a.out or os.path.join(os.path.dirname(a.session), "speed.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=("lane", "a1", "a1served", "a4", "a23lane", "layers", "speed"))
    ap.add_argument("dirs", nargs="*")
    ap.add_argument("--tag")
    ap.add_argument("--conf")
    ap.add_argument("--seconds")
    ap.add_argument("--timed-out", default="0")
    ap.add_argument("--more-off", nargs="*", default=[])
    ap.add_argument("--session")
    ap.add_argument("--out")
    a = ap.parse_args()
    globals()[a.kind](a)


if __name__ == "__main__":
    main()

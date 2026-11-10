#!/bin/bash
# served_faults.sh CONF BASE_ID [--cases LIST] [--long 0|1]: the served fault matrix (handoff section 6D, gates
# M3.fault-matrix, M3.handshake, M3.stale-cache-refusal). Each case is its own run under both GPU locks (node0's first,
# released after the case): both ranks from CONF ('$HOME/...', the same file on both boxes), a fault, and the verdict.
#   kill_r1_decode / kill_r0_decode     SIGKILL a rank while a streamed reply decodes
#   stop_r1_decode / stop_r0_decode     SIGSTOP a rank for STOP_S (10 s, past the 2 s progress deadline), then SIGCONT
#   kill_r1_prefill / stop_r1_prefill   the same inside a long prompt's prefill (--long 1: prompts over 128 rows)
#   kill_r1_load                        SIGKILL rank 1 while both ranks load: rank 0 must give up, not wait forever
#   mm_round_rows / mm_aux / mm_streams / mm_context / mm_pool / mm_kit / mm_strict / mm_conf (any other configuration byte: U46)
#                                       one rank from a configuration that differs in one setting: the peers' handshake
#                                       must refuse before serving, naming it
#   stale_bundle                        the configuration pins another input bundle hash: the launcher refuses
#   stale_cache                         rank 1's cache directory holds the right bytes under another content key's name
#                                       (a plausible stale file): it must be refused, not loaded
# Pass for a fault while a request runs: the client's stream ends without a finish (error event or closed connection)
# within DEADLINE_S of the fault (of SIGCONT for a stopped rank), at most one token arrives after the fault, rank 1
# (partner or resumed) leaves within DEADLINE_S, and rank 0, if it lives, answers a new request with an error rather
# than a reply or a hang and does not report itself healthy on /health (coordinated failure, handoff 6D). Writes ~/tfp/qual/out/BASE_ID/faults.json; exit 0 only if every case that ran passed.
set -uo pipefail
QCONF_BASE=${1:?CONF}; BASE=${2:?BASE_ID}; shift 2
CASES="kill_r1_decode,kill_r0_decode,stop_r1_decode,stop_r0_decode,kill_r1_prefill,stop_r1_prefill,kill_r1_load,mm_round_rows,mm_aux,mm_streams,mm_context,mm_pool,mm_kit,mm_strict,mm_conf,stale_bundle,stale_cache"
LONG=0
while [ $# -gt 0 ]; do
  case $1 in --cases) CASES=$2; shift 2;; --long) LONG=$2; shift 2;; *) echo "unknown $1" >&2; exit 2;; esac
done
export QLOCK_MAX_S=${QLOCK_MAX_S:-900}   # one case: a load, a fault and the teardown
DEADLINE_S=${DEADLINE_S:-15}   # the progress deadline is 2 s (COLL_PROGRESS_MS); 15 s bounds detection plus teardown
STOP_S=${STOP_S:-10}
QDIR=$(cd "$(dirname "$0")" && pwd)
QOUT_ROOT=$HOME/tfp/qual/out/$BASE
mkdir -p "$QOUT_ROOT"
export QOUT_ROOT
RESULTS=$QOUT_ROOT/faults.jsonl
: > "$RESULTS"

# variant CONF_NAME KEY=VALUE...: a copy of the base configuration with the values replaced, on both boxes
variant() {
  local name=$1; shift
  local base1=${QCONF_BASE//\$HOME/$HOME}
  python3 "$QDIR/mkconf.py" "$base1" "$HOME/tfp/qual/conf/$name.conf" "$@" || return 1
  ssh -n node0 "mkdir -p ~/tfp/qual/conf"
  scp -q "$HOME/tfp/qual/conf/$name.conf" "node0:tfp/qual/conf/$name.conf"
  echo "\$HOME/tfp/qual/conf/$name.conf"
}

record() {  # record NAME STATUS DETAIL [JSON]
  python3 -c 'import json,sys; print(json.dumps({"case": sys.argv[1], "status": sys.argv[2], "detail": sys.argv[3], "data": json.loads(sys.argv[4] or "{}")}))' \
    "$1" "$2" "$3" "${4:-}" | tee -a "$RESULTS"
}

# one case in its own run directory and output directory: QRUN=BASE-NAME
case_begin() {
  QRUN=$BASE-$1; QCONF=${QCONF:-$QCONF_BASE}
  # shellcheck source=qual_lib.sh
  source "$QDIR/qual_lib.sh"
  ql_init
  trap 'set +e; ql_stop; ql_kill_left; ql_sampler_stop; ql_copy_logs; ql_unlock' EXIT
  ql_lock || { record "$1" unrun "the GPU locks were not free"; trap - EXIT; return 1; }
}

case_end() {
  set +e
  ql_stop; ql_kill_left; ql_sampler_stop; ql_copy_logs; ql_unlock
  trap - EXIT
  unset QCONF QCONF_RANK0 QCONF_RANK1
}

# anything of this run still alive after the stop: SIGKILL by saved pid (never by pattern)
ql_kill_left() {
  local p
  p=$(ql_pid1); [ -n "$p" ] && kill -9 "$p" 2>/dev/null
  p=$(ql_pid0); [ -n "$p" ] && ssh -n node0 "kill -9 $p" 2>/dev/null
  return 0
}

# wait_gone RANK SECONDS: until that rank's process has exited; prints the seconds it took or "alive"
wait_gone() {
  local i
  for i in $(seq 0 "$2"); do
    if [ "$1" = 0 ]; then [ -z "$(ql_pid0)" ] && { echo "$i"; return 0; }; else [ -z "$(ql_pid1)" ] && { echo "$i"; return 0; }; fi
    sleep 1
  done
  echo alive; return 1
}

client_start() {  # client_start [extra fault_client args]: the streamed request in the background on node0
  ssh -n node0 "rm -f $QRUN0/client.jsonl; setsid -f nohup python3 ~/tfp/qual/bin/fault_client.py http://127.0.0.1:$QPORT $QRUN0/client.jsonl --model $QNAME $* </dev/null >/dev/null 2>&1"
}
client_events() { ssh -n node0 "cat $QRUN0/client.jsonl 2>/dev/null | grep -c '\"n\"'"; }
client_http() { ssh -n node0 "cat $QRUN0/client.jsonl 2>/dev/null | grep -c '\"http\"'"; }
client_done() { ssh -n node0 "grep -q '\"end\"' $QRUN0/client.jsonl 2>/dev/null"; }

signal_rank() {  # signal_rank RANK SIGNAL: prints the wall-clock time of delivery, taken on the rank's own box (an ssh
  # round trip to node0 takes 0.1-0.3 s, so a time taken on node1 before it would place the fault too early)
  if [ "$1" = 0 ]; then local p0; p0=$(ql_pid0); ssh -n node0 "date +%s.%N; kill -$2 $p0"; else date +%s.%N; kill "-$2" "$(ql_pid1)"; fi
}

# a fault while a request runs. phase decode: after 20 streamed tokens; prefill: 3 s into a long prompt's prefill
run_fault() {  # run_fault NAME RANK SIGNAL PHASE
  local name=$1 rank=$2 sig=$3 phase=$4
  if [ "$phase" = prefill ] && [ "$LONG" != 1 ]; then
    record "$name" skipped "needs prompts over 128 rows (--long 1); blocked until the kit has the fp32-out _gemm variant"; return 0
  fi
  case_begin "$name" || return 0
  ql_sampler_start
  if ! ql_start; then
    if ql_env_refusal; then record "$name" unrun "node0 not idle: rank 0 refused to load (memory budget, U25)"
    else record "$name" fail "the server did not come up before the fault"; fi
    ql_sampler_stop; case_end; return 0
  fi
  ql_phase "$phase"
  local ok_wait=0 i
  if [ "$phase" = decode ]; then
    client_start --max-tokens 2000
    for i in $(seq 1 240); do [ "$(client_events)" -ge 20 ] && { ok_wait=1; break; }; sleep 0.5; done
  else
    client_start --max-tokens 64 --prompt-words 6000
    for i in $(seq 1 60); do [ "$(client_http)" -ge 1 ] && { ok_wait=1; break; }; sleep 0.5; done
    sleep 3
  fi
  if [ $ok_wait != 1 ]; then record "$name" fail "the request never reached the $phase phase"; ql_sampler_stop; case_end; return 0; fi
  local before t_fault t_ref at_fault
  before=$(client_events)
  t_fault=$(signal_rank "$rank" "$sig")
  at_fault=$(client_events)   # read right after the signal (an upper bound on the events before it)
  ql_log "$name: SIG$sig to rank $rank after $before events ($at_fault when read back)"
  t_ref=$t_fault
  if [ "$sig" = STOP ]; then
    sleep "$STOP_S"
    t_ref=$(signal_rank "$rank" CONT)
    ql_log "$name: SIGCONT to rank $rank"
  fi
  local gone1 gone0
  gone1=$(wait_gone 1 "$DEADLINE_S")
  for i in $(seq 1 "$DEADLINE_S"); do client_done && break; sleep 1; done
  gone0=$(wait_gone 0 5)
  # rank 0 still alive: a new request must get an error, not a reply and not a hang
  local after_probe=none health=none
  if [ "$gone0" = alive ]; then
    health=$(ssh -n node0 "curl -s -m 10 -w ' %{http_code}' http://127.0.0.1:$QPORT/health; echo \" \$?\"" | tr -d '\n' | cut -c1-300)
    # "<http code> <curl exit>": 7 = refused (nothing listens), 28 = timed out (a hang)
    after_probe=$(ssh -n node0 "curl -s -m 20 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d '{\"model\":\"$QNAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Answer in one word: what colour is the sky on a clear day?\"}],\"max_tokens\":4}' http://127.0.0.1:$QPORT/v1/chat/completions; echo \" \$?\"")
  fi
  ql_sampler_stop
  ssh -n node0 "cat $QRUN0/client.jsonl" > "$QOUT/client.jsonl"
  python3 - "$name" "$t_fault" "$t_ref" "$sig" "$rank" "$gone1" "$gone0" "$after_probe" "$DEADLINE_S" "$QOUT/client.jsonl" "$at_fault" "$health" <<'PY' | tee -a "$RESULTS"
import json, sys
name, t_fault, t_ref, sig, rank, gone1, gone0, probe, deadline, path, at_fault, health = sys.argv[1:13]
t_fault, t_ref, deadline = float(t_fault), float(t_ref), float(deadline)
lines = [json.loads(l) for l in open(path) if l.strip()]
ev = [l for l in lines if "n" in l]
end = next((l for l in lines if "end" in l), None)
# events after the fault: by the clocks (the client's arrival times vs the delivery time taken on the signalled rank's
# box, NTP-synchronised; 0.25 s grace) and by the count read back right after the signal; the larger is used
after_clock = [l for l in ev if l["w"] > t_fault + 0.25]
after = after_clock if len(after_clock) >= len(ev) - int(at_fault) else ev[int(at_fault):]
end_lag = (end["w_end"] - t_ref) if end else None
checks = {
    # the reply in flight must not finish as if nothing happened, and must end (error event or closed) in time
    "client_not_completed": bool(end) and end["end"] != "done",
    "client_ended_within_deadline": end_lag is not None and end_lag <= deadline,
    # no tokens committed after the failure boundary (one already on the wire is tolerated)
    "at_most_one_token_after_fault": len(after) <= 1,
    # rank 1 (the partner, or the faulty rank once resumed) must not linger
    "rank1_left_within_deadline": gone1 != "alive",
    # rank 0 serves the API: gone, or alive and refusing new work with an error
    "rank0_left_or_refuses": gone0 != "alive" or probe.split()[-1] == "7" or (probe.split()[0].isdigit() and int(probe.split()[0]) >= 400),
    # coordinated failure: a rank 0 that outlives its partner must not report itself healthy
    "rank0_left_or_unhealthy": gone0 != "alive" or not (health.split() and health.split()[-2] == "200" and '"ok"' in health),
}
status = "pass" if all(checks.values()) else "fail"
data = {"signal": sig, "rank": int(rank), "events_before": len(ev) - len(after), "events_after_fault": len(after),
        "client_end": end, "client_end_after_fault_s": round(end_lag, 3) if end_lag is not None else None,
        "rank1_gone_after_s": gone1, "rank0_gone_after_s": gone0, "probe_after_http": probe, "health_after": health, "checks": checks}
print(json.dumps({"case": name, "status": status, "detail": ", ".join(k for k, v in checks.items() if not v) or "all checks", "data": data}))
PY
  case_end
}

# a mismatch: rank 0 from the base configuration, rank 1 from VARIANT; both must stop before serving
run_mismatch() {  # run_mismatch NAME EXPECT_REGEX KEY=VALUE...
  local name=$1 expect=$2; shift 2
  local v; v=$(variant "q-$name" "$@") || { record "$name" fail "could not write the variant configuration"; return 0; }
  QCONF_RANK1=$v
  case_begin "$name" || return 0
  ql_phase load
  r1 start --locked >> "$QOUT/driver.log" 2>&1
  local rc1=$?
  r0 start --locked >> "$QOUT/driver.log" 2>&1
  local g0 g1 served
  g0=$(wait_gone 0 240); g1=$(wait_gone 1 30)
  ql_copy_logs
  served=$(grep -c "serving dsv41" "$QOUT/rank0.log" 2>/dev/null || true)
  python3 - "$name" "$expect" "$rc1" "$g0" "$g1" "${served:-0}" "$QOUT/rank0.log" "$QOUT/rank1.log" "$*" <<'PY' | tee -a "$RESULTS"
import json, re, sys
name, expect, rc1, g0, g1, served, l0, l1, change = sys.argv[1:10]
text = lambda p: open(p, errors="replace").read() if __import__("os").path.exists(p) else ""
t0, t1 = text(l0), text(l1)
refused = [l for l in (t0 + "\n" + t1).splitlines() if re.search(r"refus|mismatch|Mismatch|handshake", l)]
named = any(re.search(expect, l) for l in refused)
checks = {"both_ranks_stopped": g0 != "alive" and g1 != "alive", "never_served": served == "0",
          "refusal_names_the_setting": named}
print(json.dumps({"case": name, "status": "pass" if all(checks.values()) else "fail",
                  "detail": ", ".join(k for k, v in checks.items() if not v) or "refused: " + (refused[0][:200] if refused else ""),
                  "data": {"rank1_variant": change, "rank0_gone_after_s": g0, "rank1_gone_after_s": g1, "refusal_lines": refused[:6],
                           "checks": checks}}))
PY
  case_end
}

run_kill_load() {
  case_begin kill_r1_load || return 0
  ql_phase load
  r1 start --locked >> "$QOUT/driver.log" 2>&1
  r0 start --locked >> "$QOUT/driver.log" 2>&1
  sleep 8
  local p1; p1=$(ql_pid1)
  [ -n "$p1" ] && kill -9 "$p1"
  local t0; t0=$(date +%s)
  # rank 0 waits at most LINK_WAIT_S for its peer to link (fan.link_wait_s, 300 s since abfb508), then must leave
  local g0; g0=$(wait_gone 0 "${LINK_WAIT_S:-330}")
  ql_copy_logs
  local served; served=$(grep -c "serving dsv41" "$QOUT/rank0.log" 2>/dev/null || true)
  if [ "$g0" != alive ] && [ "${served:-0}" = 0 ]; then
    record kill_r1_load pass "rank 0 left ${g0}s after rank 1 was killed during load, never serving" "{\"rank0_gone_after_s\": $g0}"
  else
    record kill_r1_load fail "rank 0 ${g0} ${LINK_WAIT_S:-330} s after rank 1 was killed during load (served lines: ${served:-0})" "{\"rank0_gone_after_s\": \"$g0\"}"
  fi
  case_end
}

run_stale_bundle() {
  local v; v=$(variant q-stale-bundle BUNDLE_SHA256=0000000000000000000000000000000000000000000000000000000000000000)
  QCONF_RANK1=$v
  case_begin stale_bundle || return 0
  local out rc
  out=$(r1 start --locked 2>&1); rc=$?
  if [ $rc != 0 ] && echo "$out" | grep -q "inputs_manifest.json"; then
    record stale_bundle pass "the launcher refused: $(echo "$out" | tail -1)"
  else
    record stale_bundle fail "the launcher did not refuse (rc $rc): $(echo "$out" | tail -1)"
  fi
  case_end
}

run_stale_cache() {
  # rank 1's real cache file under another content key's name, rank1of2-<20 hex>.bin, as a REGULAR file (the engine
  # skips symbolic links): a hard link (same btrfs volume; no data copied), else a reflink copy
  local real dir stale
  real=$(ls "$HOME"/models/tf-port/inputs-tp2-c178a012/rank1/rank-cache/rank1of2-*.bin | head -1)
  dir=$HOME/tfp/qual/stale-cache/rank1/rank-cache
  stale=$dir/rank1of2-00000000000000000000.bin
  mkdir -p "$dir"; rm -f "$dir"/*
  ln "$real" "$stale" 2>/dev/null || cp --reflink=always "$real" "$stale" || { record stale_cache fail "could not make the stale-named file"; return 0; }
  [ -f "$stale" ] && [ ! -L "$stale" ] || { record stale_cache fail "the stale-named file is not a regular file"; return 0; }
  local v; v=$(variant q-stale-cache "RANK_CACHE='$dir'")
  QCONF_RANK1=$v
  case_begin stale_cache || return 0
  ql_phase load
  local out rc
  out=$(r1 start --locked 2>&1); rc=$?
  if [ $rc != 0 ]; then
    record stale_cache pass "the launcher refused: $(echo "$out" | tail -1)"; case_end; return 0
  fi
  r0 start --locked >> "$QOUT/driver.log" 2>&1
  local linked=0 i
  for i in $(seq 1 180); do
    grep -q '"loaded_s"' "$QRUN1/rank1.log" 2>/dev/null && { linked=1; break; }
    [ -z "$(ql_pid1)" ] && break
    sleep 1
  done
  ql_copy_logs
  if [ $linked = 1 ]; then
    record stale_cache fail "rank 1 loaded a cache file whose name (content key) is not the bundle manifest's and linked (U16 open)"
  else
    record stale_cache pass "rank 1 refused the cache: $(tail -2 "$QOUT/rank1.log" | tr '\n' ' ')"
  fi
  case_end
  rm -f "$stale"   # the extra name only (a hard link: the real file is untouched)
}

IFS=, read -ra list <<< "$CASES"
for c in "${list[@]}"; do
  case $c in
    kill_r1_decode) run_fault "$c" 1 KILL decode;;
    kill_r0_decode) run_fault "$c" 0 KILL decode;;
    stop_r1_decode) run_fault "$c" 1 STOP decode;;
    stop_r0_decode) run_fault "$c" 0 STOP decode;;
    kill_r1_prefill) run_fault "$c" 1 KILL prefill;;
    stop_r1_prefill) run_fault "$c" 1 STOP prefill;;
    kill_r1_load) run_kill_load;;
    mm_round_rows) run_mismatch "$c" "round_rows" ROUND_ROWS=8;;
    mm_aux) run_mismatch "$c" "aux" AUX=triton;;
    mm_streams) run_mismatch "$c" "streams" STREAMS=2;;
    mm_context) run_mismatch "$c" "pool|context" CONTEXT=4096;;
    mm_pool) run_mismatch "$c" "pool" POOL=$(( $(source "${QCONF_BASE//\$HOME/$HOME}"; echo "${POOL:-$CONTEXT}") + 2048 ));;
    mm_kit) run_mismatch "$c" "kit" "DEPLOY_DIR='\$HOME/tfp/deploy/009fe20cfddb'";;
    mm_strict) run_mismatch "$c" "strict" GPU_STRICT=$([ "$(source "${QCONF_BASE//\$HOME/$HOME}"; echo "$GPU_STRICT")" = 1 ] && echo 0 || echo 1);;
    mm_conf) run_mismatch "$c" "conf" MAX_TOKENS=511;;
    stale_bundle) run_stale_bundle;;
    stale_cache) run_stale_cache;;
    *) record "$c" fail "unknown case";;
  esac
done

BASE1=${QCONF_BASE//\$HOME/$HOME}
DEP1=$(source "$BASE1"; echo "${DEPLOY_DIR//\$HOME/$HOME}")
python3 - "$RESULTS" "$QOUT_ROOT/faults.json" "$BASE1" "$DEP1" "$CASES" <<'PY'
import hashlib, json, os, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
summary = {s: sum(1 for r in rows if r["status"] == s) for s in ("pass", "fail", "skipped", "unrun")}
dep = json.load(open(os.path.join(sys.argv[4], "DEPLOY.json")))
json.dump({"schema": "tf-port-served-faults/1", "deploy_commit": dep.get("commit"), "config": sys.argv[3],
           "config_sha256": hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest(), "cases_requested": sys.argv[5].split(","),
           "summary": summary, "cases": rows}, open(sys.argv[2], "w"), indent=1)
print(json.dumps(summary))
sys.exit(1 if summary["fail"] or summary["unrun"] else 0)
PY

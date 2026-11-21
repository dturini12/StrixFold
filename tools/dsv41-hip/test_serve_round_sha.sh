#!/bin/bash
# test_serve_round_sha.sh: serve_lib.sh's ROUND_LOGITS_SHA key (TF_DS_ROUND_LOGITS_SHA to rank 0 only; nothing when unset
# or empty; never on rank 1's argv). Host only: serve.conf with its paths pointed at a scratch deploy, resolved by
# serve_load + serve_env (+ serve_lanes_cmd on rank 1). Exit 0 when every case passes.
set -u
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/deploy/ds-kit"
echo '{"commit": "70cfb8c"}' >"$tmp/deploy/DEPLOY.json"
for f in rope-plain-cos rope-plain-sin rope-compressed-cos rope-compressed-sin; do
  truncate -s $((16384 * 128)) "$tmp/deploy/ds-kit/$f.f32"
done
pass=0 fail=0

# resolve KEY=VALUE... (appended to serve.conf) for RANK; prints SERVE_ENV and SERVE_CMD one word a line, or the die
resolve() {
  local rank=$1; shift
  { cat "$here/serve.conf"; printf "DEPLOY_DIR='%s'\n" "$tmp/deploy"; printf '%s\n' "$@"; } >"$tmp/serve.conf"
  (
    export TF_SERVE_CONF=$tmp/serve.conf
    # shellcheck source=serve_lib.sh
    source "$here/serve_lib.sh"
    serve_load "$rank"
    serve_env
    if [ "$rank" = 1 ]; then serve_lanes_cmd; fi
    printf 'ENV %s\n' "${SERVE_ENV[@]}"
    [ "$rank" = 1 ] && printf 'CMD %s\n' "${SERVE_CMD[@]}"
    exit 0
  ) 2>&1
}

# check NAME WANT_STATUS PATTERN_PRESENT PATTERN_ABSENT RANK KEY=VALUE...: the resolution's exit status, a line it must
# print (grep -x; empty: none) and a line it must not print (grep -x; empty: none)
check() {
  local name=$1 want=$2 has=$3 hasnt=$4 rank=$5; shift 5
  local out st ok=1
  out=$(resolve "$rank" "$@"); st=$?
  if [ "$want" = 0 ] && [ "$st" != 0 ]; then ok=0; fi
  if [ "$want" != 0 ] && [ "$st" = 0 ]; then ok=0; fi
  if [ -n "$has" ] && ! grep -qxE -- "$has" <<<"$out"; then ok=0; fi
  if [ -n "$hasnt" ] && grep -qxE -- "$hasnt" <<<"$out"; then ok=0; fi
  if [ "$ok" = 1 ]; then
    pass=$((pass + 1)); echo "PASS $name"
  else
    fail=$((fail + 1)); echo "FAIL $name (status $st)"; sed 's/^/    /' <<<"$out" | tail -n 20
  fi
}

check "off by default (rank 0)" 0 "ENV TF_DS_AUX=hip" "ENV TF_DS_ROUND_LOGITS_SHA.*" 0
check "empty exports nothing" 0 "ENV TF_DS_AUX=hip" "ENV TF_DS_ROUND_LOGITS_SHA.*" 0 ROUND_LOGITS_SHA=
check "ROUND_LOGITS_SHA=1 reaches rank 0" 0 "ENV TF_DS_ROUND_LOGITS_SHA=1" "" 0 ROUND_LOGITS_SHA=1
check "ROUND_LOGITS_SHA=1 beside TRACE_ROUNDS=1 (rank 0)" 0 "ENV TF_DS_TRACE_ROUNDS=1" "" 0 ROUND_LOGITS_SHA=1 TRACE_ROUNDS=1
check "ROUND_LOGITS_SHA=1 not in rank 1 environment" 0 "ENV TF_DS_AUX=hip" "ENV TF_DS_ROUND_LOGITS_SHA.*" 1 ROUND_LOGITS_SHA=1
check "ROUND_LOGITS_SHA=1 not on rank 1 argv" 0 "CMD --side" "CMD .*(round-logits|round_logits|ROUND_LOGITS).*" 1 ROUND_LOGITS_SHA=1
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]

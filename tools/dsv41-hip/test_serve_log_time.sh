#!/bin/bash
# test_serve_log_time.sh: serve_lib.sh's LOG_TIME key (0 or 1; TF_LOG_TIME=1 to both ranks when on, nothing when off).
# Host only: serve.conf with its paths pointed at a scratch deploy (DEPLOY.json and sparse RoPE tables), resolved by
# serve_load + serve_env (+ serve_cmd's rank 1 argv) in a subshell per case.
# Run: tools/dsv41-hip/test_serve_log_time.sh (exit 0 when every case passes).
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

check "off by default: no TF_LOG_TIME (rank 0)" 0 'ENV TF_DS_AUX=hip' 'ENV TF_LOG_TIME.*' 0
check "off by default: no TF_LOG_TIME (rank 1)" 0 'ENV TF_DS_AUX=hip' 'ENV TF_LOG_TIME.*' 1
check "a conf written before the key: off" 0 'ENV TF_DS_AUX=hip' 'ENV TF_LOG_TIME.*' 0 'unset LOG_TIME'
check "LOG_TIME= (empty) exports nothing" 0 'ENV TF_DS_AUX=hip' 'ENV TF_LOG_TIME.*' 1 LOG_TIME=
check "LOG_TIME=0 exports nothing" 0 'ENV TF_DS_AUX=hip' 'ENV TF_LOG_TIME.*' 0 LOG_TIME=0
check "LOG_TIME=1 reaches rank 0" 0 'ENV TF_LOG_TIME=1' '' 0 LOG_TIME=1
check "LOG_TIME=1 reaches rank 1 (environment)" 0 'ENV TF_LOG_TIME=1' '' 1 LOG_TIME=1
check "LOG_TIME=1 adds no rank 1 argument" 0 'CMD --side' 'CMD --(log-time|time).*' 1 LOG_TIME=1
check "LOG_TIME=2 refused" 1 '.*LOG_TIME must be 0 or 1, not 2' 'ENV .*' 0 LOG_TIME=2
check "LOG_TIME=yes refused" 1 '.*LOG_TIME must be 0 or 1, not yes' 'ENV .*' 1 LOG_TIME=yes

echo "$pass passed, $fail failed"
[ "$fail" = 0 ]

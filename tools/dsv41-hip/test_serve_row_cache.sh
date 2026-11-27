#!/bin/bash
# test_serve_row_cache.sh: serve_lib.sh's ENGRAM_ROW_CACHE_MIB key (a whole number of MiB up to 65536; to both ranks as
# TF_DS_ENGRAM_ROW_CACHE_MIB when above 0, nothing when empty or 0). Host only: serve.conf with its paths pointed at a
# scratch deploy (DEPLOY.json and sparse RoPE tables), resolved by serve_load + serve_env in a subshell per case.
# Run: tools/dsv41-hip/test_serve_row_cache.sh (exit 0 when every case passes).
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

check "off by default: no TF_DS_ENGRAM_ROW_CACHE_MIB (rank 0)" 0 'ENV TF_DS_AUX=hip' 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB.*' 0
check "off by default: no TF_DS_ENGRAM_ROW_CACHE_MIB (rank 1)" 0 'ENV TF_DS_AUX=hip' 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB.*' 1
check "a conf written before the key: off" 0 'ENV TF_DS_AUX=hip' 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB.*' 0 'unset ENGRAM_ROW_CACHE_MIB'
check "ENGRAM_ROW_CACHE_MIB=0 exports nothing" 0 'ENV TF_DS_AUX=hip' 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB.*' 0 ENGRAM_ROW_CACHE_MIB=0
check "ENGRAM_ROW_CACHE_MIB=512 reaches rank 0" 0 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB=512' '' 0 ENGRAM_ROW_CACHE_MIB=512
check "ENGRAM_ROW_CACHE_MIB=512 reaches rank 1" 0 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB=512' '' 1 ENGRAM_ROW_CACHE_MIB=512
check "65536 allowed" 0 'ENV TF_DS_ENGRAM_ROW_CACHE_MIB=65536' '' 0 ENGRAM_ROW_CACHE_MIB=65536
check "65537 refused" 1 '.*ENGRAM_ROW_CACHE_MIB must be a whole number of MiB up to 65536.*' 'ENV .*' 0 ENGRAM_ROW_CACHE_MIB=65537
check "a unit refused" 1 '.*ENGRAM_ROW_CACHE_MIB must be a whole number of MiB up to 65536.*not 512M' 'ENV .*' 0 ENGRAM_ROW_CACHE_MIB=512M
check "a negative refused" 1 '.*ENGRAM_ROW_CACHE_MIB must be a whole number.*' 'ENV .*' 1 ENGRAM_ROW_CACHE_MIB=-1

echo "$pass passed, $fail failed"
[ "$fail" = 0 ]

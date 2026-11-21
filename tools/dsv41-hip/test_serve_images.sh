#!/bin/bash
# test_serve_images.sh: serve_lib.sh's image keys (IMAGE_ROWS_CACHE_MIB, IMAGE_ROWS_BUDGET_MIB, IMAGE_MEMO, FILL_IMAGES,
# TOWER_SLICE_MS): validated on both ranks, exported to rank 0 only and only when on (TF_DS_* in its environment,
# --vision-image-memo on its argv), nothing when unset, empty or 0; FILL_IMAGES=1 refused without FILL_SLICE_MS > 0.
# Host only, as test_serve_round_sha.sh: serve.conf with its paths pointed at a scratch deploy. The last case writes
# rank 0's resolved.json (serve_resolved, a dry run with a stand-in binary) and checks every key in it.
# Run: tools/dsv41-hip/test_serve_images.sh (exit 0 when every case passes).
set -u
here=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp/deploy/ds-kit" "$tmp/deploy/native-hip/bin" "$tmp/kit" "$tmp/cache"
echo '{"commit": "0ac8c68c5afc"}' >"$tmp/deploy/DEPLOY.json"
for f in rope-plain-cos rope-plain-sin rope-compressed-cos rope-compressed-sin; do
  truncate -s $((16384 * 128)) "$tmp/deploy/ds-kit/$f.f32"
done
echo stand-in >"$tmp/deploy/native-hip/bin/tensorfold-native"
echo '{}' >"$tmp/kit/manifest.json"
touch "$tmp/cache/rank0of2.bin"
pass=0 fail=0
all=(IMAGE_ROWS_CACHE_MIB=64 IMAGE_ROWS_BUDGET_MIB=1280 IMAGE_MEMO=65536 FILL_IMAGES=1 TOWER_SLICE_MS=200)

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
    if [ "$rank" = 1 ]; then serve_lanes_cmd; else serve_cmd; fi
    printf 'ENV %s\n' "${SERVE_ENV[@]}"
    printf 'CMD %s\n' "${SERVE_CMD[@]}"
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

none="(ENV TF_DS_(FILL_IMAGES|IMAGE_ROWS_CACHE_MIB|IMAGE_ROWS_BUDGET_MIB|TOWER_SLICE_MS)=.*|CMD --vision-image-memo)"
check "unset: nothing (rank 0)" 0 "ENV TF_FILL_SLICE_MS=1000" "$none" 0
check "empty: nothing (rank 0)" 0 "ENV TF_FILL_SLICE_MS=1000" "$none" 0 IMAGE_ROWS_CACHE_MIB= IMAGE_ROWS_BUDGET_MIB= IMAGE_MEMO= FILL_IMAGES= TOWER_SLICE_MS=
check "0: nothing (rank 0)" 0 "ENV TF_FILL_SLICE_MS=1000" "$none" 0 IMAGE_ROWS_CACHE_MIB=0 IMAGE_ROWS_BUDGET_MIB=00 IMAGE_MEMO=0 FILL_IMAGES=0 TOWER_SLICE_MS=0
for kv in "TF_DS_IMAGE_ROWS_CACHE_MIB=64" "TF_DS_IMAGE_ROWS_BUDGET_MIB=1280" "TF_DS_FILL_IMAGES=1" "TF_DS_TOWER_SLICE_MS=200"; do
  check "on: $kv (rank 0)" 0 "ENV $kv" "" 0 "${all[@]}"
done
check "on: --vision-image-memo on rank 0's argv" 0 "CMD 65536" "" 0 "${all[@]}"
check "on: the flag before its value" 0 "CMD --vision-image-memo" "" 0 "${all[@]}"
check "leading zeros dropped" 0 "ENV TF_DS_TOWER_SLICE_MS=50" "" 0 TOWER_SLICE_MS=050
check "on: nothing in rank 1's environment or argv" 0 "CMD --side" "(ENV TF_DS_(FILL_IMAGES|IMAGE_ROWS_CACHE_MIB|IMAGE_ROWS_BUDGET_MIB|TOWER_SLICE_MS)=.*|CMD .*image-memo.*|CMD .*(rows-cache|rows-budget|tower|fill-images).*)" 1 "${all[@]}"
check "FILL_IMAGES=1 needs FILL_SLICE_MS > 0" 1 ".*FILL_IMAGES=1 needs FILL_SLICE_MS above 0.*" "" 0 FILL_IMAGES=1 FILL_SLICE_MS=0
check "FILL_IMAGES=1 refused on rank 1 too (same conf)" 1 ".*FILL_IMAGES=1 needs FILL_SLICE_MS above 0.*" "" 1 FILL_IMAGES=1 FILL_SLICE_MS=0
check "FILL_IMAGES=2 refused" 1 ".*FILL_IMAGES must be 0 or 1.*" "" 0 FILL_IMAGES=2
for k in IMAGE_ROWS_CACHE_MIB IMAGE_ROWS_BUDGET_MIB TOWER_SLICE_MS; do
  check "$k=-1 refused" 1 ".*$k must be a whole number.*" "" 0 "$k=-1"
  check "$k=1234567 refused" 1 ".*$k must be a whole number.*" "" 0 "$k=1234567"
done
check "IMAGE_MEMO=x refused" 1 ".*IMAGE_MEMO must be a whole number.*" "" 0 IMAGE_MEMO=x
check "IMAGE_MEMO=12345678 refused" 1 ".*IMAGE_MEMO must be a whole number.*" "" 0 IMAGE_MEMO=12345678

# the dry run: rank 0's resolved.json names every key (env and argv); one without them names none
resolved() {
  local out=$1; shift
  { cat "$here/serve.conf"; printf "DEPLOY_DIR='%s'\nKIT_SUBDIR=../kit\nRANK_CACHE='%s'\n" "$tmp/deploy" "$tmp/cache"; printf '%s\n' "$@"; } >"$tmp/serve.conf"
  (
    export TF_SERVE_CONF=$tmp/serve.conf
    # shellcheck source=serve_lib.sh
    source "$here/serve_lib.sh"
    serve_load 0
    serve_env
    serve_cmd
    serve_resolved "$out"
  ) 2>&1
}
if resolved "$tmp/on.json" "${all[@]}" && resolved "$tmp/off.json" &&
  python3 - "$tmp/on.json" "$tmp/off.json" <<'PY'
import json, sys
on, off = (json.load(open(p)) for p in sys.argv[1:3])
want = {"TF_DS_IMAGE_ROWS_CACHE_MIB": "64", "TF_DS_IMAGE_ROWS_BUDGET_MIB": "1280", "TF_DS_FILL_IMAGES": "1", "TF_DS_TOWER_SLICE_MS": "200"}
for k, v in want.items():
    assert on["env"].get(k) == v, (k, on["env"].get(k))
    assert k not in off["env"], k
i = on["argv"].index("--vision-image-memo")
assert on["argv"][i + 1] == "65536", on["argv"]
assert "--vision-image-memo" not in off["argv"], off["argv"]
for k, v in want.items():
    print(f"  resolved-rank0.json env {k}={on['env'][k]}")
print(f"  resolved-rank0.json argv --vision-image-memo {on['argv'][i + 1]}")
PY
then
  pass=$((pass + 1)); echo "PASS resolved-rank0.json shows each key (and none when unset)"
else
  fail=$((fail + 1)); echo "FAIL resolved-rank0.json"
fi
echo "$pass passed, $fail failed"
[ "$fail" = 0 ]

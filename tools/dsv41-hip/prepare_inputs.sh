#!/bin/bash
# prepare_inputs.sh MODEL_DIR ENGRAM_DIR OUT_DIR [ROWS]: the CPU-only TP2 inputs bundle (BUILD_AND_RUN.md section 3), no CUDA, no
# NVIDIA container, no GPU:
#   model/{config.json,tokenizer.json,chat_template.jinja (865cb39b…)}, dsv41_token_map.json, kit/engram.json,
#   kit/rope-*.f32 + rope.json (ROWS rows, default 65536; the engine reads pool + 49 rows, so ROWS covers any
#   --context up to ROWS - 49), rank{0,1}/rank-cache/rank{R}of2-<index hash>.bin, inputs_manifest.json.
# Every checkpoint and Engram file is authenticated against the pinned HF metadata first; each rank cache is written
# by tf-dsv41-rank-cache and byte-checked in full (tf-dsv41-check --bytes 1) against the Python rank plan's exact
# entry count and data bytes; the RoPE tables must hash to a reference derived from the creator kit's served tables
# (rope_reference.py); every output file is hashed into the manifest.
#
# Environment:
#   TFP_INPUT_TOOLS   the input-tool Python modules (prepare_inputs.py, inventory_checkpoint.py, ...): default this
#                     directory if it holds prepare_inputs.py, else ~/tfp/kit/tensorfold-strix/tools/dsv41-hip; a kit
#                     directory must be a clean checkout at kit_pin.sh's commit (def87879...) or the script stops
#   TFP_INPUT_PYTHON  a Python with numpy 2.3.5, tokenizers 0.22.1 and torch 2.9.0+cpu (default ~/venvs/tfp-inputs)
#   TFP_BIN           tf-dsv41-rank-cache and tf-dsv41-check (default this checkout's zig-out/bin: `zig build install`
#                     with no -Dnvcc builds them host-only)
# It runs itself under a confinement (CPUQuota 800%, cores 8-15/24-31, MemoryHigh 4G / MemoryMax 8G,
# nice 19, idle I/O) so its page cache cannot push other work out of RAM; as a capped streaming CPU job it does not take
# /var/tmp/halo-gpu.lock.
# About 1 TB of reads and 212 GB of writes; OUT_DIR must be new or empty.
set -euo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
[ $# -ge 3 ] && [ $# -le 4 ] || usage
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
if [ -z "${TFP_SCOPED:-}" ]; then
  export TFP_SCOPED=1
  exec systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryHigh=4G -p MemoryMax=8G -- \
    taskset -c 8-15,24-31 /usr/bin/nice -n 19 /usr/bin/ionice -c3 "$0" "$@"
fi
MODEL=$1 ENGRAM=$2 OUT=$3 ROWS=${4:-65536}
. "$HERE/kit_pin.sh"
TOOLS=$(input_tools "$HERE")
PY=${TFP_INPUT_PYTHON:-$HOME/venvs/tfp-inputs/bin/python}
BIN=${TFP_BIN:-$ROOT/zig-out/bin}
for x in "$TOOLS/prepare_inputs.py" "$TOOLS/source_lock.json" "$BIN/tf-dsv41-rank-cache" "$BIN/tf-dsv41-check" "$PY"; do
  [ -e "$x" ] || { echo "missing: $x" >&2; exit 1; }
done
"$PY" - <<'PY'
from importlib.metadata import version
want = {"numpy": "2.3.5", "tokenizers": "0.22.1", "torch": "2.9.0+cpu"}
got = {k: version(k) for k in want}
assert got == want, f"pinned CPU dependencies differ: {got} != {want}"
PY
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=
mkdir -p "$(dirname "$OUT")"
REF=$OUT.rope-reference.json  # kept beside the bundle: the bundle itself is exactly what the manifest lists
"$PY" "$HERE/rope_reference.py" "$TOOLS" "$MODEL/config.json" "$ROWS" "$REF"
echo "$(date '+%F %T') prepare_inputs.py full mode: $MODEL -> $OUT ($ROWS RoPE rows)"
"$PY" "$TOOLS/prepare_inputs.py" "$MODEL" "$OUT" --rows "$ROWS" --engram-dir "$ENGRAM" \
  --rank-cache-bin "$BIN/tf-dsv41-rank-cache" --check-bin "$BIN/tf-dsv41-check" --rope-reference "$REF"
echo "$(date '+%F %T') done: $OUT/inputs_manifest.json"

#!/bin/bash
# deploy_rank0.sh [--host HOST] [--dest DIR] [--bin DIR --bin-commit SHA] [--native DIR] [--ds-kit DIR] [--integ BRANCH]: put this
# commit's tools (and optionally its CPU/HIP binaries) on the rank-0 box, which has no checkout (node0 = rank 0;
# node1 builds and is rank 1).
#
# The deployment rule: build on node1 from a clean, committed tree (tools/dsv41-hip/zig_build.sh), then copy with this
# script. Nothing on node0 is built or edited by hand. The copy lands in HOST:DIR/<commit12>/ (default
# node0:~/tfp/deploy) as
#   tools/dsv41-hip/...   the tracked files of tools/dsv41-hip at HEAD (git archive, not the worktree)
#   bin/...               the binaries from --bin DIR, if given (--bin-commit must equal HEAD: tools and binaries
#                         come from one commit)
#   native-hip/...        --native DIR: the HIP build's tree as zig build native-hip installs it (bin/tensorfold-native,
#                         bin/tf-dsv41-lanes, bin/tf-dsv41-layer, bin/tf-hip-kit, share/tensorfold/hip/<arch>/dsv41: the
#                         AMD kit), whole (the server finds the kit beside itself); --bin-commit as for --bin
#   ds-kit/...            --ds-kit DIR: the RoPE tables (rope-*.f32, rope.json) and engram.json, the files only
#                         (serve.conf DS_KIT; their rows set the served context limit, serve_lib.sh)
# --host local deploys on this box (no ssh): rank 1's box runs the same bytes as rank 0's (serve.conf DEPLOY_DIR).
#   DEPLOY.json           commit, tree, source host, time, and the SHA-256 of every deployed file
#   SHA256SUMS            the same hashes for sha256sum -c
# then checked on HOST with sha256sum -c and made read-only (chmod -R a-w: nothing rewrites a deployed file in place; a
# redeploy of the same commit makes it writable for the copy first), and HOST:DIR/current points at it (switched atomically; a tools-only deploy
# leaves a native-hip current alone, see switch_current; no other deploy directory is touched). provenance.py reads
# DEPLOY.json, so
# outputs made on HOST (preflight, verify_inputs) name the source commit. Refuses a tree with uncommitted changes to
# tracked files. Small (tools: a few MB): no link lock needed.
set -euo pipefail
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
HOST=node0 DEST='~/tfp/deploy' BIN= BIN_COMMIT= NATIVE= DSKIT= INTEG=main
while [ $# -gt 0 ]; do
  case $1 in
    --host) HOST=$2; shift 2 ;;
    --dest) DEST=$2; shift 2 ;;
    --bin) BIN=$2; shift 2 ;;
    --bin-commit) BIN_COMMIT=$2; shift 2 ;;
    --native) NATIVE=$2; shift 2 ;;
    --ds-kit) DSKIT=$2; shift 2 ;;
    --integ) INTEG=$2; shift 2 ;;
    *) usage ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(git -C "$HERE" rev-parse --show-toplevel)
HEAD=$(git -C "$ROOT" rev-parse HEAD)
[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || { echo "deploy_rank0: uncommitted changes to tracked files; commit first" >&2; exit 1; }
if [ -n "$BIN$NATIVE" ] && [ "$(git -C "$ROOT" rev-parse "${BIN_COMMIT:-none}" 2>/dev/null)" != "$HEAD" ]; then
  echo "deploy_rank0: --bin / --native need --bin-commit equal to HEAD ($HEAD): build the binaries from this commit first" >&2; exit 1
fi
SHORT=${HEAD:0:12}
STAGE=$(mktemp -d)
chmod 755 "$STAGE"  # mktemp makes it 0700; the deployed directory should read like a checkout
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive HEAD tools/dsv41-hip | tar -x -C "$STAGE"
if [ -n "$BIN" ]; then mkdir -p "$STAGE/bin"; cp -p "$BIN"/* "$STAGE/bin/"; fi
if [ -n "$NATIVE" ]; then mkdir -p "$STAGE/native-hip"; cp -pR "$NATIVE"/. "$STAGE/native-hip/"; fi
if [ -n "$DSKIT" ]; then mkdir -p "$STAGE/ds-kit"; cp -p "$DSKIT"/engram.json "$DSKIT"/rope.json "$DSKIT"/rope-*.f32 "$STAGE/ds-kit/"; fi
# the image routing biases both ranks load with MODALITIES=text_image (tools/dsv41-zig/kit/make_bias_vl.py), when made
if [ -n "$DSKIT" ] && [ -f "$DSKIT/vision/gate_bias_vl.safetensors" ]; then
  mkdir -p "$STAGE/ds-kit/vision"; cp -p "$DSKIT/vision/gate_bias_vl.safetensors" "$STAGE/ds-kit/vision/"
fi
(cd "$STAGE" && find . -type f ! -name SHA256SUMS ! -name DEPLOY.json -printf '%P\n' | sort | xargs -d '\n' sha256sum > SHA256SUMS)
python3 - "$STAGE" "$HEAD" "$(git -C "$ROOT" rev-parse HEAD^{tree})" "$HOST" "$DEST/$SHORT" "${BIN_COMMIT:-}" <<'PY'
import json, socket, sys, time
stage, head, tree, host, dest, bin_commit = sys.argv[1:7]
files = dict(reversed(l.split("  ", 1)) for l in open(f"{stage}/SHA256SUMS").read().splitlines())
json.dump({"schema": "tf-port-deploy/1", "commit": head, "tree": tree, "clean": True, "host": socket.gethostname(),
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": f"{host}:{dest}",
           "bin_commit": bin_commit or None, "files": files}, open(f"{stage}/DEPLOY.json", "w"), indent=1)
PY
# switch_current DEST SHORT: the shell command that points DEST/current at SHORT atomically (a new link renamed over
# the old one), unless current is a native-hip deploy and SHORT is not: serving runs from current (serve.conf), so a
# tools-only deploy never takes the binaries away from it. Only a commit on the integration branch (--integ, default
# main) ever becomes current: a worker branch's build is deployed beside it (use its own --dest) and current is
# left alone. Other deploy directories are never touched.
ON_INTEG=0
git -C "$ROOT" merge-base --is-ancestor "$HEAD" "$INTEG" 2>/dev/null && ON_INTEG=1
switch_current() {
  if [ "$ON_INTEG" != 1 ]; then echo "echo 'current unchanged: $HEAD is not on $INTEG, a worker branch build'"; return; fi
  echo "if [ -d $1/$2/native-hip ] || [ ! -d $1/current/native-hip ]; then ln -sfn $2 $1/.current.new && mv -Tf $1/.current.new $1/current && echo current: $2; else echo current stays \$(readlink $1/current): a native-hip deploy, this one is tools only; fi"
}

if [ "$HOST" = local ]; then
  DEST=${DEST/#\~/$HOME}
  mkdir -p "$DEST/$SHORT" && chmod -R u+w "$DEST/$SHORT"
  rsync -a --delete "$STAGE/" "$DEST/$SHORT/"
  (cd "$DEST/$SHORT" && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && eval "$(switch_current "$DEST" "$SHORT")" && echo "deployed $HEAD to $(hostname):$(pwd) (read-only)")
  exit 0
fi
ssh "$HOST" "mkdir -p $DEST/$SHORT && chmod -R u+w $DEST/$SHORT"
rsync -a --delete "$STAGE/" "$HOST:$DEST/$SHORT/"
ssh "$HOST" "cd $DEST/$SHORT && sha256sum -c --quiet SHA256SUMS && chmod -R a-w . && $(switch_current "$DEST" "$SHORT") && echo deployed $HEAD to \$(hostname):\$(pwd) '(read-only)'"


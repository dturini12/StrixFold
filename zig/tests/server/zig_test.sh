#!/usr/bin/env bash
# The server's unit tests (its module with the engine seam, tokenizer and template), then the engine seam's own. ZIG overrides the compiler.
set -euo pipefail
zig="${ZIG:-zig}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
# dsv41_assets as build.zig's assetsModule makes it: the deployed chat template embedded beside its module file
assets="$(mktemp -d)"
trap 'rm -rf "$assets"' EXIT
cp "$root/tools/dsv41-hip/assets/chat_template.jinja" "$assets/"
printf 'pub const chat_template = @embedFile("chat_template.jinja");\n' > "$assets/dsv41_assets.zig"
"$zig" test -lc --dep engine_api --dep tokenizer --dep template --dep dsv41_assets \
  "-Mroot=$root/zig/src/server/root.zig" \
  --dep lanes "-Mengine_api=$root/zig/src/core/engine_api.zig" \
  "-Mlanes=$root/zig/src/core/lanes/lanes.zig" \
  "-Mtokenizer=$root/zig/src/core/tokenizer/tokenizer.zig" \
  "-Mtemplate=$root/zig/src/core/template/template.zig" "-Mdsv41_assets=$assets/dsv41_assets.zig" "$@"
"$zig" test -lc --dep lanes "-Mroot=$root/zig/src/core/engine_api.zig" "-Mlanes=$root/zig/src/core/lanes/lanes.zig"

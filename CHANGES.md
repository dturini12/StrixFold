# Changes in this fork

This fork starts from Bertholomus AI's DeepSeek-V4.1 fork of TensorFold at commit 3ff3b2ac (18 Oct 2026). Every change
below is ours, made between 18 October and 13 November 2026 and released under Apache-2.0 as upstream, except one Linux
kernel module in `tools/m5-usb4/`, which is GPL-2.0. Upstream's LICENSE, LICENSES/ and THIRD_PARTY_NOTICES.md are
unchanged; NOTICE gains one section at its end.

## Added

| Path | What it is |
|---|---|
| `zig/src/gpu/hip/` | The AMD HIP backend: devices, memory, streams, events, graphs, staging and the kit loader |
| `zig/src/gpu/` (kernel, interface, manifest, staging, msgpack, kit_tool) | A backend-neutral GPU layer the CUDA and HIP paths share |
| `zig/src/native/hip.zig` | The native HIP entry point |
| `zig/kernels/hip/dsv41/` | Strix Halo (gfx1151) kernels: EXL3 weight matmul, sparse attention, top-k, routed experts, prompt reading, RoPE, norms |
| `zig/src/families/dsv41/` (16 new files) | The DeepSeek-V4.1 family's HIP paths: two-box collectives over TCP, the cable stripe, kept states for long context, drafting switch |
| `zig/src/core/lanes/` (8 new files) | Grammar lanes and lane scheduling used by the server |
| `zig/tests/dsv41_hip/` | GPU smoke, ABI, glue, code-object and collectives tests |
| `zig/src/server/` (reply_dsv41, tests) | DeepSeek-V4.1 reply handling on this backend, and tests for top-k, structure and telemetry |
| `tools/dsv41-hip/` | Build, input preparation, serving scripts, output checks (gates, numerics, qualification), the benchmark, and the design notes (ADAPTER, COLLECTIVES-DESIGN, KERNELS, BUILD_AND_RUN, RESULTS) |
| `tools/dsv41-hip/` input tools (prepare_inputs.py, inventory_checkpoint.py, input_common.py, snapshot_files.py, source_lock.json, assets/) | The CPU input preparation `prepare_inputs.sh` and `verify_inputs.sh` run, vendored unchanged from the component kit at def87879; the chat template is Tech2wild's (MIT, see NOTICE) |
| `tools/dsv41-zig/inputs/`, `tools/dsv41-zig/kit/` | Token map and per-node input preparation, and kit packing |
| `docs/RESULTS.md`, `docs/TUNING.md`, `docs/img/` | Measured results with charts, and the two-box tuning notes |
| `tools/dsv41-hip/ctx_sweep.py` | Writing speed against context length, drafts on and off (RESULTS, Speed at long context) |
| `docs/COMMUNITY-RESULTS.md`, `.github/ISSUE_TEMPLATE/` | Results from other people's hardware, and the issue form that collects them |
| `tools/m5-usb4/` | The Bosgame M5's optional USB4 host-to-host setting: a kernel module (GPL-2.0, its text in `COPYING`), a script and a service |
| `README.md`, `CHANGES.md` | This fork's README and this file. Upstream's README moved to `docs/UPSTREAM-README.md`. |

## Modified upstream files

| Path | Change |
|---|---|
| `build.zig`, `zig/build/cuda.zig` | Build steps for the HIP backend and its kernels |
| `zig/src/core/engine_api.zig`, `lane_host.zig`, `lanes/*.zig` | The lane host and engine API work across both GPU backends; requests wait for room in the pool |
| `zig/src/families/dsv41/*.zig` (42 files) | The family runs on HIP as well as CUDA: weights, prompt and decode paths, sampling, drafting, checkpoints and the two-box link |
| `zig/src/native/cuda.zig` | The CUDA entry point uses the shared GPU layer |
| `zig/src/server/*.zig` (9 files) | Reply parity with the official DeepSeek encoder and parser on this backend, and server settings for two-box serving |

The exact diff against upstream is `git diff 3ff3b2ac..main`.

## Ported from upstream TensorFold

Fixes and features taken from [TensorFold](https://github.com/ashhart/TensorFold) (ashhart) and from
[Bertholomus AI's DeepSeek fork](https://github.com/bertholomus/TensorFold) (bertholomus) after this fork started.
Each one is its own commit. A cherry-pick keeps its author and names its source commit; a port rewritten for this tree
names the source commit and its author in the message. Thanks to everyone below.

| Upstream commit | Author | What it does here |
|---|---|---|
| bertholomus 12132870b | Blackwellboy | A refused Engram row id no longer leaks the read batch's buffers |
| ashhart c611adeb2 | akol1 | TCP_NODELAY on accepted HTTP connections, so streamed tokens leave at once |
| ashhart 4284048d1 / bertholomus 0937cc13f | Jay Leaton | The JSON writer writes U+FFFD for a UTF-8 sequence cut off at a string's end instead of reading past it |
| ashhart 2193d1453 | chaog992 | A request keeps its own copy of its key label (use-after-free after a key-file reload) |
| bertholomus d5868eabe | bertholomus | A replay cut uploads its rows' positions into the chunk's aligned buffer (the same bug was fixed independently by mrsergiopascal in f7b0e41b5) |
| ashhart 2d2930d8b | plotarmordev | The tokenizer memo: long segments keep their ids between encodes |
| ashhart ff0551146, 9fa8f8773 | ashhart | Memo ids equal a fresh encode's wherever a segment sits; recently used segments are kept |
| ashhart a56c63daf, 81d2a912c, 6ed8a455a | mrv777 | Memo segments from 64 bytes; each entry's overhead counts against the cap |
| bertholomus 16c8d30e0, 837205e21 | bertholomus | The memo works between chat markers; a long encode gives its scratch memory back |
| ashhart d41127c06, 2be7146bc | Robert Wojciechowski | Tokenizer and template scratch on their own allocator, not the request's |
| bertholomus f4d53987f, 746f03b2a | bertholomus | `TENSORFOLD_SSE_KEEPALIVE`: SSE comments while a long prompt is read (off by default) |
| ashhart 2f25b0443 (part) | ashhart | Per-frame SSE memory, no silent token drop on allocation failure, a connection whose reply never completed is closed |
| ashhart 48e60da98 | ashhart | `cached_tokens` never exceeds the prompt |
| ashhart 5aedb86f7 | ashhart | A request's top-level `thinking` switch is read, as DeepSeek's own API sends it |
| ashhart c7f583d00 | chaog992 | Non-finite floats spelled as Python's repr does |
| ashhart a442ce6b9 | evilpsycho42 | A streamed Anthropic `message_start` carries the prompt's token count |
| ashhart 30f05c6d4 | ashhart | A failed round, a cancellation or a close ends every job it holds, even when memory runs out |
| bertholomus cf6aae2b8 | evilpsycho42 | A prompt ending within the window past a kept state resumes an earlier one and keeps it |
| bertholomus a6fed17ee | bertholomus | A kept state the lanes would refuse is passed over for an earlier one; each refusal says why |
| bertholomus 33b521d7d | bertholomus | A follower whose GPU failed for good stops at once, so rank 0 stops too (rewritten for HIP) |
| ashhart 4ded1802d (part) | Jay Leaton | `SCHEMA_PROMPT`: a request's `response_format` schema shown in its prompt as DeepSeek's encoder writes it, "## Response Format" (off by default) |

Our own changes made while porting: each half of a split prompt chunk passes its own positions to the replay cut
(needed once the cut uploads positions); and tests for the memo's exactness on DeepSeek-shaped tokenizers.

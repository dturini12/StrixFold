# w-tri: the dsv41 auxiliary kernels for gfx1151

Owner: w-tri. These are the DeepSeek-V4.1 family's non-EXL3 kernels: mHC, norms and RoPE, the window KV and the FP4 caches, the router, the indexer and top-k, sparse attention, Engram, the DSpark taps and the Markov steps.

## The Triton kernels (39 entry kernels + w-exl3's `_gemm`; the 204 recorded launches and 76 derived ones, 202 variants)

1. `fetch_upstream.sh` extracts the pinned sources from this repository's history into `$TRI_WORK/upstream` (default `~/tfp/tri`) and checks them against `SHA256SUMS`:
   - `kernels.py`, `markov.py` and `prefill.py` at 222a9c0b;
   - the recorded launches at strix-port-base.
2. `make_port.py UPSTREAM tri_kernels.py` copies every `@triton.jit` body verbatim. It replaces only the NVIDIA-only constructs, each by name and an exact number of times:
   - PDL `gdc_wait` / `gdc_launch_dependents` become no-ops. The `PDL` constexpr stays a variant key.
   - PTX `prefetch.global.L2` / `discard.global.L2` become no-ops.
   - The `float8e4nv` conversions become integer RNE code.
   One edit is a correctness fix rather than a port: the router score (`_route_row`, `_route_parts`) becomes `_sqrt_softplus`. Upstream's `sqrt(log(1 + exp(lg)))` in fp32 scored 0 below lg = -16.6, so such an expert was picked by its bias alone. `_sqrt_softplus` uses F.softplus's formula with a Cody-Waite exp (`_exp_rr`; `tl.exp` here is `exp2(fl(x log2 e))`, about |x| u32 off) and `libdevice.log1p`, a correctly rounded sqrt, and `exp(lg / 2)` below -16. It is within 4 u32 over [-100, 30] (numerics.json router.noaux_tc), checked by `t_route.py t_score_grid`.
   `tri_kernels.py` is the generated, committed result.
2b. `variants.py` writes `zig/src/families/dsv41/fixtures/derived_launches.json`: the launches the engine makes that the recording lacks. The prompt GEMM `_gemm` gets one variant per (K, N, OUT dtype, o_stride) of every linear the prompt path sends through it, read from the bundle's per-rank shapes (`inventory.json`) and the call sites' output types. The decode and verify window kernels (`_sparse_attn_merge_rot2`, `_rowmm_parts`, `_rowmm2r`, `_hc_mix_part`) get every R from 1 to 16 rows for each recorded configuration, by the Zig wrappers' rules. The indexer's launches (`_index_score`, `_score_keys`, `_index_score_cand`, `_topk_sel`, `_topk_finish`, `_gather_tiles`) get every context the engine serves up to the pool limit (1,048,527 positions, `--pool-max`; a ratio-2 layer of a 1M round scans 524,263 keys): the rounds' context buckets (powers of two from 1,024, or the pool's own size) and every prompt chunk's compressed-key count, each int argument by its Triton specialization (1, a multiple of 16, other). Only variants the recording does not already compile are written. `tri.zig`'s conformance tests replay the derived launches through the wrappers as they do the recording, so each one must be exactly what the engine's wrapper asks for. Rerun it when the weights' shapes or those wrappers change.
3. `aot_build.py LAUNCHES OUT --extra .../fixtures/derived_launches.json --jobs 8` compiles each recorded and derived launch with Triton (the pinned AMD build: `aot.json`'s `triton` record) into `OUT/hsaco/<sha256>.hsaco`. It writes `OUT/aot.json` in the CUDA kit's KernelJson fields plus `arch`, `warp_size`, `block`, `file`, `options` and `isa_stats`; the toolchain record holds both inputs' hashes. `tf-hip-kit scan --triton OUT` imports it into a kit manifest. Run it under the house limits:
   `systemd-run --user --scope -p CPUQuota=800% -p MemoryHigh=12G -p MemoryMax=16G taskset -c 8-15,24-31 nice -n 10 ~/venvs/triton/bin/python aot_build.py ...`

Toolchain: `triton==3.8.0+git675c5987.rocm10.1.0a20260822` (cp313), the wheel from `https://rocm.nightlies.amd.com/whl-multi-arch/` with sha256 `1bab2eeb8f4beccad389a497a45b3018fa6291ab93c577f04b1376cf12422499`, installed `--no-deps` into a user venv. Tests also need `ml_dtypes==0.5.3` (sha256 `6936283b56d74fbec431ca57ce58a90a908fdbd14d4e2d22eea6d72bb208a7b7`).

## The native kernels

`zig/kernels/hip/dsv41/aux.hip` holds the 12 `tf_ds_aux_*` kernels. They are registered in `registry.json` (source `aux.hip`, kit `dsv41`).

`zig/kernels/hip/dsv41/aux_attn.hip` holds `tf_ds_aux_sparse_attn_h32_d512` (`aux.sparse_attn`). It is a WMMA replacement for `_sparse_attn_part`: split mode writes the Triton partial layout, which the Triton merge kernels combine, and one-pass mode writes the output directly. It is tested by `t_hipattn.py` and timed by `x_hipattn_perf.py`. `t_wmma.py` probes the WMMA operand layouts it relies on.

`zig/kernels/hip/dsv41/aux_index.hip` holds `tf_ds_aux_index_rows_h32_d128` (`aux.index_rows`), the prompt path's indexer scores (a WMMA replacement for `_index_score_rows`, keys and tile maxima included). `aux_hc.hip` holds `tf_ds_aux_hc_mix_pf_d5120` (`aux.hc_mix_pf`), the prompt path's mHC posted streams and mixes (replacing `_hc_mix_pf`; its posted streams are bit-identical to the Triton kernel's). `aux_gate.hip` holds `tf_ds_aux_gate_logits_d5120` (`aux.gate_logits`), the prompt path's MoE router logits (x bf16 times the fp16 gate weight with each weight split exactly into two bf16, replacing two fp32 conversions and a hipBLAS sgemm); `t_hipgate.py` grades it and the routing it produces (against the exact logits and against the hipBLAS path), `x_hipgate_perf.py` times it. They are tested by `t_hipindex.py` and `t_hiphc.py` and timed by `x_hipindex_perf.py` and `x_hiphc_perf.py` under `rocprofv3 --kernel-trace`, whose runs `x_runs.py` prints. `x_bw.py` with `bw_probe.hip` measures the streaming bandwidth reference (about 225-245 GB/s on node1). Timing scripts call `common.quiet_window()`: the measuring process takes `/var/tmp/halo-gpu.lock` itself for the run and opens a quiet window with `~/bin/halo-quiet on|off <tag>` (closed at exit). `TRI_GPU_LOCK=1` makes any test take the lock the same way. `x_attn_ab.py name:block:hsaco ...` times builds of `aux_attn.hip` against each other on one set of inputs with the rounds interleaved (the GPU shares DRAM and power with the CPU jobs, so separate runs drift by 20-30%); `x_attn_one.py` launches one build for `rocprofv3 --pmc` counter passes; `x_wmma_rate.py` with `wmma_rate.hip` measures the WMMA ceiling (about 50 TFLOPS bf16 on node1, one WMMA about every 32 cycles a SIMD, a dependent chain as fast as independent ones from 2 waves a SIMD).

## Tests (GPU: run under the GPU lock)

`run_tests.sh [t_*.py ...]`. The test programs:
- `t_basic`, `t_hc`, `t_route`, `t_index`, `t_attn` and `t_markov` cover every recorded variant, and the derived ones (`common.launches()`).
- `t_gemm` covers every `_gemm` variant under numerics.json exl3.linear's acceptance (hard and probabilistic bounds, nrmse).
- `t_cover` (no GPU work) checks that every recorded and derived launch has its own variant in the kit by the strict-mode rule; `aot_build.py` skips a variant that fails to compile.
- `t_real` runs on checkpoint tensors.
- `t_native` covers the aux object (`TRI_AUX_HSACO`) and hipBLAS.

They all launch the kit's code objects through `hiprt.py`, a ctypes HIP binding and a variant selector with `aot.zig`'s rules; there is no torch at run time. They compare against `refs.py`, float64 references written from DeepSeek's `inference/model.py` and `kernel.py` definitions. Contracts:
- exact for integer, selection and packed outputs;
- γ_m·Σ|terms| bounds for fp32;
- the rounding interval of that bound for bf16/fp16;
- near-tie classification for discrete picks;
- bit-equality across row counts (row invariance).

Results go to `$TRI_WORK/results/*.jsonl`. `status.py` summarises them into `kernel_status.json`.

`t_perf.py` and the `x_*.py` experiments measure device time. Run them under `rocprofv3 --kernel-trace` with the GPU lock (`/var/tmp/halo-gpu.lock`) held for the run, never a whole sweep, inside a quiet window (`~/bin/halo-quiet on <tag>` ... `~/bin/halo-quiet off <tag>`, always paired); the quiet lock is retired. `prof_table.py` aligns the trace with the log.

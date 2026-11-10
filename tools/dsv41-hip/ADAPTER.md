# dsv41 on HIP: GPU adapter, kernel ABI, kit manifest and build (design v1)

Owner: w-arch. Status: v1 approved by the integration owner on 2026-10-19 with the decisions in section 13. Every number
below was read from the sources or measured read-only on node1 on 2026-10-18 (PDT), and is marked where it is a decision
rather than a fact.

Inputs read:
- baseline tree `1beea23831603b98242311979137e91da4e92f25` (bertholomus/TensorFold 3ff3b2ac + TP2 patches 0001-0003), the
  copy in `tf-port/critic-a/wt-am` (same tree hash as node1 `~/tfp/src/TensorFold` branch `strix-port-base`);
- ashhart/TensorFold `d46d17a784758b8a8b66badde1f728f36bdc9541` `zig/src/hip/*`, `zig/build/hip.zig`, the gfx1151 receipt;
- the Triton sources at bertholomus `222a9c0b` (`families/deepseek_v41/cuda/kernels.py`, `markov.py`, `cuda/exl3/prefill.py`);
- node1's installed toolchain (HIP 7.15.26333, AMD clang 23.0.0git 8f497e09, ROCm 10.0 under `/opt/rocm/core-10.0`).

Generated companions (regenerate with the two scripts; both read-only, both pinned to tree 1beea238 line numbers):
- `CUDA-SITES.md`, `cuda_sites.json` from `inventory_cuda.py ROOT`: every CUDA-coupled line with category, reachability
  and HIP replacement (1,196 rows; 476 distinct served lines, 27 behind off switches, 251 excluded configurations).
- `KERNELS.md`, `kernels.json` from `kernel_registry.py ROOT`: every kernel and library call the text configuration
  reaches, with owner, CUDA origin, recorded variants or specializations and Zig launch sites.

## 1. Decisions (summary)

| # | Decision | Why |
|---|---|---|
| D1 | One comptime-selected backend module `gpu` replaces `@import("cuda")` in the family. `gpu` re-exports either a CUDA facade over the untouched `zig/src/cuda` runtime or the new `zig/src/gpu/hip` backend. No runtime vtable. | The binary is built per backend; a vtable adds cost and indirection for no benefit. The family diff stays mechanical and reviewable against CUDA semantics. |
| D2 | Device addresses stay `u64` in the family (`gpu.Addr`). The HIP backend converts at its call boundary. | 170 served type references and every kernel argument already use `u64`; HIP pointers are 64-bit. |
| D3 | The HIP backend is a backport of ashhart d46d17a leaf modules (abi, runtime, context, memory, stream, module, launch, args, graph, graph_split, caps, code_object, device_arch.c), with `sys` replaced by `std`, error logging added, and per-thread binding by `hipSetDevice` rather than the deprecated `hipCtx*` API. | Proven on gfx1151 (receipt 833bbfcb); all of its hand-declared HIP enum values and struct layouts were re-checked against the HIP 7.15 headers on node1 and match (section 9.2). |
| D4 | Every direct driver-table call in the family (`d.api.cu*`, 19 served sites) becomes a named facade function. The family never touches a backend API table again. | Removes the remaining CUDA spelling and gives one place for HIP-specific semantics. |
| D5 | Host-to-device copies from pageable memory (`prompt.upload` and 25 more callers passing stack arrays) go through an adapter-owned pinned staging ring: the caller's bytes are copied before the call returns, the ring slot is recycled behind an event. Refused while the stream is capturing. | The family relies on CUDA's documented "pageable source is staged before return" rule (prompt.zig:515). HIP does not document the same guarantee; we do not depend on it. |
| D6 | Kernels are loaded only from a manifest-described kit of raw gfx1151 code objects (`.hsaco` ELF, not offload bundles). The manifest is generated at build time from each object's own AMDGPU metadata note and re-verified at load: exact arch (`gcnArchName` and ELF `e_flags` mach 0x4a), ABI version, SHA-256 per object, and per-kernel explicit-argument sizes/offsets against the Zig declaration. No CUBIN/fatbin loading, no C++ name parsing. | Handoff 6A. The metadata carries everything needed (verified on node1, section 6.4). |
| D7 | Every hand-written HIP kernel has a stable `extern "C"` name `tf_ds_<group>_<op>[_<spec>]`; template instantiations get explicit wrapper names; the manifest maps `(id, spec)` to the symbol so the loader never parses names. | Replaces `elf.Symbols` + mangled-template parsing (25 served sites). |
| D8 | A kernel's ABI is a Zig `extern struct` of its parameters (`gpu.Kernel(id, Params)`), one field per C parameter, or one field holding a by-value C struct declared in a shared header that both Zig (translate-c) and hipcc compile. Load-time check compares the Params table to the hsaco metadata; an echo kernel test proves every field end to end on the GPU. | Handoff 6A: sizes/offsets/alignment asserted on both sides plus a real launch test of every field. |
| D9 | Triton kernels keep their parameter-list ABI and the existing variant-selection rules (`aot.zig`), now over manifest entries for AMD-compiled objects. Strict mode (`TF_GPU_STRICT=1`, always on in qualification) refuses the "assumes less" fallback and any launch not in the manifest. | Same semantics the family already tests; fallbacks become visible failures instead of warnings. |
| D10 | No PDL, clusters or cooperative launches on HIP: `gpu.caps.pdl = false`; the family derives `Tri.pdl` and the EXL3 `pdl` flag from caps; a `Config` with `pdl`/`cluster` set is refused on HIP. Plain stream order replaces programmatic serialization. | No AMD equivalent; stream order is strictly stronger, so correctness is preserved. |
| D11 | Collectives move behind a family-level `Coll` interface (stream-ordered rank-ordered all-gather + `capturable` flag + abort). NCCL becomes the CUDA implementation; w-link supplies the host-staged TCP implementation and later a capturable device-proxy one (the rdma_gather stage/collect pattern over TCP). | 10 served collective call sites + NCCL bootstrap; graphs need to know whether a collective can live inside a capture. |
| D12 | Graphs are off for first light. They are allowed only when the active `Coll` is capturable (device proxy); otherwise `TF_DS_GRAPHS=1` is refused at startup, with ashhart's `graph_split.Recorder` as the explicit fallback option. | Splitting at ~94 gathers per round would leave graph pieces of one layer each; the device-proxy design keeps a stretch whole. |
| D13 | Text-only capability is an explicit `Modalities` option in the open/load contract (`native.Options`, `model.Options`, rank 1's flags), defaulting to `text` on HIP (`text_image` opens the HIP tower: section 8.1). With `text`, Vision.open (native.zig:148) and the routing-bias load (model.zig:330-333) never run; config.json and its hash are untouched; images are refused by the server's existing text-only path. | Handoff 6A. |
| D14 | New build file `zig/build/hip_dsv41.zig`: step `native-hip` (x86_64 host + gfx1151 kit, no CUDA module in its graph), `dsv41-cpu` (CPU-only prep/check tools built from a GPU-free `dsv41_cpu` module root), host tests and GPU tests as separate steps. | Handoff 4, 6A, PLAN shared rules. |
| D15 | The CUDA runtime `zig/src/cuda/*`, CUDA kernels and CUDA build file stay byte-identical. The family edits are mechanical; the CUDA facade keeps the family compiling host-only (no nvcc) as a cheap regression check, not a gate. | PLAN: CUDA build dropped as a gate, CUDA source kept for reference. Owner may drop the CUDA facade if it costs more than a day. |

## 2. Current CUDA coupling (counts)

From `cuda_sites.json` (served = either rank's serving path in the advertised configuration TP2, text, 1 stream, 8K):

| Category | Served sites | Behind off switches | What it is |
|---|---:|---:|---|
| `cuda.*` type references (rename only) | 170 | 7 | Stream/Function/Args/DeviceBuffer/Module/... in signatures and fields, 25 files |
| Direct driver-table calls | 19 | 0 | cuMemcpyDtoH_v2 ×6, cuMemcpyDtoDAsync_v2 ×5, cuMemsetD8Async ×4, cuMemsetD32Async ×2, cuMemcpyHtoDAsync_v2 ×1, cuCtxSynchronize ×1 |
| Kernel launches (`cuda.launch.launch`) | 20 | 1 | ops.zig 10, exact.zig 5, exl3_* 4 (+ generic launchers), L2 prefetch optional |
| Triton launch sites (`t.run`) | 45 | 0 | 40 entry kernels, 204 recorded specializations |
| Module loads / embedded fatbins | 8 / 3 | 0 | 4 kit cubins + 3 embedded images (pointwise, movement, dsv41_ops, dsv41_torch) |
| Mangled-name parsing (`elf.Symbols`, template args) | 25 | 0 | exl3_linear, exl3_prefill, exl3_experts |
| Triton AOT set (`cuda.aot`) | 10 | 0 | load at model.zig:224, the Tri type in 7 files |
| PDL (launch attribute or `PDL` constexpr) | 47 | 0 | 45 Triton constexprs, exl3_linear.zig:172/219 |
| Graph capture/replay | 0 | 14 | round.zig:848-890, draft.zig:117-152 (TF_DS_GRAPHS) |
| Streams / events / synchronize | 3 / 4 / 15 | 0 / 2 / 1 | main stream, side and prefetch streams, PhaseTimer events |
| Blocking copies / async copies / fills | 32 / 6 / 6 | 1 / 0 / 0 | 26 `prompt.upload` callers pass pageable (stack) memory |
| cuBLAS | 8 API calls, 4 GEMM sites | 0 | cublasSgemm_v2 at prompt.zig:946, 1041, draft.zig:631 (xwT) and draft.zig:541 (xv) |
| NCCL | 11 API calls, 10 collective call sites | 0 | bootstrap model.zig:206-209; prompt.zig:1012/1232/1348; round.zig:978/998/1008/1027; draft.zig:108/504/511 |
| Direct NVIDIA dlopen | 1 | 0 | `libcublas.so.13` (cublas.zig:38); `libcuda.so.1` at rdma.zig:96 is excluded |
| Engine host (`native/cuda.zig`) | 22 | 0 | driver/context/kernel-dir/usage; registry `{nemotron, dsv41}` |

Excluded on HIP (refused with a named error, never silently skipped): verbs RDMA ring (rdma.zig, rdma_seq.zig, ring2d.zig:
66 sites, also `libcuda` host-register), the four-node 2D split (prompt2d.Two, round2d, draft2d, exl3_experts2d), vision
(vision/vit/fmha/picture/pil/cublaslt: 18 sites in served files, the fmha CUTLASS cubin, cuBLASLt, 8 vision kernels in
dsv41_ops.cu), and the NVIDIA L2 prefetch (`TF_DS_L2_PREFETCH`).

Device code that needs an AMD implementation (detail in section 10): 40 Triton entry kernels (204 recorded
specializations), 20 required native CUDA kernels from 4 sources (+1 profiling-only), 14 EXL3 kernel ids with 29
checkpoint-required specializations (from 4 kit cubins), 2 cuBLAS SGEMM shapes, and the collective transport.

## 3. The backend-neutral GPU interface (`gpu`)

### 3.1 Module layout

```
zig/src/gpu/gpu.zig          facade: pub const backend = .hip | .cuda (build option); re-exports the selected backend
zig/src/gpu/cuda_facade.zig  CUDA: re-exports zig/src/cuda (untouched) + the new free functions (copies, fills, staging)
zig/src/gpu/hip/             HIP backend (ashhart backport + new files):
  abi.zig runtime.zig context.zig memory.zig stream.zig module.zig args.zig launch.zig graph.zig graph_split.zig
  caps.zig code_object.zig device_arch.c          (backported, adapted)
  hip_abi_check.c                                 (new: _Static_assert of every enum value / layout abi.zig declares)
  staging.zig                                     (new: pinned staging ring for pageable uploads, D5)
  hsaco.zig msgpack.zig                           (new: ELF note + AMDGPU metadata reader)
  kit.zig kernel.zig triton.zig                   (new: manifest loader, typed kernels, Triton variant set)
  blas.zig                                        (new: hipBLAS by dlopen)
zig/src/families/dsv41/coll.zig                   Coll interface (family-level); NCCL impl wraps comm.zig
zig/src/families/dsv41/cpu.zig                    GPU-free module root for the CPU tools (section 9.3)
zig/kernels/hip/dsv41/abi.h                       shared C ABI header (by-value structs, ABI version)
zig/kernels/hip/dsv41/<group>.hip                 HIP kernel sources, one file per owner group
zig/build/hip_dsv41.zig                           build steps (section 9)
tools/dsv41-hip/                                  ADAPTER.md, inventory scripts, kernels.zon registry, kit tools
```

### 3.2 Interface table

Ownership column: who frees, and what must have completed first. "Stream-ordered" = runs after earlier work on that
stream and before later work.

| `gpu.` API | Semantics and ownership | CUDA facade | HIP backend |
|---|---|---|---|
| `backend`, `Caps` | comptime backend tag; `Caps{ arch[32], wave, cus, lds_per_block, integrated, wall_clock_hz, pdl, clusters, graphs, host_mapped, wmma_f16, wmma_bf16, dot2_f16, sdot4 }` from the device, never from a name compare in family code | capability, attributes | `hipGetDevicePropertiesR0600` via device_arch.c (gcnArchName), `hipDeviceGetAttribute` (integrated 16, CUs 63, warp 87, LDS 74, wall clock 10017), ashhart `caps.zig` table + `wmma_f16` |
| `Runtime.open() / close / deviceCount / errorName / errorText` | complete symbol table or refuse (`MissingSymbol`) | `Driver.open` (libcuda.so.1) | dlopen `libamdhip64.so.7`, `hipInit(0)`, every `abi.Api` field resolved |
| `Context.init(rt, ordinal) / deinit / makeCurrent / synchronize / memInfo` | one device; `makeCurrent` on every thread that calls the runtime (lane thread) | primary ctx retain, cuCtxSetCurrent | `hipSetDevice` (thread-local current device), `hipDeviceSynchronize`, `hipMemGetInfo` |
| `Addr = u64` | a device address; offsets are plain adds | `CUdeviceptr` | `@intFromPtr` of `hipDeviceptr_t` |
| `DeviceBuffer.alloc(rt, len) / fromHost / free / at / upload / download / uploadAsync / downloadAsync / copyFrom / fill8 / fill32` | owned by one value; `free` only after every stream that used it has passed a completed event or the context is synchronized; zero length holds address 0 | cuMemAlloc etc. | `hipMalloc` (coarse-grained; on gfx1151 backed by GTT system memory), `hipFree` (note: implicitly synchronizing on AMD, so no frees in steady state), `hipMemcpy*`, `hipMemsetD8/D32[Async]` |
| `HostBuffer.alloc(rt, len, flags) / free / device() / slice` | pinned host bytes; an async copy that reads or writes them keeps them alive and unmodified until its completion event | cuMemHostAlloc | `hipHostMalloc(Portable | Mapped [| Coherent])` (flag values 1, 2, 0x40000000 checked against headers); `hipHostGetDevicePointer` |
| `MappedSlab.alloc(rt, len, .{ .coherent = true }) / register(bytes) / free` | host-coherent memory both CPU and GPU address, for transport slabs (w-link owns the slot state machine); never freed while a transfer or kernel may reference it | cuMemHostRegister (rdma.zig today) | `hipHostMalloc(Portable|Mapped|Coherent)` or `hipHostRegister(Portable|Mapped)` + `hipHostGetDevicePointer`; visibility proven by w-link's test, not assumed |
| `copyD2H(rt, out, src)` `copyH2D(rt, dst, bytes)` | blocking; replaces `d.api.cuMemcpyDtoH_v2/HtoD_v2` | cuMemcpyDtoH_v2 / HtoD_v2 | `hipMemcpyDtoH / hipMemcpyHtoD` |
| `copyD2DAsync(stream, dst, src, n)` | stream-ordered | cuMemcpyDtoDAsync_v2 | `hipMemcpyDtoDAsync` |
| `uploadStaged(stream, dst, bytes)` | stream-ordered; **the caller's bytes may be reused when it returns** (D5); error `InCapture` if the stream is capturing | cuMemcpyHtoDAsync_v2 (CUDA stages pageable memory itself) | memcpy into a slot of a pinned ring (default 4 MiB, slots recycled behind events; larger requests use a one-off pinned buffer + synchronize), then `hipMemcpyHtoDAsync` |
| `fill8Async / fill32Async(stream, dst, value, n)`, `fill8(rt, ...)` | stream-ordered / blocking | cuMemsetD8Async / D32Async / D8_v2 | `hipMemsetD8Async / hipMemsetD32Async / hipMemsetD8` |
| `Stream.init(rt, .{ .non_blocking = true }) / deinit / synchronize / synchronizeFor(ns) / done / wait(event)` | `synchronizeFor` polls `done` to a deadline and returns `Timeout` (the work is not cancelled; the caller aborts the request and the peer) | cuStream* | `hipStreamCreateWithFlags(1)`, `hipStreamSynchronize`, `hipStreamQuery`, `hipStreamWaitEvent` |
| `Event.init(rt, timing) / record / synchronize / done / elapsedMs` | ordering-only events skip timestamps | cuEvent* | `hipEventCreateWithFlags(0 or 2)`, `hipEventRecord`, `hipEventElapsedTime` |
| `Module.load(rt, image) / unload / function / global` | raw image load, used only by `Kit` | cuModuleLoadData | `hipModuleLoadData` (raw ELF, 8-byte aligned), `hipModuleGetFunction`, `hipModuleGetGlobal` |
| `Function.attribute / occupancy / allowDynamicShared(bytes)` | dynamic LDS opt-in is a check on HIP | cuFuncSetAttribute | `hipFuncGetAttribute`, `hipModuleOccupancyMaxActiveBlocksPerMultiprocessor`; `allowDynamicShared` returns `Unsupported` if static + dynamic > 64 KiB |
| `Args`, `Config{ grid, block, shared, pdl=false, cooperative=false, cluster=null }`, `launch(f, cfg, stream, *Args)`, `launch.count` | one parameter value per C parameter at its C alignment (unchanged packing rules) | cuLaunchKernel(Ex) | `hipModuleLaunchKernel(kernelParams)`; `pdl`/`cluster` refused (`Unsupported`); strict mode checks sizes against metadata |
| `Kernel(id, Params)` | typed kernel: `bind(kit, spec)` resolves and verifies, `launch(stream, cfg, params)` packs fields in order (section 6) | binds a Function, no metadata check | binds from `Kit`, verifies against hsaco metadata |
| `Kit.open(rt, dir, required) / get(id, spec) / deinit / counters` | every required kernel resolved at open, or open fails naming the missing ids | CUDA kit dir (cubins + aot.json) through the existing loaders | manifest kit (section 7) |
| `TritonSet` (`run(stream, name, grid, args, consts)`, `smallestConst`) | the `aot.Set` API and selection rules, strict mode refuses fallbacks | `cuda.aot.Set` | manifest Triton entries, `block = num_warps * wave` from metadata |
| `graph.beginCapture / endCapture / captureStatus / Graph / Exec{ instantiate, upload, launchOn, update, deinit }` | `thread_local` capture mode; an Exec must be destroyed before any buffer it references | cuStream*Capture, cuGraph* | `hipStreamBeginCapture`, `hipStreamEndCapture`, `hipGraphInstantiateWithFlags`, `hipGraphUpload`, `hipGraphLaunch`, `hipGraphExecUpdate` (HIP's 4-argument form) |
| `graph_split.Recorder` | graphs cut at each collective (fallback when `Coll` is not capturable) | n/a | ashhart graph_split.zig over `Coll` |
| `Blas.open(stream, workspace) / close / xwT / xv` | fp32 SGEMM exactly as torch issues it (row-major via the transposed column-major call) | cuBLAS | dlopen `libhipblas.so.3`: `hipblasCreate`, `hipblasSetStream`, `hipblasSetWorkspace` (32 MiB, fixed), `hipblasSetAtomicsMode(NOT_ALLOWED)` for run-to-run determinism, `hipblasSgemm` (all five symbols present on node1) |
| `usage(reset_peak)` | bytes held in DeviceBuffers/HostBuffers and device peak | same | same counters |
| `health() / markFault(code)` | first device-fatal code seen (illegal address, launch failure, ...) recorded process-wide so lanes/Coll can tell the peer before exiting | none today | set by the error mapper; read by lanes.zig and `Coll.abort` |

### 3.3 Error model

- Every facade call returns `gpu.Error!T`, `Error = error{ DriverUnavailable, MissingSymbol, GpuFailed, OutOfDeviceMemory,
  NotReady, NotFound, Invalid, Unsupported, DeviceFault, Timeout, InCapture, AbiMismatch, MissingKernel, NoKernelImage }`.
- The HIP checker logs `call: hipErrorName (code) hipGetErrorString` once per failure (ashhart's `check` drops this) and
  maps: 2 → `OutOfDeviceMemory`, 600 → `NotReady` (a state, not a failure), 500 → `NotFound`, 98/209 (no binary for the
  device) → `NoKernelImage`, 700/719/720 and other sticky codes → `DeviceFault` + `markFault`; everything else `GpuFailed`.
- `TF_GPU_SYNC=1` (bring-up): synchronize after every launch and name the kernel id on failure. This is how #25-style
  faults get attributed.
- Note for the fault matrix (w-link): on ROCm a GPU memory-access fault normally terminates the process from the runtime
  (`Memory access fault by GPU node`), so a rank-1 fault reaches rank 0 as a closed TCP link, not as an error return.
  Peer-death detection, not error codes, is the primary path for #25 on AMD.
- Kernels that wait on host-written flags (device proxy) must check an abort word and a device-side `wall_clock64()`
  deadline; no kernel may spin without bound.

### 3.4 Lifetime rules (enforced where cheap)

1. A `DeviceBuffer` is freed only after `Context.synchronize`, or after an `Event` recorded after its last use has
   completed. Debug builds track the last stream that used an Arena block and assert on free.
2. A `HostBuffer`/`MappedSlab` read or written by an async copy or kernel stays alive and unmodified until that work's
   completion event. `uploadStaged` exists so pageable callers never have to know this.
3. A graph `Exec` references device addresses captured at capture time: Arena blocks outlive every `Graphs` map (already
   true: model.zig:347-348 before 365).
4. Nothing pageable is captured into a graph (`uploadStaged` refuses during capture).
5. A module (and so every Function from it) outlives every Exec and every queued launch of its kernels.

## 4. Call-site mapping (served code)

The complete per-line table is `CUDA-SITES.md`. The non-mechanical sites:

| Site(s) | Today | Replacement |
|---|---|---|
| native.zig:28, model.zig:9, lanes.zig:23, round.zig:14, draft.zig:13, prompt.zig:14, prompt2d.zig:15, comm.zig:5, weights.zig:5, exl3_{linear,prefill,experts}.zig, tri*.zig, ops.zig:8, exact.zig:6, cublas.zig:6, lanes_main.zig:12 | `@import("cuda")` | `@import("gpu")`; `cuda.X` → `gpu.X` (170 type references) |
| native/cuda.zig:28,318,329,335-337,343,353 | Driver/Context, kernel dir `share/tensorfold/cuda/sm<cap>`, `TENSORFOLD_CUDA_KERNELS`, chip `nvidia-sm121` | `zig/src/native/gpu.zig` (same host logic over `gpu`; registry `{dsv41}` on HIP): kit dir `share/tensorfold/hip/<arch>/dsv41` or `TF_DS_HIP_KIT`, chip class `amd-gfx1151`, messages say HIP |
| lanes_main.zig:127-130 | `cuda.Driver.open`, `cuda.Context.init(&driver, 0)` | `gpu.Runtime.open`, `gpu.Context.init` (+ `--modalities`, `--hip-kit`) |
| model.zig:189, round.zig:342, 362 | `cuda.Stream.init(d, true)` | `gpu.Stream.init(rt, .{ .non_blocking = true })` |
| model.zig:224 | `cuda.aot.Set.load(..., kit/aot)` | `gpu.TritonSet` from the kit manifest |
| model.zig:226-235, exl3_prefill.zig:22-39, exl3_linear.zig:97-116, exl3_experts.zig:67-118, ops.zig:42-72, exact.zig:37-48 | `Module.load(cubin)` + `elf.Symbols` + mangled-template parsing; embedded fatbins `cuda.kernels.*` | `Kit.open` once in model.open with the required id set; each loader becomes `Kernel(id, Params).bind(kit, spec)` per kernel/spec (comptime branch keeps the CUDA loader as it is) |
| model.zig:287, round.zig:597, draft.zig:533, prompt.zig:522 | `cuMemsetD8Async` | `gpu.fill8Async` |
| draft.zig:314, prompt.zig:527 | `cuMemsetD32Async` | `gpu.fill32Async` |
| model.zig:491, 498, round.zig:316, prompt.zig:1067-1068 (vision) | `cuMemcpyDtoDAsync_v2` | `gpu.copyD2DAsync` |
| prompt.zig:517 (`upload`, 26 callers incl. draft.zig:318-467, round.zig:265, 478, 488, 1108) | `cuMemcpyHtoDAsync_v2` on pageable memory | `gpu.uploadStaged` (D5) |
| draft.zig:472-473, model.zig:682 (profile), 741, 748, 756 | `cuMemcpyDtoH_v2` | `gpu.copyD2H` |
| model.zig:303 | `cuCtxSynchronize` | `ctx.synchronize()` |
| weights.zig:179-181, 347-357, model.zig:241-250 | `DeviceBuffer.alloc` + blocking `upload` of staged weight bytes | unchanged API; HIP `hipMalloc` + `hipMemcpyHtoD` (bounded staging is already in weights.zig) |
| model.zig:206-209, 378; prompt2d.zig:66-84; comm.zig:17-54 | NCCL unique id over the link, `ncclCommInitRank`, `ncclAllGather`, send/recv (world 4) | `Coll.open(fan.links, rank, world, cfg)` after the identity handshake; NCCL impl is CUDA-only |
| prompt.zig:131-134 (`gatherF32`), 1010-1013, 1232, 1348; round.zig:978, 998, 1008, 1027; draft.zig:108, 504, 511 | `e.comm.allGather` / RDMA ring | `e.coll.allGather(stream, send, recv, count, .f32)`; the rank-ordered add stays in the family (`add2_bf16`, the `_hc_*` mixes), numerics unchanged |
| model.zig:210-221, 259-275, 389-410 | RDMA ring / world-4 rings | not compiled on HIP (`error.NoVerbsRdma` / `error.BadWorld` with an explain text if `TF_RDMA_DEVICES` or world 4 is set) |
| model.zig:280-281, 364; prompt.zig:946, 1041; draft.zig:541, 631; cublas.zig | cuBLAS | `gpu.Blas` (hipBLAS); `linearBf16` is vision-only and not provided on HIP |
| round.zig:321-375 (side stream, prefetch stream) | events + streams | unchanged over `gpu`; `usePrefetch` refused on HIP (L2 prefetch is NVIDIA-only), side stream allowed |
| round.zig:780-819 (PhaseTimer), model.zig:671-682 (clock) | timing events, `%globaltimer` kernel | unchanged over `gpu`; clock kernel ported with `wall_clock64()` and `Caps.wall_clock_hz` |
| round.zig:842-891, draft.zig:117-152 | CUDA graph capture of stretches containing gathers | `gpu.graph`; allowed only with a capturable `Coll` (D12) |
| exl3_linear.zig:172, 219; every `t.pdlConst()` | PDL | `pdl = gpu.caps.pdl` (false on HIP); Triton variants compiled with `PDL=0` |
| native.zig:146-150, model.zig:329-333, lanes.zig:230, lanes_main.zig `--vision-bias` | Vision tower + routing bias opened when config.json has `vision_config` | `Modalities` (section 8) |

## 5. Collectives and graphs (boundary with w-link)

```zig
// zig/src/families/dsv41/coll.zig (w-arch writes the interface and the NCCL wrapper; w-link writes the TCP ones)
pub const Coll = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    rank: u32,
    world: u32,
    /// Device work captured into a graph may call allGather (device-proxy implementations only).
    capturable: bool,
    pub const VTable = struct {
        /// recv[r * bytes .. (r + 1) * bytes] = rank r's send, for every rank, rank order. Stream-ordered: reads send
        /// after earlier work on `stream`; later work on `stream` sees recv. Errors: CollectiveFailed, PeerLost, Timeout.
        all_gather: *const fn (*anyopaque, stream: gpu.Stream, send: gpu.Addr, recv: gpu.Addr, bytes: usize) Error!void,
        /// Fail every in-flight and future call and wake any device waiter (abort word); idempotent.
        abort: *const fn (*anyopaque, reason: Reason) void,
        close: *const fn (*anyopaque) void,
    };
};
```

- Implementations: `nccl` (CUDA, wraps comm.zig unchanged), `tcp_host` (w-link, first light: stream sync or event wait,
  copy out, TCP exchange over thunderbolt_net, copy in), `tcp_proxy` (w-link, graphs: the rdma_gather stage/collect kernel
  pair writing to and polling a `MappedSlab`, with a host proxy thread doing TCP; abort word + deadline).
- `Engine.comm` becomes `Engine.coll`; `gatherF32` keeps its contract (`count` fp32 a rank into `dst`, rank after rank).
- Bootstrap order on HIP: `Fan.open` (unchanged TCP link) → identity handshake (w-link; includes `gpu.kit.abi_version`, the
  kit manifest SHA-256, `Modalities`, world/rank, rank-cache schema) → `Coll.open`. NCCL's unique-id exchange disappears.
- Graph rule (D12): `round.Graphs.run` and `draft.PassGraphs.run` check `e.coll.capturable` at model open; with a
  non-capturable Coll, `TF_DS_GRAPHS=1` fails startup with an explain text. Fallback to eager after a capture failure uses
  the same buffers and `Carry` state as today (round.zig:877-884).
- `add2_bf16` (the rank-ordered fp32 add and RNE to bf16) is a glue kernel owned by w-arch; its contract is exact
  (one fp32 add in rank order, then round-to-nearest-even), recorded in numerics.json.

## 6. Kernel ABI

### 6.1 Names and ABI version

- Hand-written HIP kernels: `extern "C" __global__ __launch_bounds__(N) void tf_ds_<group>_<op>[_<spec>](...)`; groups:
  `glue`, `exl3`, `aux` (w-tri's HIP kernels), `xport` (w-link's stage/collect). Template code is instantiated through
  thin `extern "C"` wrappers whose names carry the spec (`tf_ds_exl3_glinear_k12_w4`).
- Triton kernels keep Triton's function name (`_rmsnorm`, ...); variants are distinguished by the manifest, not by name.
- `TF_DS_HIP_ABI_VERSION` (starts at 1) lives in `zig/kernels/hip/dsv41/abi.h` and in `gpu/hip/kit.zig`; it is bumped
  on any change to a parameter list, a by-value struct, a kernel's documented semantics, or the manifest schema. The kit
  and the binary must agree exactly; peers compare it in the handshake.

### 6.2 Parameter declarations (Zig side)

```zig
// one field per C parameter, in order; pointers are u64 device addresses; ints and floats at their C width
pub const Add2Bf16 = gpu.Kernel("glue.add2_bf16", extern struct { g0: u64, g1: u64, out: u64, count: u64 });
// one by-value C struct (declared once in abi.h, imported into Zig by translate-c, so there is a single definition)
pub const GLinear = gpu.Kernel("exl3.glinear", extern struct { a: abi.tf_ds_exl3_glinear_args });
```

`gpu.Kernel` computes at comptime each field's size, alignment and C offset in the kernarg segment, packs a launch's
values into `Args` in field order (byte-identical to today's `a.add(...)` sequences, so the CUDA path is unchanged), and
`bind` compares that table with the kernel's hsaco metadata: same count of explicit (non-hidden) arguments, and for each
one the same size and offset, `value_kind` `global_buffer` or `by_value` consistent with a `u64` address field. Any
difference fails `Kit.open` with `AbiMismatch` naming the kernel and field.

### 6.3 By-value structs (C side)

```c
// zig/kernels/hip/dsv41/abi.h  (included by every .hip file and translated for Zig by the build)
#define TF_DS_HIP_ABI_VERSION 1
typedef struct { uint64_t xh, T, svh, bias, y, Z, counters; int64_t ldx, ldy, stride_k, stride_nb;
                 int32_t y_dtype, K, N, SK, first; ... } tf_ds_exl3_glayer;     /* example only: w-exl3 defines it */
_Static_assert(sizeof(tf_ds_exl3_glayer) == 176, "abi");
_Static_assert(offsetof(tf_ds_exl3_glayer, y_dtype) == 88, "abi");          /* every field */
```

Rules: fixed-width types only (`uint64_t`, `int64_t`, `int32_t`, `uint32_t`, `float`, fixed arrays); no `bool`, `enum`,
`long`, pointers (use `uint64_t`), bitfields or packed attributes; natural alignment; explicit padding fields. Every
struct gets size + every-offset `_Static_assert`s in the header (checked by hipcc for the device and by Zig's translate-c
host compile), and Zig `comptime` asserts the same literals. The owner of a kernel group writes its structs; w-arch
reviews them against these rules.

### 6.4 What the code object tells us (measured on node1)

A test kernel compiled with the planned flags gives a raw ELF (`ELF64, EM_AMDGPU, e_flags 0x4a = gfx1151, OS/ABI
AMDGPU-HSA, ABI version 4`) whose AMDGPU metadata note lists per kernel: `.name`, `.symbol` (`<name>.kd`),
`.args[]{ .offset, .size, .value_kind (by_value | global_buffer | hidden_*), .address_space }`, `.kernarg_segment_size`,
`.wavefront_size: 32`, `.max_flat_workgroup_size` (from `__launch_bounds__`), `.group_segment_fixed_size` (static LDS),
`.private_segment_fixed_size` (scratch, i.e. spills), `.sgpr_count`, `.vgpr_count`, `.uses_dynamic_stack`. A 48-byte
by-value struct argument appears as one `by_value` arg of size 48 at offset 0. Hidden arguments are listed after the
explicit ones and are filled by HIP itself. `hipcc --genco` and `--offload-device-only` alone both emit a clang offload
bundle; `--offload-device-only --no-gpu-bundle-output` emits the raw ELF the manifest tool reads.

### 6.5 Launch metadata and checks

- Recorded per kernel in the manifest: `max_flat_workgroup_size`, `wave`, `lds_static`, `scratch`, `vgpr`, `sgpr`,
  `kernarg_size`, and from the source registry `block` (`fixed: [x,y,z]` or `max`) and `lds_dynamic_max`.
- Kit.open refuses `wave != 32`, `uses_dynamic_stack`, and (in strict mode) `scratch > 0` unless the registry entry
  allows spills for that kernel.
- Launch (strict mode): block threads ≤ `max_flat_workgroup_size`, static + dynamic LDS ≤ 65,536 and ≤ the declared
  maximum, grid dims > 0, `Args` sizes equal the bound table. Release mode keeps only the cheap geometry check that exists
  today (`Config.validate`).

### 6.6 The real launch test (handoff 6A)

`tf-dsv41-hip-abi-test` (GPU, under the lock): for every by-value struct type in abi.h a generated echo kernel
`tf_ds_abi_echo_<struct>` copies the bytes it received into an output buffer; for every typed parameter list an echo kernel
writes each parameter to its slot. The host fills every field with a distinct pattern (address-like values above 2^40,
negative ints, NaN-boxed floats, per-byte markers), launches through the same `gpu.Kernel.launch` path the family uses,
and compares every byte. It also launches each kit kernel once with a zero-size problem where the kernel allows it, to
prove `hipModuleGetFunction` and the launch path for every manifest entry. Results go to a JSON receipt with the kit
manifest SHA-256.

## 7. AMD kit and manifest

### 7.1 Layout

```
<kit>/manifest.json
<kit>/objects/<sha256>.hsaco        raw gfx1151 ELF code objects
```

Found via `TF_DS_HIP_KIT`, else `<exe>/../share/tensorfold/hip/<arch>/dsv41`. The non-kernel assets kit (`TF_DS_KIT`: RoPE
tables, engram.json) is unchanged and no longer needs `cubins/`, `aot/` or `vision/`.

### 7.2 Schema (`tensorfold.dsv41.hip-kit/1`)

```json
{
  "schema": "tensorfold.dsv41.hip-kit/1",
  "abi_version": 1,
  "arch": "gfx1151",
  "wave": 32,
  "source": { "git_tree": "<tree of the strix-port commit the kit was built from>", "dirty": false },
  "toolchain": { "hipcc": "HIP 7.15.26333-0000000", "clang": "AMD clang 23.0.0git 8f497e0992fb", "triton": "3.8.0+git...",
                 "flags": ["--offload-arch=gfx1151", "--offload-device-only", "--no-gpu-bundle-output", "-O3",
                           "-std=c++20", "-fno-gpu-rdc", "-mno-wavefrontsize64", "-ffp-contract=off"] },
  "objects": [ { "file": "objects/9f3c....hsaco", "sha256": "9f3c...", "bytes": 48216, "origin": "hip:glue.hip",
                 "owner": "w-arch", "e_flags": "0x4a" } ],
  "kernels": [
    { "id": "glue.add2_bf16", "symbol": "tf_ds_glue_add2_bf16", "object": "objects/9f3c....hsaco", "abi": "params",
      "spec": {}, "args": [ {"offset":0,"size":8,"kind":"global_buffer"}, {"offset":8,"size":8,"kind":"global_buffer"},
                            {"offset":16,"size":8,"kind":"global_buffer"}, {"offset":24,"size":8,"kind":"by_value"} ],
      "kernarg_size": 32, "max_flat_workgroup_size": 256, "block": {"fixed": [256,1,1]}, "lds_static": 0,
      "lds_dynamic_max": 0, "scratch": 0, "vgpr": 12, "sgpr": 18, "numerics": "add2_bf16" },
    { "id": "exl3.glinear", "symbol": "tf_ds_exl3_glinear_k12_w4", "abi": "struct", "struct": "tf_ds_exl3_glinear_args",
      "spec": {"k2": 12, "cb": 2, "warps": 4}, "...": "..." },
    { "id": "tri._rmsnorm", "symbol": "_rmsnorm", "abi": "triton", "object": "objects/...hsaco",
      "triton": { "params": [ {"name":"X","type":"*bf16","div16":true,"nospec":false}, ... ],
                  "consts": { "D": {"int": 5120}, "BLOCK": {"int": 8192}, "PDL": {"int": 0} },
                  "num_warps": 8, "shared": 0, "hidden": ["global_scratch", "profile_scratch"] },
      "args": [ "... from hsaco metadata ..." ] }
  ],
  "required": { "always": ["glue.add2_bf16", "..."], "drafts": ["tri._tap", "..."], "engram": ["tri._engram_gate"],
                "profile": ["glue.clock"] }
}
```

### 7.3 Building and checking a kit

- `zig build native-hip` compiles every `zig/kernels/hip/dsv41/*.hip` (one code object each) and runs `tf-hip-kit scan`
  over them with the source registry `tools/dsv41-hip/kernels.zon` (id → symbol, spec, block, LDS, numerics key, owner).
- Triton objects come from w-tri's pinned AOT step (a directory of `.hsaco` + Triton metadata JSON per variant, same
  fields as today's `aot.json`); `-Dhip-triton-kit=<dir>` makes `tf-hip-kit merge` fold them into the same manifest.
  The seed variant list is the 204 recorded launches; strict-mode runs report any launch the kit lacks, and those are
  added and rebuilt. A qualification run has zero fallbacks and zero missing variants (counters in the run's JSON).
- `tf-hip-kit verify <kit> [--config drafts,engram]`: re-hash, re-parse metadata, check arch/ABI, list required ids that are
  missing. Used by `qualify.sh`. `tf-hip-kit coverage <kit> --counters run.json` lists required ids never launched in a run.

### 7.4 Loading

`Kit.open` (once in `model.open`, both ranks): parse manifest → refuse schema/ABI/arch mismatch (runtime `gcnArchName`
must equal `arch`, ELF `e_flags` mach must be 0x4a) → per object: read, SHA-256, `hipModuleLoadData` → per kernel:
`hipModuleGetFunction(symbol)` and metadata re-parse → resolve the required set for the active features (drafts, Engram,
profile) → bind every typed kernel and the TritonSet. Any failure stops startup with the kernel id; nothing is resolved
lazily mid-request. Per-id launch counters are kept when `TF_GPU_STRICT=1` or `--profile`.

## 8. Text-only capability

- `pub const Modalities = enum { text, text_image };` in native.zig, carried in `native.Options`, `model.Options` and
  rank 1's flags (`--modalities text`, replacing `--vision-bias` there).
- Default: `text` on HIP; on CUDA `text_image` when config.json has `vision_config` (today's behaviour). `TF_DS_MODALITIES`
  overrides. `text_image` on HIP fails startup: "image input is not implemented on the HIP backend; serve text only".
- With `text`: native.zig:147-150 does not call `Vision.open`; model.zig:329-333 does not load `gate_bias_vl`; `Loaded.vision`
  is null, so the server's own path refuses image parts ("this server accepts text only; image, audio and video inputs are
  unsupported", server/prompt.zig:66) and does not advertise images. `cfg.vision` stays true and config.json and its hash
  are never modified; the startup line says "config declares a vision tower; not initialised (text capability)".
- On HIP the family root does not import vision.zig, vit.zig, fmha.zig, picture.zig, pil.zig or cublaslt.zig at all
  (`const vision_mod = if (gpu.backend == .cuda) @import("vision.zig") else @import("vision_none.zig")`, the latter an
  opaque type whose open returns `error.VisionUnsupported`); lanes.zig:230's field uses the same alias.
- Both ranks must agree: `Modalities` is a handshake field.
- Tests: (1) open with the real LibertAI config.json (vision_config present) and an empty kit `vision/` directory: startup
  succeeds, `strace -e openat` shows no read under `vision/`; (2) a chat request with an image part gets the refusal
  through the real HTTP path; (3) a text prompt containing the literal placeholder text renders as ordinary tokens (the
  #17 escape) and matches the encoder's token ids.

### 8.1 Image input on HIP (`text_image`, local/hip-vision)

Text stays the HIP default; `MODALITIES=text_image` (`TF_DS_MODALITIES` / rank 1 `--modalities`) opens the vision
tower on rank 0. Everything above about `text` still holds: a text rank opens, loads and binds nothing of vision.

- **Tower:** `vit_hip.zig` (vit.zig's `Tower` surface: open, reserve, encode with probes, spanRows) on the AMD kit's
  `vision.*` kernels (`zig/kernels/hip/dsv41/vision.hip`: linear, rms_norm, attention, rope, add, silu_mul, gelu,
  unfold, span), bound by `hip_load.vision` as an optional group. vision.zig picks the backend's tower at compile time;
  picture.zig, pil.zig, the payload and the server hook are shared with CUDA.
- **Numerical identity:** engineLab's declared arithmetic (its HIP tower, `pr/ds41-vision` 481141258, adapted, arithmetic
  unchanged), not torch's kernels: every linear one FP32 fma chain over k ascending per output (bf16 after the bias),
  RMSNorm a wave-order sum with a correctly rounded root, attention a fused online softmax over 64-key tiles with a
  declared exp, the 2D RoPE tables built on the host in double precision (libc pow / cos / sin), erf-GELU the fdlibm
  approximations. Gate: `tf-dsv41-vision-check` (dsv41-hip-gpu-test) against engineLab's `encode_reference` dumps
  (`tools/vision-golden` in the deployment repo): every step (RoPE tables, patch embed, all 32 blocks, final norm,
  aligner rows) and the span byte for byte, and picture.zig's patches against the reference's prep. Fidelity (not
  bitwise; CUDA's vit.zig bytes differ too), mean row cosine of the aligner rows on six pictures (2026-11-10): 0.9978 to
  0.9997 against the release's FP32 torch tower, 0.9993 to 0.9998 against its BF16 CUDA run (the release's own BF16
  run is 0.9977 to 0.9998 against FP32); the release's prep gave the same patches on all six.
- **Assets:** rank 0 reads the 266 tower tensors (926 MiB BF16) from `TF_DS_VISION_DIR`, a one-shard sidecar made by
  `tools/dsv41-zig/kit/make_vision_sidecar.py` from any checkpoint that holds them (the LibertAI EXL3 1017 checkpoint's
  equal the release's: identity `32e3e2cc...`). Both ranks read the image routing biases from
  `DS_KIT/vision/gate_bias_vl.safetensors` (`make_bias_vl.py`; required on both ranks: a rank without them would refuse
  the spans its peer routes). Authenticate the source shards first (`verify_hf_cache.py`). serve_lib.sh checks both
  (`VISION_DIR`'s SHA256SUMS, `VISION_BIAS_SHA256`) before a text_image start.
- **Memory:** rank 0 holds the weights (926 MiB) and a workspace for the largest picture (1021 cells, 9189 patches:
  324 MiB), a 10 MiB span buffer, and both ranks 4096 image rows (40 MiB) from the open, so an image request allocates
  nothing on the device.
- **Pair:** rank 0 runs the tower before `fill_begin` and sends the span rows (`fill_images`, 32 KiB pieces); rank 1
  never runs it. A tower or decode error refuses the request before any frame; a rank-0 error between pieces leaves
  rank 1's partial rows to the next `fill_begin`; rank 1 failing to take the complete rows ends the follower (the pair
  restarts) instead of diverging; a sticky device fault (`DeviceFault`) is a broken pair. `TF_DS_TEST_FAULT`
  (`tower`, `fill_images:K`, `set_images`; once a process) injects each for the recovery tests.
- **Behaviour:** an image prompt skips the RAM prompt cache, the disk tier and the reuse hook (its tokens are
  placeholders); a 1920x1080 picture encodes in about 2.9 s on gfx1151, during which other streams' rounds wait.
  PNG decodes in Zig; other formats need Python 3 with Pillow (the runtime image's `rocm10.0-vision` tag).

## 9. Build

### 9.1 Steps (`zig/build/hip_dsv41.zig`, wired from build.zig's `.linux` branch next to the CUDA targets)

| Step | Produces | Needs |
|---|---|---|
| `native-hip` | `zig-out/native-hip/bin/tensorfold-native` (rank 0 server), `tf-dsv41-lanes` (rank 1 / lane gate), `tf-hip-kit`; `zig-out/native-hip/share/tensorfold/hip/gfx1151/dsv41/` (kit) | Zig 0.17.0, hipcc; no CUDA, NCCL, nvcc, cuBLAS |
| `dsv41-hip-kit` | the kit alone | hipcc (+ optional `-Dhip-triton-kit`) |
| `dsv41-cpu` | `zig-out/bin/tf-dsv41-check`, `tf-dsv41-rank-cache`, `tf-dsv41-engram-check`, `tf-hip-kit` | Zig only; no GPU module in their graph |
| `test-dsv41-hip-host` | host unit tests: facade conformance (both backends declare the interface), mock `libamdhip64` (ashhart's hip_mock.c pattern) for error mapping and staging, msgpack/hsaco parser against `llvm-readelf` output of committed fixtures, manifest/kit verification, Kernel param tables vs fixture metadata | Zig only |
| `dsv41-hip-gpu-test` | `tf-dsv41-hip-abi-test` (echo kernels, copies/fills/events, staging reuse, graph capture/replay/update, mapped-memory round trip) | GPU, run under `flock /var/tmp/halo-gpu.lock` |
| `dsv41-hip-no-cuda` | check step: the native-hip modules never import `zig/src/cuda` or `fabric`; `strings` of the installed binaries finds no `libcuda`, `libnccl`, `libcublas`, `libnvrtc`, `nvcc` | after `native-hip` |

Options (ashhart names, so the receipt's command shape carries over): `-Dhip-arch` (default `gfx1151`, must be an exact
caps-table name, no feature suffix), `-Dhipcc` (default `/usr/bin/hipcc`), `-Dhip-include` (default
`<rocm>/include` = `/opt/rocm/include`), `-Dhip-device-lib` (default `<rocm>/lib/llvm/amdgcn/bitcode`; node1 also has
`/opt/rocm/amdgcn/bitcode`), plus `-Dhip-triton-kit=<dir>`. The CUDA options and steps are untouched.

HIP compile flags (decision): `--offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output -O3 -std=c++20
-fno-gpu-rdc -mno-wavefrontsize64 -ffp-contract=off -D__HIP_PLATFORM_AMD__ --rocm-path=<rocm>
--rocm-device-lib-path=<bitcode> -I zig/kernels/hip`. No `-ffast-math`, no `-fgpu-flush-denormals-to-zero`; a kernel
that wants contraction or other relaxed math declares it in `kernels.zon` with a numerics.json key, and the build adds
the flag to that object only. The flags and the compiler version text are recorded in the manifest.

Example (node1):
```
zig build native-hip dsv41-cpu -Doptimize=ReleaseFast -Dhip-arch=gfx1151 -Dhipcc=/usr/bin/hipcc \
  -Dhip-include=/opt/rocm/include -Dhip-device-lib=/opt/rocm/lib/llvm/amdgcn/bitcode \
  [-Dhip-triton-kit=$HOME/tfp/tri/kit/gfx1151]
```

Per-worktree caches (`ZIG_LOCAL_CACHE_DIR`, `ZIG_GLOBAL_CACHE_DIR` under the worktree) so parallel workers never share
build state.

### 9.2 Toolchain facts checked on node1 (2026-10-18)

- `/usr/bin/hipcc` → HIP 7.15.26333-0000000, AMD clang 23.0.0git (8f497e0992fb), `InstalledDir /opt/rocm/core-10.0/lib/llvm/bin`;
  `llvm-readelf`, `clang-offload-bundler`, `ld.lld` present there.
- `libamdhip64.so.7` exports every symbol ashhart's `abi.Api` needs plus `hipStreamWaitValue32/WriteValue32`,
  `hipLaunchHostFunc`, `hipGraphAddHostNode`, `hipStreamBeginCaptureToGraph`, `hipExtModuleLaunchKernel`,
  `hipEventRecordWithFlags`.
- HIP 7.15 header values equal ashhart's hand-declared ones: device attributes integrated 16, clock 5, cooperative 10,
  L2 19, max threads 56, CUs 63, LDS/block 74, optin 75, warp 87, LDS/CU 10002 (+ wall clock rate 10017);
  `hipKernelNodeParams` 64 bytes with block 0, extra 16, func 24, grid 32, params 48, shared 56; hipHostMalloc
  portable 1, mapped 2, write-combined 4, coherent 0x40000000, non-coherent 0x80000000; hipHostRegister portable 1,
  mapped 2; capture modes 0/1/2; `hipGraphExecUpdate(exec, graph, *node, *result)`; error codes NotReady 600,
  IllegalAddress 700, LaunchFailure 719, OOM 2. `hip_abi_check.c` turns each of these into a `_Static_assert` so a ROCm
  upgrade that changes one fails the build instead of the run.
- `libhipblas.so.3` exports hipblasCreate/Destroy/SetStream/SetWorkspace/SetAtomicsMode/Sgemm/GemmEx/SetMathMode.
- `import triton` fails on the system Python (w-tri pins its own AMD Triton for the build-time AOT step only).
- Zig is not installed yet (w-src is installing 0.17.0 to `~/opt/zig`).

### 9.3 CPU-only targets

`zig/src/families/dsv41/cpu.zig` re-exports config, checkpoint, rank_cache, plan, engram, engram_io and engram_aio, which
import only `std`, `builtin` and `core` (whose module root pulls safetensors/checkpoint/direct_io/draft_depth/
copy_index/tokenizer/ids_json: no GPU code). `check_main.zig`, `rank_cache_main.zig` and `engram_check_main.zig` only
reference those declarations, so they build on this root with no `gpu`, `cuda` or `fabric` module. The token map and
RoPE/Engram-constant generators that replace `node_inputs.sh`'s Python/container fallbacks belong to w-src and join this
step when they exist.

## 10. Kernels reachable in the text configuration, by owner

Full table with CUDA origin and Zig launch sites: `KERNELS.md` / `kernels.json` (81 required ids).

| Owner | Ids | Detail |
|---|---:|---|
| w-tri | 51 | 39 Triton entry kernels (196 of the 204 recorded variants): mHC 11 (`_embed_init`, `_hc_partial`, `_hc_finish`, `_hc_finish_s`, `_hc_sinkhorn`, `_hc_mix_part`, `_hc_mix_pf`, `_hc_post`, `_hc_post_only`, `_collapse`, `_collapse_norm`), norm/rope 5 (`_rmsnorm`, `_rmsnorm_rot`, `_q_kv_norm`, `_kv_norm_rope`, `_rope_heads`), router/small GEMM 5 (`_rowmm2`, `_rowmm2r`, `_rowmm_parts`, `_route`, `_route_parts`), indexer 7 (`_index_score`, `_index_score_rows`, `_index_score_cand`, `_score_keys`, `_gather_tiles`, `_topk_sel`, `_topk_finish`), attention 3 (`_sparse_attn_part`, `_sparse_attn_merge`, `_sparse_attn_merge_rot2`), cache 2 (`_fp4_store`, `_fp4qd_p2`), Engram 1 (`_engram_gate`), draft 1 (`_tap`), Markov 4 (`_bias`, `_score`, `_finish`, `_pick`); 12 native kernels: indexer/pool 7 from dsv41_ops.cu (`topk_i64`, `topk_indices`, `cand_fast`, `iota_vis`, `block_max`, `pool_pick`, `apply_pool`) and 5 from dsv41_torch.cu (`rms_norm`, `compress2`, `rope`, `bf16_scale`, `hc_mean4`); numerics owner for the hipBLAS fp32 GEMMs (router logits, indexer weights, draft gate, confidence) |
| w-exl3 | 15 | 14 EXL3 ids with 29 checkpoint-required specializations (dense K {4,5,6,8} → K2 {8,10,12,16}; routed experts K {2,3} → K2 range (2,10); MTP experts K 4 → (8,8)): `rot_many`, `glinear` ×8 (K2 × warps 4/8), `rot_in`, `unpack` ×4, experts glue 7 (`group_count`, `group_place`, `work_list`, `rot_in<bf16>`, `gateup_epilogue`, `down_combine`, `decode_prep<bf16>`), `grouped_mma` ×2, `grouped_rows` ×2, `grouped_cp` ×4 (2 ranges × 2 epilogues); plus the Triton `_gemm` (exl3/prefill.py, launched from tri_markov.zig:161, 8 recorded variants) |
| w-arch | 11 | glue 9: `bf16_to_f32` (pointwise.cu), `strided_copy`, `gather_rows` (movement.cu), `f16_to_f32`, `add2_bf16`, `scatter_rows`, `copy_rows16`, `argmax_rows`, `clock` (profile only) from dsv41_ops.cu; hipBLAS binding for `sgemm_xwT` and `sgemm_xv`; ABI echo kernels |
| w-link | 4 | bootstrap, prompt all-gather (3 sites), decode all-gather (7 sites), device-proxy stage/collect pair for graphs (replacing rdma_gather.cu's 3 kernels) |

Not required, with the reason (each refused or unreachable, none stubbed): Triton `_hc_finish_u` (hidden 5120 is a
multiple of 1024, tri_hc.zig:137), `_sparse_attn_merge_rot` (head_dim 512 is a power of two, tri_attn.zig:122), `_rowmm`
(never recorded fallback, tri_norm.zig:97); `tf_f32_to_bf16_kernel` (loaded by ops.zig:65 but never called: the HIP loader
does not require it); `tf_ds_l2_prefetch_kernel` (NVIDIA L2 only, `TF_DS_L2_PREFETCH=1` refused on HIP); 8 vision kernels
in dsv41_ops.cu, the fmha cubin and cuBLASLt (text capability); rdma_gather (verbs); exl3_experts2d (world 4); sample.cu
(the DeepSeek sampler runs on the host, sampling.zig). Shape-conditional kernels are checked at model open: if this
checkpoint's config ever reaches an unreached kernel, open fails naming it.

Cross-owner dependencies to watch: EXL3 prefill uses w-exl3's `_gemm` through `tri_markov.matmul`; the routed-expert
path consumes w-tri's router outputs (`_route*`) and w-arch's `scatter_rows`/`gather_rows`; every mix after a gather
(`_hc_post`, `add2_bf16`) depends on w-link's rank order.

## 11. Implementation plan (w-arch, after review)

Ordered commits in `~/tfp/wt/arch` (branch `arch` from `strix-port`), each with reason, behaviour change and validation:

1. `tools/dsv41-hip/`: this document, the two inventory scripts and their outputs (no code change).
2. `zig/src/gpu/hip/` backport (abi, runtime, context, memory, stream, module, args, launch, graph, graph_split, caps,
   code_object, device_arch.c) + `hip_abi_check.c` + error mapping; host tests with the mock library. Validation: `zig build
   test-dsv41-hip-host` on node1.
3. `gpu` facade (both backends) + staging ring + conformance test; GPU smoke test (copies, fills, events, staging reuse,
   graph capture/replay/update, mapped round trip). Validation: `dsv41-hip-gpu-test` under the lock, receipt JSON.
4. `hsaco.zig` + `msgpack.zig` + `tf-hip-kit scan/merge/verify/coverage` + `Kit` + `gpu.Kernel` + `TritonSet`; fixtures
   compiled on node1 and compared against `llvm-readelf --notes`. Validation: host tests + the echo test.
5. Family mechanical pass: `cuda` → `gpu`, the 19 driver-table sites, `uploadStaged`, `Modalities`, PDL from caps,
   vision/RDMA/world-4 gating, `Coll` interface + NCCL wrapper, typed glue kernels; `dsv41_cpu` root. Validation: CUDA
   facade compiles host-only (`zig build test` CUDA host tests), HIP family compiles against a stub kit.
6. Glue kernels in HIP (9) with CPU reference tests (exact gates: casts, copies, gather/scatter, argmax with first-index
   ties, add2 rank order), hipBLAS binding with an FP64 reference test of the four real shapes.
7. `zig/build/hip_dsv41.zig` steps incl. `dsv41-hip-no-cuda`; `native/gpu.zig` engine host for HIP.
8. Integration hooks for the other workstreams: `kernels.zon` entries and Zig `Kernel` declarations as their kernels land.

Estimate: about 3.5k lines including tests (adapter ~1.3k, kit/ABI tooling ~0.9k, family edits ~0.4k changed lines,
build ~0.35k, glue HIP ~0.3k), 5-7 agent-days. Steps 2-4 do not need the strix-port branch or Zig on node1 to be
designed, but do need Zig 0.17.0 to build.

## 12. Open questions for the integration owner

1. D15: keep the CUDA facade compiling host-only (cost ≈ 1 day, catches non-mechanical family edits), or let the family
   become HIP-only on the branch? Recommendation: keep it until first light, then decide.
2. `Coll` interface (section 5) needs w-link's agreement before step 5; who arbitrates the handshake field list (w-link
   owns the handshake, w-arch supplies ABI version and kit digest)?
3. w-tri's choice per Triton kernel (AMD Triton AOT vs hand-written HIP) changes only the manifest `abi` field, but w-tri
   must pin its Triton build (AMD nightly 3.8.0 per Kyojin) and produce per-variant metadata in the format of section 7.2.
4. w-exl3 will likely reshape the EXL3 argument structs (Kyojin layouts); the ABI rules allow it, but exl3_linear/prefill/
   experts.zig wrappers then change beyond a mechanical port. Confirm that is within w-exl3's slice.
5. Default `TF_GPU_STRICT`: on for every non-benchmark run (cheap checks only), required for `qualify.sh`; off for the
   frozen benchmark only if measured overhead is non-zero.
6. Is `hipMalloc` on gfx1151 the right allocator for ~99 GiB of weights (coarse-grained GTT) versus
   `hipExtMallocWithFlags`? The adapter exposes both; w-link's G-c measurement should decide before the first full load.
7. Graphs: D12 makes the device-proxy transport the graph path. If w-link's visibility gate fails for coherent host
   memory, graphs fall back to `graph_split` and the per-round host cost has to be measured before P5.

## 13. Review decisions (integration owner, 2026-10-19)

1. D15: keep the CUDA facade compiling host-only until first light, capped at about one day of work; if it costs more,
   drop it and report.
2. `Coll` approved as written. w-link owns the handshake and the TCP implementations; w-arch supplies the ABI version, the
   kit manifest SHA-256 and `Modalities`; the integration owner arbitrates the final handshake field list. `tcp_host` may
   stripe large prefill all-gathers across both USB4 cables (no interface change).
3. w-tri pins its AMD Triton build and emits per-variant metadata in the section 7.2 format.
4. The EXL3 wrappers (exl3_linear/prefill/experts.zig) are w-exl3's, under these ABI rules (extern struct params,
   manifest entries, echo test).
5. `TF_GPU_STRICT` is on by default for every non-benchmark run and required by qualify.sh; the frozen benchmark turns it
   off only if its overhead is measured non-zero.
6. Allocator: wait for w-link's G-c measurement; expose both `hipMalloc` and `hipExtMallocWithFlags`.
7. Graph fallback agreed.

## 14. Implementation notes (plan steps 1-4, branch `arch`)

Where the code differs from sections 3-9, and what the GPU runs taught:

- The `gpu` surface keeps the CUDA runtime's names exactly (`Driver`, not `Runtime`; `DeviceBuffer.ptr: u64`;
  `Stream.init(d, non_blocking)`; `Config{ .pdl, .cluster, .cooperative }`), so the family changes only its import and the
  19 direct driver-table calls. `zig/src/gpu/interface.zig` lists that surface (declarations, 63 methods with parameter
  counts, fields) and checks both facades at compile time.
- Module roots: `zig/src/gpu/hip_facade.zig` (HIP builds) and `zig/src/gpu/cuda_facade.zig` (CUDA builds, over the
  untouched `zig/src/cuda`), both in `zig/src/gpu/` because Zig modules cannot import outside their root directory.
- Blocking calls are complete on return. HIP's `hipMemsetD8/D32` and `hipMemcpyDtoD` may return with the work queued on
  the null stream, which a non-blocking engine stream does not wait for: the first smoke run saw a "blocking" fill land
  after a later kernel. The HIP `DeviceBuffer.fill8/fill32/copyFrom(null stream)` now synchronize the null stream.
  The family's `cuCtxSynchronize` after setup (model.zig:303) stays meaningful on HIP.
- The kernel registry is JSON (`zig/kernels/hip/dsv41/registry.json`), not ZON: the kit tool, Python tooling (w-tri's
  Triton AOT step) and reviewers read it without Zig. Entries: id, symbol, spec, block, lds_dynamic_max, numerics,
  allow_scratch, feature (always/drafts/engram/profile/graphs), optional `triton` metadata.
- ABI kinds in the manifest: `params` (one C parameter per Zig field; a by-value struct is one parameter) and `triton`.
  The by-value structs and `TF_DS_HIP_ABI_VERSION` come from `zig/kernels/hip/dsv41/abi.h` through the build's
  translate-C step, so Zig and hipcc share one definition; the header carries a size + every-offset assertion per struct.
- Strict mode is `launch.before` (a hook in the HIP launch path) set by `Kit.open`; one kit per process. It checks
  each launch's packed offsets/sizes against the code object's explicit arguments, the block against the declared block
  and `max_flat_workgroup_size`, and dynamic LDS against `lds_dynamic_max`, and counts launches per entry. A launch of a
  kernel outside the manifest fails. `TF_GPU_STRICT=0` turns it off (benchmark only, per section 13 item 5).
- Code objects: `hipcc --offload-arch=gfx1151 --offload-device-only --no-gpu-bundle-output` emits the raw ELF (hipcc
  prints a harmless "argument unused: --hip-link" warning). `hsaco.zig` reads the metadata note; a host test compares it
  field by field with `llvm-readelf --notes` on the smoke and echo objects (its first run caught a bug in the test's own
  YAML reader, not in hsaco.zig).
- GPU programs: `tf-dsv41-hip-smoke` (runtime behaviour) and `tf-dsv41-hip-abi-test` (kit, typed kernels, Triton
  variant set, refusals), run by `tools/dsv41-hip/run_gpu_tests.sh` under the GPU lock; receipts in
  `~/tfp/arch/receipts/<UTC time>/` (run.json + one JSON per program).
- Kit provenance is by content, not by commit: the manifest's `inputs` list the SHA-256 of each `.hip` source, `abi.h`
  and the registry, and `toolchain.hipcc` is `hipcc --version` captured at build time (never cached; a toolchain change
  rebuilds every object). The section 7.2 example's `source.git_tree` is gone: Zig 0.17 caches the configure step, so a
  configure-time `git rev-parse` went stale, and a kit should not change identity on an unrelated commit. The GPU run
  receipt records git HEAD, tree and dirty state at run time.
- D10 revised with w-tri's AMD Triton kit (2026-10-19): the recorded variant keys keep the `PDL` constexpr (102 of 126
  variants have PDL=1) because `gdc_wait`/`gdc_launch_dependents` compile to no-ops on AMD; so `Tri.pdl` stays a
  variant key. Only the CUDA launch attribute (`Config.pdl`, `cluster`, `cooperative`) is refused on HIP, and the
  Triton set never sets it. `exl3_linear.zig`'s `.pdl` launch flag must still come from `gpu.features.pdl`.
- `tf-hip-kit scan --triton DIR` imports w-tri's kit (DIR/aot.json in the CUDA KernelJson schema plus arch, warp_size,
  block, file; DIR/hsaco/<sha256>.hsaco): each variant must be gfx1151, wave 32, block = num_warps x 32, one CTA, no
  PDL launch, zero global/profile scratch, and its object's explicit arguments must be the runtime parameters (sizes by
  type) plus two 8-byte scratch pointers. Spilling variants are allowed (`allow_scratch`) and counted: speed, not ABI.
- The registry may declare `required` kernel ids per kit and feature apart from any source, so the serving kit's
  required set (KERNELS.md) can be stated before every kernel exists, and Kit.open names whatever is missing.
- Device flags add `-fno-fast-math -fdenormal-fp-math=ieee`; configure refuses a set HIPCC_COMPILE_FLAGS_APPEND /
  HIPCC_LINK_FLAGS_APPEND (both after the component kit). `tools/dsv41-hip/zig_build.sh` runs builds under the house
  limits; the user manager here does not delegate cpuset, so it pins cores with taskset.
- `tf-hip-kit lint` (part of test-dsv41-hip-host) refuses any `__builtin_bit_cast` / `bit_cast<` in device sources
  without a `bitcast-ok:` note on the line: clang 23 / HIP 7.15 returns lane 0 for a bit cast of an f16x2 lane (w-exl3).
  The smoke test probes the bug and checks the memcpy idiom the kernels use instead.

### Step 5 (family pass) notes

- The rename is a code-only transform (`tools/dsv41-hip/rename_cuda_to_gpu.py`: never in comments or strings; idempotent):
  398 `cuda` tokens in 31 family module files, plus rank 1's `lanes_main.zig`. The CUDA-only tools (`layer_main.zig`,
  `rdma_bench_main.zig`, `ring2d_check_main.zig`, `comm_main.zig`, `vit_main.zig`, `picture_main.zig`) keep importing
  `cuda` directly; their `dsv41` module now sees the CUDA facade as `gpu`.
- CUDA-only code is removed from HIP builds at compile time, not stubbed: `gpu.features.{world4, verbs_rdma, vision,
  l2_prefetch}` gate it, and `prompt.two2d(e)` (an inline function returning comptime null on HIP) replaces the 27
  `e.two` uses, so every four-node branch drops out. A HIP model refuses world 4, TF_RDMA_DEVICES, TF_DS_L2_PREFETCH
  and image input at open with named errors (native.zig explain texts).
- `Fan` and `split` moved from prompt2d.zig to fan.zig (re-exported there); NCCL's bootstrap is `comm.Comm.fromFan`,
  and `Comm.coll()` is the NCCL `Coll`. `Engine.comm` is a `*const coll.Coll`; `model.Options.open_coll` takes the HIP
  transport (w-link), and without one a HIP model refuses at open (`error.NoCollective`). `TF_DS_GRAPHS=1` is refused
  when the collective is not capturable.
- `model.Modalities` (`text`, `text_image`), `TF_DS_MODALITIES`, `tf-dsv41-lanes --modalities`: default `text` on HIP.
- The kernel sets on HIP come from `hip_load.zig` (only HIP builds reference it, so the CUDA test build's refAll never
  sees `gpu.Kit`): ops/exact by kit id (section 6.1's `glue.*` and `aux.*` since step 6), the EXL3 sets refuse with
  `error.Exl3KernelsNotIntegrated` until w-exl3's loaders replace them. Kit-owned sets do not unload the kit's modules.
- `blas.zig` selects cuBLAS or hipBLAS (`hipblas.zig`: no atomics, host scalars, a fixed 32 MiB workspace from the
  arena; values and prototypes checked by `hipblas_abi_check.c`; hipBLAS's OP_N/OP_T are 111/112, not cuBLAS's 0/1).
- `zig/build/cuda.zig` changed in two places only: the dsv41 module and `tf-dsv41-lanes` import the CUDA facade as
  `gpu` (D15's "untouched" applies to the CUDA runtime and kernels).
- `zig build dsv41-cpu` builds tf-dsv41-check, tf-dsv41-rank-cache and tf-dsv41-engram-check from `cpu.zig` with no
  GPU module, into `zig-out/dsv41-cpu/bin`.

### Step 6 (glue kernels, hipBLAS) notes

- `zig/kernels/hip/dsv41/glue.hip` holds w-arch's nine glue kernels as `tf_ds_glue_<op>` (ids `glue.<op>`, section
  6.1): `bf16_to_f32`, `f16_to_f32`, `add2_bf16`, `scatter_rows`, `gather_rows`, `strided_copy`, `copy_rows16`,
  `argmax_rows`, and `clock` (feature `profile`). The CUDA bodies are kept (sources and their SHA-256 in the file
  header). KERNELS.md's inventory ids map as `native.tf_bf16_to_f32` -> `glue.bf16_to_f32`, `native.tf_ds_<op>` ->
  `glue.<op>` (w-arch) or `aux.<op>` (w-tri's aux.hip); `native.tf_f32_to_bf16` (never launched) and
  `native.tf_ds_l2_prefetch` (NVIDIA only) have no HIP kernel.
- `hip_load.zig` binds every helper with a typed parameter list (`hip_load.params`), so a compiled kernel whose
  arguments differ from what ops.zig / exact.zig pack refuses the model's open (AbiMismatch, by id). A host test runs
  every launcher (16 ops paths, 5 exact) through `launch.before` and checks the packed offsets and sizes against those
  lists, so the declarations and the hand-written packing cannot drift apart.
- The clock: CUDA reads `%globaltimer` (ns). gfx1151 has `wall_clock64()`, a constant-rate counter that the runtime
  reports as 99,810 kHz (hipDeviceAttributeWallClockRate, whole kHz). `glue.clock` takes one more parameter than the
  CUDA kernel, `ns_q32` (ns a tick, 32.32 fixed point, from the caps: `hip_load.ops(kit, caps.wall_clock_hz)`), and
  writes nanoseconds, so the profile's readout (model.zig) is unchanged. On the GPU, 2 s of host time read within the
  launch bracket and a few ppm of it; reading the same ticks as 10 ns each (a round 100 MHz) falls outside it.
- bf16 NaN rule: `.to(bf16)` in glue.hip and aux.hip is clang's float -> `__bf16` conversion. On gfx1151 a NaN keeps
  its sign and high payload bits and gets the quiet bit (`(bits >> 16) | 0x40`, measured on every NaN sum in the test).
  CUDA's `cvt.rn.bf16.f32` gives a canonical NaN instead (PTX ISA; not measured here: there is no NVIDIA GPU on these
  hosts). Finite and infinite values round identically (to nearest even). A NaN in the served path is a fault either
  way; the numerical contract is "NaN stays NaN".
- The dsv41 kit's native half (glue.hip + aux.hip, 2 objects, 21 kernels) is built and `tf-hip-kit verify`-ed by
  `test-dsv41-hip-host`; `dsv41-hip-gpu-test` installs it as `zig-out/native-hip/test/dsv41-kit` with
  `tf-dsv41-hip-glue-test` (ReleaseSafe, for the FP64 reference GEMMs). The kit's `required` lists hold only what its
  sources declare: the Triton ids join when the serve kit is assembled with `--triton` (plan step 7), the EXL3 ids
  with w-exl3's kernels.
- hipBLAS in the family's four call shapes (indexer weights k 5120 n 32, MoE gate n 384, draft gate n 128 at 17 to
  2,048 rows; the confidence head's xv at k 5,376) against FP64: worst error 7.3 u·Σ|x·w| against the γ_k bound of
  ~5,120 u·Σ|x·w|; that is fp32 accumulation (inputs rounded to a TF32-style 10-bit mantissa would each be off by up
  to 2^-11 = 8,192 u), and reruns are bit-identical (atomics off).

### Step 7 (native-hip) notes

- `zig build native-hip` installs into `zig-out/native-hip`: `bin/tensorfold-native` (rank 0: the server with
  `zig/src/native/hip.zig` as its engine host), `bin/tf-dsv41-lanes` (rank 1), `bin/tf-hip-kit`, and the dsv41 kit at
  `share/tensorfold/hip/<arch>/dsv41` (where the host looks unless `TENSORFOLD_HIP_KERNELS` names another).
  `-Dhip-triton-kit=DIR` folds w-tri's AOT Triton kit into it; `zig build dsv41-hip-kit` makes the kit alone.
  `dsv41-hip-no-cuda` (part of native-hip) checks both binaries for NVIDIA library and entry-point names.
- native/hip.zig follows native/cuda.zig: device by `TF_HIP_DEVICE` / `--device`, the device made current on the lane
  thread (hipSetDevice is per thread), the same admission arithmetic (cuda_memory.zig reads no device): an integrated
  GPU counts /proc/meminfo's MemAvailable less the reserve, and `TENSORFOLD_HIP_MEMORY_LIMIT_GB` defaults to 110
  (w-link's allocation gate). A family's refusal carries its explain text. It registers dsv41 alone.
- The serve kit requires, besides what its sources declare, the registry's `required.dsv41` lists (generated from
  kernels.json: 34 Triton + 14 EXL3 ids always, 5 Triton with drafts, 1 with Engram). Test kits are scanned with
  `tf-hip-kit scan --required declared`. With w-tri's kit folded in the serve kit holds 120 objects and 147 kernels;
  `tf-hip-kit verify` names the 14 EXL3 ids as its only gap.
- On node1 (GPU lock held), tensorfold-native serve on the 2.9 bpw checkpoint went through the runtime, the device's
  caps, the kit path, admission and the family's open, loaded and checked all 120 code objects, and refused with the
  14 missing EXL3 ids and the MissingKernel explain text; the weights were never read (the kit opens first).
- On HIP the family passes the host's kit to the model (`TF_DS_HIP_KIT` still overrides it); `TF_DS_KIT` keeps the
  RoPE tables and engram.json.

### Layer grading and the launch tooling (after step 7)

- `tf-dsv41-layer` (layer_main.zig) runs on both backends. On HIP its kernels come from the kit (`--hip-kit`, opened
  before the weights; every EXL3 specialization the weights reach checked), REC_DIR holds the fixtures, and `--kit-dir`
  the RoPE tables and engram.json. `native-hip` installs it. Fixture files are read from REC/rank<R>/layers/ or
  REC/layers/ (w-ref's layout); a fixture line's own `key` is kept.
- For w-ref's FP64 fixtures: `--anchor 1` writes Ref.layer_in h/pre over the layer's input (prompt chunks; decode rounds
  through round.Probe.anchor) and drops the pending post; a prompt layer's output is then posted in place and checked
  against Ref.layer_out. `--dump DIR` writes every checked point in the fixtures' schema (DIR/rank<R>/layers.jsonl,
  DIR/layers/r<R>.<key>.bin), which grade.py reads. `--keep-going 1` reports a difference and goes on.
  `--replay-gather 1` replaces the transport by the fixtures' Comm.gather outputs (one rank graded alone, no links).
  Ref.route pick/wts are checked as the first top_k slots of the routing.
- The bounded replay against an uncut reference: from the decoder's first layer (20) the served prefill keeps only rows
  >= prompt - window (128). A fixture that holds the whole chunk there (w-ref's: 160 rows) is uploaded, checked and
  dumped on the engine's rows only: its rows dimension (a gather's [world, rows, ..] each rank's part) from row_off =
  the cut's first row on. Each dump line carries `row_start` (0 when nothing was cut) and the engine's shape. At layer 20
  `--anchor` writes the whole chunk before the cut, since the cut's KV-source update reads every row. A tap keeps the
  rows of the layer it was taken at, and the head's points (one row) are never cut. `--no-replay-cut 1` runs every row
  through every layer instead (replay 0; the transport's plan names the mode, so both ranks must agree).
- `--dump` also writes the router's input, which no fixture records, under `STEP:LAYER:route_logits` with `equal: null`
  (prompt chunks and decode rounds; not the drafter's stages): fp32 [n, kc, 384] rowmm_gate chunk sums when n <= 16
  (kc 20), else fp32 [n, 384] logits (hipBLAS; rowmm2's fallback writes [n, 384] too). prompt.Chunk.route_in / route_kc
  name what tri_norm.route read; a round's probe sees it as round.Point.route_in (kc from tri_norm.gateParts). Ref.route
  pick/wts are checked on rounds too (round.Point.route_pick / route_wts), when the fixtures record that round's.
  An index layer's selection is dumped the same way, `STEP:LAYER:Ref.indexer.cidx`: int64 [n, kk], the compressed
  entries each row attends (kk the selection's width; -1 past a row's visible entries; at short contexts every visible
  entry, 0 .. nb-1). A round hands it to the harness through round.Probe.put (dump-only buffers with the engine's shape).
- Serving (handoff 10): `serve.conf` is the one configuration; `serve_lib.sh` resolves it per rank into an argv and a
  whole environment (`env -i`: nothing else leaks in) and writes resolved.json with identities (deploy commit, binary,
  kit manifest, config, rank cache file). `serve-rank1.sh` (node1) and `serve-rank0.sh` (node0) start, check, lock,
  wait, probe and stop their rank; `first_light.sh` (on node1) brings both up in order (identities compared; GPU locks
  node0 then node1; rank 1, then rank 0; ready = /health and a probe chat answered; rank 1 linked) and down again.
  Binaries reach node0 (and node1's own deployment) through `deploy_rank0.sh --native zig-out/native-hip`, from a clean
  commit, SHA-256 checked on arrival.


### w-tri's HIP kernels in place of Triton launches (tri_aux.zig)

- Three of w-tri's kernels take Triton launches inside the family's Triton wrappers, so every call site is unchanged.
  `Tri.aux` holds the bound kernels. When it is null, or when a kernel does not take the shape, the Triton launch runs.
  - `aux.sparse_attn` replaces `_sparse_attn_part` (tri_attn.sparseAttn).
    - Prompt chunks run in one pass straight into `out`. Decode and verify windows write the 8-split partials, and the
      Triton merges (`_sparse_attn_merge`, `_rot`, `_rot2`) read them unchanged.
    - It gets the picks list and n_idx exactly as the Triton launch would. The kernel clamps to 512 picks, so it is
      refused for n_idx > 512, h ≠ 32, hd ≠ 512 or window ≠ 128.
  - `aux.index_rows` replaces the indexer's scores without a base or candidates (tri_index.indexScore):
    `_index_score_rows`, and `_index_score` for prompt chunks of ≤ 16 rows.
    - Every round passes a per-row compressed base, so the rounds' scoring and keys stay on Triton. A stream's decode
      and verify rows always use one kernel.
    - bf16 keys and other head shapes stay on Triton.
  - `aux.hc_mix_pf` replaces `_hc_mix_pf` (tri_hc.hcMixPf) at every row count. A token's bits never depend on the size
    of the chunk it landed in, which chunked prefill and kept prefixes need. The integration owner chose this over
    w-tri's 16/256-row crossover; the cost is at most 38 µs a call.
    - The PART slots are the Triton layout, so `_hc_finish*` is unchanged.
    - d ≠ 5120 or world > 2 stays on Triton.
- `TF_DS_AUX`: `hip` (the default), `triton`, or a comma list of `attn`, `index`, `hc` (the ones on HIP). The server
  (model.open) and tf-dsv41-layer both read it.
  - It appears in the collectives handshake's plan as `aux=<name>`, so both ranks must agree.
  - The layer gate prints `{"aux": ...}` and grades either path, which gives the per-kernel A/B.
- The by-value structs are abi.h's (`tf_ds_aux_attn_args`, `tf_ds_aux_index_args`, `tf_ds_aux_hc_mix_args`); hip_load
  checks their layout at compile time.
- The three ids are in `required.dsv41.always`, so a kit without them is refused at open.
- Host tests:
  - each wrapper's choice is checked by shape and mode (the Tri log records the HIP structs);
  - hip_load's packing test launches all three through the wrappers and checks the bytes against their parameter lists.

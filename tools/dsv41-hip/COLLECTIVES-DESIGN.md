# dsv41 TP2 collectives over USB4: design

Owner: w-link (l-link since 2026-10-19 10:30). Status: draft 6 (2026-10-19 12:00 PDT). The `Coll` interface changes of
section 2.1, `tcp_host`, `tcp_proxy` and the pipelined large gathers are on `strix-port` (merged through 297a98f). Branch
`link`, rebased onto strix-port aa0e5cf, adds the split all-gather (`gatherBegin`/`gatherEnd`, section 3.5) and the
prefill stress / overlap checks (sections 4, 5.3).
Link MTU 1500 is permanent on both boxes since 2026-10-19 (section 4; closed). Numbers are
measured on node0/node1 unless marked *estimate* or *inferred*; anything still being measured is marked PENDING.

Inputs:
- Baseline tree 1beea238 (commit 7f25031), `zig/src/families/dsv41/`: comm.zig, model.zig, prompt.zig, prompt2d.zig
  (`Fan`), link.zig, rdma.zig, rdma_seq.zig, round.zig, draft.zig, lanes.zig, plan.zig, and
  `zig/kernels/cuda/dsv41/rdma_gather.cu`. Line numbers below refer to that tree.
- `Coll` interface, tf-port/w-arch/ADAPTER.md section 5 (approved), and my proposed changes (section 2.1).
- Kit `~/tfp/kit/tensorfold-strix` def87879: `tools/dsv41-hip/transport/` (tf_tp2 TCP transport + HIP bounce bridge,
  tp2_hip_check, verify_pair.py), `zig/src/families/dsv41/hip_comm.zig`.
- ds4 f46631e `rocm/ds4_rocm_tp.cuh` (host-coherent gate design), `ds4_tp.c`, `ds4_tp_io.h`.
- My Phase 0 measurements (G-b/G-c): tcplat (link only), gpuvis (GPU -> coherent slab -> TCP -> GPU), memgate,
  allocgate, reliability sessions. Sources and raw data: node0 `~/tfp/link/`.

## 1. What has to be carried (TP2, world 2)

Every model reduction is an fp32 all-gather on the single model stream, followed by a rank-ordered add on each rank
(comm.zig:1-3). In TP2 all of them go through `Engine.gatherF32` (decode, drafter; prompt.zig:131-134) or
`e.comm.allGather` (prefill; prompt.zig:1012, 1232, 1348). `Comm.exchange` is used only by the four-node 2D path
(prompt2d.zig:235, 237). The RDMA QP-info gathers at model.zig:400/406 exist only for the RDMA ring and disappear.
hidden D = 5120 (config.zig:263); Engram width per rank (hc+1)*D = 25600 (plan.zig:78-80); head columns per rank 64640.

| Site | Phase | Bytes per rank | Captured in graphs today |
|---|---|---|---|
| round.zig:998 attention gather, each of 40 layers | decode round, R rows | R*5120*4 | yes (`round.Graphs`) |
| round.zig:1008 MoE gather, each of 40 layers | decode | R*5120*4 | yes |
| round.zig:978 Engram gather, layers 1 and 14 | decode | R*25600*4 | yes |
| round.zig:1027 head gather | decode | R*64640*4 | yes |
| draft.zig:504 / 511, 3 drafter stages x 2 | DSpark draft pass, R_d = 5 x streams | R_d*5120*4 | yes (`PassGraphs`) |
| draft.zig:108 Markov bests, once per draft step (<= 5) | draft | streams*4*4 | yes |
| prompt.zig:1012 attention + MoE gathers | prefill chunk, n <= 2048 rows | n*5120*4 (40 MiB at 2048) | no |
| prompt.zig:1232 Engram gather, layers 1 and 14 | prefill | n*25600*4 (200 MiB at 2048) | no |
| prompt.zig:1348 head gather | last prefill chunk | 64640*4 | no |

Per round counts: 83 target gathers (80 + 2 + 1) plus 6 drafter + up to 5 Markov gathers = 94 with DSpark (one stream,
5 drafts, R = 6). Prefill uses a fixed 2048-row chunk (model.zig:32, not configurable at this tree); with bounded replay
an ordinary chunk runs only layers 0-19 (20 x 2 hidden gathers + 2 Engram gathers, 2.0 GiB per rank per direction), and
only the last <= 128 rows go through layers 20-39 and the head (model.zig:589-601, prompt.zig:1293-1314).

Consumers and the layout contract: recv is `[rank0 | rank1]`, each block `count` elements. `add2_bf16`
(dsv41_ops.cu:16-20) computes `bf16_rn(f32(g0) + f32(g1))` with g0 = rank 0's block; the hc kernels (`_hc_mix_part`,
`_hc_post`, `_hc_post_only`, `_hc_mix_pf`) read G = [world, rows, D] with stride RS = rows*D; decode logits copy the two
head blocks into vocabulary order (round.zig:1029); prompt logits use head_g directly as the 129280-wide row
(model.zig:660-663). The transport never adds, converts or reorders: it moves bytes into that layout. The FP32
gather-then-rank-ordered-add contract therefore stays exactly as in the reference (handoff 6D).

Today nothing has a timeout, `Ring.abort` has no caller, and the ranks exchange no identity (the only cross-rank checks
are the NCCL id length, the ring Info sanity and the control-frame step counter). Known hang paths: a peer dying during a
collective; a rank-local failure part-way through a primitive, after which collectives pair up out of order; ranks
started with different TF_DS settings. The design below turns each of these into a bounded, attributed error.

## 2. The collective abstraction

### 2.1 Interface (w-arch's `Coll`, with my changes)

```zig
// zig/src/families/dsv41/coll.zig on strix-port (w-arch), with Open taking the context on branch `link`
pub const Site = struct { phase: Phase, layer: u16 };   // Phase: round_engram, round_attn, round_moe, round_head,
                                                         // prompt_engram, prompt_attn, prompt_moe, prompt_head,
                                                         // draft_attn, draft_moe, draft_markov, ..., test_only
pub const Error = error{ CollectiveFailed, PeerLost, Timeout, Mismatch, Aborted };
pub const Coll = struct {
    ptr: *anyopaque, vtable: *const VTable, rank: u32, world: u32, capturable: bool,
    pub const VTable = struct {
        all_gather: *const fn (*anyopaque, stream: gpu.Stream, site: Site, send: u64, recv: u64, bytes: usize) Error!void,
        exchange: ?*const fn (...) Error!void = null,           // world-4 split only (NCCL)
        abort: *const fn (*anyopaque, reason: Reason) void,    // any thread, idempotent
        close: *const fn (*anyopaque) void,                    // bounded; aborts a call in flight
    };
};
pub const Kit = if (gpu.backend == .hip) gpu.Kit else void;
pub const Open = *const fn (gpa, io, ctx: *const gpu.Context, kit: ?*const Kit, fan: *const Fan,
                            id: *const Identity) anyerror!Coll;
```

Changes against ADAPTER.md section 5, each with its reason:
1. `site` argument. It is written into the wire header; both ranks must present the same (sequence, site, bytes) or the
   call fails with `Mismatch`. This converts the silent "collectives pair out of order" failures into an immediate,
   attributable error at the first divergent call. The family already has the information (round.zig `Point`/`Phase`,
   draft.zig `Point`).
2. `Mismatch` and `Aborted` errors (a `status` query was proposed but not adopted; not needed). Semantics are terminal: the first failure poisons the Coll; every
   later call returns that error without touching the network; a fresh Coll (new session epoch) is the only recovery.
   This matches the kit's terminal-session contract.
3. Stream semantics are unchanged: `send` is read after earlier work on `stream`; later work on `stream` sees `recv`.
   For `tcp_host` the call returns after the H2D of the peer's block (and the D2D of the own block) is enqueued on
   `stream`; for `tcp_proxy` it returns immediately after enqueueing kernels.
4. In-place use is allowed: if `send == recv + rank*bytes` the own block is not copied.

State on `strix-port` (coll.zig, w-arch, agreed 2026-10-19): `all_gather(stream, site, send, recv, bytes)`, the
`Phase` enum and `Site{phase, layer}`, `Error{CollectiveFailed, PeerLost, Timeout, Mismatch, Aborted}` with the terminal
contract, and `Identity{rank, world, kit_abi, kit_sha256, modalities, plan}` handed to `Open`. `Open` also receives
the `gpu.Context` (the transport needs the driver for its copies and events) and, on branch `link` (a5aed15, w-arch's
review), the typed kit: `tcp_proxy` loads `coll.stage`/`coll.collect` from it and refuses a null kit
(`error.ProxyNeedsKit`); `tcp_host` ignores it. No `status` call was
added; the first error is returned by every later call instead.

No other operation is needed in TP2: no reduce, no broadcast, no point-to-point. Control traffic (prompt ids, chunk
starts, pass tokens, snapshot ops) stays on the existing `Fan` control link (lanes.zig:40-95).

### 2.2 Ordering and numerics
- Calls on a Coll are a single FIFO. Both ranks must issue the same sequence; the 64-bit sequence number (starting at 1
  for a session, +1 per call) and the site are checked on every exchange.
- The transport moves bytes only; dtype is implicit (fp32 at every TP2 site) and recorded in the header for checking.
- The own block is written by the local rank (D2D copy, or in place), the peer block by the transport. Both ranks end
  with byte-identical recv buffers; the rank-ordered add is then bit-identical on both ranks.

### 2.3 Completion, deadlines and errors
- `tcp_host`: synchronous for the host thread. Steps: wait for the producer (event recorded on `stream`, polled with
  `hipEventQuery` in a spin loop, no sleeps), D2H of `send` into pinned TX, network exchange, H2D of the peer block into
  `recv`, D2D of the own block. (Chunked pipelining for large payloads is designed, not built: section 3.4.)
- `tcp_proxy`: asynchronous by stream order; the device waits are bounded by a deadline in `wall_clock64` ticks (100 MHz
  on gfx1151), after which the wait kernel sets the abort word and returns (ds4 pattern). The proxy thread then poisons
  the Coll, so the next eager call (and the host's check at every round boundary, before results are committed)
  returns the error.
- Deadlines are progress-based, not per-operation absolutes: once the peer's first byte of an exchange has arrived, the
  exchange fails with `Timeout` when no byte has moved for `T_progress` (default 2 s; a byte-scaled hard cap is
  planned, not built). Connect and handshake: 300 s (the current link's retry window, prompt2d.zig:53). GPU waits:
  10 s. Reason: a 200 MiB prefill Engram gather legitimately takes ~60 ms on one cable, and one TCP retransmission
  timeout must not kill a decode gate (section 4).
- A late peer is not a lost one: before the peer's first byte (it has not entered the exchange yet: its own GPU wait,
  or the other stream's work ahead of it, e.g. a vision tower) the exchange waits up to `T_gpu_wait + T_progress`
  (12 s by default) from its start, not `T_progress`. The peer's own GPU wait therefore expires first and names the
  cause. A closed or reset peer still fails the exchange at once; the cost is that a peer hung with its connection
  open, or a link gone without a FIN, between exchanges is found after 10-12 s instead of 2 s. Both limits come from
  settings the handshake already compares (`progress_ms`, `gpu_wait_s`), so both ranks apply the same one.
- Errors map: peer EOF/RST or an abort frame -> `PeerLost`; deadline -> `Timeout`; header/sequence/site/epoch mismatch
  -> `Mismatch`; local `abort` -> `Aborted`; anything else (HIP error, ENOMEM) -> `CollectiveFailed`.

### 2.4 Abort and cancellation
- `abort(reason)` (built in tcp_host): sets the session state (first reason wins) and shuts down the data sockets, so
  the peer's exchange in flight or its next one fails with PeerLost (fault matrix: 20 ms after a kill). tcp_proxy adds
  the abort word in the coherent slab (device waits and the proxy thread see it within one poll) and a watchdog thread
  on the peer that sets its own abort word when its data socket dies, so a rank stuck in a GPU wait also leaves. The
  control link (`Fan`) carries the lanes protocol and gets no abort frame of ours.
- Cancellation of a request (client disconnect, stop) does not abort the Coll. It is handled at primitive boundaries on
  the control link as today (lanes.zig:477), so both ranks stay symmetric. Aborting is for transport or rank failure,
  and it ends the session; the engine then reports an error to every admitted request, drains and restarts the session.
- Order of teardown (handoff 6D): stop new work -> publish abort -> join the proxy thread -> wait for in-flight GPU work
  (bounded) -> unmap/free slabs. No buffer is freed while a kernel or the proxy can still reference it.

### 2.5 Generation counters and slot ownership (tcp_proxy)
Slab (hipHostMalloc Mapped|Coherent): `Shared { u32 abort; ...; Flags slot[NSLOT] }`, `Flags { u64 ready, done,
consumed }` 64-byte aligned; TX slots `NSLOT x slot_bytes`; RX slots `NSLOT x slot_bytes` (peer block only).
Per gather with sequence q (device-side u64 counter, so graph replays advance it), slot s = q % NSLOT:
1. Host enqueue (eager) or graph replay: the stage kernel waits until `consumed[s] == q - NSLOT` (or q <= NSLOT), then
   copies `send` into TX[s] (or the producer writes TX[s] directly), `__threadfence_system()`, and one thread stores
   `ready[s] = q` (release, system scope). It also copies the own block into `recv + rank*bytes`.
2. Proxy thread: acquire-loads `ready[s] == q`; `ready > q` is a protocol error (abort). Sends TX[s] with header
   (epoch, q, site, bytes) and receives the peer's header + payload into RX[s]; checks the header.
3. Proxy: `done[s] = q` (release).
4. Wait/collect kernel: one thread acquire-loads `done[s] == q` with `s_sleep(1)` between loads and a wall-clock
   deadline; then all blocks copy RX[s] into `recv + peer*bytes` (or the consumer reads RX[s] directly).
   One guard load per block (LDS broadcast), never one per thread: measured 12-17 us vs 71-88 us for a 120 KiB consume.
5. Release kernel (or the collect kernel's last block): `consumed[s] = q` (release). The slot is reusable.
Stale or duplicate generations are rejected at both ends (`ready`/`done` greater than expected, header sequence not
+1). Counter wrap: 64-bit sequences do not wrap in practice; the 32-bit fields of the wire header are not used for
sequencing. A fresh session gets a fresh epoch (random nonces from both ranks, XORed), and a header from another epoch
is a `Mismatch`.
Validated already in Phase 0 (gpuvis, ds4-style kernels): 0 element errors over 161,930 network exchanges per rank and
204,000 local ones per box, 8 slots each reused >= 1,275 times per run, payloads 20 KiB to 1.5 MiB and the full
94-exchange round shape.

### 2.6 tcp_host vs tcp_proxy
- `tcp_host` (first light, eager only, `capturable = false`): simplest; costs a producer-event wait, a D2H, the exchange,
  an H2D and a stream dependency per gather. Measured: 105 us per 120 KiB gather (~50 us of it the exchange), 11.0 ms
  per DSpark round (section 5.2).
- `tcp_proxy` (graphs, `capturable = true`): stage/collect kernels on the coherent slab plus a pinned proxy thread
  that spins. Measured GPU-observed cost per exchange = network exchange + 1.2-1.9 us p50 visibility overhead. Through
  the family's Coll, two boxes, MTU 1500: DSpark round 7.25 ms p50 / 7.53 ms p95, plain round 2.16 ms, 1.3 us of host
  time per call, and a captured 94-gather DSpark pass replays in 7.52 ms p50 (section 5.2).
  Prefill (eager, 40-200 MiB per gather) uses the tcp_host path with chunked pipelining even when the proxy exists,
  because slabs sized for 200 MiB are pointless.

## 3. Transport over thunderbolt0

### 3.1 Sockets and the wait loop (measured, section 5)
- One data socket per cable: thunderbolt0 (primary, 10.40.0.0/24) and thunderbolt1 (10.40.1.0/24, `TF_TP_STRIPE`) for
  striping. The data links listen on rank 0's control-link address (TF_TP_MASTER, which should be 10.40.0.1) at the
  control port + 16 (`TF_TP_DATA_PORT` overrides). The control link stays on thunderbolt0: its frames are small and per
  round, and section 4 shows that bulk traffic on the same cable is what hurts. Listeners bind only to the USB4
  addresses.
- Socket options on data sockets: `TCP_NODELAY=1`; `SO_BUSY_POLL=50` (the unprivileged maximum because
  net.core.busy_read=50; check that it reads back 50 and that `SO_INCOMING_NAPI_ID` is non-zero after the first receive);
  `TCP_RTO_MIN_US=8000` (unprivileged on kernel 7.2; HZ=250 makes 8 ms the smallest value it accepts; tcp_info then
  reports rto 12 ms instead of 204 ms); buffers autotuned (tcp_wmem max 4 MiB, tcp_rmem max 32 MiB).
- The exchange loop spins on `send(MSG_DONTWAIT|MSG_NOSIGNAL)` and `recv(MSG_DONTWAIT)` without poll(), sleeps or
  yields while a gather is in flight. Each `recv` on an empty queue with SO_BUSY_POLL > 0 runs one thunderbolt_net NAPI
  poll in the calling thread. Waiting for the RX interrupt instead costs ~128 us per direction (interrupt throttling in
  the thunderbolt NHI driver; measured 40 B request/response 130 us without busy polling vs 15 us with).
- Header and payload go out in one `sendmsg` (iovec) and come in through one `recvmsg` loop (no separate header round
  trip, unlike the kit). Header (48 B): magic, wire version, flags, epoch (16 B), sequence u64, bytes u64, site u32,
  sender rank, part index and count. The receiver checks every field as soon as the header is in, before the call
  completes.
- The proxy/transport thread is pinned to one core in 5-7 (house rule: engines and networking on 0-7/16-23; USB4 IRQs
  are on 1-4 and 18-22) and spins: one full core per rank while the engine runs (measured 1.00 core per side).

### 3.2 Striping across both cables
- For payloads >= a threshold (default 4 MiB, tuned from measurements) the payload is split into two contiguous halves,
  each with its own header (stripe 0/1) on its own socket, both progressed in the same spin loop.
- Decode does not stripe: a measured 94-exchange round is 5.37 ms striped vs 5.06 ms on one cable, because per-message
  overhead doubles and only the 1.5 MiB head gather gains (432 vs 478 us).
- Prefill gathers (40 and 200 MiB) stripe. Measured per-gather medians at MTU 65520: 200 MiB 50.9 ms striped vs 84.1 ms
  on one cable; 40 MiB 10.8 vs 16.6 ms; 10 MiB 2.5 vs 3.7 ms. Whole 2048-row chunks did not gain as much (744 vs 835 ms
  median) because frame loss at these sizes caused retransmission stalls (section 4). At MTU 1500 the loss is gone and
  a striped chunk takes 512 ms (one cable 807 ms); through tcp_host 641 ms striped.

### 3.3 The coherent slab (tcp_proxy)
`hipHostMalloc(Mapped|Coherent)`; RLIMIT_MEMLOCK (8 MiB) does not constrain it (256 MiB measured). Slot size = largest
decode gather at the configured `round_rows`: head R*64640*4 (1.55 MB at R=6, 4.1 MB at R=16), Engram R*25600*4
(1.6 MB at R=16). With NSLOT = 8 and 4.25 MiB slots: TX + RX = 68 MiB per rank. GPU writes into coherent memory run at
72-94 GB/s; the visibility handshake costs 1.2-1.9 us p50 per exchange. The slab layout is part of the kit ABI:
`tf_ds_coll_slot` (64 B: ready, done, consumed generations, bytes, site) and `tf_ds_coll_shared` (1088 B: abort word,
device sequence counter, staged/collected counts, 16 slots) in `zig/kernels/hip/dsv41/abi.h` with per-field offset
asserts; tcp_proxy.zig checks the same layout at compile time against the generated `gpu.kernel_abi`.

The proxy thread pins itself to `TF_COLL_PROXY_CPU` (default 6) and spins. That core must carry no other spinning
thread: with both loopback ranks' proxy threads and rank 0's main thread on CPU 6, a DSpark pass took 405 ms instead of
7.7 ms (every handoff waited for a scheduler tick, 4 ms at HZ=250). serve.conf names the core (`COLL_PROXY_CPU`) and
keeps both ranks off it and its SMT sibling (`ENGINE_CPUS=0-5,7-21,23-31`, w-arch 12d8d78); the proxy thread's own
affinity call still places it on 6 (a thread can leave its process's taskset mask; checked on node1).

### 3.4 Chunked pipelining for large eager gathers (tcp_host, prefill)
Implemented (4439929): gathers >= 8 MiB (`TF_COLL_PIPE_MIB`) go through 4 MiB chunks (`TF_COLL_CHUNK_MIB`) on a copy
stream. Measured per 2048-row prefill chunk through the Coll, two boxes, MTU 1500: 593 ms pipelined vs 638 ms serial
striped over both cables, 810 vs 968 ms on one cable (wire alone: 512 / 807 ms, section 5.3).
The first `tcp_host` copied D2H, exchanged, then H2D, in series. Measured cost of the serial
copies: a 40 MiB gather takes 23 ms through tcp_host vs 14-17 ms on the wire alone (one link), 14 ms vs ~10.3-10.8 ms
striped; a 200 MiB gather 106 vs 81-86 ms. So the copies add ~25-40 % to prefill gathers, worth pipelining: two pinned
TX and two pinned RX chunk buffers (4 MiB); D2H of chunk k+1 overlaps the send of chunk k; the H2D of chunk k overlaps
the receive of chunk k+1. Since branch `link` (split gather), the own block's D2D copy also runs on the copy stream,
during the exchange, instead of on the caller's stream after it (19.5 ms of stream time per 2048-row chunk, section 5.2).

### 3.5 The split all-gather (overlap with compute)
`coll.Coll.gatherBegin(site, send, recv, count, dtype, stream)` / `gatherEnd(pending, stream)` (branch `link`): begin
takes `send` as the work queued on `stream` so far leaves it and returns at once; work queued between the two calls
must neither write `send` nor touch `recv`; end completes the gather, and work queued after it sees `recv`. One begun
gather per Coll at a time, no other call in between, both ranks pair them alike.
- tcp_host: begin records an event on `stream`, makes the copy stream wait for it and queues the chunks' D2H copies and
  the own block's copy there; end spins on each chunk's event, exchanges it and queues its upload on the copy stream,
  then makes `stream` wait for the last upload. The host thread blocks only in end, while the GPU runs the work queued
  in between. Every gather >= `TF_COLL_PIPE_MIB` runs as begin + end; smaller split gathers are one chunk.
- tcp_proxy: begin launches the stage kernel (the proxy thread sends as soon as the slot is ready); end launches the
  collect kernel; gathers larger than a slot use tcp_host's begin/end.
- Transports without the split (NCCL `Comm.coll`, test fakes) gather whole at end: the same bytes, no overlap, no
  caller branches.
Only rows of other work can overlap a gather (everything after a gather inside a layer depends on it), so the engine
side runs a prefill chunk as two halves zipped layer by layer (section 5.3).

### 3.6 RDMA write with immediate (rdma_proxy, pairs with RDMA network cards)
`TF_COLL_KIND=rdma` (serve.conf `TRANSPORT=rdma`) is tcp_proxy with the TCP exchange replaced by one RC RDMA write with
immediate per staged slot. Everything else is tcp_proxy's: the hipHostMalloc(Mapped|Coherent) slab and its
`tf_ds_coll_shared` header, the `coll.stage` / `coll.collect` kernels, the ready / done / consumed generation protocol,
the abort word and the fault contract (section 2.3 to 2.5); graph capture is unchanged because the proxy never sees a
host call. Gathers larger than a slot keep tcp_host (prefill stays on TCP).

Per gather q in slot s = q mod NSLOT, the proxy thread (pinned to `TF_COLL_PROXY_CPU`, busy-polling, never a completion
channel) sees ready[s] == q; writes a 64-byte trailer (magic, session epoch, q, bytes, site, rank) into TX[s] at
roundup(bytes, 64), so one SGE carries payload and trailer contiguously; posts one signaled RDMA_WRITE_WITH_IMM of
roundup(bytes, 64) + 64 bytes from TX[s] into the peer's RX[s] with imm = the low 32 bits of q; polls its send CQ and
receive CQ until both the local send completion and the peer's write of q are in; checks the receive completion's
byte_len and immediate and the trailer the peer wrote against its own slot words (any difference: Mismatch); then
stores done[s] = q. TX ownership is tcp_proxy's (consumed gates the stage kernel, done waits for the send completion).
RX ownership is transitive and holds for NSLOT >= 2, so NSLOT < 2 and the stripe thread (`TF_COLL_PROXY_STRIPE_KIB`
> 0) are refused. The peer writes gather q + NSLOT into RX[s] only after this rank's write of q + NSLOT - 1, which the
proxy posts only once stage(q + NSLOT - 1) set ready. The engine issues every gather on its one stream, one begun
gather at a time, so collect(q), the last reader of RX[s], is ahead of that stage in stream order. gatherEnd refuses
an end on a different stream from its begin (CollectiveFailed), the one way a caller could break that order inside a
gather. Two MRs cover the TX slots and the RX slots only, never the slab header. The handshake carries
`kind=rdma` and the two Infos (QP number, PSN, rkey, GID, MTU, slot count and size, receive region), and refuses a
mixed pair or differing slot geometry; a second small frame each is the start barrier once both QPs are in RTS.

mlx5 notes: receives are posted with a 64-byte SGE (a write with immediate lands at remote_addr, the SGE only reserves
a ring slot; a large SGE costs several striding-RQ strides and overruns the ring); the send and receive CQs are
separate; every send is signaled (mlx5 reaps unsignaled SQ WQEs only when a signaled completion is polled).

Failure: a completion with an error status poisons the collective (RETRY_EXC, RNR_RETRY_EXC, FLUSH and the remote
errors: PeerLost; others CollectiveFailed); no completion for `TF_COLL_PROGRESS_MS`: Timeout (the proxy's deadline
owns Timeout; the QP's own retries, `TF_RDMA_QP_TIMEOUT` exponent 14 by default, fail an unanswered write after a few
seconds on a ConnectX-6, inside it); the peer's tcp_host socket closing: PeerLost within milliseconds (a MSG_PEEK on
the control socket while waiting). abort and close move the QP to ERR so posted work flushes and the loop leaves.

Settings: tcp_proxy's `TF_COLL_SLOTS` (8), `TF_COLL_SLOT_KIB` (4352), `TF_COLL_PROXY_CPU` (6); `TF_RDMA_DEVICES` (one
ibdev name for both ranks, or one per rank separated by `,`; unset: the ibdev whose device/net holds
`TF_RDMA_IFNAME`, or the box's only ibdev); `TF_RDMA_GID_INDEX` (3: usually the RoCE v2 entry of the link's address);
plus tcp_coll.zig's. `tools/dsv41-hip/rdma_check.py` checks a box's device, port, GID, memlock and uverbs before a
rank starts (serve_lib.sh runs it under `TRANSPORT=rdma`; `preflight.sh --rdma-check` runs it alone).

Bring-up: the slab is registered with `ibv_reg_mr` on plain `hipHostMalloc` pages (no special allocation), so
`RLIMIT_MEMLOCK` must cover 2 x slots x slot size (8 MiB, the default in ssh sessions and systemd units, is not
enough: `limits.d` and `LimitMEMLOCK=` in any unit); an ssh `ControlMaster` session that predates a limits change
keeps the old limit; the HIP build has no `fabric` module, so `rdma_proxy` takes libibverbs through the small
`verbs` module (dlopen, no link dependency). The host suite's loopback tests need `TF_RDMA_TEST_DEVICE=<ibdev>`: both
ranks' QPs on one HCA, skipped without it.

## 4. Link reliability (USB4 frame loss and TCP retransmission timeouts)
Measured 2026-10-19 under the GPU/link lock and the quiet lock on both boxes (sessions logged in tf-port/w-link/
reliab.log on the Mac; raw latencies node0 ~/tfp/link/raw/rel_*). Settings were changed only for the session and
restored (MTU 65520 at the time, default routes, netdev_max_backlog 1000 at every session end). Outcome: the
integration owner made MTU 1500 permanent on both boxes (NetworkManager usb4-a/usb4-b, applied live, 018a4a8); later
sessions restore 1500.

What drops: thunderbolt_net loses whole frames when its RX ring overruns (`rx_missed_errors` counts frame-index gaps
inside a packet). The ring size and interrupt coalescing are fixed (SIOCETHTOOL ring/channels/coalesce: EOPNOTSUPP).
Losses come from bulk transfers at line rate, not from decode traffic:
- decode-shaped spin+busy-poll rounds: 0-2 frames per 200 rounds;
- 1 MiB symmetric exchanges: 0 frames with busy-poll, 45-90 per 3,000 without (one 217 ms RTO);
- one-way iperf3 at 28 Gb/s: 4.7 % of frames; bidirectional: 2.9-5.5 %; 2048-row prefill-shaped exchanges
  (40/200 MiB both ways at once): 2.6-3.7 %, with RTOs.
netdev_max_backlog 1000 -> 10000 changed nothing (the drop happens before the stack).

Each loss costs a retransmission; when it hits the tail of a message it costs an RTO, 204 ms at the kernel's floor.

Remedies, measured:

| | MTU 65520 (before) | MTU 9000 | MTU 1500 (now permanent) |
|---|---|---|---|
| iperf3 one-way: rate / frames missed | 28.0 Gb/s / 4.7 % (loaded sender) | 30.9 Gb/s / 0.22 % | 31.0 Gb/s / 0.04 % |
| iperf3 both ways (same session): rate / missed / bytes retransmitted | 21.1+20.6 Gb/s / 2.9-3.0 % / ~254 MB | 23.2+21.0 / 1.8-2.1 % / ~191 MB | 20.8+22.2 / 0.5-0.7 % / ~49 MB |
| 2048-row prefill chunk, one cable (p50) | 767 ms | 796 ms | 807 ms |
| same, striped over both cables (p50 / max) | 942 / 959 ms (5 RTOs; min 544) | 536 / 543 ms, 0 RTOs | 512 / 521 ms, 0 RTOs |
| DSpark round, 94 exchanges (p50 / p95) | 5.45 / 5.68 ms | 5.55 / 5.76 ms | 5.43 / 5.60 ms |
| plain 1-row round, 83 exchanges (p50) | 1.457 ms | 1.523 ms | 1.517 ms |
| 120 KiB request/response (p50) | 95-102 us | 96.0 us | 98.9 us |

- MTU 1500 on both cables removes most of the loss at no cost to decode (plain decode +4 %, about 60 us per token) and
  makes striped prefill stable (512 ms per 2048-row chunk). MTU 9000 sits in between.
- RTO floor: `TCP_RTO_MIN_US=8000` per socket (unprivileged) makes a lost tail cost ~12 ms instead of 204 ms; tcp_info
  rto 12000 us; no latency cost (DSpark round 5.52 / 5.73 ms p50/p95 vs 5.53 / 7.79 without, the latter with one RTO).
  The per-route equivalent `ip route change ... rto_min 8ms quickack 1` gave the same floor but quickack tripled
  spurious retransmits (681 per 5,500 request/responses); rto_min without quickack was not measured separately.
- Sender pacing (SO_MAX_PACING_RATE) was prepared but not run: the MTU change made it unnecessary for now.
- CPU energy preference (EPP, all cores, both boxes, 07:38-07:40, non-persistent and restored to `performance`):
  `balance_performance` against `performance`, interleaved twice. Link-only results agree in both repeats:
  120 KiB request/response 107.8-109.8 us p50 vs 96.5-99.6 us, DSpark transport round 6.02-6.05 ms vs 4.97-5.28 ms.
  The tcp_proxy rounds through the Coll did not separate (7.16 and 13.9 ms vs 7.10 and 7.97 ms); the 13.9 ms run
  coincided with node1 load rising from 4.9 to 6.3 (other workers' builds), so that comparison is inconclusive.
  Keep `performance` (the current setting) on both boxes.
- Decode rounds next to a bulk iperf3 flow (DSpark rounds, 150 each, MTU 65520 and 1500):
  - bulk on the OTHER cable (thunderbolt1): rounds unaffected, 5.37-5.46 ms p50 and <= 5.96 ms p95 at both MTUs, with
    or without the RTO floor;
  - bulk on the SAME cable (thunderbolt0): rounds collapse: MTU 65520 p95 146-165 ms, p99 373-521 ms; MTU 1500 p99 1.3 s
    (max 2.4 s) without the floor, and with the 8 ms floor p50 24 ms / p95 41 ms (queueing behind the bulk flow fires
    spurious timeouts; our socket's retransmissions rose from 155 to 958).
  So the decode cable must carry no bulk traffic, and the 8 ms RTO floor is only safe on a cable that carries none.

Recommended settings:
1. MTU 1500 on thunderbolt0 and thunderbolt1 on both boxes. Done 2026-10-19 by the integration owner (persistent in
   NetworkManager usb4-a/usb4-b, applied live); closed.
2. thunderbolt0 carries latency-critical traffic only (decode gates, the engine's collectives); bulk copies (model and
   cache transfers, iperf, scp/rsync between the boxes) go over thunderbolt1 (10.40.1.0/24). Prefill gathers may stripe
   over both because one engine never runs a prefill gather and a decode gather at the same time.
3. TCP_RTO_MIN_US=8000 on every latency-critical socket (tcp_wire sets it on the data links; DwarfStar's data sockets
   should too), given (2). No box-wide route rto_min: with quickack it raised spurious retransmits, and under
   same-cable contention a low floor fires spuriously.

RTO-floor stress (the question DwarfStar raised: two RTOs in one 42 MB exchange broke its 750 ms deadline at MTU 1500;
does tcp_host at the 8 ms floor stall on its own 40/200 MiB gathers?). `tf-dsv41-coll-check --prefill-stress 200`,
2026-10-19 10:36-10:46, both GPU locks and quiet windows, MTU 1500, 2 s progress deadline, 424 GB each way per box per
mode:

| | chunks | stalls | chunk p50 / p95 / p99 / max | worst 200 MiB / 40 MiB gather | RTOs h1/h2 | retransmitted segs h1/h2 | frames missed h1/h2 |
|---|---|---|---|---|---|---|---|
| striped | 200/200 | 0 | 622 / 659 / 678 / 705 ms | 79 / 42 ms | 34 / 13 | 134k / 125k | 8,687 / 8,955 (0.11 %) |
| one cable | 200/200 | 0 | 831 / 867 / 928 / 1,107 ms | 118 / 40 ms | 32 / 12 | 239k / 207k | 19,872 / 21,464 (0.25-0.28 %) |

At the 8 ms floor a loss costs tens of ms; the worst gather sat ~17x inside the progress deadline, so the planned
size-class RTO switch (200 ms for >= 1 MiB) was not built. Bulk prefill traffic still loses 0.1-0.3 % of frames at MTU
1500 (more than iperf3's 0.04 %), absorbed by fast retransmit.

Sender pacing (SO_MAX_PACING_RATE per socket, both ends; tcplat2, wire only, 2048-row chunk shape, 3 chunks per row,
10:58-11:00): it removes the loss but never reaches its set rate and costs 16-25 % of bulk time; decode rounds paced at
2.5 GB/s take 6.51 ms instead of 5.3-5.5. tcp_host stays unpaced.

| link | pacing per socket | chunk p50 | 200 MiB p50 | frames missed h1/h2 | retransmitted segs h1/h2 |
|---|---|---|---|---|---|
| thunderbolt0 | none | 809.5 ms | 81.5 ms | 560 / 456 | 4,039 / 7,299 |
| thunderbolt0 | 3.0 GB/s | 941.5 | 92.3 | 0 / 0 | 0 / 0 |
| thunderbolt0 | 2.5 GB/s | 964.0 | 95.2 | 0 / 0 | 0 / 0 |
| thunderbolt0 | 2.0 GB/s | 1,071.9 | 107.2 | 0 / 0 | 0 / 0 |
| thunderbolt0 | 1.5 GB/s | 1,410.7 | 140.4 | 0 / 0 | 0 / 0 |
| both | none | 517.9 | 51.2 | 443 / 513 | 4,933 / 4,605 |
| both | 2.0 GB/s | 645.2 | 60.4 | 0 / 0 | 0 / 0 |
| both | 1.5 GB/s | 764.7 | 74.4 | 0 / 0 | 0 / 4 (3 RTOs) |

## 5. Measured costs and projections
### 5.1 Single exchanges (thunderbolt0, spin + SO_BUSY_POLL=50, pinned, quiet boxes)
| Size | request/response p50 / p95 (us) | symmetric exchange p50 / p95 (us) |
|---|---|---|
| 40 B | 15.4 / 17.9 | 7.6 / 13.5 |
| 20 KiB | 29.2 / 33.4 | 15.3 / 26.5 |
| 120 KiB | 95.3-102.4 / 107.4-118.2 | 49.4 / 65.1 |
| 1 MiB | 599 / 638 | 378 / 618 |

### 5.2 Decode rounds (symmetric, back-to-back, one cable, transport only)
| Round | p50 | p95 |
|---|---|---|
| DSpark, call-site sizes (80 x 120 KiB, 2 x 600 KiB, 1.55 MB head, 6 x 100 KiB drafter, 5 x 16 B Markov; 94 exchanges, 13.2 MB each way) | 5.45-5.53 ms | 5.68-5.73 ms (7.79 ms in a run with one RTO, without the RTO floor) |
| DSpark, first Phase-0 mix (2 x 300 KiB Engram, 11 x 20 KiB drafter) | 5.06 ms | 5.25 ms |
| plain decode, 83 exchanges at 1 row (80 x 20 KiB, 2 x 100 KiB, 252.5 KiB head) | 1.46-1.47 ms | 1.53-1.56 ms |

GPU in the loop (gpuvis, ds4-style coherent-slab handshake, per-block guard; node0 carrying 13-17 busy cores of other
work): DSpark round 8.3-8.9 ms of exchange time p50 and 10.6 ms for the whole synthetic GPU round; plain decode 1.9-2.4
ms. Under the same load, 300 us compute gaps between exchanges raised link time per round from 7.6-7.9 to 10.8 ms.

Clean repeat (07:37-07:39, MTU 1500, quiet windows on both boxes, node0 load 0.7-1.2; node1 load 2.6-3.9 from other
workers' builds confined to cores 8-15/24-31, which the quiet window does not pause because each build process stays
under 50 % CPU; tests on cores 5-6):
| Round | p50 | p95 |
|---|---|---|
| DSpark, transport only, back to back | 5.33 ms | 5.43 ms |
| DSpark, 300 us gap before each exchange, spinning through the gap (link time, gaps excluded) | 9.22 ms | 9.62 ms |
| same, sleeping through the gap | 12.08 ms | 13.26 ms |
| same, sleeping, blocking poll() without busy-poll | 19.10 ms | 19.65 ms |
| plain, transport only | 1.475 ms | 1.557 ms |
| plain, 300 us gap, spinning | 4.86 ms | 5.85 ms |
| GPU in the loop (gpuvis), DSpark: whole GPU round / sum of exchanges | 6.8 / 5.1 ms | 7.3 / 5.6 ms |
| GPU in the loop (gpuvis), plain: whole GPU round / sum of exchanges | 2.4 / 1.5 ms | 2.5 / 1.6 ms |
No RTOs in any of these (the earlier contaminated runs had 215-230 ms tails). The gap cost is real: when the receiver
is not inside recv while the peer's data arrives, busy-poll does not run and the frame waits for the throttled
interrupt (120 KiB exchanges go from 51 us p50 to a bimodal 71 / 222-226 us at p50 / p90). The proxy thread spins in
recv while the GPU computes, which is the "spinning" row; keeping its socket polled is worth ~3 ms per DSpark round
over a thread that sleeps.

tcp_proxy through the family's Coll (tf-dsv41-coll-check TF_COLL_KIND=proxy, both boxes, MTU 1500, both cables for
gathers that stripe; session 2026-10-19 05:45, node0 load 3.7-5.0, node1 1.1-1.3): every word validated on both
ranks (1,896 eager calls including three 2048-row prefill chunks, 100 graph replays of a captured 94-gather pass), 0
bad. DSpark round 7.25 ms p50 / 7.53 ms p95 to the last upload (tcp_host 10.3-11.7 ms),
plain round 2.16 ms (tcp_host 3.4-3.6 ms), host time per call 1.3 us p50 for every size, graph replay of the DSpark
pass 7.52 ms p50, launch to sync. Repeats during the EPP session at `performance` (one cable): 7.10 and 7.97 ms DSpark,
2.16 and 2.27 ms plain.
tcp_host through the family's Coll (tf-dsv41-coll-check, branch `link`, both boxes, 03:31; node0 carried 10-14 busy
cores of other work, MTU 65520): every word of 1,896 calls validated on both ranks, 0 bad, one link and striped.
Host time per call, p50: 120 KiB 105 us (transport alone ~50 us), 600 KiB 330 us, 1.55 MB head 670 us, 100 KiB 107 us,
16 B 27 us; 20 KiB 36 us. Whole passes to the last upload: DSpark round 11.0 ms p50 / 11.6-11.7 ms p95 (transport
alone 5.45 ms), plain round 3.44-3.49 ms (transport alone 1.46 ms), 2048-row prefill chunk 1,161 ms on one link and
728 ms striped (200 MiB gathers 106 / 60 ms, 40 MiB 23 / 14 ms). So the staging (D2H, event wait, H2D and D2D on the
stream, API calls) costs ~55 us per 120 KiB gather, ~20 us per 20 KiB one: ~5.5 ms per DSpark round and ~2 ms per
plain token on top of the network. During this run the link lost 4.7-5.5 % of frames (prefill traffic at MTU 65520) and
saw 4-16 RTOs, which the 8 ms RTO floor contained (worst 120 KiB call 2.8 ms, worst 40 MiB call 42 ms).

Staging variants (same tool, 03:45-03:50, both MTUs): DMA copies through ordinary pinned memory vs device-side copies
through mapped coherent pinned memory ("blit") measure alike: DSpark round 11.6-11.7 ms (dma) vs 12.2-12.7 ms (blit),
plain round 3.48-3.59 ms vs 3.62-3.87 ms (node0 carried 3-11 busy cores of other work). The cost is the per-call
synchronization (the host waits for this rank's block to reach host memory, which waits for the previous gather's
upload and own-block copy in stream order: three GPU operations and an event round trip per gather), not the copy
engine. `tcp_host` therefore defaults to dma.

Projection for TP2 decode, per round: tcp_host as built ~11-12 ms per DSpark round and ~3.5 ms per plain 1-row token,
tcp_proxy ~7.1-7.5 ms and ~2.2 ms (measured above); the transport floor is ~5.3-5.5 ms and ~1.5 ms. Ways to close the
gap, in order of payoff (2 is done):
1. in-place gathers (the family passes send = recv + rank*bytes, as NCCL allows) remove the own-block copy: one GPU
   operation fewer per gather (w-arch's call sites). Measured (d2dcost, node1 10:58, quiet window): a decode-size copy
   (120 KiB-1.5 MB) costs 2.2-5.5 us of stream time, ~0.2 ms per round with tcp_host and ~nothing with tcp_proxy (its
   stage kernel writes the own block); at prefill sizes D2D runs at ~107 GB/s (40 MiB 0.39 ms, 200 MiB 1.95 ms; 19.5
   ms per 2048-row chunk), which the split tcp_host already moves off the caller's stream. Worth ~1-2 % at most;
2. tcp_proxy: kernels enqueued ahead, the host off the GPU's critical path, ~1.5 us of flag visibility per exchange,
   and graphs; per exchange the GPU still runs a stage and a collect copy (gpuvis: 4 us to write 120 KiB into coherent
   memory, 12-17 us to read it back with one guard load per block);
3. fusing stage and collect into the neighbouring kernels (the producing GEMM's epilogue writes its partial into the
   TX slab; the hc/add kernels read their peer operand straight from the RX slab), as ds4 does: per exchange ≈ wire
   time + ~1.5 us. This changes kernels owned by w-arch/w-tri/w-exl3 and is a P5 item.

### 5.3 Prefill
Per prompt token through the encoder-only layers: 40 x 5120 x 4 + 2 x 25600 x 4 = 1.024 MB per rank per direction;
2.0 GiB per 2048-row chunk. Measured per chunk (transport only, both directions at once):
- one cable: 767-835 ms p50 at MTU 65520 (with frame loss and occasional RTOs), 807 ms at MTU 1500;
- striped over both cables at MTU 1500: 512 ms p50, 521 ms max, no RTOs.
Through tcp_host (staging copies included, not pipelined): 1,066-1,082 ms per chunk on one cable, 641 ms striped at MTU
1500; pipelined (section 3.4): 810 ms on one cable, 593 ms striped. So the link costs ~0.25 ms per prompt token striped at MTU 1500 on the wire alone, 0.31 ms through tcp_host
(~0.39 / 0.52 ms on one cable): a link-only ceiling of
~4,000 prompt tokens/s striped (~2,500 on one cable) if nothing overlaps. Prefill gathers sit on the critical path
(prompt.zig: between attention and the FFN mix, and between MoE and the next mix, on one stream), so at an expected
GPU prefill rate of a few hundred to ~1,000 tokens/s the link adds ~10-25 % unless the gathers are overlapped with
compute (split each chunk's rows in two halves and gather half A while computing half B; a later optimization that
changes the prompt schedule but not the numerics).

Served engine (lead, 194530f, 2026-10-19): 2,010-token prompt 538 tok/s, 4,002-token 571 tok/s, so a 2048-row chunk
takes ~3.5 s, of which the gathers are ~0.6 s (~17 %).

Overlap, transport side measured (`tf-dsv41-coll-check --overlap 8`, both boxes, tcp_host striped, MTU 1500, both GPU
locks and quiet windows, 12:28-12:31; node0 load decaying 11.6 -> 1.8 after the window froze its jobs, node1 2.1 ->
1.2). A 2048-row chunk's 42 gathers, each step's work either a one-thread spin kernel (no memory traffic) or a D2D
copy (DRAM- and copy-engine-bound). "halves" = the same bytes as 84 half-size gathers, no work; "overlap" = the two
halves zipped (begin A, B's work, end A, begin B, A's next work, end B):

| work per half-step | work alone | gathers alone | halves | serial (today) | overlap | hidden (of the smaller) |
|---|---|---|---|---|---|---|
| 4 ms spin | 341 ms | 660 ms | 650 ms | 997 ms | 644 ms | 103 % |
| 8 ms spin | 681 | 659 | 664 | 1,364 | 785 | 88 % |
| 16 ms spin | 1,361 | 656 | 654 | 2,057 | 1,431 | 95 % |
| 768 MiB D2D copy | 646 | 686 | 639 | 1,347 | 1,192 | 24 % |
| 256 / 768 / 1,536 MiB copy (11:05, n=10) | 213 / 646 / 1,297 | 680 / 673 / 685 | - | 867 / 1,353 / 2,210 | 786 / 1,231 / 1,666 | 38 / 19 / 80 % |

Splitting a gather in two costs nothing (halves = gathers alone). With compute that leaves DRAM and the copy engines
alone, 88-100 % of the gathers hide; with a copy saturating DRAM only 19-38 %, because the staging copies and the
network compete with it. Real prefill compute (EXL3 GEMMs, attention) sits between the two.

Served engine (branch `link` 031e229 on strix-port 194530f, tri kit v3, fl5.conf: GPU_STRICT=1, tcp_host striped,
16K pool; both locks, quiet windows; 2026-10-19 13:57-14:02, node0 load 14 -> 2 decaying, node1 ~1.5; three requests
per prompt). `TF_DS_PREFILL_SPLIT=256` zips two halves (fillSplit); `TF_DS_PREFILL_SPLIT_ZIP=0` runs the same halves
one after the other (no overlap):

| | 2,010-token prompt (1 chunk) | 4,002-token prompt (2 chunks) |
|---|---|---|
| split off (one 2048-row chunk) | 3.68 s, 546 tok/s | 6.92 s, 579 tok/s |
| halves in sequence (no overlap) | 4.02 s, 500 tok/s | 7.48 s, 535 tok/s |
| halves zipped (overlap) | 3.25 s, 619 tok/s (+13 %) | 6.09 s, 658 tok/s (+14 %) |

The zip hides 0.77 s (one chunk) and 1.39 s (two chunks): ~0.7 s per 2048-row chunk, the whole of its gathers; the
halves themselves cost ~0.2-0.3 s per chunk (smaller GEMMs, more launches). An earlier session (13:26-13:31, 9c937cb)
gave the same picture (off 3.79 / 7.03 s, zipped 3.35 / 5.97 s).
Exactness on that base: rank 0's prompt-logits SHA-256 (TF_DS_LOGITS_SHA) is identical between zipped and sequential
halves for all 12 requests, so the overlap itself changes no bit; split vs whole chunk differed there (the 4K greedy
answer changed at its 4th token).

On strix-port 58fd1a9 (+ qual 686bd84; tri kit v4, exl3.gemm and the prompt experts; branch `link` e091f1e;
14:16-14:24, node0 load 1.5-4, node1 1.1-2.9; two requests per prompt) the split is bit-exact against the whole chunk:
prompt-logits SHA-256 equal for all 20 requests (12 teacher-forced prompts, 4K and 8K twice each), and the replay
window's teacher-forced logits (TF_DS_WINDOW_LOGITS, llink_tfgrade.py) bit-identical at every position: 18 prompts
(12 prose, 4 code, 2 README; 1.8K-10.9K tokens), 2,284 positions, KL(off||on) 0, top-1 100 %, NLL difference 0. That
passes the lead's acceptance rule (>= 2,048 positions, KL <= 6e-3, top-1 >= 96 %, paired NLL upper 95 % CI <= +0.005)
trivially; logs node1 ~/tfp/link/logs/tf-1019-1413 and tf-1019-1424, grade node0 ~/tfp/link/tf-grade-all.jsonl.
Speed:

| | 4,002-token prompt | 8,165-token prompt |
|---|---|---|
| split off | 4.80 / 5.43 s, mean 782 tok/s | 9.46 / 9.99 s, mean 840 tok/s |
| split on (zipped halves) | 4.04 / 4.39 s, mean 950 tok/s (+21 %) | 8.12 / 9.05 s, mean 952 tok/s (+13 %) |

Re-verified after the rebase onto strix-port f461120 (a-arch's in-place gathers: fillSplit sends attnOut / moeOut /
engramOut, engramApply gathers from engramOut; branch `link` e96af69, kit v4; 14:38-14:47, node0 load 1.3-3.1, node1
1.1-3.8): prompt-logits SHA-256 equal split on vs off for all 26 requests (and equal to the e091f1e run's, so the
in-place gathers changed no bit either); window logits bit-identical at all 2,284 positions (KL 0, top-1 100 %); greedy
answers equal. Speed (two requests each): 4K off 4.69 / 5.05 s (mean 821 tok/s), on 3.86 / 4.21 s (992 tok/s, +21 %);
8K off 9.37 / 9.99 s (844 tok/s), on 7.68 / 8.47 s (1,011 tok/s, +20 %). Arena (rank 1, 16K pool): 2.735 GB off,
3.209 GB on; 8K and 65K pools not measured, DRAFTS=1 not compared (a-arch's review).

## 6. Peer handshake (w-link owns it; w-arch supplies the kit fields)
On the control link right after `Fan.open`, before `Coll.open` and before any weights are trusted:
1. hello: magic, wire version, world, rank, 16-byte random nonce.
2. identity block, compared field by field; any difference refuses startup with the field name in the error:
   - engine: git commit + patch-set hash, build mode, `TF_GPU_STRICT`;
   - kit: kit ABI version, kit manifest SHA-256, gfx arch (w-arch);
   - model: checkpoint revision, config.json SHA-256, tokenizer/encoder SHA-256, chat template SHA-256, Modalities;
   - partition: world, rank-cache schema, rank-cache key (the `<key>` in `rank{r}of{w}-<key>.bin`, rank_cache.zig:143-157,
     never compared today), Engram constants hash, RoPE table hash;
   - execution: KV precision, ctx, chunk_rows, round_rows, streams, DSpark on/off and drafts, replay floor, graphs on/off;
   - transport: cables in use, stripe threshold, MTU, NSLOT, slot_bytes, deadlines, Coll kind.
3. epoch = nonce0 XOR nonce1 (non-zero); max payload = min of both.
4. Data sockets connect (thunderbolt0, then thunderbolt1); each opens with a hello carrying the epoch and its stripe
   index; a socket with a wrong epoch is closed.
The kit's 208-byte identity block (world, ABI, max_payload, six SHA-256 values) covers part of this; the field list above
extends it and is arbitrated by the integration owner.

## 7. Fault-injection harness
Handoff 6D requires: kill either rank; suspend a rank with the connection open; stall the proxy; dropped/blackholed
transport; cancel with a collective in flight; delayed completion across slot reuse; duplicate/stale generations;
counter wrap; fresh-session epochs; across startup, prefill, decode and verification.

### 7.1 No-model harness (implemented: `tf-dsv41-coll-check --fault`, `tools/dsv41-hip/coll_fault_matrix.sh`)
- Wire level (`zig build test-dsv41-hip-host`, tcp_wire.zig, loopback socketpairs): payload integrity one link and
  striped; Mismatch on a different site, size or order; Timeout when the peer stays silent (progress deadline); PeerLost
  when the peer closes; the failure sticks; abort from another thread ends a waiting call and the peer then loses its
  link; a peer from another session epoch is refused. coll_deadline_test.zig: a late peer completes the exchange (small,
  and large enough to stall the sender); a peer that stops mid-transfer fails at the progress deadline, one that never
  arrives at the arrival limit, a closed one at once; tcp_proxy's Core keeps a pair whose peer stages a gather late.
- Collective level, real GPU copies, both ranks on one box over loopback (no USB4 traffic, the box's GPU lock only):
  `tf-dsv41-coll-check --fault kill|stall|site|abort|close --at N` on one rank, `--peer-faults 1` on the other, eight
  cases. Run on node1 2026-10-19 03:16 (logs/fault-matrix-1.log): 8/8 pass. These runs predate the arrival limit
  (section 2.3); the script now sets `TF_COLL_GPU_WAIT_S=2`, so a 5 s stall or stop still outlasts the 4 s limit.

| Case | Faulty rank | The other rank | Time to fail |
|---|---|---|---|
| kill rank 1 at call 500 (SIGKILL) | killed | PeerLost | 20 ms |
| kill rank 0 at call 500 | killed | PeerLost | 21 ms |
| rank 1 stops calling for 5 s | exits after 5 s | Timeout | 2000 ms (the deadline) |
| rank 0 issues a different site at call 500 | Mismatch | Mismatch | immediate |
| abort from another thread on rank 0 | Aborted | PeerLost | immediate |
| rank 1 closes its collective | closed | PeerLost | immediate |
| SIGSTOP rank 1 for 5 s, then SIGCONT | resumes: PeerLost (does not continue) | Timeout | 2000 ms |
| ranks differ in TF_COLL_RTO_MIN_US | handshake refused, setting named | handshake refused | at open |

- The same eight cases against `tcp_proxy` (TF_COLL_KIND=proxy, node1, 04:2x, logs/fault-matrix-proxy-1.log): 8/8
  pass. With the proxy a call only enqueues kernels, so the error surfaces at the faulty rank's next call or at the
  stream synchronize that ends the round (the peer's proxy poisons the collective and sets the abort word, which ends
  its device waits): kill, stall, close and SIGSTOP are reported within one 94-gather round of the fault; a different
  site gives Mismatch on both ranks; abort gives Aborted / PeerLost.
- tcp_proxy slot protocol, GPU-free (`Core.run`, host tests in tcp_proxy.zig): 2,000 gathers through 4 reused slots;
  a collect that never publishes `consumed` blocks reuse of its slot (the proxy waits, it does not overwrite); a stale
  generation in a slot is refused.
- Negative controls of the word-for-word check (`--corrupt R:N` on both ranks: rank R flips one bit of its send pattern
  at call N, and in the first graph replay when the transport is capturable): each rank must find exactly that one bad
  word in half R and nothing else. Two cases in coll_fault_matrix.sh (a 20 KiB call and a 16 B Markov call). Run on
  node1 07:43 (logs/fault-matrix-negctl-0742.log): tcp_host 10/10 (the eight fault cases and both controls, one bad
  word found by each rank); tcp_proxy 10/10 (both controls found twice by each rank, once in the eager call and once
  in the first graph replay).
- Still to add: delayed `done` publication across slot reuse and duplicate/stale generations for `tcp_proxy` (its
  slots do not exist yet); counter wrap is moot for the 64-bit sequence but gets a test that starts near 2^64; a
  blackholed link (packets dropped silently, inside a private network namespace) to confirm the progress deadline and
  the RTO floor together; abort/kill during a 200 MiB prefill gather.
### 7.2 Real-link subset (both boxes, under the locks)
Kill each rank during decode and during a prefill gather; SIGSTOP a rank for longer than the deadline; unplug-equivalent
(`ip link set thunderbolt1 down` on the stripe/control cable only, restored afterwards); RTO behaviour under induced
loss; a 15-minute soak with the counters from section 4.

Run 2026-10-19 07:49-07:51 (both GPU locks, both cables, MTU 1500, progress deadline 2 s; `--fault-calls prefill` runs
2048-row prefill chunks so an external signal lands inside a 40 or 200 MiB striped gather; external signals by saved
PID 1 s after the collective opened; reliab.log on the Mac): 14/14 pass, 7 cases each for tcp_host and tcp_proxy.
These runs predate the arrival limit (section 2.3); reliab.py now sets `TF_COLL_GPU_WAIT_S=2` for the same reason.

| Case | tcp_host: rank 0 / rank 1 | tcp_proxy: rank 0 / rank 1 |
|---|---|---|
| rank 1 SIGKILLs itself at decode call 500 | PeerLost in 22 ms / killed | PeerLost at the round's sync (call 564) / killed |
| rank 0 SIGKILLs itself at decode call 500 | killed / PeerLost in 24 ms | killed / PeerLost at the round's sync |
| rank 1 stops calling for 5 s at call 500 | Timeout at 2,000 ms / exits | Timeout at the round's sync / exits |
| SIGSTOP rank 1 for 5 s during decode | Timeout at 2,001 ms / PeerLost on resume | Timeout / PeerLost on resume |
| SIGKILL rank 1 inside a prefill chunk | PeerLost in 36 ms / killed | PeerLost in 27 ms / killed |
| SIGKILL rank 0 inside a prefill chunk | killed / PeerLost in 54 ms | killed / PeerLost in 53 ms |
| SIGSTOP rank 1 for 5 s inside a prefill chunk | Timeout at 2,009 ms / PeerLost on resume | Timeout at 2,019 ms / PeerLost on resume |
No RTOs in any case; the killed or stopped bulk transfers left 46-1,558 retransmitted segments per box, which the
deadline bounds. Not yet run: thunderbolt1 down (needs the integration owner's OK; it changes a link setting).

15-minute soak (`tf-dsv41-coll-check --soak 900`, d7d3b8c; 2026-10-19 08:14-08:30, both GPU locks, no quiet window,
tcp_proxy with graphs, both cables, MTU 1500): per iteration one validated DSpark and plain round, ten timed of each,
one validated and ten timed graph replays; a validated 2048-row prefill chunk every 50th iteration.
- Integrity: 1,886 iterations, 5,625,606 gathers, 1,886 validated graph replays, 37 validated prefill chunks, 0 bad
  words on both ranks; both exit 0.
- Link: 672 GB received per box (~750 MB/s each way on average); frames missed 0.017 % (node0) / 0.007 % (node1) of
  received packets; retransmits 0.05 / 0.09 per 1,000 segments; 39 / 26 RTOs (at the 8 ms floor) and 45 / 1,431 tail
  loss probes in 15 minutes.
- Times (p50 / p99 / max): DSpark pass 13.7 / 15.9 / 31.4 ms, plain 3.93 / 4.63 / 6.37 ms, graph replay 13.8 / 16.1 /
  22.3 ms. These are about 1.9x the 05:45 session's (7.25 / 2.16 / 7.52 ms p50) because node0 carried load 17-27 all
  through (w-ref's reference traces, ~10 cores, confined to cores 8-15/24-31; the USB4 interrupts are on cores 1-4 and
  18-22, the test on 5-7). An A/B with and without a quiet window follows.

## 8. Implementation status and plan (branch `link`, node1 ~/tfp/wt/link)
Done:
1. `tcp_wire.zig` (GPU-free transport): spin loop + SO_BUSY_POLL + TCP_RTO_MIN_US, one `sendmsg`/`recvmsg` per step
   for header and payload, 48-byte header (magic, version, epoch, sequence, bytes, site, rank, part, parts), 2-link
   striping, progress deadline, terminal failure, abort from any thread. Host tests: 5.
2. `tcp_coll.zig` (`tcp_host`, `coll.Open` for HIP): handshake on the control link (hello with nonce, then the
   identity text: coll.Identity + transport settings, compared key by key, every difference logged by name), data
   links, pinned staging buffers that grow to the largest call, event spin waits, H2D/D2D on the caller's stream.
   Plugged into native.zig (rank 0) and lanes_main.zig (rank 1) for HIP builds.
3. `tf-dsv41-coll-check` (two-box validation + timing, and the fault mode) and `coll_fault_matrix.sh`.
Measured since (this document): two-box validation and timing of tcp_host (section 5.2), link settings (section 4),
staging variants (dma kept as default).
4. Done (4439929 after the rebase): chunked pipelining for tcp_host gathers >= 8 MiB (4 MiB chunks on a copy
   stream): 593 vs 638 ms per prefill chunk striped, 810 vs 968 ms on one cable.
5. Done (fbe3c79): `tcp_proxy` (TF_COLL_KIND=proxy): kit kernels `coll.stage` / `coll.collect` (coll.hip, registered in
   the dsv41 kit), a mapped coherent slab (8 slots of 4.25 MiB, 68 MiB), a pinned proxy thread with its own link,
   gathers larger than a slot through an embedded tcp_host (refused inside a capture), abort word + deadlines on every
   device wait. Two boxes: DSpark 7.25 ms, plain 2.16 ms, graph replay 7.52 ms, fault matrix 8/8 (section 5.2, 7.1).
   a5aed15: `coll.Open` takes the typed kit, the slab structs live in abi.h, and the dsv41 kit's required list names
   `coll.stage`/`coll.collect` as graph kernels.
6. Done on branch `link` (2026-10-19): the split all-gather (section 3.5), tcp_host's own-block copy on the copy
   stream, coll-check --prefill-stress and --overlap, and a draft of the engine side (`TF_DS_PREFILL_SPLIT`, a chunk
   as two halves zipped layer by layer; for w-arch's review).
Next:
7. In-place gathers at the family's call sites (w-arch): measured at ~1-2 % at most (section 5.2), low priority.
8. Real-link fault subset: 14/14 (section 7.2) except the cable-down case; the 15-minute soak: 0 bad words in 5.6 M
   gathers. (The clean repeat of the compute-gap and GPU-in-the-loop rounds is done, section 5.2.)
9. With the other workstreams (P5): fused stage/collect in the producing and consuming kernels.

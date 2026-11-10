//! DeepSeek-V4.1-Flash behind the native CUDA family interface, tensor parallel over two GPUs: the server's process is
//! rank 0 (the lane core over lanes.zig's Backend on model.zig), rank 1 is `tf-dsv41-lanes` following it on the other
//! node (the same model directory, rank cache, kit, pool and port). The ranks link over TCP and NCCL; the inputs
//! beyond the checkpoint come from the environment:
//!   TF_TP_WORLD (2), TF_TP_MASTER, TF_TP_PORT   the ranks: rank 0's fabric address and port (rank 1 connects)
//!   TF_DS_RANK_CACHE                            the lane's per-rank weight files (the Python lane's TF_DS_RANK_CACHE)
//!   TF_DS_KIT                                   cubins/ (the served extension cubins), the served RoPE tables
//!                                               (rope-*.f32, as many rows as the context) and engram.json
//!   TF_DS_ENGRAM, TF_DS_TOKEN_MAP               the Engram tables and the compressed token map
//!   TF_DS_HIP_KIT                               HIP: the AMD kit (unset: the server's, native/hip.zig)
//!   TF_DS_AUX                                   HIP: hip (default), triton, or a list of attn,index,hc: which Triton
//!                                               launches take w-tri's HIP kernels (tri_aux.zig; both ranks alike)
//!   TF_DS_ARENA_GIB                             a first block of device memory for the buffers (unset: they take what
//!                                               they need, the pool's caches by the context)
//!   TF_DS_ROUND_MS                              the lane core's round costs by rows (ms, comma separated; the
//!                                               served ROUND_MS unless set: a four-node lane's rounds cost less)
//!   TF_DS_DRAFT_MS                              the same profile's drafter pass cost (ms; GB10's 3.5 unless set)
//!   TF_DS_ROUND_ROWS, TF_DS_STREAMS             a round's rows at most (16; the four-node lane's 48) and the pool's
//!                                               streams (4; its 16): --parallel at most the streams
//!   TF_RDMA_DEVICES                             the decode gathers over the RDMA ring on these devices (else NCCL)
//!   TF_DS_GRAPHS=1                              the rounds' stretches as CUDA graphs
//!   TF_DS_HC_SIDE=1                             the mixes' side work on a stream of its own
//!   TF_DS_L2_PREFETCH=1                         the paced L2 prefetch of the next kernels' weights
//!   TF_DS_ENGRAM_AIO=1                          a round's Engram reads by Linux AIO on O_DIRECT
//!   TF_DS_FIX_K=K                               a measurement: every drafted stream verifies K drafts a round
//!   TF_DS_PROMPT_DISK_*                         sf-disk: kept prompt states on disk on both ranks (disk.zig)
//!   TF_DS_POOL                                  the streams' shared pool in positions (unset: --context); at least
//!                                               --context, which stays one request's window (and the RoPE tables')
//! The kernel set (TENSORFOLD_CUDA_KERNELS) is the recorded Triton set (aot_pack.py). The pool holds TF_DS_POOL (else --context)
//! positions for every stream together (each takes an extent: its prompt, its max_tokens and a round's rows).
//! TF_TP_WORLD=4: the four nodes of the exact 2D split (prompt2d.zig), ranks 1-3 `tf-dsv41-lanes` following on the
//! other nodes; TF_RDMA_DEVICES (and TF_DS_RDMA_KERNELS: rdma_gather.cu's fatbin, unset: the embedded one) put its
//! decode-size exchanges on the RDMA rings, as the four-node Python lane's TF_RDMA_DEVICES does.
const std = @import("std");
const gpu = @import("gpu");
const lanes = @import("lanes");
const model = @import("model.zig");
const lanes_mod = @import("lanes.zig");
const round = @import("round.zig");
const draft_mod = @import("draft.zig");
const link = @import("link.zig");
const rank_cache = @import("rank_cache.zig");
const sampling = @import("sampling.zig");
/// The vision tower exists in CUDA builds only (gpu.features.vision); a HIP build compiles none of it.
const vision_mod = if (gpu.features.vision) @import("vision.zig") else struct {
    pub const Vision = lanes_mod.VisionTower;
};

pub const model_type = "deepseek_v41";
pub const formats: []const []const u8 = &.{"exl3-mul1"};
/// The Python lane's default window; the server's --context moves it up to the model's 1,048,576.
pub const default_context: i64 = 262144;
pub const max_segments: u32 = 1;
/// The served build's prompt chunk (TF_DS_PREFILL_CHUNK).
pub const prompt_rows: u32 = 2048;
/// The served lane's reasoning effort numbers (text.zig): the server passes each name and the template hears its number.
pub const efforts = @import("text.zig").efforts;
/// The served sampler's top_k (sampling.zig: top_k + MARGIN candidates, at most 64), which a sampled request must set:
/// the nucleus path for top_k off is not ported (U49). The server refuses the rest before the engine sees them.
pub const top_k_max: u32 = sampling.max_candidates - sampling.margin;
pub const needs_top_k = true;

/// `modalities` null: TF_DS_MODALITIES (text or text_image), else the backend's default (model.default_modalities).
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, modalities: ?model.Modalities = null };
pub const Modalities = model.Modalities;

/// The request kinds this rank serves: the caller's, else TF_DS_MODALITIES, else the backend's default.
pub fn modalities(o: Options) !Modalities {
    if (o.modalities) |x| return x;
    const v = getenv("TF_DS_MODALITIES") orelse return model.default_modalities;
    return std.meta.stringToEnum(Modalities, v) orelse error.BadModalities;
}

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence.
    stream_bytes: usize,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    /// Kept prompt states for the host's prompt cache (core/prompt_cache Snapshots: `ptr` and its functions).
    snaps: ?Snaps = null,
    /// Images in requests (the server renders their placeholders; the tower runs on this rank).
    vision: ?lanes.stream.Vision = null,
    /// Structured output and required calls (the tokenizer's vocabulary read at open).
    structures: bool = false,
    /// The context window the server advertises and checks requests against: the most prompt plus max_tokens tokens
    /// the pool holds (lanes.usableContext), below --context by a round's rows and the extents' chunk alignment.
    context_window: ?usize = null,
};

pub const Snaps = struct {
    ptr: *anyopaque,
    bytes: *const fn (*anyopaque, u32) u64,
    save: *const fn (*anyopaque, ?*anyopaque, u32) anyerror!*anyopaque,
    restore: *const fn (*anyopaque, ?*anyopaque, *anyopaque) anyerror!void,
    drop: *const fn (*anyopaque, *anyopaque) void,
};

/// The rank's model and the backend over it (what `open` hands the server, freed by `deinit`).
const State = struct {
    gpa: std.mem.Allocator,
    m: *model.Model,
    lanes: lanes_mod.Lanes,
    vision: ?*vision_mod.Vision = null,
    vocab: ?*lanes.grammar.Vocab = null,
};

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// An empty variable as none (a container passes TF_RDMA_DEVICES empty when it is unset).
fn nonEmpty(v: ?[]const u8) ?[]const u8 {
    const s = v orelse return null;
    return if (s.len == 0) null else s;
}

/// The streams' shared pool: TF_DS_POOL (empty or unset: the context, one stream may take it all), at least the
/// context (a request's window: model.contextCap refuses a larger one).
fn poolOf(context: usize) !usize {
    return poolFrom(nonEmpty(getenv("TF_DS_POOL")), context);
}

fn poolFrom(v: ?[]const u8, context: usize) !usize {
    const t = v orelse return context;
    const p = std.fmt.parseInt(usize, t, 10) catch return error.BadPool;
    if (p < context) return error.BadPool;
    return p;
}

test "TF_DS_POOL: the context unless set, never below it" {
    try std.testing.expectEqual(@as(usize, 1048527), try poolFrom(null, 1048527));
    try std.testing.expectEqual(@as(usize, 2097152), try poolFrom("2097152", 1048527));
    try std.testing.expectError(error.BadPool, poolFrom("65536", 1048527));
    try std.testing.expectError(error.BadPool, poolFrom("2M", 1048527));
}

/// The device bytes this rank's weights take, for the host's memory check: its file in TF_DS_RANK_CACHE (the
/// checkpoint holds both ranks'); 0 when there is none to read (open then says what to set).
pub fn weightBytes(io: std.Io, dir: []const u8) u64 {
    _ = dir;
    var d = std.Io.Dir.cwd().openDir(io, getenv("TF_DS_RANK_CACHE") orelse return 0, .{ .iterate = true }) catch return 0;
    defer d.close(io);
    var buf: [512]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return 0;
    const name = rank_cache.find(fba.allocator(), io, d, 0, world) catch return 0;
    const st = d.statFile(io, name, .{}) catch return 0;
    return st.size;
}

/// Loads rank 0 and links rank 1 (which must start against TF_TP_MASTER:TF_TP_PORT); the pool holds TF_DS_POOL
/// positions (else `o.context`), a request at most `o.context`. Needs `ctx` current.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const gpu.Context, dir: []const u8, kernels: []const u8, o: Options) !Loaded {
    const world = std.fmt.parseInt(u32, getenv("TF_TP_WORLD") orelse "2", 10) catch return error.BadWorld;
    if (world != 2 and world != 4) return error.BadWorld; // TP2, or the four-node 2D split
    const master = link.parseIp(getenv("TF_TP_MASTER") orelse return error.NoMaster) catch return error.BadMaster;
    const port = std.fmt.parseInt(u16, getenv("TF_TP_PORT") orelse "29620", 10) catch return error.BadPort;
    const arena_gib = if (getenv("TF_DS_ARENA_GIB")) |t| std.fmt.parseInt(usize, t, 10) catch return error.BadArena else 0;
    // each prompt's logits' SHA-256 in the log (rank 1: --logits-sha), to compare two builds or settings
    if (getenv("TF_DS_LOGITS_SHA")) |v| lanes_mod.logits_sha = !std.mem.eql(u8, v, "0");
    // each verify window's position, ids and drawn tokens on stderr (rank 0; comparing two runs' schedules)
    if (getenv("TF_DS_TRACE_ROUNDS")) |v| lanes_mod.trace_rounds = std.fmt.parseInt(u8, v, 10) catch return error.BadSetting;
    // the scheduler's cost profile, measured on this backend (the model keeps its own copy)
    const round_costs = if (getenv("TF_DS_ROUND_MS")) |v| lanes_mod.parseMs(gpa, v) catch return error.BadRoundMs else null;
    defer if (round_costs) |v| gpa.free(v);
    const draft_cost = if (getenv("TF_DS_DRAFT_MS")) |v| draft_mod.scheduler_costs.parseDraft(v) catch return error.BadDraftMs else draft_mod.draft_ms;
    const mods = try modalities(o);
    const st = try gpa.create(State);
    errdefer gpa.destroy(st);
    st.gpa = gpa;
    st.m = try model.Model.open(gpa, io, ctx, .{
        .model_dir = dir,
        .cache_dir = getenv("TF_DS_RANK_CACHE") orelse return error.NoRankCache,
        .kit_dir = getenv("TF_DS_KIT") orelse return error.NoKit,
        .aot_dir = kernels,
        .rank = 0,
        .world = world,
        .master = master,
        .port = port,
        .engram_dir = getenv("TF_DS_ENGRAM"),
        .token_map = getenv("TF_DS_TOKEN_MAP"),
        .rdma_devices = nonEmpty(getenv("TF_RDMA_DEVICES")),
        .rdma_kernels = getenv("TF_DS_RDMA_KERNELS"),
        .graphs = if (getenv("TF_DS_GRAPHS")) |v| !std.mem.eql(u8, v, "0") else false,
        .side = if (getenv("TF_DS_HC_SIDE")) |v| !std.mem.eql(u8, v, "0") else false,
        .prefetch = if (getenv("TF_DS_L2_PREFETCH")) |v| !std.mem.eql(u8, v, "0") else false,
        .engram_aio = if (getenv("TF_DS_ENGRAM_AIO")) |v| !std.mem.eql(u8, v, "0") else false,
        .pool = poolOf(o.context) catch return error.BadPool,
        .context = o.context,
        .drafts = o.drafts,
        .arena_bytes = arena_gib << 30,
        .round_rows = if (getenv("TF_DS_ROUND_ROWS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadRoundRows else round.default_rows,
        .round_ms = round_costs,
        .draft_ms = draft_cost,
        .streams = if (getenv("TF_DS_STREAMS")) |v| std.fmt.parseInt(usize, v, 10) catch return error.BadStreams else model.max_streams_default,
        .modalities = mods,
        .hip_kit = getenv("TF_DS_HIP_KIT") orelse if (gpu.backend == .hip) kernels else null, // HIP: the host's kit
        .open_coll = if (gpu.backend == .hip) @import("tcp_coll.zig").open else null,
        .disk = @import("disk.zig").wanted(world),
    });
    errdefer st.m.close();
    st.lanes = lanes_mod.Lanes.init(gpa, st.m);
    // sf-disk: this rank's tier, reconciled with rank 1's before the server takes a request (off: nothing changes)
    try st.lanes.attachDisk();
    if (getenv("TF_DS_FIX_K")) |t| st.lanes.fix_k = std.fmt.parseInt(usize, t, 10) catch return error.BadFixK;
    // structured output: each token's bytes, for the draws that keep a reply inside its grammar (rank 0 draws)
    st.vocab = lanes.grammar.vocab.load(gpa, io, dir) catch |e| blk: {
        std.log.warn("structured output off: {s}/tokenizer.json is not a byte-level BPE vocabulary this engine reads ({s})", .{ dir, @errorName(e) });
        break :blk null;
    };
    errdefer if (st.vocab) |v| v.deinit(gpa);
    st.lanes.vocab = st.vocab;
    // a vision checkpoint's tower on this rank, from the kit's vision/ (torch's attention cubin, the routing bias),
    // unless this rank serves text only: config.json still declares the tower, nothing of it is opened, and the server
    // refuses image parts (Loaded.vision null)
    st.vision = null;
    if (st.m.cfg.vision and mods == .text) std.log.info("config.json declares a vision tower; not opened (text capability)", .{});
    if (gpu.features.vision and st.m.cfg.vision and mods == .text_image) st.vision = try vision_mod.Vision.open(gpa, io, st.m, dir, getenv("TF_DS_KIT").?);
    errdefer if (gpu.features.vision) if (st.vision) |v| v.close();
    st.lanes.vision = st.vision;
    // the pool is the model's (preallocated): a stream takes no device memory of its own
    const L = lanes_mod.Lanes;
    const snaps: Snaps = .{ .ptr = &st.lanes, .bytes = L.snapBytesFn, .save = L.snapSaveFn, .restore = L.snapRestoreFn, .drop = L.snapDropFn };
    return .{ .backend = st.lanes.backend(), .facts = st.lanes.facts(), .rows = @intCast(st.m.eng.round_rows), .stream_bytes = 0, .ctx = st, .deinit = deinitFn, .snaps = snaps, .vision = if (gpu.features.vision) (if (st.vision) |v| v.hook() else null) else null, .structures = st.vocab != null, .context_window = lanes_mod.usableContext(st.m.context_cap, st.m.eng.round_rows) };
}

fn deinitFn(ptr: *anyopaque) void {
    const st: *State = @ptrCast(@alignCast(ptr));
    st.lanes.deinit(); // rank 1 hears done
    if (st.vocab) |v| v.deinit(st.gpa);
    if (gpu.features.vision) if (st.vision) |v| v.close();
    st.m.close();
    st.gpa.destroy(st);
}

/// A request or a start this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.BadWorld => "the native DeepSeek-V4.1 engine runs tensor parallel over two GPUs or the four nodes of the 2D split (CUDA builds): set TF_TP_WORLD=2, or 4 on CUDA",
        error.NotPortedYet => "this request reaches a path the port does not have yet (the engine's log names its site)",
        error.NeedsTopK => "sampling with top_k off (the nucleus over the whole vocabulary) is not ported yet: send top_k from 1 to 56, or temperature 0",
        error.TooManyCandidates => "top_k above 56 is beyond this engine's sampler (top_k + 8 candidates, at most 64): send top_k from 1 to 56",
        error.NoMaster, error.BadMaster => "set TF_TP_MASTER to rank 0's fabric address (rank 1: tf-dsv41-lanes MODEL CACHE 1 2 MASTER PORT KIT; ranks 1-3 of four: ... G 4 ...)",
        error.NoRdmaKernels => "TF_RDMA_DEVICES at world 4 needs TF_DS_RDMA_KERNELS (rdma_gather.cu's fatbin) in a build without the kernel images",
        error.NoRankCache => "set TF_DS_RANK_CACHE to the lane's per-rank weight files",
        error.NoKit => "set TF_DS_KIT to the kernel kit (cubins/, the RoPE tables, engram.json)",
        error.NoFreeSlot => "this engine serves TF_DS_STREAMS streams at once (default 4): --parallel at most that",
        error.BadRoundRows => "TF_DS_ROUND_ROWS: a round's rows at most, 16 .. 48",
        error.BadRoundMs => "TF_DS_ROUND_MS: a round's cost in ms by its rows from 1, comma separated, each finite and above 0 (at most 48)",
        error.BadDraftMs => "TF_DS_DRAFT_MS: a drafter pass's cost in ms, finite and not negative",
        error.BadStreams => "TF_DS_STREAMS: the pool's streams, 1 .. 16",
        error.BadPool => "TF_DS_POOL: the streams' shared pool in positions, at least --context (unset: --context)",
        error.ContextFull => "the streams' prompts and budgets fill the context's pool: lower max_tokens, or retry when a stream ends",
        error.PromptTooLong => "the prompt and its max_tokens do not fit the context window",
        error.NoVisionKit => "this checkpoint reads images: its kit (TF_DS_KIT) needs vision/torch_fmha_sm120.cubin and vision/gate_bias_vl.safetensors",
        error.VisionUnsupported => "image input is not implemented on this backend: serve text only (TF_DS_MODALITIES=text)",
        error.BadModalities => "TF_DS_MODALITIES: text or text_image",
        error.NoCollective => "this build has no collectives transport for the ranks yet (HIP: w-link's TCP transport, model.Options.open_coll)",
        error.MissingKernel => "the kernel kit lacks a kernel this configuration reaches (the log names its id; EXL3's are w-exl3's exl3.* entries)",
        error.NoVerbsRdma => "TF_RDMA_DEVICES: the verbs RDMA ring is CUDA's; this build exchanges over its collectives transport",
        error.NoL2Prefetch => "TF_DS_L2_PREFETCH needs the kit's L2 prefetch kernel (CUDA: tf_ds_l2_prefetch_kernel; HIP: exl3.l2_prefetch), and this kit has none",
        error.GraphsNeedCapturableCollective => "TF_DS_GRAPHS=1 needs a collectives transport that runs inside graphs; this one does not: unset it",
        error.NoStructures => "structured output needs the checkpoint's tokenizer.json as a byte-level BPE vocabulary (the engine's log says why it was not read)",
        error.BadStructure => "the request's structure does not compile (the engine's log has the compiler's words)",
        else => null,
    };
}

test "refusals name what to set" {
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoMaster).?, "TF_TP_MASTER") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NoFreeSlot).?, "TF_DS_STREAMS") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.BadWorld).?, "TF_TP_WORLD") != null);
    // a request that reaches an unported path is not told to change the ranks (U49)
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NotPortedYet).?, "TF_TP_WORLD") == null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.NeedsTopK).?, "top_k") != null);
    try std.testing.expect(std.mem.indexOf(u8, explain(null, error.TooManyCandidates).?, "top_k") != null);
    try std.testing.expectEqual(@as(?[]const u8, null), explain(null, error.OutOfMemory));
}

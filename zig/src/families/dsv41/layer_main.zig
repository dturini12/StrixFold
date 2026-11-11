//! tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N] [--chunks C] [--check-from F]
//!   [--engram DIR --token-map FILE] [--dump DIR]: the Zig port's layer gate (M3).
//! WORLD 4: the four nodes of the exact 2D split (prompt2d.zig), RANK the node, the links on PORT .. PORT + 2.
//! Both ranks run the recorded prompt's first C chunks (each Model.forward of the layer fixtures, in order; the
//! compressed caches and the window rings carried from chunk to chunk) through the port's prompt forward (prompt.zig)
//! on their GPUs, at most N layers a chunk, with the recording's bounded replay (replay = the prompt's length - the
//! window: the decoder's first layer cuts a chunk to its rows from there, an encoder-only chunk ends at it), and from
//! chunk F on compare every point a layer exchanges with the served build's bytes: the attention's input rows and
//! partial, its gather, the MoE's input rows and partial, its gather, on an Engram layer its rows' projection and their
//! gather; after a chunk's last layer the DSpark taps, the head's columns, their gather and the prompt's logits.
//! With --rounds K the recorded decode rounds follow (eager rounds: RoundDecoder.run's rows, the stream in pool slot 0;
//! serial, or a drafting recording's windows of the pending token and its drafts): the first K of the prompt's request
//! through round.zig, each checked at every layer's exchanges (Engram's projection and gather, the attention's input
//! rows, partial and gather, the MoE's), the head's columns and gather, the logits and the taps, and the served
//! acceptance (the rows' greedy tokens against the drafts the next round kept and its first row). With --drafts B (a
//! drafting recording; B the request's max_tokens) the drafter runs as served around them (draft.zig): the prompt's taps
//! absorbed into its rings, then before each round the batched pass (every recorded drafter point: each stage's
//! attention rows, ring plane, partial and gather, the MoE's, the stages' streams, the head's columns, the Markov loop's
//! drafts) and the served depth from its confidences (_choose_k) against the round's window, after it the window's taps
//! absorbed.
//! With --light 1 (a token recording: TF_ZREC_ONLY=Model.forward,RoundDecoder.run) every recorded request instead: its
//! prompt's chunks through every layer unchecked, the prompt's logits checked, then its rounds (at most K with
//! --rounds K, all without), each round's logits checked and its greedy token against the recorded next one; the
//! caches and rings cleared between requests.
//! One JSON line a point; the first difference stops the run (its first differing element and how many differ).
//! REC_DIR: aot/ (the recording's Triton set), cubins/ (linear.cubin, linear_grouped.cubin, experts.cubin,
//! experts_cb.cubin: the served extension cubins), rope-{plain,compressed}-{cos,sin}.f32, rope.json and engram.json (zrec_fixtures.py), rank<R>/layers.jsonl and
//! layers/. Engram layers need the original Engram tables (DIR) and the compressed token map (the lane's JSON cache).
//! TF_DS_ROUND_MS / TF_DS_DRAFT_MS: the cost profile the recording chose its draft depths with (unset: GB10's, the
//! served lane's), for the replayed _choose_k.
//! On HIP (the gpu module's AMD backend): the kernels come from the AMD kit (--hip-kit DIR, as tf-dsv41-lanes), not
//! REC_DIR's cubins and aot/ (REC_DIR still holds the fixtures and RoPE tables); no RDMA, no 2D split. TF_DS_AUX
//! (hip, triton, or a list of attn,index,hc) chooses w-tri's HIP kernels or the Triton launches, as the server does.
//! Layer grading (w-ref's FP64-derived fixtures, tools/dsv41-hip/ADAPTER.md 14):
//!   --dump DIR        every checked point's bytes as the engine holds them, in the fixtures' own schema (w-ref's
//!                     grade.py reads it): DIR/rank<R>/layers.jsonl (the fixture line's fields and key, the engine's
//!                     sha256, equal) and DIR/layers/r<R>.<key>.bin. The key is the fixture's own when it carries one,
//!                     else STEP:LAYER:WHERE.ARG (STEP the prompt chunk, then the decode round; LAYER -1 for the
//!                     model's own points; a gather's WHERE Comm.gather.<engram|attn|moe|head>). Each line carries
//!                     row_start: after the replay cut a point the fixture holds for the whole chunk is checked and
//!                     dumped on the engine's rows only (its rows dimension the engine's; a gather's [world, rows, ..]
//!                     each rank's), the first of them the fixture's row row_start. Also dumped, with equal null: the
//!                     router's input, STEP:LAYER:route_logits, fp32 [n, kc, E] chunk sums (n <= 16) or [n, E] logits
//!                     (prompt chunks and rounds); and an index layer's selection, STEP:LAYER:Ref.indexer.cidx, int64
//!                     [n, kk] (kk the selection's width, -1 past a row's visible compressed entries)
//!   --no-replay-cut 1 no bounded replay: every layer runs the chunk's every row (a fixture of an uncut reference)
//!   --anchor 1        before each layer, the fixture's Ref.layer_in h (bf16 [n, hc, D], the streams after the
//!                     previous layer's post) and pre (fp32 [n, hc], the carried pre-mix) replace the engine's, and the
//!                     pending post is dropped: every layer starts from the reference (prompt chunks, rounds and the
//!                     drafter's stages, fixture layers LAYERS + stage). A
//!                     prompt layer's output is then posted in place and checked against Ref.layer_out (h, pre)
//!   --kit-dir DIR     the RoPE tables and engram.json (default REC_DIR; w-ref's fixtures: the inputs' kit)
//!   Ref.route pick / wts, when the fixtures record them, are checked as the first top_k slots of the routing (prompt
//!   chunks and rounds)
//!   --keep-going 1    a point that differs is reported and the run goes on (the grader decides)
//!   --replay-gather 1 no transport: each all-gather checks this rank's partial against the recording's Comm.gather in1
//!                     and returns its out (one rank graded alone on one box; no links opened). Without it a HIP gate
//!                     links to its peer and gathers over w-link's TCP transport (tcp_coll.zig; TF_TP_* / TF_COLL_*
//!                     as the server's, equal on both ranks)
const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const coll = dsv41.coll;
const prompt = dsv41.prompt;
const round_mod = dsv41.round;
const draft_mod = dsv41.draft;
const sampling = dsv41.sampling;

/// Backend-only types, comptime-gated: a HIP build holds no NCCL, no 2D split and no verbs; a CUDA build no AMD kit.
const Nccl = if (gpu.backend == .cuda) dsv41.comm.Comm else void;
const HipKit = if (gpu.backend == .hip) *gpu.Kit else void;
const Two = if (gpu.features.world4) dsv41.prompt2d.Two else void;
const Rings2d = if (gpu.features.world4 and gpu.features.verbs_rdma) dsv41.ring2d.Rings else void;

const usage = "usage: tf-dsv41-layer MODEL_DIR CACHE_DIR RANK WORLD MASTER_IP PORT REC_DIR [--layers N] [--chunks C] [--check-from F] [--rounds K] [--drafts B] [--engram DIR --token-map FILE] [--dump DIR] [--anchor 0|1] [--keep-going 0|1] [--replay-gather 0|1] [--no-replay-cut 0|1] [--hip-kit DIR] [--kit-dir DIR]\n";
const chunk_rows = 2048;
const cache_tokens = 4096; // the compressed caches' positions at least (the recorded prompt's)
/// The served pool's window (--context): the bucket rule's cap (graph.py bucket_for).
const pool_window = 1 << 20;
const max_chunks = 8;
/// The most dimensions a fixture's shape has (the replay cut rewrites its rows dimension in a copy).
const max_dims = 8;
/// --side 1: the rounds' mixes' side work on a stream of its own (round.Round.useSide).
var side_on = false;
/// --prefetch 1: the rounds' paced L2 prefetch (round.Round.usePrefetch).
var prefetch_on = false;
/// --aio 1: the rounds' Engram reads by Linux AIO on O_DIRECT (engram_aio.zig).
var aio_on = false;

/// One fixture line of layers.jsonl.
const Point = struct {
    call: u64 = 0,
    where: []const u8,
    layer: ?i64 = null,
    arg: []const u8,
    shape: []const i64,
    dtype: []const u8,
    sha256: ?[]const u8 = null,
    head: ?[]const u8 = null, // the first 64 bytes, hex
    file: ?[]const u8 = null,
    key: ?[]const u8 = null, // a keyed fixture's STEP:LAYER:POINT (w-ref's): a dump of this point carries it
};

const Fixtures = struct {
    points: []Point,
    dir: []const u8,

    /// The first line after call `after` at `where` (and `layer`, when given) for argument `arg`.
    fn find(f: *const Fixtures, after: u64, where: []const u8, layer: ?i64, arg: []const u8) ?*const Point {
        for (f.points) |*p| {
            if (p.call <= after or !std.mem.eql(u8, p.where, where) or !std.mem.eql(u8, p.arg, arg)) continue;
            if (layer) |l| if (p.layer == null or p.layer.? != l) continue;
            return p;
        }
        return null;
    }
};

/// A fixture tensor's path: REC/rank<R>/layers/<name> (a recording's), else REC/layers/<name> (both ranks' files in one
/// directory, as w-ref's FP64 fixtures keep them).
fn fixturePath(a: std.mem.Allocator, io: std.Io, rank_dir: []const u8, name: []const u8) ![]const u8 {
    const own = try std.fs.path.join(a, &.{ rank_dir, "layers", name });
    std.Io.Dir.cwd().access(io, own, .{}) catch return std.fs.path.join(a, &.{ std.fs.path.dirname(rank_dir) orelse ".", "layers", name });
    return own;
}

fn dtypeBytes(name: []const u8) usize {
    if (std.mem.eql(u8, name, "float32") or std.mem.eql(u8, name, "int32")) return 4;
    if (std.mem.eql(u8, name, "int64")) return 8;
    return 2;
}

fn numel(shape: []const i64) usize {
    var n: usize = 1;
    for (shape) |s| n *= @intCast(s);
    return n;
}

const Run = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const gpu.Driver,
    s: gpu.Stream,
    fx: *const Fixtures,
    rank: u32,
    out: *std.Io.Writer,
    host: []u8,
    row: []f32 = &.{}, // a logits row on the host (argmax, the sampler)
    ok: bool = true,
    /// The cost profile the recording's depths were chosen with (_choose_k): TF_DS_ROUND_MS / TF_DS_DRAFT_MS, else
    /// GB10's (a recording from the served GB10 lane).
    costs: draft_mod.scheduler_costs.Profile = draft_mod.scheduler_costs.spark,
    /// --dump: every checked point written (points.jsonl and a .bin each).
    dump: ?Dump = null,
    /// --keep-going: a difference is reported, and the run goes on.
    keep_going: bool = false,
    // the dump key's cursors: the step (prompt chunk, then decode round), the layer being run (-1: the model's own
    // points), and the gather being checked (engram, attn, moe, head)
    step: usize = 0,
    layer: i64 = -1,
    phase: []const u8 = "",
    // the served bounded replay: from the decoder's first layer a chunk holds only its rows from `row_off` on (`rows`
    // of them), while a fixture of the whole chunk holds all; points are then taken from row_off on (rows-first
    // tensors, and each rank's part of a gather's [world, rows, ...]) and dumped with row_start = row_off
    row_off: usize = 0,
    rows: usize = 0,

    /// How point `pt` lines up with the engine's rows: null when it does not depend on them (or nothing is cut), else
    /// (parts, fixture rows a part, row bytes): the engine holds rows [row_off, row_off + rows) of each part.
    fn cut(r: *const Run, pt: *const Point) ?struct { lead: usize, parts: usize, fix_rows: usize, row_bytes: usize } {
        if (r.row_off == 0 or pt.shape.len == 0 or pt.shape.len > max_dims) return null;
        const es = dtypeBytes(pt.dtype);
        const gather = std.mem.eql(u8, pt.where, "Comm.gather") and std.mem.eql(u8, pt.arg, "out") and pt.shape.len >= 2;
        const lead: usize = if (gather) 1 else 0;
        const fix_rows: usize = @intCast(pt.shape[lead]);
        if (fix_rows != r.rows + r.row_off) return null;
        return .{ .lead = lead, .parts = if (gather) @intCast(pt.shape[0]) else 1, .fix_rows = fix_rows, .row_bytes = numel(pt.shape[lead + 1 ..]) * es };
    }

    /// `pt` as the engine holds it: after the replay cut, its rows dimension the engine's (in `buf`).
    fn held(r: *const Run, pt: *const Point, buf: *[max_dims]i64) Point {
        const c = r.cut(pt) orelse return pt.*;
        var q = pt.*;
        @memcpy(buf[0..pt.shape.len], pt.shape);
        buf[c.lead] = @intCast(r.rows);
        q.shape = buf[0..pt.shape.len];
        return q;
    }

    /// Where the engine's rows start in the fixture's: row_off for a cut point, else 0.
    fn rowStart(r: *const Run, pt: *const Point) usize {
        return if (r.cut(pt) != null) r.row_off else 0;
    }

    /// The bytes of `pt` the engine's buffer holds (the fixture's own, or its rows from row_off on in each part).
    fn fixtureBytes(r: *Run, pt: *const Point) ![]u8 {
        const name = pt.file orelse return error.NotSavedWhole;
        const data = try std.Io.Dir.cwd().readFileAlloc(r.io, try fixturePath(r.a, r.io, r.fx.dir, name), r.a, .limited(1 << 31));
        if (data.len != numel(pt.shape) * dtypeBytes(pt.dtype)) return error.FixtureSizeDiffers;
        const c = r.cut(pt) orelse return data;
        const out = try r.a.alloc(u8, c.parts * r.rows * c.row_bytes);
        for (0..c.parts) |k| @memcpy(out[k * r.rows * c.row_bytes ..][0 .. r.rows * c.row_bytes], data[(k * c.fix_rows + r.row_off) * c.row_bytes ..][0 .. r.rows * c.row_bytes]);
        r.a.free(data);
        return out;
    }

    /// The key a dumped point carries: STEP:LAYER:WHERE.ARG, a gather's WHERE naming its sublayer.
    fn key(r: *const Run, buf: []u8, pt: *const Point) ![]const u8 {
        if (pt.key) |k| return std.fmt.bufPrint(buf, "{s}", .{k});
        const layer: i64 = pt.layer orelse r.layer;
        if (std.mem.eql(u8, pt.where, "Comm.gather")) return std.fmt.bufPrint(buf, "{d}:{d}:Comm.gather.{s}.{s}", .{ r.step, layer, r.phase, pt.arg });
        return std.fmt.bufPrint(buf, "{d}:{d}:{s}.{s}", .{ r.step, layer, pt.where, pt.arg });
    }

    /// The fixture tensor of `pt` (saved whole; its rows from row_off on after the replay cut) into device memory at
    /// `dev`: `bytes` of them, or an error naming why not.
    fn upload(r: *Run, pt: *const Point, dev: u64, bytes: usize) !void {
        const data = try r.fixtureBytes(pt);
        defer r.a.free(data);
        if (data.len != bytes) return error.FixtureSizeDiffers;
        try r.s.synchronize();
        try gpu.copyH2D(r.d, dev, data);
    }

    /// The host row buffer for n fp32 values (made once, grown when wider).
    fn rowBuf(r: *Run, n: usize) ![]f32 {
        if (r.row.len < n) r.row = try r.a.alloc(f32, n); // the arena: no free
        return r.row[0..n];
    }

    /// Our buffer `dev` against fixture point `p`: the same bytes (sha256), or where they first differ.
    fn check(r: *Run, label: []const u8, p: ?*const Point, dev: u64) !bool {
        const pt = p orelse {
            try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"error\": \"no fixture\"}}\n", .{ r.rank, label });
            try r.out.flush();
            r.ok = false;
            return false;
        };
        try r.s.synchronize();
        const bytes = if (r.cut(pt)) |c| c.parts * r.rows * c.row_bytes else numel(pt.shape) * dtypeBytes(pt.dtype);
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes); // the arena: no free
        try gpu.copyD2H(r.d, r.host[0..bytes], dev);
        return r.finish(label, pt, bytes);
    }

    /// As check, for a point the engine holds as the first columns of wider rows: `src_cols` elements a row on the
    /// device, the fixture's [rows, cols] the first cols of each (the routing's picks and weights beside the shared
    /// expert's slot).
    fn checkCols(r: *Run, label: []const u8, p: ?*const Point, dev: u64, src_cols: usize) !bool {
        const pt = p orelse {
            try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"error\": \"no fixture\"}}\n", .{ r.rank, label });
            try r.out.flush();
            r.ok = false;
            return false;
        };
        if (pt.shape.len != 2) return error.NotRows;
        const rows: usize = if (r.cut(pt) != null) r.rows else @intCast(pt.shape[0]);
        const cols: usize = @intCast(pt.shape[1]);
        const es = dtypeBytes(pt.dtype);
        try r.s.synchronize();
        const wide = try r.a.alloc(u8, rows * src_cols * es);
        defer r.a.free(wide);
        try gpu.copyD2H(r.d, wide, dev);
        const bytes = rows * cols * es;
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes);
        for (0..rows) |i| @memcpy(r.host[i * cols * es ..][0 .. cols * es], wide[i * src_cols * es ..][0 .. cols * es]);
        return r.finish(label, pt, bytes);
    }

    /// The point's `bytes` in r.host against the fixture: reported, dumped, and (with --keep-going) never a stop.
    fn finish(r: *Run, label: []const u8, pt: *const Point, bytes: usize) !bool {
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(r.host[0..bytes], &h, .{});
        var hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&h}) catch unreachable;
        const cutp = r.cut(pt);
        const same = if (cutp != null) blk: {
            const want = r.fixtureBytes(pt) catch break :blk false;
            defer r.a.free(want);
            break :blk std.mem.eql(u8, want, r.host[0..bytes]);
        } else pt.sha256 != null and std.mem.eql(u8, &hex, pt.sha256.?);
        try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"call\": {d}, \"bytes\": {d}, \"equal\": {}", .{ r.rank, label, pt.call, bytes, same });
        if (!same) {
            r.ok = false;
            if (pt.file != null) try r.diff(pt, bytes);
        }
        if (cutp != null) try r.out.print(", \"row_start\": {d}, \"rows\": {d}", .{ r.row_off, r.rows });
        if (r.dump) |*dp| {
            var kb: [160]u8 = undefined;
            const k = try r.key(&kb, pt);
            var sb: [max_dims]i64 = undefined;
            try dp.put(r, k, &r.held(pt, &sb), r.rowStart(pt), r.host[0..bytes], &hex, same);
            try r.out.print(", \"key\": \"{s}\"", .{k});
        }
        try r.out.print("}}\n", .{});
        try r.out.flush();
        return same or r.keep_going;
    }

    /// --dump only: an engine buffer no fixture records (`shape`, its `dtype`), dumped under STEP:LAYER:`name` with
    /// fixture call `call` (the point it sits beside), its rows from the chunk's row_off.
    fn dumpOnly(r: *Run, name: []const u8, call: u64, dev: u64, shape: []const i64, dtype: []const u8) !void {
        const dp = if (r.dump) |*x| x else return;
        const bytes = numel(shape) * dtypeBytes(dtype);
        try r.s.synchronize();
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes); // the arena: no free
        try gpu.copyD2H(r.d, r.host[0..bytes], dev);
        var h: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(r.host[0..bytes], &h, .{});
        var hex: [64]u8 = undefined;
        _ = std.fmt.bufPrint(&hex, "{x}", .{&h}) catch unreachable;
        var kb: [160]u8 = undefined;
        const k = try std.fmt.bufPrint(&kb, "{d}:{d}:{s}", .{ r.step, r.layer, name });
        const pt: Point = .{ .call = call, .where = "Engine.dump", .layer = r.layer, .arg = name, .shape = shape, .dtype = dtype };
        try dp.put(r, k, &pt, r.row_off, r.host[0..bytes], &hex, null);
        try r.out.print("{{\"rank\": {d}, \"point\": \"L{d} {s}\", \"bytes\": {d}, \"key\": \"{s}\", \"row_start\": {d}}}\n", .{ r.rank, r.layer, name, bytes, k, r.row_off });
        try r.out.flush();
    }

    /// The first differing element against the fixture's saved tensor (its rows the engine holds), and how many
    /// elements differ.
    fn diff(r: *Run, pt: *const Point, bytes: usize) !void {
        const want = r.fixtureBytes(pt) catch |err| {
            try r.out.print(", \"diff\": \"{s} unreadable: {s}\"", .{ pt.file orelse "", @errorName(err) });
            return;
        };
        defer r.a.free(want);
        if (want.len != bytes) {
            try r.out.print(", \"diff\": \"{d} bytes saved\"", .{want.len});
            return;
        }
        const esize = dtypeBytes(pt.dtype);
        var first: ?usize = null;
        var count: usize = 0;
        var i: usize = 0;
        while (i < bytes) : (i += esize) {
            if (!std.mem.eql(u8, r.host[i..][0..esize], want[i..][0..esize])) {
                if (first == null) first = i / esize;
                count += 1;
            }
        }
        if (first) |f| {
            var sb: [max_dims]i64 = undefined;
            const row_len = numel(r.held(pt, &sb).shape[1..]);
            try r.out.print(", \"first\": {d}, \"row\": {d}, \"col\": {d}, \"differ\": {d}, \"of\": {d}", .{ f, f / @max(row_len, 1), f % @max(row_len, 1), count, bytes / esize });
            try r.out.print(", \"ours\": \"{x}\", \"served\": \"{x}\"", .{ r.host[f * esize ..][0..esize], want[f * esize ..][0..esize] });
        }
    }

    /// Our buffer `dev` against fixture point `p` (saved whole) on the listed rows only: a pool plane whose other rows
    /// the step does not read (the served pool's rows hold its warm-up's keys).
    fn checkRows(r: *Run, label: []const u8, p: ?*const Point, dev: u64, rows: []const usize) !bool {
        const pt = p orelse {
            try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"error\": \"no fixture\"}}\n", .{ r.rank, label });
            try r.out.flush();
            r.ok = false;
            return false;
        };
        const name = pt.file orelse return error.NotSavedWhole;
        try r.s.synchronize();
        const bytes = numel(pt.shape) * dtypeBytes(pt.dtype);
        const row_bytes = numel(pt.shape[1..]) * dtypeBytes(pt.dtype);
        if (bytes > r.host.len) r.host = try r.a.alloc(u8, bytes); // the arena: no free
        try gpu.copyD2H(r.d, r.host[0..bytes], dev);
        const want = try std.Io.Dir.cwd().readFileAlloc(r.io, try fixturePath(r.a, r.io, r.fx.dir, name), r.a, .limited(1 << 31));
        defer r.a.free(want);
        if (want.len != bytes) return error.SavedSizeDiffers;
        var differ: usize = 0;
        var first: ?usize = null;
        for (rows) |row| {
            const o = row * row_bytes;
            if (o + row_bytes > bytes) return error.RowOutOfRange;
            if (!std.mem.eql(u8, r.host[o..][0..row_bytes], want[o..][0..row_bytes])) {
                differ += 1;
                if (first == null) first = row;
            }
        }
        const same = differ == 0;
        try r.out.print("{{\"rank\": {d}, \"point\": \"{s}\", \"call\": {d}, \"rows\": {d}, \"equal\": {}", .{ r.rank, label, pt.call, rows.len, same });
        if (first) |f| try r.out.print(", \"first_row\": {d}, \"rows_differ\": {d}", .{ f, differ });
        try r.out.print("}}\n", .{});
        try r.out.flush();
        if (!same) r.ok = false;
        return same or r.keep_going;
    }
};

/// --dump DIR: the engine's checked points in the fixtures' own schema (w-ref's grade.py reads it as it reads the
/// fixtures): DIR/rank<R>/layers.jsonl, a line a point (the fixture line's call, where, layer, arg, shape, dtype and key,
/// with this rank's step, the engine's sha256 and whether it equals the fixture's), and DIR/layers/<file>, the buffer's
/// bytes as the engine held them (the fixture's shape, contiguous, its dtype). Files are named r<R>.<key>.bin (':' as '_').
/// After the replay cut a point of the whole chunk's rows is dumped with the engine's rows (its shape's rows dimension)
/// and row_start, the fixture row the first of them is; a dump-only point (no fixture: the router's input) has equal null.
const Dump = struct {
    dir: std.Io.Dir,
    lines: std.Io.File,
    at: u64 = 0,

    fn open(io: std.Io, path: []const u8, rank: u32) !Dump {
        var buf: [32]u8 = undefined;
        const rank_sub = try std.fmt.bufPrint(&buf, "rank{d}", .{rank});
        try std.Io.Dir.cwd().createDirPath(io, path);
        const dir = try std.Io.Dir.cwd().openDir(io, path, .{});
        try dir.createDirPath(io, "layers");
        try dir.createDirPath(io, rank_sub);
        var pb: [48]u8 = undefined;
        return .{ .dir = dir, .lines = try dir.createFile(io, try std.fmt.bufPrint(&pb, "rank{d}/layers.jsonl", .{rank}), .{}) };
    }

    fn close(d: *Dump, io: std.Io) void {
        d.lines.close(io);
        d.dir.close(io);
    }

    fn put(d: *Dump, r: *Run, k: []const u8, pt: *const Point, row_start: usize, bytes: []const u8, sha: *const [64]u8, equal: ?bool) !void {
        var nb: [192]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "r{d}.{s}.bin", .{ r.rank, k });
        std.mem.replaceScalar(u8, name, ':', '_');
        var pb: [208]u8 = undefined;
        var f = try d.dir.createFile(r.io, try std.fmt.bufPrint(&pb, "layers/{s}", .{name}), .{});
        defer f.close(r.io);
        try f.writePositionalAll(r.io, bytes, 0);
        var line: std.ArrayList(u8) = .empty;
        defer line.deinit(r.a);
        try line.print(r.a, "{{\"call\": {d}, \"where\": \"{s}\", \"layer\": ", .{ pt.call, pt.where });
        if (pt.layer) |l| try line.print(r.a, "{d}", .{l}) else try line.appendSlice(r.a, "null");
        try line.print(r.a, ", \"arg\": \"{s}\", \"shape\": [", .{pt.arg});
        for (pt.shape, 0..) |x, i| try line.print(r.a, "{s}{d}", .{ if (i == 0) "" else ", ", x });
        try line.print(r.a, "], \"dtype\": \"{s}\", \"sha256\": \"{s}\", \"file\": \"{s}\", \"key\": \"{s}\", \"step\": {d}, \"rank\": {d}, \"row_start\": {d}, \"equal\": ", .{ pt.dtype, sha, name, k, r.step, r.rank, row_start });
        if (equal) |e| try line.print(r.a, "{}}}\n", .{e}) else try line.appendSlice(r.a, "null}\n");
        try d.lines.writePositionalAll(r.io, line.items, d.at);
        d.at += line.items.len;
    }
};

/// --replay-gather 1: the ranks' collectives from the fixtures, no transport: each all-gather takes the recording's next
/// Comm.gather (call order), checks this rank's partial (its in1, as any point), and writes its out (every rank's part,
/// rank order) into recv. One rank runs alone on one box, exactly as with a peer but for where the peer's half comes from.
const Replay = struct {
    run: ?*Run = null,
    after: u64 = 0, // the last gather consumed

    const vtable: coll.Coll.VTable = .{ .all_gather = gather, .abort = abort, .close = close };

    fn coll_(rp: *Replay, rank: u32, world: u32) coll.Coll {
        return .{ .ptr = rp, .vtable = &vtable, .rank = rank, .world = world, .capturable = false };
    }

    fn gather(ptr: *anyopaque, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) coll.Error!void {
        const rp: *Replay = @ptrCast(@alignCast(ptr));
        rp.step(stream, site, send, recv, bytes) catch |err| {
            std.log.err("replayed gather {s} layer {d}: {s}", .{ @tagName(site.phase), site.layer, @errorName(err) });
            return error.Mismatch;
        };
    }

    fn step(rp: *Replay, stream: gpu.Stream, site: coll.Site, send: u64, recv: u64, bytes: usize) !void {
        const r = rp.run orelse return error.NoRun;
        const out = r.fx.find(rp.after, "Comm.gather", null, "out") orelse return error.NoGatherFixture;
        _ = stream;
        r.phase = switch (site.phase) {
            .prompt_engram, .round_engram => "engram",
            .prompt_attn, .round_attn, .draft_attn => "attn",
            .prompt_moe, .round_moe, .draft_moe => "moe",
            .prompt_head, .round_head => "head",
            else => @tagName(site.phase),
        };
        if (r.fx.find(out.call - 1, "Comm.gather", null, "in1")) |in1| if (in1.call == out.call) {
            var buf: [64]u8 = undefined;
            _ = try r.check(try std.fmt.bufPrint(&buf, "replayed gather {s} L{d} partial", .{ r.phase, site.layer }), in1, send);
        };
        const world: usize = if (r.cut(out)) |c| c.parts else @intCast(@divExact(numel(out.shape) * dtypeBytes(out.dtype), @max(bytes, 1)));
        try r.upload(out, recv, bytes * world);
        rp.after = out.call;
    }

    fn abort(_: *anyopaque, _: coll.Reason) void {}
    fn close(_: *anyopaque) void {}
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 8) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(u32, args[3], 10);
    const world = try std.fmt.parseInt(u32, args[4], 10);
    const ip = try dsv41.link.parseIp(args[5]);
    const port = try std.fmt.parseInt(u16, args[6], 10);
    const rec = args[7];
    var layers: usize = 1;
    var chunks: usize = 1;
    var check_from: usize = 0;
    var rounds: usize = 0;
    var budgets: []usize = &.{}; // --drafts: each drafting request's max_tokens, in admission order (none: no drafter)
    var seeds: []u64 = &.{}; // --seeds: each request's sampling seed, in admission order (none: greedy)
    var smp: sampling.Sampling = .{ .seed = 0 }; // --temperature, --top-k, --top-p: the requests' sampling
    var light = false;
    var use_graphs = false; // --graphs 1: a light run's rounds as captured stretches (round.Graphs)
    var engram_dir: ?[]const u8 = null;
    var token_map: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var rdma_devices: ?[]const u8 = null; // 2D: the decode-size exchanges over RDMA rings (ring2d.zig) on these devices
    var rdma_kernels: ?[]const u8 = null; // their kernels' image (rdma_gather.cu's fatbin)
    var anchor = false;
    var keep_going = false;
    var replay_gather = false;
    var no_replay_cut = false;
    var hip_kit: ?[]const u8 = null;
    var kit_dir: ?[]const u8 = null; // the RoPE tables and engram.json (a fixture set without them: the inputs' kit)
    var ai: usize = 8;
    while (ai + 1 < args.len) : (ai += 2) {
        const key = args[ai];
        const val = args[ai + 1];
        if (std.mem.eql(u8, key, "--layers")) {
            layers = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--chunks")) {
            chunks = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--check-from")) {
            check_from = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--rounds")) {
            rounds = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--drafts")) {
            var list: std.ArrayList(usize) = .empty;
            var it = std.mem.splitScalar(u8, val, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(usize, x, 10));
            budgets = list.items;
        } else if (std.mem.eql(u8, key, "--seeds")) {
            var list: std.ArrayList(u64) = .empty;
            var it = std.mem.splitScalar(u8, val, ',');
            while (it.next()) |x| try list.append(a, try std.fmt.parseInt(u64, x, 10));
            seeds = list.items;
        } else if (std.mem.eql(u8, key, "--temperature")) {
            smp.temperature = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, key, "--top-k")) {
            smp.top_k = try std.fmt.parseInt(usize, val, 10);
        } else if (std.mem.eql(u8, key, "--top-p")) {
            smp.top_p = try std.fmt.parseFloat(f64, val);
        } else if (std.mem.eql(u8, key, "--aio")) {
            aio_on = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--prefetch")) {
            prefetch_on = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--side")) {
            side_on = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--graphs")) {
            use_graphs = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--light")) {
            light = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--engram")) {
            engram_dir = val;
        } else if (std.mem.eql(u8, key, "--token-map")) {
            token_map = val;
        } else if (std.mem.eql(u8, key, "--dump")) {
            dump = val;
        } else if (std.mem.eql(u8, key, "--rdma")) {
            rdma_devices = val;
        } else if (std.mem.eql(u8, key, "--rdma-kernels")) {
            rdma_kernels = val;
        } else if (std.mem.eql(u8, key, "--anchor")) {
            anchor = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--keep-going")) {
            keep_going = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--replay-gather")) {
            replay_gather = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--no-replay-cut")) {
            no_replay_cut = !std.mem.eql(u8, val, "0");
        } else if (std.mem.eql(u8, key, "--hip-kit")) {
            hip_kit = val;
        } else if (std.mem.eql(u8, key, "--kit-dir")) {
            kit_dir = val;
        } else return error.BadArgument;
    }

    var out_buf: [1 << 14]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &out_buf);
    const w_out = &out.interface;

    const cfg = try dsv41.Config.read(a, io, args[1]);
    var driver = try gpu.Driver.open();
    defer driver.close();
    var ctx = try gpu.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try gpu.Stream.init(&driver, true);
    defer stream.deinit();

    // the fixtures first (a bad REC_DIR fails before the weights load)
    const rank_dir = try std.fmt.allocPrint(a, "{s}/rank{d}", .{ rec, rank });
    const jl = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ rank_dir, "layers.jsonl" }), a, .limited(1 << 26));
    var points: std.ArrayList(Point) = .empty;
    var lines = std.mem.tokenizeScalar(u8, jl, '\n');
    while (lines.next()) |line| {
        const p = try std.json.parseFromSliceLeaky(Point, a, line, .{ .ignore_unknown_fields = true });
        try points.append(a, p);
    }
    const fx: Fixtures = .{ .points = points.items, .dir = rank_dir };
    // the prompt's chunks: each Model.forward's ids, and the layers the recording ran on it
    var chunk_call: [max_chunks + 1]u64 = undefined;
    var chunk_ids: [max_chunks][]const i64 = undefined;
    var n_chunks: usize = 0;
    for (fx.points) |p| {
        if (n_chunks == chunks or n_chunks == max_chunks) break;
        if (!std.mem.eql(u8, p.where, "Model.forward") or !std.mem.eql(u8, p.arg, "in2") or p.shape.len != 1) continue;
        const ids_bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try fixturePath(a, io, rank_dir, p.file orelse return error.NoChunkFixture), a, .limited(1 << 20), .@"8", null);
        chunk_call[n_chunks] = p.call;
        chunk_ids[n_chunks] = std.mem.bytesAsSlice(i64, ids_bytes);
        if (chunk_ids[n_chunks].len > chunk_rows) return error.ChunkTooLong;
        n_chunks += 1;
    }
    if (n_chunks < chunks) return error.NoChunkFixture;
    chunk_call[n_chunks] = std.math.maxInt(u64);
    var total: usize = 0;
    for (chunk_ids[0..n_chunks]) |ids| total += ids.len;
    // the recorded prompt's length (its chunks' rows: every chunk before the first round, a recording of several
    // requests holding later prompts after it): the bounded replay starts a window before its end
    var prompt_len: usize = 0;
    for (fx.points) |p| {
        if (std.mem.eql(u8, p.where, "RoundDecoder.run")) break;
        if (std.mem.eql(u8, p.where, "Model.forward") and std.mem.eql(u8, p.arg, "in2") and p.shape.len == 1) prompt_len += @intCast(p.shape[0]);
    }
    const replay_rows: usize = if (no_replay_cut) 0 else prompt_len -| cfg.window;

    // HIP: the AMD kit before the weights (a refused kit fails in seconds): every kernel this configuration reaches
    // resolved and checked against its metadata
    var kit: HipKit = undefined;
    if (gpu.backend == .hip) {
        const caps = try ctx.caps();
        kit = try gpu.Kit.open(gpa, io, &driver, hip_kit orelse return error.NoHipKit, caps.archName(), .{ .drafts = budgets.len > 0, .engram = cfg.engram_layers.slice().len > 0 });
    }
    defer if (gpu.backend == .hip) kit.deinit();
    // HIP: which Triton launches take w-tri's kernels (TF_DS_AUX, as the server reads it: the A/B of the two paths)
    const aux_which: dsv41.tri_aux.Which = if (gpu.backend == .hip) try dsv41.tri_aux.Which.fromEnv() else .triton;
    var aux_name: [32]u8 = undefined;
    var aux: dsv41.tri_aux.Aux = .{};

    // the weights (the DSpark blocks with --drafts) from the lane's rank cache
    const t0 = std.Io.Timestamp.now(io, .awake);
    var cache_dir = try std.Io.Dir.cwd().openDir(io, args[2], .{ .iterate = true });
    defer cache_dir.close(io);
    const cache_name = try dsv41.rank_cache.find(a, io, cache_dir, rank, world);
    var ix = try dsv41.rank_cache.open(gpa, io, cache_dir, cache_name);
    defer ix.deinit();
    var cache_file = try cache_dir.openFile(io, cache_name, .{});
    defer cache_file.close(io);
    const sp = try dsv41.prompt2d.split(rank, world); // world 4: the 2D split (prompt2d.zig)
    var w = try dsv41.weights.load(gpa, &driver, .{ .cache = .{ .file = cache_file, .index = &ix, .io = io } }, cfg, sp, budgets.len > 0);
    defer w.deinit();
    const load_s = @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;

    // the collectives: replayed from the fixtures (no links), else over the links: NCCL (CUDA), or w-link's
    // host-staged TCP (HIP, tcp_coll.zig) after the peers' handshake on coll.Identity (the kit, and this gate's run
    // settings: both ranks must run the same gate)
    var replay: Replay = .{};
    var fan: ?dsv41.fan.Fan = null;
    defer if (fan) |*f| f.close();
    var nccl: Nccl = undefined;
    var coll_: coll.Coll = undefined;
    if (replay_gather) {
        coll_ = replay.coll_(rank, world);
    } else if (gpu.backend == .cuda) {
        fan = try dsv41.fan.Fan.open(io, ip, port, rank, world);
        nccl = try fan.?.comm(rank, world);
        coll_ = nccl.coll(); // the engine's collectives (coll.zig) over this NCCL communicator
    } else {
        fan = try dsv41.fan.Fan.open(io, ip, port, rank, world);
        var plan: [256]u8 = undefined;
        const id: coll.Identity = .{
            .rank = rank,
            .world = world,
            .kit_abi = gpu.kit.abi_version,
            .kit_sha256 = kit.manifest_sha256,
            .modalities = 0, // text: the gate runs no vision tower
            .plan = try std.fmt.bufPrint(&plan, "gate=tf-dsv41-layer layers={d} chunks={d} check_from={d} rounds={d} drafts={d} anchor={} light={} replay_cut={} aux={s}", .{ layers, chunks, check_from, rounds, budgets.len, anchor, light, !no_replay_cut, aux_which.name(&aux_name) }),
        };
        coll_ = try dsv41.tcp_coll.open(gpa, io, &ctx, kit, &fan.?, &id);
    }
    defer if (gpu.backend == .cuda and !replay_gather) nccl.deinit();
    defer if (gpu.backend == .hip and !replay_gather) coll_.close(); // (before the links: defers run in reverse)

    // kernels: CUDA the recording's Triton set, the served extension cubins and the torch-op images; HIP the AMD kit's
    // (every one the configuration reaches resolved and checked against its metadata, every EXL3 specialization the
    // weights reach present)
    var set: gpu.aot.Set = undefined;
    var pf: dsv41.exl3_prefill.Kernels = undefined;
    var lg: dsv41.exl3_linear.Kernels = undefined;
    var exk: dsv41.exl3_experts.Kernels = undefined;
    var ops: dsv41.ops.Ops = undefined;
    var ex: dsv41.exact.Exact = undefined;
    if (gpu.backend == .cuda) {
        set = try gpu.aot.Set.load(gpa, io, &driver, ctx.device, try std.fs.path.join(a, &.{ rec, "aot" }));
        const lin = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "linear.cubin" }), a, .limited(1 << 28), .@"16", null);
        pf = try dsv41.exl3_prefill.Kernels.load(&driver, lin);
        const lgc = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "linear_grouped.cubin" }), a, .limited(1 << 28), .@"16", null);
        lg = try dsv41.exl3_linear.Kernels.load(&driver, lgc);
        const exb = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts.cubin" }), a, .limited(1 << 28), .@"16", null);
        const exc = try std.Io.Dir.cwd().readFileAllocOptions(io, try std.fs.path.join(a, &.{ rec, "cubins", "experts_cb.cubin" }), a, .limited(1 << 28), .@"16", null);
        exk = try dsv41.exl3_experts.Kernels.load(&driver, exb, exc, dsv41.exl3_linear.codebook_mul1);
        if (!gpu.kernels.available) return error.NoKernelImages;
        ops = try dsv41.ops.Ops.load(&driver, gpu.kernels.torch_pointwise, gpu.kernels.torch_movement, gpu.kernels.dsv41_ops);
        ex = try dsv41.exact.Exact.load(&driver, gpu.kernels.dsv41_torch);
    } else {
        const caps = try ctx.caps();
        set = try gpu.aot.Set.fromKit(gpa, kit);
        ops = try dsv41.hip_load.ops(kit, caps.wall_clock_hz);
        ex = try dsv41.hip_load.exact(kit);
        pf = try dsv41.hip_load.exl3Prefill(kit);
        lg = try dsv41.hip_load.exl3Linear(kit);
        exk = try dsv41.hip_load.exl3Experts(kit);
        try dsv41.hip_load.coverage(&w, &pf, &lg, &exk);
        aux = try dsv41.hip_load.aux(kit, aux_which);
        try w_out.print("{{\"rank\": {d}, \"aux\": \"{s}\"}}\n", .{ rank, aux_which.name(&aux_name) });
        try w_out.flush();
    }
    defer {
        ex.unload();
        ops.unload();
        exk.unload();
        lg.unload();
        pf.unload();
        set.deinit();
    }

    // the caches' positions: the deepest round's bucket and the widest extent the recording reaches (every row's: a
    // concurrent round's rows are several streams'); the deepest position any row or prompt reaches
    var tokens: usize = cache_tokens;
    var deepest: usize = 0;
    var filled: usize = 0; // the current prompt's positions so far (its chunks, until a round)
    for (fx.points) |*q| {
        if (std.mem.eql(u8, q.where, "Model.forward") and std.mem.eql(u8, q.arg, "in2") and q.shape.len == 1) {
            filled += @intCast(q.shape[0]);
            deepest = @max(deepest, filled);
            continue;
        }
        if (!std.mem.eql(u8, q.where, "RoundDecoder.run")) continue;
        filled = 0;
        const in2 = std.mem.eql(u8, q.arg, "in2");
        if (!in2 and !std.mem.eql(u8, q.arg, "in5")) continue;
        var most: i64 = headInt(q) catch 0;
        if (q.file) |name| {
            const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, try fixturePath(a, io, rank_dir, name), a, .limited(1 << 20), .@"8", null);
            for (std.mem.bytesAsSlice(i64, bytes)) |v| most = @max(most, v);
        }
        const m: usize = @intCast(@max(most, 0));
        if (in2) deepest = @max(deepest, m + round_mod.max_rows);
        tokens = @max(tokens, if (in2) round_mod.bucketFor(m + 1, pool_window) else m);
    }
    // the RoPE tables' rows every position reaches (the fixtures hold the served tables' first rows)
    const rope_rows = @max(@min(tokens, deepest + 1), chunk_rows);
    var rope_buf = try gpu.DeviceBuffer.alloc(&driver, 4 * rope_rows * (cfg.rope_dim / 2) * 4);
    defer rope_buf.free();
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }, 0..) |name, j| {
        var f = try std.Io.Dir.cwd().openFile(io, try std.fs.path.join(a, &.{ kit_dir orelse rec, name }), .{});
        defer f.close(io);
        const part = try a.alloc(u8, rope_rows * (cfg.rope_dim / 2) * 4);
        if (try f.readPositionalAll(io, part, 0) != part.len) return error.ShortRopeTable; // (a fixture of fewer rows)
        try rope_buf.upload(j * part.len, part);
        a.free(part);
    }
    const tbl = rope_rows * (cfg.rope_dim / 2) * 4;
    const rope: prompt.Rope = .{ .cos = rope_buf.ptr, .sin = rope_buf.ptr + tbl };
    const rope_c: prompt.Rope = .{ .cos = rope_buf.ptr + 2 * tbl, .sin = rope_buf.ptr + 3 * tbl };

    var arena = try prompt.Arena.init(&driver, 5 << 30); // (a 1M pool's caches, the candidate pool's buffers)
    defer arena.deinit();
    var blas_ws_ptr: u64 = undefined;
    var two: ?Two = null;
    if (gpu.features.world4 and sp.pair != null) two = try Two.init(&cfg, &w, &arena, sp, chunk_rows);
    // 2D with --rdma: the rings (their infos go round over NCCL world 4)
    var rdma_mod: ?gpu.Module = null;
    defer if (rdma_mod) |*m| m.unload();
    var rdma_rings: ?Rings2d = null;
    defer if (gpu.features.world4 and gpu.features.verbs_rdma) if (rdma_rings) |*r| r.close(gpa);
    if (gpu.features.world4 and gpu.features.verbs_rdma and two != null and rdma_devices != null) {
        const image = try std.Io.Dir.cwd().readFileAllocOptions(io, rdma_kernels orelse return error.NoRdmaKernels, a, .limited(1 << 26), .@"16", null);
        rdma_mod = try gpu.Module.load(&driver, image);
        var devices: std.ArrayList([]const u8) = .empty;
        var dit = std.mem.splitScalar(u8, rdma_devices.?, ',');
        while (dit.next()) |d| try devices.append(a, d);
        rdma_rings = try Rings2d.open(gpa, &driver, try dsv41.rdma.Kernels.load(rdma_mod.?), devices.items, rank, .{ .max_bytes = 4456448, .gid_index = 5 }, &nccl, stream);
        two.?.rings = &rdma_rings.?;
    }
    // HIP: host bytes reach the device through a pinned staging ring (prompt.upload)
    var staging: ?gpu.Staging = if (gpu.backend == .hip) try gpu.Staging.init(&driver, 4 << 20) else null;
    defer if (staging) |*x| x.deinit();
    var eng: prompt.Engine = .{ .d = &driver, .s = stream, .t = .{ .set = &set, .stream = stream, .aux = if (gpu.backend == .hip) &aux else null }, .blas = undefined, .comm = &coll_, .pf = &pf, .lin = &lg, .ex = &exk, .ops = &ops, .exact = &ex, .c = &cfg, .w = &w, .world = sp.world, .plain = rope, .compressed = rope_c, .two = if (gpu.features.world4) (if (two) |*t| t else null) else null, .staging = if (staging) |*x| x else null };
    // TP2 with --rdma: the decode gathers over the RDMA ring (the served lane's), their bytes checked as NCCL's were
    var tp2_mod: ?gpu.Module = null;
    defer if (tp2_mod) |*mm| mm.unload();
    var tp2_ring: ?*prompt.Ring = null;
    defer if (gpu.features.verbs_rdma) if (tp2_ring) |r| r.destroy(gpa);
    if (gpu.features.verbs_rdma and two == null and rdma_devices != null) {
        tp2_mod = try gpu.Module.load(&driver, gpu.kernels.dsv41_rdma);
        var devices: std.ArrayList([]const u8) = .empty;
        var dit = std.mem.splitScalar(u8, rdma_devices.?, ',');
        while (dit.next()) |dv| try devices.append(a, dv);
        tp2_ring = try dsv41.model.Model.openRing(gpa, &driver, try dsv41.rdma.Kernels.load(tp2_mod.?), devices.items, rank, &coll_, stream);
        eng.ring = tp2_ring;
    }
    var ch = try prompt.Chunk.init(&eng, &arena, chunk_rows, tokens);
    // a drafting recording's pool: MultiDecoder's slots (each stream's window rings and positional stores)
    const slots: usize = if (budgets.len > 0) draft_mod.default_streams else 1;
    const caches = try prompt.Caches.init(&eng, &arena, tokens, slots);
    blas_ws_ptr = ch.blas_ws;
    var blas = try dsv41.blas.Blas.open(stream, blas_ws_ptr);
    defer blas.close();
    eng.blas = &blas;
    const rings = try a.alloc(u64, cfg.layers);
    for (rings) |*r| {
        r.* = try arena.take(slots * eng.ringBytes());
        try gpu.fill8Async(&driver, stream.handle, r.*, 0, slots * eng.ringBytes());
    }
    try stream.synchronize();
    try ctx.synchronize(); // the setup's legacy-stream copies too

    try w_out.print("{{\"rank\": {d}, \"loaded_s\": {d:.1}, \"weights_bytes\": {d}, \"chunks\": {d}, \"rows\": {d}, \"arena_bytes\": {d}}}\n", .{ rank, load_s, w.bytes, n_chunks, total, arena.used });
    try w_out.flush();

    // Engram on the host: the compressed token map, the recording's multipliers, the tables and 64 readers
    var eh: ?prompt.EngramHost = null;
    var tables: dsv41.engram_io.Tables = undefined;
    var pool: ?*dsv41.engram_io.Pool = null;
    defer if (pool) |p| p.deinit(gpa);
    defer if (eh != null) tables.close();
    var aio: ?*dsv41.engram_aio.Aio = null;
    defer if (aio) |x| x.deinit();
    if (engram_dir) |edir| {
        const map_text = try std.Io.Dir.cwd().readFileAlloc(io, token_map orelse return error.NoTokenMap, a, .limited(1 << 26));
        const map = try std.json.parseFromSliceLeaky([]i32, a, map_text, .{});
        const EngramJson = struct { multipliers: [][]i64 };
        const ej_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ kit_dir orelse rec, "engram.json" }), a, .limited(1 << 26));
        const ej = try std.json.parseFromSliceLeaky(EngramJson, a, ej_text, .{ .ignore_unknown_fields = true });
        var mult: [dsv41.engram.max_layers][dsv41.engram.max_ngram]i64 = @splat(@splat(0));
        for (ej.multipliers, 0..) |row, l| for (row, 0..) |v, k| {
            mult[l][k] = v;
        };
        tables = try dsv41.engram_io.Tables.open(gpa, io, edir);
        pool = try dsv41.engram_io.Pool.init(gpa, io, 64);
        eh = try prompt.EngramHost.init(a, &cfg, dsv41.engram.Hasher.init(cfg, map, mult), &tables, pool.?, sp.rank, sp.world, chunk_rows);
        if (aio_on) {
            aio = try dsv41.engram_aio.Aio.init(gpa, io, 1024);
            eh.?.aio = aio;
        }
    }
    // the whole sequence's ids (Engram hashes n-grams across chunk boundaries)
    const seq = try a.alloc(i32, total + rounds * round_mod.max_rows);
    {
        var at: usize = 0;
        for (chunk_ids[0..n_chunks]) |ids| for (ids) |id| {
            seq[at] = @intCast(id);
            at += 1;
        };
    }

    var run: Run = .{ .a = a, .io = io, .d = &driver, .s = stream, .fx = &fx, .rank = rank, .out = w_out, .host = try a.alloc(u8, 2 * chunk_rows * cfg.hidden * 4), .keep_going = keep_going };
    if (dump) |dir| run.dump = try Dump.open(io, dir, rank);
    defer if (run.dump) |*dp| dp.close(io);
    replay.run = &run;
    if (std.c.getenv("TF_DS_ROUND_MS")) |v| run.costs.round_ms = try draft_mod.scheduler_costs.parseRounds(a, std.mem.span(v));
    if (std.c.getenv("TF_DS_DRAFT_MS")) |v| run.costs.draft_ms = try draft_mod.scheduler_costs.parseDraft(std.mem.span(v));
    try run.costs.validate();
    if (light and budgets.len > 0) {
        const ok = try runDrafted(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, a, &arena, budgets, seeds, smp, sp);
        try w_out.print("{{\"rank\": {d}, \"light\": true, \"drafted\": true, \"all_equal\": {}}}\n", .{ rank, ok });
        try w_out.flush();
        return if (ok) 0 else 1;
    }
    if (light) {
        var gs = round_mod.Graphs.init(gpa);
        defer gs.deinit();
        const ok = try runLight(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, a, &arena, rounds, tokens, if (use_graphs) &gs else null);
        if (use_graphs) {
            try w_out.print("{{\"rank\": {d}, \"graphs\": {d}}}\n", .{ rank, gs.count() });
            try w_out.flush();
        }
        try w_out.print("{{\"rank\": {d}, \"light\": true, \"all_equal\": {}}}\n", .{ rank, ok });
        try w_out.flush();
        return if (ok) 0 else 1;
    }
    const host_pos = try a.alloc(i64, chunk_rows);
    var start: usize = 0;
    var ran: usize = 0;
    for (chunk_ids[0..n_chunks], 0..) |ids, ci| {
        // the layers the recording ran on this chunk (its attention points before the next chunk)
        var rec_layers: usize = 0;
        for (fx.points) |p| {
            if (p.call <= chunk_call[ci] or p.call >= chunk_call[ci + 1] or !std.mem.eql(u8, p.where, "Model.attention_k")) continue;
            if (p.layer) |l| rec_layers = @max(rec_layers, @as(usize, @intCast(l)) + 1);
        }
        const n_layers = @min(layers, cfg.layers);
        try w_out.print("{{\"rank\": {d}, \"chunk\": {d}, \"call\": {d}, \"start\": {d}, \"rows\": {d}, \"replay\": {d}, \"layers\": {d}, \"recorded_layers\": {d}, \"checked\": {}}}\n", .{ rank, ci, chunk_call[ci], start, ids.len, replay_rows, n_layers, rec_layers, ci >= check_from });
        try w_out.flush();
        var shared: prompt.Shared = .{};
        try prompt.begin(&eng, &ch, ids, start, host_pos);
        run.step = ci;
        if (!try runChunk(&run, &eng, &ch, &caches, &shared, rings, if (eh) |*x| x else null, seq[0 .. start + ids.len], chunk_call[ci], n_layers, ci >= check_from, dump, replay_rows, host_pos, anchor)) {
            run.ok = false;
            break;
        }
        start += ids.len;
        ran += 1;
    }
    // (--keep-going: the rounds run after a prompt that differed, so a reference's every step is graded)
    if (rounds > 0 and (run.ok or keep_going) and ran == n_chunks) {
        if (n_chunks != chunks or start != prompt_len) return error.RoundsNeedTheWholePrompt;
        run.ok = try runRounds(&run, &eng, &ch, &caches, rings, if (eh) |*x| x else null, seq, start, rounds, a, &arena, if (budgets.len > 0) budgets[0] else 0, sp, n_chunks, anchor);
    }
    try w_out.print("{{\"rank\": {d}, \"chunks\": {d}, \"layers\": {d}, \"rounds\": {d}, \"all_equal\": {}}}\n", .{ rank, ran, layers, rounds, run.ok });
    try w_out.flush();
    return if (run.ok) 0 else 1;
}

/// The recorded rounds' checker: each point found after the last one matched, as the recorder numbered the calls.
const RoundCheck = struct {
    run: *Run,
    eng: *const prompt.Engine,
    after: u64,
    att: u64 = 0,
    moe: u64 = 0,
    call: u64, // the round's RoundDecoder.run
    fwd: u64, // its RoundRunner.forward
    rows: usize = 0, // the round's rows (--anchor's)
    cfg: ?*const dsv41.Config = null, // set: --anchor

    fn at(ctx: *anyopaque, what: round_mod.Point, layer: usize, dev: u64) anyerror!bool {
        const rc: *RoundCheck = @ptrCast(@alignCast(ctx));
        const fx = rc.run.fx;
        const l: i64 = @intCast(layer);
        rc.run.layer = switch (what) {
            .head_cols, .head_gather, .logits, .taps => -1,
            else => l,
        };
        rc.run.phase = switch (what) {
            .engram_proj, .engram_gather => "engram",
            .attn_gather => "attn",
            .moe_gather => "moe",
            .head_cols, .head_gather => "head",
            else => rc.run.phase,
        };
        var buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "R{d} L{d} {s}", .{ rc.call, layer, @tagName(what) });
        switch (what) {
            // the router's input: dumped only (no fixture records it)
            .route_in => {
                if (rc.run.dump == null) return true;
                const c = rc.eng.c;
                const kc: i64 = @intCast(dsv41.tri_norm.gateParts(rc.rows, c.hidden, c.experts));
                const n: i64 = @intCast(rc.rows);
                const e: i64 = @intCast(c.experts);
                try rc.run.dumpOnly("route_logits", rc.moe, dev, if (kc > 0) &.{ n, kc, e } else &.{ n, e }, "float32");
                return true;
            },
            // the routing, when the fixtures record this round's (w-ref's Ref.route): the slots beside the shared expert's
            .route_pick, .route_wts => {
                const q = fx.find(rc.att, "Ref.route", l, if (what == .route_pick) "pick" else "wts") orelse return true;
                if (fx.find(rc.call, "RoundDecoder.run", null, "in1")) |next| if (q.call > next.call) return true;
                return rc.run.checkCols(label, q, dev, rc.eng.slots());
            },
            else => {},
        }
        const pt: ?*const Point = switch (what) {
            .engram_proj => fx.find(rc.after, "Comm.gather", null, "in1"),
            .engram_gather => fx.find(rc.after, "Comm.gather", null, "out"),
            .attn_in => fx.find(rc.after, "RoundDecoder._attention2", l, "in2"),
            .attn_out => fx.find(rc.after, "RoundDecoder._attention2", l, "out"),
            .attn_gather => fx.find(rc.att, "Comm.gather", null, "out"),
            .moe_in => fx.find(rc.att, "Model.moe", l, "in2"),
            .moe_out => fx.find(rc.att, "Model.moe", l, "out"),
            .moe_gather => fx.find(rc.moe, "Comm.gather", null, "out"),
            .head_cols => fx.find(rc.after, "Comm.gather", null, "in1"),
            .head_gather => fx.find(rc.after, "Comm.gather", null, "out"),
            .logits => fx.find(rc.call - 1, "RoundDecoder.run", null, "out"),
            .taps => fx.find(rc.fwd - 1, "RoundRunner.forward", null, "out.1"),
            .route_in, .route_pick, .route_wts => unreachable,
        };
        const ok = try rc.run.check(label, pt, dev);
        if (pt) |q| switch (what) {
            .engram_gather => rc.after = q.call,
            .attn_in => rc.att = q.call,
            .moe_in => rc.moe = q.call,
            .moe_gather => rc.after = q.call,
            else => {},
        };
        return ok;
    }

    /// --anchor: round layer `layer` starts from the fixture's Ref.layer_in (the round's rows).
    fn put(ctx: *anyopaque, name: []const u8, layer: usize, dev: u64, shape: []const i64, dtype: []const u8) anyerror!void {
        const rc: *RoundCheck = @ptrCast(@alignCast(ctx));
        if (rc.run.dump == null) return;
        rc.run.layer = @intCast(layer);
        try rc.run.dumpOnly(name, rc.att, dev, shape, dtype);
    }

    fn anchor(ctx: *anyopaque, layer: usize, h: u64, pre: u64) anyerror!bool {
        const rc: *RoundCheck = @ptrCast(@alignCast(ctx));
        rc.run.layer = @intCast(layer);
        try anchorLayer(rc.run, rc.call, layer, h, pre, rc.rows, rc.cfg orelse return false);
        return true;
    }
};

/// A fixture tensor of int64 values, whole (a round's input rows).
fn ints(run: *Run, p: ?*const Point) ![]const i64 {
    const pt = p orelse return error.NoRoundFixture;
    const name = pt.file orelse return error.NoRoundFixture;
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(run.io, try fixturePath(run.a, run.io, run.fx.dir, name), run.a, .limited(1 << 20), .@"8", null);
    return std.mem.bytesAsSlice(i64, bytes);
}

/// The logits' width: the ranks' vocabulary parts (a 2D rank's part is both pairs' quarters).
fn vocabOf(eng: *const prompt.Engine) usize {
    return eng.world * (if (prompt.two2d(eng)) |t| t.hw[0] + t.hw[1] else eng.w.head.n);
}

/// The index of the first largest of n fp32 values on the device (torch.argmax's choice).
fn argmax(run: *Run, dev: u64, n: usize) !usize {
    try run.s.synchronize();
    const host = try run.rowBuf(n);
    try gpu.copyD2H(run.d, std.mem.sliceAsBytes(host), dev);
    var best: usize = 0;
    for (host, 0..) |v, i| {
        if (v > host[best]) best = i;
    }
    return best;
}

/// The recorded drafter pass's checker: its points after the pass's BatchDraftGraph.run call, stage j's carrying layer
/// `layers` + j, each gather the first after its stage's attention or MoE call.
const DraftCheck = struct {
    run: *Run,
    call: u64, // the pass's BatchDraftGraph.run
    layers: usize,
    after: u64,
    vis: []const usize, // the plane rows the pass reads: each stream's window positions in its slot
    att: u64 = 0,
    moe: u64 = 0,
    rows: usize = 0, // the pass's rows (--anchor's): streams x the drafter's block
    cfg: ?*const dsv41.Config = null, // set: --anchor

    /// --anchor: stage `stage` (fixture layer layers + stage) starts from the fixture's Ref.layer_in.
    fn anchor(ctx: *anyopaque, stage: usize, h: u64, pre: u64) anyerror!bool {
        const dc: *DraftCheck = @ptrCast(@alignCast(ctx));
        dc.run.layer = @intCast(dc.layers + stage);
        try anchorLayer(dc.run, dc.call, dc.layers + stage, h, pre, dc.rows, dc.cfg orelse return false);
        return true;
    }

    fn at(ctx: *anyopaque, what: draft_mod.Point, stage: usize, dev: u64) anyerror!bool {
        const dc: *DraftCheck = @ptrCast(@alignCast(ctx));
        const fx = dc.run.fx;
        const l: i64 = @intCast(dc.layers + stage);
        dc.run.layer = l;
        dc.run.phase = switch (what) {
            .attn_gather => "attn",
            .moe_gather => "moe",
            else => dc.run.phase,
        };
        var buf: [64]u8 = undefined;
        const label = try std.fmt.bufPrint(&buf, "D{d} S{d} {s}", .{ dc.call, stage, @tagName(what) });
        const pt: ?*const Point = switch (what) {
            .attn_in => fx.find(dc.after, "BatchDraftGraph._attention", l, "in2"),
            .attn_ring => fx.find(dc.after, "BatchDraftGraph._attention", l, "in3"),
            .attn_out => fx.find(dc.after, "BatchDraftGraph._attention", l, "out"),
            .attn_gather => fx.find(dc.att, "Comm.gather", null, "out"),
            .moe_in => fx.find(dc.att, "Model.moe", l, "in2"),
            .moe_out => fx.find(dc.att, "Model.moe", l, "out"),
            .moe_gather => fx.find(dc.moe, "Comm.gather", null, "out"),
            .stages_h => fx.find(dc.call, "BatchDraftGraph._stages_fused", null, "out.0"),
            .stages_pre => fx.find(dc.call, "BatchDraftGraph._stages_fused", null, "out.1"),
            .local => fx.find(dc.call, "Markov.steps", null, "in1"),
            .markov => fx.find(dc.call, "Markov.steps", null, "post.in2"),
        };
        const ok = if (what == .attn_ring) try dc.run.checkRows(label, pt, dev, dc.vis) else try dc.run.check(label, pt, dev);
        if (pt) |q| switch (what) {
            .attn_in => dc.att = q.call,
            .moe_in => dc.moe = q.call,
            .moe_gather => dc.after = q.call,
            else => {},
        };
        return ok;
    }
};

/// sample_rows of one row of n fp32 logits on the device at absolute position `position` (sampling.zig).
fn sampleDev(run: *Run, dev: u64, n: usize, position: usize, s: sampling.Sampling) !i64 {
    if (s.temperature <= 0) return @intCast(try argmax(run, dev, n));
    try run.s.synchronize();
    const host = try run.rowBuf(n);
    try gpu.copyD2H(run.d, std.mem.sliceAsBytes(host), dev);
    return sampling.sampleRow(host, position, s);
}

/// The recorded decode rounds after the prompt: each RoundDecoder.run's rows (ids, positions, the stream's extent) through
/// round.zig with every point checked. A round's rows are its window: the pending token and the drafts it verifies (one
/// row a round when serial). The rows' greedy tokens (their targets) decide the next round as the served round keeps
/// them (multi.py _round): the drafts while each equals its target, then the target after them; the next round starts
/// at the kept position with that target, its rows overwriting the rejected ones (in the Engram sequence too). Only the
/// recorded prompt's request: a later prompt ends the rounds. `drafts` (the request's max_tokens; 0: none): the
/// drafter around each round, its pass checked at every recorded point, its drafts and served depth against the
/// round's window, the window's taps absorbed after it. False on the first difference.
fn runRounds(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, seq: []i32, prompt_len: usize, rounds: usize, a: std.mem.Allocator, arena: *prompt.Arena, drafts: usize, sp: dsv41.plan.Split, first_step: usize, anchor: bool) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const vocab = vocabOf(eng);
    const R_max = round_mod.default_rows; // the recordings' round bound
    var rd = try round_mod.Round.init(eng, arena, a, ch.max_comp);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    var targets: [R_max]usize = undefined;
    targets[0] = try argmax(run, ch.head_g, vocab); // greedy from the prompt's logits
    var prev_ids: [R_max]i64 = undefined;
    var prev_n: usize = 1; // the prompt: its last row's target only
    var prev_pos: i64 = @as(i64, @intCast(prompt_len)) - 1;
    // from the prompt's last chunk (the latest Model.forward before the first round): a later one is another request's
    var after: u64 = 0;
    if (fx.find(0, "RoundDecoder.run", null, "in1")) |first| {
        for (fx.points) |*q| {
            if (q.call < first.call and isPoint(q, "Model.forward", "in2")) after = q.call;
        }
    }
    // the drafter: its pool (MultiDecoder's slots), the prompt's taps (the replayed rows, the tap layers' rows side by
    // side) absorbed into slot 0's rings from the taps' first position (Drafter.absorb after the prompt's last chunk)
    var pool: draft_mod.Pool = undefined;
    var dr: draft_mod.Drafter = undefined;
    if (drafts > 0) {
        pool = try draft_mod.Pool.init(eng, arena, draft_mod.default_streams);
        dr = try draft_mod.Drafter.init(eng, arena, sp);
        const d = c.hidden;
        for (0..c.dspark_taps.slice().len) |j| try eng.ops.copyRows(eng.s, prompt.tapRows(eng, ch, j), d * 2, dr.at + j * d * 2, dr.taps_w * 2, d * 2, ch.n);
        if (fx.find(after, "Drafter.absorb", null, "in3")) |ap| {
            if (!try run.check("prompt absorb taps", ap, dr.at)) return false;
        }
        dr.absorb(eng, ch, &pool, 0, dr.at, ch.n, ch.start) catch |err| {
            try w_out.print("{{\"rank\": {d}, \"absorb\": \"prompt\", \"error\": \"{s}\"}}\n", .{ run.rank, @errorName(err) });
            try w_out.flush();
            return false;
        };
    }
    for (0..rounds) |k| {
        run.step = first_step + k;
        const rp = fx.find(after, "RoundDecoder.run", null, "in1") orelse {
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"note\": \"no more recorded rounds\"}}\n", .{ run.rank, k });
            try w_out.flush();
            break;
        };
        if (fx.find(after, "Model.forward", null, "in2")) |mf| if (mf.call < rp.call) {
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"note\": \"a later request's prompt: its rounds are not this prompt's\"}}\n", .{ run.rank, k });
            try w_out.flush();
            break;
        };
        const ids = try ints(run, rp);
        const pos = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in2"));
        const base = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in4"));
        const end = try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in5"));
        if (ids.len == 0 or ids.len > R_max or pos.len != ids.len) return error.BadRound;
        // the previous round's acceptance: its kept drafts equal their targets, its first rejected one does not, and
        // this round's first row is the target after the kept ones
        const kept = pos[0] - prev_pos - 1;
        var accept_ok = kept >= 0 and kept < prev_n;
        if (accept_ok) {
            const ka: usize = @intCast(kept);
            for (0..ka) |j| {
                if (prev_ids[j + 1] != @as(i64, @intCast(targets[j]))) accept_ok = false;
            }
            if (ka + 1 < prev_n and prev_ids[ka + 1] == @as(i64, @intCast(targets[ka]))) accept_ok = false;
            if (ids[0] != @as(i64, @intCast(targets[ka]))) accept_ok = false;
        }
        try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"call\": {d}, \"rows\": {d}, \"pos\": {d}, \"id\": {d}, \"kept\": {d}, \"accept_equal\": {}}}\n", .{ run.rank, k, rp.call, ids.len, pos[0], ids[0], kept, accept_ok });
        try w_out.flush();
        if (!accept_ok) run.ok = false;
        if (drafts > 0) {
            // the round's drafter pass (its BatchDraftGraph.run, before the round's forward)
            const bp = fx.find(after, "BatchDraftGraph.run", null, "in1") orelse return error.NoDraftFixture;
            if (bp.call > rp.call) return error.NoDraftFixture;
            const tok = try ints(run, bp);
            const q0 = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in2"));
            const sl = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in3"));
            const most = 5; // the engine's drafts (--mtp-drafts 5)
            const steps = draft_mod.passSteps(tok.len, pool.slots, most, R_max, dr.n);
            // the plane rows the pass reads: each stream's window (its 128 positions before q0) in its slot
            var vis: [draft_mod.max_streams * 128]usize = undefined;
            var nv: usize = 0;
            for (q0, sl) |q, slot| {
                var p: i64 = @max(0, q - @as(i64, @intCast(c.window)));
                while (p < q) : (p += 1) {
                    vis[nv] = @as(usize, @intCast(slot)) * pool.ring + @as(usize, @intCast(p)) % pool.ring;
                    nv += 1;
                }
            }
            var dc: DraftCheck = .{ .run = run, .call = bp.call, .layers = c.layers, .after = bp.call, .vis = vis[0..nv], .rows = tok.len * dr.n, .cfg = if (anchor) c else null };
            dr.pass(eng, ch, &pool, tok, q0, sl, steps, null, .{ .ctx = &dc, .at = DraftCheck.at, .anchor = if (anchor) DraftCheck.anchor else null }) catch |err| {
                if (err == error.DraftMismatch) return false;
                try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"draft_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, k, bp.call, @errorName(err) });
                try w_out.flush();
                return false;
            };
            // its drafts against the recorded ones, the served depth from its confidences against the window's
            var arg_buf: [16]u8 = undefined;
            const want = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, try std.fmt.bufPrint(&arg_buf, "out.{d}", .{0})));
            var drafts_ok = want.len == steps;
            if (drafts_ok) for (0..steps) |j| {
                if (dr.drafts[0][j] != want[j]) drafts_ok = false;
            };
            const emitted: usize = @intCast(pos[0] - @as(i64, @intCast(prompt_len)) + 1);
            const cap_k = @min(draft_mod.depth(1, most, R_max), drafts -| emitted);
            const kk = try draft_mod.chooseK(dr.confs[0][0..steps], @min(cap_k, steps), 1, most, R_max, run.costs);
            var window_ok = kk + 1 == ids.len;
            if (window_ok) for (1..ids.len) |j| {
                if (ids[j] != dr.drafts[0][j - 1]) window_ok = false;
            };
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"draft_call\": {d}, \"steps\": {d}, \"drafts_equal\": {}, \"k\": {d}, \"conf\": [", .{ run.rank, k, bp.call, steps, drafts_ok, kk });
            for (dr.confs[0][0..steps], 0..) |cf, j| try w_out.print("{s}{e}", .{ if (j == 0) "" else ", ", cf });
            try w_out.print("], \"window_equal\": {}}}\n", .{window_ok});
            try w_out.flush();
            if (!drafts_ok or !window_ok) run.ok = false;
        }
        const p0: usize = @intCast(pos[0]);
        if (pos[0] < 0 or p0 + ids.len > seq.len) return error.RoundNotNext;
        for (ids, 0..) |id, i| seq[p0 + i] = @intCast(id);
        // the RoundRunner.forward around this run: the last one before it
        var fcall: u64 = 0;
        for (fx.points) |*q| {
            if (q.call < rp.call and std.mem.eql(u8, q.where, "RoundRunner.forward")) fcall = q.call;
        }
        var rc: RoundCheck = .{ .run = run, .eng = eng, .after = rp.call, .call = rp.call, .fwd = fcall, .rows = ids.len, .cfg = if (anchor) c else null };
        const rows: round_mod.Rows = .{ .ids = ids, .pos = pos, .base = base[0], .end = end[0] };
        round_mod.forward(eng, &rd, ch, caches, rings, eh, seq[0 .. p0 + ids.len], rows, pool_window, .{ .ctx = &rc, .at = RoundCheck.at, .anchor = if (anchor) RoundCheck.anchor else null, .put = RoundCheck.put }, null) catch |err| {
            if (err == error.RoundMismatch) return false;
            try w_out.print("{{\"rank\": {d}, \"round\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, k, @errorName(err) });
            try w_out.flush();
            return false;
        };
        if (drafts > 0) {
            // the window's rows into slot 0's rings at once (TF_DS_EAGER_ABSORB: before the round's tokens are known)
            const items = [_]draft_mod.Item{.{ .slot = 0, .taps = rd.taps, .n = ids.len, .start = p0 }};
            try dr.absorbMany(eng, ch, &pool, &items);
            if (fx.find(rp.call, "Drafter.absorb_many", null, "in3.0.1")) |ap| {
                var label: [48]u8 = undefined;
                if (!try run.check(try std.fmt.bufPrint(&label, "R{d} absorb taps", .{rp.call}), ap, dr.at)) return false;
            }
        }
        for (0..ids.len) |j| targets[j] = try argmax(run, rd.logits + j * vocab * 4, vocab);
        @memcpy(prev_ids[0..ids.len], ids);
        prev_n = ids.len;
        prev_pos = pos[0];
        after = rp.call;
    }
    return run.ok;
}

/// --anchor: layer `li` of the step whose Model.forward / RoundDecoder.run is call `call` starts from the fixture's
/// Ref.layer_in: h (bf16 [n, hc, D], the streams after the previous layer's post) and pre (fp32 [n, hc], the carried
/// pre-mix: the previous sublayer's), written over the engine's `h` and `pre` (the caller drops its pending post).
fn anchorLayer(run: *Run, call: u64, li: usize, h: u64, pre: u64, n: usize, c: *const dsv41.Config) !void {
    const l: i64 = @intCast(li);
    const ph = run.fx.find(call - 1, "Ref.layer_in", l, "h") orelse return error.NoAnchorFixture;
    const pp = run.fx.find(call - 1, "Ref.layer_in", l, "pre") orelse return error.NoAnchorFixture;
    try run.upload(ph, h, n * c.hc * c.hidden * 2);
    try run.upload(pp, pre, n * c.hc * 4);
    try run.out.print("{{\"rank\": {d}, \"anchor\": {d}, \"layer\": {d}, \"rows\": {d}}}\n", .{ run.rank, run.step, li, n });
    try run.out.flush();
}

/// One chunk's layers through the prompt forward, each exchange checked against the fixtures after the chunk's call
/// when `checked` (else only run); at the decoder's first layer the replay cut (an encoder-only chunk ends there); a
/// chunk through every layer ends with its taps and the head, checked too. False on the first difference or a step's
/// error (reported).
fn runChunk(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, shared: *prompt.Shared, rings: []const u64, eh: ?*prompt.EngramHost, seq: []const i32, call: u64, layers: usize, checked: bool, dump: ?[]const u8, replay: usize, host_pos: []i64, anchor: bool) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const rank = run.rank;
    const c = eng.c;
    var after = call;
    var floor: usize = 0;
    var kv_done: ?usize = null;
    var label_buf: [64]u8 = undefined;
    // the DSpark taps only when the recorded forward took them (a drafting engine: Model.forward's taps list)
    const want_taps = if (fx.find(call - 1, "Model.forward", null, "post.in.taps.0")) |q| q.call == call else false;
    // the replay cut's rows (Run.row_off): none until the decoder's first layer, and each tap's as it was taken
    const start0 = ch.start;
    run.row_off = 0;
    run.rows = ch.n;
    defer run.row_off = 0;
    var tap_cut: [c.dspark_taps.items.len][2]usize = undefined;
    for (0..layers) |li| {
        const l: i64 = @intCast(li);
        run.layer = l;
        // --anchor: the layer starts from the reference streams and pre (all the chunk's rows: at the decoder's first
        // layer before the cut, whose KV source reads them all)
        if (anchor) anchorLayer(run, call, li, ch.h, ch.pre, ch.n, c) catch |err| return report(w_out, rank, li, "anchor", err);
        if (anchor) ch.pending = false;
        if (li == c.layers / 2) {
            // the cut layer's attention input on every row (its KV-source update reads them all), dumped only: what an
            // operator replay of the cut needs below row_start (anchored: the mixes are a pure function of h and pre,
            // so the cut's own mixes give the same bytes)
            if (checked and anchor and run.dump != null and replay > ch.start) {
                prompt.attnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "attention mixes", err);
                const n: i64 = @intCast(ch.n);
                const d: i64 = @intCast(c.hidden);
                run.dumpOnly("kv_source_in", call, ch.x, &.{ n, d }, "bfloat16") catch |err| return report(w_out, rank, li, "kv source dump", err);
            }
            const go_on = prompt.replayCut(eng, ch, caches, shared, li, replay, host_pos) catch |err| return report(w_out, rank, li, "replay cut", err);
            try w_out.print("{{\"rank\": {d}, \"layer\": {d}, \"replay\": {d}, \"rows\": {d}, \"start\": {d}, \"encoder_only\": {}}}\n", .{ rank, li, replay, ch.n, ch.start, !go_on });
            try w_out.flush();
            if (!go_on) break;
            floor = replay;
            kv_done = li;
            run.row_off = ch.start - start0;
            run.rows = ch.n;
        }
        if (eng.w.layers[li].engram_wkv != null) {
            const e_h = eh orelse return report(w_out, rank, li, "engram", error.NoEngramTables);
            prompt.engramApply(eng, ch, e_h, li, seq) catch |err| return report(w_out, rank, li, "engram", err);
            if (dump) |dir| dumpEngram(run.a, run.io, dir, li, rank, e_h, ch.n) catch |err| {
                try w_out.print("{{\"rank\": {d}, \"dump\": \"{s}\"}}\n", .{ rank, @errorName(err) });
                try w_out.flush();
            };
            if (checked) {
                run.phase = "engram";
                const eg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "engram fixture", error.NoLayerFixture);
                // (2D: the projection here is a column part of the rank's; the gather below holds the ranks' whole ones)
                if (prompt.two2d(eng) == null and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram projection", .{li}), eg, ch.engramOut(eng, eng.w.layers[li].engram_wkv.?.n))) return false;
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} engram gather", .{li}), fx.find(after, "Comm.gather", null, "out"), ch.ekg)) return false;
                after = eg.call;
            }
        }
        // the fixtures' points of this layer (a checked run only)
        const att: ?*const Point = if (checked) (fx.find(after, "Model.attention_k", l, "in2") orelse return report(w_out, rank, li, "attention fixture", error.NoLayerFixture)) else null;
        prompt.attnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "attention mixes", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention in", .{li}), att, ch.x)) return false;
        // a DSpark tap reads the streams after the layer's Engram (_forward_k's order), after the mixes as the engine takes it (U52)
        if (want_taps) {
            if (std.mem.indexOfScalar(u16, c.dspark_taps.slice(), @intCast(li))) |j| {
                prompt.tap(eng, ch, j) catch |err| return report(w_out, rank, li, "tap", err);
                tap_cut[j] = .{ run.row_off, run.rows };
            }
        }
        prompt.attention(eng, ch, caches, shared, li, rings[li], floor, kv_done != null and kv_done.? == li) catch |err| return report(w_out, rank, li, "attention", err);
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention partial", .{li}), fx.find(after, "Model.attention_k", l, "out"), ch.attnOut(eng))) return false;
        // an index layer's selection, dumped only: int64 [n, kk], -1 past each row's visible entries
        if (checked and run.dump != null and eng.w.layers[li].ratio != 0 and eng.w.layers[li].idx_wq_b != null and (shared.kk orelse 0) > 0) {
            const n: i64 = @intCast(ch.n);
            const kk: i64 = @intCast(shared.kk.?);
            run.dumpOnly("Ref.indexer.cidx", att.?.call, ch.cidx, &.{ n, kk }, "int64") catch |err| return report(w_out, rank, li, "indexer dump", err);
        }
        try prompt.gather(eng, ch, ch.attnOut(eng), ch.ga, .at(.prompt_attn, li));
        run.phase = "attn";
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} attention gather", .{li}), fx.find(att.?.call, "Comm.gather", null, "out"), ch.ga)) return false;
        prompt.ffnMixes(eng, ch, li) catch |err| return report(w_out, rank, li, "ffn mixes", err);
        const mo: ?*const Point = if (checked) (fx.find(att.?.call, "Model.moe", l, "in2") orelse return report(w_out, rank, li, "moe fixture", error.NoLayerFixture)) else null;
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe in", .{li}), mo, ch.x)) return false;
        prompt.moe(eng, ch, li) catch |err| return report(w_out, rank, li, "moe", err);
        // the router's input, dumped only (no fixture records it): the chunk sums or the logits route() read
        if (checked and run.dump != null) {
            const e: i64 = @intCast(c.experts);
            const n: i64 = @intCast(ch.n);
            const kc: i64 = @intCast(ch.route_kc);
            run.dumpOnly("route_logits", mo.?.call, ch.route_in, if (kc > 0) &.{ n, kc, e } else &.{ n, e }, "float32") catch |err| return report(w_out, rank, li, "route dump", err);
        }
        if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe partial", .{li}), fx.find(att.?.call, "Model.moe", l, "out"), ch.moeOut(eng))) return false;
        // the routing (a fixture that records it: w-ref's Ref.route), the routed slots beside the shared expert's
        if (checked) if (fx.find(att.?.call, "Ref.route", l, "pick")) |rpk| {
            if (!try run.checkCols(try std.fmt.bufPrint(&label_buf, "L{d} route picks", .{li}), rpk, ch.pick, eng.slots())) return false;
            if (!try run.checkCols(try std.fmt.bufPrint(&label_buf, "L{d} route weights", .{li}), fx.find(att.?.call, "Ref.route", l, "wts"), ch.wts, eng.slots())) return false;
        };
        try prompt.gather(eng, ch, ch.moeOut(eng), ch.gm, .at(.prompt_moe, li));
        run.phase = "moe";
        if (checked) {
            const mg = fx.find(mo.?.call, "Comm.gather", null, "out") orelse return report(w_out, rank, li, "moe gather fixture", error.NoLayerFixture);
            if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} moe gather", .{li}), mg, ch.gm)) return false;
            after = mg.call;
        }
        prompt.endLayer(ch);
        if (anchor) {
            // the layer's output (the next layer starts from the reference anyway): the post applied here, in place, as
            // the engine's own taps and head apply it, then the streams and the carried pre against Ref.layer_out
            try dsv41.tri_basic.hcPost(eng.t, ch.gm, ch.h, ch.post, ch.comb, ch.h, eng.world, ch.n, c.hidden);
            ch.pending = false;
            if (checked) if (fx.find(att.?.call, "Ref.layer_out", l, "h")) |ho| {
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} layer out", .{li}), ho, ch.h)) return false;
                if (!try run.check(try std.fmt.bufPrint(&label_buf, "L{d} layer out pre", .{li}), fx.find(att.?.call, "Ref.layer_out", l, "pre"), ch.pre)) return false;
            };
        }
        if (li + 1 < c.layers) continue;
        run.layer = -1; // the model's own points: the taps, the head, the logits
        run.phase = "head";
        // the chunk's end: the taps (Model.forward's list), the head on the last row, the prompt's logits
        for (0..if (want_taps) c.dspark_taps.slice().len else 0) |j| {
            var arg_buf: [32]u8 = undefined;
            const arg = try std.fmt.bufPrint(&arg_buf, "post.in.taps.{d}", .{j});
            run.row_off, run.rows = tap_cut[j];
            if (checked and !try run.check(try std.fmt.bufPrint(&label_buf, "tap {d}", .{j}), fx.find(call - 1, "Model.forward", null, arg), prompt.tapRows(eng, ch, j))) return false;
        }
        run.row_off = 0; // the head's points: the last row's
        prompt.head(eng, ch) catch |err| return report(w_out, rank, li, "head", err);
        if (checked) {
            const hg = fx.find(after, "Comm.gather", null, "in1") orelse return report(w_out, rank, li, "head fixture", error.NoLayerFixture);
            // (2D: the columns here are a quarter of the rank's; the gather holds both ranks' halves)
            if (prompt.two2d(eng) == null and !try run.check("head columns", hg, ch.headOut(eng))) return false;
            if (!try run.check("head gather", fx.find(after, "Comm.gather", null, "out"), ch.head_g)) return false;
            if (!try run.check("prompt logits", fx.find(call - 1, "Model.forward", null, "out"), ch.head_g)) return false;
        }
    }
    try run.s.synchronize();
    return true;
}

/// A fixture's one-element int64 (a round's id, position or extent) from its recorded first bytes.
fn headInt(p: *const Point) !i64 {
    const h = p.head orelse return error.NoHead;
    if (h.len < 16) return error.NoHead;
    var b: [8]u8 = undefined;
    _ = try std.fmt.hexToBytes(&b, h[0..16]);
    return std.mem.readInt(i64, &b, .little);
}

fn isPoint(p: *const Point, where: []const u8, arg: []const u8) bool {
    return std.mem.eql(u8, p.where, where) and std.mem.eql(u8, p.arg, arg);
}

/// A light run's round checker: the round's logits only (the recording holds no layer points).
const LightCheck = struct {
    run: *Run,
    call: u64,

    fn at(ctx: *anyopaque, what: round_mod.Point, layer: usize, dev: u64) anyerror!bool {
        _ = layer;
        const lc: *LightCheck = @ptrCast(@alignCast(ctx));
        if (what != .logits) return true;
        var buf: [48]u8 = undefined;
        return lc.run.check(try std.fmt.bufPrint(&buf, "R{d} logits", .{lc.call}), lc.run.fx.find(lc.call - 1, "RoundDecoder.run", null, "out"), dev);
    }
};

/// Every recorded request of a token recording: its prompt (the chunks' ids) through every layer, the prompt's logits
/// checked; then its rounds (at most max_rounds, 0: all), each round's logits checked and the greedy token against the
/// recorded next one; the caches and rings cleared between requests. A request whose first chunks the served engine
/// restored from a kept prompt (its recorded chunks start past 0: the first round's position says how many are missing)
/// takes them from an earlier request's prompt and computes them again: the state they leave is the restored one (the
/// same ids through the same encoder-only chunks). False on the first difference.
fn runLight(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, a: std.mem.Allocator, arena: *prompt.Arena, max_rounds: usize, tokens: usize, graphs: ?*round_mod.Graphs) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const pts = fx.points;
    const vocab = vocabOf(eng);
    var rd = try round_mod.Round.init(eng, arena, a, tokens);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    const seq = try a.alloc(i32, tokens + 16);
    const host_pos = try a.alloc(i64, chunk_rows);
    var prompts: std.ArrayList([]const i32) = .empty; // the earlier requests' prompts (kept prompts' prefixes)
    var i: usize = 0;
    var req: usize = 0;
    while (true) : (req += 1) {
        // the request: its chunks (Model.forward's ids), then its rounds (RoundDecoder.run's)
        while (i < pts.len and !(isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1)) i += 1;
        if (i == pts.len) break;
        var chunks: std.ArrayList(*const Point) = .empty;
        var rounds: std.ArrayList(*const Point) = .empty;
        while (i < pts.len and !isPoint(&pts[i], "RoundDecoder.run", "in1")) : (i += 1) {
            if (isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1) try chunks.append(a, &pts[i]);
        }
        while (i < pts.len and !(isPoint(&pts[i], "Model.forward", "in2") and pts[i].shape.len == 1)) : (i += 1) {
            if (isPoint(&pts[i], "RoundDecoder.run", "in1")) try rounds.append(a, &pts[i]);
        }
        // the prompt: every recorded chunk's ids, after the kept prefix the served engine restored (if any)
        var tail: usize = 0;
        for (chunks.items) |cp| tail += (try ints(run, cp)).len;
        const total: usize = if (rounds.items.len > 0) @intCast(try headInt(fx.find(rounds.items[0].call - 1, "RoundDecoder.run", null, "in2") orelse return error.NoRoundFixture)) else tail;
        if (total < tail or total > seq.len) return error.ChunkTooLong;
        const missing = total - tail;
        if (missing % chunk_rows != 0) return error.KeptPrefixNotChunked;
        if (missing > 0) {
            var found = false;
            for (prompts.items) |pp| {
                if (pp.len < missing) continue;
                @memcpy(seq[0..missing], pp[0..missing]);
                found = true;
                break;
            }
            if (!found) return error.NoKeptPrefix;
        }
        var len: usize = missing;
        for (chunks.items) |cp| {
            const ids = try ints(run, cp);
            for (ids, 0..) |id, j| seq[len + j] = @intCast(id);
            len += ids.len;
        }
        try prompts.append(a, try a.dupe(i32, seq[0..len]));
        const replay = len -| c.window;
        try caches.clear(eng);
        for (rings) |r| try gpu.fill8Async(eng.d, eng.s.handle, r, 0, eng.ringBytes());
        var start: usize = 0;
        const ids64 = try a.alloc(i64, chunk_rows);
        while (start < len) {
            const n: usize = @min(chunk_rows, len - start);
            for (0..n) |j| ids64[j] = seq[start + j];
            var shared: prompt.Shared = .{};
            try prompt.begin(eng, ch, ids64[0..n], start, host_pos);
            if (!try runChunk(run, eng, ch, caches, &shared, rings, eh, seq[0 .. start + n], chunks.items[0].call, c.layers, false, null, replay, host_pos, false)) return false;
            start += n;
        }
        const last = chunks.items[chunks.items.len - 1];
        var buf: [48]u8 = undefined;
        const prompt_ok = try run.check(try std.fmt.bufPrint(&buf, "request {d} prompt logits", .{req}), fx.find(last.call - 1, "Model.forward", null, "out"), ch.head_g);
        var greedy = try argmax(run, ch.head_g, vocab);
        var have = len;
        var equal_tokens: usize = 0;
        var equal_logits: usize = 0;
        const n_rounds = if (max_rounds == 0) rounds.items.len else @min(max_rounds, rounds.items.len);
        for (rounds.items[0..n_rounds]) |rp| {
            const id = try headInt(rp);
            const pos = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in2") orelse return error.NoRoundFixture);
            const base = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in4") orelse return error.NoRoundFixture);
            const end = try headInt(fx.find(rp.call - 1, "RoundDecoder.run", null, "in5") orelse return error.NoRoundFixture);
            if (id == @as(i64, @intCast(greedy))) equal_tokens += 1;
            if (pos != @as(i64, @intCast(have)) or have >= seq.len) return error.RoundNotNext;
            seq[have] = @intCast(id);
            have += 1;
            var lc: LightCheck = .{ .run = run, .call = rp.call };
            const ids = [_]i64{id};
            const posv = [_]i64{pos};
            const rows: round_mod.Rows = .{ .ids = &ids, .pos = &posv, .base = 0, .end = end - base };
            const before = run.ok;
            round_mod.forward(eng, &rd, ch, caches, rings, eh, seq[0..have], rows, pool_window, .{ .ctx = &lc, .at = LightCheck.at }, graphs) catch |err| {
                if (err != error.RoundMismatch) {
                    try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"round_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, req, rp.call, @errorName(err) });
                    try w_out.flush();
                    return false;
                }
            };
            if (run.ok and before) equal_logits += 1;
            greedy = try argmax(run, rd.logits, vocab);
        }
        try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"prompt_tokens\": {d}, \"kept_prefix\": {d}, \"chunks\": {d}, \"prompt_logits_equal\": {}, \"rounds\": {d}, \"round_logits_equal\": {d}, \"greedy_equal\": {d}}}\n", .{ run.rank, req, len, missing, chunks.items.len, prompt_ok, n_rounds, equal_logits, equal_tokens });
        try w_out.flush();
        if (!run.ok) return false;
    }
    return run.ok;
}

/// A stream of the drafted light run (multi.py's Stream in MultiDecoder): its pool slot and extent, its ids (the prompt,
/// the kept tokens, the current window's rows), its kept length, pending token, tokens out and budget; its window.
const DStream = struct {
    req: usize,
    slot: usize,
    base: i64 = 0,
    end: i64 = 0,
    seq: []i32,
    len: usize = 0, // positions kept (sc.length)
    pending: i64 = 0,
    out: usize = 0, // tokens taken (len(s.out))
    budget: usize,
    done: bool = false,
    rounds: usize = 0,
    tokens: []i64, // the reply: the first token, then each round's new tokens
    smp: sampling.Sampling, // temperature 0: greedy
    // this round's window: its rows in the round, from row `row`
    want: usize = 0,
    win: [round_mod.max_rows]i64 = undefined,
    n: usize = 0,
    row: usize = 0,
};

/// The first point after call `after` at `where` with argument `arg` and call before `before` (a step between two).
fn findBefore(fx: *const Fixtures, after: u64, before: u64, where: []const u8, arg: []const u8) ?*const Point {
    const q = fx.find(after, where, null, arg) orelse return null;
    return if (q.call < before) q else null;
}

/// Every request of a drafting recording (the served concurrent decoder with drafts, greedy), as MultiDecoder serves
/// them: each request's prompt through every layer into its stream's slot (the lowest free one) and extent (the base
/// the recording's rounds show), its logits checked, its taps absorbed into the drafter's rings; then each recorded
/// round from the port's own scheduling: the live streams in admission order, the draft depth by their count, each
/// stream's want (its budget's rest), one batched drafter pass for the drafting streams (every recorded drafter point
/// checked), each stream's depth from its confidences (_choose_k), the windows [pending, drafts]; the rows against the
/// recorded round's, the round through every layer (every recorded point checked); each stream's targets (greedy), the
/// drafts kept while equal to them, the target after them, the end of its budget; every window's taps absorbed. A
/// round the recording holds and the scheduling does not make (or the other way) fails the run. `budgets`: each
/// request's max_tokens in admission order; `seeds` each request's sampling seed (with `smp`'s temperature, top_k and
/// top_p: the served sampler, sampling.zig, every target checked against the recorded DsEngine._sample's), none: greedy.
fn runDrafted(run: *Run, eng: *const prompt.Engine, ch: *prompt.Chunk, caches: *const prompt.Caches, rings: []const u64, eh: ?*prompt.EngramHost, a: std.mem.Allocator, arena: *prompt.Arena, budgets: []const usize, seeds: []const u64, smp: sampling.Sampling, sp: dsv41.plan.Split) !bool {
    const fx = run.fx;
    const w_out = run.out;
    const c = eng.c;
    const pts = fx.points;
    const vocab = vocabOf(eng);
    const R_max = round_mod.default_rows; // the recordings' round bound
    const most = 5; // the engine's drafts (--mtp-drafts 5)
    var rd = try round_mod.Round.init(eng, arena, a, ch.max_comp);
    if (side_on) try rd.useSide(eng);
    defer rd.dropSide(eng);
    if (prefetch_on) try rd.usePrefetch(eng);
    defer rd.dropPrefetch();
    var pool = try draft_mod.Pool.init(eng, arena, draft_mod.default_streams);
    var dr = try draft_mod.Drafter.init(eng, arena, sp);
    var streams: std.ArrayList(DStream) = .empty;
    const host_pos = try a.alloc(i64, chunk_rows);
    const ids64 = try a.alloc(i64, chunk_rows);
    const ring_view = try a.alloc(u64, rings.len);
    var prev_round: u64 = 0; // the last round's RoundDecoder.run call
    var i: usize = 0;
    while (i < pts.len) : (i += 1) {
        const p = &pts[i];
        if (isPoint(p, "Model.forward", "in2") and p.shape.len == 1) {
            // a request's prompt: its chunks to the one that reaches the head (its logits)
            const req = streams.items.len;
            if (req >= budgets.len) return error.MoreRequestsThanBudgets;
            var chunks: std.ArrayList(*const Point) = .empty;
            var last: ?*const Point = null;
            var j = i;
            while (j < pts.len) : (j += 1) {
                const q = &pts[j];
                if (isPoint(q, "RoundDecoder.run", "in1")) break; // (a round inside a prompt's fill: not ported)
                if (!(isPoint(q, "Model.forward", "in2") and q.shape.len == 1)) continue;
                try chunks.append(a, q);
                if (fx.find(q.call - 1, "Model.forward", null, "out")) |o| if (o.call == q.call) {
                    last = q;
                    break;
                };
            }
            const lp = last orelse return error.PromptWithoutLogits;
            var len: usize = 0;
            for (chunks.items) |cp| len += (try ints(run, cp)).len;
            // its slot: the lowest one no live stream holds; its extent: the first round's rows in that slot
            var held: [draft_mod.max_streams]bool = @splat(false);
            for (streams.items) |*st| {
                if (!st.done) held[st.slot] = true;
            }
            const slot = std.mem.indexOfScalar(bool, &held, false) orelse return error.NoFreeSlot;
            var base: i64 = 0;
            var end: i64 = 0;
            var found = false;
            var k = j;
            while (k < pts.len and !found) : (k += 1) {
                const q = &pts[k];
                if (!isPoint(q, "RoundDecoder.run", "in3")) continue;
                const sl = try ints(run, q);
                for (sl, 0..) |v, r| {
                    if (v != @as(i64, @intCast(slot))) continue;
                    const rpos = try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in2"));
                    if (rpos[r] != @as(i64, @intCast(len))) return error.SlotNotThePrompts;
                    base = (try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in4")))[r];
                    end = (try ints(run, fx.find(q.call - 1, "RoundDecoder.run", null, "in5")))[r];
                    found = true;
                    break;
                }
            }
            const st = try streams.addOne(a);
            var ss = smp;
            if (seeds.len > 0) {
                if (req >= seeds.len) return error.MoreRequestsThanSeeds;
                ss.seed = seeds[req];
            } else ss.temperature = 0;
            st.* = .{ .req = req, .slot = slot, .base = base, .end = end, .seq = try a.alloc(i32, len + budgets[req] + R_max), .budget = budgets[req], .tokens = try a.alloc(i64, budgets[req]), .smp = ss };
            var at: usize = 0;
            for (chunks.items) |cp| for (try ints(run, cp)) |id| {
                st.seq[at] = @intCast(id);
                at += 1;
            };
            // the prompt into its slot's rings and positional stores, its extent's compressed rows
            const view = try caches.view(eng, slot, @intCast(base));
            for (rings, 0..) |r, li| ring_view[li] = r + slot * eng.ringBytes();
            const replay = len -| c.window;
            var start: usize = 0;
            while (start < len) {
                const n: usize = @min(chunk_rows, len - start);
                for (0..n) |jj| ids64[jj] = st.seq[start + jj];
                var shared: prompt.Shared = .{};
                try prompt.begin(eng, ch, ids64[0..n], start, host_pos);
                if (!try runChunk(run, eng, ch, &view, &shared, ring_view, eh, st.seq[0 .. start + n], lp.call, c.layers, false, null, replay, host_pos, false)) return false;
                start += n;
            }
            var buf: [64]u8 = undefined;
            const prompt_ok = try run.check(try std.fmt.bufPrint(&buf, "request {d} prompt logits", .{req}), fx.find(lp.call - 1, "Model.forward", null, "out"), ch.head_g);
            // its taps (the replayed rows, the tap layers side by side) into its slot's drafter rings
            const d = c.hidden;
            for (0..c.dspark_taps.slice().len) |jj| try eng.ops.copyRows(eng.s, prompt.tapRows(eng, ch, jj), d * 2, dr.at + jj * d * 2, dr.taps_w * 2, d * 2, ch.n);
            if (fx.find(lp.call, "Drafter.absorb", null, "in3")) |ap| {
                if (!try run.check(try std.fmt.bufPrint(&buf, "request {d} absorb taps", .{req}), ap, dr.at)) return false;
            }
            try dr.absorb(eng, ch, &pool, slot, dr.at, ch.n, ch.start);
            st.len = len;
            st.pending = try sampleDev(run, ch.head_g, vocab, len, st.smp);
            st.tokens[0] = st.pending;
            st.out = 1;
            if (st.pending == c.eos or st.out >= st.budget) st.done = true;
            try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"prompt_tokens\": {d}, \"chunks\": {d}, \"slot\": {d}, \"base\": {d}, \"prompt_logits_equal\": {}, \"first\": {d}}}\n", .{ run.rank, req, len, chunks.items.len, slot, base, prompt_ok, st.pending });
            try w_out.flush();
            if (!found and !st.done) return error.NoRoundForThePrompt;
            i = j;
            continue;
        }
        if (!isPoint(p, "RoundDecoder.run", "in1")) continue;
        // a round: the live streams in admission order, each one's want and window
        const rp = p;
        var live: [draft_mod.max_streams]*DStream = undefined;
        var nl: usize = 0;
        for (streams.items) |*st| {
            if (st.done) continue;
            if (nl == live.len) return error.TooManyStreams;
            live[nl] = st;
            nl += 1;
        }
        if (nl == 0) return error.RoundWithoutStreams;
        const kdep = draft_mod.depth(nl, most, R_max);
        var tok: [draft_mod.max_streams]i64 = undefined;
        var q0: [draft_mod.max_streams]i64 = undefined;
        var sl: [draft_mod.max_streams]i64 = undefined;
        var nd: usize = 0;
        for (live[0..nl]) |st| {
            st.want = @min(kdep, st.budget -| st.out);
            st.win[0] = st.pending;
            st.n = 1;
            if (st.want == 0) continue;
            tok[nd] = st.pending;
            q0[nd] = @intCast(st.len);
            sl[nd] = @intCast(st.slot);
            nd += 1;
        }
        var label: [64]u8 = undefined;
        if (nd > 0) {
            // the drafting streams' pass: its recorded inputs, every recorded point, its drafts; each stream's depth
            const bp = findBefore(fx, prev_round, rp.call, "BatchDraftGraph.run", "in1") orelse return error.NoDraftFixture;
            const rtok = try ints(run, bp);
            const rq0 = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in2"));
            const rsl = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, "in3"));
            const inputs_ok = std.mem.eql(i64, rtok, tok[0..nd]) and std.mem.eql(i64, rq0, q0[0..nd]) and std.mem.eql(i64, rsl, sl[0..nd]);
            if (!inputs_ok) {
                try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"draft_call\": {d}, \"draft_inputs_equal\": false}}\n", .{ run.rank, rp.call, bp.call });
                try w_out.flush();
                return false;
            }
            const steps = draft_mod.passSteps(nd, pool.slots, most, R_max, dr.n);
            var vis: [draft_mod.max_streams * 128]usize = undefined;
            var nv: usize = 0;
            for (q0[0..nd], sl[0..nd]) |q, slot| {
                var pp: i64 = @max(0, q - @as(i64, @intCast(c.window)));
                while (pp < q) : (pp += 1) {
                    vis[nv] = @as(usize, @intCast(slot)) * pool.ring + @as(usize, @intCast(pp)) % pool.ring;
                    nv += 1;
                }
            }
            var dc: DraftCheck = .{ .run = run, .call = bp.call, .layers = c.layers, .after = bp.call, .vis = vis[0..nv] };
            dr.pass(eng, ch, &pool, tok[0..nd], q0[0..nd], sl[0..nd], steps, null, .{ .ctx = &dc, .at = DraftCheck.at }) catch |err| {
                if (err == error.DraftMismatch) return false;
                try w_out.print("{{\"rank\": {d}, \"draft_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, bp.call, @errorName(err) });
                try w_out.flush();
                return false;
            };
            var di: usize = 0;
            for (live[0..nl]) |st| {
                if (st.want == 0) continue;
                const want = try ints(run, fx.find(bp.call - 1, "BatchDraftGraph.run", null, try std.fmt.bufPrint(&label, "out.{d}", .{di})));
                if (want.len != steps or !std.mem.eql(i64, want, dr.drafts[di][0..steps])) {
                    try w_out.print("{{\"rank\": {d}, \"draft_call\": {d}, \"stream\": {d}, \"drafts_equal\": false}}\n", .{ run.rank, bp.call, st.req });
                    try w_out.flush();
                    return false;
                }
                const kk = try draft_mod.chooseK(dr.confs[di][0..steps], @min(st.want, steps), nl, most, R_max, run.costs);
                for (0..kk) |jj| st.win[1 + jj] = dr.drafts[di][jj];
                st.n = 1 + kk;
                di += 1;
            }
        }
        // the rows: each live stream's window at its kept length, its slot and extent; its ids through the window
        var r_ids: [R_max]i64 = undefined;
        var r_pos: [R_max]i64 = undefined;
        var r_sl: [R_max]i64 = undefined;
        var r_base: [R_max]i64 = undefined;
        var r_end: [R_max]i64 = undefined;
        var wins: [draft_mod.max_streams]round_mod.Window = undefined;
        var R: usize = 0;
        for (live[0..nl], 0..) |st, wi| {
            if (R + st.n > R_max) return error.RoundTooWide;
            st.row = R;
            for (0..st.n) |jj| {
                r_ids[R + jj] = st.win[jj];
                r_pos[R + jj] = @intCast(st.len + jj);
                r_sl[R + jj] = @intCast(st.slot);
                r_base[R + jj] = st.base;
                r_end[R + jj] = st.end;
                st.seq[st.len + jj] = @intCast(st.win[jj]);
            }
            wins[wi] = .{ .row = R, .n = st.n, .seq = st.seq[0 .. st.len + st.n] };
            R += st.n;
        }
        const rows_ok = std.mem.eql(i64, try ints(run, rp), r_ids[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in2")), r_pos[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in3")), r_sl[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in4")), r_base[0..R]) and
            std.mem.eql(i64, try ints(run, fx.find(rp.call - 1, "RoundDecoder.run", null, "in5")), r_end[0..R]);
        try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"streams\": {d}, \"drafting\": {d}, \"rows\": {d}, \"rows_equal\": {}}}\n", .{ run.rank, rp.call, nl, nd, R, rows_ok });
        try w_out.flush();
        if (!rows_ok) return false;
        var fcall: u64 = 0;
        for (fx.points) |*q| {
            if (q.call < rp.call and std.mem.eql(u8, q.where, "RoundRunner.forward")) fcall = q.call;
        }
        var rc: RoundCheck = .{ .run = run, .eng = eng, .after = rp.call, .call = rp.call, .fwd = fcall, .rows = R };
        const rows: round_mod.Rows = .{ .ids = r_ids[0..R], .pos = r_pos[0..R], .slots = r_sl[0..R], .bases = r_base[0..R], .ends = r_end[0..R], .windows = wins[0..nl] };
        round_mod.forward(eng, &rd, ch, caches, rings, eh, &.{}, rows, pool_window, .{ .ctx = &rc, .at = RoundCheck.at, .put = RoundCheck.put }, null) catch |err| {
            if (err == error.RoundMismatch) return false;
            try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"error\": \"{s}\"}}\n", .{ run.rank, rp.call, @errorName(err) });
            try w_out.flush();
            return false;
        };
        // every window's taps into its slot's rings at once (the served eager absorb), then each stream's acceptance
        var items: [draft_mod.max_streams]draft_mod.Item = undefined;
        for (live[0..nl], 0..) |st, wi| items[wi] = .{ .slot = st.slot, .taps = rd.taps + st.row * dr.taps_w * 2, .n = st.n, .start = st.len };
        try dr.absorbMany(eng, ch, &pool, items[0..nl]); // (its rows: the round's taps, checked)
        var sample_after = rp.call;
        for (live[0..nl]) |st| {
            var targets: [R_max]i64 = undefined;
            for (0..st.n) |jj| targets[jj] = try sampleDev(run, rd.logits + (st.row + jj) * vocab * 4, vocab, st.len + 1 + jj, st.smp);
            // the served sampler's tokens for this window (a recording with sampler points)
            if (fx.find(sample_after, "DsEngine._sample", null, "out")) |sq| {
                const want = try ints(run, sq);
                if (!std.mem.eql(i64, want, targets[0..st.n])) {
                    try w_out.print("{{\"rank\": {d}, \"round_call\": {d}, \"stream\": {d}, \"sample_call\": {d}, \"samples_equal\": false}}\n", .{ run.rank, rp.call, st.req, sq.call });
                    try w_out.flush();
                    return false;
                }
                sample_after = sq.call;
            }
            var acc: usize = 0;
            while (acc + 1 < st.n and st.win[acc + 1] == targets[acc]) acc += 1;
            st.len += acc + 1;
            pool.absorbed[st.slot] = st.len;
            // the new tokens: the kept drafts and the target after them, to an end token or the budget
            var take: usize = acc + 1;
            for (0..acc + 1) |jj| {
                const t = if (jj < acc) st.win[jj + 1] else targets[acc];
                if (t == c.eos) {
                    take = jj + 1;
                    st.done = true;
                    break;
                }
            }
            take = @min(take, st.budget -| st.out);
            for (0..take) |jj| st.tokens[st.out + jj] = if (jj < acc) st.win[jj + 1] else targets[acc];
            st.out += take;
            if (st.out >= st.budget) st.done = true;
            st.pending = if (take == 0) st.pending else (if (take - 1 < acc) st.win[take] else targets[acc]);
            st.rounds += 1;
        }
        prev_round = rp.call;
    }
    // every stream's rounds ended where the recording's did
    var ok = run.ok;
    for (streams.items) |*st| {
        try w_out.print("{{\"rank\": {d}, \"request\": {d}, \"rounds\": {d}, \"tokens\": {d}, \"done\": {}, \"reply\": [", .{ run.rank, st.req, st.rounds, st.out, st.done });
        for (st.tokens[0..st.out], 0..) |t, jj| try w_out.print("{s}{d}", .{ if (jj == 0) "" else ", ", t });
        try w_out.print("]}}\n", .{});
        if (!st.done) ok = false;
    }
    try w_out.flush();
    return ok;
}

/// The Engram rows' ids and their bf16 rows of layer li, for the served Engram.rows (zrec_engram_ref.py).
fn dumpEngram(a: std.mem.Allocator, io: std.Io, dir: []const u8, li: usize, rank: u32, eh: *const prompt.EngramHost, n: usize) !void {
    const m = n * (eh.hi - eh.lo);
    const t = eh.tables.layers.get(@intCast(li)) orelse return error.MissingEngramTable;
    for ([_][]const u8{ "ids.i64", "rows.bf16" }, [_][]const u8{ std.mem.sliceAsBytes(eh.flat[0..m]), std.mem.sliceAsBytes(eh.rows[0 .. m * t.row_w]) }) |ext, bytes| {
        const path = try std.fmt.allocPrint(a, "{s}/engram-L{d}-r{d}.{s}", .{ dir, li, rank, ext });
        var f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        try f.writePositionalAll(io, bytes, 0);
    }
}

fn report(w: *std.Io.Writer, rank: u32, li: usize, step: []const u8, err: anyerror) !bool {
    try w.print("{{\"rank\": {d}, \"layer\": {d}, \"step\": \"{s}\", \"error\": \"{s}\"}}\n", .{ rank, li, step, @errorName(err) });
    try w.flush();
    return false;
}

test "replay cut: a whole-chunk fixture is read, checked and dumped on the engine's rows only" {
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    // the fixtures of an uncut 5-row chunk: a gather's out [2, 5, 3], a rows-first [5, 3] and a one-row [1, 3] (bf16),
    // each element part * 100 + row * 10 + col
    var g: [2 * 5 * 3]u16 = undefined;
    for (&g, 0..) |*v, i| v.* = @intCast(i / 15 * 100 + i % 15 / 3 * 10 + i % 3);
    try tmp.dir.createDirPath(io, "layers");
    try tmp.dir.writeFile(io, .{ .sub_path = "layers/g.bin", .data = std.mem.sliceAsBytes(&g) });
    try tmp.dir.writeFile(io, .{ .sub_path = "layers/x.bin", .data = std.mem.sliceAsBytes(g[0..15]) });
    try tmp.dir.writeFile(io, .{ .sub_path = "layers/h.bin", .data = std.mem.sliceAsBytes(g[0..3]) });
    var points = [_]Point{
        .{ .call = 7, .where = "Comm.gather", .arg = "out", .shape = &.{ 2, 5, 3 }, .dtype = "bfloat16", .file = "g.bin" },
        .{ .call = 5, .where = "Model.moe", .layer = 20, .arg = "in2", .shape = &.{ 5, 3 }, .dtype = "bfloat16", .file = "x.bin", .key = "0:20:moe_in" },
        .{ .call = 9, .where = "Comm.gather", .arg = "in1", .shape = &.{ 1, 3 }, .dtype = "bfloat16", .file = "h.bin" },
    };
    const fx: Fixtures = .{ .points = &points, .dir = try std.fs.path.join(a, &.{ root, "rank0" }) };
    var wb: [4096]u8 = undefined;
    var w: std.Io.Writer = .fixed(&wb);
    var r: Run = .{ .a = a, .io = io, .d = undefined, .s = undefined, .fx = &fx, .rank = 0, .out = &w, .host = &.{}, .layer = 20, .row_off = 2, .rows = 3 };
    r.dump = try Dump.open(io, try std.fs.path.join(a, &.{ root, "dump" }), 0);
    defer r.dump.?.close(io);

    // the gather: each rank's rows 2..4; the rows-first point: rows 2..4; the one-row point: whole
    const cg = r.cut(&points[0]).?;
    try std.testing.expectEqual(@as(usize, 1), cg.lead);
    try std.testing.expectEqual(@as(usize, 2), cg.parts);
    // (byte compares: the arena's bytes need not be u16-aligned)
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 20, 21, 22, 30, 31, 32, 40, 41, 42, 120, 121, 122, 130, 131, 132, 140, 141, 142 }), try r.fixtureBytes(&points[0]));
    const xb = try r.fixtureBytes(&points[1]);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&[_]u16{ 20, 21, 22, 30, 31, 32, 40, 41, 42 }), xb);
    try std.testing.expect(r.cut(&points[2]) == null);
    try std.testing.expectEqual(@as(usize, 6), (try r.fixtureBytes(&points[2])).len);
    var sb: [max_dims]i64 = undefined;
    try std.testing.expectEqualSlices(i64, &.{ 2, 3, 3 }, r.held(&points[0], &sb).shape);
    try std.testing.expectEqual(@as(usize, 2), r.rowStart(&points[1]));
    try std.testing.expectEqual(@as(usize, 0), r.rowStart(&points[2]));

    // the engine's rows equal the fixture's from row 2: equal, dumped with the engine's shape and row_start 2; one
    // element off: a difference, reported at the engine's row
    r.host = try a.dupe(u8, xb);
    try std.testing.expect(try r.finish("L20 moe in", &points[1], 18));
    r.host[8] ^= 1; // engine row 1 (the fixture's row 3), col 1
    try std.testing.expect(!try r.finish("L20 moe in", &points[1], 18));
    try std.testing.expect(!r.ok);
    const said = w.buffered();
    try std.testing.expect(std.mem.indexOf(u8, said, "\"equal\": false, \"first\": 4, \"row\": 1, \"col\": 1, \"differ\": 1, \"of\": 9") != null);
    try std.testing.expect(std.mem.indexOf(u8, said, "\"row_start\": 2, \"rows\": 3") != null);
    // a dump-only point: equal null
    const ro: Point = .{ .call = 5, .where = "Engine.dump", .layer = 20, .arg = "route_logits", .shape = &.{ 3, 384 }, .dtype = "float32" };
    try r.dump.?.put(&r, "0:20:route_logits", &ro, 2, "abcd", &(@as([64]u8, @splat('0'))), null);
    const lines = try tmp.dir.readFileAlloc(io, "dump/rank0/layers.jsonl", a, .limited(1 << 20));
    var it = std.mem.splitScalar(u8, lines, '\n');
    const first = it.next().?;
    try std.testing.expect(std.mem.indexOf(u8, first, "\"shape\": [3, 3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, first, "\"key\": \"0:20:moe_in\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, first, "\"row_start\": 2, \"equal\": true}") != null);
    try std.testing.expect(std.mem.indexOf(u8, it.next().?, "\"equal\": false}") != null);
    try std.testing.expect(std.mem.endsWith(u8, it.next().?, "\"row_start\": 2, \"equal\": null}"));
    const bin = try tmp.dir.readFileAlloc(io, "dump/layers/r0.0_20_moe_in.bin", a, .limited(1 << 20));
    try std.testing.expectEqual(@as(usize, 18), bin.len);

    // no cut (row_off 0): the whole fixture, its own shape
    r.row_off = 0;
    try std.testing.expect(r.cut(&points[0]) == null);
    try std.testing.expectEqual(@as(usize, 60), (try r.fixtureBytes(&points[0])).len);
}

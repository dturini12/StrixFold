//! markov.py's Markov loop of the DSpark drafter (_bias, _score, _finish, _pick) and exl3/prefill.py's prompt GEMM
//! (_gemm), each launched as its Python wrapper launches it.
const std = @import("std");
const gpu = @import("gpu");
const tri = @import("tri.zig");
const config = @import("config.zig");
const Config = config.Config;
const Split = @import("plan.zig").Split;
const aot = gpu.aot;
const Tri = tri.Tri;
const u = tri.u;
const p = aot.ptr;
const tri_aux = @import("tri_aux.zig");
const sampling = @import("sampling.zig");

/// markov.BN: vocabulary entries a dot (V / world is a multiple of it).
pub const bn = 64;
/// markov.SUB: dots a bias program, one after another.
pub const sub = 4;
/// markov.BC: vocabulary entries a scoring program.
pub const bc = 1024;
/// markov.NP: rows a launch at most (tl.dot's smallest N); a step's rows are its streams.
pub const np = 16;
/// markov.CACHE_ROWS: the cached bias rows (TF_DS_MARKOV_CACHE unset in the served lane: 256).
pub const cache_rows = 256;

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

fn int(name: []const u8, v: usize) aot.Arg {
    return aot.int(name, @intCast(v));
}

/// Markov._scratch(N): pv fp32 and pi int32 [N, n_part], send fp32 [N, 4] (zeroed), the staging rows fp16 [N, cols].
pub const Scratch = struct { pv: u64, pi: u64, send: u64, stage: u64 };

/// comm.gather for steps: every rank's (value, index) bests [rows, 4] fp32 gathered at step `step`; returns the
/// [world, rows, 4] address.
pub const Gather = struct {
    ctx: *anyopaque,
    run: *const fn (ctx: *anyopaque, send: u64, rows: usize, step: usize) anyerror!u64,
};

/// A sampled pass's draws (HIP): aux.markov_race and the rows' keys (tri_aux.MarkovKey [rows], on the device).
pub const Race = struct { f: gpu.Function, keys: u64 };

/// aux.markov_race's launch: a row a workgroup.
pub fn race(t: Tri, f: gpu.Function, a: tri_aux.MarkovRaceArgs) !void {
    return t.runHipBlock("aux.markov_race", f, .{ @intCast(a.n_rows), 1, 1 }, tri_aux.markov_race_block, a);
}

extern "c" fn logf(x: f32) f32;
extern "c" fn log1pf(x: f32) f32;

/// aux.markov_race's Gumbel draw of the key bits (sampling.keyBits), its fp32 arithmetic: -log(-log(u)) of the
/// sampler's u, the double nearest ((bits >> 11) + 0.5) 2^-53. Below a half that is exact, and -log(u) is taken
/// from it; from a half on the double is (a + (a & 1)) 2^-53 (a tie, to even), and -log(u) is taken from 1 - u (the
/// sampler's u of the last a is 1 and its draw +inf: here too).
pub fn raceGumbel(bits: u64) f32 {
    const a = bits >> 11;
    const e: f32 = if (a < (1 << 52))
        -logf((@as(f32, @floatFromInt(a)) + 0.5) * 0x1p-53)
    else
        -log1pf(-(@as(f32, @floatFromInt((1 << 53) - a - (a & 1))) * 0x1p-53));
    return -logf(e);
}

/// _score's order of two (value, token) (kernels.py _better, torch.argmax's): a NaN above everything, then the larger
/// value, ties (and a NaN against a NaN) to the lower token.
fn raceBetter(v: f32, i: u32, bv: f32, bi: u32) bool {
    const n1 = v != v;
    const n2 = bv != bv;
    return (n1 and (!n2 or i < bi)) or (!n1 and !n2 and (v > bv or (v == bv and i < bi)));
}

/// aux.markov_race's pick of one row on the host (its GPU test's reference and the rule's statement): the best, in
/// _score's order, of score[c] = lg[c] + bias[c] (one fp32 add), times the key's inv_t plus the draw of token
/// id0 + c at `position` when inv_t > 0 (a greedy row: _score's and _finish's pick). (value, token).
pub fn raceRow(lg: []const f32, bias: []const f16, key: tri_aux.MarkovKey, position: u64, id0: u32) struct { f32, u32 } {
    var bv = -std.math.inf(f32);
    var bi: u32 = std.math.maxInt(u32);
    for (lg, bias, 0..) |l, b, c| {
        const score = l + @as(f32, b);
        const id = id0 + @as(u32, @intCast(c));
        const v = if (key.inv_t > 0) score * key.inv_t + raceGumbel(sampling.keyBits(key.seed, position, id)) else score;
        if (raceBetter(v, id, bv, bi)) {
            bv = v;
            bi = id;
        }
    }
    return .{ bv, if (bi == std.math.maxInt(u32)) id0 else bi };
}

/// markov.Markov: one drafter's Markov loop on one rank: the head rows it scores, the cached bias rows, every token's slot.
pub const Markov = struct {
    head: u64, // markov_head [V, rank] fp16
    emb: u64, // markov_embed [V, rank] bf16
    none: u64, // int32 [V], every entry -1: the slot table before fill
    slot: u64, // int32 [V]: a token's cached row, -1 none
    cache: u64 = 0, // fp16 [rows, cols]: the cached bias rows (fill)
    world: usize,
    rank_dim: usize,
    split: bool,
    n_cols: usize,
    segs: usize,
    seg0: usize,
    cols: usize,
    b_tiles: usize,
    s_tiles: usize,
    n_part: usize,
    bp: usize,
    // the head's logits one row of lg_w columns a row (a 2D node's quarter, draft2d.zig); 0: TP2's [segs, rows, n_cols]
    lg_w: usize = 0,

    /// Markov.__init__'s geometry for rank s.rank of s.world (head and emb [V, rank]); fill then makes the cached rows.
    pub fn init(c: Config, s: Split, head: u64, emb: u64, none: u64) !Markov {
        const world: usize = s.world;
        // TF_DS_MARKOV_SPLIT unset in the served lane: each rank scores its own vocabulary part whenever TP > 1
        const split = world > 1;
        const n_cols = c.vocab / world;
        if (n_cols * world != c.vocab or n_cols % bn != 0) return error.VocabSplit;
        const segs: usize = if (split) 1 else world;
        const s_tiles: usize = tri.cdiv(n_cols, bc);
        return .{
            .head = head,
            .emb = emb,
            .none = none,
            .slot = none,
            .world = world,
            .rank_dim = c.markov_rank,
            .split = split,
            .n_cols = n_cols,
            .segs = segs,
            .seg0 = if (split) s.rank else 0,
            .cols = segs * n_cols,
            .b_tiles = tri.cdiv(n_cols, bn * sub),
            .s_tiles = s_tiles,
            .n_part = s_tiles * segs,
            .bp = tri.pow2(s_tiles * segs),
        };
    }

    /// Markov.bias: _bias for n rows (tokens tok[r * t_stride] int64) into out [n, cols] fp16 (row stride o_row), each row whose token has no slot.
    pub fn bias(m: Markov, t: Tri, tok: u64, t_stride: usize, slot: u64, out: u64, o_row: usize, n: usize) !void {
        try t.run("_bias", .{ u(m.b_tiles), u(m.segs), 1 }, &.{
            p("TOK", "*i64", tok),      int("t_stride", t_stride), p("SLOT", "*i32", slot), p("EMB", "*bf16", m.emb),
            p("HEAD", "*fp16", m.head), p("OUT", "*fp16", out),    int("o_row", o_row),     int("n_rows", n),
            int("n_cols", m.n_cols),    int("seg0", m.seg0),
        }, &.{ ci("BN", bn), ci("SUB", sub), ci("RANK", m.rank_dim), ci("NP", np), t.pdlConst() });
    }

    /// Markov.fill: k tokens' rows (tok int64 [k]) into cache fp16 [max(k, 1), cols], NP a launch, slots none; slot int32 [V] (tok[j] -> j) is the caller's.
    pub fn fill(m: *Markov, t: Tri, tok: u64, k: usize, cache: u64, slot: u64) !void {
        var r0: usize = 0;
        while (r0 < k) : (r0 += np) try m.bias(t, tok + 8 * r0, 1, m.none, cache + 2 * r0 * m.cols, m.cols, @min(np, k - r0));
        m.cache = cache;
        m.slot = slot;
    }

    /// Markov.local_best: step i of out [N, steps + 1] int64 (row stride ts) on this rank's columns: out[:, i + 1], or the returned send [N, 4] (split).
    pub fn localBest(m: Markov, t: Tri, lg: u64, out: u64, ts: usize, n: usize, block: usize, i: usize, sc: Scratch) !?u64 {
        if (n > np) return error.TooManyRows;
        const nc = m.n_cols;
        const lw = if (m.lg_w != 0) m.lg_w else nc; // the logits' row width
        const l_seg = if (m.lg_w != 0) nc else n * block * nc; // a segment's offset in them
        const tok = out + 8 * i;
        try m.bias(t, tok, ts, m.slot, sc.stage, m.cols, n);
        try t.run("_score", .{ u(m.s_tiles), u(m.segs), 1 }, &.{
            p("LG", "*fp32", lg + 4 * i * lw), int("l_seg", l_seg),       int("l_row", block * lw),     p("TOK", "*i64", tok),
            int("t_stride", ts),               p("SLOT", "*i32", m.slot), p("CACHE", "*fp16", m.cache), int("c_row", m.cols),
            p("STAGE", "*fp16", sc.stage),     int("s_row", m.cols),      p("PV", "*fp32", sc.pv),      p("PI", "*i32", sc.pi),
            int("n_rows", n),                  int("n_cols", nc),         int("seg0", m.seg0),          int("n_part", m.n_part),
        }, &.{ ci("BC", bc), t.pdlConst() });
        try t.run("_finish", .{ u(n), 1, 1 }, &.{
            p("PV", "*fp32", sc.pv), p("PI", "*i32", sc.pi),      int("n_part", m.n_part),            p("OUT", "*i64", out + 8 * (i + 1)),
            int("o_stride", ts),     p("SEND", "*fp32", sc.send), int("send", @intFromBool(m.split)),
        }, &.{ ci("BP", m.bp), t.pdlConst() });
        return if (m.split) sc.send else null;
    }

    /// Whether a pass's steps can draw with aux.markov_race: this rank scores its own vocabulary columns, one segment
    /// of the head's TP2 logits, and hands its bests to the ranks' gather.
    pub fn races(m: Markov) bool {
        return m.split and m.segs == 1 and m.lg_w == 0;
    }

    /// localBest for a pass with sampled streams: step i's (value, token) of each row on this rank's columns into
    /// the returned send [N, 4], a sampled row's by its score over the draft temperature plus its key's draw at the
    /// step's position, a greedy row's as localBest picks it.
    pub fn raceBest(m: Markov, t: Tri, rc: Race, lg: u64, out: u64, ts: usize, n: usize, block: usize, i: usize, sc: Scratch) !u64 {
        if (n > np) return error.TooManyRows;
        if (!m.races()) return error.BadDraftPass;
        const nc = m.n_cols;
        const tok = out + 8 * i;
        try m.bias(t, tok, ts, m.slot, sc.stage, m.cols, n);
        try race(t, rc.f, .{
            .lg = lg + 4 * i * nc,          .tok = tok,                  .slot = m.slot,              .cache = m.cache,
            .stage = sc.stage,              .keys = rc.keys,             .send = sc.send,             .l_row = @intCast(block * nc),
            .t_stride = @intCast(ts),       .c_row = @intCast(m.cols),   .s_row = @intCast(m.cols),   .n_rows = @intCast(n),
            .n_cols = @intCast(nc),         .id0 = @intCast(m.seg0 * nc), .step = @intCast(i),
        });
        return sc.send;
    }

    /// Markov.pick: out[:, i + 1] (out [N, steps + 1] int64, row stride ts) from every rank's (value, index) bests g [world, N, 4] fp32.
    pub fn pick(_: Markov, t: Tri, g: u64, world: usize, out: u64, ts: usize, n: usize, i: usize) !void {
        try t.run("_pick", .{ u(n), 1, 1 }, &.{ p("G", "*fp32", g), int("n_rows", n), p("OUT", "*i64", out + 8 * (i + 1)), int("o_stride", ts) }, &.{ ci("WORLD", world), t.pdlConst() });
    }

    /// Markov.steps: out [N, count + 1] int64 (out[:, 0] the last tokens) -> out[:, 1:] the drafts, each step's bests gathered over the ranks before its pick.
    /// `rc` (a pass with sampled streams, races()): each step's bests are raceBest's.
    pub fn steps(m: Markov, t: Tri, lg: u64, out: u64, ts: usize, n: usize, block: usize, count: usize, sc: Scratch, gather: Gather, rc: ?Race) !void {
        for (0..count) |i| {
            const send = if (rc) |r| try m.raceBest(t, r, lg, out, ts, n, block, i, sc) else (try m.localBest(t, lg, out, ts, n, block, i, sc) orelse continue);
            try m.pick(t, try gather.run(gather.ctx, send, n, i), m.world, out, ts, n, i);
        }
    }
};

/// prefill.HAD_SCALE: 1 / sqrt(128).
pub const had_scale: f32 = 0.08838834764831845;
/// prefill.BN: a GEMM program's columns, one Hadamard block.
pub const gemm_bn = 128;

/// The prompt GEMM's output dtype: the caller's out tensor's.
pub const OutType = enum { bf16, fp32 };

/// prefill.matmul's _gemm after the extension's rot_in (xh fp16 [M, K]) and unpack (wq fp16 [K, N]): out [M, N] (row stride os) = (xh wq) H SCALE svh + bias.
pub fn matmul(t: Tri, xh: u64, wq: u64, h: u64, svh: u64, bias: ?u64, out: u64, out_type: OutType, os: usize, m: usize, k: usize, n: usize) !void {
    // TF_EXL3_PREFILL_TILES unset in the served lane: tiles() is 128 rows, a 32-wide K step, 8 warps, 4 stages and groups of 8 for every shape
    const bm = 128;
    const bk = 32;
    const group = 8;
    const ty: []const u8 = switch (out_type) {
        .bf16 => "*bf16",
        .fp32 => "*fp32",
    };
    try t.run("_gemm", .{ tri.cdiv(m, bm) * u(n / gemm_bn), 1, 1 }, &.{
        p("X", "*fp16", xh),                 p("W", "*fp16", wq), p("H", "*bf16", h), p("SVH", "*fp16", svh),
        p("BIAS", "*fp16", bias orelse svh), p("OUT", ty, out),   int("M", m),        int("o_stride", os),
    }, &.{ ci("K", k), ci("N", n), ci("BM", bm), ci("BK", bk), ci("GROUP", group), ci("HAS_BIAS", @intFromBool(bias != null)), aot.cf("SCALE", had_scale) });
}

// -- conformance with the recorded launches ----------------------------------------------------------------------

const Case = tri.Case;

/// The recording's lane: TP2 over the two GB10s.
const tp = 2;
/// A device address for a tensor the checked launch does not take.
const elsewhere: u64 = 0x7e00_0000_0000;

/// Our checkpoint's config.json (vocabulary, Markov rank, DSpark block).
fn modelConfig() !Config {
    var why: config.Why = .{};
    return config.parse(std.testing.allocator, config.test_config, &why);
}

/// The Markov of the case's rank.
fn markovOf(cfg: Config, c: Case, head: u64, emb: u64, none: u64) !Markov {
    return Markov.init(cfg, .{ .rank = @intCast(c.v.get("rank").?.integer), .world = tp }, head, emb, none);
}

fn dtype(c: Case, tensor: []const u8) []const u8 {
    return c.v.get("tensors").?.object.get(tensor).?.array.items[0].string;
}

test "Markov bias rows (fill at load, each step's staging rows) launch as recorded" {
    try tri.conform("_bias", struct {
        fn call(t: Tri, c: Case) !void {
            const cfg = try modelConfig();
            var m = try markovOf(cfg, c, c.ptr("HEAD"), c.ptr("EMB"), c.ptr("SLOT"));
            // fill's launch at r0 reads tok[r0:] (TOK) into cache[r0] (OUT): the first launch of a fill of TOK's tokens
            if (std.mem.eql(u8, c.v.get("phase").?.string, "init")) return m.fill(t, c.ptr("TOK"), c.dim("TOK", 0), c.ptr("OUT"), elsewhere);
            m.slot = c.ptr("SLOT");
            // a step's bias reads out[:, i] (TOK) and nothing else of the step's own: step 0 of out = TOK
            const sc: Scratch = .{ .pv = elsewhere, .pi = elsewhere, .send = elsewhere, .stage = c.ptr("OUT") };
            _ = try m.localBest(t, elsewhere, c.ptr("TOK"), c.stride("TOK", 0), c.dim("TOK", 0), cfg.dspark_block, 0, sc);
        }
    }.call);
}

test "Markov scoring launches as recorded" {
    try tri.conform("_score", struct {
        fn call(t: Tri, c: Case) !void {
            const cfg = try modelConfig();
            var m = try markovOf(cfg, c, elsewhere, elsewhere, elsewhere);
            m.slot = c.ptr("SLOT");
            m.cache = c.ptr("CACHE");
            const n = c.dim("TOK", 0);
            const nc = cfg.vocab / tp;
            // LG is lg.view(-1)[i * nc:] of the drafter head's [N block, nc] logits: its length gives the step i
            const i = n * cfg.dspark_block - c.dim("LG", 0) / nc;
            const sc: Scratch = .{ .pv = c.ptr("PV"), .pi = c.ptr("PI"), .send = elsewhere, .stage = c.ptr("STAGE") };
            _ = try m.localBest(t, c.ptr("LG") - 4 * i * nc, c.ptr("TOK") - 8 * i, c.stride("TOK", 0), n, cfg.dspark_block, i, sc);
        }
    }.call);
}

test "Markov step finish launches as recorded" {
    try tri.conform("_finish", struct {
        fn call(t: Tri, c: Case) !void {
            const cfg = try modelConfig();
            const m = try markovOf(cfg, c, elsewhere, elsewhere, elsewhere);
            // the finish writes out[:, i + 1] (OUT) and takes nothing else of the step's own: step 0 of out = OUT less a column
            const sc: Scratch = .{ .pv = c.ptr("PV"), .pi = c.ptr("PI"), .send = c.ptr("SEND"), .stage = elsewhere };
            _ = try m.localBest(t, elsewhere, c.ptr("OUT") - 8, c.stride("OUT", 0), c.dim("OUT", 0), cfg.dspark_block, 0, sc);
        }
    }.call);
}

test "Markov pick launches as recorded" {
    try tri.conform("_pick", struct {
        fn call(t: Tri, c: Case) !void {
            const m = try markovOf(try modelConfig(), c, elsewhere, elsewhere, elsewhere);
            // out[:, i + 1] is OUT: step 0 of out = OUT less a column
            try m.pick(t, c.ptr("G"), c.dim("G", 0), c.ptr("OUT") - 8, c.stride("OUT", 0), c.dim("OUT", 0), 0);
        }
    }.call);
}

test "the EXL3 prompt GEMM launches as recorded" {
    try tri.conform("_gemm", struct {
        fn call(t: Tri, c: Case) !void {
            const out_type: OutType = if (std.mem.eql(u8, dtype(c, "OUT"), "float32")) .fp32 else .bf16;
            // the model's EXL3 linears carry no bias (weights.Linear): matmul passes svh as BIAS
            try matmul(t, c.ptr("X"), c.ptr("W"), c.ptr("H"), c.ptr("SVH"), null, c.ptr("OUT"), out_type, c.stride("OUT", 0), c.dim("OUT", 0), c.dim("X", 1), c.dim("OUT", 1));
        }
    }.call);
}

test "steps: each step's bias, score and finish, then the pick of the ranks' gathered bests" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const cfg = try modelConfig();
    const m = try Markov.init(cfg, .{ .rank = 1, .world = tp }, elsewhere, elsewhere, elsewhere);
    const Ranks = struct {
        gathered: u64 = 0,
        fn run(ctx: *anyopaque, send: u64, rows: usize, step: usize) anyerror!u64 {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            try std.testing.expectEqual(@as(u64, 0x7c00_0000_0200), send);
            try std.testing.expectEqual(@as(usize, 3), rows);
            try std.testing.expectEqual(@as(usize, r.gathered), step); // the step, for the collective's site
            r.gathered += 1;
            return 0x7d00_0000_0000 + 0x1000 * r.gathered;
        }
    };
    var ranks: Ranks = .{};
    const out: u64 = 0x7b00_0000_0000;
    const sc: Scratch = .{ .pv = 0x7c00_0000_0000, .pi = 0x7c00_0000_0100, .send = 0x7c00_0000_0200, .stage = 0x7c00_0000_0300 };
    try m.steps(.{ .log = &log }, elsewhere, out, 4, 3, cfg.dspark_block, 3, sc, .{ .ctx = &ranks, .run = Ranks.run }, null);
    try std.testing.expectEqual(@as(u64, 3), ranks.gathered);
    try std.testing.expectEqual(@as(usize, 12), log.items.items.len);
    const names = [_][]const u8{ "_bias", "_score", "_finish", "_pick" };
    for (log.items.items, 0..) |k, j| {
        try std.testing.expectEqualStrings(names[j % 4], k.name);
        if (j % 4 != 3) continue;
        try std.testing.expectEqual(0x7d00_0000_0000 + 0x1000 * @as(u64, j / 4 + 1), k.args[0].value.ptr.addr);
        try std.testing.expectEqual(out + 8 * @as(u64, j / 4 + 1), k.args[2].value.ptr.addr);
    }
}

test "a sampled pass's steps: each step's bias, the race at its position, then the pick of the ranks' gathered bests" {
    var log: tri.Log = .{ .gpa = std.testing.allocator };
    defer log.deinit();
    const cfg = try modelConfig();
    var m = try Markov.init(cfg, .{ .rank = 1, .world = tp }, elsewhere, elsewhere, 0x7a00_0000_0000);
    m.cache = 0x7a00_0000_1000;
    try std.testing.expect(m.races());
    const Ranks = struct {
        gathered: u64 = 0,
        fn run(ctx: *anyopaque, send: u64, rows: usize, step: usize) anyerror!u64 {
            const r: *@This() = @ptrCast(@alignCast(ctx));
            try std.testing.expectEqual(@as(u64, 0x7c00_0000_0200), send);
            try std.testing.expectEqual(@as(usize, 3), rows);
            try std.testing.expectEqual(@as(usize, r.gathered), step);
            r.gathered += 1;
            return 0x7d00_0000_0000 + 0x1000 * r.gathered;
        }
    };
    var ranks: Ranks = .{};
    const out: u64 = 0x7b00_0000_0000;
    const lg: u64 = 0x7900_0000_0000;
    const keys: u64 = 0x7800_0000_0000;
    const sc: Scratch = .{ .pv = 0x7c00_0000_0000, .pi = 0x7c00_0000_0100, .send = 0x7c00_0000_0200, .stage = 0x7c00_0000_0300 };
    const f: gpu.Function = undefined; // the log takes the launch
    try m.steps(.{ .log = &log }, lg, out, 4, 3, cfg.dspark_block, 3, sc, .{ .ctx = &ranks, .run = Ranks.run }, .{ .f = f, .keys = keys });
    try std.testing.expectEqual(@as(u64, 3), ranks.gathered);
    try std.testing.expectEqual(@as(usize, 9), log.items.items.len);
    const names = [_][]const u8{ "_bias", "aux.markov_race", "_pick" };
    const nc = cfg.vocab / tp;
    for (log.items.items, 0..) |k, j| {
        try std.testing.expectEqualStrings(names[j % 3], k.name);
        const i = j / 3;
        if (j % 3 != 1) continue;
        try std.testing.expectEqual([3]u32{ 3, 1, 1 }, k.grid);
        try std.testing.expectEqual(@as(u32, tri_aux.markov_race_block), k.block);
        const a = log.hipArgs(tri_aux.MarkovRaceArgs, j);
        // step i: its block row's logits, the tokens the last step picked, this rank's half of the vocabulary
        try std.testing.expectEqual(lg + 4 * i * nc, a.lg);
        try std.testing.expectEqual(out + 8 * i, a.tok);
        try std.testing.expectEqual(@as(i64, @intCast(cfg.dspark_block * nc)), a.l_row);
        try std.testing.expectEqual(@as(i64, 4), a.t_stride);
        try std.testing.expectEqual(m.slot, a.slot);
        try std.testing.expectEqual(m.cache, a.cache);
        try std.testing.expectEqual(sc.stage, a.stage);
        try std.testing.expectEqual(keys, a.keys);
        try std.testing.expectEqual(sc.send, a.send);
        try std.testing.expectEqual(@as(i32, 3), a.n_rows);
        try std.testing.expectEqual(@as(i32, @intCast(nc)), a.n_cols);
        try std.testing.expectEqual(@as(i32, @intCast(nc)), a.id0);
        try std.testing.expectEqual(@as(i32, @intCast(i)), a.step);
    }
    // a single rank (no gather to hand the bests to) keeps the Triton steps
    const one = try Markov.init(cfg, .{ .rank = 0, .world = 1 }, elsewhere, elsewhere, elsewhere);
    try std.testing.expect(!one.races());
}

test "raceGumbel is the sampler's draw of the same key bits" {
    // the sampler's -log(-log(uniform)) in float64, over the uniform's whole range and the key's ordinary bits
    const edge = [_]u64{ 0, 1 << 11, (1 << 62) - 1, 1 << 62, (1 << 63) - (1 << 11), 1 << 63, (1 << 63) + (1 << 11), std.math.maxInt(u64) - (1 << 11), std.math.maxInt(u64) };
    for (edge) |bits| {
        const want = -@log(-@log(@as(f64, @floatFromInt(bits >> 11)) * 0x1p-53 + 0x1p-54));
        // (the last value's uniform rounds to 1: +inf, there and here)
        if (std.math.isInf(want)) try std.testing.expect(std.math.isPositiveInf(raceGumbel(bits))) else try std.testing.expectApproxEqAbs(want, @as(f64, raceGumbel(bits)), 1e-5 * @max(1.0, @abs(want)));
    }
    try std.testing.expect(std.math.isPositiveInf(raceGumbel(std.math.maxInt(u64))));
    for (0..20000) |i| {
        const bits = sampling.keyBits(0x1234_5678_9abc, 100000 + i / 100, i);
        const want = -@log(-@log(sampling.uniform(0x1234_5678_9abc, 100000 + i / 100, i)));
        try std.testing.expectApproxEqAbs(want, @as(f64, raceGumbel(bits)), 1e-5 * @max(1.0, @abs(want)));
    }
}

test "raceRow: the served sampler's token on the same scores, the greedy token without a key" {
    const n = 48; // a row the sampler takes whole (top_k 48: every token a candidate)
    var lg: [n]f32 = undefined;
    var bias: [n]f16 = undefined;
    var sum: [n]f32 = undefined;
    var x: u64 = 0x9E3779B97F4A7C15;
    var compared: usize = 0;
    for (0..500) |r| {
        for (&lg, &bias, &sum) |*l, *b, *s| {
            x = x *% 6364136223846793005 +% 1442695040888963407;
            l.* = @as(f32, @floatFromInt(@as(i32, @intCast(x >> 44)) - (1 << 19))) / 131072.0; // -4 .. 4
            x = x *% 6364136223846793005 +% 1442695040888963407;
            b.* = @floatCast(@as(f32, @floatFromInt(@as(i32, @intCast(x >> 52)) - 2048)) / 2048.0); // -1 .. 1
            s.* = l.* + @as(f32, b.*);
        }
        const position: u64 = 5000 + r;
        const seed: u64 = 77 + r;
        // a greedy row: sampling.argmax of the scores
        const g = raceRow(&lg, &bias, .{}, position, 0);
        try std.testing.expectEqual(@as(u32, @intCast(sampling.argmax(&sum))), g[1]);
        try std.testing.expectEqual(sum[g[1]], g[0]);
        for ([_]f64{ 1.0, 0.7, 0.25 }) |temp| {
            for ([_]u32{ 0, 64640 }) |id0| {
                // the sampler's float64 scores of tokens id0 + c; a row whose two best are nearer than the fp32
                // draw's error may go either way
                var best = -std.math.inf(f64);
                var second = best;
                var at: u32 = 0;
                for (sum, 0..) |v, c| {
                    const sc = @as(f64, v) / temp - @log(-@log(sampling.uniform(seed, position, @as(u64, id0) + c)));
                    if (sc > best) {
                        second = best;
                        best = sc;
                        at = id0 + @as(u32, @intCast(c));
                    } else if (sc > second) second = sc;
                }
                if (best - second < 1e-3) continue;
                const got = raceRow(&lg, &bias, .{ .seed = seed, .position = position, .inv_t = @floatCast(1.0 / temp) }, position, id0);
                try std.testing.expectEqual(at, got[1]);
                try std.testing.expectApproxEqAbs(best, @as(f64, got[0]), 1e-3);
                compared += 1;
                // on token ids from 0 that is the served sampler's draw
                if (id0 == 0) try std.testing.expectEqual(@as(i64, at), try sampling.sampleRow(&sum, position, .{ .seed = seed, .temperature = temp, .top_k = n, .top_p = 1.0 }));
            }
        }
    }
    try std.testing.expect(compared > 2800);
    // _score's order (torch.argmax's), which a greedy row in a pass with sampled ones must keep: a NaN above every
    // value, the first of two; then the larger value, the first of equals
    const nan = std.math.nan(f32);
    @memset(&bias, 0);
    for (&lg, 0..) |*l, c| l.* = @floatFromInt(c % 7);
    lg[20] = 100;
    try std.testing.expectEqual(@as(u32, 20), raceRow(&lg, &bias, .{}, 9, 0)[1]);
    lg[17] = nan;
    lg[31] = nan;
    const g = raceRow(&lg, &bias, .{}, 9, 1000);
    try std.testing.expectEqual(@as(u32, 1017), g[1]);
    try std.testing.expect(std.math.isNan(g[0]));
    try std.testing.expectEqual(@as(u32, 1017), raceRow(&lg, &bias, .{ .seed = 1, .position = 9, .inv_t = 1 }, 9, 1000)[1]);
    lg[17] = 6;
    lg[31] = 6;
    lg[20] = 6;
    try std.testing.expectEqual(@as(u32, 6), raceRow(&lg, &bias, .{}, 9, 0)[1]); // 6, 13, 17, 20, ... tie at 6
}

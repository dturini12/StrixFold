//! w-tri's HIP kernels in place of Triton launches and hipBLAS GEMMs (HIP builds; TF_DS_AUX, default hip):
//!   aux.sparse_attn  for _sparse_attn_part: split partials (the Triton merge kernels unchanged) or one pass
//!   aux.index_rows   for the indexer's scores without a base or candidates (_index_score_rows; _index_score of a
//!                    prompt chunk of at most 16 rows)
//!   aux.hc_mix_pf    for _hc_mix_pf at every row count (a token's bits never depend on the chunk it landed in:
//!                    chunked prefill and kept prefixes; Triton's lead at 32-128 rows is at most 38 us a call)
//!   aux.gate_logits  (gate) for a prompt chunk's router gate past decode_rows rows: x bf16 . gate_w fp16^T into fp32
//!                    logits from the weight as stored, in place of toF32 + f16ToF32 + the hipBLAS sgemm (within
//!                    gamma(5120) of fp64, as the sgemm)
//!   aux.gate_logits  (idxw, its ne = 32 build) for the indexer's head weights past decode_rows rows: x . weights_proj^T
//!                    from the fp16 weight, in place of toF32 + the hipBLAS sgemm on the fp32 copy
//!   aux.rope_heads   (rope) for _rope_heads at every row count: _rope_heads' bits
//! and one kernel that is not a Triton launch's replacement: aux.markov_race, the drafter's Markov step of a pass with
//! sampled streams (tri_markov.raceBest; bound whenever the kit has it, used only when a pass carries keys).
//! A shape a kernel does not take keeps its Triton launch or GEMM (tri_attn.sparseAttn, tri_index.indexScore,
//! tri_hc.hcMixPf, tri_norm.gateLogits, tri_norm.ropeHeads ask first). The by-value arguments are abi.h's structs
//! (hip_load.zig checks the layouts). CUDA builds never set Tri.aux.
const std = @import("std");
const gpu = @import("gpu");

/// tf_ds_aux_attn_args. flags: 1 ring, 2 compressed keys, 4 packed FP4, 8 one pass, 16 per-row bases.
pub const AttnArgs = extern struct {
    q: u64,
    wsrc: u64,
    wlo: u64,
    comp: u64,
    csc: u64,
    idx: u64,
    pos: u64,
    pm: u64,
    pl: u64,
    po: u64,
    sink: u64,
    out: u64,
    rbase: u64,
    cbase: u64,
    ring_size: i32,
    n_idx: i32,
    rows: i32,
    heads: i32,
    win: i32,
    splits: i32,
    flags: i32,
    pad0: i32 = 0,
    scale: f32,
    pad1: i32 = 0,
};

/// tf_ds_aux_index_args. flags: 1 int64 keys, 2 (with keys) tile maxima into tmax.
pub const IndexArgs = extern struct { q: u64, k: u64, ks: u64, w: u64, vis: u64, out: u64, tmax: u64, rows: i32, n: i32, nt: i32, flags: i32 };

/// tf_ds_aux_hc_mix_args. flags: 1 posted (xo written from x, the gathered g, post and comb).
pub const HcMixArgs = extern struct { x: u64, xo: u64, g: u64, post: u64, comb: u64, @"fn": u64, part: u64, rs: i64, rows: i32, world: i32, flags: i32, pad0: i32 = 0 };

/// tf_ds_aux_gate_args: out fp32 [rows, ne] = x bf16 [rows, 5120] . w fp16 [ne, 5120]^T. flags 0.
pub const GateArgs = extern struct { x: u64, w: u64, out: u64, rows: i32, ne: i32, flags: i32 = 0, pad0: i32 = 0 };

/// tf_ds_aux_rope_heads_args: x bf16 [rows, h, hd] rotated in place on its last rd dims. flags: 1 inverse.
pub const RopeHeadsArgs = extern struct { x: u64, cos: u64, sin: u64, pos: u64, rows: i32, h: i32, hd: i32, rd: i32, flags: i32, pad0: i32 = 0 };

/// tf_ds_markov_key: a drafting row's key, its stream's sampling seed, the absolute position its first draft is drawn
/// at and 1 / its draft temperature (0: a greedy row, drafted as _score drafts it).
pub const MarkovKey = extern struct { seed: u64 = 0, position: u64 = 0, inv_t: f32 = 0, pad0: u32 = 0 };

/// tf_ds_aux_markov_race_args: send fp32 [n_rows, 4] = each row's best (value, token id, 0, 0) of this rank's n_cols
/// columns, score * inv_t + the key's Gumbel draw at position + step (a greedy row: the score).
pub const MarkovRaceArgs = extern struct { lg: u64, tok: u64, slot: u64, cache: u64, stage: u64, keys: u64, send: u64, l_row: i64, t_stride: i64, c_row: i64, s_row: i64, n_rows: i32, n_cols: i32, id0: i32, step: i32 };

/// aux.markov_race's block: a row a workgroup.
pub const markov_race_block = 1024;

/// The kernels' compiled shapes (registry.json's specs) and limits.
pub const attn_heads = 32;
pub const attn_hd = 512;
pub const attn_window = 128;
/// aux.sparse_attn reads at most this many picks a row (it clamps, so a longer list keeps the Triton launch).
pub const attn_max_idx = 512;
pub const index_heads = 32;
pub const index_dim = 128;
/// aux.index_rows: rows a workgroup, and about this many workgroups in all (grid y splits the key tiles).
pub const index_rows_wg = 8;
pub const index_target_wgs = 240;
pub const hc_d = 5120;
pub const hc_nb = 40;
/// aux.hc_mix_pf: rows a workgroup, and the column blocks of a row group.
pub const hc_rows_wg = 32;
pub const hc_cols = 10;
/// aux.gate_logits: the row width both builds are compiled for; the router build's experts a workgroup (ne a multiple
/// of it) and rows a workgroup; the indexer build's ne, rows a workgroup and block.
pub const gate_d = 5120;
pub const gate_bn = 64;
pub const gate_rows_wg = 128;
pub const gate32_ne = 32;
pub const gate32_rows_wg = 32;
pub const gate32_block = 128;
/// aux.rope_heads: pairs a thread (rd a multiple of 2 x this; 16-byte loads of x, cos and sin).
pub const rope_pairs = 4;

/// Which of them take their HIP kernel (the field names are TF_DS_AUX's list items).
pub const Which = struct {
    attn: bool = true,
    index: bool = true,
    hc: bool = true,
    gate: bool = true,
    idxw: bool = true,
    rope: bool = true,

    pub const triton: Which = .{ .attn = false, .index = false, .hc = false, .gate = false, .idxw = false, .rope = false };

    /// TF_DS_AUX's value: hip (all), triton (none), or a comma list of attn, index, hc, gate, idxw, rope (those on HIP).
    pub fn parse(spec: []const u8) error{BadSetting}!Which {
        if (std.mem.eql(u8, spec, "hip")) return .{};
        if (std.mem.eql(u8, spec, "triton")) return triton;
        var w = triton;
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |item| {
            var known = false;
            inline for (@typeInfo(Which).@"struct".field_names) |f| if (std.mem.eql(u8, item, f)) {
                @field(w, f) = true;
                known = true;
            };
            if (!known) return error.BadSetting;
        }
        return w;
    }

    /// TF_DS_AUX from the environment (unset: hip).
    pub fn fromEnv() error{BadSetting}!Which {
        const v = std.c.getenv("TF_DS_AUX") orelse return .{};
        return parse(std.mem.span(v));
    }

    /// The selection as the handshake's plan and the logs name it: hip, triton, or the HIP ones listed (buf: 32 bytes).
    pub fn name(w: Which, buf: []u8) []const u8 {
        if (std.meta.eql(w, Which{})) return "hip";
        if (std.meta.eql(w, triton)) return "triton";
        var n: usize = 0;
        inline for (@typeInfo(Which).@"struct".field_names) |f| if (@field(w, f)) {
            if (n > 0) {
                buf[n] = ',';
                n += 1;
            }
            @memcpy(buf[n..][0..f.len], f);
            n += f.len;
        };
        return buf[0..n];
    }
};

/// The bound kernels (null: that launch stays Triton, or the GEMM hipBLAS).
pub const Aux = struct {
    attn: ?gpu.Function = null,
    index: ?gpu.Function = null,
    hc_mix: ?gpu.Function = null,
    gate: ?gpu.Function = null, // aux.gate_logits, ne a multiple of 64 (the router)
    gate32: ?gpu.Function = null, // aux.gate_logits, ne = 32 (the indexer's head weights)
    rope: ?gpu.Function = null,
    markov_race: ?gpu.Function = null, // not one of Which's: a sampled pass's Markov step (no Triton form)
};

/// aux.index_rows' grid: groups of 8 rows by key-tile ranges, about 240 workgroups in all.
pub fn indexGrid(rows: usize, nt: usize) [3]u32 {
    const groups = (rows + index_rows_wg - 1) / index_rows_wg;
    const splits = @max(1, @min(nt, index_target_wgs / groups));
    return .{ @intCast(groups), @intCast(splits), 1 };
}

/// aux.hc_mix_pf's grid: 10 column blocks a group of 32 rows.
pub fn hcGrid(rows: usize) [3]u32 {
    return .{ @intCast(hc_cols * ((rows + hc_rows_wg - 1) / hc_rows_wg)), 1, 1 };
}

/// aux.gate_logits' grid: the router build's 128 rows by 64 experts a workgroup, or the indexer build's 32 rows.
pub fn gateGrid(rows: usize, ne: usize) [3]u32 {
    if (ne == gate32_ne) return .{ @intCast((rows + gate32_rows_wg - 1) / gate32_rows_wg), 1, 1 };
    return .{ @intCast((rows + gate_rows_wg - 1) / gate_rows_wg), @intCast(ne / gate_bn), 1 };
}

/// aux.rope_heads' grid: a thread four pairs of a row's head, 256 threads a workgroup.
pub fn ropeGrid(rows: usize, h: usize, rd: usize) [3]u32 {
    return .{ @intCast((rows * h * (rd / (2 * rope_pairs)) + 255) / 256), 1, 1 };
}

test "TF_DS_AUX: hip, triton, or the kernels listed" {
    try std.testing.expectEqual(Which{}, try Which.parse("hip"));
    try std.testing.expectEqual(Which.triton, try Which.parse("triton"));
    try std.testing.expectEqual(Which{ .attn = true, .index = false, .hc = true, .gate = false, .idxw = false, .rope = false }, try Which.parse("attn,hc"));
    try std.testing.expectEqual(Which{ .attn = true, .index = true, .hc = true, .gate = false, .idxw = false, .rope = false }, try Which.parse("attn,index,hc"));
    try std.testing.expectEqual(Which{ .attn = false, .index = false, .hc = false, .gate = true, .idxw = false, .rope = true }, try Which.parse("rope,gate"));
    try std.testing.expectError(error.BadSetting, Which.parse("attn,mla"));
    try std.testing.expectError(error.BadSetting, Which.parse(""));
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("index,hc,gate,idxw,rope", (Which{ .attn = false }).name(&b));
    try std.testing.expectEqualStrings("attn,index,hc", (try Which.parse("attn,index,hc")).name(&b));
    try std.testing.expectEqualStrings("hip", (Which{}).name(&b));
    try std.testing.expectEqualStrings("triton", Which.triton.name(&b));
    // the longest list (all but one) fits the 32-byte buffers the logs and the handshake name it in
    try std.testing.expectEqualStrings("attn,index,hc,gate,idxw", (Which{ .rope = false }).name(&b));
}

test "the grids follow w-tri's launches" {
    // 2048 prompt rows over 8192 keys (128 tiles): 256 groups, one range each; one row: 128 tiles over 128 ranges
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, indexGrid(2048, 128));
    try std.testing.expectEqual([3]u32{ 1, 128, 1 }, indexGrid(1, 128));
    try std.testing.expectEqual([3]u32{ 3, 80, 1 }, indexGrid(17, 128));
    try std.testing.expectEqual([3]u32{ 10, 1, 1 }, hcGrid(1));
    try std.testing.expectEqual([3]u32{ 640, 1, 1 }, hcGrid(2048));
    // the router gate: (ceil(rows / 64), ne / 128); the indexer's weights (ne 32): ceil(rows / 32)
    try std.testing.expectEqual([3]u32{ 16, 6, 1 }, gateGrid(2048, 384));
    try std.testing.expectEqual([3]u32{ 1, 6, 1 }, gateGrid(17, 384));
    try std.testing.expectEqual([3]u32{ 64, 1, 1 }, gateGrid(2048, 32));
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, gateGrid(24, 32));
    // rope_heads: rows x heads x rd / 8 threads (aux_rope.hip: ceil(rows h rd / 8 / 256))
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, ropeGrid(1, 32, 64));
    try std.testing.expectEqual([3]u32{ 2048, 1, 1 }, ropeGrid(2048, 32, 64));
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, ropeGrid(6, 1, 64));
}

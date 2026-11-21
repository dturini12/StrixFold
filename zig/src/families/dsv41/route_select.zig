//! aux.route_pick's selection (aux_route.hip) beside _route_parts' 512-lane loop, literally: the host equivalence test.
const std = @import("std");

pub const ne = 384;
pub const nb = 512;
pub const waves = ne / 32;
pub const max_k = 8;

pub const Out = struct {
    pick: [max_k]i32 = @splat(0),
    wts: [max_k]f32 = @splat(0),
    /// kernel: took the literal loop; triton: a round among the first topk found best == -inf
    degenerate: bool = false,
};

fn maxnum(a: f32, b: f32) f32 {
    if (std.math.isNan(a)) return b;
    if (std.math.isNan(b)) return a;
    return if (b > a) b else a;
}

/// _route_parts' selection over the 512 lanes, literally.
pub fn triton(sel_in: *const [ne]f32, sc_in: *const [ne]f32, sc_pad: f32, topk: usize, slots: usize, scale: f32, shared_id: i32) Out {
    var sel: [nb]f32 = undefined;
    var sc: [nb]f32 = undefined;
    for (0..nb) |e| {
        sel[e] = if (e < ne) sel_in[e] else -std.math.inf(f32);
        sc[e] = if (e < ne) sc_in[e] else sc_pad;
    }
    var o: Out = .{};
    var picks: [max_k]i32 = @splat(shared_id);
    var ws: [max_k]f32 = @splat(0);
    var tot: f32 = 0.0;
    for (0..topk) |k| {
        var best: f32 = sel[0];
        for (sel[1..]) |x| best = maxnum(best, x);
        if (best == -std.math.inf(f32)) o.degenerate = true;
        var idx: usize = nb;
        for (0..nb) |e| if (sel[e] == best) {
            idx = @min(idx, e);
        };
        var v: f32 = 0.0;
        for (0..nb) |e| v += if (e == idx) sc[e] else 0.0;
        picks[k] = @intCast(idx);
        ws[k] = v;
        tot += v;
        for (0..nb) |e| if (e == idx) {
            sel[e] = -std.math.inf(f32);
        };
    }
    for (0..slots) |s| {
        o.pick[s] = picks[s];
        o.wts[s] = if (s < topk) ws[s] / (tot + 1e-20) * scale else 1.0;
    }
    return o;
}

/// a beats b: greater sel first, equal sel lower index first (NaN never beats, never is beaten)
fn beats(a: f32, ia: usize, b: f32, ib: usize) bool {
    return a > b or (a == b and ia < ib);
}

/// tf_ds_aux_route_pick_ne384's selection for one row, step for step.
pub fn kernel(sel: *const [ne]f32, sc: *const [ne]f32, sc_pad: f32, topk: usize, slots: usize, scale: f32, shared_id: i32) Out {
    var o: Out = .{};
    // each wave's best valid element: (valid ? sel : -inf, index) per lane, then the shuffles' xor butterfly 16..1
    var wv: [waves]f32 = undefined;
    var wi: [waves]usize = undefined;
    var n_valid: usize = 0;
    for (0..waves) |w| {
        var bv: [32]f32 = undefined;
        var bi: [32]usize = undefined;
        for (0..32) |l| {
            const e = w * 32 + l;
            const valid = sel[e] > -std.math.inf(f32);
            n_valid += @intFromBool(valid);
            bv[l] = if (valid) sel[e] else -std.math.inf(f32);
            bi[l] = e;
        }
        var off: usize = 16;
        while (off > 0) : (off >>= 1) {
            const pv = bv;
            const pi = bi;
            for (0..32) |l| if (beats(pv[l ^ off], pi[l ^ off], pv[l], pi[l])) {
                bv[l] = pv[l ^ off];
                bi[l] = pi[l ^ off];
            };
        }
        wv[w] = bv[0];
        wi[w] = bi[0];
    }
    if (n_valid >= topk) {
        var s_pick: [max_k]usize = @splat(std.math.maxInt(usize));
        for (0..ne) |e| {
            if (!(sel[e] > -std.math.inf(f32))) continue;
            var above: usize = 0;
            for (0..waves) |w| above += @intFromBool(beats(wv[w], wi[w], sel[e], e));
            if (above >= topk) continue;
            var rank: usize = 0;
            for (0..ne) |j| rank += @intFromBool(beats(sel[j], j, sel[e], e));
            if (rank < topk) {
                std.debug.assert(s_pick[rank] == std.math.maxInt(usize)); // ranks are distinct
                s_pick[rank] = e;
            }
        }
        var v: [max_k]f32 = undefined;
        var tot: f32 = 0.0;
        for (0..topk) |k| {
            v[k] = sc[s_pick[k]] + 0.0;
            tot += v[k];
        }
        const den = tot + 1e-20;
        for (0..slots) |s| {
            o.pick[s] = if (s < topk) @intCast(s_pick[s]) else shared_id;
            o.wts[s] = if (s < topk) v[s] / den * scale else 1.0;
        }
        return o;
    }
    // the degenerate row: thread 0's literal loop (aux_route.hip)
    o.degenerate = true;
    var s_sel = sel.*;
    var pk: [max_k]usize = undefined;
    var v: [max_k]f32 = undefined;
    var tot: f32 = 0.0;
    for (0..topk) |k| {
        var best: f32 = -std.math.inf(f32);
        for (s_sel) |x| best = maxnum(best, x);
        var idx: usize = nb;
        for (0..ne) |j| if (s_sel[j] == best) {
            idx = j;
            break;
        };
        if (idx == nb) idx = ne;
        pk[k] = idx;
        v[k] = (if (idx < ne) sc[idx] else sc_pad) + 0.0;
        tot += v[k];
        if (idx < ne) s_sel[idx] = -std.math.inf(f32);
    }
    const den = tot + 1e-20;
    for (0..slots) |s| {
        o.pick[s] = if (s < topk) @intCast(pk[s]) else shared_id;
        o.wts[s] = if (s < topk) v[s] / den * scale else 1.0;
    }
    return o;
}

// -- the equivalence test --------------------------------------------------------------------------------------

const sc_pad_value: f32 = 0x1.aa4182p-1; // about sqrt(ln 2): the score of lg +0 (any value: both read the same one)

fn bitsEqual(a: Out, b: Out, slots: usize) bool {
    for (0..slots) |s| {
        if (a.pick[s] != b.pick[s]) return false;
        if (@as(u32, @bitCast(a.wts[s])) != @as(u32, @bitCast(b.wts[s]))) return false;
    }
    return a.degenerate == b.degenerate;
}

const Regime = enum { normal, ties, signed_zero, pos_inf, neg_inf_bias, nan_some, few_finite, all_nan, all_neg_inf, mixed };

fn fill(r: std.Random, regime: Regime, sel: *[ne]f32, sc: *[ne]f32, few: usize) void {
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    for (0..ne) |e| {
        var s: f32 = @abs(@as(f32, @floatCast(r.floatNorm(f64)))) * 0.3;
        var b: f32 = @as(f32, @floatCast(r.floatNorm(f64))) * 0.1;
        switch (regime) {
            .normal => {},
            .ties => {
                s = @as(f32, @floatFromInt(r.uintLessThan(u32, 4))) * 0.25;
                b = if (r.boolean()) 0.0 else 0.125;
            },
            .signed_zero => {
                s = if (r.uintLessThan(u32, 4) == 0) 0.5 else 0.0;
                b = if (r.boolean()) -0.0 else 0.0;
            },
            .pos_inf => if (r.uintLessThan(u32, 50) == 0) {
                s = inf;
            },
            .neg_inf_bias => if (r.uintLessThan(u32, 3) == 0) {
                b = -inf;
            },
            .nan_some => switch (r.uintLessThan(u32, 40)) {
                0 => s = nan,
                1 => b = nan,
                2 => b = -inf,
                else => {},
            },
            .few_finite => {},
            .all_nan => s = nan,
            .all_neg_inf => b = -inf,
            .mixed => switch (r.uintLessThan(u32, 8)) {
                0 => s = nan,
                1 => b = -inf,
                2 => s = inf,
                3 => b = -0.0,
                4 => s = 0.0,
                else => {},
            },
        }
        sc[e] = s;
        sel[e] = s + b;
    }
    if (regime == .few_finite) {
        // exactly `few` elements above -inf at random places; the others -inf or NaN
        for (0..ne) |e| sel[e] = if (r.boolean()) -inf else nan;
        var placed: usize = 0;
        while (placed < few) {
            const e = r.uintLessThan(usize, ne);
            if (sel[e] > -inf) continue;
            sel[e] = sc[e] + 0.05;
            placed += 1;
        }
    }
}

test "aux.route_pick's selection equals _route_parts' 512-lane loop: picks, order, weights' bits, the degenerate decision" {
    var prng: std.Random.DefaultPrng = .init(0x7f2_0517);
    const r = prng.random();
    var sel: [ne]f32 = undefined;
    var sc: [ne]f32 = undefined;
    var counts: [@typeInfo(Regime).@"enum".field_names.len]usize = @splat(0);
    var degenerate: usize = 0;
    const shapes = [_][2]usize{ .{ 6, 7 }, .{ 6, 8 }, .{ 8, 8 }, .{ 1, 2 }, .{ 3, 4 }, .{ 6, 6 }, .{ 6, 3 } };
    for (0..1500) |i| {
        const regime: Regime = @enumFromInt(i % counts.len);
        const shape = if (i % 5 == 0) shapes[r.uintLessThan(usize, shapes.len)] else shapes[0];
        const few = r.uintLessThan(usize, shape[0] + 3); // around topk: 0 .. topk + 2
        fill(r, regime, &sel, &sc, few);
        const scale: f32 = 1.5;
        const a = triton(&sel, &sc, sc_pad_value, shape[0], shape[1], scale, 384);
        const b = kernel(&sel, &sc, sc_pad_value, shape[0], shape[1], scale, 384);
        if (!bitsEqual(a, b, shape[1])) {
            std.debug.print("row {d} regime {s} topk {d} slots {d}: triton {any} {any} {} kernel {any} {any} {}\n", .{ i, @tagName(regime), shape[0], shape[1], a.pick[0..shape[1]], a.wts[0..shape[1]], a.degenerate, b.pick[0..shape[1]], b.wts[0..shape[1]], b.degenerate });
            return error.TestUnexpectedResult;
        }
        counts[@intFromEnum(regime)] += 1;
        degenerate += @intFromBool(a.degenerate);
    }
    // every regime ran, and the degenerate rows (all NaN, all -inf, fewer than topk above -inf) were reached
    for (counts) |c| try std.testing.expect(c > 0);
    try std.testing.expect(degenerate >= 300);
}

test "the selection on hand-made rows: ties to the lower index, -0 equals +0, NaN never picked, pads repeat" {
    const inf = std.math.inf(f32);
    var sel: [ne]f32 = @splat(-inf);
    var sc: [ne]f32 = @splat(0.25);
    // ties: experts 7, 3, 300 share the best value; 5 holds -0 and 9 holds +0 (equal); 11 is NaN
    sel[7] = 2.0;
    sel[3] = 2.0;
    sel[300] = 2.0;
    sel[5] = -0.0;
    sel[9] = 0.0;
    sel[11] = std.math.nan(f32);
    sel[100] = 1.0;
    const a = kernel(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384);
    try std.testing.expect(!a.degenerate);
    try std.testing.expectEqualSlices(i32, &.{ 3, 7, 300, 100, 5, 9, 384 }, a.pick[0..7]);
    try std.testing.expect(bitsEqual(a, triton(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384), 7));
    // only 4 valid: the literal loop; after them the first -inf expert (0), picked again and again (masking keeps -inf)
    sel[5] = -inf;
    sel[9] = -inf;
    const b = kernel(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384);
    try std.testing.expect(b.degenerate);
    try std.testing.expectEqualSlices(i32, &.{ 3, 7, 300, 100, 0, 0, 384 }, b.pick[0..7]);
    try std.testing.expect(bitsEqual(b, triton(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384), 7));
    // every sel NaN: no expert equals the best (-inf, the pads'): the first pad (384), with the pads' score
    for (&sel) |*x| x.* = std.math.nan(f32);
    const c = kernel(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384);
    try std.testing.expectEqualSlices(i32, &.{ 384, 384, 384, 384, 384, 384, 384 }, c.pick[0..7]);
    var tot: f32 = 0.0;
    for (0..6) |_| tot += sc_pad_value + 0.0;
    try std.testing.expectEqual(sc_pad_value / (tot + 1e-20) * 1.5, c.wts[0]);
    try std.testing.expect(bitsEqual(c, triton(&sel, &sc, sc_pad_value, 6, 7, 1.5, 384), 7));
}

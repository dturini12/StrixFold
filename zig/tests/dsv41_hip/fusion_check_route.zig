//! tf-dsv41-fusion-check's route step: tri_norm.route with TF_DS_AUX=hip (_route_parts) vs hip,route (aux.route_pick).

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const tri_norm = dsv41.tri_norm;
const cm = @import("fusion_check_common.zig");
const Env = cm.Env;
const Report = cm.Report;
const Msg = cm.Msg;
const Rng = cm.Rng;
const ne = cm.ne;
const kc = cm.kc;
const slots = cm.slots;
const max_rows = cm.max_rows;

/// The round's routes a graph replay (one a layer).
pub const launches = 40;

const Regime = enum { normal, below, wide, negative };

/// tools/dsv41-hip/tri/t_route.py's REGIMES: a row's logit drawn from each.
fn regimeDraw(rng: *Rng, g: Regime) f64 {
    return switch (g) {
        .normal => rng.normal() * 1.5,
        .below => rng.uniform(-40, -17),
        .wide => rng.uniform(-100, 30),
        .negative => rng.normal() * 4 - 12,
    };
}

/// Special rows: logits that make ties, infinities, NaN, or fewer than six finite selection values.
const Special = enum { none, ties, tie_zero, pos_inf, neg_inf, inf_nan, nan_some, nan_six, nan_all };

const Case = struct {
    name: []const u8,
    rows: usize,
    /// Rows of S before the launch's first (prompt.zig routeSpans offsets S alone; PICK and WTS are the span's own).
    off: usize,
    bias: usize,
    regime: Regime,
    special: Special,
};

const bias_names = [_][]const u8{ "bias N(0,0.1)", "bias coarse, ±0, two -inf", "bias -inf but 4" };

/// The three biases: N(0, 0.1); coarse (ties) with ±0 and two -inf; -inf except 4 experts (every row degenerate).
fn makeBiases(rng: *Rng, out: *[3][ne]f32) void {
    for (&out[0]) |*x| x.* = @floatCast(rng.normal() * 0.1);
    for (&out[1], 0..) |*x, i| {
        const q: f32 = @floatFromInt(@as(i64, @intCast(rng.below(7))) - 3);
        x.* = q * 0.125;
        if (i % 37 == 0) x.* = 0.0;
        if (i % 41 == 0) x.* = -0.0;
    }
    out[1][5] = -std.math.inf(f32);
    out[1][200] = -std.math.inf(f32);
    for (&out[2]) |*x| x.* = -std.math.inf(f32);
    for ([_]usize{ 3, 90, 91, 383 }) |i| out[2][i] = @floatCast(rng.normal() * 0.1);
}

/// One row's chunk sums [kc, ne] (t_route.py: the row's logit over kc plus N(0, 0.35) a chunk), then `special`.
fn fillRow(rng: *Rng, row: []f32, regime: Regime, special: Special) void {
    for (0..ne) |ex| {
        const lg = regimeDraw(rng, regime);
        for (0..kc) |c| row[c * ne + ex] = @floatCast(lg / kc + rng.normal() * 0.35);
    }
    const inf = std.math.inf(f32);
    const nan = std.math.nan(f32);
    switch (special) {
        .none => {},
        // every expert's chunks equal: equal scores, the bias decides, equal bias the lower index
        .ties => for (0..kc) |c| {
            const v: f32 = @floatCast(rng.normal() * 0.1);
            for (0..ne) |ex| row[c * ne + ex] = v;
        },
        // logits far below: every score 0 (or its smallest), ties at the bias' ±0
        .tie_zero => @memset(row, -200.0 / @as(f32, kc)),
        .pos_inf => for ([_]usize{ 7, 8, 300 }) |ex| {
            row[rng.below(kc) * ne + ex] = inf;
        },
        .neg_inf => for (0..ne / 2) |k| {
            row[rng.below(kc) * ne + 2 * k] = -inf;
        },
        // +inf and -inf in one expert's chunks: its sum is NaN
        .inf_nan => for ([_]usize{ 1, 2, 250 }) |ex| {
            row[ex] = inf;
            row[ne + ex] = -inf;
        },
        .nan_some => for (0..1 + rng.below(9)) |_| {
            row[rng.below(kc) * ne + rng.below(ne)] = nan;
        },
        // NaN in all but five experts: fewer than six finite selection values
        .nan_six => for (0..ne) |ex| {
            if (ex % 77 != 0) row[rng.below(kc) * ne + ex] = nan;
        },
        .nan_all => @memset(row, nan),
    }
}

/// Every case the step runs (the dry run counts them).
pub fn cases(a: std.mem.Allocator) ![]Case {
    var list: std.ArrayList(Case) = .empty;
    for (1..max_rows + 1) |rows| for (std.enums.values(Regime)) |g| for (0..2) |b| {
        try list.append(a, .{ .name = @tagName(g), .rows = rows, .off = 0, .bias = b, .regime = g, .special = .none });
    };
    // the special rows: in every row at 1 and 16 rows, in every other row at 6 (mixed with ordinary rows)
    for ([_]usize{ 1, 6, 16 }) |rows| for (std.enums.values(Special)[1..]) |sp| for (0..2) |b| {
        try list.append(a, .{ .name = @tagName(sp), .rows = rows, .off = 0, .bias = b, .regime = .normal, .special = sp });
    };
    for ([_]usize{ 1, 7, 16 }) |rows| for (std.enums.values(Regime)) |g| {
        try list.append(a, .{ .name = "degenerate bias", .rows = rows, .off = 0, .bias = 2, .regime = g, .special = .none });
    };
    // an image span: the launch's rows start o[0] rows into the chunk's S
    for ([_]usize{ 1, 5, 13 }) |rows| for (0..2) |b| {
        try list.append(a, .{ .name = "row offset", .rows = rows, .off = 3, .bias = b, .regime = .normal, .special = .none });
        try list.append(a, .{ .name = "row offset, NaN rows", .rows = rows, .off = 3, .bias = b, .regime = .wide, .special = .nan_some });
    };
    return list.toOwnedSlice(a);
}

const row_floats = kc * ne;
/// Rows of PICK / WTS checked past the launch's last (nothing may write there).
const guard_rows = 4;
const names = [_][]const u8{ "_route_parts", "aux.route_pick" };

const Bufs = struct {
    s: gpu.DeviceBuffer,
    bias: gpu.DeviceBuffer,
    pick: [2]gpu.DeviceBuffer,
    wts: [2]gpu.DeviceBuffer,
    /// S's rows: the offset rows, the launch's and the guard rows.
    const span_rows = 3 + max_rows + guard_rows;
    const words = span_rows * slots;

    fn init(e: *const Env) !Bufs {
        return .{
            .s = try e.alloc(span_rows * row_floats * 4),
            .bias = try e.alloc(3 * ne * 4),
            .pick = .{ try e.alloc(words * 4), try e.alloc(words * 4) },
            .wts = .{ try e.alloc(words * 4), try e.alloc(words * 4) },
        };
    }

    fn deinit(b: *Bufs) void {
        b.s.free();
        b.bias.free();
        for (&b.pick) |*x| x.free();
        for (&b.wts) |*x| x.free();
    }
};

/// tri_norm.route as the round calls it, `on` choosing TF_DS_AUX=hip,route over hip.
fn route(e: *const Env, on: bool, s: u64, bias: u64, pick: u64, wts: u64, rows: usize) !void {
    try tri_norm.route(e.t(on), s, kc, bias, cm.topk, cm.route_scale, cm.shared_id, pick, wts, rows, ne, slots);
}

/// One case through both paths (k 0 off, k 1 on), each into its own PICK / WTS filled alike; returns differing words.
fn runCase(e: *const Env, b: *Bufs, c: Case, rng: *Rng, host_s: []f32, msg: *Msg, launch_bad: *bool) !usize {
    @memset(host_s, 0);
    for (0..c.rows) |i| {
        const sp: Special = if (c.special != .none and (c.rows == 1 or i % 2 == 0 or c.rows == 16)) c.special else .none;
        fillRow(rng, host_s[(c.off + i) * row_floats ..][0..row_floats], c.regime, sp);
    }
    try e.put(b.s, 0, host_s[0 .. (c.off + c.rows) * row_floats]);
    for (0..2) |k| {
        try e.fill(b.pick[k].ptr, 0xC3, Bufs.words * 4);
        try e.fill(b.wts[k].ptr, 0xC3, Bufs.words * 4);
        const C = cm.Counts(&names);
        const c0 = C.take(e.kit);
        try route(e, k == 1, b.s.ptr + c.off * row_floats * 4, b.bias.ptr + c.bias * ne * 4, b.pick[k].ptr, b.wts[k].ptr, c.rows);
        const ran = C.take(e.kit).since(c0);
        // off: one _route_parts and no aux.route_pick; on: the reverse
        const want: [2]u64 = if (k == 0) .{ 1, 0 } else .{ 0, 1 };
        if (!std.mem.eql(u64, &ran, &want)) {
            if (!launch_bad.*) msg.put("{s} path launched _route_parts {d}x, aux.route_pick {d}x ({s} rows {d}); ", .{ if (k == 0) "off" else "on", ran[0], ran[1], c.name, c.rows });
            launch_bad.* = true;
        }
    }
    var hp: [2][Bufs.words]u32 = undefined;
    var hw: [2][Bufs.words]u32 = undefined;
    for (0..2) |k| {
        try e.get(b.pick[k], 0, &hp[k]);
        try e.get(b.wts[k], 0, &hw[k]);
    }
    var bad: usize = 0;
    for (0..Bufs.words) |i| {
        const pd = hp[0][i] != hp[1][i];
        const wd = hw[0][i] != hw[1][i];
        if (!pd and !wd) continue;
        if (bad == 0) {
            const row = i / slots;
            msg.put("first difference: case \"{s}\" rows {d} {s}, row {d}{s} slot {d}: pick {d} vs {d}, wts 0x{x:0>8} vs 0x{x:0>8}; ", .{ c.name, c.rows, bias_names[c.bias], row, if (row >= c.rows) " (a guard row)" else "", i % slots, @as(i32, @bitCast(hp[0][i])), @as(i32, @bitCast(hp[1][i])), hw[0][i], hw[1][i] });
        }
        bad += @intFromBool(pd) + @intFromBool(wd);
    }
    return bad;
}

pub fn step(e: *const Env, r: *Report) !void {
    var msg: Msg = .{};
    const aux_ok = cm.auxObjectOk(e.kit.m, &msg, e.o.aux_object);
    r.check("route.aux_object", aux_ok, msg.text());
    if (!aux_ok) return error.AuxObject;
    if (e.aux_on.route == null or e.aux_off.route != null) {
        r.check("route.binding", false, "TF_DS_AUX=hip,route must bind aux.route_pick and hip must not");
        return error.Binding;
    }
    var b = try Bufs.init(e);
    defer b.deinit();
    var rng: Rng = .{ .s = e.o.seed ^ 0x726f757465 };
    var biases: [3][ne]f32 = undefined;
    makeBiases(&rng, &biases);
    try e.put(b.bias, 0, std.mem.asBytes(&biases));
    const all = try cases(e.gpa);
    defer e.gpa.free(all);
    const host_s = try e.gpa.alloc(f32, Bufs.span_rows * row_floats);
    defer e.gpa.free(host_s);
    msg = .{};
    var bad: usize = 0;
    var launch_bad = false;
    const before = try cm.entryCounts(e.gpa, e.kit);
    defer e.gpa.free(before);
    for (all) |c| bad += try runCase(e, &b, c, &rng, host_s, &msg, &launch_bad);
    try baseline(e, r, before);
    msg.put("{d} cases (rows 1-16, 4 regimes x 2 biases; ties, ±inf, NaN, degenerate rows, row offset): {d} words of PICK/WTS (and guard rows) differ", .{ all.len, bad });
    r.check("route.bits", bad == 0 and !launch_bad, msg.text());

    // timing: the round's 40 routes captured as one graph, replayed back to back
    for (cm.timed_rows) |rows| {
        const host = host_s[0 .. rows * row_floats];
        for (0..rows) |i| fillRow(&rng, host[i * row_floats ..][0..row_floats], .normal, .none);
        try e.put(b.s, 0, host);
        var us: [2]f64 = undefined;
        for (0..2) |k| us[k] = try cm.timeGraph(e, e.o.replays, Graph{ .e = e, .b = &b, .on = k == 1, .rows = rows }) * 1000.0 / launches;
        const gain = (us[0] - us[1]) * launches / 1000.0;
        r.timing("\"timing\": \"route\", \"rows\": {d}, \"launches\": {d}, \"replays\": {d}, \"route_parts_us\": {d:.3}, \"route_pick_us\": {d:.3}", .{ rows, launches, e.o.replays, us[0], us[1] }, rows, gain);
    }
}

/// The off path's _route_parts launches all came from the deployed object (--route-object), and some did.
fn baseline(e: *const Env, r: *Report, before: []const u64) !void {
    var msg: Msg = .{};
    const want = e.o.route_object.?;
    var from_it: u64 = 0;
    var elsewhere: u64 = 0;
    for (e.kit.entries, before, 0..) |entry, b0, i| {
        const tt = entry.k.triton orelse continue;
        if (!std.mem.eql(u8, tt.@"fn", "_route_parts")) continue;
        const n = e.kit.launches(@intCast(i)) - b0;
        if (n == 0) continue;
        const sha = cm.objectSha(e.kit.m, entry.k.object) orelse "?";
        if (std.mem.eql(u8, sha, want)) from_it += n else {
            if (elsewhere == 0) msg.put("_route_parts launched {d}x from {s} (sha256 {s}), not the deployed object; ", .{ n, entry.k.object, sha });
            elsewhere += n;
        }
    }
    msg.put("the off path launched _route_parts {d}x from the deployed object {s}, {d}x from another", .{ from_it, want, elsewhere });
    r.check("route.baseline", from_it > 0 and elsewhere == 0, msg.text());
}

const Graph = struct {
    e: *const Env,
    b: *Bufs,
    on: bool,
    rows: usize,

    pub fn record(g: Graph) !void {
        const k: usize = @intFromBool(g.on);
        for (0..launches) |l| try route(g.e, g.on, g.b.s.ptr, g.b.bias.ptr + (l % 2) * ne * 4, g.b.pick[k].ptr, g.b.wts[k].ptr, g.rows);
    }
};

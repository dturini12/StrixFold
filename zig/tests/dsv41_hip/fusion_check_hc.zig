//! tf-dsv41-fusion-check's hc step: tri_hc.hcPre2 as round.zig's mix calls it, tri_hc.inline_mix off vs on.

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const tri_hc = dsv41.tri_hc;
const cm = @import("fusion_check_common.zig");
const Env = cm.Env;
const Report = cm.Report;
const Msg = cm.Msg;
const Rng = cm.Rng;
const hc_d = cm.hc_d;
const hc_nb = cm.hc_nb;
const max_rows = cm.max_rows;
const world = cm.world;

/// A round's mixes (80), unposted where no gather is pending: the first, and after the two Engram layers' posts.
pub const round_sites = 80;
const round_unposted = [_]usize{ 0, 2, 28 };

/// The identity chains, a site a letter: 'U' unposted, 'P' posted (each site's post / comb feed the next post).
const chains = [_][]const u8{ "UPP", "PUP" };
pub const runs = max_rows * chains.len * 2;
const names = [_][]const u8{ "_hc_post_only", "_hc_mix_part", "_hc_finish_s", "_hc_sinkhorn", "_hc_fused" };

/// The outputs a site snapshots, in this order.
const Field = enum { h_out, x, part, pre_out, post, comb };

fn fieldBytes(f: Field, rows: usize) usize {
    return switch (f) {
        .h_out => rows * 4 * hc_d * 2,
        .x => rows * hc_d * 2,
        .part => rows * hc_nb * 32 * 4,
        .pre_out, .post => rows * 4 * 4,
        .comb => rows * 16 * 4,
    };
}

fn snapBytes(rows: usize) usize {
    var n: usize = 0;
    for (std.enums.values(Field)) |f| n += fieldBytes(f, rows);
    return n;
}

const Bufs = struct {
    h: [2]gpu.DeviceBuffer,
    g: [3]gpu.DeviceBuffer,
    fnw: [3]gpu.DeviceBuffer,
    scale: [3]gpu.DeviceBuffer,
    base: [3]gpu.DeviceBuffer,
    normw: [3]gpu.DeviceBuffer,
    pre: [2]gpu.DeviceBuffer,
    x: gpu.DeviceBuffer,
    post: gpu.DeviceBuffer,
    comb: gpu.DeviceBuffer,
    part: gpu.DeviceBuffer,
    /// [mode]: a chain's snapshots, site after site.
    snap: [2]gpu.DeviceBuffer,
    fork: gpu.Event,

    fn init(e: *const Env) !Bufs {
        var b: Bufs = undefined;
        for (&b.h) |*x| x.* = try e.alloc(max_rows * 4 * hc_d * 2);
        for (&b.g) |*x| x.* = try e.alloc(world * max_rows * hc_d * 4);
        for (&b.fnw) |*x| x.* = try e.alloc(cm.hc_mixes * 4 * hc_d * 4);
        for (&b.scale) |*x| x.* = try e.alloc(3 * 4);
        for (&b.base) |*x| x.* = try e.alloc(cm.hc_mixes * 4);
        for (&b.normw) |*x| x.* = try e.alloc(hc_d * 2);
        for (&b.pre) |*x| x.* = try e.alloc(max_rows * 4 * 4);
        b.x = try e.alloc(max_rows * hc_d * 2);
        b.post = try e.alloc(max_rows * 4 * 4);
        b.comb = try e.alloc(max_rows * 16 * 4);
        b.part = try e.alloc(max_rows * hc_nb * 32 * 4);
        for (&b.snap) |*x| x.* = try e.alloc(3 * snapBytes(max_rows));
        b.fork = try gpu.Event.init(e.d, false);
        return b;
    }

    fn deinit(b: *Bufs) void {
        inline for (.{ &b.h, &b.g, &b.fnw, &b.scale, &b.base, &b.normw, &b.pre, &b.snap }) |arr| for (arr) |*x| x.free();
        b.x.free();
        b.post.free();
        b.comb.free();
        b.part.free();
        b.fork.deinit();
    }
};

/// The synthetic state at max_rows: streams [16, 4, D] bf16, 3 sites' partials [world][16][D], pre, post, comb.
const Host = struct {
    h: []u16,
    g: [3][]f32,
    pre: [max_rows * 4]f32,
    post: [max_rows * 4]f32,
    comb: [max_rows * 16]f32,

    fn make(a: std.mem.Allocator, rng: *Rng) !Host {
        var hh: Host = undefined;
        hh.h = try a.alloc(u16, max_rows * 4 * hc_d);
        for (hh.h) |*v| v.* = cm.bf16(@floatCast(rng.normal()));
        for (&hh.g) |*gg| {
            gg.* = try a.alloc(f32, world * max_rows * hc_d);
            for (gg.*) |*v| v.* = @floatCast(rng.normal() * 0.5);
        }
        for (&hh.pre) |*v| v.* = @floatCast(rng.uniform(0.05, 1.0));
        for (&hh.post) |*v| v.* = @floatCast(rng.uniform(0.0, 2.0));
        // comb: positive, roughly doubly stochastic, a 4 x 4 a row
        for (&hh.comb) |*v| v.* = @floatCast(0.25 + rng.uniform(-0.1, 0.1));
        return hh;
    }

    fn free(hh: Host, a: std.mem.Allocator) void {
        a.free(hh.h);
        for (hh.g) |gg| a.free(gg);
    }
};

/// Three sites' own weights: fn [24, 4D] fp32, scale [3], base [24], the norm's weight [D] bf16.
fn weights(e: *const Env, b: *Bufs, rng: *Rng) !void {
    const w = try e.gpa.alloc(f32, cm.hc_mixes * 4 * hc_d);
    defer e.gpa.free(w);
    const nw = try e.gpa.alloc(u16, hc_d);
    defer e.gpa.free(nw);
    for (0..3) |k| {
        for (w) |*v| v.* = @floatCast(rng.normal() * 0.02);
        try e.put(b.fnw[k], 0, w);
        var sc: [3]f32 = undefined;
        for (&sc) |*v| v.* = @floatCast(rng.uniform(0.5, 1.5));
        try e.put(b.scale[k], 0, &sc);
        var base: [cm.hc_mixes]f32 = undefined;
        for (&base) |*v| v.* = @floatCast(rng.normal() * 0.5);
        try e.put(b.base[k], 0, &base);
        for (nw) |*v| v.* = cm.bf16(@floatCast(1.0 + rng.normal() * 0.1));
        try e.put(b.normw[k], 0, nw);
    }
}

/// The state a chain starts from for `rows` rows (the host state's leading rows); outputs filled with 0x5A first.
fn reset(e: *const Env, b: *Bufs, hh: *const Host, rows: usize) !void {
    try e.put(b.h[0], 0, hh.h[0 .. rows * 4 * hc_d]);
    try e.put(b.pre[0], 0, hh.pre[0 .. rows * 4]);
    try e.put(b.post, 0, hh.post[0 .. rows * 4]);
    try e.put(b.comb, 0, hh.comb[0 .. rows * 16]);
    try e.fill(b.h[1].ptr, 0x5A, max_rows * 4 * hc_d * 2);
    try e.fill(b.pre[1].ptr, 0x5A, max_rows * 4 * 4);
    try e.fill(b.x.ptr, 0x5A, max_rows * hc_d * 2);
    try e.fill(b.part.ptr, 0x5A, max_rows * hc_nb * 32 * 4);
}

/// The gathered partials laid out [world, rows, D], as the gather leaves them for `rows` rows.
fn gathered(e: *const Env, b: *Bufs, hh: *const Host, rows: usize) !void {
    for (0..3) |k| for (0..world) |q| {
        try e.put(b.g[k], q * rows * hc_d * 4, hh.g[k][q * max_rows * hc_d ..][0 .. rows * hc_d]);
    };
}

/// One site as round.zig's mix: the round's sink (this stream: production's HC_SIDE=0), none with tri_hc.inline_mix.
fn site(e: *const Env, b: *Bufs, hi: *usize, k: usize, posted: bool, rows: usize) !u64 {
    const sk: tri_hc.Sink = .{ .s = e.s, .fork = b.fork, .cnt = 0 };
    const sink: ?tri_hc.Sink = if (tri_hc.inline_mix) null else sk;
    const t = e.t(true);
    const pre_in = b.pre[k % 2].ptr;
    const pre_out = b.pre[(k + 1) % 2].ptr;
    const w = k % 3;
    const h = b.h[hi.*].ptr;
    const eps = cm.hc_eps_rms;
    // posted: the pending partials posted into the spare streams, which become h
    const ps: ?tri_hc.Posted = if (posted) .{ .gathered = b.g[w].ptr, .world = world, .h_out = b.h[1 - hi.*].ptr } else null;
    const src = try tri_hc.hcPre2(t, h, b.fnw[w].ptr, b.scale[w].ptr, b.base[w].ptr, pre_in, b.normw[w].ptr, eps, cm.hc_eps, cm.hc_iters, b.x.ptr, pre_out, b.post.ptr, b.comb.ptr, b.part.ptr, ps, sink, rows, hc_d);
    if (posted) hi.* = 1 - hi.*;
    return src;
}

/// Site `k`'s outputs copied (in stream order) into the mode's snapshot buffer.
fn snapshot(e: *const Env, b: *Bufs, mode: usize, k: usize, src: u64, pre_out: u64, rows: usize) !void {
    var at = b.snap[mode].ptr + k * snapBytes(rows);
    const srcs = [_]u64{ src, b.x.ptr, b.part.ptr, pre_out, b.post.ptr, b.comb.ptr };
    for (std.enums.values(Field), srcs) |f, p| {
        const n = fieldBytes(f, rows);
        try gpu.copyD2DAsync(e.d, e.s.handle, at, p, n);
        at += n;
    }
}

/// One chain in one mode; flags launch_bad (first case in msg) unless it made exactly the launches the mode must.
fn chainRun(e: *const Env, b: *Bufs, hh: *const Host, chain: []const u8, mode: usize, rows: usize, msg: *Msg, launch_bad: *bool) !void {
    tri_hc.inline_mix = mode == 1;
    defer tri_hc.inline_mix = false;
    try reset(e, b, hh, rows);
    const C = cm.Counts(&names);
    const c0 = C.take(e.kit);
    var hi: usize = 0;
    var unposted: u64 = 0;
    for (chain, 0..) |kind, k| {
        const src = try site(e, b, &hi, k, kind == 'P', rows);
        unposted += @intFromBool(kind == 'U');
        try snapshot(e, b, mode, k, src, b.pre[(k + 1) % 2].ptr, rows);
    }
    const ran = C.take(e.kit).since(c0);
    const n = chain.len;
    // off: posted post_only, finish_s, mix_part, sinkhorn; unposted mix_part, sinkhorn, finish_s; on: the 2 inline
    const want: [names.len]u64 = if (mode == 0) .{ n - unposted, n, n, n, 0 } else .{ 0, n, n, 0, 0 };
    if (!std.mem.eql(u64, &ran, &want)) {
        if (!launch_bad.*) msg.put("inline_mix {s}, rows {d}, chain {s}: launches post_only {d} mix_part {d} finish_s {d} sinkhorn {d} fused {d}; ", .{ if (mode == 1) "on" else "off", rows, chain, ran[0], ran[1], ran[2], ran[3], ran[4] });
        launch_bad.* = true;
    }
}

/// Off's and on's snapshots compared field by field; returns the differing bytes (the first one in msg).
fn compare(host: [2][]u8, chain: []const u8, rows: usize, msg: *Msg, first: bool) usize {
    var bad: usize = 0;
    var at: usize = 0;
    for (0..chain.len) |k| for (std.enums.values(Field)) |f| {
        const fb = fieldBytes(f, rows);
        const x = host[0][at..][0..fb];
        const y = host[1][at..][0..fb];
        if (std.mem.indexOfDiff(u8, x, y)) |i| {
            if (first and bad == 0) msg.put("first difference: rows {d}, chain {s}, site {d} ({c}), {s} byte {d}: 0x{x:0>2} off vs 0x{x:0>2} on; ", .{ rows, chain, k, chain[k], @tagName(f), i, x[i], y[i] });
            for (x, y) |p, q| bad += @intFromBool(p != q);
        }
        at += fb;
    };
    return bad;
}

pub fn step(e: *const Env, r: *Report) !void {
    if (tri_hc.fuse) return error.HcFuseSet; // (TF_HC_FUSE: not the path this check is about)
    var b = try Bufs.init(e);
    defer b.deinit();
    var rng: Rng = .{ .s = e.o.seed ^ 0x6863 };
    try weights(e, &b, &rng);
    const hh = try Host.make(e.gpa, &rng);
    defer hh.free(e.gpa);
    const sb = 3 * snapBytes(max_rows);
    const host = [2][]u8{ try e.gpa.alloc(u8, sb), try e.gpa.alloc(u8, sb) };
    defer for (host) |x| e.gpa.free(x);
    var msg: Msg = .{};
    var bad: usize = 0;
    var launch_bad = false;
    for (1..max_rows + 1) |rows| {
        try gathered(e, &b, &hh, rows);
        for (chains) |chain| {
            for (0..2) |mode| try chainRun(e, &b, &hh, chain, mode, rows, &msg, &launch_bad);
            const n = chain.len * snapBytes(rows);
            try e.get(b.snap[0], 0, host[0][0..n]);
            try e.get(b.snap[1], 0, host[1][0..n]);
            bad += compare(host, chain, rows, &msg, bad == 0);
        }
    }
    msg.put("rows 1-16, chains UPP and PUP (post / comb feeding the next post), inline_mix off vs on: {d} bytes of h_out, x, PART, pre_out, post, comb differ", .{bad});
    r.check("hc.bits", bad == 0 and !launch_bad, msg.text());

    // timing: a round's 80 sites as one graph
    for (cm.timed_rows) |rows| {
        var ms: [2]f64 = undefined;
        for (0..2) |mode| {
            tri_hc.inline_mix = mode == 1;
            defer tri_hc.inline_mix = false;
            try reset(e, &b, &hh, rows);
            ms[mode] = try cm.timeGraph(e, e.o.replays, Graph{ .e = e, .b = &b, .rows = rows });
        }
        r.timing("\"timing\": \"hc\", \"rows\": {d}, \"sites\": {d}, \"replays\": {d}, \"off_ms_round\": {d:.4}, \"inline_ms_round\": {d:.4}", .{ rows, round_sites, e.o.replays, ms[0], ms[1] }, rows, ms[0] - ms[1]);
    }
}

/// A round's 80 sites (the streams' values carry over from replay to replay: the timing does not depend on them).
const Graph = struct {
    e: *const Env,
    b: *Bufs,
    rows: usize,

    pub fn record(g: Graph) !void {
        var hi: usize = 0;
        for (0..round_sites) |k| {
            const posted = std.mem.indexOfScalar(usize, &round_unposted, k) == null;
            _ = try site(g.e, g.b, &hi, k, posted, g.rows);
        }
    }
};

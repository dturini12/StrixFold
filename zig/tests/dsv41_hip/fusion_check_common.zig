//! tf-dsv41-fusion-check's shared parts: the shapes, options, report, seeded inputs, environment and graph timing.

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const tri = dsv41.tri;
const tri_aux = dsv41.tri_aux;

// the main model's router (fixtures/launches.json `_route_parts` NE 384) and mHC (D 5120, NB 40, eps as recorded)
pub const ne = 384;
pub const kc = 20;
pub const topk = 6;
pub const slots = 7;
pub const route_scale: f32 = 1.5;
pub const shared_id = 384;
pub const hc_d = 5120;
pub const hc_nb = dsv41.tri_basic.hc_blocks;
pub const hc_mixes = 24;
pub const hc_eps_rms: f32 = 1e-20;
pub const hc_eps: f32 = 1e-6;
pub const hc_iters = 20;
pub const max_rows = 16;
pub const world = 2;
/// The row counts the graphs are timed at.
pub const timed_rows = [_]usize{ 1, 3, 6, 16 };
/// The gain each change must show at one row before pair time is spent (PLAN 6.2), ms a round.
pub const min_gain_ms = 0.2;
/// The score step's full sweep: 256 batches of 2^24 inputs, every fp32 pattern.
pub const score_batches_all = 256;

pub const Steps = struct { score: bool = false, route: bool = false, hc: bool = false };

pub const Opts = struct {
    kit: []const u8,
    steps: Steps,
    oracle: ?[]const u8 = null,
    oracle_sha: ?[]const u8 = null,
    aux_object: ?[]const u8 = null,
    score_batches: usize = score_batches_all,
    replays: usize = 2000,
    seed: u64 = 1,
    receipt: ?[]const u8 = null,
    dry_run: bool = false,
};

/// Every check and timing as one JSON line, on stderr and in the receipt.
pub const Report = struct {
    gpa: std.mem.Allocator,
    json: std.ArrayList(u8) = .empty,
    failed: u32 = 0,
    passed: u32 = 0,
    timing_short: u32 = 0,

    pub fn add(r: *Report, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.allocPrint(r.gpa, fmt, args) catch return;
        if (r.json.items.len > 0) r.json.appendSlice(r.gpa, ",\n") catch return;
        r.json.appendSlice(r.gpa, "    ") catch return;
        r.json.appendSlice(r.gpa, s) catch return;
        std.debug.print("{s}\n", .{s});
    }

    /// One check's line: {"check": name, "ok": ok, "detail": detail}.
    pub fn check(r: *Report, name: []const u8, ok: bool, detail: []const u8) void {
        if (ok) r.passed += 1 else r.failed += 1;
        var esc: std.ArrayList(u8) = .empty;
        for (detail) |c| {
            if (c == '"' or c == '\\') esc.append(r.gpa, '\\') catch return;
            esc.append(r.gpa, if (c < 0x20) ' ' else c) catch return;
        }
        r.add("{{\"check\": \"{s}\", \"ok\": {}, \"detail\": \"{s}\"}}", .{ name, ok, esc.items });
    }

    /// A timing line; `gain` is ms a round, judged against min_gain_ms at one row.
    pub fn timing(r: *Report, comptime fmt: []const u8, args: anytype, rows: usize, gain: f64) void {
        const short = rows == 1 and gain < min_gain_ms;
        if (short) r.timing_short += 1;
        const verdict: []const u8 = if (rows != 1) "" else if (short) ", \"verdict\": \"SHORT of 0.2 ms\"" else ", \"verdict\": \"OK\"";
        const s = std.fmt.allocPrint(r.gpa, fmt, args) catch return;
        r.add("{{{s}, \"gain_ms_round\": {d:.4}{s}}}", .{ s, gain, verdict });
    }
};

/// A fixed-size message for a check's detail.
pub const Msg = struct {
    buf: [1024]u8 = undefined,
    len: usize = 0,

    pub fn put(m: *Msg, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(m.buf[m.len..], fmt, args) catch return;
        m.len += s.len;
    }

    pub fn text(m: *const Msg) []const u8 {
        return m.buf[0..m.len];
    }
};

pub fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// splitmix64, the seeded inputs' source.
pub const Rng = struct {
    s: u64,

    pub fn next(r: *Rng) u64 {
        r.s +%= 0x9E3779B97F4A7C15;
        var x = r.s;
        x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
        x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
        return x ^ (x >> 31);
    }

    /// Uniform in [0, 1).
    pub fn unit(r: *Rng) f64 {
        return @as(f64, @floatFromInt(r.next() >> 11)) * (1.0 / @as(f64, 1 << 53));
    }

    pub fn uniform(r: *Rng, lo: f64, hi: f64) f64 {
        return lo + (hi - lo) * r.unit();
    }

    /// Standard normal (Box-Muller).
    pub fn normal(r: *Rng) f64 {
        const u1_ = @max(r.unit(), 1e-300);
        const u2_ = r.unit();
        return @sqrt(-2.0 * @log(u1_)) * @cos(2.0 * std.math.pi * u2_);
    }

    pub fn below(r: *Rng, n: usize) usize {
        return @intCast(r.next() % n);
    }
};

/// fp32 to bf16 bits, round to nearest even (finite inputs).
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn isNan32(u: u32) bool {
    return u & 0x7f800000 == 0x7f800000 and u & 0x007fffff != 0;
}

/// One GPU, one stream, the candidate's kit with its Triton set and both TF_DS_AUX selections bound.
pub const Env = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const gpu.Driver,
    s: gpu.Stream,
    kit: *gpu.Kit,
    set: *const gpu.aot.Set,
    aux_off: tri_aux.Aux,
    aux_on: tri_aux.Aux,
    o: *const Opts,

    /// The family's launcher as the served build runs it (PDL on), TF_DS_AUX=hip (off) or hip,route (on).
    pub fn t(e: *const Env, on: bool) tri.Tri {
        return .{ .set = e.set, .stream = e.s, .pdl = true, .aux = if (on) &e.aux_on else &e.aux_off };
    }

    pub fn alloc(e: *const Env, bytes: usize) !gpu.DeviceBuffer {
        return gpu.DeviceBuffer.alloc(e.d, bytes);
    }

    /// `bytes` of `value` from `dst`, in stream order.
    pub fn fill(e: *const Env, dst: u64, value: u8, bytes: usize) !void {
        try gpu.fill8Async(e.d, e.s.handle, dst, value, bytes);
    }

    /// `host` to `dst`, the stream idle first (a synchronous copy; the stream is non-blocking).
    pub fn put(e: *const Env, dst: gpu.DeviceBuffer, offset: usize, host: anytype) !void {
        try e.s.synchronize();
        try dst.upload(offset, std.mem.sliceAsBytes(host));
    }

    pub fn get(e: *const Env, src: gpu.DeviceBuffer, offset: usize, out: anytype) !void {
        try e.s.synchronize();
        try src.download(offset, std.mem.sliceAsBytes(out));
    }
};

/// Launches so far (strict-mode counts) of kit kernel `name`: a native id, or every variant of a Triton function.
pub fn launchesOf(kit: *const gpu.Kit, name: []const u8) u64 {
    var n: u64 = 0;
    for (kit.entries, 0..) |entry, i| {
        const is = std.mem.eql(u8, entry.k.id, name) or (if (entry.k.triton) |tt| std.mem.eql(u8, tt.@"fn", name) else false);
        if (is) n += kit.launches(@intCast(i));
    }
    return n;
}

/// Launch counts of a list of kernels, to diff around a run.
pub fn Counts(comptime names: []const []const u8) type {
    return struct {
        v: [names.len]u64,

        pub fn take(kit: *const gpu.Kit) @This() {
            var c: @This() = undefined;
            inline for (names, 0..) |nm, i| c.v[i] = launchesOf(kit, nm);
            return c;
        }

        pub fn since(now: @This(), before: @This()) [names.len]u64 {
            var d: [names.len]u64 = undefined;
            for (&d, now.v, before.v) |*x, a, b| x.* = a - b;
            return d;
        }
    };
}

/// aux.route_pick and aux.route_score in one object, and that object the one T2's note names (when given).
pub fn auxObjectOk(m: gpu.manifest.Manifest, msg: *Msg, want: ?[]const u8) bool {
    var pick: ?[]const u8 = null;
    var score: ?[]const u8 = null;
    for (m.kernels) |k| {
        if (std.mem.eql(u8, k.id, "aux.route_pick")) pick = k.object;
        if (std.mem.eql(u8, k.id, "aux.route_score")) score = k.object;
    }
    if (pick == null or score == null) {
        msg.put("the kit lacks {s}{s}", .{ if (pick == null) "aux.route_pick " else "", if (score == null) "aux.route_score" else "" });
        return false;
    }
    if (!std.mem.eql(u8, pick.?, score.?)) {
        msg.put("aux.route_pick ({s}) and aux.route_score ({s}) are in different objects", .{ pick.?, score.? });
        return false;
    }
    const sha = for (m.objects) |ob| {
        if (std.mem.eql(u8, ob.file, pick.?)) break ob.sha256;
    } else {
        msg.put("object {s} not in the manifest", .{pick.?});
        return false;
    };
    msg.put("aux_route object {s} sha256 {s}", .{ pick.?, sha });
    if (want) |w| if (!std.mem.eql(u8, w, sha)) {
        msg.put("; T2-score-asm.md names {s}: not this object", .{w});
        return false;
    };
    return true;
}

/// `work.record()` captured as a graph, uploaded, 20 warm replays, then `replays` back to back: mean ms a replay.
pub fn timeGraph(e: *const Env, replays: usize, work: anytype) !f64 {
    try e.s.synchronize();
    try gpu.graph.beginCapture(e.s, .thread_local);
    work.record() catch |err| {
        if (gpu.graph.endCapture(e.s)) |got| {
            var x = got;
            x.deinit();
        } else |_| {}
        return err;
    };
    var graph = try gpu.graph.endCapture(e.s);
    defer graph.deinit();
    var exec = try graph.instantiate();
    defer exec.deinit();
    try exec.upload(e.s);
    for (0..20) |_| try exec.launchOn(e.s);
    var start = try gpu.Event.init(e.d, true);
    defer start.deinit();
    var end = try gpu.Event.init(e.d, true);
    defer end.deinit();
    try start.record(e.s);
    for (0..replays) |_| try exec.launchOn(e.s);
    try end.record(e.s);
    try end.synchronize();
    const ms = try gpu.Event.elapsedMs(start, end);
    return @as(f64, ms) / @as(f64, @floatFromInt(replays));
}

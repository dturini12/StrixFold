//! tf-dsv41-hip-glue-test --kit DIR [--receipt PATH]: w-arch's glue kernels and the hipBLAS binding on the GPU (plan
//! step 6), launched through the family's own code (ops.zig's launchers bound by hip_load.zig, hipblas.zig) and compared
//! with host references that share nothing with the kernels:
//! - the casts bit for bit over every one of the 65,536 bf16 and fp16 inputs (fp16 through Zig's own conversion);
//! - add2_bf16 bit for bit against the host's IEEE fp32 add and round-to-nearest-even (ties, denormals, overflow, ±0);
//!   a NaN must stay a NaN (its payload rule is reported, ADAPTER.md 14);
//! - the row movers byte for byte, including every byte they must leave alone and the gather's invalid flag;
//! - argmax_rows against sampling.argmax, the host rule it implements (ties, NaN, ±inf, ±0, row lengths);
//! - aux.markov_race (the drafter's Markov step of a sampled pass) against tri_markov.raceRow, the host statement of
//!   its rule: greedy rows exactly, sampled rows' tokens wherever the two best scores are not within the fp32 draw's
//!   error, cached and staged bias rows, both vocabulary halves; and its time a launch;
//! - the profile clock: count, cap, reset, order, and its nanoseconds bracketed by the host's clock;
//! - hipBLAS's fp32 GEMMs in the family's four call shapes against an FP64 reference: every element within
//!   γ_k·Σ|x·w| (u = 2^-24), and a second run bit-identical.
//! With --cross LABEL=DIR (repeatable): another implementation's code objects (CUDA kernel names), each glue operation
//! run through the same launcher on the same inputs and compared byte for byte with glue.* (the cross-check of w-tri's
//! gfx1151-native ports and the component kit's). Run under the GPU lock (tools/dsv41-hip/run_gpu_tests.sh).

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const ops_mod = dsv41.ops;
const hip_load = dsv41.hip_load;
const sampling = dsv41.sampling;
const Blas = dsv41.blas.Blas;

const Fail = error{CheckFailed};

const Receipt = struct {
    gpa: std.mem.Allocator,
    checks: std.ArrayList(u8) = .empty,
    passed: u32 = 0,
    failed: u32 = 0,
    detail: [1536]u8 = undefined,
    detail_len: usize = 0,

    fn note(r: *Receipt, comptime fmt: []const u8, args: anytype) void {
        const s = std.fmt.bufPrint(r.detail[r.detail_len..], fmt, args) catch return;
        r.detail_len += s.len;
    }

    fn expect(r: *Receipt, ok: bool, comptime fmt: []const u8, args: anytype) Fail!void {
        if (ok) return;
        r.note(" FAILED: " ++ fmt, args);
        return error.CheckFailed;
    }

    fn run(r: *Receipt, name: []const u8, comptime f: anytype, ctx: anytype) !void {
        r.detail_len = 0;
        const t0 = nowNs();
        const ok = if (f(r, ctx)) |_| true else |e| blk: {
            if (e != error.CheckFailed) r.note(" error {s}", .{@errorName(e)});
            break :blk false;
        };
        r.note(" ({d} ms)", .{(nowNs() - t0) / std.time.ns_per_ms});
        if (ok) r.passed += 1 else r.failed += 1;
        std.debug.print("{s} {s}: {s}\n", .{ if (ok) "PASS" else "FAIL", name, r.detail[0..r.detail_len] });
        const sep: []const u8 = if (r.checks.items.len == 0) "" else ",\n";
        try r.checks.appendSlice(r.gpa, try std.fmt.allocPrint(r.gpa, "{s}    {{\"name\": \"{s}\", \"ok\": {}, \"detail\": \"", .{ sep, name, ok }));
        for (r.detail[0..r.detail_len]) |c| {
            if (c == '"' or c == '\\') try r.checks.append(r.gpa, '\\');
            try r.checks.append(r.gpa, if (c < 0x20) ' ' else c);
        }
        try r.checks.appendSlice(r.gpa, "\"}");
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

/// splitmix64
fn mix(x0: u64) u64 {
    var x = x0 +% 0x9E3779B97F4A7C15;
    x = (x ^ (x >> 30)) *% 0xBF58476D1CE4E5B9;
    x = (x ^ (x >> 27)) *% 0x94D049BB133111EB;
    return x ^ (x >> 31);
}

const Env = struct {
    gpa: std.mem.Allocator,
    d: *const gpu.Driver,
    s: gpu.Stream,
    kit: *gpu.Kit,
    o: ops_mod.Ops,
    wall_clock_hz: u64,
};

/// A device copy of `items` (any slice).
fn upload(e: *const Env, items: anytype) !gpu.DeviceBuffer {
    return gpu.DeviceBuffer.fromHost(e.d, std.mem.sliceAsBytes(items));
}

/// The stream's work done, then `out` from `buf` at `offset` bytes.
fn download(e: *const Env, buf: gpu.DeviceBuffer, offset: usize, out: anytype) !void {
    try e.s.synchronize();
    try buf.download(offset, std.mem.sliceAsBytes(out));
}

fn isNan32(u: u32) bool {
    return u & 0x7f800000 == 0x7f800000 and u & 0x007fffff != 0;
}

fn isNan16(u: u16) bool {
    return u & 0x7f80 == 0x7f80 and u & 0x007f != 0;
}

/// fp32 bits to bf16 by round to nearest even (a non-NaN input).
fn bf16Rne(u: u32) u16 {
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// ---- the casts

/// Every bf16 pattern, then more values than the 65,535-block grid holds (the grid-stride loop's second pass).
fn bf16ToF32(r: *Receipt, e: *const Env) !void {
    const count: usize = 65535 * 256 + 4451;
    const in = try e.gpa.alloc(u16, count);
    defer e.gpa.free(in);
    for (in, 0..) |*v, i| v.* = if (i < 65536) @intCast(i) else @truncate(mix(i));
    var din = try upload(e, in);
    defer din.free();
    var dout = try gpu.DeviceBuffer.alloc(e.d, count * 4);
    defer dout.free();
    try dout.fill8(0xA5, null);
    try e.o.toF32(e.s, din.ptr, dout.ptr, count);
    const out = try e.gpa.alloc(u32, count);
    defer e.gpa.free(out);
    try download(e, dout, 0, out);
    var bad: usize = 0;
    for (in, out, 0..) |b, got, i| if (got != @as(u32, b) << 16) {
        if (bad == 0) r.note(" first difference at {d}: bf16 {x:0>4} gave {x:0>8}", .{ i, b, got });
        bad += 1;
    };
    r.note("{d} values (all 65,536 patterns incl. NaN payloads, then past the 65,535-block grid): {d} differ from the high-half rule", .{ count, bad });
    try r.expect(bad == 0, "bf16 -> fp32 is not exact", .{});
}

/// Every fp16 pattern against Zig's own fp16 -> fp32 conversion; NaN in, NaN out (payload rule reported).
fn f16ToF32(r: *Receipt, e: *const Env) !void {
    const count: usize = 65536;
    const in = try e.gpa.alloc(u16, count);
    defer e.gpa.free(in);
    for (in, 0..) |*v, i| v.* = @intCast(i);
    var din = try upload(e, in);
    defer din.free();
    var dout = try gpu.DeviceBuffer.alloc(e.d, count * 4);
    defer dout.free();
    try e.o.f16ToF32(e.s, din.ptr, dout.ptr, count);
    const out = try e.gpa.alloc(u32, count);
    defer e.gpa.free(out);
    try download(e, dout, 0, out);
    var bad: usize = 0;
    var nans: usize = 0;
    var nan_same: usize = 0;
    var subnormals: usize = 0;
    for (in, out) |b, got| {
        const want: u32 = @bitCast(@as(f32, @as(f16, @bitCast(b))));
        if (b & 0x7c00 == 0x7c00 and b & 0x03ff != 0) {
            nans += 1;
            if (!isNan32(got)) {
                if (bad == 0) r.note(" fp16 NaN {x:0>4} gave {x:0>8}", .{ b, got });
                bad += 1;
            }
            if (got == want) nan_same += 1;
            continue;
        }
        if (b & 0x7c00 == 0 and b & 0x03ff != 0) subnormals += 1;
        if (got != want) {
            if (bad == 0) r.note(" fp16 {x:0>4} gave {x:0>8}, host {x:0>8}", .{ b, got, want });
            bad += 1;
        }
    }
    r.note("all 65,536 fp16 patterns: {d} non-NaN ({d} subnormal) exact, {d} NaN stay NaN ({d} with the host's bits); {d} wrong", .{ count - nans, subnormals, nans, nan_same, bad });
    try r.expect(bad == 0, "fp16 -> fp32 is not exact", .{});
}

/// add2_bf16's inputs: the designed edge cases, then random pairs (raw bit patterns, and partial-sized values).
fn add2Inputs(gpa: std.mem.Allocator, g0: *std.ArrayList(u32), g1: *std.ArrayList(u32)) !void {
    // round-to-nearest-even at every kind of low half (exact sums: x + 0), both parities, near the top and bottom
    const highs = [_]u16{ 0x0000, 0x0001, 0x0002, 0x0080, 0x0081, 0x3f80, 0x3f81, 0x7f7e, 0x7f7f, 0x8000, 0x8001, 0x8080, 0xbf80, 0xbf81, 0xff7e, 0xff7f };
    const lows = [_]u16{ 0x0000, 0x0001, 0x7fff, 0x8000, 0x8001, 0xffff };
    for (highs) |h| for (lows) |l| {
        try g0.append(gpa, @as(u32, h) << 16 | l);
        try g1.append(gpa, 0);
    };
    for (0..4096) |i| {
        const h: u32 = @as(u32, @truncate(mix(i))) & 0xffff0000;
        if (isNan32(h | 0x8000)) continue;
        try g0.append(gpa, h | 0x8000); // a tie
        try g1.append(gpa, 0);
    }
    // denormal sums (both operands subnormal, either sign), signed zeros, overflow to infinity, inf - inf, NaN inputs
    for (0..4096) |i| {
        const x = mix(@as(u64, 1) << 56 ^ i);
        try g0.append(gpa, @as(u32, @truncate(x)) & 0x807fffff);
        try g1.append(gpa, @as(u32, @truncate(x >> 32)) & 0x807fffff);
    }
    const specials = [_][2]u32{
        .{ 0x00000000, 0x80000000 }, .{ 0x80000000, 0x80000000 }, .{ 0x80000000, 0x00000000 }, .{ 0x3f800000, 0xbf800000 },
        .{ 0x7f7fffff, 0x7f7fffff }, .{ 0xff7fffff, 0xff7fffff }, .{ 0x7f7f8000, 0x00000000 }, .{ 0x7f7f7fff, 0x00000000 },
        .{ 0x7f800000, 0x3f800000 }, .{ 0xff800000, 0x3f800000 }, .{ 0x7f800000, 0xff800000 }, .{ 0x7fc12345, 0x3f800000 },
        .{ 0x3f800000, 0xffa00001 }, .{ 0x7f800001, 0x00000000 }, .{ 0xffffffff, 0x7f800000 }, .{ 0x00800000, 0x80000001 },
    };
    for (specials) |p| {
        try g0.append(gpa, p[0]);
        try g1.append(gpa, p[1]);
    }
    // random: raw bit patterns (every exponent pairing), then fp32 partials of similar size (the reduction's case)
    for (0..1 << 21) |i| {
        const x = mix(@as(u64, 3) << 56 ^ i);
        try g0.append(gpa, @truncate(x));
        try g1.append(gpa, @truncate(x >> 32));
    }
    for (0..1 << 21) |i| {
        const x = mix(@as(u64, 5) << 56 ^ i);
        const sa: u32 = @truncate(x >> 63 << 31);
        const sb: u32 = @truncate(x >> 62 << 31);
        const ea: u32 = 127 - 20 + @as(u32, @truncate((x >> 48) % 40));
        const eb: u32 = ea + 2 - @as(u32, @truncate((x >> 56) % 5));
        // (Zig's | and & share one precedence level: the mantissa mask needs its parentheses)
        try g0.append(gpa, sa | ea << 23 | (@as(u32, @truncate(x)) & 0x7fffff));
        try g1.append(gpa, sb | eb << 23 | (@as(u32, @truncate(x >> 23)) & 0x7fffff));
    }
}

/// out = bf16(g0 + g1): the designed edge cases, then random pairs (raw bit patterns, and partial-sized values).
fn add2Bf16(r: *Receipt, e: *const Env) !void {
    var g0: std.ArrayList(u32) = .empty;
    defer g0.deinit(e.gpa);
    var g1: std.ArrayList(u32) = .empty;
    defer g1.deinit(e.gpa);
    try add2Inputs(e.gpa, &g0, &g1);
    const count = g0.items.len;
    var d0 = try upload(e, g0.items);
    defer d0.free();
    var d1 = try upload(e, g1.items);
    defer d1.free();
    var dout = try gpu.DeviceBuffer.alloc(e.d, count * 2);
    defer dout.free();
    try e.o.add2Bf16(e.s, d0.ptr, d1.ptr, dout.ptr, count);
    const out = try e.gpa.alloc(u16, count);
    defer e.gpa.free(out);
    try download(e, dout, 0, out);
    var bad: usize = 0;
    var nans: usize = 0;
    var nan_quiet_rule: usize = 0;
    var ties: usize = 0;
    var subnormal_sums: usize = 0;
    var normal_pairs: usize = 0; // both operands normal and within a factor 2^5 of each other (partials)
    for (g0.items, g1.items, out) |a, b, got| {
        const sum: f32 = @as(f32, @bitCast(a)) + @as(f32, @bitCast(b));
        const u: u32 = @bitCast(sum);
        if (isNan32(u)) {
            nans += 1;
            if (!isNan16(got)) {
                if (bad == 0) r.note(" {x:0>8} + {x:0>8} is NaN, gave {x:0>4}", .{ a, b, got });
                bad += 1;
            }
            if (got == @as(u16, @truncate(u >> 16)) | 0x40) nan_quiet_rule += 1;
            continue;
        }
        if (u & 0xffff == 0x8000) ties += 1;
        const ea = (a >> 23) & 0xff;
        const eb = (b >> 23) & 0xff;
        if (ea != 0 and eb != 0 and ea != 0xff and eb != 0xff and @max(ea, eb) - @min(ea, eb) <= 5) normal_pairs += 1;
        if (u & 0x7f800000 == 0 and u & 0x7fffff != 0) subnormal_sums += 1;
        if (got != bf16Rne(u)) {
            if (bad == 0) r.note(" {x:0>8} + {x:0>8} = {x:0>8} gave {x:0>4}, want {x:0>4}", .{ a, b, u, got, bf16Rne(u) });
            bad += 1;
        }
    }
    r.note("{d} pairs ({d} exact ties, {d} subnormal sums, {d} normal pairs of similar size): {d} wrong; {d} NaN sums stay NaN, {d} of them as (sum bits >> 16) | 0x40 (quieted, payload kept; CUDA's cvt.rn.bf16.f32 gives a canonical NaN)", .{ count, ties, subnormal_sums, normal_pairs, bad, nans, nan_quiet_rule });
    try r.expect(bad == 0, "add2_bf16 is not the fp32 add rounded to nearest even", .{});
}

// ---- the row movers

/// `n` bytes of a pattern (distinct from anything a mover writes).
fn pattern(a: std.mem.Allocator, n: usize, seed: u64) ![]u8 {
    const out = try a.alloc(u8, n);
    for (out, 0..) |*b, i| b.* = @truncate(mix(seed ^ (i *% 0x100000001b3)) >> 17);
    return out;
}

/// The first `rows` of a random permutation of [0, n).
fn distinct(a: std.mem.Allocator, n: usize, rows: usize, seed: u64) ![]i64 {
    const all = try a.alloc(i64, n);
    defer a.free(all);
    for (all, 0..) |*v, i| v.* = @intCast(i);
    for (0..rows) |i| {
        const j = i + @as(usize, @intCast(mix(seed + i) % (n - i)));
        std.mem.swap(i64, &all[i], &all[j]);
    }
    return a.dupe(i64, all[0..rows]);
}

const ScatterCase = struct { rows: usize, bytes: usize, src_ld: usize, dst_ld: usize, dst_rows: usize, src_off: usize, dst_off: usize };

fn scatterRows(r: *Receipt, e: *const Env) !void {
    const cases = [_]ScatterCase{
        // the window ring's rows (512 bf16), aligned: 16-byte words
        .{ .rows = 64, .bytes = 1024, .src_ld = 1024, .dst_ld = 1024, .dst_rows = 256, .src_off = 0, .dst_off = 0 },
        .{ .rows = 33, .bytes = 1152, .src_ld = 2048, .dst_ld = 1280, .dst_rows = 97, .src_off = 16, .dst_off = 32 },
        // unaligned rows and bases: byte copies
        .{ .rows = 37, .bytes = 1000, .src_ld = 1003, .dst_ld = 1007, .dst_rows = 61, .src_off = 3, .dst_off = 5 },
    };
    for (cases, 0..) |c, ci| {
        const src = try pattern(e.gpa, c.src_off + c.rows * c.src_ld, 11 + ci);
        defer e.gpa.free(src);
        const dst0 = try pattern(e.gpa, c.dst_off + c.dst_rows * c.dst_ld + 64, 23 + ci);
        defer e.gpa.free(dst0);
        const idx = try distinct(e.gpa, c.dst_rows, c.rows, 31 + ci);
        defer e.gpa.free(idx);
        var dsrc = try upload(e, src);
        defer dsrc.free();
        var ddst = try upload(e, dst0);
        defer ddst.free();
        var didx = try upload(e, idx);
        defer didx.free();
        try e.o.scatterRows(e.s, dsrc.ptr + c.src_off, c.src_ld, didx.ptr, ddst.ptr + c.dst_off, c.dst_ld, c.bytes, c.rows);
        const want = try e.gpa.dupe(u8, dst0);
        defer e.gpa.free(want);
        for (idx, 0..) |t, i| {
            const at = c.dst_off + @as(usize, @intCast(t)) * c.dst_ld;
            @memcpy(want[at..][0..c.bytes], src[c.src_off + i * c.src_ld ..][0..c.bytes]);
        }
        const got = try e.gpa.alloc(u8, dst0.len);
        defer e.gpa.free(got);
        try download(e, ddst, 0, got);
        const bad = std.mem.indexOfDiff(u8, got, want);
        r.note("{s}{d} rows of {d} B (ld {d}/{d}, bases +{d}/+{d}): {s}", .{ if (ci == 0) "" else "; ", c.rows, c.bytes, c.src_ld, c.dst_ld, c.src_off, c.dst_off, if (bad == null) "exact" else "DIFFERENT" });
        try r.expect(bad == null, "first wrong byte {d}", .{bad orelse 0});
    }
}

/// dst row i = src row idx[i] over the Markov embedding's real size (129,280 rows of 512 B); out-of-range indices set
/// the flag and leave their rows alone.
fn gatherRows(r: *Receipt, e: *const Env) !void {
    const src_rows: usize = 129280;
    const bytes: usize = 512;
    const rows: usize = 64;
    const src = try pattern(e.gpa, src_rows * bytes, 41);
    defer e.gpa.free(src);
    var dsrc = try upload(e, src);
    defer dsrc.free();
    const dst0 = try pattern(e.gpa, rows * bytes, 43);
    defer e.gpa.free(dst0);
    var ddst = try upload(e, dst0);
    defer ddst.free();
    var dflag = try gpu.DeviceBuffer.alloc(e.d, 4);
    defer dflag.free();
    const idx = try e.gpa.alloc(i64, rows);
    defer e.gpa.free(idx);
    for (idx, 0..) |*v, i| v.* = @intCast(mix(i + 47) % src_rows);
    idx[0] = 0;
    idx[rows - 1] = @intCast(src_rows - 1);
    for (0..2) |pass| {
        // pass 1: two indices out of range (-1 and src_rows), the rest as before
        if (pass == 1) {
            idx[5] = -1;
            idx[9] = @intCast(src_rows);
            try ddst.upload(0, dst0);
        }
        var didx = try upload(e, idx);
        defer didx.free();
        try dflag.fill32(0, null);
        try e.o.gatherRows(e.s, dsrc.ptr, src_rows, didx.ptr, ddst.ptr, bytes, rows, dflag.ptr);
        const got = try e.gpa.alloc(u8, rows * bytes);
        defer e.gpa.free(got);
        try download(e, ddst, 0, got);
        var flag: [1]u32 = undefined;
        try download(e, dflag, 0, flag[0..]);
        const want = try e.gpa.dupe(u8, dst0);
        defer e.gpa.free(want);
        for (idx, 0..) |t, i| if (t >= 0 and t < @as(i64, @intCast(src_rows))) @memcpy(want[i * bytes ..][0..bytes], src[@as(usize, @intCast(t)) * bytes ..][0..bytes]);
        const bad = std.mem.indexOfDiff(u8, got, want);
        const want_flag: u32 = if (pass == 1) 1 else 0;
        r.note("{s}{d} rows of {d} B from {d}: {s}, invalid flag {d}", .{ if (pass == 0) "" else "; with indices -1 and 129280: ", rows, bytes, src_rows, if (bad == null) "exact" else "DIFFERENT", flag[0] });
        try r.expect(bad == null and flag[0] == want_flag, "first wrong byte {d}, flag {d} (want {d})", .{ bad orelse 0, flag[0], want_flag });
    }
}

/// ops.copyRows in the draft's confidence-head shapes (16-byte words) and an unaligned one (tf_strided_copy), then the
/// strided copy's general three-level form through its bound kernel.
fn copyRows(r: *Receipt, e: *const Env) !void {
    const rows: usize = 64;
    const d: usize = 5120;
    const rank: usize = 256;
    const w = d + rank;
    const total = rows * w * 4 + 256;
    const xcf = try pattern(e.gpa, rows * d * 4, 51);
    defer e.gpa.free(xcf);
    const ef = try pattern(e.gpa, rows * rank * 4, 53);
    defer e.gpa.free(ef);
    const conf0 = try pattern(e.gpa, total, 57);
    defer e.gpa.free(conf0);
    var dx = try upload(e, xcf);
    defer dx.free();
    var de = try upload(e, ef);
    defer de.free();
    var dc = try upload(e, conf0);
    defer dc.free();
    try e.o.copyRows(e.s, dx.ptr, d * 4, dc.ptr, w * 4, d * 4, rows);
    try e.o.copyRows(e.s, de.ptr, rank * 4, dc.ptr + d * 4, w * 4, rank * 4, rows);
    const want = try e.gpa.dupe(u8, conf0);
    defer e.gpa.free(want);
    for (0..rows) |i| {
        @memcpy(want[i * w * 4 ..][0 .. d * 4], xcf[i * d * 4 ..][0 .. d * 4]);
        @memcpy(want[i * w * 4 + d * 4 ..][0 .. rank * 4], ef[i * rank * 4 ..][0 .. rank * 4]);
    }
    const got = try e.gpa.alloc(u8, total);
    defer e.gpa.free(got);
    try download(e, dc, 0, got);
    var bad = std.mem.indexOfDiff(u8, got, want);
    r.note("confidence-head rows [{d}, {d}] + [{d}, {d}] into [{d}, {d}] fp32 (words): {s}", .{ rows, d, rows, rank, rows, w, if (bad == null) "exact" else "DIFFERENT" });
    try r.expect(bad == null, "first wrong byte {d}", .{bad orelse 0});

    // unaligned: tf_strided_copy's row form (ops.copyRows' fallback)
    try dc.upload(0, conf0);
    try e.o.copyRows(e.s, dx.ptr + 3, 1001, dc.ptr + 7, 1013, 999, 37);
    @memcpy(want, conf0);
    for (0..37) |i| @memcpy(want[7 + i * 1013 ..][0..999], xcf[3 + i * 1001 ..][0..999]);
    try download(e, dc, 0, got);
    bad = std.mem.indexOfDiff(u8, got, want);
    r.note("; 37 unaligned rows of 999 B (ld 1001/1013): {s}", .{if (bad == null) "exact" else "DIFFERENT"});
    try r.expect(bad == null, "first wrong byte {d}", .{bad orelse 0});

    // the general [outer, middle, inner] form
    const K = gpu.Kernel("glue.strided_copy", hip_load.params.StridedCopy);
    const k = try K.bind(e.kit, &.{});
    try dc.upload(0, conf0);
    const p: hip_load.params.StridedCopy = .{ .src = dx.ptr + 1, .dst = dc.ptr + 2, .outer = 9, .middle = 5, .inner = 7, .src_outer = 211, .src_middle = 13, .dst_outer = 101, .dst_middle = 17 };
    try k.launch(e.s, .{ .grid = .{ .x = 9 }, .block = .{ .x = 256 } }, p);
    @memcpy(want, conf0);
    for (0..9) |g| for (0..5) |m| @memcpy(want[2 + g * 101 + m * 17 ..][0..7], xcf[1 + g * 211 + m * 13 ..][0..7]);
    try download(e, dc, 0, got);
    bad = std.mem.indexOfDiff(u8, got, want);
    r.note("; [9, 5, 7 B] with strides 211/13 -> 101/17: {s}", .{if (bad == null) "exact" else "DIFFERENT"});
    try r.expect(bad == null, "first wrong byte {d}", .{bad orelse 0});
}

// ---- argmax

fn argmaxRows(r: *Receipt, e: *const Env) !void {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const lens = [_]usize{ 129280, 1, 31, 1023, 1025, 4097 };
    var cases: usize = 0;
    for (lens) |n| {
        const rows: usize = 12;
        const ld = n + 37;
        const x = try e.gpa.alloc(f32, rows * ld);
        defer e.gpa.free(x);
        for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(mix(i + n) % 2_000_001)) - 1_000_000)) / 64.0;
        const at = struct {
            fn f(row: []f32, i: usize) *f32 {
                return &row[@min(i, row.len - 1)];
            }
        }.f;
        for (0..rows) |ri| {
            const row = x[ri * ld ..][0..n];
            switch (ri) {
                0 => {}, // random
                1 => for ([_]usize{ n * 3 / 5, n / 7, n - 1 }) |i| {
                    at(row, i).* = 1e9; // a tie at three places: the first wins
                },
                2 => row[0] = nan, // a leading NaN: 0
                3 => for ([_]usize{ n / 3, n / 2, n - 1 }) |i| {
                    at(row, i).* = nan; // NaNs never win
                },
                4 => @memset(row, -inf),
                5 => @memset(row, nan),
                6 => {
                    at(row, n / 2).* = inf;
                    at(row, n / 4).* = inf;
                },
                7 => {
                    @memset(row, -1);
                    at(row, n / 3).* = -0.0;
                    at(row, n / 5).* = 0.0;
                },
                8 => row[0] = 1e9,
                9 => row[n - 1] = 1e9,
                10 => {
                    @memset(row, -inf);
                    at(row, n - 1).* = nan;
                },
                else => @memset(row, 3.5), // all equal
            }
        }
        var dx = try upload(e, x);
        defer dx.free();
        var dout = try gpu.DeviceBuffer.alloc(e.d, rows * 4);
        defer dout.free();
        try e.o.argmaxRows(e.s, dx.ptr, ld, n, rows, dout.ptr);
        var got: [12]u32 = undefined;
        try download(e, dout, 0, got[0..]);
        for (0..rows) |ri| {
            const want = sampling.argmax(x[ri * ld ..][0..n]);
            cases += 1;
            try r.expect(got[ri] == want, "n {d} row {d}: {d}, sampling.argmax {d}", .{ n, ri, got[ri], want });
        }
    }
    r.note("{d} rows (n {any}, row stride n + 37; ties, NaN first/inside/everywhere, all -inf, +inf twice, +0 before -0, ends): each equals sampling.argmax", .{ cases, lens });
}

// ---- the drafter's sampled Markov step

fn markovRace(r: *Receipt, e: *const Env) !void {
    const tri_markov = dsv41.tri_markov;
    const tri_aux = dsv41.tri_aux;
    const ax = try hip_load.aux(e.kit, tri_aux.Which.triton);
    const f = ax.markov_race orelse return r.expect(false, "the kit has no aux.markov_race", .{});
    const t: dsv41.tri.Tri = .{ .stream = e.s };
    const nc = 64640; // a TP2 rank's vocabulary columns
    const rows = 6;
    const block = 5;
    const ts = block + 1;
    const cached = 4;
    const lg = try e.gpa.alloc(f32, rows * block * nc);
    defer e.gpa.free(lg);
    const cache = try e.gpa.alloc(f16, cached * nc);
    defer e.gpa.free(cache);
    const stage = try e.gpa.alloc(f16, rows * nc);
    defer e.gpa.free(stage);
    const slot = try e.gpa.alloc(i32, 2 * nc);
    defer e.gpa.free(slot);
    // logits spread over 24 with a few near the top (a sampled row's draw is among them), bias rows within 2
    for (lg, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(mix(i) % 3_000_001)) - 1_500_000)) / 125_000.0;
    for (0..rows * block) |q| for (0..12) |j| {
        lg[q * nc + mix(q * 131 + j) % nc] = 13.0 + @as(f32, @floatFromInt(mix(q * 977 + j) % 2001)) / 1000.0;
    };
    // _score's order for a NaN (above everything, the first of two): a greedy row's and a sampled row's, at steps 1
    // and 3 (rows 2 and 3 of the six; row r's step i is logits row r * block + i)
    for ([_]usize{ 2 * block + 1, 3 * block + 3 }) |q| for ([_]usize{ 40_000, 123, 64_000 }) |c| {
        lg[q * nc + c] = std.math.nan(f32);
    };
    for (cache, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i64, @intCast(mix(i + 7) % 4001)) - 2000)) / 1000.0);
    for (stage, 0..) |*v, i| v.* = @floatCast(@as(f32, @floatFromInt(@as(i64, @intCast(mix(i + 11) % 4001)) - 2000)) / 1000.0);
    @memset(slot, -1);
    // each row's previous token at every step: rows 1 and 4 read cached bias rows, the others their staging rows
    var tok: [rows * ts]i64 = undefined;
    for (&tok, 0..) |*v, i| v.* = @intCast(mix(i + 3) % (2 * nc));
    for ([_]usize{ 1, 4 }, 0..) |row, j| for (0..ts) |i| {
        slot[@intCast(tok[row * ts + i])] = @intCast((j + i) % cached);
    };
    for ([_]usize{ 0, 2, 3, 5 }) |row| for (0..ts) |i| {
        if (slot[@intCast(tok[row * ts + i])] >= 0) tok[row * ts + i] = @intCast((@as(usize, @intCast(tok[row * ts + i])) + 1) % (2 * nc));
    };
    const keys = [rows]tri_aux.MarkovKey{
        .{ .seed = 0x1122334455667788, .position = 4097, .inv_t = 1.0 },
        .{ .seed = 7, .position = 1_048_000, .inv_t = 1.0 / 0.7 },
        .{}, // a greedy stream beside sampled ones
        .{ .seed = 0x7fff_ffff_ffff_ffff, .position = 0, .inv_t = 4.0 },
        .{ .seed = 99, .position = 333_333, .inv_t = 0.5 },
        .{ .seed = 12345, .position = 77, .inv_t = 0 },
    };
    var dlg = try upload(e, lg);
    defer dlg.free();
    var dcache = try upload(e, cache);
    defer dcache.free();
    var dstage = try upload(e, stage);
    defer dstage.free();
    var dslot = try upload(e, slot);
    defer dslot.free();
    var dtok = try upload(e, tok[0..]);
    defer dtok.free();
    var dkeys = try upload(e, keys[0..]);
    defer dkeys.free();
    var dsend = try gpu.DeviceBuffer.alloc(e.d, rows * 4 * 4);
    defer dsend.free();
    var cases: usize = 0;
    var near: usize = 0;
    var nans: usize = 0;
    for ([_]u32{ 0, nc }) |id0| for (0..block) |i| {
        const sentinel: [rows * 4]f32 = @splat(-7.5);
        try dsend.upload(0, std.mem.sliceAsBytes(sentinel[0..]));
        const a: tri_aux.MarkovRaceArgs = .{
            .lg = dlg.ptr + 4 * i * nc, .tok = dtok.ptr + 8 * i, .slot = dslot.ptr, .cache = dcache.ptr, .stage = dstage.ptr, .keys = dkeys.ptr, .send = dsend.ptr,
            .l_row = block * nc, .t_stride = ts, .c_row = nc, .s_row = nc, .n_rows = rows, .n_cols = nc, .id0 = @intCast(id0), .step = @intCast(i),
        };
        try tri_markov.race(t, f, a);
        var got: [rows * 4]f32 = undefined;
        try download(e, dsend, 0, got[0..]);
        for (0..rows) |row| {
            const sl = slot[@intCast(tok[row * ts + i])];
            const bias = if (sl >= 0) cache[@as(usize, @intCast(sl)) * nc ..][0..nc] else stage[row * nc ..][0..nc];
            const row_lg = lg[(row * block + i) * nc ..][0..nc];
            const key = keys[row];
            const want = tri_markov.raceRow(row_lg, bias, key, key.position + i, id0);
            // the runner-up on the host: a sampled row whose two best are within the device logs' error may go either way
            var second = -std.math.inf(f32);
            for (row_lg, bias, 0..) |l, b, c| {
                const id = id0 + @as(u32, @intCast(c));
                if (id == want[1]) continue;
                const score = l + @as(f32, b);
                const v = if (key.inv_t > 0) score * key.inv_t + tri_markov.raceGumbel(sampling.keyBits(key.seed, key.position + i, id)) else score;
                if (v > second) second = v;
            }
            cases += 1;
            const gv = got[row * 4];
            if (std.math.isNan(want[0])) {
                // a row with NaNs: the first of them, in either kind of row
                nans += 1;
                try r.expect(std.math.isNan(gv) and got[row * 4 + 1] == @as(f32, @floatFromInt(want[1])), "id0 {d} step {d} row {d}: ({d}, {d}), host NaN at {d}", .{ id0, i, row, gv, got[row * 4 + 1], want[1] });
                continue;
            }
            try r.expect(got[row * 4 + 1] >= 0 and got[row * 4 + 1] < 2 * nc, "id0 {d} step {d} row {d}: token {d} is no token", .{ id0, i, row, got[row * 4 + 1] });
            const gi: u32 = @intFromFloat(got[row * 4 + 1]);
            try r.expect(got[row * 4 + 2] == 0 and got[row * 4 + 3] == 0, "id0 {d} step {d} row {d}: send[2..] not 0", .{ id0, i, row });
            if (key.inv_t > 0) {
                try r.expect(@abs(gv - want[0]) <= 1e-4 * @max(1.0, @abs(want[0])), "id0 {d} step {d} row {d}: value {d}, host {d}", .{ id0, i, row, gv, want[0] });
                if (want[0] - second < 1e-4) {
                    near += 1;
                    continue;
                }
                try r.expect(gi == want[1], "id0 {d} step {d} row {d}: token {d}, host {d} (value {d} against {d}, runner-up {d})", .{ id0, i, row, gi, want[1], gv, want[0], second });
            } else {
                try r.expect(gi == want[1] and gv == want[0], "id0 {d} step {d} row {d} (greedy): ({d}, {d}), host ({d}, {d})", .{ id0, i, row, gv, gi, want[0], want[1] });
            }
        }
    };
    // its time a launch: 200 steps of four rows queued, then one wait
    const a4: tri_aux.MarkovRaceArgs = .{
        .lg = dlg.ptr, .tok = dtok.ptr, .slot = dslot.ptr, .cache = dcache.ptr, .stage = dstage.ptr, .keys = dkeys.ptr, .send = dsend.ptr,
        .l_row = block * nc, .t_stride = ts, .c_row = nc, .s_row = nc, .n_rows = 4, .n_cols = nc, .id0 = 0, .step = 0,
    };
    for (0..20) |_| try tri_markov.race(t, f, a4);
    try e.s.synchronize();
    const t0 = nowNs();
    for (0..200) |_| try tri_markov.race(t, f, a4);
    try e.s.synchronize();
    const us = @as(f64, @floatFromInt(nowNs() - t0)) / 200.0 / 1000.0;
    r.note("{d} rows of {d} columns (both vocabulary halves, steps 0-{d}, cached and staged bias rows, keys at inv_t 1, 1/0.7, 4, 0.5 and two greedy rows): greedy rows equal the host's (value, token); sampled rows' values within 1e-4 and their tokens the host's ({d} within 1e-4 of the runner-up, not compared); {d} rows with NaNs give the first NaN, _score's order; {d:.1} us a four-row launch", .{ cases, nc, block - 1, near, nans, us });
    try r.expect(nans == 4, "{d} rows with NaNs, 4 placed", .{nans});
}

// ---- the profile clock

fn clock(r: *Receipt, e: *const Env) !void {
    const cap: u32 = 4;
    var buf = try gpu.DeviceBuffer.alloc(e.d, (1 + 2 * cap) * 8);
    defer buf.free();
    try buf.fill8(0xEE, null);
    // long enough that the bracket (two launch-to-sync latencies, ~1-2 ms on an idle APU) is narrower than the 0.19%
    // between 99.81 MHz and a round 100 MHz (3.8 ms over 2 s)
    const wait_ns: u64 = 2000 * std.time.ns_per_ms;
    // a launch first, so the bracket below holds no first-launch cost
    try e.o.gpuClock(e.s, buf.ptr, 0, true, cap);
    try e.s.synchronize();
    const h0 = nowNs();
    try e.o.gpuClock(e.s, buf.ptr, 250, true, cap);
    try e.s.synchronize();
    const h1 = nowNs();
    while (nowNs() - h1 < wait_ns) {}
    const h2 = nowNs();
    try e.o.gpuClock(e.s, buf.ptr, 7, false, cap);
    try e.s.synchronize();
    const h3 = nowNs();
    for ([_]u32{ 8, 9, 10 }) |tag| try e.o.gpuClock(e.s, buf.ptr, tag, false, cap); // the third is past the cap
    var h: [1 + 2 * cap]u64 = undefined;
    try download(e, buf, 0, h[0..]);
    try r.expect(h[0] == cap, "count {d}, want the cap {d}", .{ h[0], cap });
    const tags = [_]u64{ 250, 7, 8, 9 };
    for (0..cap) |i| try r.expect(h[2 + 2 * i] == tags[i], "tag {d} is {d}", .{ i, h[2 + 2 * i] });
    for (1..cap) |i| try r.expect(h[1 + 2 * i] >= h[1 + 2 * (i - 1)], "clock {d} before clock {d}", .{ i, i - 1 });
    const gap = h[3] - h[1];
    const lo = h2 - h1;
    const hi = h3 - h0;
    // the scale comes from a rate in whole kHz (±5 ppm) and the two oscillators may differ by a few more: 25 ppm slack
    const slack = gap / 40_000;
    r.note("count {d} at cap {d}, tags in order, clocks non-decreasing; across a {d} ms host wait the device clock moved {d} ns; the host's bracket [{d}, {d}] ns ({d} us wide, +-{d} us for the kHz rate and oscillators; rate {d} Hz from the caps)", .{ h[0], cap, wait_ns / std.time.ns_per_ms, gap, lo, hi, (hi - lo) / std.time.ns_per_us, slack / std.time.ns_per_us, e.wall_clock_hz });
    try r.expect(gap + slack >= lo and gap <= hi + slack, "outside the bracket", .{});
    // the same ticks read as 10 ns each (a round 100 MHz): the bracket must be able to tell the two apart
    const ticks_ns: u64 = @intCast(@as(u128, gap) * e.wall_clock_hz / 100_000_000);
    const tells = ticks_ns + slack < lo or ticks_ns > hi + slack;
    r.note("; the same ticks at a round 100 MHz would read {d} ns: {s}", .{ ticks_ns, if (tells) "outside it" else "inside it (this wait cannot tell them apart)" });
    try r.expect(tells, "the wait is too short to check the rate", .{});
    try e.o.gpuClock(e.s, buf.ptr, 1, true, cap);
    try download(e, buf, 0, h[0..]);
    r.note("; reset: count {d}", .{h[0]});
    try r.expect(h[0] == 1 and h[2] == 1, "reset left count {d} tag {d}", .{ h[0], h[2] });
}

// ---- hipBLAS against FP64

const Gemm = struct { what: []const u8, rows: usize, k: usize, n: usize, xv: bool = false, w_f16: bool };

fn hipblas(r: *Receipt, e: *const Env) !void {
    var ws = try gpu.DeviceBuffer.alloc(e.d, Blas.workspace_bytes);
    defer ws.free();
    var b = try Blas.open(e.s, ws.ptr);
    defer b.close();
    const shapes = [_]Gemm{
        .{ .what = "indexer weights (prompt.zig:962)", .rows = 17, .k = 5120, .n = 32, .w_f16 = false },
        .{ .what = "indexer weights", .rows = 2048, .k = 5120, .n = 32, .w_f16 = false },
        .{ .what = "MoE gate logits (prompt.zig:1057)", .rows = 17, .k = 5120, .n = 384, .w_f16 = true },
        .{ .what = "MoE gate logits", .rows = 2048, .k = 5120, .n = 384, .w_f16 = true },
        .{ .what = "draft gate (draft.zig:632)", .rows = 17, .k = 5120, .n = 128, .w_f16 = true },
        .{ .what = "draft gate", .rows = 256, .k = 5120, .n = 128, .w_f16 = true },
        .{ .what = "confidence head xv (draft.zig:542)", .rows = 1, .k = 5376, .n = 1, .xv = true, .w_f16 = false },
        .{ .what = "confidence head xv", .rows = 64, .k = 5376, .n = 1, .xv = true, .w_f16 = false },
    };
    const u: f64 = std.math.pow(f64, 2, -24);
    for (shapes, 0..) |g, si| {
        // x: bf16 values widened (toF32's output); w: fp16 values (the gate weights) or plain fp32
        const x = try e.gpa.alloc(f32, g.rows * g.k);
        defer e.gpa.free(x);
        for (x, 0..) |*v, i| {
            const m = mix(@as(u64, si) << 48 ^ i);
            const f = (@as(f32, @floatFromInt(m >> 40)) / 16777216.0 - 0.5) * std.math.pow(f32, 2, @as(f32, @floatFromInt(@as(i32, @intCast((m >> 8) % 9)) - 4)));
            v.* = @bitCast(@as(u32, @bitCast(f)) & 0xffff0000);
        }
        const w = try e.gpa.alloc(f32, g.n * g.k);
        defer e.gpa.free(w);
        for (w, 0..) |*v, i| {
            const m = mix(@as(u64, si + 16) << 48 ^ i);
            const f = (@as(f32, @floatFromInt(m >> 40)) / 16777216.0 - 0.5) * 0.1;
            v.* = if (g.w_f16) @as(f16, @floatCast(f)) else f;
        }
        var dx = try upload(e, x);
        defer dx.free();
        var dw = try upload(e, w);
        defer dw.free();
        var d1 = try gpu.DeviceBuffer.alloc(e.d, g.rows * g.n * 4);
        defer d1.free();
        var d2 = try gpu.DeviceBuffer.alloc(e.d, g.rows * g.n * 4);
        defer d2.free();
        try d1.fill8(0xFF, null);
        try d2.fill8(0x7F, null);
        for ([_]u64{ d1.ptr, d2.ptr }) |out| {
            if (g.xv) try b.xv(dx.ptr, dw.ptr, out, g.rows, g.k) else try b.xwT(dx.ptr, dw.ptr, out, g.rows, g.k, g.n);
        }
        const got = try e.gpa.alloc(f32, g.rows * g.n);
        defer e.gpa.free(got);
        const again = try e.gpa.alloc(f32, g.rows * g.n);
        defer e.gpa.free(again);
        try download(e, d1, 0, got);
        try download(e, d2, 0, again);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(again));
        const gamma = @as(f64, @floatFromInt(g.k)) * u / (1 - @as(f64, @floatFromInt(g.k)) * u);
        var worst: f64 = 0; // |error| in units of u * sum |x w|
        var over: usize = 0;
        for (0..g.rows) |i| for (0..g.n) |j| {
            var acc: f64 = 0;
            var mag: f64 = 0;
            const xr = x[i * g.k ..][0..g.k];
            const wr = w[j * g.k ..][0..g.k];
            for (xr, wr) |a, c| {
                const p = @as(f64, a) * @as(f64, c);
                acc += p;
                mag += @abs(p);
            }
            const err = @abs(@as(f64, got[i * g.n + j]) - acc);
            // the reference's own rounding: k additions in fp64
            const bound = gamma * mag + @as(f64, @floatFromInt(g.k)) * std.math.pow(f64, 2, -53) * mag;
            if (!(err <= bound)) {
                if (over == 0) r.note(" {s} [{d},{d}]: {e} vs fp64 {e}, bound {e};", .{ g.what, i, j, got[i * g.n + j], acc, bound });
                over += 1;
            }
            if (mag > 0) worst = @max(worst, err / (u * mag));
        };
        r.note("{s}{s} rows {d} k {d} n {d}: worst {d:.2} u*sum|xw| (bound {d}), {d} over, rerun {s}", .{ if (si == 0) "" else "; ", g.what, g.rows, g.k, g.n, worst, g.k, over, if (same) "bit-identical" else "DIFFERENT" });
        try r.expect(over == 0 and same, "{s}", .{g.what});
    }
}

// ---- another implementation of the glue kernels, bit for bit (--cross LABEL=DIR)

/// Code objects in `dir` (raw gfx1151 ELF) holding the glue operations under their CUDA names: each operation runs
/// through the same launcher (ops.zig) on the same inputs as glue.* here, and the outputs are compared byte for byte.
/// A difference in a NaN's payload is reported; any other difference fails (one of the two is wrong: glue.* is held to
/// host references above).
const Cross = struct { e: *const Env, io: std.Io, label: []const u8, dir: []const u8 };

const CrossSet = struct {
    mods: std.ArrayList(gpu.Module) = .empty,
    metas: std.ArrayList(gpu.hsaco.CodeObject) = .empty,

    /// The function named `name` in whichever object defines it (from the objects' metadata: no failed lookup), or null.
    fn get(c: *const CrossSet, name: [:0]const u8) !?gpu.Function {
        for (c.metas.items, c.mods.items) |m, mod| if (m.find(name) != null) return try mod.function(name);
        return null;
    }
};

fn crossLoad(x: *const Cross, a: std.mem.Allocator) !CrossSet {
    var set: CrossSet = .{};
    var d = try std.Io.Dir.cwd().openDir(x.io, x.dir, .{ .iterate = true });
    defer d.close(x.io);
    var it = d.iterate();
    while (try it.next(x.io)) |ent| {
        if (!std.mem.endsWith(u8, ent.name, ".hsaco")) continue;
        const bytes = try d.readFileAllocOptions(x.io, ent.name, a, .limited(1 << 28), .@"16", null);
        try set.metas.append(a, try gpu.hsaco.parse(a, bytes));
        try set.mods.append(a, try gpu.Module.load(x.e.d, bytes));
    }
    return set;
}

/// How two outputs differ: bytes outside NaN payloads, and elements that are NaN in both but with other bits.
const Diff = struct { other: usize = 0, nan_payload: usize = 0, first: ?usize = null };

fn diffWords(comptime T: type, a: []const T, b: []const T, comptime isNan: fn (T) bool) Diff {
    var d: Diff = .{};
    for (a, b, 0..) |x, y, i| if (x != y) {
        if (isNan(x) and isNan(y)) {
            d.nan_payload += 1;
        } else {
            d.other += 1;
            if (d.first == null) d.first = i;
        }
    };
    return d;
}

fn never8(_: u8) bool {
    return false;
}

fn never32(_: u32) bool {
    return false;
}

fn noteDiff(r: *Receipt, label: []const u8, what: []const u8, n: usize, d: Diff) void {
    if (d.other == 0 and d.nan_payload == 0) return r.note(" {s} identical ({d});", .{ what, n });
    if (d.other == 0) return r.note(" {s} identical but for {d} NaN payloads ({d});", .{ what, d.nan_payload, n });
    r.note(" {s}: {d} of {d} differ from {s}, first at {d};", .{ what, d.other, n, label, d.first.? });
}

fn crossCheck(r: *Receipt, x: *const Cross) !void {
    const e = x.e;
    var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var set = try crossLoad(x, a);
    defer for (set.mods.items) |*m| m.unload();
    // these objects are not the kit's: strict mode's launch check would refuse them
    const saved = gpu.launch.before;
    gpu.launch.before = null;
    defer gpu.launch.before = saved;
    var other = e.o;
    var missing: usize = 0;
    var bad: usize = 0;
    r.note("{d} objects from {s}:", .{ set.mods.items.len, x.dir });

    // the casts: every bf16 and fp16 pattern
    const bits = try a.alloc(u16, 65536);
    for (bits, 0..) |*v, i| v.* = @intCast(i);
    var din = try upload(e, bits);
    defer din.free();
    var d1 = try gpu.DeviceBuffer.alloc(e.d, 65536 * 4);
    defer d1.free();
    var d2 = try gpu.DeviceBuffer.alloc(e.d, 65536 * 4);
    defer d2.free();
    const o1 = try a.alloc(u32, 65536);
    const o2 = try a.alloc(u32, 65536);
    for ([_]struct { [:0]const u8, []const u8, bool }{ .{ "tf_bf16_to_f32_kernel", "bf16_to_f32", false }, .{ "tf_ds_f16_to_f32_kernel", "f16_to_f32", true } }) |c| {
        const f = try set.get(c[0]) orelse {
            r.note(" {s} absent;", .{c[1]});
            missing += 1;
            continue;
        };
        if (c[2]) {
            other.f16_to_f32 = f;
            try e.o.f16ToF32(e.s, din.ptr, d1.ptr, 65536);
            try other.f16ToF32(e.s, din.ptr, d2.ptr, 65536);
        } else {
            other.to_f32 = f;
            try e.o.toF32(e.s, din.ptr, d1.ptr, 65536);
            try other.toF32(e.s, din.ptr, d2.ptr, 65536);
        }
        try download(e, d1, 0, o1);
        try download(e, d2, 0, o2);
        const d = diffWords(u32, o1, o2, isNan32);
        noteDiff(r, x.label, c[1], 65536, d);
        if (d.other > 0) bad += 1;
    }

    // the rank-ordered add: add2_bf16's 4.2M pairs
    if (try set.get("tf_ds_add2_bf16_kernel")) |f| {
        other.add2_bf16 = f;
        var g0: std.ArrayList(u32) = .empty;
        var g1: std.ArrayList(u32) = .empty;
        try add2Inputs(a, &g0, &g1);
        const n = g0.items.len;
        var dg0 = try upload(e, g0.items);
        defer dg0.free();
        var dg1 = try upload(e, g1.items);
        defer dg1.free();
        var y1 = try gpu.DeviceBuffer.alloc(e.d, n * 2);
        defer y1.free();
        var y2 = try gpu.DeviceBuffer.alloc(e.d, n * 2);
        defer y2.free();
        try e.o.add2Bf16(e.s, dg0.ptr, dg1.ptr, y1.ptr, n);
        try other.add2Bf16(e.s, dg0.ptr, dg1.ptr, y2.ptr, n);
        const h1 = try a.alloc(u16, n);
        const h2 = try a.alloc(u16, n);
        try download(e, y1, 0, h1);
        try download(e, y2, 0, h2);
        const d = diffWords(u16, h1, h2, isNan16);
        noteDiff(r, x.label, "add2_bf16", n, d);
        if (d.other > 0) bad += 1;
    } else {
        r.note(" add2_bf16 absent;", .{});
        missing += 1;
    }

    // the row movers: one aligned and one unaligned call each, every byte of the destination compared
    const src = try pattern(a, 1 << 20, 71);
    var dsrc = try upload(e, src);
    defer dsrc.free();
    const dst0 = try pattern(a, 1 << 20, 73);
    var y1 = try upload(e, dst0);
    defer y1.free();
    var y2 = try upload(e, dst0);
    defer y2.free();
    const b1 = try a.alloc(u8, 1 << 20);
    const b2 = try a.alloc(u8, 1 << 20);
    const idx = try distinct(a, 256, 64, 79);
    var didx = try upload(e, idx);
    defer didx.free();
    var flags = try gpu.DeviceBuffer.alloc(e.d, 8);
    defer flags.free();
    const movers = [_]struct { [:0]const u8, []const u8 }{
        .{ "tf_ds_scatter_rows_kernel", "scatter_rows" }, .{ "tf_gather_rows_kernel", "gather_rows" },
        .{ "tf_ds_copy_rows16_kernel", "copy_rows16" },   .{ "tf_strided_copy_kernel", "strided_copy" },
    };
    for (movers, 0..) |m, mi| {
        const f = try set.get(m[0]) orelse {
            r.note(" {s} absent;", .{m[1]});
            missing += 1;
            continue;
        };
        for ([_]bool{ true, false }) |aligned| {
            try y1.upload(0, dst0);
            try y2.upload(0, dst0);
            try flags.fill8(0, null);
            const so: u64 = if (aligned) 0 else 3;
            const do: u64 = if (aligned) 0 else 5;
            const bytes: usize = if (aligned) 1024 else 999;
            const ld: usize = if (aligned) 1024 else 1003;
            inline for (.{ false, true }) |second| {
                const o = if (second) &other else &e.o;
                const y = if (second) y2.ptr else y1.ptr;
                switch (mi) {
                    0 => {
                        other.scatter = f;
                        try o.scatterRows(e.s, dsrc.ptr + so, ld, didx.ptr, y + do, ld, bytes, 64);
                    },
                    1 => {
                        other.gather = f;
                        try o.gatherRows(e.s, dsrc.ptr + so, 900, didx.ptr, y + do, bytes, 64, flags.ptr + @as(u64, if (second) 4 else 0));
                    },
                    2 => {
                        other.copy16 = f;
                        if (aligned) try o.copyRows(e.s, dsrc.ptr, 2048, y, 1024 + 512, 1024, 300);
                    },
                    else => {
                        other.strided = f;
                        if (!aligned) try o.copyRows(e.s, dsrc.ptr + 1, 1001, y + 2, 1013, 999, 300);
                    },
                }
            }
            try download(e, y1, 0, b1);
            try download(e, y2, 0, b2);
            var fl: [2]u32 = undefined;
            try download(e, flags, 0, fl[0..]);
            var d = diffWords(u8, b1, b2, never8);
            if (fl[0] != fl[1]) d.other += 1;
            if ((mi == 2 and !aligned) or (mi == 3 and aligned)) continue; // one path each through copyRows
            noteDiff(r, x.label, m[1], b1.len, d);
            if (d.other > 0) bad += 1;
        }
    }

    // argmax: the designed rows at the vocabulary's width and at 1,025
    if (try set.get("tf_ds_argmax_rows_kernel")) |f| {
        other.argmax_rows = f;
        const nan = std.math.nan(f32);
        for ([_]usize{ 129280, 1025 }) |n| {
            const rows: usize = 8;
            const xs = try a.alloc(f32, rows * n);
            for (xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(mix(i + n * 7) % 2_000_001)) - 1_000_000)) / 64.0;
            xs[1 * n + n / 7] = 1e9;
            xs[1 * n + n - 1] = 1e9;
            xs[2 * n] = nan;
            xs[3 * n + n / 2] = nan;
            @memset(xs[4 * n ..][0..n], -std.math.inf(f32));
            @memset(xs[5 * n ..][0..n], nan);
            @memset(xs[6 * n ..][0..n], 3.5);
            var dx = try upload(e, xs);
            defer dx.free();
            var z1 = try gpu.DeviceBuffer.alloc(e.d, rows * 4);
            defer z1.free();
            var z2 = try gpu.DeviceBuffer.alloc(e.d, rows * 4);
            defer z2.free();
            try e.o.argmaxRows(e.s, dx.ptr, n, n, rows, z1.ptr);
            try other.argmaxRows(e.s, dx.ptr, n, n, rows, z2.ptr);
            var t1: [8]u32 = undefined;
            var t2: [8]u32 = undefined;
            try download(e, z1, 0, t1[0..]);
            try download(e, z2, 0, t2[0..]);
            const d = diffWords(u32, &t1, &t2, never32);
            noteDiff(r, x.label, if (n == 1025) "argmax_rows n 1025" else "argmax_rows n 129280", rows, d);
            if (d.other > 0) bad += 1;
        }
    } else {
        r.note(" argmax_rows absent;", .{});
        missing += 1;
    }
    r.note(" {d} absent", .{missing});
    try r.expect(bad == 0, "{d} operations disagree with glue.* outside NaN payloads", .{bad});
}

/// w-exl3's EXL3 kernels bound as the model binds them (hip_load.exl3Prefill / exl3Linear / exl3Experts: every kit entry
/// of those ids with the family's parameter list, AbiMismatch otherwise), and every specialization the TP2 checkpoint
/// reaches present (ADAPTER section 10: unpack and glinear k2 {8, 10, 12, 16}, glinear at 4 and 8 warps; grouped_mma,
/// grouped_rows and grouped_cp epilogues 1 and 2 for the K2 ranges (8, 8) and (2, 10)). The kernels' arithmetic is
/// graded on real tensors by w-exl3's harness (contract per numerics.json); this checks the binding.
fn exl3Bindings(r: *Receipt, e: *const Env) !void {
    const cb = 2;
    var pf = try hip_load.exl3Prefill(e.kit);
    defer pf.unload();
    var lg = try hip_load.exl3Linear(e.kit);
    defer lg.unload();
    var xk = try hip_load.exl3Experts(e.kit);
    defer xk.unload();
    var missing: usize = 0;
    for ([_]usize{ 8, 10, 12, 16 }) |k2| {
        if (pf.unpack[k2][cb] == null) missing += 1;
        for (lg.glinear[k2][cb]) |g| if (g == null) {
            missing += 1;
        };
    }
    for ([_]usize{ 0, 1 }) |rg| {
        if (xk.mma[rg] == null) missing += 1;
        if (xk.rows[rg] == null) missing += 1;
        for (xk.cp[rg]) |c| if (c == null) {
            missing += 1;
        };
    }
    var n_entries: usize = 0;
    for (e.kit.entries) |entry| n_entries += @intFromBool(std.mem.startsWith(u8, entry.k.id, "exl3."));
    r.note("{d} exl3 kit entries bound with the family's parameter lists; {d} of the 29 TP2 specializations missing", .{ n_entries, missing });
    try r.expect(missing == 0, "exl3 specializations missing: {d}", .{missing});
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator(); // everything here lives until exit
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    var kit_dir: ?[]const u8 = null;
    var receipt_path: ?[]const u8 = null;
    var crosses: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i + 1 < args.len) : (i += 2) {
        if (std.mem.eql(u8, args[i], "--kit")) kit_dir = args[i + 1];
        if (std.mem.eql(u8, args[i], "--receipt")) receipt_path = args[i + 1];
        if (std.mem.eql(u8, args[i], "--cross")) try crosses.append(gpa, args[i + 1]);
    }
    if (kit_dir == null) {
        std.debug.print("usage: tf-dsv41-hip-glue-test --kit DIR [--receipt PATH] [--cross LABEL=DIR]...\n", .{});
        return 2;
    }
    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try ctx.caps();
    // the serve kit's native half (glue.hip + aux.hip), every kernel bound with the family's parameter lists
    const kit = try gpu.Kit.open(gpa, io, &d, kit_dir.?, caps.archName(), .{ .profile = true, .drafts = true });
    defer kit.deinit();
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    var o = try hip_load.ops(kit, caps.wall_clock_hz);
    defer o.unload();
    var x = try hip_load.exact(kit);
    defer x.unload();
    std.debug.print("kit {s}: {d} objects, {d} kernels, manifest sha256 {s}, strict {}; every helper bound\n", .{ kit_dir.?, kit.m.objects.len, kit.m.kernels.len, &kit.manifest_sha256, kit.strict });
    const env: Env = .{ .gpa = std.heap.page_allocator, .d = &d, .s = s, .kit = kit, .o = o, .wall_clock_hz = caps.wall_clock_hz };
    var r: Receipt = .{ .gpa = gpa };
    try r.run("bf16_to_f32", bf16ToF32, &env);
    try r.run("f16_to_f32", f16ToF32, &env);
    try r.run("add2_bf16", add2Bf16, &env);
    try r.run("scatter_rows", scatterRows, &env);
    try r.run("gather_rows", gatherRows, &env);
    try r.run("copy_rows_strided_copy", copyRows, &env);
    try r.run("argmax_rows", argmaxRows, &env);
    try r.run("markov_race", markovRace, &env);
    try r.run("clock", clock, &env);
    try r.run("hipblas_fp64", hipblas, &env);
    try r.run("exl3_bindings", exl3Bindings, &env);
    for (crosses.items) |spec| {
        const eq = std.mem.indexOfScalar(u8, spec, '=') orelse return error.BadCross; // --cross LABEL=DIR
        const cx: Cross = .{ .e = &env, .io = io, .label = spec[0..eq], .dir = spec[eq + 1 ..] };
        try r.run(try std.fmt.allocPrint(gpa, "cross_{s}", .{cx.label}), crossCheck, &cx);
    }
    // strict mode counted every launch: the glue kernels each ran
    var unlaunched: usize = 0;
    for (kit.entries, 0..) |entry, j| {
        if (kit.strict and std.mem.startsWith(u8, entry.k.id, "glue.") and kit.launches(@intCast(j)) == 0) {
            std.debug.print("never launched: {s}\n", .{entry.k.id});
            unlaunched += 1;
        }
    }
    std.debug.print("{d} passed, {d} failed; glue kernels never launched: {d}\n", .{ r.passed, r.failed, unlaunched });
    if (receipt_path) |p| {
        const text = try std.fmt.allocPrint(gpa, "{{\n  \"receipt\": \"tf-dsv41-hip-glue-test\",\n  \"schema\": 1,\n  \"arch\": \"{s}\",\n  \"kit\": \"{s}\",\n  \"kit_manifest_sha256\": \"{s}\",\n  \"strict\": {},\n  \"wall_clock_hz\": {d},\n  \"glue_never_launched\": {d},\n  \"passed\": {d},\n  \"failed\": {d},\n  \"checks\": [\n{s}\n  ]\n}}\n", .{ caps.archName(), kit_dir.?, &kit.manifest_sha256, kit.strict, caps.wall_clock_hz, unlaunched, r.passed, r.failed, r.checks.items });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = text });
    }
    return if (r.failed == 0 and unlaunched == 0) 0 else 1;
}

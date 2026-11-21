//! The collectives' deadlines without a GPU: a late peer against a lost one (tcp_wire, tcp_proxy), and their settings.
const std = @import("std");
const wire = @import("tcp_wire.zig");
const tcp_coll = @import("tcp_coll.zig");
const proxy = @import("tcp_proxy.zig");

const c = std.c;
const testing = std.testing;
const ms = std.time.ns_per_ms;
const epoch: [16]u8 = @splat(5);
const opt: wire.Options = .{ .busy_poll_us = 0, .rto_min_us = 0, .progress_ns = 200 * ms, .arrival_ns = 2000 * ms };

fn socketPair() ![2]c.fd_t {
    var sv: [2]c.fd_t = undefined;
    if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
    return sv;
}

fn sleepNs(ns: u64) void {
    var ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = c.nanosleep(&ts, null);
}

/// Waits until the peer's bytes are readable on `fd` (peeked, not consumed): the peer has entered its exchange.
fn awaitPeer(fd: c.fd_t) !void {
    var b: [1]u8 = undefined;
    const t0 = wire.monotonicNs();
    while (c.recv(fd, &b, 1, std.os.linux.MSG.PEEK | std.os.linux.MSG.DONTWAIT) <= 0) {
        if (wire.monotonicNs() - t0 > 10 * std.time.ns_per_s) return error.PeerNeverEntered;
        sleepNs(ms);
    }
}

fn pattern(k: usize, rank: u16, i: usize) u8 {
    return @truncate(i *% 31 +% k *% 7 +% @as(usize, rank) *% 101);
}

/// One rank's exchanges of `sizes`, each `delay_ns` after the peer entered it; the peer's bytes and arrival waits kept.
const Side = struct {
    w: *wire.Wire,
    sizes: []const usize,
    delay_ns: u64 = 0,
    result: ?wire.Error = null,
    bad: usize = 0,
    waits: [2]u64 = .{ 0, 0 },

    fn run(s: *Side) void {
        const a = std.heap.page_allocator;
        const out = a.alloc(u8, 1 << 20) catch return s.broke();
        defer a.free(out);
        const in = a.alloc(u8, 1 << 20) catch return s.broke();
        defer a.free(in);
        const peer = 1 - s.w.rank;
        for (s.sizes, 0..) |n, k| {
            if (s.delay_ns > 0) {
                awaitPeer(s.w.fds[0]) catch return s.broke();
                sleepNs(s.delay_ns);
            }
            for (out[0..n], 0..) |*b, i| b.* = pattern(k, s.w.rank, i);
            s.w.exchange(1, out[0..n], in[0..n]) catch |e| {
                s.result = e;
                return;
            };
            s.waits[k] = s.w.peer_wait_ns;
            for (in[0..n], 0..) |b, i| s.bad += @intFromBool(b != pattern(k, peer, i));
        }
    }

    fn broke(s: *Side) void {
        s.bad = std.math.maxInt(usize);
    }
};

test "a peer that arrives late completes the exchange, small or filling the socket buffer" {
    const sv = try socketPair();
    var w0 = wire.Wire.fromFds(0, &.{sv[0]}, epoch, opt);
    var w1 = wire.Wire.fromFds(1, &.{sv[1]}, epoch, opt);
    defer w0.close(std.time.ns_per_s);
    defer w1.close(std.time.ns_per_s);
    const sizes = [_]usize{ 16, 1 << 20 }; // 1 MiB: rank 0's sends stall long past progress_ns before rank 1 comes
    var s0: Side = .{ .w = &w0, .sizes = &sizes };
    var s1: Side = .{ .w = &w1, .sizes = &sizes, .delay_ns = 600 * ms };
    const t = try std.Thread.spawn(.{}, Side.run, .{&s1});
    s0.run();
    t.join();
    try testing.expectEqual(@as(?wire.Error, null), s0.result);
    try testing.expectEqual(@as(?wire.Error, null), s1.result);
    try testing.expectEqual(@as(usize, 0), s0.bad + s1.bad);
    for (s0.waits) |ns| try testing.expect(ns >= 500 * ms and ns != wire.never_arrived);
    for (s1.waits) |ns| try testing.expect(ns < 500 * ms); // the late rank finds the peer's bytes queued
}

/// A wire whose other end is a raw socket the test drives by hand.
const Raw = struct {
    w: wire.Wire,
    fd: c.fd_t,

    fn open() !Raw {
        return openWith(opt);
    }

    fn openWith(o: wire.Options) !Raw {
        const sv = try socketPair();
        return .{ .w = wire.Wire.fromFds(0, &.{sv[0]}, epoch, o), .fd = sv[1] };
    }

    fn deinit(r: *Raw) void {
        r.w.close(std.time.ns_per_s);
        if (r.fd >= 0) _ = c.close(r.fd);
    }

    fn why(r: *const Raw) []const u8 {
        return std.mem.sliceTo(&r.w.why, 0);
    }
};

test "a peer that stops mid-transfer fails at the progress deadline, not the arrival limit" {
    var r = try Raw.open();
    defer r.deinit();
    const n = 4096;
    const site = 9;
    const h: wire.Header = .{ .magic = wire.magic, .version = wire.version, .flags = 0, .epoch = epoch, .seq = 1, .bytes = n, .site = site, .rank = 1, .part = 0, .parts = 1 };
    const half: [n / 2]u8 = @splat(3);
    try testing.expect(c.write(r.fd, std.mem.asBytes(&h), @sizeOf(wire.Header)) == @sizeOf(wire.Header));
    try testing.expect(c.write(r.fd, &half, half.len) == half.len);
    var out: [n]u8 = @splat(1);
    var in: [n]u8 = undefined;
    const t0 = wire.monotonicNs();
    try testing.expectError(error.Timeout, r.w.exchange(site, &out, &in));
    const took = wire.monotonicNs() - t0;
    try testing.expectEqualStrings("no progress within the deadline", r.why());
    try testing.expect(took >= 150 * ms and took < 1000 * ms);
    try testing.expect(r.w.peer_wait_ns < 150 * ms);
}

test "a peer that never arrives fails at the arrival limit" {
    var o = opt;
    o.arrival_ns = 600 * ms;
    var r = try Raw.openWith(o);
    defer r.deinit();
    var out: [16]u8 = @splat(1);
    var in: [16]u8 = undefined;
    const t0 = wire.monotonicNs();
    try testing.expectError(error.Timeout, r.w.exchange(1, &out, &in));
    const took = wire.monotonicNs() - t0;
    try testing.expectEqualStrings("the peer did not arrive within the limit", r.why());
    try testing.expect(took >= 550 * ms and took < 3000 * ms);
    try testing.expectEqual(wire.never_arrived, r.w.peer_wait_ns);
}

test "a closed peer fails at once, also while its arrival is awaited" {
    var out: [16]u8 = @splat(1);
    var in: [16]u8 = undefined;
    var a = try Raw.open();
    defer a.deinit();
    _ = c.close(a.fd);
    a.fd = -1;
    var t0 = wire.monotonicNs();
    try testing.expectError(error.PeerLost, a.w.exchange(1, &out, &in));
    try testing.expect(wire.monotonicNs() - t0 < 100 * ms);

    var b = try Raw.open();
    defer b.deinit();
    const Closer = struct {
        fn run(fd: *c.fd_t) void {
            sleepNs(100 * ms);
            _ = c.close(fd.*);
            fd.* = -1;
        }
    };
    t0 = wire.monotonicNs();
    const t = try std.Thread.spawn(.{}, Closer.run, .{&b.fd});
    const got = b.w.exchange(1, &out, &in);
    const took = wire.monotonicNs() - t0;
    t.join();
    try testing.expectError(error.PeerLost, got);
    try testing.expect(took < 1000 * ms);
}

/// One rank of tcp_proxy's slab protocol without a GPU: the test stages slot 1 by hand.
const ProxyRank = struct {
    shared: proxy.Shared = std.mem.zeroes(proxy.Shared),
    tx: [4 * 8192]u8 = @splat(0),
    rx: [4 * 8192]u8 = @splat(0),
    w: wire.Wire,
    stop: std.atomic.Value(u32) = .init(0),
    exchanged: std.atomic.Value(u64) = .init(0),
    code: u32 = 0,

    fn run(r: *ProxyRank) void {
        const k: proxy.Core = .{ .shared = &r.shared, .tx = &r.tx, .rx = &r.rx, .nslot = 4, .slot_bytes = 8192, .w = &r.w, .stop = &r.stop, .exchanged = &r.exchanged };
        r.code = k.run();
    }

    fn stage(r: *ProxyRank) void {
        for (r.tx[8192..][0..4096], 0..) |*b, i| b.* = pattern(1, r.w.rank, i);
        r.shared.slot[1].bytes = 4096;
        r.shared.slot[1].site = 7;
        @atomicStore(u64, &r.shared.slot[1].ready, 1, .release);
    }

    fn done(r: *ProxyRank) bool {
        return @atomicLoad(u64, &r.shared.slot[1].done, .acquire) == 1;
    }
};

test "the proxy keeps a pair whose peer stages a gather late" {
    const gpa = testing.allocator;
    const sv = try socketPair();
    const rk = [2]*ProxyRank{ try gpa.create(ProxyRank), try gpa.create(ProxyRank) };
    defer for (rk) |r| {
        r.w.close(std.time.ns_per_s);
        gpa.destroy(r);
    };
    for (rk, 0..) |r, i| r.* = .{ .w = wire.Wire.fromFds(@intCast(i), &.{sv[i]}, epoch, opt) };
    const th = [2]std.Thread{ try std.Thread.spawn(.{}, ProxyRank.run, .{rk[0]}), try std.Thread.spawn(.{}, ProxyRank.run, .{rk[1]}) };
    rk[0].stage();
    const entered = awaitPeer(rk[1].w.fds[0]); // rank 0's proxy is in its exchange (checked after the joins)
    sleepNs(600 * ms); // three times the progress deadline
    rk[1].stage();
    const t0 = wire.monotonicNs();
    while (!(rk[0].done() and rk[1].done()) and wire.monotonicNs() - t0 < 5 * std.time.ns_per_s) sleepNs(ms);
    for (rk) |r| r.stop.store(1, .release);
    for (th) |t| t.join();
    try entered;
    for (rk) |r| {
        try testing.expect(r.done());
        try testing.expectEqual(@as(u32, 0), r.code);
        try testing.expectEqual(@as(u32, 0), @atomicLoad(u32, &r.shared.abort, .acquire));
        for (r.rx[8192..][0..4096], 0..) |b, i| try testing.expectEqual(pattern(1, 1 - r.w.rank, i), b);
    }
    try testing.expect(rk[0].w.peer_wait_ns >= 500 * ms);
}

test "the arrival limit is TF_COLL_GPU_WAIT_S plus TF_COLL_PROGRESS_MS" {
    const env = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    defer _ = env.unsetenv("TF_COLL_GPU_WAIT_S");
    defer _ = env.unsetenv("TF_COLL_PROGRESS_MS");
    _ = env.unsetenv("TF_COLL_GPU_WAIT_S");
    _ = env.unsetenv("TF_COLL_PROGRESS_MS");
    try testing.expectEqual(@as(u64, 12_000 * ms), (try tcp_coll.settingsFromEnv(29600)).opt.arrival_ns);
    _ = env.setenv("TF_COLL_GPU_WAIT_S", "60", 1);
    try testing.expectEqual(@as(u64, 62_000 * ms), (try tcp_coll.settingsFromEnv(29600)).opt.arrival_ns);
    _ = env.setenv("TF_COLL_PROGRESS_MS", "500", 1);
    const s = try tcp_coll.settingsFromEnv(29600);
    try testing.expectEqual(@as(u64, 60_500 * ms), s.opt.arrival_ns);
    try testing.expectEqual(@as(u64, 500 * ms), s.opt.progress_ns);
}

test "a GPU wait expires only past its deadline and is slow once past the warn threshold" {
    const WaitClock = tcp_coll.WaitClock;
    const w: WaitClock = .{ .t0 = 1000 * ms, .limit_ns = 10_000 * ms, .warn_ns = 2000 * ms };
    try testing.expect(!w.expired(1000 * ms) and !w.slow(1000 * ms));
    try testing.expect(!w.slow(3000 * ms) and w.slow(3000 * ms + 1)); // strictly past 2 s
    try testing.expect(!w.expired(11_000 * ms) and w.expired(11_000 * ms + 1)); // strictly past 10 s
    try testing.expectEqual(@as(u64, 10_000), w.ms(11_000 * ms + 1));
    try testing.expect(!w.expired(0) and w.ms(0) == 0); // a clock read before t0 counts as no time
    const quiet: WaitClock = .{ .t0 = 0, .limit_ns = 60 * std.time.ns_per_s, .warn_ns = 0 };
    try testing.expect(!quiet.slow(59 * std.time.ns_per_s) and quiet.expired(61 * std.time.ns_per_s));
}

//! The TP2 collectives' data path over the USB4 links (tools/dsv41-hip/COLLECTIVES-DESIGN.md section 3): one TCP
//! socket per cable between the two ranks, a 48-byte header before every payload, and a nonblocking exchange that
//! spins on send/recv without poll() or sleeps. With SO_BUSY_POLL set, each recv on an empty queue runs the
//! thunderbolt_net NAPI poll in the calling thread; waiting for the receive interrupt instead costs ~128 us per
//! direction on these links (a 120 KiB request/response: 95-102 us spinning, 340-390 us waiting in poll()).
//! GPU-free: tcp_coll.zig stages device memory through it.
//!
//! The contract (coll.zig): both ranks call exchange with the same site and byte count in the same order; the headers
//! carry the session epoch, a sequence number, the site and the size, and any difference fails the call with
//! Mismatch. The first failure poisons the wire for good (later calls return it at once) and shuts the sockets down so
//! the peer's call fails promptly too. Deadlines: a call waits up to `arrival_ns` for the peer's first byte, then fails
//! with Timeout when no byte has moved in either direction for `progress_ns`; a closed peer fails it at once.
const std = @import("std");
const c = std.c;
const linux = std.os.linux;

extern "c" fn recvmsg(sockfd: c.fd_t, msg: *linux.msghdr, flags: u32) isize;

pub const magic: u32 = 0x31434654; // "TFC1"
pub const version: u16 = 1;
pub const max_links = 2;
/// Linux's TCP_RTO_MIN_US socket option (include/uapi/linux/tcp.h); kernels without it refuse it, which is reported.
const TCP_RTO_MIN_US: u32 = 45;

pub const Header = extern struct {
    magic: u32,
    version: u16,
    flags: u16,
    epoch: [16]u8,
    seq: u64,
    /// the whole payload of this call (before it is split over the links)
    bytes: u64,
    /// coll.Site: phase << 16 | layer
    site: u32,
    rank: u16,
    /// this link's part of the payload, and the number of parts
    part: u8,
    parts: u8,

    comptime {
        std.debug.assert(@sizeOf(Header) == 48);
    }
};

pub const Error = error{ PeerLost, Timeout, Mismatch, Aborted, SocketFailed };

pub const Options = struct {
    /// once the peer's bytes of a call flow, no byte moved in either direction for this long: Timeout
    progress_ns: u64 = 2 * std.time.ns_per_s,
    /// the peer's first byte of a call missing this long after the call began: Timeout (never less than progress_ns)
    arrival_ns: u64 = 0,
    /// SO_BUSY_POLL microseconds; 50 is the most an unprivileged socket may set here (net.core.busy_read)
    busy_poll_us: c_int = 50,
    /// TCP_RTO_MIN_US per socket (the kernel default floor is 200 ms; HZ=250 makes 8 ms the smallest it takes); 0 keeps
    /// the default
    rto_min_us: c_int = 8000,
    /// payloads of at least this many bytes are split evenly over every link
    stripe_min: usize = 4 << 20,
    /// connect retries for this long while the peer starts listening
    connect_ns: u64 = 300 * std.time.ns_per_s,
};

const State = enum(u32) { ok = 0, peer_lost, timeout, mismatch, aborted, socket_failed, closed };

fn stateError(s: State) Error {
    return switch (s) {
        .ok => unreachable,
        .peer_lost => error.PeerLost,
        .timeout => error.Timeout,
        .mismatch => error.Mismatch,
        .aborted, .closed => error.Aborted,
        .socket_failed => error.SocketFailed,
    };
}

fn errorState(e: Error) State {
    return switch (e) {
        error.PeerLost => .peer_lost,
        error.Timeout => .timeout,
        error.Mismatch => .mismatch,
        error.Aborted => .aborted,
        error.SocketFailed => .socket_failed,
    };
}

/// Wire.peer_wait_ns of a call that ended before the peer's first byte arrived.
pub const never_arrived = std.math.maxInt(u64);

pub fn monotonicNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const Wire = struct {
    fds: [max_links]c.fd_t = .{ -1, -1 },
    links: usize = 0,
    rank: u16,
    epoch: [16]u8,
    opt: Options,
    seq: u64 = 0,
    state: std.atomic.Value(u32) = .init(0),
    /// set while a call is inside exchange, so close can wait for it after aborting it
    busy: std.atomic.Value(u32) = .init(0),
    /// the first failure's description, for the log
    why: [96]u8 = @splat(0),
    /// how long the last call waited for the peer's first byte (never_arrived: it did not come); caller's thread only
    peer_wait_ns: u64 = 0,

    /// Rank 0: listen on each address (one per link, its USB4 address), take one connection on each, and check its
    /// hello (magic, version, rank, epoch, link index). Rank 1: connect to each, retrying while rank 0 starts.
    pub fn open(rank: u16, addrs: []const [4]u8, port: u16, epoch: [16]u8, opt: Options) Error!Wire {
        if (addrs.len == 0 or addrs.len > max_links) return error.SocketFailed;
        var w: Wire = .{ .rank = rank, .epoch = epoch, .opt = opt };
        errdefer w.closeFds();
        for (addrs, 0..) |ip, i| {
            const fd = if (rank == 0) try listenOne(ip, port + @as(u16, @intCast(i))) else try connectOne(ip, port + @as(u16, @intCast(i)), opt.connect_ns);
            w.fds[i] = fd;
            w.links = i + 1;
            tune(fd, opt);
            try w.hello(i);
        }
        return w;
    }

    /// Two wires over already-connected sockets (tests; socketpair or loopback).
    pub fn fromFds(rank: u16, fds: []const c.fd_t, epoch: [16]u8, opt: Options) Wire {
        var w: Wire = .{ .rank = rank, .epoch = epoch, .opt = opt };
        for (fds, 0..) |fd, i| {
            w.fds[i] = fd;
            tune(fd, opt);
        }
        w.links = fds.len;
        return w;
    }

    fn hello(w: *Wire, link: usize) Error!void {
        const mine = Header{ .magic = magic, .version = version, .flags = 0, .epoch = w.epoch, .seq = 0, .bytes = 0, .site = 0, .rank = w.rank, .part = @intCast(link), .parts = @intCast(w.links) };
        var theirs: Header = undefined;
        try sendAll(w.fds[link], std.mem.asBytes(&mine));
        try recvAll(w.fds[link], std.mem.asBytes(&theirs));
        if (theirs.magic != magic or theirs.version != version or !std.mem.eql(u8, &theirs.epoch, &w.epoch) or
            theirs.rank != 1 - w.rank or theirs.part != link) return error.Mismatch;
    }

    fn closeFds(w: *Wire) void {
        for (w.fds[0..w.links]) |*fd| {
            if (fd.* >= 0) _ = c.close(fd.*);
            fd.* = -1;
        }
    }

    /// The first failure, if any.
    pub fn status(w: *const Wire) ?Error {
        const s: State = @enumFromInt(w.state.load(.acquire));
        return if (s == .ok) null else stateError(s);
    }

    /// Fails the call in flight and every later one with Aborted (unless already failed) and shuts the sockets down,
    /// so the peer's call fails with PeerLost. Idempotent; any thread.
    pub fn abort(w: *Wire) void {
        w.fail(error.Aborted, "aborted");
    }

    fn fail(w: *Wire, e: Error, why: []const u8) void {
        if (w.state.cmpxchgStrong(0, @intFromEnum(errorState(e)), .acq_rel, .acquire) == null) {
            const n = @min(why.len, w.why.len - 1);
            @memcpy(w.why[0..n], why[0..n]);
            for (w.fds[0..w.links]) |fd| {
                if (fd >= 0) _ = c.shutdown(fd, linux.SHUT.RDWR);
            }
        }
    }

    /// Aborts a call in flight, waits up to `ns` for it to leave, and closes the sockets.
    pub fn close(w: *Wire, ns: u64) void {
        if (w.status() == null) _ = w.state.cmpxchgStrong(0, @intFromEnum(State.closed), .acq_rel, .acquire);
        for (w.fds[0..w.links]) |fd| {
            if (fd >= 0) _ = c.shutdown(fd, linux.SHUT.RDWR);
        }
        const t0 = monotonicNs();
        while (w.busy.load(.acquire) != 0 and monotonicNs() - t0 < ns) std.atomic.spinLoopHint();
        if (w.busy.load(.acquire) == 0) w.closeFds();
    }

    /// Sends `out` and receives the peer's equally long payload into `in`, both at once. The payload is split over the
    /// links when it is at least opt.stripe_min bytes and there is more than one link.
    pub fn exchange(w: *Wire, site: u32, out: []const u8, in: []u8) Error!void {
        w.peer_wait_ns = never_arrived;
        if (w.status()) |e| return e;
        if (out.len != in.len) return w.failed(error.Mismatch, "local send and receive sizes differ");
        _ = w.busy.fetchAdd(1, .acq_rel);
        defer _ = w.busy.fetchSub(1, .acq_rel);
        w.seq += 1;
        const parts: usize = if (w.links > 1 and out.len >= w.opt.stripe_min) w.links else 1;
        var io: [max_links]Part = undefined;
        const chunk = (out.len + parts - 1) / parts;
        for (0..parts) |i| {
            const lo = @min(out.len, i * chunk);
            const hi = @min(out.len, lo + chunk);
            io[i] = .{
                .fd = w.fds[i],
                .hdr_out = .{ .magic = magic, .version = version, .flags = 0, .epoch = w.epoch, .seq = w.seq, .bytes = out.len, .site = site, .rank = w.rank, .part = @intCast(i), .parts = @intCast(parts) },
                .out = out[lo..hi],
                .in = in[lo..hi],
            };
        }
        return w.run(io[0..parts]);
    }

    fn failed(w: *Wire, e: Error, why: []const u8) Error {
        w.fail(e, why);
        return w.status() orelse e;
    }

    const Part = struct {
        fd: c.fd_t,
        hdr_out: Header,
        hdr_in: Header = undefined,
        out: []const u8,
        in: []u8,
        sent: usize = 0, // header + payload bytes sent
        got: usize = 0, // header + payload bytes received
        checked: bool = false,

        fn total(p: *const Part) usize {
            return @sizeOf(Header) + p.out.len;
        }
    };

    fn run(w: *Wire, parts: []Part) Error!void {
        const t0 = monotonicNs();
        var last = t0;
        var idle: u32 = 0;
        while (true) {
            var moved = false;
            var all = true;
            for (parts) |*p| {
                if (p.sent < p.total()) {
                    all = false;
                    switch (sendPart(p)) {
                        .moved => moved = true,
                        .again => {},
                        .closed => return w.failed(error.PeerLost, "peer closed while sending"),
                        .err => return w.failed(error.SocketFailed, "send failed"),
                    }
                }
                if (p.got < p.total()) {
                    all = false;
                    switch (recvPart(p)) {
                        .moved => {
                            moved = true;
                            // marked here: a small call can end in the pass its first byte arrives
                            if (w.peer_wait_ns == never_arrived) w.peer_wait_ns = monotonicNs() - t0;
                        },
                        .again => {},
                        .closed => return w.failed(error.PeerLost, "peer closed while receiving"),
                        .err => return w.failed(error.SocketFailed, "receive failed"),
                    }
                    if (!p.checked and p.got >= @sizeOf(Header)) {
                        if (!w.headerMatches(p)) {
                            logMismatch(w, p);
                            return w.failed(error.Mismatch, "peer header differs (sequence, site, size or epoch)");
                        }
                        p.checked = true;
                    }
                }
            }
            if (all) return;
            if (w.state.load(.monotonic) != 0) return w.status().?;
            if (moved) {
                idle = 0;
                last = monotonicNs();
                continue;
            }
            idle +%= 1;
            if (idle % 1024 != 0) continue;
            const now = monotonicNs();
            if (w.peer_wait_ns == never_arrived) {
                // the peer has not entered this call: our stalled sends and empty receives are not judged yet
                if (now - t0 > @max(w.opt.arrival_ns, w.opt.progress_ns))
                    return w.failed(error.Timeout, "the peer did not arrive within the limit");
            } else if (now - last > w.opt.progress_ns) {
                return w.failed(error.Timeout, "no progress within the deadline");
            }
        }
    }

    /// Both headers of an exchange that differ, one line (no payload): where the ranks' sequences parted. Site is
    /// coll.Phase's number / layer.
    fn logMismatch(w: *const Wire, p: *const Part) void {
        const h = p.hdr_in;
        const m = p.hdr_out;
        const other = h.magic != magic or h.version != version or !std.mem.eql(u8, &h.epoch, &w.epoch);
        std.log.warn("collectives: rank {d} wire header mismatch: this rank's seq {d} site {d}/{d} bytes {d} part {d}/{d}, the peer's (rank {d}) seq {d} site {d}/{d} bytes {d} part {d}/{d}{s}", .{
            w.rank, m.seq, m.site >> 16, m.site & 0xffff, m.bytes, m.part, m.parts,
            h.rank, h.seq, h.site >> 16, h.site & 0xffff, h.bytes, h.part, h.parts, if (other) "; magic, version or epoch differs" else "",
        });
    }

    fn headerMatches(w: *const Wire, p: *const Part) bool {
        const h = p.hdr_in;
        const m = p.hdr_out;
        return h.magic == magic and h.version == version and std.mem.eql(u8, &h.epoch, &w.epoch) and h.seq == m.seq and
            h.bytes == m.bytes and h.site == m.site and h.rank == 1 - w.rank and h.part == m.part and h.parts == m.parts and
            h.flags == 0;
    }
};

const Step = enum { moved, again, closed, err };

fn sendPart(p: *Wire.Part) Step {
    const hdr = std.mem.asBytes(&p.hdr_out);
    var iov: [2]std.posix.iovec_const = undefined;
    var n_iov: usize = 0;
    if (p.sent < hdr.len) {
        iov[0] = .{ .base = hdr.ptr + p.sent, .len = hdr.len - p.sent };
        iov[1] = .{ .base = p.out.ptr, .len = p.out.len };
        n_iov = if (p.out.len > 0) 2 else 1;
    } else {
        const off = p.sent - hdr.len;
        iov[0] = .{ .base = p.out.ptr + off, .len = p.out.len - off };
        n_iov = 1;
    }
    const msg: c.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = n_iov, .control = null, .controllen = 0, .flags = 0 };
    const r = c.sendmsg(p.fd, &msg, linux.MSG.DONTWAIT | linux.MSG.NOSIGNAL);
    if (r > 0) {
        p.sent += @intCast(r);
        return .moved;
    }
    return switch (c.errno(r)) {
        .AGAIN, .INTR => .again,
        .PIPE, .CONNRESET => .closed,
        else => .err,
    };
}

fn recvPart(p: *Wire.Part) Step {
    const hdr = std.mem.asBytes(&p.hdr_in);
    var iov: [2]std.posix.iovec = undefined;
    var n_iov: usize = 0;
    if (p.got < hdr.len) {
        iov[0] = .{ .base = hdr.ptr + p.got, .len = hdr.len - p.got };
        iov[1] = .{ .base = p.in.ptr, .len = p.in.len };
        n_iov = if (p.in.len > 0) 2 else 1;
    } else {
        const off = p.got - hdr.len;
        iov[0] = .{ .base = p.in.ptr + off, .len = p.in.len - off };
        n_iov = 1;
    }
    var msg: linux.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = n_iov, .control = null, .controllen = 0, .flags = 0 };
    const r = recvmsg(p.fd, &msg, linux.MSG.DONTWAIT);
    if (r > 0) {
        p.got += @intCast(r);
        return .moved;
    }
    if (r == 0) return .closed;
    return switch (c.errno(r)) {
        .AGAIN, .INTR => .again,
        .CONNRESET => .closed,
        else => .err,
    };
}

/// TCP_NODELAY, SO_BUSY_POLL and the RTO floor; a refused option is reported, not fatal (the exchange still works,
/// only slower or with the kernel's 200 ms RTO floor).
fn tune(fd: c.fd_t, opt: Options) void {
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.IPPROTO.TCP, linux.TCP.NODELAY, std.mem.asBytes(&one), @sizeOf(c_int));
    if (opt.busy_poll_us > 0 and c.setsockopt(fd, c.SOL.SOCKET, linux.SO.BUSY_POLL, std.mem.asBytes(&opt.busy_poll_us), @sizeOf(c_int)) != 0)
        std.log.warn("tcp_wire: SO_BUSY_POLL={d} refused; receives will wait for interrupts", .{opt.busy_poll_us});
    if (opt.rto_min_us > 0 and c.setsockopt(fd, c.IPPROTO.TCP, TCP_RTO_MIN_US, std.mem.asBytes(&opt.rto_min_us), @sizeOf(c_int)) != 0)
        std.log.warn("tcp_wire: TCP_RTO_MIN_US={d} refused; retransmission timeouts keep the kernel floor", .{opt.rto_min_us});
}

fn sockaddr(ip: [4]u8, port: u16) c.sockaddr.in {
    return .{ .port = std.mem.nativeToBig(u16, port), .addr = @bitCast(ip) };
}

fn listenOne(ip: [4]u8, port: u16) Error!c.fd_t {
    const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
    if (fd < 0) return error.SocketFailed;
    defer _ = c.close(fd);
    const one: c_int = 1;
    _ = c.setsockopt(fd, c.SOL.SOCKET, c.SO.REUSEADDR, std.mem.asBytes(&one), @sizeOf(c_int));
    var a = sockaddr(ip, port);
    if (c.bind(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) != 0) return error.SocketFailed;
    if (c.listen(fd, 1) != 0) return error.SocketFailed;
    const peer = c.accept(fd, null, null);
    if (peer < 0) return error.SocketFailed;
    return peer;
}

fn connectOne(ip: [4]u8, port: u16, ns: u64) Error!c.fd_t {
    const t0 = monotonicNs();
    while (true) {
        const fd = c.socket(c.AF.INET, c.SOCK.STREAM, 0);
        if (fd < 0) return error.SocketFailed;
        var a = sockaddr(ip, port);
        if (c.connect(fd, @ptrCast(&a), @sizeOf(c.sockaddr.in)) == 0) return fd;
        _ = c.close(fd);
        if (monotonicNs() - t0 > ns) return error.SocketFailed;
        var d: c.timespec = .{ .sec = 0, .nsec = 100 * std.time.ns_per_ms };
        _ = c.nanosleep(&d, null);
    }
}

/// Blocking full send and receive for the hellos (before the wire is in use).
fn sendAll(fd: c.fd_t, buf: []const u8) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = c.send(fd, buf.ptr + done, buf.len - done, linux.MSG.NOSIGNAL);
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return error.PeerLost;
        }
        done += @intCast(n);
    }
}

fn recvAll(fd: c.fd_t, buf: []u8) Error!void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = c.recv(fd, buf.ptr + done, buf.len - done, 0);
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return error.PeerLost;
        }
        done += @intCast(n);
    }
}

// ---------------------------------------------------------------------------------------------------------------- tests

const testing = std.testing;

fn pairs(n: usize) ![2][max_links]c.fd_t {
    var a: [max_links]c.fd_t = .{ -1, -1 };
    var b: [max_links]c.fd_t = .{ -1, -1 };
    for (0..n) |i| {
        var sv: [2]c.fd_t = undefined;
        if (c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &sv) != 0) return error.SocketFailed;
        a[i] = sv[0];
        b[i] = sv[1];
    }
    return .{ a, b };
}

fn fill(buf: []u8, seed: u64) void {
    var x = seed *% 0x9E3779B97F4A7C15 +% 1;
    for (buf) |*b| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b.* = @truncate(x);
    }
}

const Peer = struct {
    w: *Wire,
    sizes: []const usize,
    sites: []const u32,
    result: ?Error = null,
    bad: usize = 0,

    fn run(p: *Peer) void {
        const a = std.heap.page_allocator;
        const out = a.alloc(u8, 3 << 20) catch return;
        defer a.free(out);
        const in = a.alloc(u8, 3 << 20) catch return;
        defer a.free(in);
        const want = a.alloc(u8, 3 << 20) catch return;
        defer a.free(want);
        for (p.sizes, p.sites, 0..) |n, s, k| {
            fill(out[0..n], k * 2 + p.w.rank);
            p.w.exchange(s, out[0..n], in[0..n]) catch |e| {
                p.result = e;
                return;
            };
            fill(want[0..n], k * 2 + (1 - p.w.rank));
            if (!std.mem.eql(u8, want[0..n], in[0..n])) p.bad += 1;
        }
    }
};

const test_epoch: [16]u8 = @splat(7);
const test_opt: Options = .{ .busy_poll_us = 0, .rto_min_us = 0, .progress_ns = 300 * std.time.ns_per_ms, .stripe_min = 1 << 20 };

fn runPair(links: usize, sizes0: []const usize, sites0: []const u32, sizes1: []const usize, sites1: []const u32) ![2]Peer {
    const fds = try pairs(links);
    var w0 = Wire.fromFds(0, fds[0][0..links], test_epoch, test_opt);
    var w1 = Wire.fromFds(1, fds[1][0..links], test_epoch, test_opt);
    defer w0.close(std.time.ns_per_s);
    defer w1.close(std.time.ns_per_s);
    var p0: Peer = .{ .w = &w0, .sizes = sizes0, .sites = sites0 };
    var p1: Peer = .{ .w = &w1, .sizes = sizes1, .sites = sites1 };
    const t = try std.Thread.spawn(.{}, Peer.run, .{&p1});
    p0.run();
    t.join();
    return .{ p0, p1 };
}

test "payloads cross in both directions, one link and striped over two" {
    const sizes = [_]usize{ 0, 40, 20480, 122880, 1048576, 3 << 20, 7 };
    const sites = [_]u32{ 1, 2, 3, 4, 5, 6, 7 };
    for ([_]usize{ 1, 2 }) |links| {
        const r = try runPair(links, &sizes, &sites, &sizes, &sites);
        try testing.expectEqual(@as(?Error, null), r[0].result);
        try testing.expectEqual(@as(?Error, null), r[1].result);
        try testing.expectEqual(@as(usize, 0), r[0].bad + r[1].bad);
    }
}

test "a different site, size or order fails both ranks with Mismatch" {
    const r = try runPair(1, &.{ 64, 64 }, &.{ 1, 2 }, &.{ 64, 64 }, &.{ 1, 3 });
    try testing.expectEqual(@as(?Error, error.Mismatch), r[0].result);
    try testing.expect(r[1].result != null);
    const r2 = try runPair(1, &.{128}, &.{1}, &.{256}, &.{1});
    try testing.expect(r2[0].result != null and r2[1].result != null);
}

test "a silent peer times out, a closed peer is lost, and the failure sticks" {
    const fds = try pairs(1);
    var w0 = Wire.fromFds(0, fds[0][0..1], test_epoch, test_opt);
    var buf: [16]u8 = @splat(1);
    var in: [16]u8 = undefined;
    const t0 = monotonicNs();
    try testing.expectError(error.Timeout, w0.exchange(1, &buf, &in));
    const took = monotonicNs() - t0;
    try testing.expect(took >= 250 * std.time.ns_per_ms and took < 3 * std.time.ns_per_s);
    try testing.expectError(error.Timeout, w0.exchange(1, &buf, &in));
    w0.close(std.time.ns_per_s);

    const fds2 = try pairs(1);
    var w1 = Wire.fromFds(0, fds2[0][0..1], test_epoch, test_opt);
    defer w1.close(std.time.ns_per_s);
    _ = c.close(fds2[1][0]);
    try testing.expectError(error.PeerLost, w1.exchange(1, &buf, &in));
}

test "abort from another thread ends a waiting call; the peer then loses its link" {
    const fds = try pairs(1);
    var w0 = Wire.fromFds(0, fds[0][0..1], test_epoch, .{ .busy_poll_us = 0, .rto_min_us = 0, .progress_ns = 10 * std.time.ns_per_s });
    var w1 = Wire.fromFds(1, fds[1][0..1], test_epoch, test_opt);
    defer w0.close(std.time.ns_per_s);
    defer w1.close(std.time.ns_per_s);
    const Aborter = struct {
        fn run(w: *Wire) void {
            var d: c.timespec = .{ .sec = 0, .nsec = 50 * std.time.ns_per_ms };
            _ = c.nanosleep(&d, null);
            w.abort();
        }
    };
    const t = try std.Thread.spawn(.{}, Aborter.run, .{&w0});
    var buf: [16]u8 = @splat(1);
    var in: [16]u8 = undefined;
    const t0 = monotonicNs();
    try testing.expectError(error.Aborted, w0.exchange(1, &buf, &in));
    try testing.expect(monotonicNs() - t0 < 2 * std.time.ns_per_s);
    t.join();
    try testing.expectError(error.PeerLost, w1.exchange(1, &buf, &in));
}

test "a peer from another session is refused" {
    const fds = try pairs(1);
    var w0 = Wire.fromFds(0, fds[0][0..1], test_epoch, test_opt);
    var w1 = Wire.fromFds(1, fds[1][0..1], @as([16]u8, @splat(9)), test_opt);
    defer w0.close(std.time.ns_per_s);
    defer w1.close(std.time.ns_per_s);
    var p0: Peer = .{ .w = &w0, .sizes = &.{32}, .sites = &.{1} };
    var p1: Peer = .{ .w = &w1, .sizes = &.{32}, .sites = &.{1} };
    const t = try std.Thread.spawn(.{}, Peer.run, .{&p1});
    p0.run();
    t.join();
    try testing.expectEqual(@as(?Error, error.Mismatch), p0.result);
    try testing.expectEqual(@as(?Error, error.Mismatch), p1.result);
}

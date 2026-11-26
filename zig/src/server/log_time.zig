//! ``TF_LOG_TIME=1``: each log line starts with its UTC time, ``2026-11-26T14:03:12.123456Z `` (RFC 3339, µs).
const std = @import("std");
const Writer = std.Io.Writer;

pub const stamp_len = 28;

/// Whether ``TF_LOG_TIME``'s value turns stamping on: exactly ``1``.
pub fn on(value: ?[]const u8) bool {
    return std.mem.eql(u8, value orelse return false, "1");
}

var state: std.atomic.Value(u8) = .init(0); // 0: not read yet, 1: off, 2: on

/// ``TF_LOG_TIME``, read once.
pub fn enabled() bool {
    const s = state.load(.monotonic);
    if (s != 0) return s == 2;
    const yes = on(if (std.c.getenv("TF_LOG_TIME")) |v| std.mem.span(v) else null);
    state.store(if (yes) 2 else 1, .monotonic);
    return yes;
}

/// Wall-clock ns.
pub fn nowNs() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.REALTIME, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// ``ns`` as a line's stamp, cut to µs; a time before the epoch is the epoch.
pub fn stamp(buf: *[stamp_len]u8, ns: i128) []const u8 {
    const t: u64 = @intCast(std.math.clamp(ns, 0, std.math.maxInt(u64)));
    const es: std.time.epoch.EpochSeconds = .{ .secs = t / std.time.ns_per_s };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const secs = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}.{d:0>6}Z ", .{
        yd.year,                md.month.numeric(),        md.day_index + 1,
        secs.getHoursIntoDay(), secs.getMinutesIntoHour(), secs.getSecondsIntoMinute(),
        t % std.time.ns_per_s / std.time.ns_per_us,
    }) catch unreachable;
}

/// A writer that passes its bytes to ``out`` with a stamp before the first byte of each line.
pub const Stamper = struct {
    out: *Writer,
    clock: *const fn () i128 = nowNs,
    at_start: bool = true,
    interface: Writer,

    pub fn init(out: *Writer, buffer: []u8) Stamper {
        return .{ .out = out, .interface = .{ .vtable = &.{ .drain = drain, .flush = flush }, .buffer = buffer } };
    }

    fn put(s: *Stamper, bytes: []const u8) Writer.Error!void {
        var rest = bytes;
        while (rest.len > 0) {
            if (s.at_start) {
                var buf: [stamp_len]u8 = undefined;
                try s.out.writeAll(stamp(&buf, s.clock()));
                s.at_start = false;
            }
            const end = if (std.mem.indexOfScalar(u8, rest, '\n')) |i| i + 1 else rest.len;
            try s.out.writeAll(rest[0..end]);
            s.at_start = rest[end - 1] == '\n';
            rest = rest[end..];
        }
    }

    fn drain(w: *Writer, data: []const []const u8, splat: usize) Writer.Error!usize {
        const s: *Stamper = @alignCast(@fieldParentPtr("interface", w));
        try s.put(w.buffered());
        w.end = 0;
        var n: usize = 0;
        for (data[0 .. data.len - 1]) |d| {
            try s.put(d);
            n += d.len;
        }
        const last = data[data.len - 1];
        for (0..splat) |_| try s.put(last);
        return n + last.len * splat;
    }

    fn flush(w: *Writer) Writer.Error!void {
        const s: *Stamper = @alignCast(@fieldParentPtr("interface", w));
        try s.put(w.buffered());
        w.end = 0;
        try s.out.flush();
    }
};

/// The binaries' ``std.log``: ``defaultLog``'s lines, stamped when ``TF_LOG_TIME`` is on.
pub fn logFn(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    if (!enabled()) return std.log.defaultLog(level, scope, format, args);
    const io = std.Options.debug_io;
    const prev = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(prev);
    var buf: [4096]u8 = undefined;
    const stderr = std.debug.lockStderr(&buf);
    defer std.debug.unlockStderr();
    var s: Stamper = .init(&stderr.file_writer.interface, &.{});
    std.log.defaultLogFileTerminal(level, scope, format, args, .{ .writer = &s.interface, .mode = stderr.terminal_mode }) catch {};
}

/// ``std.debug.print``, each line stamped when ``TF_LOG_TIME`` is on.
pub fn print(comptime fmt: []const u8, args: anytype) void {
    if (!enabled()) return std.debug.print(fmt, args);
    const io = std.Options.debug_io;
    const prev = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(prev);
    var buf: [4096]u8 = undefined;
    const stderr = std.debug.lockStderr(&buf);
    defer std.debug.unlockStderr();
    var s: Stamper = .init(&stderr.file_writer.interface, &.{});
    s.interface.print(fmt, args) catch {};
}

fn zero() i128 {
    return 0;
}

test "a stamp" {
    var buf: [stamp_len]u8 = undefined;
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000000Z ", stamp(&buf, 0));
    try std.testing.expectEqualStrings("2026-11-26T14:03:12.123456Z ", stamp(&buf, 1_795_701_792_123_456_789));
    try std.testing.expectEqualStrings("2026-11-26T14:03:12.000042Z ", stamp(&buf, 1_795_701_792_000_042_000));
    try std.testing.expectEqualStrings("2024-02-29T23:59:59.999999Z ", stamp(&buf, 1_709_251_199_999_999_999));
    try std.testing.expectEqualStrings("1970-01-01T00:00:00.000000Z ", stamp(&buf, -5));
}

test "only 1 turns it on" {
    for ([_]?[]const u8{ null, "", "0", "true", "11", "1 " }) |v| try std.testing.expect(!on(v));
    try std.testing.expect(on("1"));
}

test "a stamp a line" {
    const S = "1970-01-01T00:00:00.000000Z ";
    const cases = [_]struct { writes: []const []const u8, want: []const u8 }{
        .{ .writes = &.{"a\nb\n"}, .want = S ++ "a\n" ++ S ++ "b\n" },
        .{ .writes = &.{ "par", "tial\n" }, .want = S ++ "partial\n" },
        .{ .writes = &.{"\n"}, .want = S ++ "\n" },
        .{ .writes = &.{ "a\nhal", "f", "" }, .want = S ++ "a\n" ++ S ++ "half" },
    };
    for (cases) |c| {
        var buf: [256]u8 = undefined;
        var out: Writer = .fixed(&buf);
        var s: Stamper = .init(&out, &.{});
        s.clock = zero;
        for (c.writes) |bytes| try s.interface.writeAll(bytes);
        try s.interface.flush();
        try std.testing.expectEqualStrings(c.want, out.buffered());
    }
}

test "a log line's body stamped" {
    const S = "1970-01-01T00:00:00.000000Z ";
    inline for (.{ .default, .x }) |scope| {
        var direct_buf: [256]u8 = undefined;
        var direct: Writer = .fixed(&direct_buf);
        try std.log.defaultLogFileTerminal(.info, scope, "m {d}\nn", .{7}, .{ .writer = &direct, .mode = .no_color });
        var buf: [256]u8 = undefined;
        var out: Writer = .fixed(&buf);
        var s: Stamper = .init(&out, &.{});
        s.clock = zero;
        try std.log.defaultLogFileTerminal(.info, scope, "m {d}\nn", .{7}, .{ .writer = &s.interface, .mode = .no_color });
        var want_buf: [256]u8 = undefined;
        var want: Writer = .fixed(&want_buf);
        var lines = std.mem.splitScalar(u8, direct.buffered()[0 .. direct.buffered().len - 1], '\n');
        while (lines.next()) |l| try want.print(S ++ "{s}\n", .{l});
        try std.testing.expectEqualStrings(want.buffered(), out.buffered());
        try std.testing.expectStringStartsWith(out.buffered(), if (scope == .x) S ++ "info(x): m 7\n" else S ++ "info: m 7\n");
    }
}

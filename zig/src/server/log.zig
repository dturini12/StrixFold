//! Server output: ``[tensorfold]`` lines on stdout, and the live status line they clear first.
const std = @import("std");
const Conn = @import("http_conn.zig").Conn;
const afterChars = @import("reply_text.zig").afterChars;
const log_time = @import("log_time.zig");

var global_io: ?std.Io = null;
var mutex: std.Io.Mutex = .init;
var shown = false; // the live line is on screen
var line_start = true; // the newest write ended its line, so a redraw cannot split it
var quiet = false;

pub fn init(io_: std.Io, silent: bool) void {
    global_io = io_;
    quiet = silent;
}

pub fn io() std.Io {
    return global_io orelse std.Io.Threaded.global_single_threaded.io();
}

const clear = "\r\x1b[2K";

fn writeOut(bytes: []const u8) void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = std.posix.system.write(1, bytes[off..].ptr, bytes.len - off);
        if (std.posix.errno(rc) != .SUCCESS or rc <= 0) return;
        off += @intCast(rc);
    }
}

/// One ``[tensorfold] ...`` line.
pub fn line(comptime fmt: []const u8, args: anytype) void {
    if (quiet or @import("builtin").is_test) return; // a test's stdout is the build runner's channel
    var buf: [8192]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    w.print("[tensorfold] " ++ fmt ++ "\n", args) catch {
        buf[buf.len - 1] = '\n';
        w.end = buf.len;
    };
    emit(w.buffered(), log_time.enabled());
}

/// Text as given, the live line cleared first.
pub fn raw(text: []const u8) void {
    emit(text, false);
}

fn emit(text: []const u8, stamped: bool) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) {
        writeOut(clear);
        shown = false;
    }
    var at: [log_time.stamp_len]u8 = undefined;
    var buf: [log_time.stamp_len + 8192]u8 = undefined;
    var rest = text;
    while (rest.len > 0) writeOut(nextPiece(&buf, &rest, if (stamped) log_time.stamp(&at, log_time.nowNs()) else null));
    if (text.len > 0) line_start = text[text.len - 1] == '\n';
}

/// Without a prefix ``rest`` whole, else its next line after the prefix (in ``buf``); ``rest`` keeps what follows.
fn nextPiece(buf: []u8, rest: *[]const u8, prefix: ?[]const u8) []const u8 {
    const p = prefix orelse {
        const all = rest.*;
        rest.* = all[all.len..];
        return all;
    };
    const nl = if (std.mem.indexOfScalar(u8, rest.*, '\n')) |i| i + 1 else rest.*.len;
    const n = @min(nl, buf.len - p.len);
    @memcpy(buf[0..p.len], p);
    @memcpy(buf[p.len..][0..n], rest.*[0..n]);
    rest.* = rest.*[n..];
    return buf[0 .. p.len + n];
}

/// Redraws the status line unless a log line is half written.
pub fn status(text: []const u8, width: usize) void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (!line_start) return;
    writeOut(clear);
    writeOut(text[0..@min(text.len, @max(20, width) - 1)]);
    shown = true;
}

pub fn clearStatus() void {
    const m = io();
    mutex.lockUncancelable(m);
    defer mutex.unlock(m);
    if (shown) writeOut(clear);
    shown = false;
}

pub fn keySuffix(c: *const Conn) []const u8 {
    if (!c.auth_enabled) return "";
    return if (c.key_label) |label| label_suffix(label) else " key=unauthenticated";
}

threadlocal var suffix_buf: [80]u8 = undefined;

fn label_suffix(label: []const u8) []const u8 {
    return std.fmt.bufPrint(&suffix_buf, " key={s}", .{label}) catch " key=?";
}

/// The access line ``send_response`` prints: ``ip "GET / HTTP/1.1" 200 -``.
pub fn request(c: *const Conn, code: u16) void {
    line("{s} \"{s}\" {d} -{s}", .{ c.peer, c.requestline, code, keySuffix(c) });
}

/// ``refused <id>: <message>``, the message cut to 300 characters: why a request got an error reply.
pub fn refused(buf: []u8, id: []const u8, message: []const u8) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("refused {s}: {s}", .{ id, message[0 .. message.len - afterChars(message, 300).len] }) catch {};
    return w.buffered();
}

/// ``ended <id> reason=...``: a request that ends without its reply, because its client left (``why`` null) or it failed.
pub fn ended(buf: []u8, id: []const u8, why: ?[]const u8, prompt: usize, tokens: usize, after: f64) []const u8 {
    var w: std.Io.Writer = .fixed(buf);
    w.print("ended {s} reason=", .{id}) catch {};
    if (why) |e| w.print("error ({s})", .{e}) catch {} else w.writeAll("client-left") catch {};
    w.print(" prompt={d} tokens={d} after={d:.2}s", .{ prompt, tokens, after }) catch {};
    return w.buffered();
}

test "a line's pieces, stamped and not" {
    var buf: [64]u8 = undefined;
    var at: [log_time.stamp_len]u8 = undefined;
    const p = log_time.stamp(&at, 1_795_701_792_123_456_789);
    var rest: []const u8 = "[tensorfold] a\nb\n";
    try std.testing.expectEqualStrings("2026-11-26T14:03:12.123456Z [tensorfold] a\n", nextPiece(&buf, &rest, p));
    try std.testing.expectEqualStrings("2026-11-26T14:03:12.123456Z b\n", nextPiece(&buf, &rest, p));
    try std.testing.expectEqual(0, rest.len);
    rest = "[tensorfold] a\nb\n";
    try std.testing.expectEqualStrings("[tensorfold] a\nb\n", nextPiece(&buf, &rest, null));
    try std.testing.expectEqual(0, rest.len);
}

test "a request's refused and ended lines" {
    var buf: [1400]u8 = undefined;
    const accents: [301][2]u8 = @splat(.{ 0xc3, 0xa9 }); // two bytes a character, so a byte cut would split one
    const cut = refused(&buf, "cmpl-1", std.mem.sliceAsBytes(&accents));
    try std.testing.expectEqualStrings("refused cmpl-1: ", cut[0..16]);
    try std.testing.expectEqualStrings(std.mem.sliceAsBytes(accents[0..300]), cut[16..]);
    try std.testing.expectEqualStrings("refused cmpl-1: short", refused(&buf, "cmpl-1", "short"));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=client-left prompt=12 tokens=3 after=0.50s", ended(&buf, "chatcmpl-1", null, 12, 3, 0.5));
    try std.testing.expectEqualStrings("ended chatcmpl-1 reason=error (OutOfMemory) prompt=12 tokens=0 after=1.25s", ended(&buf, "chatcmpl-1", "OutOfMemory", 12, 0, 1.25));
}

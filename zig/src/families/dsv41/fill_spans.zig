//! fill_begin's pictures: rank 0 writes each span and digest; rank 1 checks them against its sequence and keys alike.
const std = @import("std");
const lanes_mod = @import("lanes.zig");
const image_key = @import("lanes").image_key;
const KeySpan = image_key.KeySpan;
const Wire = lanes_mod.Wire;
const Reader = lanes_mod.Reader;

/// Rank 0: the count, then each span's at, length and digest (in prompt order, apart, over image positions only).
pub fn write(w: *Wire, gpa: std.mem.Allocator, spans: []const KeySpan) !void {
    try w.int(gpa, u32, @intCast(spans.len));
    for (spans) |sp| {
        try w.int(gpa, u64, sp.at);
        try w.int(gpa, u32, sp.len);
        try w.key(gpa, sp.digest);
    }
}

/// Rank 1: the spans into `out`, each after the last, inside `seq`, over its image positions (-1) only; else BadFrame.
pub fn read(r: *Reader, gpa: std.mem.Allocator, seq: []const i32, out: *std.ArrayList(KeySpan)) !void {
    out.clearRetainingCapacity();
    const n = try r.int(u32);
    var end: u64 = 0;
    for (0..n) |_| {
        const at = try r.int(u64);
        const len = try r.int(u32);
        const digest = try r.key();
        if (at < end or at > seq.len or len == 0 or len > seq.len - at) return error.BadFrame;
        for (seq[@intCast(at)..][0..len]) |x| if (x != -1) return error.BadFrame;
        end = at + len;
        try out.append(gpa, .{ .at = @intCast(at), .len = len, .digest = digest });
    }
}

/// The key ids rank 0 keys the prompt by: `seq`'s ids, each span's positions its picture's key ids.
pub fn keyIds(gpa: std.mem.Allocator, seq: []const i32, spans: []const KeySpan, out: *std.ArrayList(u32)) !void {
    try out.resize(gpa, seq.len);
    for (out.items, seq) |*k, x| k.* = @bitCast(x);
    for (spans) |sp| for (out.items[sp.at..][0..sp.len], 0..) |*k, j| {
        k.* = image_key.imageKeyId(&sp.digest, @intCast(j));
    };
}

/// A frame of `spans` as rank 0 writes them, read back as rank 1 reads them against `seq`.
fn roundTrip(seq: []const i32, spans: []const KeySpan, at_override: ?u64) !void {
    const gpa = std.testing.allocator;
    var w: Wire = .{};
    defer w.deinit(gpa);
    try write(&w, gpa, spans);
    if (at_override) |at| std.mem.writeInt(u64, w.buf.items[4..12], at, .little);
    var r: Reader = .{ .b = w.buf.items };
    var out: std.ArrayList(KeySpan) = .empty;
    defer out.deinit(gpa);
    try read(&r, gpa, seq, &out);
    try r.end();
    try std.testing.expectEqualSlices(KeySpan, spans, out.items);
}

test "rank 1 takes only spans in order, apart, inside its sequence and over its image positions" {
    const d = image_key.imageDigest("picture");
    const seq = [_]i32{ 5, -1, -1, 6, -1, -1, -1, 7 };
    try roundTrip(&seq, &.{ .{ .at = 1, .len = 2, .digest = d }, .{ .at = 4, .len = 3, .digest = d } }, null);
    try roundTrip(&seq, &.{}, null);
    const bad = error.BadFrame;
    try std.testing.expectError(bad, roundTrip(&seq, &.{ .{ .at = 1, .len = 2, .digest = d }, .{ .at = 2, .len = 1, .digest = d } }, null)); // overlap
    try std.testing.expectError(bad, roundTrip(&seq, &.{ .{ .at = 4, .len = 3, .digest = d }, .{ .at = 1, .len = 2, .digest = d } }, null)); // order
    try std.testing.expectError(bad, roundTrip(&seq, &.{.{ .at = 0, .len = 1, .digest = d }}, std.math.maxInt(u64))); // overflow
    try std.testing.expectError(bad, roundTrip(&seq, &.{.{ .at = 0, .len = 1, .digest = d }}, null)); // over a text id
    try std.testing.expectError(bad, roundTrip(&[_]i32{ 1, 2 }, &.{.{ .at = 0, .len = 1, .digest = d }}, null));
    try std.testing.expectError(bad, roundTrip(&seq, &.{.{ .at = 6, .len = 3, .digest = d }}, null)); // past the end
    try std.testing.expectError(bad, roundTrip(&seq, &.{.{ .at = 8, .len = 1, .digest = d }}, null));
    try std.testing.expectError(bad, roundTrip(&seq, &.{.{ .at = 1, .len = 0, .digest = d }}, null)); // empty
}

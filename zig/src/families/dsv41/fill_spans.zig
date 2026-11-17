//! fill_begin's pictures: rank 0 writes each span and digest; rank 1 checks them against its sequence and keys alike.
const std = @import("std");
const lanes_mod = @import("lanes.zig");
const image_key = @import("lanes").image_key;
const KeySpan = image_key.KeySpan;
const Wire = lanes_mod.Wire;
const Reader = lanes_mod.Reader;
const prompt = @import("prompt.zig");
const Span = prompt.Span;
const Image = @import("lanes").stream.Image;

/// Whether a pass resuming at `start` still reads rows of the span [at, at + tokens): it ends past the resume.
pub fn pastResume(at: usize, tokens: usize, start: usize) bool {
    return at + tokens > start;
}

/// The spans a pass resuming at `start` takes rows for (each whole, one it lands inside too), packed in order.
pub fn rowSpans(gpa: std.mem.Allocator, images: []const Image, start: usize, out: *std.ArrayList(Span)) !void {
    out.clearRetainingCapacity();
    var row: usize = 0;
    for (images) |img| if (pastResume(img.at, img.tokens, start)) {
        try out.append(gpa, .{ .at = img.at, .len = img.tokens, .row = row });
        row += img.tokens;
    };
}

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

/// The image row each position of [start, start + n) takes: (image index, offset in its span), null for text.
fn rowsRead(gpa: std.mem.Allocator, images: []const Image, resume_at: usize, start: usize, n: usize) ![]?[2]usize {
    var spans: std.ArrayList(Span) = .empty;
    defer spans.deinit(gpa);
    try rowSpans(gpa, images, resume_at, &spans);
    // the rows buffer as the pass fills it: the kept spans' rows in order, each row named by its picture and offset
    var buf: std.ArrayList([2]usize) = .empty;
    defer buf.deinit(gpa);
    for (images, 0..) |img, k| if (pastResume(img.at, img.tokens, resume_at)) {
        for (0..img.tokens) |j| try buf.append(gpa, .{ k, j });
    };
    const out = try gpa.alloc(?[2]usize, n);
    @memset(out, null);
    for (spans.items) |sp| {
        const o = prompt.overlap(sp, start, n) orelse continue;
        for (o[0]..o[1]) |r| out[r] = buf.items[sp.rowOf(start + r)];
    }
    return out;
}

test "a resume inside a picture's span takes that whole picture's rows, and its first row is the span's row at the resume" {
    const gpa = std.testing.allocator;
    // a picture a resume at 4096 covers, one it lands inside ([4000, 4300)), one past it
    const images = [_]Image{ .{ .at = 100, .tokens = 200, .bytes = "" }, .{ .at = 4000, .tokens = 300, .bytes = "" }, .{ .at = 5000, .tokens = 50, .bytes = "" } };
    try std.testing.expect(!pastResume(100, 200, 4096));
    try std.testing.expect(pastResume(4000, 300, 4096)); // begun before the resume, ending past it: still read
    try std.testing.expect(!pastResume(4000, 96, 4096)); // ending at the resume: covered
    try std.testing.expect(pastResume(4000, 97, 4096));
    var spans: std.ArrayList(Span) = .empty;
    defer spans.deinit(gpa);
    try rowSpans(gpa, &images, 4096, &spans);
    // what fill_images sends rank 1: the two spans at their positions; rank 1 packs their rows in this order too
    try std.testing.expectEqualSlices(Span, &.{ .{ .at = 4000, .len = 300, .row = 0 }, .{ .at = 5000, .len = 50, .row = 300 } }, spans.items);
    // the first chunk after the resume starts 96 rows into the picture
    const o = prompt.overlap(spans.items[0], 4096, 2048).?;
    try std.testing.expectEqualSlices(usize, &.{ 0, 204 }, &o);
    try std.testing.expectEqual(@as(usize, 96), spans.items[0].rowOf(4096 + o[0]));
    try std.testing.expectEqual(@as(usize, 300), spans.items[1].rowOf(5000));
    // every position from the resume on reads the row a pass from 0 reads there
    const cold = try rowsRead(gpa, &images, 0, 4096, 2048);
    defer gpa.free(cold);
    const resumed = try rowsRead(gpa, &images, 4096, 4096, 2048);
    defer gpa.free(resumed);
    try std.testing.expectEqualSlices(?[2]usize, cold, resumed);
    try std.testing.expectEqualSlices(usize, &.{ 1, 96 }, &resumed[0].?);
    try std.testing.expectEqualSlices(usize, &.{ 1, 299 }, &resumed[203].?);
    try std.testing.expect(resumed[204] == null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 0 }, &resumed[5000 - 4096].?);
    // from 0 the same span sits after the first picture's rows
    try rowSpans(gpa, &images, 0, &spans);
    try std.testing.expectEqual(@as(usize, 200 + 96), spans.items[1].rowOf(4096));
    // a resume past every picture takes none
    try rowSpans(gpa, &images, 6144, &spans);
    try std.testing.expectEqual(@as(usize, 0), spans.items.len);
}

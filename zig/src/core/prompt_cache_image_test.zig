//! The prompt cache's image keys: a state with pictures resumes only on their whole digests (prompt_cache_test's fake).
const std = @import("std");
const pc = @import("prompt_cache.zig");
const t = @import("prompt_cache_test.zig");
const Fake = t.Fake;
const fresh = t.fresh;
const Store = pc.Store;

/// Two pictures whose digests agree in their first 8 bytes (a key of 64 bits would hold them alike).
fn twoPictures() [2]pc.Digest {
    const da = pc.imageDigest("picture A");
    var db = da;
    db[20] ^= 0x40;
    return .{ da, db };
}

/// The fake family at `at` of `prompt`, as a pass from 0 would leave it (so a keep there saves).
fn standAt(f: *Fake, prompt: []const u32, at: u32) void {
    f.at = at;
    f.sum = fresh(prompt[0..at]);
}

test "image key ids: a picture's span keys by its whole digest, so another picture of the same size does not resume past it" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const img: u32 = 99; // the image token: every position of a span holds it
    const ids = [_]u32{ 1, 2, img, img, img, 3, 4, 5, 9 };
    const da, const db = twoPictures();
    try std.testing.expectEqualSlices(u8, da[0..8], db[0..8]);
    const sa = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = da }};
    const sb = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = db }};
    const ka = try pc.keyIds(a, &ids, &sa);
    const kb = try pc.keyIds(a, &ids, &sb);
    try std.testing.expectEqualSlices(u32, ids[0..2], ka[0..2]); // text keeps its ids
    try std.testing.expectEqualSlices(u32, ids[5..], ka[5..]);
    for (ka[2..5], kb[2..5], 0..) |x, y, j| {
        try std.testing.expect(x >= 0x8000_0000 and y >= 0x8000_0000 and x != y);
        try std.testing.expectEqual(pc.imageKeyId(&da, @intCast(j)), x);
    }
    try std.testing.expect(ka[2] != ka[3]); // each offset its own id
    try std.testing.expectError(error.BadImageSpan, pc.keyIds(a, &ids, &.{.{ .at = 7, .len = 3, .digest = da }}));
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();
    standAt(&f, ka, 7);
    try std.testing.expect(s.keepSealed(ka, &sa, 7, null, &.{})); // past the picture
    const next = ids ++ [_]u32{ 6, 6 };
    const next_a = try pc.keyIds(a, &next, &sa);
    try std.testing.expectEqual(@as(u32, 7), s.findSealed(next_a, &sa, &.{}).?.at); // the same picture: resumes past it
    const next_b = try pc.keyIds(a, &next, &sb);
    try std.testing.expect(s.findSealed(next_b, &sb, &.{}) == null); // another picture: nothing past it
    const l = try s.lookup(a, next_b, &sb, 9, &.{}, &.{});
    try std.testing.expect(l.entry == null);
}

test "a state with a picture resumes only on that picture's whole digest, even where every span key id agrees" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const da, const db = twoPictures();
    const sa = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = da }};
    const sb = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = db }};
    const key = try pc.keyIds(a, &.{ 1, 2, 0, 0, 0, 3, 4, 5, 9, 6 }, &sa); // the key ids of A, as if B's collided with them
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();
    standAt(&f, key, 2);
    try std.testing.expect(s.keepSealed(key, &sa, 2, null, &.{})); // before the picture: no seal
    standAt(&f, key, 7);
    try std.testing.expect(s.keepSealed(key, &sa, 7, null, &.{}));
    try std.testing.expect(s.keepSealed(key, &sb, 7, null, &.{})); // the same ids, another picture: its own state
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expectEqual(s.entries.items[1], s.findSealed(key, &sa, &.{}).?);
    try std.testing.expectEqual(s.entries.items[2], s.findSealed(key, &sb, &.{}).?);
    try std.testing.expectEqual(@as(u32, 2), s.find(key, &.{}).?.at); // no pictures given: only the unsealed prefix
    const c = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = pc.imageDigest("picture C") }};
    try std.testing.expectEqual(@as(u32, 2), s.findSealed(key, &c, &.{}).?.at);
}

test "a shared cut past a picture is that picture's: the same text with another picture neither resumes it nor shares its mark" {
    const gpa = std.testing.allocator;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const img: u32 = 99;
    const ids = [_]u32{ 1, 2, img, img, img, 3, 4, 5, 6, 7, 8, 9 }; // a shared cut at 7, past the span [2, 5)
    const da, const db = twoPictures();
    const sa = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = da }};
    const sb = [_]pc.KeySpan{.{ .at = 2, .len = 3, .digest = db }};
    const ka = try pc.keyIds(a, &ids, &sa);
    const kb = try pc.keyIds(a, &ids, &sb);
    var f: Fake = .{ .gpa = gpa };
    var s = Store.init(gpa, f.snapshots(), .{ .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer s.deinit();
    const la = try s.lookup(a, ka, &sa, 11, &.{7}, &.{});
    try std.testing.expect(std.mem.indexOfScalar(u32, la.marks, 7) != null);
    standAt(&f, ka, 7);
    try std.testing.expect(s.keepSealed(ka, &sa, 7, null, &.{}));
    try std.testing.expect(s.entries.items[0].shared);
    // the other picture, the same text and the same cut: nothing of A's resumes, and its own state there is its own
    const lb = try s.lookup(a, kb, &sb, 11, &.{}, &.{});
    try std.testing.expect(lb.entry == null);
    standAt(&f, kb, 7);
    try std.testing.expect(s.keepSealed(kb, &sb, 7, null, &.{}));
    try std.testing.expectEqual(@as(usize, 2), s.entries.items.len);
    try std.testing.expect(!s.entries.items[1].shared); // A's shared mark is not B's
    // even with A's key ids (as if B's collided with them), B's digest keeps its state apart and unshared
    try std.testing.expect((try s.lookup(a, ka, &sb, 11, &.{}, &.{})).entry == null);
    try std.testing.expect(s.keepSealed(ka, &sb, 7, null, &.{}));
    try std.testing.expectEqual(@as(usize, 3), s.entries.items.len);
    try std.testing.expect(!s.entries.items[2].shared);
    // a cut before the picture holds no picture: both conversations share it
    const before = try s.lookup(a, ka, &sa, 11, &.{2}, &.{});
    try std.testing.expectEqual(@as(u32, 7), before.entry.?.at);
    standAt(&f, ka, 2);
    try std.testing.expect(s.keepSealed(ka, &sa, 2, null, &.{}));
    try std.testing.expectEqual(@as(u32, 2), s.findSealed(kb[0..6], &sb, &.{}).?.at);
}

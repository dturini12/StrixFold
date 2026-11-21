//! Rank 0's tower rows by picture digest (TF_DS_IMAGE_ROWS_CACHE_MIB), LRU, under the encoder's mutex.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The picture's BLAKE3-256 (lanes.image_key.Digest: its prepared payload, grid header and patches).
pub const Digest = [32]u8;

pub const RowsCache = struct {
    cap: usize, // bytes of rows at most (0: nothing is kept)
    used: usize = 0,
    entries: std.ArrayList(Entry) = .empty, // oldest use first

    const Entry = struct { digest: Digest, tokens: u32, rows: []u8 };

    pub fn deinit(c: *RowsCache, gpa: Allocator) void {
        for (c.entries.items) |x| gpa.free(x.rows);
        c.entries.deinit(gpa);
        c.* = .{ .cap = c.cap };
    }

    fn find(c: *const RowsCache, digest: *const Digest, tokens: u32) ?usize {
        for (c.entries.items, 0..) |x, i| if (x.tokens == tokens and std.mem.eql(u8, &x.digest, digest)) return i;
        return null;
    }

    /// Whether the picture's rows are kept (no use recorded).
    pub fn has(c: *const RowsCache, digest: *const Digest, tokens: u32) bool {
        return c.find(digest, tokens) != null;
    }

    /// Appends the picture's rows to `out` (its capacity already reserved: no allocation) and marks them used last.
    pub fn copy(c: *RowsCache, digest: *const Digest, tokens: u32, out: *std.ArrayList(u8)) bool {
        const i = c.find(digest, tokens) orelse return false;
        if (out.unusedCapacitySlice().len < c.entries.items[i].rows.len) return false; // (the caller reserves exactly)
        const x = c.entries.orderedRemove(i);
        c.entries.appendAssumeCapacity(x);
        out.appendSliceAssumeCapacity(x.rows);
        return true;
    }

    /// Keeps a copy of the rows, the oldest out first for room; rows past the cap or a failed copy are not kept.
    pub fn put(c: *RowsCache, gpa: Allocator, digest: *const Digest, tokens: u32, rows: []const u8) void {
        if (rows.len > c.cap or c.find(digest, tokens) != null) return;
        while (c.used + rows.len > c.cap) c.evictOldest(gpa);
        c.entries.ensureUnusedCapacity(gpa, 1) catch return;
        const own = gpa.dupe(u8, rows) catch return;
        c.entries.appendAssumeCapacity(.{ .digest = digest.*, .tokens = tokens, .rows = own });
        c.used += rows.len;
    }

    fn evictOldest(c: *RowsCache, gpa: Allocator) void {
        const x = c.entries.orderedRemove(0);
        c.used -= x.rows.len;
        gpa.free(x.rows);
    }
};

fn digestOf(b: u8) Digest {
    return @splat(b);
}

fn copied(c: *RowsCache, b: u8, tokens: u32, out: *std.ArrayList(u8)) !bool {
    try out.ensureUnusedCapacity(std.testing.allocator, 64);
    return c.copy(&digestOf(b), tokens, out);
}

test "a rows cache copies a kept picture's rows, by digest and span, and drops the least recently used past its cap" {
    const gpa = std.testing.allocator;
    var c: RowsCache = .{ .cap = 12 };
    defer c.deinit(gpa);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    c.put(gpa, &digestOf(1), 1, "aaaa");
    c.put(gpa, &digestOf(2), 1, "bbbb");
    c.put(gpa, &digestOf(3), 1, "cccc");
    try std.testing.expectEqual(@as(usize, 12), c.used);
    try std.testing.expect(!try copied(&c, 1, 2, &out)); // same digest, another span: not this picture
    try std.testing.expect(try copied(&c, 1, 1, &out)); // 1 used last: 2 is now the oldest
    try std.testing.expectEqualStrings("aaaa", out.items);
    c.put(gpa, &digestOf(4), 1, "dddd");
    try std.testing.expect(!c.has(&digestOf(2), 1));
    try std.testing.expect(c.has(&digestOf(1), 1) and c.has(&digestOf(3), 1) and c.has(&digestOf(4), 1));
    c.put(gpa, &digestOf(5), 2, "eeeeeeee"); // two out for it: 3 then 1
    try std.testing.expect(!c.has(&digestOf(3), 1) and !c.has(&digestOf(1), 1));
    try std.testing.expectEqual(@as(usize, 12), c.used);
    c.put(gpa, &digestOf(6), 4, "ffffffffffffff"); // larger than the cap: not kept, nothing evicted
    try std.testing.expect(!c.has(&digestOf(6), 4) and c.has(&digestOf(5), 2));
    out.clearRetainingCapacity();
    try std.testing.expect(try copied(&c, 5, 2, &out));
    try std.testing.expectEqualStrings("eeeeeeee", out.items);
}

test "a rows cache with no cap keeps nothing" {
    const gpa = std.testing.allocator;
    var c: RowsCache = .{ .cap = 0 };
    defer c.deinit(gpa);
    c.put(gpa, &digestOf(1), 1, "aaaa");
    try std.testing.expectEqual(@as(usize, 0), c.entries.items.len);
}

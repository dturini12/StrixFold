//! --vision-image-memo: a picture's span tokens, pixels and payload digest by the BLAKE3-256 of its source bytes.
const std = @import("std");
const log = @import("log.zig");

pub const Digest = [32]u8;

/// One remembered preparation.
pub const Entry = struct { tokens: u32, pixels: u64, digest: Digest };

/// Only good preparations are remembered; at `cap` entries it is cleared whole before the next new one (one log line).
pub const Memo = struct {
    gpa: std.mem.Allocator,
    cap: u32,
    mutex: std.Io.Mutex = .init,
    map: std.AutoHashMapUnmanaged(Digest, Entry) = .empty,
    /// Times it was cleared for being full.
    clears: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, cap: u32) Memo {
        return .{ .gpa = gpa, .cap = cap };
    }

    pub fn deinit(m: *Memo) void {
        m.map.deinit(m.gpa);
    }

    /// The memo's key for a picture: the BLAKE3-256 of its source bytes (the image file as the request sent it).
    pub fn key(source: []const u8) Digest {
        var out: Digest = undefined;
        std.crypto.hash.Blake3.hash(source, &out, .{});
        return out;
    }

    pub fn get(m: *Memo, io: std.Io, k: Digest) ?Entry {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        return m.map.get(k);
    }

    /// Remembers `e` under `k`; full (and `k` new): everything is forgotten first. Out of memory: not remembered.
    pub fn put(m: *Memo, io: std.Io, k: Digest, e: Entry) void {
        if (m.cap == 0) return;
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        if (m.map.count() >= m.cap and !m.map.contains(k)) {
            m.map.clearRetainingCapacity();
            m.clears += 1;
            log.line("image memo: {d} pictures remembered, the most --vision-image-memo allows; cleared (time {d})", .{ m.cap, m.clears });
        }
        m.map.put(m.gpa, k, e) catch {};
    }

    pub fn count(m: *Memo, io: std.Io) usize {
        m.mutex.lockUncancelable(io);
        defer m.mutex.unlock(io);
        return m.map.count();
    }
};

test "a memo remembers by source, keeps the newest value, and clears whole when full" {
    const io = std.testing.io;
    var m: Memo = .init(std.testing.allocator, 2);
    defer m.deinit();
    const a = Memo.key("a");
    const b = Memo.key("b");
    const c = Memo.key("c");
    try std.testing.expect(m.get(io, a) == null);
    m.put(io, a, .{ .tokens = 1, .pixels = 10, .digest = Memo.key("pa") });
    m.put(io, b, .{ .tokens = 2, .pixels = 20, .digest = Memo.key("pb") });
    try std.testing.expectEqual(@as(u32, 1), m.get(io, a).?.tokens);
    // full, but `a` is already there: no clear
    m.put(io, a, .{ .tokens = 3, .pixels = 30, .digest = Memo.key("pa") });
    try std.testing.expectEqual(@as(u64, 0), m.clears);
    try std.testing.expectEqual(@as(u32, 3), m.get(io, a).?.tokens);
    // full and `c` new: cleared, then `c` alone
    m.put(io, c, .{ .tokens = 4, .pixels = 40, .digest = Memo.key("pc") });
    try std.testing.expectEqual(@as(u64, 1), m.clears);
    try std.testing.expectEqual(@as(usize, 1), m.count(io));
    try std.testing.expect(m.get(io, a) == null and m.get(io, b) == null);
    try std.testing.expectEqualSlices(u8, &Memo.key("pc"), &m.get(io, c).?.digest);
}

test "a memo of 0 pictures remembers nothing" {
    const io = std.testing.io;
    var m: Memo = .init(std.testing.allocator, 0);
    defer m.deinit();
    m.put(io, Memo.key("a"), .{ .tokens = 1, .pixels = 1, .digest = Memo.key("p") });
    try std.testing.expectEqual(@as(usize, 0), m.count(io));
}

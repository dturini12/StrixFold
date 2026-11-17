//! A picture in the prompt caches' keys: its digest, its span's key ids, and the seal a kept state holds them by.
const std = @import("std");
const stream = @import("stream.zig");
const Allocator = std.mem.Allocator;
const Blake3 = std.crypto.hash.Blake3;

/// A picture's identity: the BLAKE3-256 digest of its prepared payload (or of its ready rows).
pub const Digest = [32]u8;

pub fn imageDigest(bytes: []const u8) Digest {
    var out: Digest = undefined;
    Blake3.hash(bytes, &out, .{});
    return out;
}

/// An image span as the caches see it: positions [at, at + len) made of the picture `digest`.
pub const KeySpan = struct { at: u32, len: u32, digest: Digest };

/// The key id at offset `j` of a picture's span: bit 31 set (no vocabulary id), 31 bits of BLAKE3 keyed by the digest.
pub fn imageKeyId(digest: *const Digest, j: u32) u32 {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, j, .little);
    var out: [4]u8 = undefined;
    Blake3.hash(&b, &out, .{ .key = digest.* });
    return 0x8000_0000 | std.mem.readInt(u32, &out, .little);
}

/// The ids kept states are found by (a new slice in `a`): `prompt`'s, each span's positions its picture's key ids.
pub fn keyIds(a: Allocator, prompt: []const u32, spans: []const KeySpan) ![]u32 {
    const out = try a.dupe(u32, prompt);
    errdefer a.free(out);
    for (spans) |sp| {
        if (@as(usize, sp.at) + sp.len > out.len) return error.BadImageSpan;
        for (out[sp.at..][0..sp.len], 0..) |*q, j| q.* = imageKeyId(&sp.digest, @intCast(j));
    }
    return out;
}

/// Each image's span and whole digest (a new slice in `a`; taken here when the server did not take it).
pub fn spansOf(a: Allocator, images: []const stream.Image) ![]KeySpan {
    const out = try a.alloc(KeySpan, images.len);
    for (images, out) |img, *sp| sp.* = .{ .at = img.at, .len = img.tokens, .digest = img.digest orelse imageDigest(img.bytes) };
    return out;
}

/// The pictures a state of the first `n` key ids holds: each span begun before `n`, its whole digest (zero: none).
pub fn sealOf(spans: []const KeySpan, n: usize) Digest {
    var h = Blake3.init(.{});
    var any = false;
    for (spans) |sp| if (sp.at < n) {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u32, b[0..4], sp.at, .little);
        std.mem.writeInt(u32, b[4..8], sp.len, .little);
        h.update(&b);
        h.update(&sp.digest);
        any = true;
    };
    var out: Digest = @splat(0);
    if (any) h.final(&out);
    return out;
}

test "a digest is the whole BLAKE3-256; two pictures alike in their first 8 bytes of digest still key apart" {
    const a = imageDigest("picture A");
    try std.testing.expect(!std.mem.allEqual(u8, a[8..], 0));
    var b = a;
    b[31] ^= 1; // the same first 8 bytes: a key of 64 bits would hold them alike
    var differ: usize = 0;
    for (0..64) |j| differ += @intFromBool(imageKeyId(&a, @intCast(j)) != imageKeyId(&b, @intCast(j)));
    try std.testing.expectEqual(@as(usize, 64), differ);
    const sa = [_]KeySpan{.{ .at = 2, .len = 3, .digest = a }};
    const sb = [_]KeySpan{.{ .at = 2, .len = 3, .digest = b }};
    try std.testing.expect(!std.mem.eql(u8, &sealOf(&sa, 3), &sealOf(&sb, 3)));
    try std.testing.expectEqualSlices(u8, &sealOf(&sa, 2), &sealOf(&sb, 2)); // no span begun before 2: both unsealed
    try std.testing.expect(std.mem.allEqual(u8, &sealOf(&sa, 2), 0));
}

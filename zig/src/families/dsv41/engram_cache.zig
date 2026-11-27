//! The prompt chunks' Engram rows kept dense (row_w + row_s bytes a row, not a 4 KiB page): 8-way sets, random ways.
const std = @import("std");
const engram_io = @import("engram_io.zig");

const ways = 8;

pub const RowCache = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    sets: usize,
    slot: usize, // bytes a row: the widest table's row_w + row_s
    keys: []u64, // [sets * ways]; 0 = empty
    data: []u8, // [sets * ways * slot]
    rng: u64 = 0x9e3779b97f4a7c15, // a full set's way to replace: random, so a text longer than the cache keeps a share
    // a gather's misses: their ids and places, read into miss_w / miss_s
    miss_ids: []i64,
    miss_at: []u32,
    miss_w: []u8,
    miss_s: []u8,
    row_w: usize,
    row_s: usize,
    hits: u64 = 0,
    misses: u64 = 0,

    /// About `bytes` of rows (at least one set) for tables of rows row_w + row_s wide, gathers of up to `cap` rows.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, bytes: usize, row_w: usize, row_s: usize, cap: usize) !*RowCache {
        const slot = row_w + row_s;
        const sets = @max(1, bytes / (slot + 8) / ways);
        const c = try gpa.create(RowCache);
        errdefer gpa.destroy(c);
        c.* = .{ .gpa = gpa, .io = io, .sets = sets, .slot = slot, .row_w = row_w, .row_s = row_s, .keys = &.{}, .data = &.{}, .miss_ids = &.{}, .miss_at = &.{}, .miss_w = &.{}, .miss_s = &.{} };
        errdefer c.free();
        c.keys = try gpa.alloc(u64, sets * ways);
        @memset(c.keys, 0);
        c.data = try gpa.alloc(u8, sets * ways * slot);
        c.miss_ids = try gpa.alloc(i64, cap);
        c.miss_at = try gpa.alloc(u32, cap);
        c.miss_w = try gpa.alloc(u8, cap * row_w);
        c.miss_s = try gpa.alloc(u8, cap * row_s);
        return c;
    }

    fn free(c: *RowCache) void {
        c.gpa.free(c.keys);
        c.gpa.free(c.data);
        c.gpa.free(c.miss_ids);
        c.gpa.free(c.miss_at);
        c.gpa.free(c.miss_w);
        c.gpa.free(c.miss_s);
    }

    pub fn deinit(c: *RowCache) void {
        c.free();
        c.gpa.destroy(c);
    }

    /// The bytes this cache holds (its rows and keys).
    pub fn held(c: *const RowCache) usize {
        return c.data.len + c.keys.len * 8;
    }

    fn key(layer: usize, id: u64) u64 {
        return (@as(u64, @intCast(layer)) << 48) | (id + 1);
    }

    fn setOf(c: *const RowCache, k: u64) usize {
        return @intCast(std.hash.int(k) % c.sets);
    }

    fn find(c: *const RowCache, k: u64) ?usize {
        const s0 = c.setOf(k) * ways;
        for (s0..s0 + ways) |i| if (c.keys[i] == k) return i;
        return null;
    }

    fn put(c: *RowCache, k: u64, w: []const u8, s: []const u8) void {
        const i = c.find(k) orelse blk: {
            const s0 = c.setOf(k) * ways;
            for (s0..s0 + ways) |j| if (c.keys[j] == 0) break :blk j;
            c.rng ^= c.rng << 13;
            c.rng ^= c.rng >> 7;
            c.rng ^= c.rng << 17;
            break :blk s0 + @as(usize, @intCast(c.rng % ways));
        };
        c.keys[i] = k;
        const d = c.data[i * c.slot ..][0..c.slot];
        @memcpy(d[0..w.len], w);
        @memcpy(d[c.row_w..][0..s.len], s);
    }

    /// Pool.gather of layer `layer`'s rows `ids`: cached rows copied from here, the misses read and kept.
    pub fn gather(c: *RowCache, pool: *engram_io.Pool, layer: usize, t: engram_io.Table, ids: []const i64, out_w: []u8, out_s: []u8) !void {
        if (t.row_w > c.row_w or t.row_s > c.row_s or ids.len > c.miss_ids.len or layer >= 1 << 15) return pool.gather(t, ids, out_w, out_s);
        if (out_w.len != ids.len * t.row_w or out_s.len != ids.len * t.row_s) return error.BadSize;
        c.mutex.lockUncancelable(c.io);
        defer c.mutex.unlock(c.io);
        var nm: usize = 0;
        for (ids, 0..) |id, i| {
            const hit = if (id >= 0 and @as(u64, @intCast(id)) < t.rows) c.find(key(layer, @intCast(id))) else null;
            if (hit) |h| {
                const d = c.data[h * c.slot ..][0..c.slot];
                @memcpy(out_w[i * t.row_w ..][0..t.row_w], d[0..t.row_w]);
                @memcpy(out_s[i * t.row_s ..][0..t.row_s], d[c.row_w..][0..t.row_s]);
                c.hits += 1;
            } else {
                c.miss_ids[nm] = id;
                c.miss_at[nm] = @intCast(i);
                nm += 1;
            }
        }
        c.misses += nm;
        if (nm == 0) return;
        try pool.gather(t, c.miss_ids[0..nm], c.miss_w[0 .. nm * t.row_w], c.miss_s[0 .. nm * t.row_s]);
        for (0..nm) |j| {
            const i: usize = c.miss_at[j];
            const w = c.miss_w[j * t.row_w ..][0..t.row_w];
            const s = c.miss_s[j * t.row_s ..][0..t.row_s];
            @memcpy(out_w[i * t.row_w ..][0..t.row_w], w);
            @memcpy(out_s[i * t.row_s ..][0..t.row_s], s);
            c.put(key(layer, @intCast(c.miss_ids[j])), w, s);
        }
    }
};

test "cached rows equal the table's reads, through misses, hits and evictions" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    // a [64, 4] weight table and its [64, 1] scales for layer 14, every byte distinct within its file
    const json = "{\"layers.14.engram.embed.weight\":{\"dtype\":\"U8\",\"shape\":[64,4],\"data_offsets\":[0,256]}," ++
        "\"layers.14.engram.embed.scale\":{\"dtype\":\"U8\",\"shape\":[64,1],\"data_offsets\":[256,320]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..320) |i| try file.append(a, @truncate(i * 7 + 3));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    const tab = tables.layers.get(14).?;
    const pool = try engram_io.Pool.init(a, io, 3);
    defer pool.deinit(a);
    // 2 sets of 8 ways: 16 rows of the 64, so a long gather evicts
    const c = try RowCache.init(a, io, 2 * ways * (4 + 1 + 8), 4, 1, 32);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 2), c.sets);
    var rng = std.Random.DefaultPrng.init(7);
    const r = rng.random();
    var ids: [32]i64 = undefined;
    var w0: [32 * 4]u8 = undefined;
    var s0: [32]u8 = undefined;
    var w1: [32 * 4]u8 = undefined;
    var s1: [32]u8 = undefined;
    for (0..40) |round| {
        // early rounds draw from 12 rows (they fit), later ones from all 64 (evictions); repeats within a gather too
        const span: u64 = if (round < 10) 12 else 64;
        for (&ids) |*x| x.* = @intCast(r.uintLessThan(u64, span));
        try pool.gather(tab, &ids, &w0, &s0);
        @memset(&w1, 0xee);
        @memset(&s1, 0xee);
        try c.gather(pool, 1, tab, &ids, &w1, &s1);
        try std.testing.expectEqualSlices(u8, &w0, &w1);
        try std.testing.expectEqualSlices(u8, &s0, &s1);
    }
    try std.testing.expect(c.hits > 0);
    try std.testing.expect(c.misses > 0);
    // the same ids of another layer are other rows: a miss, never layer 1's bytes
    const h = c.hits;
    ids[0] = 5;
    try c.gather(pool, 1, tab, ids[0..1], w1[0..4], s1[0..1]);
    try c.gather(pool, 0, tab, ids[0..1], w1[0..4], s1[0..1]);
    try std.testing.expectEqual(h + 1, c.hits);
    // a row id past the table fails as the pool's read does
    ids[0] = 64;
    try std.testing.expectError(error.ReadFailed, c.gather(pool, 1, tab, ids[0..1], w1[0..4], s1[0..1]));
}

test "a text cycled through more rows than the cache holds still hits (random ways, not first in first out)" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    const json = "{\"layers.14.engram.embed.weight\":{\"dtype\":\"U8\",\"shape\":[64,4],\"data_offsets\":[0,256]}," ++
        "\"layers.14.engram.embed.scale\":{\"dtype\":\"U8\",\"shape\":[64,1],\"data_offsets\":[256,320]}}";
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(a);
    var len: [8]u8 = undefined;
    std.mem.writeInt(u64, &len, json.len, .little);
    try file.appendSlice(a, &len);
    try file.appendSlice(a, json);
    for (0..320) |i| try file.append(a, @truncate(i * 5 + 1));
    try tmp.dir.writeFile(io, .{ .sub_path = "engram.safetensors", .data = file.items });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", &tmp.sub_path });
    defer a.free(dir);
    var tables = try engram_io.Tables.open(a, io, dir);
    defer tables.close();
    const tab = tables.layers.get(14).?;
    const pool = try engram_io.Pool.init(a, io, 2);
    defer pool.deinit(a);
    // one set of 8 ways; 16 rows read in order, 2 a gather, 10 times: first in first out would never hit
    const c = try RowCache.init(a, io, ways * (4 + 1 + 8), 4, 1, 2);
    defer c.deinit();
    try std.testing.expectEqual(@as(usize, 1), c.sets);
    var w: [8]u8 = undefined;
    var sc: [2]u8 = undefined;
    var w0: [8]u8 = undefined;
    var s0: [2]u8 = undefined;
    for (0..10) |_| {
        var id: i64 = 0;
        while (id < 16) : (id += 2) {
            const ids = [_]i64{ id, id + 1 };
            try c.gather(pool, 1, tab, &ids, &w, &sc);
            try pool.gather(tab, &ids, &w0, &s0);
            try std.testing.expectEqualSlices(u8, &w0, &w);
            try std.testing.expectEqualSlices(u8, &s0, &sc);
        }
    }
    try std.testing.expect(c.hits >= 16); // a share of the 144 reads after the first pass
}

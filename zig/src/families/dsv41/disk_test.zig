//! sf-disk for dsv41 on the host: settings, block rows, the identity, a pair's save and resume, a failed submit.
const std = @import("std");
const core = @import("core");
const disk = @import("disk.zig");
const pd = core.prompt_disk;
const Key = pd.Key;
const Settings = disk.Settings;
const Plane = disk.Plane;
const Parts = disk.Parts;
const block_tokens = disk.block_tokens;
const blockBytes = disk.blockBytes;
const rowsAt = disk.rowsAt;
const identityText = disk.identityText;
const seedText = disk.seedText;
const submitRuns = disk.submitRuns;
const image_key = @import("lanes").image_key;
const fill_spans = @import("fill_spans.zig");
const lanes_mod = @import("lanes.zig");

test "settings: off unless a positive GiB; defaults; a value that is not a number turns the tier off" {
    const a = std.testing.allocator;
    const Env = struct {
        var vars: []const [2][]const u8 = &.{};
        fn get(name: [:0]const u8) ?[]const u8 {
            for (vars) |v| if (std.mem.eql(u8, v[0], name)) return v[1];
            return null;
        }
    };
    Env.vars = &.{.{ "HOME", "/h" }};
    try std.testing.expectEqual(@as(?Settings, null), try Settings.parse(a, Env.get));
    Env.vars = &.{ .{ "HOME", "/h" }, .{ "TF_DS_PROMPT_DISK_GIB", "0" } };
    try std.testing.expectEqual(@as(?Settings, null), try Settings.parse(a, Env.get));
    Env.vars = &.{ .{ "HOME", "/h" }, .{ "TF_DS_PROMPT_DISK_GIB", "256" } };
    const s = (try Settings.parse(a, Env.get)).?;
    defer a.free(s.dir);
    try std.testing.expectEqualStrings("/h/strixfold-cache/prefix", s.dir);
    try std.testing.expectEqual(@as(u64, 256 << 30), s.budget);
    try std.testing.expectEqual(@as(u64, 16 << 30), s.floor);
    try std.testing.expectEqual(@as(u64, 1 << 30), s.queue);
    try std.testing.expectEqual(@as(u32, 8192), s.min_tokens);
    Env.vars = &.{ .{ "TF_DS_PROMPT_DISK_GIB", "2.5" }, .{ "TF_DS_PROMPT_DISK_DIR", "/x" }, .{ "TF_DS_PROMPT_DISK_QUEUE_MIB", "64" }, .{ "TF_DS_PROMPT_DISK_FLOOR_GIB", "1" }, .{ "TF_DS_PROMPT_DISK_MIN_TOKENS", "100" } };
    const t = (try Settings.parse(a, Env.get)).?;
    defer a.free(t.dir);
    try std.testing.expectEqualStrings("/x", t.dir);
    try std.testing.expectEqual(@as(u64, 5 << 29), t.budget);
    try std.testing.expectEqual(@as(u64, 64 << 20), t.queue);
    try std.testing.expectEqual(@as(u64, 1 << 30), t.floor);
    try std.testing.expectEqual(@as(u32, block_tokens), t.min_tokens); // at least a block
    Env.vars = &.{.{ "TF_DS_PROMPT_DISK_GIB", "lots" }};
    try std.testing.expectError(error.BadSetting, Settings.parse(a, Env.get));
}

test "a block's rows: each plane's place in an extent and its bytes" {
    const p: Plane = .{ .base = 1 << 20, .row = 256, .ratio = 4 };
    try std.testing.expectEqual(@as(u64, block_tokens / 4 * 256), blockBytes(p));
    try std.testing.expectEqual(@as(u64, (1 << 20) + 4096 / 4 * 256), rowsAt(p, 4096, 1));
    try std.testing.expectEqual(@as(u64, (1 << 20) + (4096 + 2 * block_tokens) / 4 * 256), rowsAt(p, 4096, 3));
    // consecutive blocks' rows are consecutive in the plane (a run is one copy a plane)
    try std.testing.expectEqual(rowsAt(p, 0, 2), rowsAt(p, 0, 1) + blockBytes(p));
}

test "the identity: every input on its own line; the plan's conf= and disk= tokens left out" {
    const a = std.testing.allocator;
    const parts: Parts = .{ .rank = 1, .world = 2, .deploy = "70cfb8c", .bundle = "c178", .kit_sha256 = "abcd", .hip_runtime = 70051831, .arch = "gfx1151", .plan = "drafts=true pool=8192 conf=1234 kv=x disk=1", .engram = "e[t:1]", .token_map = "tm", .ds_kit = "dk", .window = 128, .snap_bytes = 6 << 20, .planes = "256:4" };
    const text = try identityText(a, parts);
    defer a.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "plan.pool=8192\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "conf") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "disk=") == null);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank=1\n") != null);
    // a changed setting is a new namespace; the same inputs the same one
    var other = parts;
    other.plan = "drafts=true pool=16384 conf=1234 kv=x disk=1";
    const t2 = try identityText(a, other);
    defer a.free(t2);
    try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(text), &pd.namespaceOf(t2)));
    var diff: [128]u8 = undefined;
    try std.testing.expectEqualStrings("plan.pool", pd.identityDiff(&diff, text, t2));
    var same = parts;
    same.plan = "drafts=true pool=8192 conf=9999 kv=x"; // serve.conf's hash and the disk flag do not change it
    const t3 = try identityText(a, same);
    defer a.free(t3);
    try std.testing.expectEqualStrings(text, t3);
}

test "a block's key holds the whole digest of each picture that meets it, so key ids alike cannot resume it" {
    var ids: [3 * block_tokens]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(i * 7 % 1000);
    const da = image_key.imageDigest("picture A");
    var db = da;
    db[20] ^= 1; // alike in the first 8 bytes; the ids below are A's for both (as if B's had collided)
    const h0 = pd.seed("id");
    var text: [3]Key = undefined;
    pd.chain(h0, std.mem.sliceAsBytes(&ids), block_tokens * 4, &text);
    var none: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{}, &none);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&text), std.mem.sliceAsBytes(&none)); // text: pd.chain's
    var ka: [3]Key = undefined;
    var kb: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = block_tokens - 2, .len = 4, .digest = da }}, &ka);
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = block_tokens - 2, .len = 4, .digest = db }}, &kb);
    for (ka, kb) |x, y| try std.testing.expect(!std.mem.eql(u8, &x, &y)); // the span meets blocks 1 and 2
    for (ka, text) |x, y| try std.testing.expect(!std.mem.eql(u8, &x, &y)); // nor is either the same ids' text key
    var late: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(&ids), &.{.{ .at = 2 * block_tokens + 5, .len = 4, .digest = da }}, &late);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(text[0..2]), std.mem.sliceAsBytes(late[0..2])); // before it: text
    try std.testing.expect(!std.mem.eql(u8, &text[2], &late[2]));
}

fn testRoot(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-sfdisk-dsv41-{s}-{d}", .{ tag, std.c.getpid() });
}

fn testRmTree(root: []const u8) void {
    var buf: [512]u8 = undefined;
    const r = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return;
    const d = std.c.opendir(r) orelse return;
    while (std.c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        var sub: [512]u8 = undefined;
        const p = std.fmt.bufPrint(&sub, "{s}/{s}", .{ root, name }) catch continue;
        var zb: [512]u8 = undefined;
        if (std.c.unlink(std.fmt.bufPrintSentinel(&zb, "{s}", .{p}, 0) catch continue) != 0) testRmTree(p);
    }
    _ = std.c.closedir(d);
    _ = std.c.rmdir(r);
}

const test_planes = [_]u64{ 64, 16 };
const test_slot = 96;

fn testParts(rank: u32) Parts {
    // the rank-local inputs may differ between the boxes (their own engram listing here): only the namespace sees them
    return .{ .rank = rank, .world = 2, .deploy = "4a5cb43", .bundle = "c178a012", .kit_sha256 = "73944f4e", .hip_runtime = 70051831, .arch = "gfx1151", .plan = "drafts=true pool=327680 conf=1 kv=x disk=1", .engram = if (rank == 0) "e[a:1]" else "e[a:1,b:2]", .token_map = "tm", .ds_kit = "dk", .window = 128, .snap_bytes = test_slot, .planes = "64:4,16:4" };
}

/// One rank's save of the state at k blocks, as diskSave and the disk_save handler make it (bytes depend on the rank).
fn testSave(t: *pd.Tier, chain_keys: []const Key, k: u32, rank: u8) !void {
    const gpa = std.testing.allocator;
    var p = try t.plan(chain_keys[0..k], true, false);
    defer p.deinit(gpa);
    try std.testing.expect(p.result == .ok);
    const Fill = struct {
        rank: u8,
        fn f(c: @This(), j0: u32, n: u32) ![]u8 {
            const data = try std.testing.allocator.alloc(u8, n * pd.sum(&test_planes));
            for (data, 0..) |*b, i| b.* = @truncate(c.rank *% 101 +% j0 *% 7 +% i);
            return data;
        }
    };
    try std.testing.expect(submitRuns(t, &p, chain_keys[0..k], 1, true, Fill{ .rank = rank }, Fill.f));
    const slot = try gpa.alloc(u8, test_slot);
    @memset(slot, rank +% 0x30);
    try t.submitState(chain_keys[0..k], slot, k);
}

test "rank 0 and rank 1 name a prompt's states alike: a pair's save, reconcile and resume all find them" {
    const gpa = std.testing.allocator;
    const p0 = testParts(0);
    const p1 = testParts(1);
    const id0 = try identityText(gpa, p0);
    defer gpa.free(id0);
    const id1 = try identityText(gpa, p1);
    defer gpa.free(id1);
    const s0 = try seedText(gpa, p0);
    defer gpa.free(s0);
    const s1 = try seedText(gpa, p1);
    defer gpa.free(s1);
    // each rank's files in a namespace of its own; one chain seed for both
    try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf(id0), &pd.namespaceOf(id1)));
    try std.testing.expectEqualStrings(s0, s1);
    try std.testing.expect(std.mem.indexOf(u8, s0, "rank") == null);
    // the same prompt on both ranks (rank 0 hashes its u32 tokens, rank 1 its i32 sequence: the same bytes)
    var toks: [6 * block_tokens]u32 = undefined;
    for (&toks, 0..) |*x, i| x.* = @intCast(i * 31 % 129280);
    var seq: [6 * block_tokens]i32 = undefined;
    for (&seq, toks) |*q, x| q.* = @intCast(x);
    var k0: [6]Key = undefined;
    var k1: [6]Key = undefined;
    pd.chain(pd.seed(s0), std.mem.sliceAsBytes(&toks), block_tokens * 4, &k0);
    pd.chain(pd.seed(s1), std.mem.sliceAsBytes(&seq), block_tokens * 4, &k1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), std.mem.sliceAsBytes(&k1));
    // (the namespace's own text as the seed would name them differently: no hit would be possible)
    var w0: [1]Key = undefined;
    var w1: [1]Key = undefined;
    pd.chain(pd.seed(id0), std.mem.sliceAsBytes(&toks), block_tokens * 4, &w0);
    pd.chain(pd.seed(id1), std.mem.sliceAsBytes(&seq), block_tokens * 4, &w1);
    try std.testing.expect(!std.mem.eql(u8, &w0[0], &w1[0]));

    var rb0: [128]u8 = undefined;
    var rb1: [128]u8 = undefined;
    const root0 = try testRoot(&rb0, "pair0");
    const root1 = try testRoot(&rb1, "pair1");
    defer testRmTree(root0);
    defer testRmTree(root1);
    const cfg: pd.Config = .{ .root = undefined, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 };
    var c0 = cfg;
    c0.root = root0;
    var c1 = cfg;
    c1.root = root1;
    c1.follower = true;
    {
        const t0 = try pd.Tier.open(gpa, c0, id0, 0, test_slot, &test_planes);
        defer t0.close();
        const t1 = try pd.Tier.open(gpa, c1, id1, 1, test_slot, &test_planes);
        defer t1.close();
        // rank 0 keeps the states at 3 and 5 blocks; rank 1 hears each key and finds it in its own chain
        for ([_]u32{ 3, 5 }) |k| {
            try testSave(t0, &k0, k, 0);
            try std.testing.expectEqualSlices(u8, &k0[k - 1], &k1[k - 1]); // lanes.zig's disk_save check
            try testSave(t1, &k1, k, 1);
        }
        try std.testing.expect(t0.flush(5 * std.time.ns_per_s) and t1.flush(5 * std.time.ns_per_s));
    }
    // a restart: both reopen and reconcile, rank 0's list against rank 1's bitmap
    const t0 = try pd.Tier.open(gpa, c0, id0, 0, test_slot, &test_planes);
    defer t0.close();
    const t1 = try pd.Tier.open(gpa, c1, id1, 1, test_slot, &test_planes);
    defer t1.close();
    const listed = try t0.completeKeys();
    defer gpa.free(listed);
    try std.testing.expectEqual(@as(usize, 2), listed.len);
    const map = try pd.bitmap(gpa, t1, listed);
    defer gpa.free(map);
    for (0..listed.len) |i| try std.testing.expect(pd.bit(map, i)); // nothing to drop on either side
    // the resume: rank 0 picks the longest state of the prompt, both ranks read their own part back
    const k = t0.longest(&k0, 0, 6).?;
    try std.testing.expectEqual(@as(u32, 5), k);
    const keys = try gpa.alloc(Key, k);
    defer gpa.free(keys);
    const slot = try gpa.alloc(u8, test_slot);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, k * pd.sum(&test_planes));
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + pd.sum(&test_planes));
    defer gpa.free(file);
    for ([_]*pd.Tier{ t0, t1 }, [_]u8{ 0, 1 }) |t, rank| {
        try std.testing.expectEqual(pd.Load.ok, t.readState(k0[k - 1], k, keys, slot));
        try std.testing.expect(t.readRun(keys, 1, data, file));
        try std.testing.expectEqual(rank +% 0x30, slot[0]);
    }
    try std.testing.expectEqual(pd.Outcome.hit, pd.Outcome.of(.ok, .ok));
}

test "a run whose submit fails queues nothing: its reservation comes back and its blocks go" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{});
    const gpa = failing.allocator();
    var rb: [128]u8 = undefined;
    const root = try testRoot(&rb, "submit");
    defer testRmTree(root);
    const t = try pd.Tier.open(gpa, .{ .root = root, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 }, "id", 0, test_slot, &test_planes);
    defer {
        failing.fail_index = std.math.maxInt(usize);
        t.close();
    }
    var toks: [3 * block_tokens]u32 = @splat(7);
    var keys: [3]Key = undefined;
    pd.chain(pd.seed("seed"), std.mem.sliceAsBytes(&toks), block_tokens * 4, &keys);
    var p = try t.plan(&keys, false, true); // a pass's streamed run of three blocks
    defer {
        failing.fail_index = std.math.maxInt(usize);
        p.deinit(gpa);
    }
    try std.testing.expect(t.queueRoom() < 1 << 20);
    const Fill = struct {
        f: *std.testing.FailingAllocator,
        fn f_(c: @This(), j0: u32, n: u32) ![]u8 {
            _ = j0;
            const data = try c.f.allocator().alloc(u8, n * pd.sum(&test_planes));
            c.f.fail_index = c.f.alloc_index; // the submit's copy of the run's keys fails
            return data;
        }
    };
    try std.testing.expect(!submitRuns(t, &p, &keys, 1, false, Fill{ .f = &failing }, Fill.f_));
    failing.fail_index = std.math.maxInt(usize);
    try std.testing.expectEqual(@as(u64, 1 << 20), t.queueRoom());
    try std.testing.expectEqual(@as(usize, 0), t.blocks.count());
    try std.testing.expectEqual(@as(u64, 0), t.bytes);
}

test "rank 1 keys an image prompt's blocks as rank 0 does, from fill_begin's spans, a span across a block end too" {
    const gpa = std.testing.allocator;
    const image: u32 = 128815; // the image token: rank 0's prompt holds it, its sequence -1 there
    var prompt: [3 * block_tokens]u32 = undefined;
    for (&prompt, 0..) |*x, i| x.* = @intCast(i * 31 % 129280 % image);
    const spans = [_]image_key.KeySpan{ .{ .at = block_tokens - 3, .len = 10, .digest = image_key.imageDigest("picture A") }, .{ .at = 2 * block_tokens + 100, .len = 64, .digest = image_key.imageDigest("picture B") } };
    var seq0: [prompt.len]i32 = undefined;
    for (&seq0, prompt) |*q, x| q.* = @intCast(x);
    for (spans) |sp| for (prompt[sp.at..][0..sp.len], seq0[sp.at..][0..sp.len]) |*x, *q| {
        x.* = image;
        q.* = -1;
    };
    // rank 0: its key from the prompt (the lane host's), its chain, and fill_begin's sequence and spans
    const key0 = try image_key.keyIds(gpa, &prompt, &spans);
    defer gpa.free(key0);
    const h0 = pd.seed("seed");
    var k0: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(key0), &spans, &k0);
    var w: lanes_mod.Wire = .{};
    defer w.deinit(gpa);
    try w.ids(gpa, &seq0);
    try fill_spans.write(&w, gpa, &spans);
    // rank 1: the frame read back, its key ids made from it, its chain
    var r: lanes_mod.Reader = .{ .b = w.buf.items };
    var seq1: std.ArrayList(i32) = .empty;
    defer seq1.deinit(gpa);
    try r.ids(gpa, &seq1);
    var spans1: std.ArrayList(image_key.KeySpan) = .empty;
    defer spans1.deinit(gpa);
    try fill_spans.read(&r, gpa, seq1.items, &spans1);
    try r.end();
    var key1: std.ArrayList(u32) = .empty;
    defer key1.deinit(gpa);
    try fill_spans.keyIds(gpa, seq1.items, spans1.items, &key1);
    try std.testing.expectEqualSlices(u32, key0, key1.items);
    var k1: [3]Key = undefined;
    disk.chainKeys(h0, std.mem.sliceAsBytes(key1.items), spans1.items, &k1);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&k0), std.mem.sliceAsBytes(&k1));
    var text: [3]Key = undefined; // every block's key holds its pictures: none is the text chain's
    pd.chain(h0, std.mem.sliceAsBytes(&prompt), block_tokens * 4, &text);
    for (k0, text) |a, b| try std.testing.expect(!std.mem.eql(u8, &a, &b));
}

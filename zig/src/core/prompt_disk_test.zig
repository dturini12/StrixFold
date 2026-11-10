//! sf-disk's store on the host: keys, file formats, the index, the writer's files, the scan at open, eviction,
//! the free-space floor and the queue bound, and the reconcile's bitmap.
const std = @import("std");
const pd = @import("prompt_disk.zig");
const Key = pd.Key;
const Tier = pd.Tier;
const c = std.c;

const planes = [_]u64{ 96, 32, 48, 8 }; // a block's bytes, plane by plane (small stand-ins)
const block_bytes = pd.sum(&planes);
const slot_bytes = 200;
const B = 16; // token bytes a block (4 tokens of 4 bytes)

fn scratch(buf: []u8, tag: []const u8) ![]const u8 {
    return std.fmt.bufPrint(buf, "/tmp/tf-sfdisk-{s}-{d}", .{ tag, c.getpid() });
}

fn rmTree(root: []const u8) void {
    var buf: [512]u8 = undefined;
    const r = std.fmt.bufPrintSentinel(&buf, "{s}", .{root}, 0) catch return;
    const d = c.opendir(r) orelse return;
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        var sub: [512]u8 = undefined;
        const p = std.fmt.bufPrint(&sub, "{s}/{s}", .{ root, name }) catch continue;
        var zb: [512]u8 = undefined;
        if (c.unlink(std.fmt.bufPrintSentinel(&zb, "{s}", .{p}, 0) catch continue) != 0) rmTree(p);
    }
    _ = c.closedir(d);
    _ = c.rmdir(r);
}

fn exists(file: []const u8) bool {
    var buf: [600]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&buf, "{s}", .{file}, 0) catch return false;
    const fd = c.open(p, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return false;
    _ = c.close(fd);
    return true;
}

fn countFiles(dir: []const u8, ext: []const u8) usize {
    var buf: [512]u8 = undefined;
    const d = c.opendir(std.fmt.bufPrintSentinel(&buf, "{s}", .{dir}, 0) catch return 0) orelse return 0;
    defer _ = c.closedir(d);
    var n: usize = 0;
    while (c.readdir(d)) |ent| {
        const name = std.mem.sliceTo(&ent.name, 0);
        if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
        if (std.mem.endsWith(u8, name, ext)) {
            n += 1;
        } else if (ext.len > 0) {
            var sub: [512]u8 = undefined;
            n += countFiles(std.fmt.bufPrint(&sub, "{s}/{s}", .{ dir, name }) catch continue, ext);
        }
    }
    return n;
}

/// A prompt of n blocks: tokens 4 a block, token i of block j = j * 100 + i (alter: one token changed in block `alter`).
fn prompt(out: []u32, alter: ?usize) []const u8 {
    for (out, 0..) |*t, i| t.* = @intCast((i / 4) * 100 + i % 4);
    if (alter) |j| out[j * 4] += 7;
    return std.mem.sliceAsBytes(out);
}

/// A run's bytes (plane-major over n blocks), each byte a function of its block j and plane.
fn runBytes(gpa: std.mem.Allocator, first_j: u32, n: usize) ![]u8 {
    const data = try gpa.alloc(u8, n * block_bytes);
    for (planes, 0..) |pb, p| for (0..n) |i| {
        const off = pd.rangeOffset(&planes, n, p) + i * pb;
        @memset(data[off..][0..pb], @intCast((first_j + i) * 16 + p));
    };
    return data;
}

fn slotBytes(gpa: std.mem.Allocator, tag: u8) ![]u8 {
    const s = try gpa.alloc(u8, slot_bytes);
    for (s, 0..) |*b, i| b.* = tag +% @as(u8, @intCast(i % 251));
    return s;
}

fn cfgFor(root: []const u8) pd.Config {
    return .{ .root = root, .budget = 1 << 30, .floor = 0, .queue = 1 << 20, .rate = 0, .free_override = 1 << 40 };
}

/// Saves the state of `keys` (all of H_1..H_k) through plan + submit, as a leader does; returns the plan's result.
fn save(t: *Tier, keys: []const Key) !pd.Plan {
    const gpa = std.testing.allocator;
    const p = try t.plan(keys, true, false);
    if (p.result != .ok) return p;
    for (p.runs) |r| try t.submitRun(keys[r.first..][0..r.n], r.first + 1, try runBytes(gpa, r.first + 1, r.n));
    try t.submitState(keys, try slotBytes(gpa, @intCast(keys.len)), @intCast(blk: {
        var n: u32 = 0;
        for (p.runs) |r| n += r.n;
        break :blk n;
    }));
    return p;
}

test "chain keys: prefixes share them, a changed token changes every later one, identities separate them" {
    var a: [32]u32 = undefined;
    var b: [32]u32 = undefined;
    var ka: [8]Key = undefined;
    var kb: [8]Key = undefined;
    const h0 = pd.seed("identity A");
    pd.chain(h0, prompt(&a, null), B, &ka);
    pd.chain(h0, prompt(&b, 5), B, &kb);
    for (0..5) |j| try std.testing.expectEqualSlices(u8, &ka[j], &kb[j]);
    for (5..8) |j| try std.testing.expect(!std.mem.eql(u8, &ka[j], &kb[j]));
    // a shorter prompt's keys are the longer one's first keys
    var ks: [3]Key = undefined;
    pd.chain(h0, prompt(&a, null)[0 .. 3 * B + 5], B, &ks);
    for (0..3) |j| try std.testing.expectEqualSlices(u8, &ka[j], &ks[j]);
    var kc: [8]Key = undefined;
    pd.chain(pd.seed("identity B"), prompt(&a, null), B, &kc);
    for (0..8) |j| try std.testing.expect(!std.mem.eql(u8, &ka[j], &kc[j]));
    try std.testing.expect(!std.mem.eql(u8, &pd.namespaceOf("identity A"), &pd.namespaceOf("identity B")));
}

test "a header reads back as written; any changed byte is refused" {
    const h: pd.Head = .{ .kind = .state, .ns = @splat(3), .key = @splat(9), .index = 77, .rank = 1, .payload = 6 << 20, .sum = 0x1234_5678_9abc_def0 };
    var b: [pd.head_bytes]u8 = undefined;
    h.encode(&b);
    try std.testing.expectEqual(h, try pd.Head.decode(&b));
    for (0..pd.head_bytes) |i| {
        var x = b;
        x[i] ^= 0x10;
        try std.testing.expectError(error.BadHeader, pd.Head.decode(&x));
    }
}

test "pair outcomes: both ok resume, a bad side forgets, a busy side only misses" {
    try std.testing.expectEqual(pd.Outcome.hit, pd.Outcome.of(.ok, .ok));
    try std.testing.expectEqual(pd.Outcome.forget, pd.Outcome.of(.ok, .bad));
    try std.testing.expectEqual(pd.Outcome.forget, pd.Outcome.of(.bad, .busy));
    try std.testing.expectEqual(pd.Outcome.miss, pd.Outcome.of(.busy, .ok));
    try std.testing.expectEqual(pd.Outcome.miss, pd.Outcome.of(.ok, .busy));
}

test "identity differences name the keys that changed, appeared or went" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("none", pd.identityDiff(&buf, "a=1\nb=2\n", "a=1\nb=2\n"));
    try std.testing.expectEqualStrings("b,c", pd.identityDiff(&buf, "a=1\nb=2\nc=3\n", "a=1\nb=9\n"));
    try std.testing.expectEqualStrings("d", pd.identityDiff(&buf, "a=1\n", "a=1\nd=4\n"));
}

test "a saved state reads back byte for byte after a reopen; the longest state on disk is found" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "reopen");
    defer rmTree(root);
    var toks: [40]u32 = undefined;
    var keys: [10]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    {
        const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
        defer t.close();
        var p4 = try save(t, keys[0..4]);
        p4.deinit(gpa);
        var p9 = try save(t, keys[0..9]); // blocks 1-4 are already there: only 5-9 are written
        defer p9.deinit(gpa);
        try std.testing.expectEqual(@as(usize, 1), p9.runs.len);
        try std.testing.expectEqual(pd.Run{ .first = 4, .n = 5 }, p9.runs[0]);
        try std.testing.expect(t.flush(5 * std.time.ns_per_s));
        try std.testing.expectEqual(@as(?u32, 9), t.longest(&keys, 0, 10));
        try std.testing.expectEqual(@as(?u32, 4), t.longest(&keys, 0, 8));
        try std.testing.expectEqual(@as(?u32, null), t.longest(&keys, 9, 10)); // nothing past the in-memory state
    }
    var db: [256]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 9), countFiles(try std.fmt.bufPrint(&db, "{s}/{s}/b", .{ root, &std.fmt.bytesToHex(pd.namespaceOf("id"), .lower) }), ".blk"));
    try std.testing.expectEqual(@as(usize, 0), countFiles(root, ".tmp"));
    const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
    defer t.close();
    try std.testing.expectEqual(@as(?u32, 9), t.longest(&keys, 0, 10));
    var got_keys: [9]Key = undefined;
    const slot = try gpa.alloc(u8, slot_bytes);
    defer gpa.free(slot);
    try std.testing.expectEqual(pd.Load.ok, t.readState(keys[8], 9, &got_keys, slot));
    const want_slot = try slotBytes(gpa, 9);
    defer gpa.free(want_slot);
    try std.testing.expectEqualSlices(u8, want_slot, slot);
    for (0..9) |j| try std.testing.expectEqualSlices(u8, &keys[j], &got_keys[j]);
    const data = try gpa.alloc(u8, 9 * block_bytes);
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + block_bytes);
    defer gpa.free(file);
    try std.testing.expect(t.readRun(&got_keys, 1, data, file));
    const want = try runBytes(gpa, 1, 9);
    defer gpa.free(want);
    try std.testing.expectEqualSlices(u8, want, data);
    // a run read in two parts lays each part out as its own run
    const part = try gpa.alloc(u8, 4 * block_bytes);
    defer gpa.free(part);
    try std.testing.expect(t.readRun(got_keys[5..9], 6, part, file));
    const want_part = try runBytes(gpa, 6, 4);
    defer gpa.free(want_part);
    try std.testing.expectEqualSlices(u8, want_part, part);
    // the wrong index for a block (a file under another key's place) is refused
    try std.testing.expect(!t.readRun(got_keys[5..9], 5, part, file));
}

/// The path of block `k` or state `k` in root's namespace "id".
fn path(buf: []u8, root: []const u8, k: Key, block: bool) ![:0]u8 {
    const ns = std.fmt.bytesToHex(pd.namespaceOf("id"), .lower);
    const hex = std.fmt.bytesToHex(k, .lower);
    return if (block) std.fmt.bufPrintSentinel(buf, "{s}/{s}/b/{s}/{s}.blk", .{ root, &ns, hex[0..2], &hex }, 0) else std.fmt.bufPrintSentinel(buf, "{s}/{s}/s/{s}.st", .{ root, &ns, &hex }, 0);
}

fn flipByte(p: [:0]const u8, at: u64) !void {
    const fd = c.open(p, .{ .ACCMODE = .RDWR }, @as(c.mode_t, 0));
    if (fd < 0) return error.TestFile;
    defer _ = c.close(fd);
    var b: [1]u8 = undefined;
    if (c.pread(fd, &b, 1, @intCast(at)) != 1) return error.TestFile;
    b[0] ^= 0x40;
    if (c.pwrite(fd, &b, 1, @intCast(at)) != 1) return error.TestFile;
}

test "a corrupt or missing file is a bad resume; a state still being written is busy" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "corrupt");
    defer rmTree(root);
    var toks: [24]u32 = undefined;
    var keys: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
    defer t.close();
    var p = try save(t, &keys);
    p.deinit(gpa);
    var shorter = try save(t, keys[0..4]); // shares blocks 1-4
    shorter.deinit(gpa);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    var got: [6]Key = undefined;
    const slot = try gpa.alloc(u8, slot_bytes);
    defer gpa.free(slot);
    const data = try gpa.alloc(u8, 6 * block_bytes);
    defer gpa.free(data);
    const file = try gpa.alloc(u8, pd.head_bytes + block_bytes);
    defer gpa.free(file);
    var pb: [512]u8 = undefined;
    // a flipped payload byte of block 3
    try std.testing.expect(t.readRun(&keys, 1, data, file));
    try flipByte(try path(&pb, root, keys[2], true), pd.head_bytes + 5);
    try std.testing.expect(!t.readRun(&keys, 1, data, file));
    // a flipped byte of the state's own bytes
    try std.testing.expectEqual(pd.Load.ok, t.readState(keys[5], 6, &got, slot));
    try flipByte(try path(&pb, root, keys[5], false), pd.head_bytes + 6 * 32 + 17);
    try std.testing.expectEqual(pd.Load.bad, t.readState(keys[5], 6, &got, slot));
    // a missing block
    _ = c.unlink(try path(&pb, root, keys[0], true));
    try std.testing.expect(!t.readRun(keys[0..1], 1, data, file));
    // the bad block's key is named, and every state that lists it goes
    var bad: Key = undefined;
    try std.testing.expect(!t.readRunOr(&keys, 1, data, file, &bad));
    try std.testing.expectEqualSlices(u8, &keys[0], &bad);
    try std.testing.expectEqual(@as(usize, 2), t.dropBlock(bad));
    try std.testing.expectEqual(@as(?u32, null), t.longest(&keys, 0, 6));
    // an unknown state, and one whose k is not the asked one
    try std.testing.expectEqual(pd.Load.bad, t.readState(keys[3], 4, got[0..4], slot));
    // a state planned but not yet written is busy
    var toks2: [24]u32 = undefined;
    var keys2: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks2, 0), B, &keys2);
    var p2 = try t.plan(&keys2, true, false);
    defer p2.deinit(gpa);
    try std.testing.expectEqual(pd.Load.busy, t.readState(keys2[5], 6, &got, slot));
    t.abandon(&p2, &keys2, true, 0);
    try std.testing.expectEqual(pd.Load.bad, t.readState(keys2[5], 6, &got, slot));
    try std.testing.expectEqual(@as(u64, 1 << 20), t.queueRoom()); // the reservation came back
}

test "the scan at open removes temporary files, unlisted blocks and states missing a block" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "scan");
    defer rmTree(root);
    var toks: [32]u32 = undefined;
    var keys: [8]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    var toks2: [32]u32 = undefined;
    var keys2: [8]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks2, 2), B, &keys2);
    {
        const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
        defer t.close();
        var p = try save(t, keys[0..3]);
        p.deinit(gpa);
        var q = try save(t, &keys2); // shares blocks 1-2 with the first
        q.deinit(gpa);
        // streamed blocks of a pass that kept nothing
        var s = try t.plan(keys[3..8], false, true);
        for (s.runs) |r| try t.submitRun(keys[3 + r.first ..][0..r.n], 4 + r.first, try runBytes(gpa, 4 + r.first, r.n));
        s.deinit(gpa);
        try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    }
    var pb: [512]u8 = undefined;
    const ns = std.fmt.bytesToHex(pd.namespaceOf("id"), .lower);
    // a crash's leftovers: a .tmp, and the second state losing a block of its own
    const tmp = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}/s/x.st.tmp", .{ root, &ns }, 0);
    const fd = c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true }, @as(c.mode_t, 0o600));
    _ = c.close(fd);
    _ = c.unlink(try path(&pb, root, keys2[6], true));
    const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
    defer t.close();
    try std.testing.expectEqual(@as(?u32, 3), t.longest(&keys, 0, 8));
    try std.testing.expectEqual(@as(?u32, null), t.longest(&keys2, 0, 8));
    var sb: [256]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 0), countFiles(root, ".tmp"));
    try std.testing.expectEqual(@as(usize, 1), countFiles(try std.fmt.bufPrint(&sb, "{s}/{s}/s", .{ root, &ns }), ".st"));
    // only the first state's three blocks stay: the second's own and the streamed ones no state listed go
    try std.testing.expectEqual(@as(usize, 3), countFiles(try std.fmt.bufPrint(&sb, "{s}/{s}/b", .{ root, &ns }), ".blk"));
    try std.testing.expectEqual(3 * (pd.head_bytes + block_bytes) + pd.head_bytes + 3 * 32 + slot_bytes, t.bytes);
}

test "a pass's streamed blocks a state takes stay; the rest go when the pass ends" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "stream");
    defer rmTree(root);
    var toks: [32]u32 = undefined;
    var keys: [8]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    const t = try Tier.open(gpa, cfgFor(root), "id", 0, slot_bytes, &planes);
    defer t.close();
    var s = try t.plan(&keys, false, true);
    for (s.runs) |r| try t.submitRun(keys[r.first..][0..r.n], 1 + r.first, try runBytes(gpa, 1 + r.first, r.n));
    s.deinit(gpa);
    var p = try save(t, keys[0..5]); // its blocks were streamed: only its own bytes are written now
    defer p.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), p.runs.len);
    t.endPass();
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    var sb: [256]u8 = undefined;
    const ns = std.fmt.bytesToHex(pd.namespaceOf("id"), .lower);
    try std.testing.expectEqual(@as(usize, 5), countFiles(try std.fmt.bufPrint(&sb, "{s}/{s}/b", .{ root, &ns }), ".blk"));
    try std.testing.expectEqual(@as(?u32, 5), t.longest(&keys, 0, 8));
}

test "eviction frees the least recently used state and only the blocks no other state holds" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "evict");
    defer rmTree(root);
    var ta: [24]u32 = undefined;
    var ka: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&ta, null), B, &ka);
    var tb: [24]u32 = undefined;
    var kb: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&tb, 0), B, &kb); // shares nothing with a
    const blk = pd.head_bytes + block_bytes;
    const st = struct {
        fn f(k: u64) u64 {
            return pd.head_bytes + 32 * k + slot_bytes;
        }
    }.f;
    var cfg = cfgFor(root);
    cfg.budget = 6 * blk + st(3) + st(6) + 4 * blk + st(4) - 1; // a's 3 and 6 fit with b's 4 only if one goes
    const t = try Tier.open(gpa, cfg, "id", 0, slot_bytes, &planes);
    defer t.close();
    var p1 = try save(t, ka[0..3]);
    p1.deinit(gpa);
    var p2 = try save(t, ka[0..6]);
    p2.deinit(gpa);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    t.touch(ka[2]); // the shorter one was resumed since: the longer is now the least recently used
    var p3 = try save(t, kb[0..4]);
    defer p3.deinit(gpa);
    try std.testing.expect(p3.result == .ok);
    try std.testing.expectEqual(@as(usize, 1), p3.victims.len);
    try std.testing.expectEqualSlices(u8, &ka[5], &p3.victims[0]);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(?u32, 3), t.longest(&ka, 0, 6)); // its blocks 1-3 stayed with the shorter state
    try std.testing.expectEqual(@as(?u32, 4), t.longest(&kb, 0, 6));
    var pb: [512]u8 = undefined;
    for (0..3) |j| try std.testing.expect(exists(try path(&pb, root, ka[j], true)));
    for (3..6) |j| try std.testing.expect(!exists(try path(&pb, root, ka[j], true)));
    try std.testing.expect(t.bytes <= cfg.budget);
    // the follower never evicts: past its budget (and its slack) it refuses
    t.cfg.budget = 1;
    t.cfg.follower = true;
    var p4 = try save(t, ka[0..6]);
    defer p4.deinit(gpa);
    try std.testing.expectEqual(pd.Reason.budget, p4.result.skip);
    try std.testing.expectEqual(@as(?u32, 4), t.longest(&kb, 0, 6));
}

test "the free-space floor and the queue bound skip a save and change nothing" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "floor");
    defer rmTree(root);
    var toks: [24]u32 = undefined;
    var keys: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    var cfg = cfgFor(root);
    cfg.floor = 16 << 30;
    cfg.free_override = 16 << 30; // exactly the floor: nothing more may be written
    const t = try Tier.open(gpa, cfg, "id", 0, slot_bytes, &planes);
    defer t.close();
    var p = try save(t, &keys);
    defer p.deinit(gpa);
    try std.testing.expectEqual(pd.Reason.floor, p.result.skip);
    try std.testing.expectEqual(@as(u64, 0), t.bytes);
    t.cfg.free_override = 1 << 40;
    t.cfg.queue = 6 * block_bytes; // the blocks fit, the state's own bytes do not
    var q = try save(t, &keys);
    defer q.deinit(gpa);
    try std.testing.expectEqual(pd.Reason.queue, q.result.skip);
    try std.testing.expectEqual(@as(?pd.Status, null), t.stateOf(keys[5]));
    t.cfg.queue = 1 << 20;
    var r = try save(t, &keys);
    defer r.deinit(gpa);
    try std.testing.expect(r.result == .ok);
    try std.testing.expect(t.queueRoom() <= (1 << 20)); // reserved until written
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(u64, 1 << 20), t.queueRoom());
    var again = try save(t, &keys);
    defer again.deinit(gpa);
    try std.testing.expectEqual(pd.Reason.exists, again.result.skip);
}

test "reconcile: a bitmap of the other side's keys, and the states each side drops" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "reconcile");
    defer rmTree(root);
    var toks: [24]u32 = undefined;
    var keys: [6]Key = undefined;
    pd.chain(pd.seed("id"), prompt(&toks, null), B, &keys);
    const t = try Tier.open(gpa, cfgFor(root), "id", 1, slot_bytes, &planes);
    defer t.close();
    var p = try save(t, keys[0..2]);
    p.deinit(gpa);
    var q = try save(t, keys[0..5]);
    q.deinit(gpa);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    // the leader lists the states at 2, 4 and 5 blocks; this side lacks the one at 4
    const listed = [_]Key{ keys[1], keys[3], keys[4] };
    const map = try pd.bitmap(gpa, t, &listed);
    defer gpa.free(map);
    try std.testing.expect(pd.bit(map, 0) and !pd.bit(map, 1) and pd.bit(map, 2) and !pd.bit(map, 9));
    // a state here the leader does not list goes
    const mine = try t.completeKeys();
    defer gpa.free(mine);
    try std.testing.expectEqual(@as(usize, 2), mine.len);
    t.drop(&.{keys[1]});
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(?u32, 5), t.longest(&keys, 0, 6));
    try std.testing.expectEqual(@as(?u32, null), t.longest(&keys, 0, 4));
}

test "another identity's directory counts against the budget and goes first when room is needed" {
    const gpa = std.testing.allocator;
    var rb: [128]u8 = undefined;
    const root = try scratch(&rb, "others");
    defer rmTree(root);
    var toks: [24]u32 = undefined;
    var keys: [6]Key = undefined;
    pd.chain(pd.seed("old"), prompt(&toks, null), B, &keys);
    var old_ns: [32]u8 = undefined;
    {
        const t = try Tier.open(gpa, cfgFor(root), "old", 0, slot_bytes, &planes);
        defer t.close();
        old_ns = t.ns_hex;
        var p = try save(t, &keys);
        p.deinit(gpa);
        try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    }
    var cfg = cfgFor(root);
    const blk = pd.head_bytes + block_bytes;
    cfg.budget = 6 * blk + pd.head_bytes + 6 * 32 + slot_bytes + 100; // room for one state of six blocks in all
    const t = try Tier.open(gpa, cfg, "new", 0, slot_bytes, &planes);
    defer t.close();
    try std.testing.expect(t.others_bytes > 0);
    var nk: [6]Key = undefined;
    pd.chain(pd.seed("new"), prompt(&toks, null), B, &nk);
    var p = try save(t, &nk);
    defer p.deinit(gpa);
    try std.testing.expect(p.result == .ok);
    try std.testing.expect(t.flush(5 * std.time.ns_per_s));
    var ob: [256]u8 = undefined;
    try std.testing.expect(!exists(try std.fmt.bufPrint(&ob, "{s}/{s}/IDENTITY", .{ root, &old_ns })));
    // a namespace whose IDENTITY is another text is refused (a hash prefix collision)
    var ib: [256]u8 = undefined;
    const ip = try std.fmt.bufPrintSentinel(&ib, "{s}/{s}/IDENTITY", .{ root, &t.ns_hex }, 0);
    const fd = c.open(ip, .{ .ACCMODE = .WRONLY, .TRUNC = true }, @as(c.mode_t, 0));
    _ = c.write(fd, "other", 5);
    _ = c.close(fd);
    try std.testing.expectError(error.ForeignNamespace, Tier.open(gpa, cfg, "new", 0, slot_bytes, &planes));
}

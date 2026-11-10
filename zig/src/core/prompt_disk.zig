//! A persistent tier for kept prompt states (sf-disk): content-addressed blocks of a state's position rows plus small
//! state files, written by a background thread and read back after a restart. Family-neutral: a family hands it host
//! bytes laid out by `planes` (each block's bytes, plane by plane) and its states' own bytes (`slot_bytes`), and does
//! every device copy itself (dsv41: families/dsv41/disk.zig). One Tier per rank and box; in a TP pair the leader (rank
//! 0) decides saves, evictions and resumes and the follower (rank 1) mirrors them (it never evicts on its own).
//!
//! Keys are chain hashes over the prompt's tokens: H_0 = SHA-256("sf-disk/1" || identity), H_j = SHA-256(H_{j-1} ||
//! block j's token bytes). Block j (positions [(j-1)B, jB)) is file b/xx/<H_j>.blk; the state at jB is s/<H_j>.st and
//! lists H_1..H_j. A state file is written only after its blocks, so a state on disk promises its blocks (the scan at
//! open checks it again). Every file carries a header with its kind, namespace, key, index, rank and an XXH3-64 of its
//! payload, itself checksummed; a resume verifies every byte it reads.
const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const c = std.c;
const linux = std.os.linux;
const Sha256 = std.crypto.hash.sha2.Sha256;
const XxHash3 = std.hash.XxHash3;

pub const Key = [32]u8;
pub const version: u32 = 1;
/// Every file's header: magic, version, namespace, key, index, rank, payload bytes, payload sum, header sum.
pub const head_bytes = 88;
const magic_block = "SFDB".*;
const magic_state = "SFDS".*;
/// The most blocks a state lists (2^22 tokens at 2048 a block: past any context this engine serves).
pub const max_blocks = 4096;
/// A writer's write call at most (its rate cap sleeps between them).
const piece_bytes = 8 << 20;

// -- keys ---------------------------------------------------------------------------------------------------------

/// H_0 for a namespace's identity text.
pub fn seed(identity: []const u8) Key {
    var h = Sha256.init(.{});
    h.update("sf-disk/1");
    h.update(identity);
    return h.finalResult();
}

/// The namespace (directory name) of an identity: the first 16 bytes of its SHA-256.
pub fn namespaceOf(identity: []const u8) [16]u8 {
    var d: [32]u8 = undefined;
    Sha256.hash(identity, &d, .{});
    return d[0..16].*;
}

/// H_1 .. H_n of `tokens` (the ids' bytes) in blocks of `block` bytes: out[j-1] = H_j, n = out.len (at most the whole
/// blocks the bytes hold).
pub fn chain(h0: Key, tokens: []const u8, block: usize, out: []Key) void {
    std.debug.assert(out.len * block <= tokens.len);
    var prev = h0;
    for (out, 0..) |*o, i| {
        var h = Sha256.init(.{});
        h.update(&prev);
        h.update(tokens[i * block ..][0..block]);
        o.* = h.finalResult();
        prev = o.*;
    }
}

/// A key's first 16 hex digits (log lines).
pub fn short(k: Key) [16]u8 {
    return std.fmt.bytesToHex(k[0..8].*, .lower);
}

// -- headers ------------------------------------------------------------------------------------------------------

pub const Kind = enum { block, state };

pub const Head = struct {
    kind: Kind,
    ns: [16]u8,
    key: Key,
    index: u32, // a block's j, a state's k
    rank: u32,
    payload: u64,
    sum: u64,

    pub fn encode(h: Head, out: *[head_bytes]u8) void {
        out[0..4].* = if (h.kind == .block) magic_block else magic_state;
        std.mem.writeInt(u32, out[4..8], version, .little);
        out[8..24].* = h.ns;
        out[24..56].* = h.key;
        std.mem.writeInt(u32, out[56..60], h.index, .little);
        std.mem.writeInt(u32, out[60..64], h.rank, .little);
        std.mem.writeInt(u64, out[64..72], h.payload, .little);
        std.mem.writeInt(u64, out[72..80], h.sum, .little);
        std.mem.writeInt(u64, out[80..88], XxHash3.hash(0, out[0..80]), .little);
    }

    pub fn decode(b: *const [head_bytes]u8) !Head {
        if (std.mem.readInt(u64, b[80..88], .little) != XxHash3.hash(0, b[0..80])) return error.BadHeader;
        const kind: Kind = if (std.mem.eql(u8, b[0..4], &magic_block)) .block else if (std.mem.eql(u8, b[0..4], &magic_state)) .state else return error.BadHeader;
        if (std.mem.readInt(u32, b[4..8], .little) != version) return error.BadHeader;
        return .{ .kind = kind, .ns = b[8..24].*, .key = b[24..56].*, .index = std.mem.readInt(u32, b[56..60], .little), .rank = std.mem.readInt(u32, b[60..64], .little), .payload = std.mem.readInt(u64, b[64..72], .little), .sum = std.mem.readInt(u64, b[72..80], .little) };
    }
};

// -- layout -------------------------------------------------------------------------------------------------------

/// A run of n blocks in host memory: plane p of the run at rangeOffset(planes, n, p), block i's part of it at
/// + i * planes[p] (each plane's rows of consecutive blocks are consecutive, as in the device plane).
pub fn rangeOffset(planes: []const u64, n: usize, p: usize) u64 {
    var off: u64 = 0;
    for (planes[0..p]) |x| off += x * n;
    return off;
}

pub fn sum(planes: []const u64) u64 {
    var n: u64 = 0;
    for (planes) |x| n += x;
    return n;
}

// -- the index ----------------------------------------------------------------------------------------------------

pub const Status = enum(u8) { pending, complete };

const StateRec = struct { k: u32, bytes: u64, used: i128, status: Status, keys: []Key };
const BlockRec = struct { bytes: u64, refs: u32 = 0, status: Status, pinned: bool = false, guard: bool = false };

/// A resume's answer on one rank, and the pair's outcome from both.
pub const Load = enum(u8) {
    ok = 0,
    bad = 1, // missing or corrupt: both ranks forget the state
    busy = 2, // still being written here: a miss, kept

    pub fn name(l: Load) []const u8 {
        return @tagName(l);
    }
};
pub const Outcome = enum(u8) {
    hit = 0,
    miss = 1,
    forget = 2,

    pub fn of(a: Load, b: Load) Outcome {
        if (a == .ok and b == .ok) return .hit;
        if (a == .bad or b == .bad) return .forget;
        return .miss;
    }
};

pub const Reason = enum { exists, queue, floor, budget, io, hash, off };

pub const Config = struct {
    root: []const u8,
    budget: u64,
    floor: u64 = 16 << 30,
    queue: u64 = 1 << 30,
    rate: u64 = 1 << 30, // the writer's bytes a second at most (0: no cap)
    follower: bool = false, // rank 1: refuses what passes its budget instead of evicting
    free_override: ?u64 = null, // tests: the free bytes the filesystem reports
    drain_ns: u64 = 20 * std.time.ns_per_s, // close: how long the queued writes may still run
    block_tokens: u64 = 2048, // a block's positions (log lines' tokens=)
};

/// A run of blocks a save or a stream writes: indices into the keys it was planned for.
pub const Run = struct { first: u32, n: u32 };

/// What `plan` decided: the evicted states (the leader tells its follower), the runs of blocks absent here, and the
/// queue bytes reserved for them (and the state's own bytes).
pub const Plan = struct {
    result: union(enum) { ok, skip: Reason },
    victims: []Key = &.{},
    runs: []Run = &.{},
    charge: u64 = 0,

    pub fn deinit(p: *Plan, gpa: Allocator) void {
        gpa.free(p.victims);
        gpa.free(p.runs);
        p.* = undefined;
    }
};

const Job = union(enum) {
    run: struct { keys: []Key, first_j: u32, data: []u8, charge: u64, t0: i128 },
    state: struct { key: Key, k: u32, keys: []Key, slot: []u8, charge: u64, t0: i128, new_blocks: u32, new_bytes: u64 },
    delete: struct { states: []Key, blocks: []Key },
    sweep: struct { name: [32]u8 },
};

const Done = struct { job: Job, ok: bool, reason: Reason = .io };

const Other = struct { name: [32]u8, bytes: u64, used: i128 };

pub const Counts = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    forgets: u64 = 0,
    busy: u64 = 0,
    saved: u64 = 0,
    skipped: u64 = 0,
    failed: u64 = 0,
    evicted: u64 = 0,
};

pub const Tier = struct {
    gpa: Allocator,
    cfg: Config,
    rank: u32,
    ns: [16]u8,
    ns_hex: [32]u8,
    dir: [:0]u8,
    slot_bytes: u64,
    planes: []u64,
    block_bytes: u64,
    states: std.AutoHashMapUnmanaged(Key, StateRec) = .empty,
    blocks: std.AutoHashMapUnmanaged(Key, BlockRec) = .empty,
    bytes: u64 = 0, // this namespace's files, pending and complete
    others: std.ArrayList(Other) = .empty,
    others_bytes: u64 = 0,
    counts: Counts = .{},
    // the writer (`mu` guards jobs, done, queued, stop)
    mu: c.pthread_mutex_t = .{},
    cv: c.pthread_cond_t = .{},
    jobs: std.ArrayList(Job) = .empty,
    head: usize = 0,
    done: std.ArrayList(Done) = .empty,
    queued: u64 = 0,
    stop: bool = false,
    running: bool = false, // the writer holds a job
    deadline: i128 = 0,
    thread: ?std.Thread = null,
    written: std.atomic.Value(u64) = .init(0), // the writer's totals (its own lines)
    clock: i128 = 0, // the last `used` handed out (wall ns, strictly increasing)

    /// The store of `identity` under cfg.root (made when missing), its index rebuilt from the files, the writer started.
    pub fn open(gpa: Allocator, cfg: Config, identity: []const u8, rank: u32, slot_bytes: u64, planes: []const u64) !*Tier {
        const t = try gpa.create(Tier);
        errdefer gpa.destroy(t);
        const ns = namespaceOf(identity);
        t.* = .{ .gpa = gpa, .cfg = cfg, .rank = rank, .ns = ns, .ns_hex = std.fmt.bytesToHex(ns, .lower), .dir = undefined, .slot_bytes = slot_bytes, .planes = undefined, .block_bytes = sum(planes) };
        t.cfg.root = try gpa.dupe(u8, cfg.root);
        errdefer gpa.free(t.cfg.root);
        t.planes = try gpa.dupe(u64, planes);
        errdefer gpa.free(t.planes);
        t.dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ cfg.root, &t.ns_hex }, 0);
        errdefer gpa.free(t.dir);
        try makePath(gpa, t.dir);
        try t.sub("s");
        try t.sub("b");
        try t.claim(identity);
        errdefer t.freeIndex();
        try t.scan();
        t.scanOthers(identity);
        t.thread = try std.Thread.spawn(.{}, writer, .{t});
        return t;
    }

    /// Stops the writer once the queued writes are done (at most cfg.drain_ns; the rest are dropped) and frees all.
    pub fn close(t: *Tier) void {
        _ = c.pthread_mutex_lock(&t.mu);
        t.stop = true;
        t.deadline = now() + t.cfg.drain_ns;
        _ = c.pthread_cond_broadcast(&t.cv);
        _ = c.pthread_mutex_unlock(&t.mu);
        if (t.thread) |th| th.join();
        t.drain();
        for (t.jobs.items[t.head..]) |*j| freeJob(t.gpa, j);
        t.jobs.deinit(t.gpa);
        t.done.deinit(t.gpa);
        t.freeIndex();
        t.others.deinit(t.gpa);
        t.gpa.free(t.dir);
        t.gpa.free(t.planes);
        t.gpa.free(t.cfg.root);
        t.gpa.destroy(t);
    }

    fn freeIndex(t: *Tier) void {
        var it = t.states.valueIterator();
        while (it.next()) |s| t.gpa.free(s.keys);
        t.states.deinit(t.gpa);
        t.blocks.deinit(t.gpa);
        t.states = .empty;
        t.blocks = .empty;
    }

    fn sub(t: *Tier, name: []const u8) !void {
        var buf: [1100]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&buf, "{s}/{s}", .{ t.dir, name }, 0);
        const rc = c.mkdir(p, 0o700);
        if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
    }

    /// IDENTITY: written when new; another identity in this namespace (a hash prefix collision) is refused.
    fn claim(t: *Tier, identity: []const u8) !void {
        var buf: [1100]u8 = undefined;
        const p = try std.fmt.bufPrintSentinel(&buf, "{s}/IDENTITY", .{t.dir}, 0);
        if (readSmall(t.gpa, p)) |have| {
            defer t.gpa.free(have);
            if (!std.mem.eql(u8, have, identity)) return error.ForeignNamespace;
            touchPath(p);
            return;
        }
        var tb: [1100]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&tb, "{s}/IDENTITY.tmp", .{t.dir}, 0);
        try writeFile(tmp, &.{identity});
        if (c.rename(tmp, p) != 0) return error.DiskWrite;
    }

    // -- open: the index from the files ------------------------------------------------------------------------

    fn scan(t: *Tier) !void {
        // blocks: name and size (headers are read at a resume)
        var bdir: [1100]u8 = undefined;
        for (0..256) |x| {
            const hex = std.fmt.bytesToHex([1]u8{@intCast(x)}, .lower);
            const d = try std.fmt.bufPrint(&bdir, "{s}/b/{s}", .{ t.dir, &hex });
            var it = Entries.open(d) orelse continue;
            defer it.close();
            while (it.next()) |name| {
                var pb: [1400]u8 = undefined;
                const path = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ d, name }, 0) catch continue;
                if (std.mem.endsWith(u8, name, ".tmp")) {
                    _ = c.unlink(path);
                    continue;
                }
                const key = keyName(name, ".blk") orelse continue;
                const st = statPath(path) orelse continue;
                if (st.size != head_bytes + t.block_bytes or !std.mem.eql(u8, name[0..2], &hex)) {
                    _ = c.unlink(path);
                    continue;
                }
                try t.blocks.put(t.gpa, key, .{ .bytes = st.size, .status = .complete });
            }
        }
        // states: header and block list; a state missing a block, or not this namespace's, goes
        var sd: [1100]u8 = undefined;
        const sdir = try std.fmt.bufPrint(&sd, "{s}/s", .{t.dir});
        var it = Entries.open(sdir) orelse return error.DiskDir;
        defer it.close();
        while (it.next()) |name| {
            var pb: [1400]u8 = undefined;
            const path = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ sdir, name }, 0) catch continue;
            if (std.mem.endsWith(u8, name, ".tmp")) {
                _ = c.unlink(path);
                continue;
            }
            const key = keyName(name, ".st") orelse continue;
            const rec = t.readStateMeta(path, key) catch {
                _ = c.unlink(path);
                continue;
            };
            const whole = for (rec.keys) |bk| {
                if (!t.blocks.contains(bk)) break false;
            } else true;
            if (!whole) {
                t.gpa.free(rec.keys);
                _ = c.unlink(path);
                continue;
            }
            for (rec.keys) |bk| t.blocks.getPtr(bk).?.refs += 1;
            t.states.put(t.gpa, key, rec) catch |e| {
                t.gpa.free(rec.keys);
                return e;
            };
            t.clock = @max(t.clock, rec.used);
        }
        // blocks no state lists (a crash between a run and its state, an eviction cut short)
        var gone: std.ArrayList(Key) = .empty;
        defer gone.deinit(t.gpa);
        var bi = t.blocks.iterator();
        while (bi.next()) |e| if (e.value_ptr.refs == 0) try gone.append(t.gpa, e.key_ptr.*);
        for (gone.items) |k| {
            _ = t.blocks.remove(k);
            var pb: [1400]u8 = undefined;
            _ = c.unlink(t.blockPath(&pb, k, false) catch continue);
        }
        var bv = t.blocks.valueIterator();
        while (bv.next()) |b| t.bytes += b.bytes;
        var sv = t.states.valueIterator();
        while (sv.next()) |s| t.bytes += s.bytes;
    }

    fn readStateMeta(t: *Tier, path: [:0]const u8, key: Key) !StateRec {
        const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0) return error.DiskRead;
        defer _ = c.close(fd);
        var hb: [head_bytes]u8 = undefined;
        if (!readAt(fd, &hb, 0)) return error.DiskRead;
        const h = try Head.decode(&hb);
        if (h.kind != .state or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &key) or h.rank != t.rank or h.index == 0 or h.index > max_blocks) return error.BadHeader;
        const file = head_bytes + 32 * @as(u64, h.index) + t.slot_bytes;
        const st = statFd(fd) orelse return error.DiskRead;
        if (st.size != file or h.payload != t.slot_bytes) return error.BadSize;
        const keys = try t.gpa.alloc(Key, h.index);
        errdefer t.gpa.free(keys);
        if (!readAt(fd, std.mem.sliceAsBytes(keys), head_bytes)) return error.DiskRead;
        if (!std.mem.eql(u8, &keys[h.index - 1], &key)) return error.BadHeader;
        return .{ .k = h.index, .bytes = file, .used = st.mtime, .status = .complete, .keys = keys };
    }

    /// Other namespaces under the root: their bytes count against the budget; each logged with what differs.
    fn scanOthers(t: *Tier, identity: []const u8) void {
        var it = Entries.open(t.cfg.root) orelse return;
        defer it.close();
        while (it.next()) |name| {
            if (name.len != 32 or std.mem.eql(u8, name, &t.ns_hex) or !allHex(name)) continue;
            {
                var pb: [1100]u8 = undefined;
                const path = std.fmt.bufPrint(&pb, "{s}/{s}", .{ t.cfg.root, name }) catch continue;
                const bytes = treeBytes(path, false);
                var ib: [1200]u8 = undefined;
                const ip = std.fmt.bufPrintSentinel(&ib, "{s}/IDENTITY", .{path}, 0) catch continue;
                const used: i128 = if (statPath(ip)) |st| st.mtime else 0;
                t.others.append(t.gpa, .{ .name = name[0..32].*, .bytes = bytes, .used = used }) catch return;
                t.others_bytes += bytes;
                var diff_buf: [1024]u8 = undefined;
                const differs = if (readSmall(t.gpa, ip)) |theirs| blk: {
                    defer t.gpa.free(theirs);
                    break :blk identityDiff(&diff_buf, identity, theirs);
                } else "no-identity";
                t.line("event=foreign ns={s} differs={s} mib={d}", .{ name, differs, bytes >> 20 });
            }
        }
    }

    // -- the leader's and the follower's decisions (their engine thread) ----------------------------------------

    /// The writer's results applied: written states complete, failed ones gone with their blocks.
    pub fn drain(t: *Tier) void {
        _ = c.pthread_mutex_lock(&t.mu);
        const list = t.done;
        t.done = .empty;
        _ = c.pthread_mutex_unlock(&t.mu);
        var l = list;
        defer l.deinit(t.gpa);
        for (l.items) |*d| {
            switch (d.job) {
                .run => |r| for (r.keys) |k| {
                    const b = t.blocks.getPtr(k) orelse continue;
                    if (d.ok) {
                        b.status = .complete;
                    } else if (b.refs == 0 and !b.pinned) {
                        t.forgetBlock(k, null);
                    } // (else it stays pending: the state that needs it fails at its own write and lets it go)
                },
                .state => |s| if (t.states.getPtr(s.key)) |rec| {
                    if (d.ok) {
                        rec.status = .complete;
                        t.counts.saved += 1;
                    } else {
                        t.counts.failed += 1;
                        t.dropState(s.key);
                    }
                },
                else => {},
            }
            freeJob(t.gpa, &d.job);
        }
    }

    /// Waits until the writer has nothing queued or running, then applies its results (tests, and a resume that wants a
    /// state being written: bounded by `max_ns`). True when it emptied in time.
    pub fn flush(t: *Tier, max_ns: u64) bool {
        const until = now() + max_ns;
        while (true) {
            _ = c.pthread_mutex_lock(&t.mu);
            const idle = t.head == t.jobs.items.len and !t.running;
            _ = c.pthread_mutex_unlock(&t.mu);
            if (idle) break;
            if (now() > until) return false;
            sleepNs(std.time.ns_per_ms);
        }
        t.drain();
        return true;
    }

    /// A state's place in the store: none, being written, or on disk.
    pub fn stateOf(t: *Tier, key: Key) ?Status {
        t.drain();
        return if (t.states.get(key)) |s| s.status else null;
    }

    /// The largest k in (k_min, k_max] whose state is on disk under hashes[k-1] (the prompt's chain).
    pub fn longest(t: *Tier, hashes: []const Key, k_min: u32, k_max: u32) ?u32 {
        t.drain();
        var k = @min(k_max, @as(u32, @intCast(hashes.len)));
        while (k > k_min) : (k -= 1) if (t.states.get(hashes[k - 1])) |s| if (s.status == .complete and s.k == k) return k;
        return null;
    }

    /// The bytes on the root's filesystem a writer may still use (statvfs, less nothing).
    pub fn freeBytes(t: *const Tier) u64 {
        if (t.cfg.free_override) |v| return v;
        return fsFree(t.dir);
    }

    pub fn queueRoom(t: *Tier) u64 {
        _ = c.pthread_mutex_lock(&t.mu);
        defer _ = c.pthread_mutex_unlock(&t.mu);
        return t.cfg.queue -| t.queued;
    }

    /// The files a run of n blocks makes, and the host bytes it holds while queued.
    fn runFiles(t: *const Tier, n: u64) u64 {
        return n * (head_bytes + t.block_bytes);
    }

    pub fn stateFile(t: *const Tier, k: u64) u64 {
        return head_bytes + 32 * k + t.slot_bytes;
    }

    /// Plans writing the blocks `keys` (consecutive blocks of one prompt) and, when `state` is set, the state at the
    /// last of them (then `keys` holds all of H_1..H_k). Blocks absent here become pending (pinned
    /// to the prompt pass when `pin`); the state becomes pending with a reference on each of its blocks; the queue
    /// bytes are reserved. The leader evicts least recently used states for room (plan.victims: delete jobs queued
    /// here, the follower hears them); the follower refuses what passes its budget. Skipped plans change nothing.
    pub fn plan(t: *Tier, keys: []const Key, state: bool, pin: bool) !Plan {
        t.drain();
        const gpa = t.gpa;
        if (keys.len == 0) return .{ .result = .ok };
        if (state and t.states.contains(keys[keys.len - 1])) return .{ .result = .{ .skip = .exists } };
        var runs: std.ArrayList(Run) = .empty;
        defer runs.deinit(gpa);
        var victims: std.ArrayList(Key) = .empty;
        defer victims.deinit(gpa);
        var missing: u64 = 0;
        for (keys, 0..) |k, i| {
            if (t.blocks.contains(k)) continue;
            const items = runs.items;
            if (items.len > 0 and items[items.len - 1].first + items[items.len - 1].n == i) {
                items[items.len - 1].n += 1;
            } else try runs.append(gpa, .{ .first = @intCast(i), .n = 1 });
            missing += 1;
        }
        if (!state and missing == 0) return .{ .result = .ok };
        const kk: u64 = keys.len;
        const charge_ = missing * t.block_bytes + (if (state) kk * 32 + t.slot_bytes else 0);
        const files = t.runFiles(missing) + (if (state) t.stateFile(kk) else 0);
        if (charge_ > t.queueRoom()) return t.skipped(.queue);
        if (t.freeBytes() < t.cfg.floor + files) return t.skipped(.floor);
        if (t.bytes + t.others_bytes + files > t.cfg.budget) {
            t.sweepOthers(files);
            if (t.cfg.follower) {
                if (t.bytes + t.others_bytes + files > t.cfg.budget + t.cfg.budget / 16) return t.skipped(.budget);
            } else {
                // this plan's blocks stay while others go (an older state of the same conversation shares them)
                for (keys) |k| if (t.blocks.getPtr(k)) |b| {
                    b.guard = true;
                };
                defer for (keys) |k| if (t.blocks.getPtr(k)) |b| {
                    b.guard = false;
                };
                while (t.bytes + t.others_bytes + files > t.cfg.budget) {
                    const v = t.victim() orelse break;
                    try victims.append(gpa, v);
                    t.dropState(v);
                    t.counts.evicted += 1;
                }
                if (t.bytes + t.others_bytes + files > t.cfg.budget) {
                    var p = t.skipped(.budget);
                    p.victims = try victims.toOwnedSlice(gpa);
                    return p;
                }
            }
        }
        // the plan stands: blocks pending, the state pending with its references, the queue reserved
        try t.blocks.ensureUnusedCapacity(gpa, @intCast(missing));
        if (state) try t.states.ensureUnusedCapacity(gpa, 1);
        const owned: []Key = if (state) try gpa.dupe(Key, keys) else &.{};
        const out_victims = victims.toOwnedSlice(gpa) catch |e| {
            gpa.free(owned);
            return e;
        };
        const out_runs = runs.toOwnedSlice(gpa) catch |e| {
            gpa.free(owned);
            gpa.free(out_victims);
            return e;
        };
        for (out_runs) |r| for (keys[r.first..][0..r.n]) |k| {
            t.blocks.putAssumeCapacity(k, .{ .bytes = head_bytes + t.block_bytes, .status = .pending });
            t.bytes += head_bytes + t.block_bytes;
        };
        if (pin) for (keys) |k| if (t.blocks.getPtr(k)) |b| {
            b.pinned = true;
        };
        if (state) {
            t.states.putAssumeCapacity(keys[keys.len - 1], .{ .k = @intCast(kk), .bytes = t.stateFile(kk), .used = t.tick(), .status = .pending, .keys = owned });
            t.bytes += t.stateFile(kk);
            for (keys) |k| t.blocks.getPtr(k).?.refs += 1;
        }
        _ = c.pthread_mutex_lock(&t.mu);
        t.queued += charge_;
        _ = c.pthread_mutex_unlock(&t.mu);
        return .{ .result = .ok, .victims = out_victims, .runs = out_runs, .charge = charge_ };
    }

    fn skipped(t: *Tier, why: Reason) Plan {
        t.counts.skipped += 1;
        return .{ .result = .{ .skip = why } };
    }

    /// A run's bytes (laid out by `planes`, plane-major over its n blocks) handed to the writer; it owns `data` now
    /// (gpa) and releases `charge` of the reservation when done. keys: the run's own keys; first_j: its first block's j.
    pub fn submitRun(t: *Tier, keys: []const Key, first_j: u32, data: []u8) !void {
        const owned = t.gpa.dupe(Key, keys) catch |e| {
            t.gpa.free(data);
            return e;
        };
        t.push(.{ .run = .{ .keys = owned, .first_j = first_j, .data = data, .charge = t.block_bytes * keys.len, .t0 = now() } });
    }

    /// The state's own bytes (gpa, owned now) after its runs; written once they are.
    pub fn submitState(t: *Tier, keys: []const Key, slot: []u8, new_blocks: u32) !void {
        const owned = t.gpa.dupe(Key, keys) catch |e| {
            t.gpa.free(slot);
            return e;
        };
        const kk = keys.len;
        t.push(.{ .state = .{ .key = keys[kk - 1], .k = @intCast(kk), .keys = owned, .slot = slot, .charge = kk * 32 + t.slot_bytes, .t0 = now(), .new_blocks = new_blocks, .new_bytes = new_blocks * (head_bytes + t.block_bytes) + t.stateFile(kk) } });
    }

    /// A plan whose bytes never came (a device copy failed): its reservation released, its state and blocks gone.
    pub fn abandon(t: *Tier, p: *const Plan, keys: []const Key, state: bool, submitted_runs: usize) void {
        var left = p.charge;
        for (p.runs[0..submitted_runs]) |r| left -|= t.block_bytes * r.n;
        _ = c.pthread_mutex_lock(&t.mu);
        t.queued -|= left;
        _ = c.pthread_mutex_unlock(&t.mu);
        if (state) t.dropState(keys[keys.len - 1]);
        for (p.runs[submitted_runs..]) |r| for (keys[r.first..][0..r.n]) |k| if (t.blocks.get(k)) |b| if (b.refs == 0) t.forgetBlock(k, null);
    }

    /// The prompt pass ended: its streamed blocks no state took go.
    pub fn endPass(t: *Tier) void {
        t.drain();
        var gone: std.ArrayList(Key) = .empty;
        defer gone.deinit(t.gpa);
        var it = t.blocks.iterator();
        while (it.next()) |e| if (e.value_ptr.pinned) {
            e.value_ptr.pinned = false;
            if (e.value_ptr.refs == 0) gone.append(t.gpa, e.key_ptr.*) catch {};
        };
        if (gone.items.len == 0) return;
        var del: std.ArrayList(Key) = .empty;
        for (gone.items) |k| t.forgetBlock(k, &del);
        t.pushDelete(&.{}, del.toOwnedSlice(t.gpa) catch return);
    }

    /// A block found missing or corrupt: every state on disk that lists it goes (each would fail the same way).
    pub fn dropBlock(t: *Tier, key: Key) usize {
        t.drain();
        var gone: std.ArrayList(Key) = .empty;
        defer gone.deinit(t.gpa);
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete) for (e.value_ptr.keys) |bk| if (std.mem.eql(u8, &bk, &key)) {
            gone.append(t.gpa, e.key_ptr.*) catch {};
            break;
        };
        for (gone.items) |k| t.dropState(k);
        t.line("event=drop-block states={d} key={s}", .{ gone.items.len, &short(key) });
        return gone.items.len;
    }

    /// States forgotten (evicted by the leader, a resume found bad, rank 0 lacks them): their files and the blocks only
    /// they held go.
    pub fn drop(t: *Tier, keys: []const Key) void {
        t.drain();
        for (keys) |k| t.dropState(k);
    }

    fn dropState(t: *Tier, key: Key) void {
        const kv = t.states.fetchRemove(key) orelse return;
        t.bytes -|= kv.value.bytes;
        var del: std.ArrayList(Key) = .empty;
        for (kv.value.keys) |bk| if (t.blocks.getPtr(bk)) |b| {
            b.refs -|= 1;
            if (b.refs == 0 and !b.pinned and !b.guard) t.forgetBlock(bk, &del);
        };
        t.gpa.free(kv.value.keys);
        const states = t.gpa.dupe(Key, &.{key}) catch {
            del.deinit(t.gpa);
            return;
        };
        t.pushDelete(states, del.toOwnedSlice(t.gpa) catch &.{});
    }

    fn forgetBlock(t: *Tier, key: Key, del: ?*std.ArrayList(Key)) void {
        const b = t.blocks.fetchRemove(key) orelse return;
        t.bytes -|= b.value.bytes;
        if (del) |d| d.append(t.gpa, key) catch {} else {
            const one = t.gpa.dupe(Key, &.{key}) catch return;
            t.pushDelete(&.{}, one);
        }
    }

    /// The least recently used state on disk (pending ones are never evicted).
    fn victim(t: *Tier) ?Key {
        var best: ?Key = null;
        var used: i128 = 0;
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete and (best == null or e.value_ptr.used < used)) {
            best = e.key_ptr.*;
            used = e.value_ptr.used;
        };
        return best;
    }

    /// Other namespaces removed, least recently opened first, until `files` more fit (or none is left).
    fn sweepOthers(t: *Tier, files: u64) void {
        while (t.others.items.len > 0 and t.bytes + t.others_bytes + files > t.cfg.budget) {
            var oldest: usize = 0;
            for (t.others.items, 0..) |o, i| if (o.used < t.others.items[oldest].used) {
                oldest = i;
            };
            const o = t.others.swapRemove(oldest);
            t.others_bytes -|= o.bytes;
            t.line("event=sweep ns={s} mib={d}", .{ &o.name, o.bytes >> 20 });
            t.push(.{ .sweep = .{ .name = o.name } });
        }
    }

    /// A state was resumed: the most recently used, here and in its file's mtime (the order after a restart).
    pub fn touch(t: *Tier, key: Key) void {
        const s = t.states.getPtr(key) orelse return;
        s.used = t.tick();
        var pb: [1400]u8 = undefined;
        touchPath(t.statePath(&pb, key, false) catch return);
    }

    fn tick(t: *Tier) i128 {
        t.clock = @max(t.clock + 1, now());
        return t.clock;
    }

    /// The states on disk (gpa), for the startup reconcile.
    pub fn completeKeys(t: *Tier) ![]Key {
        t.drain();
        var out: std.ArrayList(Key) = .empty;
        var it = t.states.iterator();
        while (it.next()) |e| if (e.value_ptr.status == .complete) try out.append(t.gpa, e.key_ptr.*);
        return out.toOwnedSlice(t.gpa);
    }

    pub fn hasComplete(t: *const Tier, key: Key) bool {
        return if (t.states.get(key)) |s| s.status == .complete else false;
    }

    // -- a resume (the engine thread, both ranks) -------------------------------------------------------------

    /// State `key` of k blocks read and verified: its block keys into `keys` (k of them) and its own bytes into
    /// `slot`. busy: still being written here.
    pub fn readState(t: *Tier, key: Key, k: u32, keys: []Key, slot: []u8) Load {
        t.drain();
        const rec = t.states.get(key) orelse return .bad;
        if (rec.status == .pending) return .busy;
        if (rec.k != k or keys.len != k or slot.len != t.slot_bytes) return .bad;
        var pb: [1400]u8 = undefined;
        const path = t.statePath(&pb, key, false) catch return .bad;
        const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
        if (fd < 0) return .bad;
        defer {
            _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
            _ = c.close(fd);
        }
        var hb: [head_bytes]u8 = undefined;
        if (!readAt(fd, &hb, 0)) return .bad;
        const h = Head.decode(&hb) catch return .bad;
        if (h.kind != .state or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &key) or h.index != k or h.rank != t.rank or h.payload != t.slot_bytes) return .bad;
        if (!readAt(fd, std.mem.sliceAsBytes(keys), head_bytes) or !readAt(fd, slot, head_bytes + 32 * @as(u64, k))) return .bad;
        var x = XxHash3.init(0);
        x.update(std.mem.sliceAsBytes(keys));
        x.update(slot);
        if (x.final() != h.sum or !std.mem.eql(u8, &keys[k - 1], &key)) return .bad;
        return .ok;
    }

    /// Blocks `keys` (block first_j onward) read and verified into `data`, laid out as a run (rangeOffset); `scratch`
    /// holds one block file. False when one is missing or corrupt (its key in `bad`, when given).
    pub fn readRun(t: *Tier, keys: []const Key, first_j: u32, data: []u8, scratch: []u8) bool {
        return t.readRunOr(keys, first_j, data, scratch, null);
    }

    pub fn readRunOr(t: *Tier, keys: []const Key, first_j: u32, data: []u8, scratch: []u8, bad: ?*Key) bool {
        const n = keys.len;
        if (data.len < n * t.block_bytes or scratch.len < head_bytes + t.block_bytes) return false;
        for (keys, 0..) |k, i| {
            if (bad) |b| b.* = k;
            var pb: [1400]u8 = undefined;
            const path = t.blockPath(&pb, k, false) catch return false;
            const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
            if (fd < 0) return false;
            const file = scratch[0 .. head_bytes + t.block_bytes];
            const got = readAt(fd, file, 0);
            var extra: [1]u8 = undefined;
            const longer = readAt(fd, &extra, file.len);
            _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
            _ = c.close(fd);
            if (!got or longer) return false;
            const h = Head.decode(file[0..head_bytes]) catch return false;
            const payload = file[head_bytes..];
            if (h.kind != .block or !std.mem.eql(u8, &h.ns, &t.ns) or !std.mem.eql(u8, &h.key, &k) or h.index != first_j + i or h.rank != t.rank or h.payload != t.block_bytes) return false;
            if (XxHash3.hash(0, payload) != h.sum) return false;
            var off: u64 = 0;
            for (t.planes, 0..) |pbytes, p| {
                const dst = rangeOffset(t.planes, n, p) + i * pbytes;
                @memcpy(data[dst..][0..pbytes], payload[off..][0..pbytes]);
                off += pbytes;
            }
        }
        return true;
    }

    // -- paths ------------------------------------------------------------------------------------------------

    fn blockPath(t: *const Tier, buf: []u8, k: Key, tmp: bool) ![:0]u8 {
        const hex = std.fmt.bytesToHex(k, .lower);
        return std.fmt.bufPrintSentinel(buf, "{s}/b/{s}/{s}.blk{s}", .{ t.dir, hex[0..2], &hex, if (tmp) ".tmp" else "" }, 0);
    }

    fn statePath(t: *const Tier, buf: []u8, k: Key, tmp: bool) ![:0]u8 {
        const hex = std.fmt.bytesToHex(k, .lower);
        return std.fmt.bufPrintSentinel(buf, "{s}/s/{s}.st{s}", .{ t.dir, &hex, if (tmp) ".tmp" else "" }, 0);
    }

    // -- the writer thread ------------------------------------------------------------------------------------

    fn push(t: *Tier, j: Job) void {
        _ = c.pthread_mutex_lock(&t.mu);
        defer _ = c.pthread_mutex_unlock(&t.mu);
        t.jobs.append(t.gpa, j) catch {
            // no room for the job itself: it fails at once (its bytes freed, its reservation released)
            var x = j;
            t.queued -|= charge(&x);
            t.done.append(t.gpa, .{ .job = x, .ok = false }) catch freeJob(t.gpa, &x);
            return;
        };
        _ = c.pthread_cond_signal(&t.cv);
    }

    fn pushDelete(t: *Tier, states: []Key, blocks: []Key) void {
        if (states.len == 0 and blocks.len == 0) return;
        t.push(.{ .delete = .{ .states = states, .blocks = blocks } });
    }

    fn writer(t: *Tier) void {
        lowerPriority();
        while (true) {
            _ = c.pthread_mutex_lock(&t.mu);
            while (t.head == t.jobs.items.len and !t.stop) _ = c.pthread_cond_wait(&t.cv, &t.mu);
            if (t.head == t.jobs.items.len or (t.stop and now() > t.deadline)) {
                _ = c.pthread_mutex_unlock(&t.mu);
                return;
            }
            var j = t.jobs.items[t.head];
            t.running = true;
            t.head += 1;
            if (t.head == t.jobs.items.len) {
                t.jobs.clearRetainingCapacity();
                t.head = 0;
            }
            _ = c.pthread_mutex_unlock(&t.mu);
            var reason: Reason = .io;
            const ok = t.run(&j, &reason);
            // the bytes held while queued go now; the job's record stays for drain
            const ch = charge(&j);
            freePayload(t.gpa, &j);
            _ = c.pthread_mutex_lock(&t.mu);
            t.queued -|= ch;
            t.running = false;
            t.done.append(t.gpa, .{ .job = j, .ok = ok, .reason = reason }) catch freeJob(t.gpa, &j);
            _ = c.pthread_mutex_unlock(&t.mu);
        }
    }

    fn run(t: *Tier, j: *Job, reason: *Reason) bool {
        switch (j.*) {
            .run => |r| {
                const n = r.keys.len;
                var bytes: u64 = 0;
                for (r.keys, 0..) |k, i| {
                    if (t.freeBytes() < t.cfg.floor + head_bytes + t.block_bytes) {
                        reason.* = .floor;
                        t.line("event=blocks result=failed reason=floor count={d}", .{n});
                        return false;
                    }
                    var parts: [64][]const u8 = undefined;
                    if (t.planes.len > parts.len) return false;
                    for (t.planes, 0..) |pbytes, p| parts[p] = r.data[rangeOffset(t.planes, n, p) + i * pbytes ..][0..pbytes];
                    t.writeBlock(k, r.first_j + @as(u32, @intCast(i)), parts[0..t.planes.len]) catch |e| {
                        t.line("event=blocks result=failed reason={s} count={d}", .{ @errorName(e), n });
                        return false;
                    };
                    bytes += head_bytes + t.block_bytes;
                }
                t.syncBlockDirs(r.keys);
                _ = t.written.fetchAdd(bytes, .monotonic);
                t.line("event=blocks result=written tokens={d} count={d} bytes={d} ms={d}", .{ (r.first_j - 1) * t.cfg.block_tokens, n, bytes, ms(now() - r.t0) });
                return true;
            },
            .state => |s| {
                // its blocks were written before it (the queue's order): each must be here, whole
                for (s.keys) |k| {
                    var pb: [1400]u8 = undefined;
                    const st = statPath(t.blockPath(&pb, k, false) catch return false) orelse {
                        reason.* = .io;
                        t.line("event=save result=failed reason=block-missing tokens={d} key={s}", .{ s.k * t.cfg.block_tokens, &short(s.key) });
                        return false;
                    };
                    if (st.size != head_bytes + t.block_bytes) return false;
                }
                const file = t.stateFile(s.k);
                if (t.freeBytes() < t.cfg.floor + file) {
                    reason.* = .floor;
                    t.line("event=save result=failed reason=floor tokens={d} key={s}", .{ s.k * t.cfg.block_tokens, &short(s.key) });
                    return false;
                }
                t.writeState(s.key, s.k, s.keys, s.slot) catch |e| {
                    t.line("event=save result=failed reason={s} tokens={d} key={s}", .{ @errorName(e), s.k * t.cfg.block_tokens, &short(s.key) });
                    return false;
                };
                _ = t.written.fetchAdd(file, .monotonic);
                t.line("event=save result=written tokens={d} blocks={d} new_blocks={d} bytes={d} ms={d} key={s}", .{ s.k * t.cfg.block_tokens, s.k, s.new_blocks, s.new_bytes, ms(now() - s.t0), &short(s.key) });
                return true;
            },
            .delete => |d| {
                for (d.states) |k| {
                    var pb: [1400]u8 = undefined;
                    _ = c.unlink(t.statePath(&pb, k, false) catch continue);
                }
                for (d.blocks) |k| {
                    var pb: [1400]u8 = undefined;
                    _ = c.unlink(t.blockPath(&pb, k, false) catch continue);
                }
                return true;
            },
            .sweep => |s| {
                var pb: [1100]u8 = undefined;
                const path = std.fmt.bufPrint(&pb, "{s}/{s}", .{ t.cfg.root, &s.name }) catch return false;
                _ = treeBytes(path, true);
                return true;
            },
        }
    }

    fn writeBlock(t: *Tier, k: Key, j: u32, parts: []const []const u8) !void {
        var x = XxHash3.init(0);
        for (parts) |p| x.update(p);
        var hb: [head_bytes]u8 = undefined;
        (Head{ .kind = .block, .ns = t.ns, .key = k, .index = j, .rank = t.rank, .payload = t.block_bytes, .sum = x.final() }).encode(&hb);
        var tb: [1400]u8 = undefined;
        var fb: [1400]u8 = undefined;
        const tmp = try t.blockPath(&tb, k, true);
        const hex = std.fmt.bytesToHex(k, .lower);
        var db: [1200]u8 = undefined;
        const d = try std.fmt.bufPrintSentinel(&db, "{s}/b/{s}", .{ t.dir, hex[0..2] }, 0);
        const rc = c.mkdir(d, 0o700);
        if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
        var all: [65][]const u8 = undefined;
        all[0] = &hb;
        @memcpy(all[1..][0..parts.len], parts);
        try t.writeThrottled(tmp, all[0 .. parts.len + 1]);
        if (c.rename(tmp, try t.blockPath(&fb, k, false)) != 0) {
            _ = c.unlink(tmp);
            return error.DiskWrite;
        }
    }

    fn writeState(t: *Tier, key: Key, k: u32, keys: []const Key, slot: []const u8) !void {
        var x = XxHash3.init(0);
        x.update(std.mem.sliceAsBytes(keys));
        x.update(slot);
        var hb: [head_bytes]u8 = undefined;
        (Head{ .kind = .state, .ns = t.ns, .key = key, .index = k, .rank = t.rank, .payload = t.slot_bytes, .sum = x.final() }).encode(&hb);
        var tb: [1400]u8 = undefined;
        var fb: [1400]u8 = undefined;
        const tmp = try t.statePath(&tb, key, true);
        try t.writeThrottled(tmp, &.{ &hb, std.mem.sliceAsBytes(keys), slot });
        if (c.rename(tmp, try t.statePath(&fb, key, false)) != 0) {
            _ = c.unlink(tmp);
            return error.DiskWrite;
        }
        var db: [1200]u8 = undefined;
        syncDir(std.fmt.bufPrintSentinel(&db, "{s}/s", .{t.dir}, 0) catch return);
    }

    /// `parts` into a new file at `path`, in pieces under the rate cap, then fdatasync and out of the page cache.
    fn writeThrottled(t: *Tier, path: [:0]const u8, parts: []const []const u8) !void {
        const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
        if (fd < 0) return error.DiskWrite;
        var ok = false;
        defer {
            _ = c.close(fd);
            if (!ok) _ = c.unlink(path);
        }
        const t0 = now();
        var done: u64 = 0;
        for (parts) |p| {
            var off: usize = 0;
            while (off < p.len) {
                const n = @min(piece_bytes, p.len - off);
                try writeAll(fd, p[off..][0..n]);
                off += n;
                done += n;
                if (t.cfg.rate > 0) {
                    const want: i128 = @intCast(done * std.time.ns_per_s / t.cfg.rate);
                    const ahead = want - (now() - t0);
                    if (ahead > std.time.ns_per_ms) sleepNs(@intCast(ahead));
                }
            }
        }
        if (c.fdatasync(fd) != 0) return error.DiskWrite;
        _ = linux.fadvise(fd, 0, 0, linux.POSIX_FADV.DONTNEED);
        ok = true;
    }

    fn syncBlockDirs(t: *Tier, keys: []const Key) void {
        var seen: [256]bool = @splat(false);
        for (keys) |k| {
            if (seen[k[0]]) continue;
            seen[k[0]] = true;
            const hex = std.fmt.bytesToHex([1]u8{k[0]}, .lower);
            var db: [1200]u8 = undefined;
            syncDir(std.fmt.bufPrintSentinel(&db, "{s}/b/{s}", .{ t.dir, &hex }, 0) catch continue);
        }
    }

    /// One log line, "sf-disk: rank=R " and the event's fields (tests stay quiet).
    pub fn line(t: *const Tier, comptime fmt: []const u8, args: anytype) void {
        if (builtin.is_test) return;
        std.log.info("sf-disk: rank={d} " ++ fmt, .{t.rank} ++ args);
    }
};

fn charge(j: *const Job) u64 {
    return switch (j.*) {
        .run => |r| r.charge,
        .state => |s| s.charge,
        else => 0,
    };
}

fn freePayload(gpa: Allocator, j: *Job) void {
    switch (j.*) {
        .run => |*r| {
            gpa.free(r.data);
            r.data = &.{};
        },
        .state => |*s| {
            gpa.free(s.slot);
            s.slot = &.{};
        },
        else => {},
    }
}

fn freeJob(gpa: Allocator, j: *Job) void {
    freePayload(gpa, j);
    switch (j.*) {
        .run => |r| gpa.free(r.keys),
        .state => |s| gpa.free(s.keys),
        .delete => |d| {
            gpa.free(d.states);
            gpa.free(d.blocks);
        },
        .sweep => {},
    }
}

/// The keys of the reconcile's other side, as a bitmap of `have` over `keys` (bit i: keys[i] is on disk here).
pub fn bitmap(gpa: Allocator, t: *const Tier, keys: []const Key) ![]u8 {
    const out = try gpa.alloc(u8, (keys.len + 7) / 8);
    @memset(out, 0);
    for (keys, 0..) |k, i| if (t.hasComplete(k)) {
        out[i / 8] |= @as(u8, 1) << @intCast(i % 8);
    };
    return out;
}

pub fn bit(map: []const u8, i: usize) bool {
    return i / 8 < map.len and map[i / 8] & (@as(u8, 1) << @intCast(i % 8)) != 0;
}

/// The keys `key=value` lines of `theirs` that differ from `ours` (or are missing from either), comma separated in
/// `buf` ("none" when the texts agree line for line).
pub fn identityDiff(buf: []u8, ours: []const u8, theirs: []const u8) []const u8 {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, ours, '\n');
    while (it.next()) |l| {
        if (l.len == 0) continue;
        const name = l[0 .. std.mem.indexOfScalar(u8, l, '=') orelse l.len];
        if (lineOf(theirs, name)) |o| if (std.mem.eql(u8, o, l)) continue;
        n = appendName(buf, n, name);
    }
    var jt = std.mem.splitScalar(u8, theirs, '\n');
    while (jt.next()) |l| {
        if (l.len == 0) continue;
        const name = l[0 .. std.mem.indexOfScalar(u8, l, '=') orelse l.len];
        if (lineOf(ours, name) == null) n = appendName(buf, n, name);
    }
    return if (n == 0) "none" else buf[0..n];
}

fn lineOf(text: []const u8, name: []const u8) ?[]const u8 {
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |l| if (l.len > name.len and l[name.len] == '=' and std.mem.startsWith(u8, l, name)) return l else if (std.mem.eql(u8, l, name)) return l;
    return null;
}

fn appendName(buf: []u8, n: usize, name: []const u8) usize {
    const need = name.len + @intFromBool(n > 0);
    if (n + need > buf.len) return n;
    var m = n;
    if (m > 0) {
        buf[m] = ',';
        m += 1;
    }
    @memcpy(buf[m..][0..name.len], name);
    return m + name.len;
}

// -- small file helpers ---------------------------------------------------------------------------------------------

fn allHex(s: []const u8) bool {
    for (s) |ch| if (!std.ascii.isHex(ch)) return false;
    return true;
}

fn keyName(name: []const u8, ext: []const u8) ?Key {
    if (name.len != 64 + ext.len or !std.mem.endsWith(u8, name, ext)) return null;
    var k: Key = undefined;
    _ = std.fmt.hexToBytes(&k, name[0..64]) catch return null;
    return k;
}

/// A directory's entries, "." and ".." skipped.
const Entries = struct {
    d: *c.DIR,

    fn open(path: []const u8) ?Entries {
        var buf: [1100]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&buf, "{s}", .{path}, 0) catch return null;
        return .{ .d = c.opendir(p) orelse return null };
    }

    fn next(e: *Entries) ?[]const u8 {
        while (c.readdir(e.d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) continue;
            return name;
        }
        return null;
    }

    fn close(e: *Entries) void {
        _ = c.closedir(e.d);
    }
};

const FileStat = struct { size: u64, mtime: i128 };

fn statPath(path: [:0]const u8) ?FileStat {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = c.close(fd);
    return statFd(fd);
}

fn statFd(fd: c_int) ?FileStat {
    var sx: linux.Statx = undefined;
    const at_empty_path: u32 = 0x1000;
    if (linux.errno(linux.statx(fd, "", at_empty_path, .{ .SIZE = true, .MTIME = true }, &sx)) != .SUCCESS) return null;
    return .{ .size = sx.size, .mtime = @as(i128, sx.mtime.sec) * std.time.ns_per_s + sx.mtime.nsec };
}

/// The bytes of the regular files under `path` (two levels: a namespace's s/, b/ and b/xx/), each removed with the
/// directories when `remove`.
fn treeBytes(path: []const u8, remove: bool) u64 {
    var total: u64 = 0;
    var it = Entries.open(path) orelse return 0;
    while (it.next()) |name| {
        var pb: [1400]u8 = undefined;
        const p = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ path, name }, 0) catch continue;
        if (c.opendir(p)) |d| {
            _ = c.closedir(d);
            total += treeBytes(p, remove); // (removes p itself when `remove`)
        } else if (statPath(p)) |st| {
            total += st.size;
            if (remove) _ = c.unlink(p);
        }
    }
    it.close();
    if (remove) {
        var pb: [1400]u8 = undefined;
        _ = c.rmdir(std.fmt.bufPrintSentinel(&pb, "{s}", .{path}, 0) catch return total);
    }
    return total;
}

fn readSmall(gpa: Allocator, path: [:0]const u8) ?[]u8 {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return null;
    defer _ = c.close(fd);
    const st = statFd(fd) orelse return null;
    if (st.size > 1 << 20) return null;
    const out = gpa.alloc(u8, @intCast(st.size)) catch return null;
    if (!readAt(fd, out, 0)) {
        gpa.free(out);
        return null;
    }
    return out;
}

fn writeFile(path: [:0]const u8, parts: []const []const u8) !void {
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o600));
    if (fd < 0) return error.DiskWrite;
    defer _ = c.close(fd);
    for (parts) |p| try writeAll(fd, p);
    if (c.fdatasync(fd) != 0) return error.DiskWrite;
}

fn touchPath(path: [:0]const u8) void {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return;
    defer _ = c.close(fd);
    _ = c.futimens(fd, null);
}

fn syncDir(path: [:0]const u8) void {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(c.mode_t, 0));
    if (fd < 0) return;
    defer _ = c.close(fd);
    _ = c.fsync(fd);
}

fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return error.DiskWrite;
        }
        done += @intCast(n);
    }
}

fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) {
            if (n < 0 and c.errno(n) == .INTR) continue;
            return false;
        }
        done += @intCast(n);
    }
    return true;
}

/// `dir` and its parents (mode 0700).
fn makePath(gpa: Allocator, dir: [:0]const u8) !void {
    const tmp = try gpa.dupeSentinel(u8, dir, 0);
    defer gpa.free(tmp);
    for (tmp[1..], 1..) |ch, i| if (ch == '/') {
        tmp[i] = 0;
        _ = c.mkdir(tmp.ptr, 0o700);
        tmp[i] = '/';
    };
    const rc = c.mkdir(tmp.ptr, 0o700);
    if (rc != 0 and c.errno(rc) != .EXIST) return error.DiskDir;
}

const Statvfs = extern struct { bsize: c_ulong, frsize: c_ulong, blocks: u64, bfree: u64, bavail: u64, files: u64, ffree: u64, favail: u64, fsid: c_ulong, flag: c_ulong, namemax: c_ulong, spare: [8]c_int };
extern "c" fn statvfs(path: [*:0]const u8, buf: *Statvfs) c_int;

/// The bytes an unprivileged writer may still use on `path`'s filesystem (0 when it cannot tell: no writes).
pub fn fsFree(path: [:0]const u8) u64 {
    var s: Statvfs = undefined;
    if (statvfs(path, &s) != 0) return 0;
    return s.bavail * s.frsize;
}

/// Wall-clock ns (a state file's mtime is the same clock).
fn now() i128 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(c.CLOCK.REALTIME, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn ms(ns: i128) u64 {
    return @intCast(@max(0, @divTrunc(ns, std.time.ns_per_ms)));
}

fn sleepNs(ns: u64) void {
    const ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = c.nanosleep(&ts, null);
}

/// The writer thread in the idle I/O class and at nice 10: a decode round's Engram reads go first.
fn lowerPriority() void {
    const idle_class: usize = 3 << 13;
    _ = linux.syscall3(.ioprio_set, 1, 0, idle_class); // IOPRIO_WHO_PROCESS, this thread
    _ = linux.syscall3(.setpriority, 0, @intCast(linux.gettid()), 10); // PRIO_PROCESS on a thread id: this thread
}

test {
    _ = @import("prompt_disk_test.zig");
}

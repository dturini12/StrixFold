//! sf-disk for dsv41: kept prompt states (slot bytes, the extent's compressed rows in blocks) on disk on both TP ranks.
const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("gpu");
const core = @import("core");
const model = @import("model.zig");
const identity_mod = @import("disk_identity.zig");
const pd = core.prompt_disk;
const image_key = @import("lanes").image_key;
const Key = pd.Key;
const Model = model.Model;

/// A block's positions: chunk_rows, shared by every state of a conversation.
pub const block_tokens = model.chunk_rows;

/// TF_DS_PROMPT_DISK_*, the same on both ranks (serve_lib.sh); the handshake carries " disk=1" when on.
pub const Settings = struct {
    dir: []const u8, // _DIR (default $HOME/strixfold-cache/prefix): one directory per identity below it
    budget: u64, // _GIB: disk for kept states on this box (unset, empty or 0: off)
    floor: u64 = 16 << 30, // _FLOOR_GIB: no writes while the filesystem has less free
    queue: u64 = 1 << 30, // _QUEUE_MIB: host memory for states waiting to be written
    min_tokens: u32 = 8192, // _MIN_TOKENS: shorter states stay in memory only
    rate: u64 = 1 << 30, // _RATE_MIB: the writer's bytes a second at most (0: no cap)

    /// The tier's settings from the environment; null when off (or a value is not a number: one log line).
    pub fn fromEnv(a: std.mem.Allocator) ?Settings {
        return parse(a, envVar) catch |e| {
            note("event=off reason=settings ({s})", .{@errorName(e)});
            return null;
        };
    }

    pub fn parse(a: std.mem.Allocator, get: *const fn ([:0]const u8) ?[]const u8) !?Settings {
        const gib = try number(f64, get("TF_DS_PROMPT_DISK_GIB")) orelse return null;
        if (!(gib > 0)) return null;
        var s: Settings = .{ .dir = undefined, .budget = @intFromFloat(gib * (1 << 30)) };
        s.dir = if (get("TF_DS_PROMPT_DISK_DIR")) |d| (if (d.len > 0) try a.dupe(u8, d) else null) orelse try defaultDir(a, get) else try defaultDir(a, get);
        if (try number(f64, get("TF_DS_PROMPT_DISK_FLOOR_GIB"))) |v| s.floor = @intFromFloat(@max(v, 0) * (1 << 30));
        if (try number(u64, get("TF_DS_PROMPT_DISK_QUEUE_MIB"))) |v| s.queue = v << 20;
        if (try number(u32, get("TF_DS_PROMPT_DISK_MIN_TOKENS"))) |v| s.min_tokens = @max(v, block_tokens);
        if (try number(u64, get("TF_DS_PROMPT_DISK_RATE_MIB"))) |v| s.rate = v << 20;
        return s;
    }

    fn defaultDir(a: std.mem.Allocator, get: *const fn ([:0]const u8) ?[]const u8) ![]const u8 {
        return std.fmt.allocPrint(a, "{s}/strixfold-cache/prefix", .{get("HOME") orelse return error.NoHome});
    }
};

fn number(comptime T: type, text: ?[]const u8) !?T {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    return switch (@typeInfo(T)) {
        .float => std.fmt.parseFloat(T, t) catch error.BadSetting,
        else => std.fmt.parseInt(T, t, 10) catch error.BadSetting,
    };
}

pub fn envVar(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// Whether this build and launch may run the tier (the handshake's " disk=1"): HIP, TP2, the settings on.
pub fn wanted(world: u32) bool {
    if (gpu.backend != .hip or world != 2) return false;
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    return Settings.fromEnv(fba.allocator()) != null;
}

/// One plane of a block: where its rows are, a row's bytes, the layer's ratio.
pub const Plane = struct { base: u64, row: usize, ratio: usize };

/// Every plane copyExtent moves, in its order: each kv-source layer's codes and scales, then its index keys'.
pub fn planesOf(gpa: std.mem.Allocator, m: *const Model) ![]Plane {
    var out: std.ArrayList(Plane) = .empty;
    errdefer out.deinit(gpa);
    const c = m.eng.c;
    const cs = &m.caches;
    for (m.eng.w.layers, 0..) |lay, i| {
        if (lay.comp_wkv == null) continue;
        const r: usize = lay.ratio;
        if (r == 0 or block_tokens % r != 0) return error.BadRatio;
        try out.append(gpa, .{ .base = cs.comp_codes[i], .row = c.head_dim / 2, .ratio = r });
        try out.append(gpa, .{ .base = cs.comp_scales[i], .row = c.head_dim / 16, .ratio = r });
        if (lay.idx_wk != null) {
            try out.append(gpa, .{ .base = cs.idx_codes[i], .row = c.index_head_dim / 2, .ratio = r });
            try out.append(gpa, .{ .base = cs.idx_scales[i], .row = c.index_head_dim / 32, .ratio = r });
        }
    }
    return out.toOwnedSlice(gpa);
}

/// A plane's bytes in one block (block_tokens positions).
pub fn blockBytes(p: Plane) u64 {
    return block_tokens / p.ratio * p.row;
}

/// The device address of block j's (1-based) rows of plane p in the extent at `base`.
pub fn rowsAt(p: Plane, base: usize, j: usize) u64 {
    return p.base + (base + (j - 1) * block_tokens) / p.ratio * p.row;
}

pub const Parts = identity_mod.Parts;
pub const identityText = identity_mod.identityText;
pub const seedText = identity_mod.seedText;

/// pd.chain with each block's pictures: H_j also hashes (at, len, whole digest) of every span that meets block j.
pub fn chainKeys(h0: Key, ids: []const u8, spans: []const image_key.KeySpan, out: []Key) void {
    if (spans.len == 0) return pd.chain(h0, ids, block_tokens * 4, out);
    std.debug.assert(out.len * block_tokens * 4 <= ids.len);
    var prev = h0;
    for (out, 0..) |*o, i| {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        h.update(&prev);
        h.update(ids[i * block_tokens * 4 ..][0 .. block_tokens * 4]);
        for (spans) |sp| if (sp.at < (i + 1) * block_tokens and @as(usize, sp.at) + sp.len > i * block_tokens) {
            var b: [8]u8 = undefined;
            std.mem.writeInt(u32, b[0..4], sp.at, .little);
            std.mem.writeInt(u32, b[4..8], sp.len, .little);
            h.update(&b);
            h.update(&sp.digest);
        };
        o.* = h.finalResult();
        prev = o.*;
    }
}

/// One rank's tier: the store, the planes, the current prompt pass's chain of keys, and buffers for a resume.
pub const Disk = struct {
    gpa: std.mem.Allocator,
    m: *Model,
    tier: *pd.Tier,
    settings: Settings,
    planes: []Plane,
    plane_bytes: []u64,
    h0: Key,
    scratch: gpu.DeviceBuffer, // a resume's slot bytes on the device, then snapCopy into the slot
    chain: std.ArrayList(Key) = .empty, // the pass's H_1 .. (every whole block of its prompt)
    passing: bool = false, // a prompt pass is running (its streamed blocks pinned until it ends)
    streaming: bool = false, // rank 0: this pass writes its blocks as its chunks finish
    streamed: u32 = 0, // rank 0: blocks 1 .. streamed are planned in this pass
    stream_skips: u64 = 0,
    peer_bad: u32 = 0, // rank 0: resumes in a row rank 1 found bad (10: the tier goes off)
    off: bool = false,

    /// The tier for this rank (owning s.dir from here), or null with one log line: it then stays off.
    pub fn open(gpa: std.mem.Allocator, m: *Model, s: Settings) ?*Disk {
        return openOr(gpa, m, s) catch |e| {
            note("event=off rank={d} reason={s} dir={s}", .{ m.rank, @errorName(e), s.dir });
            gpa.free(s.dir);
            return null;
        };
    }

    fn openOr(gpa: std.mem.Allocator, m: *Model, s: Settings) !*Disk {
        if (gpu.backend != .hip or m.world != 2) return error.NotTp2Hip;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const planes = try planesOf(gpa, m);
        errdefer gpa.free(planes);
        const pb = try gpa.alloc(u64, planes.len);
        errdefer gpa.free(pb);
        for (planes, pb) |p, *b| b.* = blockBytes(p);
        const parts = try identity_mod.identity(arena.allocator(), m, planes);
        const id = try identityText(arena.allocator(), parts);
        const seed_text = try seedText(arena.allocator(), parts);
        const tier = try pd.Tier.open(gpa, .{ .root = s.dir, .budget = s.budget, .floor = s.floor, .queue = s.queue, .rate = s.rate, .follower = m.rank != 0, .block_tokens = block_tokens }, id, m.rank, m.snapBytes(), pb);
        errdefer tier.close();
        var scratch = try gpu.DeviceBuffer.alloc(m.ctx.d, m.snapBytes());
        errdefer scratch.free();
        const d = try gpa.create(Disk);
        d.* = .{ .gpa = gpa, .m = m, .tier = tier, .settings = s, .planes = planes, .plane_bytes = pb, .h0 = pd.seed(seed_text), .scratch = scratch };
        tier.line("event=open ns={s} states={d} blocks={d} mib={d} others_mib={d} budget_gib={d} block_bytes={d} slot_bytes={d} dir={s}", .{ &tier.ns_hex, tier.states.count(), tier.blocks.count(), tier.bytes >> 20, tier.others_bytes >> 20, s.budget >> 30, tier.block_bytes, tier.slot_bytes, s.dir });
        return d;
    }

    pub fn close(d: *Disk) void {
        d.tier.close();
        d.gpa.free(d.settings.dir);
        d.scratch.free();
        d.chain.deinit(d.gpa);
        d.gpa.free(d.planes);
        d.gpa.free(d.plane_bytes);
        d.gpa.destroy(d);
    }

    /// A prompt pass begins on key ids `ids` with pictures `spans` (rank 1: as rank 0 sent them): its blocks' keys.
    pub fn beginPass(d: *Disk, ids: []const u8, spans: []const image_key.KeySpan) void {
        const n = ids.len / 4 / block_tokens;
        d.chain.resize(d.gpa, n) catch {
            d.chain.clearRetainingCapacity();
            return;
        };
        chainKeys(d.h0, ids, spans, d.chain.items);
        d.passing = true;
        d.streaming = false;
        d.streamed = 0;
    }

    /// The pass ended: its streamed blocks no state took go.
    pub fn endPass(d: *Disk) void {
        if (!d.passing) return;
        d.passing = false;
        d.streaming = false;
        d.tier.endPass();
    }

    /// Blocks j0 .. j1 - 1 (1-based) of the extent at `base`, plane by plane, into host memory (gpa): one copy a plane.
    fn copyRun(d: *Disk, base: usize, j0: usize, n: usize) ![]u8 {
        const total = n * pd.sum(d.plane_bytes);
        const data = try d.gpa.alloc(u8, total);
        errdefer d.gpa.free(data);
        for (d.planes, d.plane_bytes, 0..) |p, b, i| {
            const off = pd.rangeOffset(d.plane_bytes, n, i);
            try gpu.copyD2H(d.m.ctx.d, data[off..][0 .. n * b], rowsAt(p, base, j0));
        }
        return data;
    }

    /// Writes what `plan` reserved: its runs from the extent at `base`, then the state's bytes from `slot_src` if set.
    pub fn write(d: *Disk, p: *const pd.Plan, keys: []const Key, first_j: u32, base: usize, slot_src: ?u64) bool {
        // the device copies wait for the stream (the chunks, the save's snapCopy); false: nothing more is written
        d.m.stream.synchronize() catch |e| return d.abandon(p, keys, slot_src != null, 0, e);
        const Copy = struct {
            d: *Disk,
            base: usize,
            fn f(c: @This(), j0: u32, n: u32) ![]u8 {
                return c.d.copyRun(c.base, j0, n);
            }
        };
        if (!submitRuns(d.tier, p, keys, first_j, slot_src != null, Copy{ .d = d, .base = base }, Copy.f)) return false;
        const src = slot_src orelse return true;
        const slot = d.gpa.alloc(u8, d.tier.slot_bytes) catch |e| return d.abandon(p, keys, true, p.runs.len, e);
        gpu.copyD2H(d.m.ctx.d, slot, src) catch |e| {
            d.gpa.free(slot);
            return d.abandon(p, keys, true, p.runs.len, e);
        };
        var new: u32 = 0;
        for (p.runs) |r| new += r.n;
        d.tier.submitState(keys, slot, new) catch |e| return d.abandon(p, keys, true, p.runs.len, e);
        return true;
    }

    fn abandon(d: *Disk, p: *const pd.Plan, keys: []const Key, state: bool, submitted: usize, e: anyerror) bool {
        return abandonPlan(d.tier, p, keys, state, submitted, e);
    }

    /// State `key` of k blocks read and verified into slot `slot` and the extent at `base` (else a pass from 0).
    pub fn load(d: *Disk, slot: usize, base: usize, key: Key, k: u32, bytes: *u64) pd.Load {
        return d.loadOr(slot, base, key, k, bytes) catch |e| {
            d.tier.line("event=load result=bad reason={s} blocks={d} key={s}", .{ @errorName(e), k, &pd.short(key) });
            return .bad;
        };
    }

    fn loadOr(d: *Disk, slot: usize, base: usize, key: Key, k: u32, bytes: *u64) !pd.Load {
        if (k == 0 or k > d.chain.items.len or !std.mem.eql(u8, &d.chain.items[k - 1], &key)) return error.NotThisPrompt;
        try d.m.stream.synchronize();
        const gpa = d.gpa;
        const keys = try gpa.alloc(Key, k);
        defer gpa.free(keys);
        const slot_bytes = try gpa.alloc(u8, d.tier.slot_bytes);
        defer gpa.free(slot_bytes);
        const st = d.tier.readState(key, k, keys, slot_bytes);
        if (st != .ok) return st;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(keys), std.mem.sliceAsBytes(d.chain.items[0..k]))) return error.KeysDiffer;
        const group = 32;
        const per = pd.sum(d.plane_bytes);
        const data = try gpa.alloc(u8, @min(k, group) * per);
        defer gpa.free(data);
        const file = try gpa.alloc(u8, pd.head_bytes + per);
        defer gpa.free(file);
        var j0: u32 = 0;
        while (j0 < k) {
            const n = @min(group, k - j0);
            var bad: Key = undefined;
            if (!d.tier.readRunOr(keys[j0..][0..n], j0 + 1, data, file, &bad)) {
                _ = d.tier.dropBlock(bad); // (every state listing it would fail the same way)
                return .bad;
            }
            for (d.planes, d.plane_bytes, 0..) |p, b, i| {
                const off = pd.rangeOffset(d.plane_bytes, n, i);
                try gpu.copyH2D(d.m.ctx.d, rowsAt(p, base, j0 + 1), data[off..][0 .. n * b]);
            }
            j0 += n;
        }
        try gpu.copyH2D(d.m.ctx.d, d.scratch.ptr, slot_bytes);
        try d.m.snapCopy(slot, d.scratch.ptr, false);
        bytes.* = k * (pd.head_bytes + per) + d.tier.stateFile(k);
        return .ok;
    }

    /// The pair's outcome of a resume of `key`: a hit makes it the most recently used, a forget removes it here.
    pub fn finish(d: *Disk, key: Key, out: pd.Outcome) void {
        switch (out) {
            .hit => {
                d.tier.touch(key);
                d.tier.counts.hits += 1;
            },
            .miss => d.tier.counts.misses += 1,
            .forget => {
                d.tier.drop(&.{key});
                d.tier.counts.forgets += 1;
            },
        }
    }

    pub fn line(d: *const Disk, comptime fmt: []const u8, args: anytype) void {
        d.tier.line(fmt, args);
    }
};

/// The plan's runs to the writer, each from `copy(ctx, j, n)`; a failure abandons it with the runs queued (false).
pub fn submitRuns(tier: *pd.Tier, p: *const pd.Plan, keys: []const Key, first_j: u32, state: bool, ctx: anytype, comptime copy: anytype) bool {
    for (p.runs, 0..) |r, i| {
        const data = copy(ctx, first_j + r.first, r.n) catch |e| return abandonPlan(tier, p, keys, state, i, e);
        tier.submitRun(keys[r.first..][0..r.n], first_j + r.first, data) catch |e| return abandonPlan(tier, p, keys, state, i, e);
    }
    return true;
}

fn abandonPlan(tier: *pd.Tier, p: *const pd.Plan, keys: []const Key, state: bool, submitted: usize, e: anyerror) bool {
    tier.line("event=save result=failed reason={s} tokens={d}", .{ @errorName(e), keys.len * block_tokens });
    tier.abandon(p, keys, state, submitted);
    return false;
}

/// A log line before a tier exists (tests stay quiet).
fn note(comptime fmt: []const u8, args: anytype) void {
    if (builtin.is_test) return;
    std.log.warn("sf-disk: " ++ fmt, args);
}

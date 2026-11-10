//! sf-disk for dsv41 (out/strixfold/disk-tier-design.md): kept prompt states on disk on each rank of the TP pair, read
//! back after a restart. A state on disk is what an in-memory resume in another extent restores: the slot's own bytes
//! (Model.snapCopy) and the extent's compressed rows below its position (Model.copyExtent's planes), here in blocks of
//! chunk_rows positions shared by every state of a conversation. core/prompt_disk.zig keeps the files; this file makes
//! the identity, the planes' layout and every device copy; lanes.zig sends the frames (rank 0 decides, rank 1 follows).
//!   TF_DS_PROMPT_DISK_GIB         disk for kept states on this box (unset, empty or 0: off)
//!   TF_DS_PROMPT_DISK_DIR         where (default $HOME/strixfold-cache/prefix); one directory per identity below it
//!   TF_DS_PROMPT_DISK_FLOOR_GIB   no writes while the filesystem has less free (default 16)
//!   TF_DS_PROMPT_DISK_QUEUE_MIB   host memory for states waiting to be written (default 1024)
//!   TF_DS_PROMPT_DISK_MIN_TOKENS  shorter states stay in memory only (default 8192)
//!   TF_DS_PROMPT_DISK_RATE_MIB    the writer's MiB a second at most (default 1024; 0: no cap)
//! Both ranks read the same variables (serve_lib.sh hands them to both); the handshake carries " disk=1" when on.
const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("gpu");
const core = @import("core");
const model = @import("model.zig");
const pd = core.prompt_disk;
const Key = pd.Key;
const Model = model.Model;

pub const block_tokens = model.chunk_rows;

pub const Settings = struct {
    dir: []const u8,
    budget: u64,
    floor: u64 = 16 << 30,
    queue: u64 = 1 << 30,
    min_tokens: u32 = 8192,
    rate: u64 = 1 << 30,

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

fn envVar(name: [:0]const u8) ?[]const u8 {
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

/// The identity text of a rank's states (pd.namespaceOf names its directory): what a kept state's bytes depend on.
/// `plan` is the handshake's plan text; its conf= (serve.conf's hash: any key) and disk= tokens are left out.
pub fn identityText(a: std.mem.Allocator, parts: Parts) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    const w = struct {
        fn f(o: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const s = try std.fmt.allocPrint(al, fmt ++ "\n", args);
            defer al.free(s);
            try o.appendSlice(al, s);
        }
    }.f;
    try w(&out, a, "format={d}", .{pd.version});
    try w(&out, a, "rank={d}", .{parts.rank});
    try w(&out, a, "world={d}", .{parts.world});
    try w(&out, a, "deploy={s}", .{parts.deploy});
    try w(&out, a, "bundle={s}", .{parts.bundle});
    try w(&out, a, "kit_sha256={s}", .{parts.kit_sha256});
    try w(&out, a, "hip_runtime={d}", .{parts.hip_runtime});
    try w(&out, a, "arch={s}", .{parts.arch});
    var it = std.mem.tokenizeScalar(u8, parts.plan, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "conf=") or std.mem.startsWith(u8, tok, "disk=")) continue;
        try w(&out, a, "plan.{s}", .{tok});
    }
    try w(&out, a, "engram={s}", .{parts.engram});
    try w(&out, a, "token_map_sha256={s}", .{parts.token_map});
    try w(&out, a, "ds_kit={s}", .{parts.ds_kit});
    try w(&out, a, "window={d}", .{parts.window});
    try w(&out, a, "snap_bytes={d}", .{parts.snap_bytes});
    try w(&out, a, "block_tokens={d}", .{block_tokens});
    try w(&out, a, "planes={s}", .{parts.planes});
    return out.toOwnedSlice(a);
}

/// The text the prompt chain is seeded from: only what the two ranks' handshake already makes equal (the deployed
/// commit, the bundle, the kit, the world and the plan less conf= and disk=), never the rank. Both ranks must name a
/// prompt's blocks and states by the same keys (rank 0 decides, rank 1 checks the keys against its own chain); each
/// rank's files stay apart by its namespace (identityText, which has the rank and the rank-local inputs).
pub fn seedText(a: std.mem.Allocator, parts: Parts) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const w = struct {
        fn f(o: *std.ArrayList(u8), al: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !void {
            const t = try std.fmt.allocPrint(al, fmt ++ "\n", args);
            defer al.free(t);
            try o.appendSlice(al, t);
        }
    }.f;
    try w(&out, a, "chain format={d}", .{pd.version});
    try w(&out, a, "world={d}", .{parts.world});
    try w(&out, a, "deploy={s}", .{parts.deploy});
    try w(&out, a, "bundle={s}", .{parts.bundle});
    try w(&out, a, "kit_sha256={s}", .{parts.kit_sha256});
    var it = std.mem.tokenizeScalar(u8, parts.plan, ' ');
    while (it.next()) |tok| {
        if (std.mem.startsWith(u8, tok, "conf=") or std.mem.startsWith(u8, tok, "disk=")) continue;
        try w(&out, a, "plan.{s}", .{tok});
    }
    try w(&out, a, "block_tokens={d}", .{block_tokens});
    return out.toOwnedSlice(a);
}

pub const Parts = struct {
    rank: u32,
    world: u32,
    deploy: []const u8,
    bundle: []const u8,
    kit_sha256: []const u8,
    hip_runtime: c_int,
    arch: []const u8,
    plan: []const u8,
    engram: []const u8,
    token_map: []const u8,
    ds_kit: []const u8,
    window: usize,
    snap_bytes: usize,
    planes: []const u8,
};

/// The launch identity a served start has (serve_lib.sh): without the deployed commit and the bundle's hash no state
/// may be trusted across starts.
fn launchValue(name: [:0]const u8) ?[]const u8 {
    const v = envVar(name) orelse return null;
    return if (v.len == 0 or std.mem.eql(u8, v, "-")) null else v;
}

/// The SHA-256 of a file's bytes (hex), or "missing".
fn fileSha(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch return "missing";
    defer a.free(bytes);
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return a.dupe(u8, &std.fmt.bytesToHex(d, .lower));
}

/// A directory's regular files as "name:size,..." sorted by name (the Engram tables: their names and sizes).
fn listing(a: std.mem.Allocator, io: std.Io, dir: []const u8) ![]const u8 {
    var d = std.Io.Dir.cwd().openDir(io, dir, .{ .iterate = true }) catch return "missing";
    defer d.close(io);
    var names: std.ArrayList([]const u8) = .empty;
    var it = d.iterate();
    while (try it.next(io)) |e| if (e.kind == .file) try names.append(a, try a.dupe(u8, e.name));
    std.mem.sort([]const u8, names.items, {}, struct {
        fn lt(_: void, x: []const u8, y: []const u8) bool {
            return std.mem.lessThan(u8, x, y);
        }
    }.lt);
    var out: std.ArrayList(u8) = .empty;
    for (names.items, 0..) |n, i| {
        const st = d.statFile(io, n, .{}) catch continue;
        const s = try std.fmt.allocPrint(a, "{s}{s}:{d}", .{ if (i == 0) "" else ",", n, st.size });
        try out.appendSlice(a, s);
    }
    return out.toOwnedSlice(a);
}

/// This rank's identity parts (arena `a`): identityText names its directory, seedText seeds the prompt chain.
fn identity(a: std.mem.Allocator, m: *const Model, planes: []const Plane) !Parts {
    const deploy = launchValue("TF_DS_DEPLOY") orelse return error.NoLaunchIdentity;
    const bundle = launchValue("TF_DS_BUNDLE") orelse return error.NoLaunchIdentity;
    var arch_buf: [256]u8 = undefined;
    const arch = m.ctx.arch(&arch_buf) catch "unknown";
    var pl: std.ArrayList(u8) = .empty;
    for (planes, 0..) |p, i| try pl.appendSlice(a, try std.fmt.allocPrint(a, "{s}{d}:{d}", .{ if (i == 0) "" else ",", p.row, p.ratio }));
    var ds_kit: std.ArrayList(u8) = .empty;
    try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, "engram.json:{s}", .{try fileSha(a, m.io, try std.fs.path.join(a, &.{ m.kit_dir, "engram.json" }))}));
    // the RoPE tables by size (the deploy's own files, under its commit; hashing them would read up to 512 MiB a start)
    for ([_][]const u8{ "rope-plain-cos.f32", "rope-plain-sin.f32", "rope-compressed-cos.f32", "rope-compressed-sin.f32" }) |n| {
        const st = std.Io.Dir.cwd().statFile(m.io, try std.fs.path.join(a, &.{ m.kit_dir, n }), .{}) catch {
            try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, ",{s}:missing", .{n}));
            continue;
        };
        try ds_kit.appendSlice(a, try std.fmt.allocPrint(a, ",{s}:{d}", .{ n, st.size }));
    }
    const engram = if (m.engram_dir) |e| try std.fmt.allocPrint(a, "{s}[{s}]", .{ std.fs.path.basename(e), try listing(a, m.io, e) }) else "none";
    const kit_sha: []const u8 = if (gpu.backend == .hip) &m.kit.manifest_sha256 else "cuda";
    return .{
        .rank = m.rank,
        .world = m.world,
        .deploy = deploy,
        .bundle = bundle,
        .kit_sha256 = kit_sha,
        .hip_runtime = m.ctx.d.version() catch 0,
        .arch = arch,
        .plan = m.plan_text,
        .engram = engram,
        .token_map = if (m.token_map) |t| try fileSha(a, m.io, t) else "none",
        .ds_kit = ds_kit.items,
        .window = m.cfg.window,
        .snap_bytes = m.snapBytes(),
        .planes = pl.items,
    };
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

    /// The tier for this rank, or null with one log line (it then stays off: nothing is read or written). It owns
    /// s.dir (gpa) from here.
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
        const parts = try identity(arena.allocator(), m, planes);
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

    /// A prompt pass begins on `ids` (its tokens; rank 1: its sequence as rank 0 sent it): the keys of its whole blocks.
    pub fn beginPass(d: *Disk, ids: []const u8) void {
        const n = ids.len / 4 / block_tokens;
        d.chain.resize(d.gpa, n) catch {
            d.chain.clearRetainingCapacity();
            return;
        };
        pd.chain(d.h0, ids, block_tokens * 4, d.chain.items);
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

    /// Writes what `plan` reserved: its runs from the extent at `base` (keys: those it was planned for, the first being
    /// block first_j), then, when `slot_src` is set, the state's own bytes from that device buffer. The device copies
    /// wait for the stream (the pass's chunks and the save's snapCopy before them). False: nothing more of it is written.
    pub fn write(d: *Disk, p: *const pd.Plan, keys: []const Key, first_j: u32, base: usize, slot_src: ?u64) bool {
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

    /// The state of k blocks under `key` read back into slot `slot` and the extent at `base`, every byte verified on
    /// the way; the stream waits first (the extent's previous owner may still run). ok: the slot and rows [0, k blocks)
    /// are the state's; bad or busy: whatever was written there is overwritten by the pass from 0.
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

/// The plan's runs to the writer, each run's bytes from `copy(ctx, first block j, n)`. On a failure the plan is
/// abandoned with the runs actually queued: a run whose copy or submit failed queued nothing (submitRun frees its
/// bytes), so its reservation comes back. False then.
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

/// One rank's save of the state at k blocks of `tokens`, as rank 0's diskSave and rank 1's disk_save handler make it
/// (rank 1 checks rank 0's key against its own chain first); bytes depend on the rank, as each rank's part does.
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
    // (the namespace's own text as the seed, as before this fix, names them differently: no hit was possible)
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

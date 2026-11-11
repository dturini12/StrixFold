//! HIP family registration, device setup, memory admission and the native lane host (tools/dsv41-hip/ADAPTER.md
//! section 9: the `native-hip` server). It follows native/cuda.zig step for step over the backend-neutral `gpu` module
//! (zig/src/gpu/hip_facade.zig); the families it registers are the ones ported to HIP, today dsv41 alone.
//!   TF_HIP_DEVICE                      the device ordinal (--device; default 0)
//!   TENSORFOLD_HIP_KERNELS             the AMD kit (manifest.json and objects/); unset: share/tensorfold/hip/<arch>/
//!                                      <family> beside the binary
//!   TENSORFOLD_MEMORY_RESERVE_GIB      memory kept free for the host (default: a tenth, at least 4 GiB)
//!   TENSORFOLD_HIP_MEMORY_LIMIT_GB     the most this process allocates on the device (default 110: on a 128 GiB Strix
//!                                      Halo, w-link's allocation gate left ~10 GiB for the host at 112 GiB)
const std = @import("std");
const gpu = @import("gpu");
const api = @import("engine_api");
const lanes = @import("lanes");
const dsv41 = @import("dsv41");
const Allocator = std.mem.Allocator;
/// The admission arithmetic is the CUDA host's (it reads no device); only the counts come from HIP.
const budget = @import("cuda_memory.zig");
const Pool = budget.Pool;

/// HIP families provide metadata, open their lane backend and explain their refusals.
const registry = .{dsv41.native};

pub const backends: []const []const u8 = &.{"hip"};
pub const families: []const api.Family = blk: {
    var out: [registry.len]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    const final = out;
    break :blk &final;
};

const gib: f64 = 1 << 30;
/// TENSORFOLD_HIP_MEMORY_LIMIT_GB unset (w-link's allocgate on both Strix Halo boxes, 2026-10-19).
const default_limit: u64 = 110 << 30;

/// The chip class gate entries name ("amd-gfx1151" for a Strix Halo) of the device TF_HIP_DEVICE picks; null without one.
pub fn chip(a: Allocator) ?[]const u8 {
    const ordinal = envNumber(getenv("TF_HIP_DEVICE")) catch return null;
    var driver = gpu.Driver.open() catch return null;
    defer driver.close();
    var ctx = gpu.Context.init(&driver, @intCast(ordinal orelse 0)) catch return null;
    defer ctx.deinit();
    var buf: [256]u8 = undefined;
    return chipClass(a, ctx.arch(&buf) catch return null);
}

/// "amd-" and the device's gfx name without its feature suffixes ("gfx1151:xnack-" -> "amd-gfx1151").
fn chipClass(a: Allocator, arch: []const u8) ?[]const u8 {
    const base = arch[0 .. std.mem.indexOfScalar(u8, arch, ':') orelse arch.len];
    return std.fmt.allocPrint(a, "amd-{s}", .{base}) catch null;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// A variable's whole number: null when unset or empty, an error for anything else that is not one (as cuda.zig).
fn envNumber(text: ?[]const u8) error{Invalid}!?u32 {
    const t = std.mem.trim(u8, text orelse return null, " ");
    if (t.len == 0) return null;
    return std.fmt.parseInt(u32, t, 10) catch error.Invalid;
}

/// The flag's value, else the variable's, else `default`; `problem` names a variable that is not a whole number.
fn setting(a: Allocator, flag: ?u32, name: [:0]const u8, default: u32, problem: *[]const u8) !?u32 {
    if (flag) |v| return v;
    const v = envNumber(getenv(name)) catch {
        problem.* = try std.fmt.allocPrint(a, "{s}={s}: a whole number", .{ name, getenv(name).? });
        return null;
    };
    return v orelse default;
}

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// Prompt plus reply tokens a request may use: --context (0: the model's window), else the family default within it.
fn contextWindow(requested: ?i64, native: i64, default: i64) error{ Negative, NoNative, PastNative }!i64 {
    const r = requested orelse return if (native > 0) @min(default, native) else default;
    if (r < 0) return error.Negative;
    if (r == 0) return if (native > 0) native else error.NoNative;
    if (native > 0 and r > native) return error.PastNative;
    return r;
}

/// The family's AMD kit: TENSORFOLD_HIP_KERNELS, else share/tensorfold/hip/<arch>/<family> beside the binary.
fn kitDir(a: Allocator, io: std.Io, arch: []const u8, family: []const u8) ![]const u8 {
    if (getenv("TENSORFOLD_HIP_KERNELS")) |dir| return a.dupe(u8, dir);
    const exe = try std.process.executableDirPathAlloc(io, a);
    return std.fs.path.join(a, &.{ exe, "..", "share", "tensorfold", "hip", arch, family });
}

/// A /proc file's text, streamed: procfs reports size 0, and a positional read (readFileAlloc) stops there.
fn procText(a: Allocator, io: std.Io, path: []const u8) ?[]u8 {
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return null;
    defer file.close(io);
    var buf: [4096]u8 = undefined;
    var r = file.readerStreaming(io, &buf);
    return r.interface.allocRemaining(a, .limited(1 << 20)) catch null;
}

/// The pool now: an integrated GPU (Strix Halo) counts the host's MemAvailable, a discrete one HIP's free memory.
fn pool(a: Allocator, io: std.Io, ctx: *const gpu.Context, problem: *[]const u8) !?Pool {
    const unified = (try ctx.caps()).integrated;
    const card = try ctx.memInfo();
    const text = if (unified) procText(a, io, "/proc/meminfo") else null;
    const available = budget.counts(unified, .{ .total = card.total, .available = card.free }, text) catch {
        problem.* = "cannot read or parse /proc/meminfo's MemTotal and MemAvailable; refusing HIP unified-memory admission";
        return null;
    };
    const reserve = budget.reserveBytes(getenv("TENSORFOLD_MEMORY_RESERVE_GIB"), available.total) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_MEMORY_RESERVE_GIB={s}: a number of GiB from 2 to the memory's size", .{getenv("TENSORFOLD_MEMORY_RESERVE_GIB").?});
        return null;
    };
    const limit = budget.limitBytes(getenv("TENSORFOLD_HIP_MEMORY_LIMIT_GB")) catch {
        problem.* = try std.fmt.allocPrint(a, "TENSORFOLD_HIP_MEMORY_LIMIT_GB={s}: a positive number of GiB whose byte count fits in a 64-bit size", .{getenv("TENSORFOLD_HIP_MEMORY_LIMIT_GB").?});
        return null;
    };
    return .{ .free = available.available, .total = available.total, .reserve = reserve, .limit = limit orelse default_limit, .unified = unified };
}

fn readMemory(_: ?*anyopaque, reset_peak: bool) ?api.Memory {
    const u = gpu.usage(reset_peak);
    return .{ .active = u.device, .cache = 0, .peak = u.peak };
}

/// A lone driver returns false at completion or true when yield hands it to the lane core.
const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool) anyerror!bool;

/// The device a lane thread needs current: HIP selects the device per thread (hipSetDevice).
threadlocal var bound: ?*const gpu.Context = null;

const Gpu = struct { driver: gpu.Driver, ctx: gpu.Context };

/// One loaded model behind the lane host: everything the engine thread reads lives here (as cuda.zig's Host).
const Host = struct {
    gpa: Allocator,
    gpu: ?*Gpu, // null in host tests: no device to bind
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    inner: lanes.backend.Backend,
    vtable: lanes.backend.Backend.VTable,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,
    startup: []u8 = &.{},
    lone: ?LoneRun = null, // the family's driver for a lone drafted stream, called with `family`
    cache: ?api.prompt_cache.Store = null, // kept prompt states, when the family keeps them
    cache_images: bool = false, // the family resumes and keeps prompts with images (its pass keys them by picture)
    snap_vt: api.prompt_cache.Snapshots.VTable = undefined,

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        if (h.gpu) |g| g.ctx.makeCurrent() catch {};
        if (h.cache) |*store| store.deinit(); // the family's kept states go before the family
        h.release(h.family);
        if (h.gpu) |g| {
            g.ctx.deinit();
            g.driver.close();
            h.gpa.destroy(g);
        }
        h.gpa.free(h.startup);
        h.gpa.destroy(h);
    }

    fn bind(p: *anyopaque) *Host {
        const h: *Host = @ptrCast(@alignCast(p));
        if (h.gpu) |g| if (bound != &g.ctx) {
            g.ctx.makeCurrent() catch |e| std.log.err("hipSetDevice on the lane thread: {s}", .{@errorName(e)});
            bound = &g.ctx;
        };
        return h;
    }

    /// The round loop and the lane host over the family's backend, its thread started.
    fn serve(h: *Host, io: std.Io, facts: lanes.Model, rows: u32, info: api.Info, explain: ?api.Explain) !void {
        h.cfg = try lanes.Config.init(h.gpa, facts, rows, rows - 1);
        errdefer h.cfg.deinit(h.gpa);
        // TF_LANES_FROZEN_TIMING=1: widths from the tables only, never from measured round times (repeatable runs)
        if (getenv("TF_LANES_FROZEN_TIMING")) |v| h.cfg.frozen_timing = !std.mem.eql(u8, v, "0");
        if (getenv("TF_DS_DRAFT_BYPASS_AT")) |v| try h.cfg.setDraftBypassAt(try std.fmt.parseInt(u32, v, 10));
        h.clock = .{ .io = io };
        h.core = lanes.Engine.init(h.gpa, &h.cfg, h.backend(), h.clock.clock());
        errdefer h.core.deinit();
        h.host = api.LaneHost.init(h.gpa, io, &h.core, info);
        h.host.memory = if (h.gpu != null) .{ .read = readMemory } else null;
        h.host.explain = explain;
        if (h.cache) |*store| h.host.cache = store;
        h.host.cache_images = h.cache != null and h.cache_images;
        if (h.lone != null) h.host.lone = .{ .ctx = h, .run = loneRun, .sampled = true };
        try h.host.start();
    }

    /// The family's lone driver with the device current on the lane thread.
    fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const x = bind(p);
        return x.lone.?(x.family, s, hooks.ctx, hooks.committed, hooks.yield);
    }

    /// The family's backend, each call made with the device current on the calling thread.
    fn backend(h: *Host) lanes.backend.Backend {
        const v = h.inner.vtable;
        h.vtable = .{
            .prefill = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.prefill(x.inner.ptr, s);
                }
            }.f,
            .first = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.first(x.inner.ptr, s, position);
                }
            }.f,
            .queue = struct {
                fn f(p: *anyopaque, s: *lanes.Stream, feed: lanes.backend.Feed, position: u64) anyerror!u64 {
                    const x = bind(p);
                    return x.inner.vtable.queue(x.inner.ptr, s, feed, position);
                }
            }.f,
            .read = struct {
                fn f(p: *anyopaque, handle: u64) anyerror!u32 {
                    const x = bind(p);
                    return x.inner.vtable.read(x.inner.ptr, handle);
                }
            }.f,
            .verify = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, out: []lanes.backend.Verified) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.verify(x.inner.ptr, w, out);
                }
            }.f,
            .keep = struct {
                fn f(p: *anyopaque, w: []const lanes.backend.Window, paths: []const []const u32) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.keep(x.inner.ptr, w, paths);
                }
            }.f,
            .draft = struct {
                fn f(p: *anyopaque, r: []const lanes.backend.DraftRequest) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.draft(x.inner.ptr, r);
                }
            }.f,
            .unspeculate = if (v.unspeculate != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream) anyerror!void {
                    const x = bind(p);
                    return x.inner.vtable.unspeculate.?(x.inner.ptr, s);
                }
            }.f else null,
            .probabilities = if (v.probabilities != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []f64) anyerror!bool {
                    const x = bind(p);
                    return x.inner.vtable.probabilities.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .tree = if (v.tree != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, gpa: Allocator) anyerror!?lanes.stream.Held {
                    const x = bind(p);
                    return x.inner.vtable.tree.?(x.inner.ptr, s, gpa);
                }
            }.f else null,
            .alternatives = if (v.alternatives != null) struct {
                fn f(p: *anyopaque, s: *lanes.Stream, out: []lanes.backend.Alternative) anyerror!usize {
                    const x = bind(p);
                    return x.inner.vtable.alternatives.?(x.inner.ptr, s, out);
                }
            }.f else null,
            .release = struct {
                fn f(p: *anyopaque, s: *lanes.Stream) void {
                    const x = bind(p);
                    x.inner.vtable.release(x.inner.ptr, s);
                }
            }.f,
        };
        return .{ .ptr = h, .vtable = &h.vtable };
    }
};

/// The engine for `o.dir`, or null with `problem` set when no HIP family reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const native = modelContext(a, io, o.dir);
    const window = contextWindow(o.context, native, F.default_context) catch |e| {
        problem.* = switch (e) {
            error.Negative => try std.fmt.allocPrint(a, "--context {d}: a token count, or 0 for the model's window", .{o.context.?}),
            error.NoNative => "--context 0 asks for the model's window, and its config.json names none: give a token count",
            error.PastNative => try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ o.context.?, native }),
        };
        return null;
    };
    const device = try setting(a, o.device, "TF_HIP_DEVICE", 0, problem) orelse return null;
    const segments = try setting(a, o.segments, "TF_HIP_SEGMENTS", 1, problem) orelse return null;
    if (segments < 1 or segments > F.max_segments) {
        const named = if (o.segments != null) "--segments " else "TF_HIP_SEGMENTS=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: whole prompt chunks a call, 1 to {d}", .{ named, segments, F.max_segments });
        return null;
    }
    const g = try gpa.create(Gpu);
    var opened = false;
    defer if (!opened) gpa.destroy(g);
    g.driver = gpu.Driver.open() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "no HIP runtime (libamdhip64.so.7: {s})", .{@errorName(e)});
        return null;
    };
    defer if (!opened) g.driver.close();
    const count: u32 = @intCast(try g.driver.deviceCount());
    if (device >= count) {
        const named = if (o.device != null) "--device " else "TF_HIP_DEVICE=";
        problem.* = try std.fmt.allocPrint(a, "{s}{d}: this machine has {d} HIP device{s} (0 to {d})", .{ named, device, count, if (count == 1) "" else "s", @max(count, 1) - 1 });
        return null;
    }
    g.ctx = try gpu.Context.init(&g.driver, @intCast(device));
    defer if (!opened) g.ctx.deinit();
    bound = &g.ctx;
    const caps = g.ctx.caps() catch |e| {
        problem.* = try std.fmt.allocPrint(a, "HIP device {d} is outside this build's ISA table ({s}): the dsv41 kit is built for gfx1151", .{ device, @errorName(e) });
        return null;
    };
    var name_buf: [256]u8 = undefined;
    const name = g.ctx.name(&name_buf) catch "GPU";
    const kit = try kitDir(a, io, caps.archName(), "dsv41");
    std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ kit, "manifest.json" }), .{}) catch {
        problem.* = try std.fmt.allocPrint(a, "no AMD kernel kit for {s} at {s}: set TENSORFOLD_HIP_KERNELS to a kit (zig build native-hip installs one)", .{ caps.archName(), kit });
        return null;
    };
    const before = try pool(a, io, &g.ctx, problem) orelse return null;
    const weights = if (@hasDecl(F, "weightBytes")) F.weightBytes(io, o.dir) else 0;
    const held0 = gpu.usage(false).device;
    if (weights > before.room(held0)) {
        problem.* = try std.fmt.allocPrint(a, "this rank's {d:.1} GiB of weights do not fit the {d:.1} GiB the HIP memory budget grants ({d:.1} GiB available less a {d:.1} GiB reserve, at most {d:.1} GiB in all); free memory or adjust TENSORFOLD_MEMORY_RESERVE_GIB / TENSORFOLD_HIP_MEMORY_LIMIT_GB", .{ toGib(weights), toGib(before.room(held0)), toGib(before.free), toGib(before.reserve), toGib(before.limit.?) });
        return null;
    }
    const loaded = F.open(gpa, io, &g.ctx, o.dir, kit, .{ .context = @intCast(window), .drafts = o.drafts, .segments = segments }) catch |e| {
        const why = if (@hasDecl(F, "explain")) F.explain(null, e) else null;
        problem.* = try std.fmt.allocPrint(a, "the native HIP engine cannot load {s} with kit {s} ({s}){s}{s}", .{ o.dir, kit, @errorName(e), if (why != null) ": " else "", why orelse "" });
        return null;
    };
    defer if (!opened) loaded.deinit(loaded.ctx);
    const model = gpu.usage(false).device - held0;
    const after = try pool(a, io, &g.ctx, problem) orelse return null;
    const room = after.room(model);
    const streams = budget.admit(room, loaded.stream_bytes, o.lanes, o.lanes_fixed) catch |e| {
        problem.* = switch (e) {
            error.NoStream => try std.fmt.allocPrint(a, "the HIP memory budget fits no stream: one at a {d}-token window takes {d:.2} GiB, and {d:.2} GiB is left after the model's {d:.2} GiB and the {d:.1} GiB reserve; lower --context, or free memory", .{ window, toGib(loaded.stream_bytes), toGib(room), toGib(model), toGib(after.reserve) }),
            error.TooMany => try std.fmt.allocPrint(a, "--parallel {d} needs {d:.2} GiB for its streams at a {d}-token window, and the HIP memory budget leaves {d:.2} GiB: serve --parallel {d}, or lower --context", .{ o.lanes, toGib(loaded.stream_bytes * o.lanes), window, toGib(room), room / loaded.stream_bytes }),
        };
        return null;
    };
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.* = .{ .gpa = gpa, .gpu = g, .family = loaded.ctx, .release = loaded.deinit, .inner = loaded.backend, .vtable = undefined, .cfg = undefined, .clock = undefined, .core = undefined, .host = undefined, .lone = loaded.lone };
    h.startup = try std.fmt.allocPrint(gpa, "HIP {s} device {d} ({s}{s}): model {d:.2} GiB; {d} stream{s} at once, {d:.2} GiB each at a {d}-token window, of {d:.1} GiB left after a {d:.1} GiB reserve; prompts in {d}-row chunks", .{
        caps.archName(), device, name, if (after.unified) ", memory shared with the host" else "", toGib(model), streams, if (streams == 1) "" else "s", toGib(loaded.stream_bytes), window, toGib(room), toGib(after.reserve), F.prompt_rows,
    });
    errdefer gpa.free(h.startup);
    // a family that keeps prompt states (its snapshots) gets the prompt cache: states at its chunk ends
    if (@hasField(@TypeOf(loaded), "snaps")) if (loaded.snaps) |sn| {
        const want: u64 = @intFromFloat((o.prompt_cache_gib orelse 0.5) * gib);
        if (want > 0) {
            h.snap_vt = .{ .bytes = sn.bytes, .save = sn.save, .restore = sn.restore, .drop = sn.drop };
            h.cache = api.prompt_cache.Store.init(gpa, .{ .ptr = sn.ptr, .vtable = &h.snap_vt }, .{}, want);
            h.cache_images = @hasField(@TypeOf(loaded), "cache_images") and loaded.cache_images;
        }
    };
    // text only on HIP (D13): the family hands no vision hook, so the server refuses image parts
    const vision: ?api.Vision = if (@hasField(@TypeOf(loaded), "vision")) loaded.vision else null;
    // structured output and required calls when the family read its tokenizer's vocabulary (its draws enforce them)
    const structures = if (@hasField(@TypeOf(loaded), "structures")) loaded.structures else false;
    // the window the family can actually hold (a request's extent is rounded up past its tokens), else --context
    const served_window: i64 = if (@hasField(@TypeOf(loaded), "context_window")) (if (loaded.context_window) |w| @intCast(w) else window) else window;
    try h.serve(io, loaded.facts, loaded.rows, .{ .lanes = streams, .context_window = @intCast(served_window), .startup = h.startup, .efforts = familyEfforts(F), .vision = vision, .structures = structures, .top_k_max = if (@hasDecl(F, "top_k_max")) F.top_k_max else 0, .needs_top_k = @hasDecl(F, "needs_top_k") and F.needs_top_k }, .{ .ctx = loaded.ctx, .text = F.explain });
    opened = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// A family's reasoning efforts as numbers its chat template takes (F.efforts: name and value), else none.
fn familyEfforts(comptime F: type) []const api.Effort {
    if (!@hasDecl(F, "efforts")) return &.{};
    const table = comptime blk: {
        var out: [F.efforts.len]api.Effort = undefined;
        for (F.efforts, &out) |e, *o| o.* = .{ .name = e.name, .value = e.value };
        break :blk out;
    };
    return &table;
}

fn toGib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / gib;
}

test "chip classes name the gfx target without its feature suffixes" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "gfx1151", "gfx1151:xnack-" }) |arch| {
        const name = chipClass(a, arch).?;
        defer a.free(name);
        try std.testing.expectEqualStrings("amd-gfx1151", name);
    }
}

test "the HIP build registers dsv41 alone, on the hip backend" {
    try std.testing.expectEqual(@as(usize, 1), families.len);
    try std.testing.expectEqualStrings(dsv41.native.model_type, families[0].model_type);
    try std.testing.expectEqualStrings("hip", backends[0]);
}

test "context windows and variables as the CUDA host reads them" {
    try std.testing.expectEqual(@as(i64, 16384), try contextWindow(null, 262144, 16384));
    try std.testing.expectEqual(@as(i64, 262144), try contextWindow(0, 262144, 16384));
    try std.testing.expectError(error.PastNative, contextWindow(300000, 262144, 16384));
    try std.testing.expectEqual(@as(?u32, null), try envNumber(""));
    try std.testing.expectError(error.Invalid, envNumber("-1"));
}

test "the default device limit leaves the host its share of a 128 GiB Strix Halo" {
    const g: u64 = 1 << 30;
    // 120 GiB available and a reserve of a tenth of 128: with nothing held the reserve binds (107.2 GiB), with 100 GiB
    // held the 110 GiB limit does
    const p: Pool = .{ .free = 120 * g, .total = 128 * g, .reserve = try budget.reserveBytes(null, 128 * g), .limit = default_limit, .unified = true };
    try std.testing.expectEqual(p.free - p.reserve, p.room(0));
    try std.testing.expectEqual(@as(u64, 10 * g), p.room(100 * g));
}

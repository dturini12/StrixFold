//! LaneHost and the family's tower gate (Info.tower_slice_ns), and the encoder job forgotten at a pause.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const Image = lanes.stream.Image;

const io = std.testing.io;

fn sleepMs(ms: i64) void {
    std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
}

/// Fails the test binary if it is still armed after 5 s (a hang becomes a failure).
const Watchdog = struct {
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    fn arm(w: *Watchdog) !void {
        w.thread = try std.Thread.spawn(.{}, watch, .{w});
    }

    fn watch(w: *Watchdog) void {
        var ms: u32 = 0;
        while (ms < 5000) : (ms += 5) {
            if (w.done.load(.acquire)) return;
            sleepMs(5);
        }
        @panic("lane host gate test watchdog: no progress for 5 s (a hang)");
    }

    fn disarm(w: *Watchdog) void {
        w.done.store(true, .release);
        if (w.thread) |t| t.join();
    }
};

const b_images = [_]Image{.{ .at = 0, .tokens = 2, .bytes = "parked picture" }};
const a_images = [_]Image{.{ .at = 0, .tokens = 2, .bytes = "skipped picture" }};

/// The encoder as the lane host sees it, a model of encoder_gate.zig: B moves a unit a slice; A's urgent take passes.
const Tower = struct {
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    core: *lanes.Engine,
    held: bool = false,
    credit: u32 = 0,
    rush: bool = false,
    urgent_done: bool = false,
    b_active: bool = false,
    b_units: u32 = 0,
    b_need: u32 = std.math.maxInt(u32),
    stop: bool = false,
    last: enum { none, hold, slice, release } = .none,
    holds: u32 = 0,
    slices: u32 = 0,
    releases: u32 = 0,
    budget: u64 = 0,
    violations: u32 = 0,
    takes: u32 = 0,
    take_held: bool = false, // the gate was held when A's pass asked for its picture
    forgets: std.atomic.Value(u32) = .init(0),

    fn lock(t: *Tower) void {
        t.mutex.lockUncancelable(io);
    }

    fn unlock(t: *Tower) void {
        t.mutex.unlock(io);
    }

    fn of(ctx: *anyopaque) *Tower {
        return @ptrCast(@alignCast(ctx));
    }

    fn start(ctx: *anyopaque, images: []const Image, _: u32, _: lanes.stream.Wake) void {
        const t = of(ctx);
        t.lock();
        defer t.unlock();
        if (images.ptr == &b_images) t.b_active = true;
        t.cond.broadcast(io);
    }

    fn ready(ctx: *anyopaque, images: []const Image) bool {
        const t = of(ctx);
        t.lock();
        defer t.unlock();
        return images.ptr != &b_images or t.b_units >= t.b_need;
    }

    fn forget(ctx: *anyopaque, _: []const Image) void {
        _ = of(ctx).forgets.fetchAdd(1, .acq_rel);
    }

    fn hold(ctx: *anyopaque) void {
        const t = of(ctx);
        t.lock();
        defer t.unlock();
        if (t.core.activeCount() == 0 or t.last == .hold) t.violations += 1; // only around a round, and once
        t.held = true;
        t.holds += 1;
        t.last = .hold;
    }

    fn slice(ctx: *anyopaque, budget_ns: u64) void {
        const t = of(ctx);
        t.lock();
        defer t.unlock();
        if (!t.held or t.last != .hold) t.violations += 1; // after the round its hold began
        t.budget = budget_ns;
        t.slices += 1;
        t.last = .slice;
        if (!t.b_active or t.b_units >= t.b_need) return;
        t.credit = 1;
        t.cond.broadcast(io);
        while (t.credit > 0 and !t.stop) t.cond.waitUncancelable(io, &t.mutex);
    }

    fn release(ctx: *anyopaque) void {
        const t = of(ctx);
        t.lock();
        defer t.unlock();
        t.held = false;
        t.credit = 0;
        t.releases += 1;
        t.last = .release;
        t.cond.broadcast(io);
    }

    /// A's prompt pass asks for its skipped picture (the cache missed): it waits for the urgent encode.
    fn take(t: *Tower) void {
        t.lock();
        defer t.unlock();
        t.takes += 1;
        t.take_held = t.held;
        t.rush = true;
        t.cond.broadcast(io);
        while (!t.urgent_done) t.cond.waitUncancelable(io, &t.mutex);
        t.urgent_done = false;
    }

    /// The encoder thread: the urgent picture first, else B's next unit when the gate lets it.
    fn work(t: *Tower) void {
        t.lock();
        defer t.unlock();
        while (true) {
            while (!t.stop and !t.rush and !(t.b_active and t.b_units < t.b_need and (!t.held or t.credit > 0))) t.cond.waitUncancelable(io, &t.mutex);
            if (t.stop) return;
            if (t.rush) {
                t.rush = false;
                t.urgent_done = true;
            } else {
                t.b_units += 1;
                if (t.held) t.credit -= 1;
            }
            t.cond.broadcast(io);
            t.unlock();
            sleepMs(1);
            t.lock();
        }
    }

    fn halt(t: *Tower) void {
        t.lock();
        defer t.unlock();
        t.stop = true;
        t.cond.broadcast(io);
    }

    fn prepare(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: *[]const u8) lanes.stream.Vision.PrepareError!lanes.stream.Vision.Prepared {
        return error.BadImage;
    }

    fn vision(t: *Tower) lanes.stream.Vision {
        const e: lanes.stream.Encode = .{ .ctx = t, .start = start, .ready = ready, .forget = forget, .hold = hold, .slice = slice, .release = release };
        return .{ .ctx = t, .placeholder = "<image>", .token = 9, .prepare = prepare, .encode = e };
    }

    /// The fake backend's chunk hook: A's first chunk takes its picture.
    fn chunk(ctx: *anyopaque, s: *lanes.Stream, at: usize) void {
        const t = of(ctx);
        if (s.images.ptr == &a_images and at == 0) t.take();
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    tokens: usize = 0,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .tokens => |t| b.tokens += t.len,
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn wait(b: *Box) Reason {
        while (true) {
            if (b.reason()) |r| return r;
            sleepMs(1);
        }
    }

    fn reason(b: *Box) ?Reason {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done;
    }

    fn count(b: *Box) usize {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.tokens;
    }
};

fn sink(b: *Box) api.Sink {
    return .{ .ctx = b, .event = Box.event };
}

test "a lane host holds the tower around rounds only, slices after each, releases before idling and at its end; a request needing an urgent encode passes a parked job" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tower: Tower = .{ .core = &core };
    target.prefill_hook = Tower.chunk;
    target.prefill_hook_ctx = &tower;
    const worker = try std.Thread.spawn(.{}, Tower.work, .{&tower});
    defer worker.join();
    defer tower.halt();
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 3, .vision = tower.vision(), .tower_slice_ns = std.time.ns_per_ms });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    const e = host.engine();
    const prompt = [_]u32{ 9, 9, 3, 1 };
    const decoder: Request = .{ .prompt = &[_]u32{ 2, 7, 1, 8 }, .max_tokens = 100000 };
    const b_request: Request = .{ .prompt = &prompt, .images = &b_images, .max_tokens = 4 };
    const a_request: Request = .{ .prompt = &prompt, .images = &a_images, .max_tokens = 4 };
    var d: Box = .{};
    var b: Box = .{};
    var a: Box = .{};
    try e.submit(1, &decoder, sink(&d));
    try e.submit(2, &b_request, sink(&b));
    while (true) { // the decoder's rounds hold the tower; B's job runs in their slices
        tower.lock();
        const going = tower.b_units >= 3;
        tower.unlock();
        if (going) break;
        sleepMs(1);
    }
    // A is ready (its picture skipped on a lost hint): admitted past B, its pass asks for it with the tower held
    try e.submit(3, &a_request, sink(&a));
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expect(b.reason() == null and d.reason() == null);
    tower.lock();
    const takes = tower.takes;
    const take_held = tower.take_held;
    tower.b_need = tower.b_units + 2; // B's job ends after two more slices
    tower.unlock();
    try std.testing.expectEqual(@as(u32, 1), takes);
    try std.testing.expect(take_held);
    try std.testing.expectEqual(Reason.length, b.wait());
    const before = d.count();
    while (d.count() < before + 8) sleepMs(1);
    e.cancel(1);
    try std.testing.expectEqual(Reason.cancelled, d.wait());
    sleepMs(250); // idle: the host waits for work (100 ms at a time) with the tower free
    tower.lock();
    const idle_held = tower.held;
    const idle_last = tower.last;
    const holds = tower.holds;
    tower.unlock();
    try std.testing.expect(!idle_held and idle_last == .release);
    sleepMs(150);
    tower.lock();
    try std.testing.expectEqual(holds, tower.holds); // no hold while nothing decodes
    tower.unlock();
    // run returns with the tower free: a decoder still running when the host stops
    var d2: Box = .{};
    try e.submit(4, &decoder, sink(&d2));
    while (d2.count() < 4) sleepMs(1);
    host.stop();
    stopped = true;
    try std.testing.expectEqual(Reason.cancelled, d2.wait());
    tower.lock();
    defer tower.unlock();
    try std.testing.expect(!tower.held and tower.last == .release);
    try std.testing.expectEqual(@as(u32, 0), tower.violations);
    try std.testing.expectEqual(@as(u64, std.time.ns_per_ms), tower.budget);
    try std.testing.expect(tower.holds > 4 and tower.slices > 4);
}

test "with tower_slice_ns 0 a lane host never holds, slices or releases the tower" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tower: Tower = .{ .core = &core, .b_need = 0 };
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 2, .vision = tower.vision() });
    try host.start();
    defer host.stop();
    const prompt = [_]u32{ 9, 9, 3, 1 };
    const b_request: Request = .{ .prompt = &prompt, .images = &b_images, .max_tokens = 16 };
    const text: Request = .{ .prompt = &prompt, .max_tokens = 16 };
    var x: Box = .{};
    var y: Box = .{};
    try host.engine().submit(1, &text, sink(&x));
    try host.engine().submit(2, &b_request, sink(&y));
    try std.testing.expectEqual(Reason.length, x.wait());
    try std.testing.expectEqual(Reason.length, y.wait());
    tower.lock();
    defer tower.unlock();
    try std.testing.expectEqual(@as(u32, 0), tower.holds + tower.slices + tower.releases);
}

const Paused = struct {
    tower: *Tower,
    forgets_at_second: ?u32 = null,

    fn chunk(ctx: *anyopaque, s: *lanes.Stream, at: usize) void {
        const p: *Paused = @ptrCast(@alignCast(ctx));
        if (s.images.ptr == &a_images and at == 1) p.forgets_at_second = p.tower.forgets.load(.acquire);
    }
};

test "an image pass that pauses has its encoder job forgotten at the pause; a text pass's pause forgets nothing" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 3, .slice_chunks = 1 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tower: Tower = .{ .core = &core, .b_need = 0 };
    var paused: Paused = .{ .tower = &tower };
    target.prefill_hook = Paused.chunk;
    target.prefill_hook_ctx = &paused;
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 3, .vision = tower.vision(), .fill_slice_ns = std.time.ns_per_s });
    try host.start();
    defer host.stop();
    const e = host.engine();
    const prompt = [_]u32{ 9, 9, 3, 1, 4, 1 };
    const decoder: Request = .{ .prompt = &[_]u32{ 2, 7, 1, 8 }, .max_tokens = 100000 };
    const text: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    const pictured: Request = .{ .prompt = &prompt, .images = &a_images, .max_tokens = 4 };
    var d: Box = .{};
    try e.submit(1, &decoder, sink(&d));
    while (d.count() < 2) sleepMs(1); // its rounds make the passes below pause after each chunk
    var t: Box = .{};
    try e.submit(2, &text, sink(&t));
    try std.testing.expectEqual(Reason.length, t.wait());
    try std.testing.expectEqual(@as(u32, 0), tower.forgets.load(.acquire));
    var p: Box = .{};
    try e.submit(3, &pictured, sink(&p));
    try std.testing.expectEqual(Reason.length, p.wait());
    try std.testing.expectEqual(@as(?u32, 1), paused.forgets_at_second); // at the pause after its first chunk
    e.cancel(1);
    _ = d.wait();
}

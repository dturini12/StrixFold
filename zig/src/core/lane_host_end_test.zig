//! A lane host's ends with no memory left: a failed round, named cancellations and a close each end every job once.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;

/// A request's finished events: how many came, and the last one's reason.
const Box = struct {
    finished: usize = 0,
    reason: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        if (e.* == .finished) {
            b.finished += 1;
            b.reason = e.finished.reason;
        }
    }
};

const prompt = [_]u32{ 2, 7, 1, 8, 2, 8 };
const request: api.Request = .{ .prompt = &prompt, .max_tokens = 8 };

/// The ends below, called on a host with no engine thread whose allocator fails from `arm` on.
const Rig = struct {
    failing: std.testing.FailingAllocator,
    boxes: [3]Box = .{ .{}, .{}, .{} },

    fn arm(r: *Rig) void {
        r.failing.fail_index = r.failing.alloc_index;
    }
    fn disarm(r: *Rig) void {
        r.failing.fail_index = std.math.maxInt(usize);
    }
    fn expectEach(r: *Rig, reason: Reason) !void {
        for (&r.boxes) |*b| {
            try std.testing.expectEqual(@as(usize, 1), b.finished);
            try std.testing.expectEqual(reason, b.reason.?);
        }
    }
};

const End = enum { fail, cancel, close };

/// Three requests submitted, the first admitted (its stream never opened), then `end` with every allocation failing.
fn endWithoutMemory(end: End) !void {
    var rig: Rig = .{ .failing = std.testing.FailingAllocator.init(std.testing.allocator, .{}) };
    const gpa = rig.failing.allocator();
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 4 });
    defer host.stop();
    const e = host.engine();
    for (&rig.boxes, 1..) |*b, id| try e.submit(id, &request, .{ .ctx = b, .event = Box.event });
    try host.admitted.append(gpa, host.queued.orderedRemove(0));
    if (end == .fail) {
        try host.admitted.appendSlice(gpa, host.queued.items);
        host.queued.clearRetainingCapacity();
    }
    if (end == .cancel) for (1..rig.boxes.len + 1) |id| e.cancel(id);
    rig.arm();
    switch (end) {
        .fail => host.failAll("the round failed"),
        .cancel => host.takeCancels(),
        .close => host.closeAll(),
    }
    rig.disarm();
    try rig.expectEach(if (end == .fail) .failed else .cancelled);
    try std.testing.expectEqual(@as(usize, 0), host.queued.items.len + host.admitted.items.len + host.cancels.items.len);
}

test "a failed round ends every admitted job once, failed, when no allocation can succeed" {
    try endWithoutMemory(.fail);
}

test "named cancellations end their queued and admitted jobs once, cancelled, when no allocation can succeed" {
    try endWithoutMemory(.cancel);
}

test "a close ends every queued and admitted job once, cancelled, when no allocation can succeed" {
    try endWithoutMemory(.close);
}

/// A request's events as the engine thread sends them: its tokens, how many finished events came and the last reason.
const Watched = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    finished: usize = 0,
    reason: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const w: *@This() = @ptrCast(@alignCast(ctx));
        w.mutex.lockUncancelable(std.testing.io);
        defer w.mutex.unlock(std.testing.io);
        switch (e.*) {
            .tokens => |t| w.tokens.appendSlice(std.testing.allocator, t) catch {},
            .finished => |f| {
                w.finished += 1;
                w.reason = f.reason;
            },
            else => {},
        }
    }
    fn snapshot(w: *@This()) struct { tokens: usize, finished: usize, reason: ?Reason } {
        w.mutex.lockUncancelable(std.testing.io);
        defer w.mutex.unlock(std.testing.io);
        return .{ .tokens = w.tokens.items.len, .finished = w.finished, .reason = w.reason };
    }
    /// Waits for the finished event (10 s at most: a host stuck with a stale filling job never sends it).
    fn done(w: *@This()) !Reason {
        for (0..10_000) |_| {
            const s = w.snapshot();
            if (s.finished > 0) return s.reason.?;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
    fn decodes(w: *@This()) !void {
        for (0..10_000) |_| {
            if (w.snapshot().tokens > 0) return;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return error.Timeout;
    }
};

/// A stream decodes while another's prompt pass is suspended between its pieces (TF_FILL_CHUNKS=1, its turn never
/// comes: fill_rounds at its most); then `end` ends the suspended pass: its cancel, a failed round, or the host's close.
/// Both filling pointers must be cleared, every job must finish exactly once, and after a cancel or a failure another
/// request must run to its end.
fn endSuspendedFill(end: End) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4 };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    core.fill_chunks = 1;
    core.fill_rounds = std.math.maxInt(u32);
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 3 });
    var stopped = false;
    defer if (!stopped) host.stop();
    try host.start();
    const e = host.engine();
    var wa: Watched = .{};
    defer wa.tokens.deinit(gpa);
    var wf: Watched = .{};
    defer wf.tokens.deinit(gpa);
    var wc: Watched = .{};
    defer wc.tokens.deinit(gpa);
    const pa = [_]u32{ 3, 1, 4, 1, 5 };
    const pf = [_]u32{ 2, 7, 1, 8, 2, 8, 1, 8 };
    const pc = [_]u32{ 1, 6, 1, 8, 0, 3 };
    try e.submit(1, &.{ .prompt = &pa, .max_tokens = 1_000_000 }, .{ .ctx = &wa, .event = Watched.event });
    try wa.decodes();
    try e.submit(2, &.{ .prompt = &pf, .max_tokens = 8 }, .{ .ctx = &wf, .event = Watched.event });
    // the pass is suspended once both the host and the core hold it (its first piece ran)
    var suspended = false;
    for (0..10_000) |_| {
        if (@atomicLoad(@TypeOf(host.filling), &host.filling, .acquire) != null and @atomicLoad(?*lanes.stream.Stream, &core.filling, .acquire) != null) {
            suspended = true;
            break;
        }
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(suspended);
    switch (end) {
        .cancel => {
            e.cancel(2);
            try std.testing.expectEqual(Reason.cancelled, try wf.done());
            e.cancel(1);
            try std.testing.expectEqual(Reason.cancelled, try wa.done());
        },
        .fail => {
            target.fail_round.store(true, .release);
            try std.testing.expectEqual(Reason.failed, try wa.done());
            try std.testing.expectEqual(Reason.failed, try wf.done());
        },
        .close => {
            host.stop();
            stopped = true;
            try std.testing.expectEqual(Reason.cancelled, try wa.done());
            try std.testing.expectEqual(Reason.cancelled, try wf.done());
        },
    }
    if (end != .close) {
        // another request runs to its end: no stale filling job holds the admissions
        try e.submit(3, &.{ .prompt = &pc, .max_tokens = 6 }, .{ .ctx = &wc, .event = Watched.event });
        try std.testing.expectEqual(Reason.length, try wc.done());
        try std.testing.expectEqual(@as(usize, 6), wc.snapshot().tokens);
        host.stop();
        stopped = true;
    }
    try std.testing.expect(host.filling == null);
    try std.testing.expect(core.filling == null);
    try std.testing.expectEqual(@as(usize, 1), wa.snapshot().finished);
    try std.testing.expectEqual(@as(usize, 1), wf.snapshot().finished);
    if (end != .close) try std.testing.expectEqual(@as(usize, 1), wc.snapshot().finished);
}

test "a pass suspended between its pieces, cancelled: both filling pointers clear, each job ends once, the next runs" {
    try endSuspendedFill(.cancel);
}

test "a pass suspended between its pieces when a round fails: both filling pointers clear, each job ends once, the next runs" {
    try endSuspendedFill(.fail);
}

test "a pass suspended between its pieces when the host closes: both filling pointers clear, each job ends once" {
    try endSuspendedFill(.close);
}

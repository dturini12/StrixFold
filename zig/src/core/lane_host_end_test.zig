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

//! LaneHost with an off-thread tower (lanes.stream.Encode): an image job waits for its rows, the others go first.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;
const Image = lanes.stream.Image;

const io = std.testing.io;

/// The family's encoder as LaneHost sees it: every request's rows ready once `open` is set.
const Tower = struct {
    open: std.atomic.Value(bool) = .init(false),
    starts: std.atomic.Value(u32) = .init(0),
    forgets: std.atomic.Value(u32) = .init(0),
    wake: ?lanes.stream.Wake = null,

    fn start(ctx: *anyopaque, _: []const Image, _: u32, wake: lanes.stream.Wake) void {
        const t: *Tower = @ptrCast(@alignCast(ctx));
        t.wake = wake;
        _ = t.starts.fetchAdd(1, .acq_rel);
    }

    fn ready(ctx: *anyopaque, _: []const Image) bool {
        const t: *Tower = @ptrCast(@alignCast(ctx));
        return t.open.load(.acquire);
    }

    fn forget(ctx: *anyopaque, _: []const Image) void {
        const t: *Tower = @ptrCast(@alignCast(ctx));
        _ = t.forgets.fetchAdd(1, .acq_rel);
    }

    fn prepare(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: *[]const u8) lanes.stream.Vision.PrepareError!lanes.stream.Vision.Prepared {
        return error.BadImage;
    }

    fn vision(t: *Tower, encode: bool) lanes.stream.Vision {
        const e: lanes.stream.Encode = .{ .ctx = t, .start = start, .ready = ready, .forget = forget };
        return .{ .ctx = t, .placeholder = "<image>", .token = 9, .prepare = prepare, .encode = if (encode) e else null };
    }

    fn release(t: *Tower) void {
        t.open.store(true, .release);
        const w = t.wake.?;
        w.call(w.ctx);
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    seq: *std.atomic.Value(u32),
    prefilled: ?u32 = null, // its admission's place among the boxes
    done: ?Reason = null,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        switch (e.*) {
            .prefilled => b.prefilled = b.seq.fetchAdd(1, .acq_rel),
            .finished => |f| b.done = f.reason,
            else => {},
        }
    }

    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(io);
            const d = b.done;
            b.mutex.unlock(io);
            if (d) |r| return r;
            std.Io.sleep(io, .fromMilliseconds(1), .awake) catch {};
        }
    }

    fn finished(b: *Box) bool {
        b.mutex.lockUncancelable(io);
        defer b.mutex.unlock(io);
        return b.done != null;
    }
};

test "an image job waits for its encode while text jobs pass it in order; a cancelled one is forgotten" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tower: Tower = .{};
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 1, .vision = tower.vision(true) });
    try host.start();
    defer host.stop();
    var seq: std.atomic.Value(u32) = .init(0);
    const prompt = [_]u32{ 9, 9, 3, 1 };
    const images = [_]Image{.{ .at = 0, .tokens = 2, .bytes = "picture" }};
    const pictured: Request = .{ .prompt = &prompt, .images = &images, .max_tokens = 4 };
    const text: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    var a: Box = .{ .seq = &seq };
    var b: Box = .{ .seq = &seq };
    var c: Box = .{ .seq = &seq };
    const e = host.engine();
    try e.submit(1, &pictured, .{ .ctx = &a, .event = Box.event });
    try e.submit(2, &text, .{ .ctx = &b, .event = Box.event });
    try e.submit(3, &text, .{ .ctx = &c, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(Reason.length, c.wait());
    try std.testing.expect(!a.finished());
    try std.testing.expectEqual(@as(?u32, 0), b.prefilled);
    try std.testing.expectEqual(@as(?u32, 1), c.prefilled);
    try std.testing.expectEqual(@as(?u32, null), a.prefilled);
    tower.release();
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(?u32, 2), a.prefilled);
    try std.testing.expectEqual(@as(u32, 1), tower.starts.load(.acquire));
    try std.testing.expect(tower.forgets.load(.acquire) >= 1);

    tower.open.store(false, .release);
    var d: Box = .{ .seq = &seq };
    const forgets = tower.forgets.load(.acquire);
    try e.submit(4, &pictured, .{ .ctx = &d, .event = Box.event });
    e.cancel(4);
    try std.testing.expectEqual(Reason.cancelled, d.wait());
    try std.testing.expectEqual(@as(?u32, null), d.prefilled);
    try std.testing.expectEqual(forgets + 1, tower.forgets.load(.acquire));
    try std.testing.expectEqual(@as(u32, 2), tower.starts.load(.acquire));
}

test "an image job without an off-thread tower is admitted as before" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var tower: Tower = .{};
    var host = LaneHost.init(gpa, io, &core, .{ .lanes = 1, .vision = tower.vision(false) });
    try host.start();
    defer host.stop();
    var seq: std.atomic.Value(u32) = .init(0);
    const prompt = [_]u32{ 9, 9, 3, 1 };
    const images = [_]Image{.{ .at = 0, .tokens = 2, .bytes = "picture" }};
    const pictured: Request = .{ .prompt = &prompt, .images = &images, .max_tokens = 4 };
    var a: Box = .{ .seq = &seq };
    try host.engine().submit(1, &pictured, .{ .ctx = &a, .event = Box.event });
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(@as(u32, 0), tower.starts.load(.acquire));
}

//! LaneHost's submit-time hint asks the store only (peekSealed); the family's checks run at admission (the fake).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

/// The fake's snapshots, checked: `stale_at` lost its rows, `refuse_at` is refused; each check notes its thread.
const Family = struct {
    f: *lanes.fake.Fake,
    store: ?*pc.Store = null,
    submitter: std.Thread.Id = 0,
    checks: usize = 0,
    on_submitter: usize = 0,
    stale_at: u32 = 0,
    refuse_at: u32 = 0,

    fn of(ptr: *anyopaque) *Family {
        return @ptrCast(@alignCast(ptr));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        return of(ptr).f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        of(ptr).f.drop(saved);
    }
    /// The kept position of `saved` (the store's entry for it), and the check counted.
    fn checked(x: *Family, saved: pc.Saved) u32 {
        x.checks += 1;
        if (std.Thread.getCurrentId() == x.submitter) x.on_submitter += 1;
        for (x.store.?.entries.items) |e| if (e.saved == saved) return e.at;
        return 0;
    }
    fn resumes(ptr: *anyopaque, saved: pc.Saved, _: u32) bool {
        const at = of(ptr).checked(saved);
        return at != of(ptr).stale_at and at != of(ptr).refuse_at;
    }
    fn live(ptr: *anyopaque, saved: pc.Saved) bool {
        return of(ptr).checked(saved) != of(ptr).stale_at;
    }
};

/// An off-thread tower with every request's rows at once: it notes where each encode was told to start.
const Hint = struct {
    froms: [8]u32 = @splat(99),
    n: usize = 0,

    fn start(ctx: *anyopaque, _: []const api.Image, from: u32, _: lanes.stream.Wake) void {
        const t: *Hint = @ptrCast(@alignCast(ctx));
        if (t.n < t.froms.len) t.froms[t.n] = from;
        t.n += 1;
    }
    fn ready(_: *anyopaque, _: []const api.Image) bool {
        return true;
    }
    fn forget(_: *anyopaque, _: []const api.Image) void {}
    fn prepare(_: *anyopaque, _: std.mem.Allocator, _: []const u8, _: *[]const u8) lanes.stream.Vision.PrepareError!lanes.stream.Vision.Prepared {
        return error.BadImage;
    }
    fn vision(t: *Hint) lanes.stream.Vision {
        return .{ .ctx = t, .placeholder = "<image>", .token = 11, .prepare = prepare, .encode = .{ .ctx = t, .start = start, .ready = ready, .forget = forget } };
    }
};

const Box = struct {
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    cached: ?u32 = null,
    done: ?Reason = null,

    fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        switch (e.*) {
            .prefilled => |c| b.cached = c,
            .tokens => |t| b.tokens.appendSlice(std.testing.allocator, t) catch {},
            .finished => |f| b.done = f.reason,
        }
    }
    fn wait(b: *Box) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// A host over the fake with an image-keyed store whose family checks are `fam`'s.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine = undefined,
    fam: Family = undefined,
    vt: pc.Snapshots.VTable = undefined,
    store: pc.Store = undefined,
    hint: Hint = .{},
    host: LaneHost = undefined,

    fn start(r: *Rig) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa };
        r.clock = .{};
        r.hint = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.fam = .{ .f = &r.target, .submitter = std.Thread.getCurrentId() };
        r.vt = .{ .bytes = Family.bytes, .save = Family.save, .restore = Family.restore, .drop = Family.drop, .resumes = Family.resumes, .live = Family.live };
        r.store = pc.Store.init(gpa, .{ .ptr = &r.fam, .vtable = &r.vt }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
        r.fam.store = &r.store;
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 2, .vision = r.hint.vision() });
        r.host.cache = &r.store;
        r.host.cache_images = true;
        try r.host.start();
    }

    fn stop(r: *Rig) void {
        r.host.stop();
        r.store.deinit();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }

    /// One request with the picture at 2 (3 positions); its reply's tokens appended to `out`.
    fn turn(r: *Rig, id: Id, prompt: []const u32, out: *std.ArrayList(u32)) !?u32 {
        const im = [_]api.Image{.{ .at = 2, .tokens = 3, .bytes = "picture A", .digest = pc.imageDigest("picture A") }};
        var b: Box = .{};
        defer b.tokens.deinit(std.testing.allocator);
        const req: Request = .{ .prompt = prompt, .images = &im, .max_tokens = 6, .history_len = @intCast(prompt.len - 2) };
        try r.host.engine().submit(id, &req, .{ .ctx = &b, .event = Box.event });
        try std.testing.expectEqual(Reason.length, b.wait());
        try out.appendSlice(std.testing.allocator, b.tokens.items);
        return b.cached;
    }
};

/// The next turn: `prev` (its prompt and reply) and a tool result.
fn next(prev: []const u32, out: *std.ArrayList(u32)) !void {
    try out.appendSlice(std.testing.allocator, prev);
    try out.appendSlice(std.testing.allocator, &.{ 7, 7, 5, 3 });
}

test "a stale sealed image state: the submit's hint calls no family check and drops nothing; admission drops it" {
    var r: Rig = undefined;
    try r.start();
    defer r.stop();
    const gpa = std.testing.allocator;
    const img: u32 = 11;
    var t1: std.ArrayList(u32) = .empty;
    defer t1.deinit(gpa);
    try t1.appendSlice(gpa, &.{ 3, 1, img, img, img, 9, 2, 6, 5, 3 });
    try std.testing.expectEqual(@as(?u32, 0), try r.turn(1, t1.items, &t1));
    const kept = r.store.entries.items.len;
    try std.testing.expect(kept >= 1);
    const at = r.store.entries.items[kept - 1].at;
    r.fam.stale_at = at; // another stream wrote over its rows
    r.fam.checks = 0;
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try next(t1.items, &t2);
    const cached = try r.turn(2, t2.items, &t2);
    try std.testing.expectEqual(at, r.hint.froms[1]); // the hint saw the sealed state as it was, unchecked
    try std.testing.expectEqual(@as(usize, 0), r.fam.on_submitter); // no family check on the submitting thread
    try std.testing.expect(r.fam.checks > 0); // the engine thread's lookup asked the family
    try std.testing.expectEqual(@as(u64, 1), r.store.counts.evicted); // and dropped the lost state there
    try std.testing.expect(cached.? < at);
}

test "a live sealed image state the family refuses: the pass resumes an earlier one, and the refused one stays" {
    var r: Rig = undefined;
    try r.start();
    defer r.stop();
    const gpa = std.testing.allocator;
    const img: u32 = 11;
    var t1: std.ArrayList(u32) = .empty;
    defer t1.deinit(gpa);
    try t1.appendSlice(gpa, &.{ 3, 1, img, img, img, 9, 2, 6, 5, 3 });
    _ = try r.turn(1, t1.items, &t1);
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try next(t1.items, &t2);
    const first = try r.turn(2, t2.items, &t2);
    try std.testing.expect(first.? > 0); // turn 2 resumed turn 1's state
    var longest: u32 = 0;
    for (r.store.entries.items) |e| longest = @max(longest, e.at);
    try std.testing.expect(longest > first.?); // turn 2 kept a longer one
    r.fam.refuse_at = longest;
    const entries = r.store.entries.items.len;
    var t3: std.ArrayList(u32) = .empty;
    defer t3.deinit(gpa);
    try next(t2.items, &t3);
    const cached = try r.turn(3, t3.items, &t3);
    try std.testing.expectEqual(first, cached); // the earlier state, not the refused one
    try std.testing.expectEqual(@as(usize, 0), r.fam.on_submitter);
    try std.testing.expectEqual(@as(u64, 0), r.store.counts.evicted);
    var still = false;
    for (r.store.entries.items) |e| still = still or e.at == longest;
    try std.testing.expect(still and r.store.entries.items.len >= entries);
}

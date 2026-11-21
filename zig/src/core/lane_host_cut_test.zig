//! LaneHost's cut-in: a short request reads whole between a paused prompt pass's slices (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;

const decoder = [_]u32{ 2, 7, 1, 8 }; // request 1: it decodes throughout
const long = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8 }; // request 2: its pass pauses between slices
const short = [_]u32{ 9, 7, 9, 3, 2 }; // request 3: sent during the long pass's second chunk
const later = [_]u32{ 6, 2, 8, 3, 1, 8 }; // request 4: sent while request 3's cut-in is being refused for room

/// What happens once the short request has two tokens out (from the host's thread, so the long pass is still paused).
const Then = enum { nothing, cancel_short, cancel_long, fail_round };

/// Each request's tokens and end, the passes' chunks by prompt length, and the test's actions on the host's thread.
const Seen = struct {
    mutex: std.Io.Mutex = .init,
    target: *lanes.fake.Fake,
    e: api.Engine = undefined,
    short_request: Request = .{ .prompt = &short, .max_tokens = 4 },
    later_request: Request = .{ .prompt = &later, .max_tokens = 4 },
    refuse: usize = 0, // the fake refuses this many passes for room from the short request's arrival
    order: [2]usize = .{ 0, 0 }, // the prompt lengths of the first two short passes to begin
    began: usize = 0,
    cancel_short_at: ?usize = null, // the short request is cancelled in this chunk of its own pass
    then: Then = .nothing,
    acted: bool = false,
    long_chunks: usize = 0,
    short_chunks: usize = 0,
    long_at_short: ?usize = null, // the long pass's chunks read when the short's pass began
    long_at_short_done: ?usize = null, // and when the short request ended
    short_rounds: [2]u64 = .{ 0, 0 }, // the rounds run by the short pass's first and last chunk
    tokens: [5]std.ArrayList(u32) = @splat(.empty),
    cached: [5]?u32 = @splat(null),
    done: [5]?Reason = @splat(null),

    fn hook(ctx: *anyopaque, s: *lanes.stream.Stream, chunk: usize) void {
        const q: *Seen = @ptrCast(@alignCast(ctx));
        q.mutex.lockUncancelable(std.testing.io);
        var submit = false;
        var cancel = false;
        if (s.prompt_len == long.len) {
            q.long_chunks += 1;
            submit = chunk == 1;
        } else if (s.prompt_len == short.len) {
            if (q.short_chunks == 0) {
                q.long_at_short = q.long_chunks;
                q.short_rounds[0] = q.target.rounds;
            }
            q.short_rounds[1] = q.target.rounds;
            q.short_chunks += 1;
            cancel = q.cancel_short_at == chunk;
        }
        if (chunk == 0 and (s.prompt_len == short.len or s.prompt_len == later.len) and q.began < 2) {
            q.order[q.began] = s.prompt_len;
            q.began += 1;
        }
        q.mutex.unlock(std.testing.io);
        if (submit) {
            q.target.refuse_full = q.refuse;
            q.e.submit(3, &q.short_request, .{ .ctx = q, .event = event }) catch unreachable;
        }
        if (cancel) q.e.cancel(3);
    }

    /// The fake refuses a pass for room (the host's thread, inside that request's admission): request 4 arrives.
    fn refused(ctx: *anyopaque, _: *lanes.stream.Stream) void {
        const q: *Seen = @ptrCast(@alignCast(ctx));
        q.e.submit(4, &q.later_request, .{ .ctx = q, .event = event }) catch unreachable;
    }

    fn event(ctx: *anyopaque, id: Id, e: *const Event) void {
        const q: *Seen = @ptrCast(@alignCast(ctx));
        q.mutex.lockUncancelable(std.testing.io);
        const i: usize = @intCast(id);
        switch (e.*) {
            .prefilled => |c| q.cached[i] = c,
            .tokens => |t| q.tokens[i].appendSlice(std.testing.allocator, t) catch {},
            .finished => |f| {
                q.done[i] = f.reason;
                if (i == 3) q.long_at_short_done = q.long_chunks;
            },
        }
        const act = !q.acted and q.then != .nothing and q.tokens[3].items.len >= 2;
        if (act) q.acted = true;
        q.mutex.unlock(std.testing.io);
        if (act) switch (q.then) {
            .nothing => {},
            .cancel_short => q.e.cancel(3),
            .cancel_long => q.e.cancel(2),
            .fail_round => q.target.fail_round.store(true, .release),
        };
    }

    fn count(q: *Seen, id: usize) usize {
        q.mutex.lockUncancelable(std.testing.io);
        defer q.mutex.unlock(std.testing.io);
        return q.tokens[id].items.len;
    }

    fn waitDone(q: *Seen, id: usize) ?Reason {
        for (0..10000) |_| {
            q.mutex.lockUncancelable(std.testing.io);
            const d = q.done[id];
            q.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
        return null;
    }
};

/// The fake's greedy reply to `prompt`: its draws from the history each token extends.
fn fakeReply(prompt: []const u32, comptime n: usize) [n]u32 {
    var hist: [64]u32 = undefined;
    @memcpy(hist[0..prompt.len], prompt);
    var out: [n]u32 = undefined;
    for (&out, 0..) |*t, i| {
        t.* = lanes.fake.next(hist[0 .. prompt.len + i], null, prompt.len + i);
        hist[prompt.len + i] = t.*;
    }
    return out;
}

/// A lane host of three lanes over the fake: every pass of n chunks, one a slice, cut-ins up to `cut_in` tokens.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock,
    core: lanes.Engine,
    host: LaneHost,
    seen: Seen,
    stopped: bool = false,

    fn start(r: *Rig, cut_in: u32, n: usize) !void {
        const gpa = std.testing.allocator;
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa, .prefill_chunks = n, .slice_chunks = 1 };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.backend(), r.clock.clock());
        r.seen = .{ .target = &r.target };
        r.target.prefill_hook = Seen.hook;
        r.target.prefill_hook_ctx = &r.seen;
        r.host = LaneHost.init(gpa, std.testing.io, &r.core, .{ .lanes = 3, .fill_slice_ns = std.time.ns_per_s, .fill_cut_in = cut_in });
        r.stopped = false;
        try r.host.start();
        r.seen.e = r.host.engine();
    }

    fn stop(r: *Rig) void {
        if (!r.stopped) r.host.stop();
        r.stopped = true;
    }

    fn deinit(r: *Rig) void {
        r.stop();
        for (&r.seen.tokens) |*t| t.deinit(std.testing.allocator);
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(std.testing.allocator);
    }

    /// Request 1 decoding, then request 2's long prompt, whose pass pauses (request 3 arrives in its second chunk).
    fn begin(r: *Rig, long_tokens: u32) !void {
        const sink: api.Sink = .{ .ctx = &r.seen, .event = Seen.event };
        try r.seen.e.submit(1, &.{ .prompt = &decoder, .max_tokens = 1 << 30 }, sink);
        while (r.seen.count(1) < 3) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        const second: *const Request = if (long_tokens == 6) &.{ .prompt = &long, .max_tokens = 6 } else &.{ .prompt = &long, .max_tokens = 1 << 30 };
        try r.seen.e.submit(2, second, sink);
    }

    /// The decoding request cancelled: no lane and no paused pass may be left.
    fn end(r: *Rig) !void {
        r.seen.e.cancel(1);
        try std.testing.expectEqual(@as(?Reason, Reason.cancelled), r.seen.waitDone(1));
        r.stop();
        try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
        try std.testing.expect(r.host.filling == null);
    }
};

const chunks = 40;

test "a short request cuts in between a paused pass's slices and reads whole; the pass then reads on" {
    var r: Rig = undefined;
    try r.start(8, chunks);
    defer r.deinit();
    try r.begin(6);
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(3));
    try std.testing.expectEqualSlices(u32, &fakeReply(&short, 4), r.seen.tokens[3].items);
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(2));
    try std.testing.expectEqualSlices(u32, &fakeReply(&long, 6), r.seen.tokens[2].items);
    const q = &r.seen;
    // the short pass began after the long pass's second chunk, and the short request ended before that pass was in
    try std.testing.expectEqual(@as(?usize, 2), q.long_at_short);
    try std.testing.expect(q.long_at_short_done.? < chunks);
    // its own pass ran whole: every chunk, no round between its first and its last
    try std.testing.expectEqual(@as(usize, chunks), q.short_chunks);
    try std.testing.expectEqual(q.short_rounds[0], q.short_rounds[1]);
    try std.testing.expectEqual(@as(usize, chunks), q.long_chunks);
    try r.end();
}

test "without a cut-in size the short request waits for the paused pass, as before" {
    var r: Rig = undefined;
    try r.start(0, chunks);
    defer r.deinit();
    try r.begin(6);
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(3));
    try std.testing.expectEqualSlices(u32, &fakeReply(&short, 4), r.seen.tokens[3].items);
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(2));
    try std.testing.expectEqualSlices(u32, &fakeReply(&long, 6), r.seen.tokens[2].items);
    try std.testing.expectEqual(@as(?usize, chunks), r.seen.long_at_short); // the long pass was in first
    try r.end();
}

test "a request longer than the cut-in size, or with a picture's rows, does not cut in" {
    for ([_]bool{ false, true }) |picture| {
        var r: Rig = undefined;
        try r.start(if (picture) 8 else 4, chunks);
        defer r.deinit();
        const rows = [_]api.Image{.{ .at = 1, .tokens = 1, .bytes = "ab", .rows = true }};
        if (picture) r.seen.short_request.images = &rows;
        try r.begin(6);
        try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(3));
        try std.testing.expectEqual(@as(?usize, chunks), r.seen.long_at_short);
        try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(2));
        try r.end();
    }
}

test "a cut-in refused for room stays queued and runs later, with a stream decoding and with none" {
    for ([_]bool{ true, false }) |decoding| {
        var r: Rig = undefined;
        try r.start(8, chunks);
        defer r.deinit();
        r.seen.refuse = 1;
        const sink: api.Sink = .{ .ctx = &r.seen, .event = Seen.event };
        if (decoding) try r.begin(6) else try r.seen.e.submit(2, &.{ .prompt = &long, .max_tokens = 6 }, sink);
        try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(3));
        try std.testing.expectEqualSlices(u32, &fakeReply(&short, 4), r.seen.tokens[3].items);
        try std.testing.expectEqual(@as(usize, 0), r.target.refuse_full);
        try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(2));
        try std.testing.expectEqualSlices(u32, &fakeReply(&long, 6), r.seen.tokens[2].items);
        // with no stream decoding the lone pass pauses too, and the refused request cuts in at its next pause
        if (!decoding) try std.testing.expect(r.seen.long_at_short.? > 2 and r.seen.long_at_short.? < chunks);
        if (decoding) try r.end() else {
            r.stop();
            try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
        }
    }
}

test "a background cut-in refused for room goes back behind a foreground request sent meanwhile" {
    var r: Rig = undefined;
    try r.start(8, chunks);
    defer r.deinit();
    r.seen.refuse = 1;
    r.seen.short_request.background = true;
    r.target.refuse_hook = Seen.refused;
    try r.seen.e.submit(2, &.{ .prompt = &long, .max_tokens = 6 }, .{ .ctx = &r.seen, .event = Seen.event });
    for ([_]usize{ 4, 3, 2 }) |id| try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(id));
    try std.testing.expectEqualSlices(u32, &fakeReply(&later, 4), r.seen.tokens[4].items);
    try std.testing.expectEqualSlices(u32, &fakeReply(&short, 4), r.seen.tokens[3].items);
    // the foreground request cut in first, though the background one had been taken from the queue before it came
    try std.testing.expectEqualSlices(usize, &.{ later.len, short.len }, &r.seen.order);
    r.stop();
    try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
}

test "a cut-in cancelled in its own pass, or once it decodes, leaves the paused pass to read on" {
    for ([_]bool{ true, false }) |in_pass| {
        var r: Rig = undefined;
        try r.start(8, chunks);
        defer r.deinit();
        if (in_pass) r.seen.cancel_short_at = 1 else r.seen.then = .cancel_short;
        r.seen.short_request.max_tokens = 1 << 30;
        try r.begin(6);
        try std.testing.expectEqual(@as(?Reason, Reason.cancelled), r.seen.waitDone(3));
        if (in_pass) try std.testing.expectEqual(@as(usize, 0), r.seen.count(3));
        try std.testing.expect(r.seen.long_at_short_done.? < chunks);
        try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(2));
        try std.testing.expectEqualSlices(u32, &fakeReply(&long, 6), r.seen.tokens[2].items);
        try std.testing.expectEqual(@as(usize, chunks), r.seen.long_chunks);
        try r.end();
    }
}

test "the paused pass cancelled beside its cut-in frees its lane; the cut-in decodes to its end" {
    var r: Rig = undefined;
    try r.start(8, chunks);
    defer r.deinit();
    r.seen.then = .cancel_long;
    r.seen.short_request.max_tokens = 12;
    try r.begin(6);
    try std.testing.expectEqual(@as(?Reason, Reason.cancelled), r.seen.waitDone(2));
    try std.testing.expectEqual(@as(usize, 0), r.seen.count(2));
    try std.testing.expect(r.seen.long_chunks < chunks);
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(3));
    try std.testing.expectEqualSlices(u32, &fakeReply(&short, 12), r.seen.tokens[3].items);
    try r.end();
}

test "a failed round with a pass paused and a cut-in decoding ends all three; the next request runs" {
    var r: Rig = undefined;
    try r.start(8, chunks);
    defer r.deinit();
    r.seen.then = .fail_round;
    r.seen.short_request.max_tokens = 1 << 30;
    try r.begin(6);
    for (1..4) |id| try std.testing.expectEqual(@as(?Reason, Reason.failed), r.seen.waitDone(id));
    try std.testing.expect(r.seen.long_chunks < chunks);
    try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
    try r.seen.e.submit(4, &.{ .prompt = &decoder, .max_tokens = 5 }, .{ .ctx = &r.seen, .event = Seen.event });
    try std.testing.expectEqual(@as(?Reason, Reason.length), r.seen.waitDone(4));
    try std.testing.expectEqualSlices(u32, &fakeReply(&decoder, 5), r.seen.tokens[4].items);
    r.stop();
    try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
    try std.testing.expect(r.host.filling == null);
}

test "stopping the host with a pass paused and a cut-in decoding frees every lane" {
    var r: Rig = undefined;
    try r.start(8, 1 << 20); // (a pass this long is still paused when the test's thread stops the host)
    defer r.deinit();
    r.seen.short_request.max_tokens = 1 << 30;
    try r.begin(0);
    while (r.seen.count(3) < 2) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    r.stop();
    for (1..4) |id| try std.testing.expectEqual(@as(?Reason, Reason.cancelled), r.seen.done[id]);
    try std.testing.expect(r.seen.long_chunks < 1 << 20);
    try std.testing.expectEqual(@as(usize, 0), r.target.lanes.count());
    try std.testing.expect(r.host.filling == null);
}

/// A checking allocator: each allocation on pages of its own, a freed one's pages unreadable from then on.
const Fence = struct {
    const page = std.heap.page_size_min;
    const vtable: std.mem.Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };
    fn allocator() std.mem.Allocator {
        return .{ .ptr = undefined, .vtable = &vtable };
    }
    fn alloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        if (alignment.toByteUnits() > page) return null;
        const mem = std.posix.mmap(null, std.mem.alignForward(usize, @max(len, 1), page), .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0) catch return null;
        return mem.ptr;
    }
    fn resize(_: *anyopaque, mem: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
        return new_len <= std.mem.alignForward(usize, @max(mem.len, 1), page);
    }
    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }
    fn free(_: *anyopaque, mem: []u8, _: std.mem.Alignment, _: usize) void {
        const p: *align(page) anyopaque = @ptrCast(@alignCast(mem.ptr));
        _ = std.c.mprotect(p, std.mem.alignForward(usize, @max(mem.len, 1), page), .{});
    }
};

const Snaps = struct {
    fn bytes(_: *anyopaque, at: u32) u64 {
        return 64 + 4 * @as(u64, at);
    }
    fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        return f.save(@ptrCast(@alignCast(owner.?)), at);
    }
    fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
        return error.BackendRestores;
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
        f.drop(saved);
    }
};

const turn1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
const turn2 = turn1 ++ [_]u32{ 8, 4, 6, 2, 7, 7, 5, 3 }; // A: the conversation's next turn, resumed from its state at 8

/// A resumes a kept state and pauses; B cuts in and its mark takes that state's room; A reads on or is cancelled.
fn cutInEvictsSource(cancel_a: bool) !void {
    const gpa = std.testing.allocator;
    var other: [30]u32 = undefined; // B: another conversation, short enough to cut in
    for (&other, 0..) |*x, i| x.* = @intCast(i % 7 + 11);
    const Ctx = struct {
        q: Seen,
        b: Request,
        cancel_a: bool,
        fn hook(ptr: *anyopaque, s: *lanes.stream.Stream, chunk: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            if (s.prompt_len != turn2.len) return;
            if (chunk == 0) c.q.e.submit(3, &c.b, .{ .ctx = &c.q, .event = Seen.event }) catch unreachable;
            if (chunk == 2 and c.cancel_a) c.q.e.cancel(2);
        }
    };
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4, .slice_chunks = 1 };
    defer target.deinit();
    var ctx: Ctx = .{ .q = .{ .target = &target }, .b = .{ .prompt = &other, .max_tokens = 4, .history_len = 28 }, .cancel_a = cancel_a };
    defer for (&ctx.q.tokens) |*t| t.deinit(gpa);
    target.prefill_hook = Ctx.hook;
    target.prefill_hook_ctx = &ctx;
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    // entries on fenced pages (a freed one's read faults); 240 bytes: A's source 96 + its mark 128, B's mark 176
    var store = pc.Store.init(Fence.allocator(), .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 240);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2, .fill_slice_ns = std.time.ns_per_s, .fill_cut_in = 64 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const q = &ctx.q;
    q.e = host.engine();
    const sink: api.Sink = .{ .ctx = q, .event = Seen.event };
    try q.e.submit(1, &.{ .prompt = &turn1, .max_tokens = 2, .history_len = 8 }, sink);
    try std.testing.expectEqual(@as(?Reason, Reason.length), q.waitDone(1));
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len); // A's source, at 8
    try q.e.submit(2, &.{ .prompt = &turn2, .max_tokens = 6, .history_len = 16 }, sink);
    const a_end = q.waitDone(2);
    try std.testing.expectEqual(@as(?Reason, Reason.length), q.waitDone(3));
    try std.testing.expect(store.counts.evicted >= 1); // B's mark took the source's room
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits); // A's resume was reported before B's lookup
    if (cancel_a) return std.testing.expectEqual(@as(?Reason, Reason.cancelled), a_end);
    try std.testing.expectEqual(@as(?Reason, Reason.length), a_end);
    try std.testing.expectEqual(@as(?u32, 8), q.cached[2]);
    try std.testing.expectEqualSlices(u32, &fakeReply(&turn2, 6), q.tokens[2].items);
}

test "a cut-in whose keep evicts the state the paused pass resumed from leaves that pass nothing freed to read" {
    try cutInEvictsSource(false);
    try cutInEvictsSource(true);
}

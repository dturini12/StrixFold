//! LaneHost with the prompt cache: a conversation's next turn resumes from its kept prompt state (the fake backend).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const pc = @import("prompt_cache.zig");
const LaneHost = api.LaneHost;
const Id = api.Id;
const Event = api.Event;
const Request = api.Request;
const Reason = api.Reason;

test "a lane host resumes a conversation from its kept prompt state and reports the cached tokens" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores; // lane-core backends restore inside their own prompt pass
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const f: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            f.drop(saved);
        }
    };
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    var t2: std.ArrayList(u32) = .empty; // the next turn: the history, the reply, a tool result, a new generation prompt
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    var history: std.ArrayList(u32) = .empty; // the fake's tokens read the whole history: any slip in it changes them
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    try std.testing.expectEqual(@as(usize, 2), store.entries.items.len);
}

test "a turn's mark evicting the state it resumed from: the hit is reported first and the entry freed once" {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    const Snaps = struct { // a budget of one state: each keep records the hits the store had counted by then
        f: *lanes.fake.Fake,
        store: ?*pc.Store = null,
        hits_at_save: [4]u64 = @splat(0),
        saves: usize = 0,
        live: usize = 0,
        fn of(ptr: *anyopaque) *@This() {
            return @ptrCast(@alignCast(ptr));
        }
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!pc.Saved {
            const x = of(ptr);
            x.hits_at_save[x.saves] = x.store.?.counts.hits;
            x.saves += 1;
            x.live += 1;
            return x.f.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            of(ptr).live -= 1;
            of(ptr).f.drop(saved);
        }
    };
    var snaps: Snaps = .{ .f = &target };
    var store = pc.Store.init(gpa, .{ .ptr = &snaps, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 150);
    defer store.deinit();
    snaps.store = &store;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    var t2: std.ArrayList(u32) = .empty; // 18 tokens: its history's state (128 bytes) and turn 1's (96) pass the budget together
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    try std.testing.expectEqual(@as(?u32, 8), b2.cached);
    try std.testing.expectEqual(@as(usize, 2), snaps.saves);
    try std.testing.expectEqual(@as(u64, 1), snaps.hits_at_save[1]); // the resume was reported before the keep evicted it
    try std.testing.expectEqual(@as(u64, 1), store.counts.evicted);
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len);
    try std.testing.expectEqual(@as(u32, 16), store.entries.items[0].at);
    try std.testing.expectEqual(@as(usize, 1), snaps.live);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, t2.items);
    for (b2.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

/// One conversation over a fresh lane host and store: turn 1 with picture `first`, turn 2 (turn 1, four reply tokens,
/// a tool result) with picture `second`; returns turn 2's cached tokens and the store's entry count after it.
fn imageTurns(cache_images: bool, first: []const u8, second: []const u8) !struct { cached: ?u32, entries: usize, hits: u64 } {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
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
    var store = pc.Store.init(gpa, .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 1 << 20);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    host.cache_images = cache_images;
    try host.start();
    defer host.stop();
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    const e = host.engine();
    const img: u32 = 11; // the image token: the span's three positions hold it whatever the picture
    const t1 = [_]u32{ 3, 1, img, img, img, 9, 2, 6, 5, 3 };
    const im1 = [_]api.Image{.{ .at = 2, .tokens = 3, .bytes = first }};
    var b1: Box = .{};
    defer b1.tokens.deinit(gpa);
    const r1: Request = .{ .prompt = &t1, .images = &im1, .max_tokens = 6, .history_len = 8 };
    try e.submit(1, &r1, .{ .ctx = &b1, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b1.wait());
    try std.testing.expectEqual(@as(?u32, 0), b1.cached);
    var t2: std.ArrayList(u32) = .empty;
    defer t2.deinit(gpa);
    try t2.appendSlice(gpa, &t1);
    try t2.appendSlice(gpa, b1.tokens.items[0..4]);
    try t2.appendSlice(gpa, &.{ 7, 7, 5, 3 });
    const im2 = [_]api.Image{.{ .at = 2, .tokens = 3, .bytes = second, .digest = pc.imageDigest(second) }}; // as the server sets it
    var b2: Box = .{};
    defer b2.tokens.deinit(gpa);
    const r2: Request = .{ .prompt = t2.items, .images = &im2, .max_tokens = 12, .history_len = @intCast(t2.items.len - 2) };
    try e.submit(2, &r2, .{ .ctx = &b2, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b2.wait());
    return .{ .cached = b2.cached, .entries = store.entries.items.len, .hits = store.counts.hits };
}

test "a conversation with a picture resumes past it when the next turn carries the same picture, and only then" {
    const same = try imageTurns(true, "picture A", "picture A"); // turn 1's digest taken by the host, turn 2's by the server
    try std.testing.expectEqual(@as(?u32, 8), same.cached);
    try std.testing.expectEqual(@as(u64, 1), same.hits);
    const other = try imageTurns(true, "picture A", "picture B"); // the same tokens, another picture
    try std.testing.expectEqual(@as(?u32, 0), other.cached);
    try std.testing.expectEqual(@as(u64, 0), other.hits);
    const off = try imageTurns(false, "picture A", "picture A"); // a backend that keys no pictures: nothing kept
    try std.testing.expectEqual(@as(?u32, 0), off.cached);
    try std.testing.expectEqual(@as(usize, 0), off.entries);
}

/// A checking allocator: every allocation on pages of its own, and a freed one's pages made unreadable (never mapped
/// again), so a read through a stale pointer faults instead of reading what the slot holds next.
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
        _ = std.c.mprotect(p, std.mem.alignForward(usize, @max(mem.len, 1), page), .{}); // PROT_NONE
    }
};

/// A resumed request's prompt pass in pieces (TF_FILL_CHUNKS, TF_FILL_CUT_IN), a short prompt cutting in after its first
/// piece and evicting the state it resumed from; then the pass completes, or is cancelled in a later piece.
fn cutInEvictsSource(cancel_a: bool) !void {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
    defer cfg.deinit(gpa);
    const t1 = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3 };
    const t2 = t1 ++ [_]u32{ 8, 4, 6, 2, 7, 7, 5, 3 }; // A: t1's conversation, its next turn (resumes the state at 8)
    var tb: [30]u32 = undefined; // B: another conversation, short enough to cut in, whose marks need A's source's room
    for (&tb, 0..) |*x, i| x.* = @intCast(i % 7 + 11);
    const Box = struct {
        mutex: std.Io.Mutex = .init,
        tokens: std.ArrayList(u32) = .empty,
        cached: ?u32 = null,
        done: ?Reason = null,
        fn event(ctx: *anyopaque, _: Id, e: *const Event) void {
            const b: *@This() = @ptrCast(@alignCast(ctx));
            b.mutex.lockUncancelable(std.testing.io);
            defer b.mutex.unlock(std.testing.io);
            switch (e.*) {
                .prefilled => |c| b.cached = c,
                .tokens => |t| b.tokens.appendSlice(gpa, t) catch {},
                .finished => |f| b.done = f.reason,
            }
        }
        fn wait(b: *@This()) Reason {
            while (true) {
                b.mutex.lockUncancelable(std.testing.io);
                const d = b.done;
                b.mutex.unlock(std.testing.io);
                if (d) |r| return r;
                std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
            }
        }
    };
    var ba: Box = .{};
    defer ba.tokens.deinit(gpa);
    var bb: Box = .{};
    defer bb.tokens.deinit(gpa);
    var b0: Box = .{};
    defer b0.tokens.deinit(gpa);
    const r_b: Request = .{ .prompt = &tb, .max_tokens = 4, .history_len = 28 };
    // on the fake's prompt chunks (the host's thread): B is submitted during A's first piece, so it cuts in right after
    // it; with cancel_a, A is cancelled in its third piece
    const Ctx = struct {
        e: api.Engine = undefined,
        bb: *Box,
        r_b: *const Request,
        a_len: usize,
        cancel_a: bool,
        fn hook(ptr: *anyopaque, s: *lanes.stream.Stream, piece: usize) void {
            const c: *@This() = @ptrCast(@alignCast(ptr));
            if (s.prompt_len != c.a_len) return;
            if (piece == 0) c.e.submit(3, c.r_b, .{ .ctx = c.bb, .event = Box.event }) catch unreachable;
            if (piece == 2 and c.cancel_a) c.e.cancel(2);
        }
    };
    var ctx: Ctx = .{ .bb = &bb, .r_b = &r_b, .a_len = t2.len, .cancel_a = cancel_a };
    var target: lanes.fake.Fake = .{ .gpa = gpa, .prefill_chunks = 4, .prefill_hook = Ctx.hook, .prefill_hook_ctx = &ctx };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    core.fill_chunks = 1;
    core.cut_in = 64; // every pass in pieces, B (30 tokens) cuts in
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
    // the store's entries on fenced pages: A's report of a freed entry faults. 240 bytes: A's source (96) and A's own
    // mark (128) fit, B's mark (176) needs the source's room
    var store = pc.Store.init(Fence.allocator(), .{ .ptr = &target, .vtable = &.{ .bytes = Snaps.bytes, .save = Snaps.save, .restore = Snaps.restore, .drop = Snaps.drop } }, .{ .min_gap = 4, .min_prompt = 0 }, 240);
    defer store.deinit();
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    host.cache = &store;
    try host.start();
    defer host.stop();
    const e = host.engine();
    ctx.e = e;
    const r0: Request = .{ .prompt = &t1, .max_tokens = 2, .history_len = 8 };
    try e.submit(1, &r0, .{ .ctx = &b0, .event = Box.event });
    try std.testing.expectEqual(Reason.length, b0.wait());
    try std.testing.expectEqual(@as(usize, 1), store.entries.items.len); // A's source, at 8
    const r_a: Request = .{ .prompt = &t2, .max_tokens = 6, .history_len = 16 };
    try e.submit(2, &r_a, .{ .ctx = &ba, .event = Box.event });
    const ra = ba.wait();
    try std.testing.expectEqual(Reason.length, bb.wait());
    try std.testing.expect(store.counts.evicted >= 1); // B's mark took the source's room
    if (cancel_a) {
        try std.testing.expectEqual(Reason.cancelled, ra);
        return;
    }
    try std.testing.expectEqual(Reason.length, ra);
    try std.testing.expectEqual(@as(?u32, 8), ba.cached); // A resumed the state at 8, which the hit reports
    try std.testing.expectEqual(@as(u64, 1), store.counts.hits);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(gpa);
    try history.appendSlice(gpa, &t2);
    for (ba.tokens.items) |t| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), t);
        try history.append(gpa, t);
    }
}

test "a pass in pieces reports its restore after its first piece: a cut-in that evicts its source leaves it nothing to read" {
    try cutInEvictsSource(false);
}

test "the same, the pass cancelled in a later piece" {
    try cutInEvictsSource(true);
}

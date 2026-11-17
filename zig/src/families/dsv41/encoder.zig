//! Rank 0's picture encodes on a thread of their own (HIP), so the engine thread keeps decoding the other streams.
const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;
const Image = lanes.stream.Image;
const Wake = lanes.stream.Wake;

/// The tower the thread drives: `bind` once on that thread, `rows` appends a picture's span rows, returns its tokens.
pub const Tower = struct {
    ctx: *anyopaque,
    bind: *const fn (ctx: *anyopaque) anyerror!void,
    rows: *const fn (ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8)) anyerror!usize,
};

const Job = struct {
    key: usize, // the request's images.ptr
    payloads: [][]align(16) u8, // copies of the pictures to encode (empty: rows came ready, skipped, or encoded)
    tokens: []usize, // each picture's span, as its encode returned it
    ranges: [][2]usize, // where each image's rows begin and end in `rows`
    skipped: []bool, // a resume covers its span: encoded only when a prompt pass asks for it (take)
    rows: std.ArrayList(u8) = .empty,
    state: enum { queued, running, done } = .queued,
    urgent: bool = false, // a prompt pass waits for it: the thread takes it first
    err: ?anyerror = null,
    gone: bool = false, // forgotten while it ran: the thread frees it
    wake: Wake,
};

pub const Encoder = struct {
    gpa: Allocator,
    io: std.Io,
    tower: Tower,
    cap: usize, // a request's image rows at most (Model.image_rows_reserved): more fail at once, nothing encoded
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done: std.Io.Condition = .init, // a job finished (a prompt pass waits in take)
    jobs: std.ArrayList(*Job) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,

    pub fn init(gpa: Allocator, io: std.Io, tower: Tower, cap: usize) Encoder {
        return .{ .gpa = gpa, .io = io, .tower = tower, .cap = cap };
    }

    pub fn start(e: *Encoder) !void {
        e.thread = try std.Thread.spawn(.{}, run, .{e});
    }

    /// Ends the thread once its current picture is done, then frees every job.
    pub fn stop(e: *Encoder) void {
        e.lock();
        e.closing = true;
        e.cond.broadcast(e.io);
        e.done.broadcast(e.io);
        e.unlock();
        if (e.thread) |t| t.join();
        e.thread = null;
        for (e.jobs.items) |j| e.free(j);
        e.jobs.deinit(e.gpa);
    }

    /// The lane host's view (lanes.stream.Encode).
    pub fn hook(e: *Encoder) lanes.stream.Encode {
        return .{ .ctx = e, .start = startFn, .ready = readyFn, .forget = forgetFn };
    }

    fn of(ctx: *anyopaque) *Encoder {
        return @ptrCast(@alignCast(ctx));
    }

    fn startFn(ctx: *anyopaque, images: []const Image, from: u32, wake: Wake) void {
        of(ctx).submit(images, from, wake) catch |err| std.log.err("encoder: a request's pictures not queued ({s})", .{@errorName(err)});
    }

    fn readyFn(ctx: *anyopaque, images: []const Image) bool {
        return of(ctx).ready(images);
    }

    fn forgetFn(ctx: *anyopaque, images: []const Image) void {
        of(ctx).forget(images);
    }

    fn lock(e: *Encoder) void {
        e.mutex.lockUncancelable(e.io);
    }

    fn unlock(e: *Encoder) void {
        e.mutex.unlock(e.io);
    }

    /// Queues a request's pictures past `from` (a resume covers the rest), payloads copied (the request may end first).
    pub fn submit(e: *Encoder, images: []const Image, from: u32, wake: Wake) !void {
        const j = try e.gpa.create(Job);
        j.* = .{ .key = @intFromPtr(images.ptr), .payloads = &.{}, .tokens = &.{}, .ranges = &.{}, .skipped = &.{}, .wake = wake };
        errdefer e.free(j);
        j.payloads = try e.gpa.alloc([]align(16) u8, images.len);
        for (j.payloads) |*p| p.* = &.{};
        j.tokens = try e.gpa.alloc(usize, images.len);
        @memset(j.tokens, 0);
        j.ranges = try e.gpa.alloc([2]usize, images.len);
        @memset(j.ranges, .{ 0, 0 });
        j.skipped = try e.gpa.alloc(bool, images.len);
        @memset(j.skipped, false);
        var rows: usize = 0;
        for (images) |img| rows += img.tokens;
        var any = false;
        if (rows > e.cap) {
            j.state = .done;
            j.err = error.TooManyImageRows;
        } else for (images, j.payloads, j.skipped) |img, *p, *skip| {
            skip.* = !img.rows and @as(usize, img.at) + img.tokens <= from;
            if (img.rows or skip.*) continue;
            p.* = try e.gpa.alignedAlloc(u8, .@"16", img.bytes.len);
            @memcpy(p.*, img.bytes);
            any = true;
        }
        if (!any and j.err == null) j.state = .done; // nothing to encode now
        e.lock();
        defer e.unlock();
        try e.jobs.append(e.gpa, j);
        e.cond.signal(e.io);
    }

    /// Whether a request's rows are ready, or its encode failed, or it never queued (its prompt pass then says why).
    pub fn ready(e: *Encoder, images: []const Image) bool {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return true;
        return j.state == .done;
    }

    /// Appends image `k`'s rows to `out` and returns its span (a skipped one encoded now, the caller waiting).
    pub fn take(e: *Encoder, images: []const Image, k: usize, gpa: Allocator, out: *std.ArrayList(u8)) !usize {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return error.NotEncoded;
        if (j.state != .done) return error.NotEncoded;
        if (j.err) |err| return err;
        if (j.skipped[k]) try e.encodeNow(j, k, images[k].bytes);
        try out.appendSlice(gpa, j.rows.items[j.ranges[k][0]..j.ranges[k][1]]);
        return j.tokens[k];
    }

    /// Picture `k` of a done job queued first and waited for (the lock held).
    fn encodeNow(e: *Encoder, j: *Job, k: usize, bytes: []const u8) !void {
        const p = try e.gpa.alignedAlloc(u8, .@"16", bytes.len);
        @memcpy(p, bytes);
        j.payloads[k] = p;
        j.skipped[k] = false;
        j.state = .queued;
        j.urgent = true;
        e.cond.signal(e.io);
        while (j.state != .done) {
            if (e.closing) return error.Closed;
            e.done.waitUncancelable(e.io, &e.mutex);
        }
        if (j.err) |err| return err;
    }

    /// Drops a request's job; one being encoded is freed by the thread when its picture is done.
    pub fn forget(e: *Encoder, images: []const Image) void {
        e.lock();
        defer e.unlock();
        const j = e.find(images) orelse return;
        if (j.state == .running) {
            j.gone = true;
            return;
        }
        e.remove(j);
        e.free(j);
    }

    fn find(e: *Encoder, images: []const Image) ?*Job {
        if (images.len == 0) return null;
        for (e.jobs.items) |j| if (j.key == @intFromPtr(images.ptr) and !j.gone) return j;
        return null;
    }

    fn remove(e: *Encoder, j: *Job) void {
        const i = std.mem.indexOfScalar(*Job, e.jobs.items, j) orelse return;
        _ = e.jobs.orderedRemove(i);
    }

    fn freePayloads(e: *Encoder, j: *Job) void {
        for (j.payloads) |*p| {
            e.gpa.free(p.*);
            p.* = &.{};
        }
    }

    /// Whether the thread should leave job `j` before its next picture: forgotten, or the encoder stopping.
    fn stopped(e: *Encoder, j: *const Job) bool {
        e.lock();
        defer e.unlock();
        return j.gone or e.closing;
    }

    fn free(e: *Encoder, j: *Job) void {
        e.freePayloads(j);
        e.gpa.free(j.payloads);
        e.gpa.free(j.tokens);
        e.gpa.free(j.ranges);
        e.gpa.free(j.skipped);
        j.rows.deinit(e.gpa);
        e.gpa.destroy(j);
    }

    fn run(e: *Encoder) void {
        const bound = e.tower.bind(e.tower.ctx);
        e.lock();
        defer e.unlock();
        while (!e.closing) {
            const j = e.next() orelse {
                e.cond.waitUncancelable(e.io, &e.mutex);
                continue;
            };
            j.state = .running;
            e.unlock();
            var err: ?anyerror = null;
            if (bound) |_| {
                e.encode(j) catch |x| {
                    err = x;
                };
            } else |x| err = x;
            e.lock();
            j.state = .done;
            j.urgent = false;
            j.err = err;
            e.freePayloads(j);
            e.done.broadcast(e.io);
            if (j.gone) {
                e.remove(j);
                e.free(j);
                continue;
            }
            const wake = j.wake;
            e.unlock();
            wake.call(wake.ctx);
            e.lock();
        }
    }

    /// The queued job a prompt pass waits for, else the first queued (the lock held).
    fn next(e: *Encoder) ?*Job {
        var first: ?*Job = null;
        for (e.jobs.items) |x| if (x.state == .queued) {
            if (x.urgent) return x;
            if (first == null) first = x;
        };
        return first;
    }

    fn encode(e: *Encoder, j: *Job) !void {
        for (j.payloads, j.tokens, j.ranges) |p, *t, *r| if (p.len > 0) {
            if (e.stopped(j)) return error.EncodeCancelled;
            const begin = j.rows.items.len;
            t.* = try e.tower.rows(e.tower.ctx, p, e.gpa, &j.rows);
            r.* = .{ begin, j.rows.items.len };
        };
    }
};

/// A tower whose pictures are [n, v]: n rows of four bytes v; it holds each encode while `hold` is set.
const Fake = struct {
    hold: std.atomic.Value(bool) = .init(false),
    entered: std.atomic.Value(u32) = .init(0), // pictures whose encode has begun
    encodes: std.atomic.Value(u32) = .init(0),
    fail: bool = false,

    fn tower(f: *Fake) Tower {
        return .{ .ctx = f, .bind = bind, .rows = rows };
    }

    fn bind(_: *anyopaque) anyerror!void {}

    fn rows(ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8)) anyerror!usize {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        _ = f.entered.fetchAdd(1, .acq_rel);
        while (f.hold.load(.acquire)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        _ = f.encodes.fetchAdd(1, .acq_rel);
        if (f.fail) return error.DeviceFault;
        try out.appendNTimes(gpa, payload[1], 4 * @as(usize, payload[0]));
        return payload[0];
    }
};

const Woken = struct {
    n: std.atomic.Value(u32) = .init(0),

    fn call(ctx: *anyopaque) void {
        const w: *Woken = @ptrCast(@alignCast(ctx));
        _ = w.n.fetchAdd(1, .acq_rel);
    }

    fn wake(w: *Woken) Wake {
        return .{ .ctx = w, .call = call };
    }
};

fn waitReady(e: *Encoder, images: []const Image) void {
    while (!e.ready(images)) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
}

test "an encoder hands each request its pictures' rows in order, the rows that came ready skipped" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 1, .bytes = "rows", .rows = true }, .{ .at = 5, .tokens = 3, .bytes = &.{ 3, 9 } } };
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(usize, 2), try e.take(&a, 0, gpa, &out));
    try std.testing.expectEqual(@as(usize, 20), out.items.len);
    try std.testing.expect(std.mem.allEqual(u8, out.items[0..12], 9) and std.mem.allEqual(u8, out.items[12..], 7));
    // the wake comes after ready
    while (woken.n.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    try std.testing.expectEqual(@as(u32, 1), woken.n.load(.acquire));
    e.forget(&a);
    try std.testing.expect(e.ready(&a)); // no job: its prompt pass says why
    try std.testing.expectError(error.NotEncoded, e.take(&a, 0, gpa, &out));
}

test "a request forgotten while its picture encodes is freed by the thread, unwoken; a queued one at once" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }};
    const b = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 2 } }};
    try e.submit(&a, 0, woken.wake());
    try e.submit(&b, 0, woken.wake());
    while (true) { // the thread has taken a
        e.lock();
        const running = e.jobs.items[0].state == .running;
        e.unlock();
        if (running) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expect(!e.ready(&a) and !e.ready(&b));
    e.forget(&a);
    e.forget(&b);
    fake.hold.store(false, .release);
    while (true) {
        e.lock();
        const left = e.jobs.items.len;
        e.unlock();
        if (left == 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "a failed encode is the request's error at its prompt pass" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{ .fail = true };
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }};
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectError(error.DeviceFault, e.take(&a, 0, gpa, &out));
    e.forget(&a);
}

test "a request over the image row cap fails at once, its pictures never encoded" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 4);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 3, .bytes = &.{ 3, 1 } }, .{ .at = 4, .tokens = 2, .bytes = &.{ 2, 1 } } };
    try e.submit(&a, 0, woken.wake());
    try std.testing.expect(e.ready(&a));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectError(error.TooManyImageRows, e.take(&a, 0, gpa, &out));
    e.forget(&a);
    try std.testing.expectEqual(@as(u32, 0), fake.encodes.load(.acquire));
}

fn waitEntered(f: *Fake) void {
    while (f.entered.load(.acquire) == 0) std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
}

const four = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2 } }, .{ .at = 4, .tokens = 1, .bytes = &.{ 1, 3 } }, .{ .at = 6, .tokens = 1, .bytes = &.{ 1, 4 } } };

test "a request of four pictures forgotten during its first stops before the second" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    try e.submit(&four, 0, woken.wake());
    waitEntered(&fake);
    e.forget(&four);
    fake.hold.store(false, .release);
    while (true) {
        e.lock();
        const left = e.jobs.items.len;
        e.unlock();
        if (left == 0) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "stopping the encoder during the first of four pictures waits for that picture only" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 8);
    try e.start();
    var joined = false;
    errdefer if (!joined) {
        fake.hold.store(false, .release);
        e.stop();
    };
    var woken: Woken = .{};
    try e.submit(&four, 0, woken.wake());
    waitEntered(&fake);
    const stopper = try std.Thread.spawn(.{}, Encoder.stop, .{&e});
    while (true) {
        e.lock();
        const closing = e.closing;
        e.unlock();
        if (closing) break;
        std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
    }
    fake.hold.store(false, .release);
    stopper.join();
    joined = true;
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
}

test "pictures a resume covers are not encoded at submit; one its pass still asks for is encoded then" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, std.testing.io, fake.tower(), 16);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 3, .bytes = &.{ 3, 8 } }, .{ .at = 8, .tokens = 1, .bytes = &.{ 1, 9 } } };
    try e.submit(&a, 6, woken.wake()); // a resume at 6 covers the first two
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, gpa, &out)); // the resume fell short: encoded now
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 4 + 12), out.items.len);
    try std.testing.expect(std.mem.allEqual(u8, out.items[0..4], 9) and std.mem.allEqual(u8, out.items[4..], 8));
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, gpa, &out)); // again: its rows kept, no second encode
    try std.testing.expectEqual(@as(usize, 12), out.items.len);
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    const b = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 5 } }};
    try e.submit(&b, 2, woken.wake()); // every picture covered: ready at once, nothing encoded
    try std.testing.expect(e.ready(&b));
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    e.forget(&a);
    e.forget(&b);
}

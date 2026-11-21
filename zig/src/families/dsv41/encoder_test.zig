//! encoder.zig's tests: the job queue, the rows cache, pictures prepared from their source.
const std = @import("std");
const lanes = @import("lanes");
const enc = @import("encoder.zig");
const Encoder = enc.Encoder;
const Tower = enc.Tower;
const Allocator = std.mem.Allocator;
const Image = lanes.stream.Image;
const Wake = lanes.stream.Wake;
const io = std.testing.io;

pub fn sleepMs(ms: i64) void {
    std.Io.sleep(io, .fromMilliseconds(ms), .awake) catch {};
}

/// A tower of pictures [n, v, blocks]: n rows of bytes v after paced blocks; `hold` holds, `fail_v` fails picture v.
pub const Fake = struct {
    hold: std.atomic.Value(bool) = .init(false),
    hold_v: std.atomic.Value(u8) = .init(0), // with `hold`: only picture v waits (0: every one)
    entered: std.atomic.Value(u32) = .init(0), // pictures whose encode has begun
    encodes: std.atomic.Value(u32) = .init(0),
    blocks: std.atomic.Value(u32) = .init(0), // blocks run
    in_block: std.atomic.Value(bool) = .init(false),
    block_ms: i64 = 2,
    fail: bool = false,
    fail_v: std.atomic.Value(u8) = .init(0),
    fail_block: u8 = 0,
    prepares: std.atomic.Value(u32) = .init(0),
    prepare_thread: std.atomic.Value(std.Thread.Id) = .init(0),
    mutex: std.Io.Mutex = .init,
    events: [64][2]u8 = undefined, // (v, 'b' or 'e')
    n_events: usize = 0,

    pub fn tower(f: *Fake) Tower {
        return .{ .ctx = f, .bind = bind, .rows = rows, .row_bytes = 4, .prepare = prepare };
    }

    fn bind(_: *anyopaque) anyerror!void {}

    fn note(f: *Fake, v: u8, what: u8) void {
        f.mutex.lockUncancelable(io);
        defer f.mutex.unlock(io);
        f.events[f.n_events] = .{ v, what };
        f.n_events += 1;
    }

    pub fn log(f: *Fake) []const [2]u8 {
        f.mutex.lockUncancelable(io);
        defer f.mutex.unlock(io);
        return f.events[0..f.n_events];
    }

    fn rows(ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8), pace: enc.Pace) anyerror!usize {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        const v = payload[1];
        _ = f.entered.fetchAdd(1, .acq_rel);
        f.note(v, 'b');
        while (f.hold.load(.acquire) and (f.hold_v.load(.acquire) == 0 or f.hold_v.load(.acquire) == v)) sleepMs(1);
        const n_blocks: u8 = if (payload.len > 2) payload[2] else 0;
        for (0..n_blocks) |b| {
            f.in_block.store(true, .release);
            sleepMs(f.block_ms);
            _ = f.blocks.fetchAdd(1, .acq_rel);
            f.in_block.store(false, .release);
            if (f.fail_v.load(.acquire) == v and b == f.fail_block) return error.DeviceFault;
            if (pace.pausing()) pace.yield();
        }
        _ = f.encodes.fetchAdd(1, .acq_rel);
        if (f.fail) return error.DeviceFault;
        try out.appendNTimes(gpa, v, 4 * @as(usize, payload[0]));
        f.note(v, 'e');
        return payload[0];
    }

    /// A source [n, v, blocks] prepares to a payload of `lazy_bytes`: those three bytes, then zeros.
    fn prepare(ctx: *anyopaque, gpa: Allocator, source: []const u8) anyerror![]align(16) u8 {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        _ = f.prepares.fetchAdd(1, .acq_rel);
        f.prepare_thread.store(std.Thread.getCurrentId(), .release);
        const p = try gpa.alignedAlloc(u8, .@"16", lazy_bytes);
        @memset(p, 0);
        @memcpy(p[0..source.len], source);
        return p;
    }
};

const lazy_bytes = 4096;

fn lazyDigest(source: []const u8) lanes.image_key.Digest {
    var p: [lazy_bytes]u8 = @splat(0);
    @memcpy(p[0..source.len], source);
    return lanes.image_key.imageDigest(&p);
}

pub const Woken = struct {
    n: std.atomic.Value(u32) = .init(0),

    fn call(ctx: *anyopaque) void {
        const w: *Woken = @ptrCast(@alignCast(ctx));
        _ = w.n.fetchAdd(1, .acq_rel);
    }

    pub fn wake(w: *Woken) Wake {
        return .{ .ctx = w, .call = call };
    }
};

/// The capacity of a request's job rows (the exact reserve's tests).
pub fn jobRowsCapacity(e: *Encoder, images: []const Image) usize {
    e.lock();
    defer e.unlock();
    return e.find(images).?.rows.capacity;
}

pub fn jobsLeft(e: *Encoder) usize {
    e.lock();
    defer e.unlock();
    return e.jobs.items.len;
}

pub fn waitReady(e: *Encoder, images: []const Image) void {
    while (!e.ready(images)) sleepMs(1);
}

pub fn waitEntered(f: *Fake) void {
    while (f.entered.load(.acquire) == 0) sleepMs(1);
}

test "an encoder hands each request its pictures' rows in order, the rows that came ready skipped" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 1, .bytes = "rows", .rows = true }, .{ .at = 5, .tokens = 3, .bytes = &.{ 3, 9 } } };
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(usize, 20), jobRowsCapacity(&e, &a)); // the encoded pictures' 5 rows exactly
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(usize, 2), try e.take(&a, 0, gpa, &out));
    try std.testing.expectEqual(@as(usize, 20), out.items.len);
    try std.testing.expect(std.mem.allEqual(u8, out.items[0..12], 9) and std.mem.allEqual(u8, out.items[12..], 7));
    // the wake comes after ready
    while (woken.n.load(.acquire) == 0) sleepMs(1);
    try std.testing.expectEqual(@as(u32, 1), woken.n.load(.acquire));
    e.forget(&a);
    try std.testing.expect(e.ready(&a)); // no job: its prompt pass says why
    try std.testing.expectError(error.NotEncoded, e.take(&a, 0, gpa, &out));
}

test "a request forgotten while its picture encodes is freed by the thread, unwoken; a queued one at once" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, io, fake.tower(), 8);
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
        sleepMs(1);
    }
    try std.testing.expect(!e.ready(&a) and !e.ready(&b));
    e.forget(&a);
    e.forget(&b);
    fake.hold.store(false, .release);
    while (jobsLeft(&e) > 0) sleepMs(1);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "a failed encode is the request's error at its prompt pass" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{ .fail = true };
    var e = Encoder.init(gpa, io, fake.tower(), 8);
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
    var e = Encoder.init(gpa, io, fake.tower(), 4);
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

const four = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2 } }, .{ .at = 4, .tokens = 1, .bytes = &.{ 1, 3 } }, .{ .at = 6, .tokens = 1, .bytes = &.{ 1, 4 } } };

test "a request of four pictures forgotten during its first stops before the second" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, io, fake.tower(), 8);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    try e.submit(&four, 0, woken.wake());
    waitEntered(&fake);
    e.forget(&four);
    fake.hold.store(false, .release);
    while (jobsLeft(&e) > 0) sleepMs(1);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
}

test "stopping the encoder during the first of four pictures waits for that picture only" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    fake.hold.store(true, .release);
    var e = Encoder.init(gpa, io, fake.tower(), 8);
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
        sleepMs(1);
    }
    fake.hold.store(false, .release);
    stopper.join();
    joined = true;
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
}

test "pictures a resume covers are not encoded at submit; one its pass still asks for is encoded then" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 16);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 } }, .{ .at = 3, .tokens = 3, .bytes = &.{ 3, 8 } }, .{ .at = 8, .tokens = 1, .bytes = &.{ 1, 9 } } };
    try e.submit(&a, 6, woken.wake()); // a resume at 6 covers the first two
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 4), jobRowsCapacity(&e, &a)); // the one encoded picture's row exactly
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), try e.take(&a, 2, gpa, &out));
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, gpa, &out)); // the resume fell short: encoded now
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expectEqual(@as(usize, 16), jobRowsCapacity(&e, &a)); // and the late one's 3 rows, exactly
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

test "a resume inside a picture's span still encodes it at submit; a resume that falls back into a skipped one encodes it then" {
    const gpa = std.testing.allocator;
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 1024);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    // a picture the resume covers, and one at [4000, 4200) the resume at 4096 lands inside
    const a = [_]Image{ .{ .at = 100, .tokens = 200, .bytes = &.{ 200, 7 } }, .{ .at = 4000, .tokens = 200, .bytes = &.{ 200, 8 } } };
    try e.submit(&a, 4096, woken.wake());
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire)); // the second only, though it began before 4096
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 200), try e.take(&a, 1, gpa, &out)); // its whole span's rows, already there
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    try std.testing.expect(out.items.len == 4 * 200 and std.mem.allEqual(u8, out.items, 8));
    e.forget(&a);
    // the hint said 6144 (past the picture) and the pass resumed at 4096 after all: the skipped picture on request
    const b = [_]Image{.{ .at = 4000, .tokens = 200, .bytes = &.{ 200, 9 } }};
    try e.submit(&b, 6144, woken.wake());
    try std.testing.expect(e.ready(&b));
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire));
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 200), try e.take(&b, 0, gpa, &out));
    try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
    try std.testing.expect(out.items.len == 4 * 200 and std.mem.allEqual(u8, out.items, 9));
    e.forget(&b);
}

pub fn digestOf(b: u8) lanes.image_key.Digest {
    return @splat(b);
}

test "rows cache: a second request with the same picture encodes nothing and gets equal rows; off, it encodes again" {
    const gpa = std.testing.allocator;
    for ([_]usize{ 64, 0 }) |cap| {
        var fake: Fake = .{};
        var e = Encoder.init(gpa, io, fake.tower(), 64);
        e.options.rows_cache_bytes = cap;
        try e.start();
        defer e.stop();
        var woken: Woken = .{};
        const a = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 }, .digest = digestOf(7) }, .{ .at = 4, .tokens = 3, .bytes = &.{ 3, 8 }, .digest = digestOf(8) } };
        try e.submit(&a, 0, woken.wake());
        waitReady(&e, &a);
        var first: std.ArrayList(u8) = .empty;
        defer first.deinit(gpa);
        for (0..2) |k| _ = try e.take(&a, k, gpa, &first);
        e.forget(&a);
        try std.testing.expectEqual(@as(u32, 2), fake.encodes.load(.acquire));
        // the next turn carries the same two pictures (a new request: its own images) and a new one
        const b = [_]Image{ .{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 }, .digest = digestOf(7) }, .{ .at = 4, .tokens = 3, .bytes = &.{ 3, 8 }, .digest = digestOf(8) }, .{ .at = 9, .tokens = 1, .bytes = &.{ 1, 6 }, .digest = digestOf(6) } };
        try e.submit(&b, 0, woken.wake());
        waitReady(&e, &b);
        try std.testing.expectEqual(@as(u32, if (cap > 0) 3 else 5), fake.encodes.load(.acquire));
        try std.testing.expectEqual(@as(usize, 24), jobRowsCapacity(&e, &b)); // cached and encoded rows, exactly
        var second: std.ArrayList(u8) = .empty;
        defer second.deinit(gpa);
        for (0..2) |k| _ = try e.take(&b, k, gpa, &second);
        try std.testing.expectEqualSlices(u8, first.items, second.items);
        e.forget(&b);
        // every picture cached: ready at submit, nothing encoded
        const c = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 7 }, .digest = digestOf(7) }};
        try e.submit(&c, 0, woken.wake());
        if (cap > 0) try std.testing.expect(e.ready(&c));
        waitReady(&e, &c);
        try std.testing.expectEqual(@as(u32, if (cap > 0) 3 else 6), fake.encodes.load(.acquire));
        // a picture a resume skipped and its pass asks for after all: copied, nothing waited for
        const d = [_]Image{.{ .at = 0, .tokens = 3, .bytes = &.{ 3, 8 }, .digest = digestOf(8) }};
        try e.submit(&d, 3, woken.wake());
        try std.testing.expect(e.ready(&d));
        second.clearRetainingCapacity();
        try std.testing.expectEqual(@as(usize, 3), try e.take(&d, 0, gpa, &second));
        try std.testing.expect(std.mem.allEqual(u8, second.items, 8) and second.items.len == 12);
        try std.testing.expectEqual(@as(u32, if (cap > 0) 3 else 7), fake.encodes.load(.acquire));
        e.forget(&c);
        e.forget(&d);
    }
}

/// An allocator that counts the live blocks of `size` bytes (a lazy picture's payload) and their most at once.
const Counting = struct {
    child: Allocator,
    size: usize,
    live: usize = 0,
    most: usize = 0,
    mutex: std.Io.Mutex = .init,

    fn allocator(c: *Counting) Allocator {
        return .{ .ptr = c, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }

    fn count(c: *Counting, len: usize, d: isize) void {
        if (len != c.size) return;
        c.mutex.lockUncancelable(io);
        defer c.mutex.unlock(io);
        c.live = @intCast(@as(isize, @intCast(c.live)) + d);
        c.most = @max(c.most, c.live);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawAlloc(len, alignment, ret) orelse return null;
        c.count(len, 1);
        return p;
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret: usize) bool {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        if (!c.child.rawResize(memory, alignment, new_len, ret)) return false;
        c.count(memory.len, -1);
        c.count(new_len, 1);
        return true;
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, new_len: usize, ret: usize) ?[*]u8 {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        const p = c.child.rawRemap(memory, alignment, new_len, ret) orelse return null;
        c.count(memory.len, -1);
        c.count(new_len, 1);
        return p;
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ret: usize) void {
        const c: *Counting = @ptrCast(@alignCast(ctx));
        c.count(memory.len, -1);
        c.child.rawFree(memory, alignment, ret);
    }
};

test "a picture sent by its source is prepared on the thread, one payload alive at a time; a changed one fails" {
    var counting: Counting = .{ .child = std.testing.allocator, .size = lazy_bytes };
    const gpa = counting.allocator();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    const s1 = [_]u8{ 2, 7, 2 };
    const s2 = [_]u8{ 3, 8, 2 };
    const s3 = [_]u8{ 1, 9, 2 };
    const a = [_]Image{
        .{ .at = 0, .tokens = 2, .bytes = &.{}, .source = &s1, .digest = lazyDigest(&s1) },
        .{ .at = 4, .tokens = 3, .bytes = &.{}, .source = &s2, .digest = lazyDigest(&s2) },
        .{ .at = 9, .tokens = 1, .bytes = &.{}, .source = &s3, .digest = lazyDigest(&s3) },
    };
    try e.submit(&a, 0, woken.wake());
    waitReady(&e, &a);
    try std.testing.expectEqual(@as(u32, 3), fake.prepares.load(.acquire));
    try std.testing.expect(fake.prepare_thread.load(.acquire) != std.Thread.getCurrentId()); // on the encoder's thread
    try std.testing.expectEqual(@as(usize, 1), counting.most); // one payload at a time
    try std.testing.expectEqual(@as(usize, 0), counting.live); // and none kept after its picture
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 3), try e.take(&a, 1, std.testing.allocator, &out));
    try std.testing.expect(out.items.len == 12 and std.mem.allEqual(u8, out.items, 8));
    e.forget(&a);
    // the source's payload is not the one the server measured: the request fails, its picture never encoded
    const b = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{}, .source = &s1, .digest = lazyDigest(&s2) }};
    try e.submit(&b, 0, woken.wake());
    waitReady(&e, &b);
    try std.testing.expectError(error.ImageChanged, e.take(&b, 0, std.testing.allocator, &out));
    try std.testing.expectEqual(@as(u32, 3), fake.encodes.load(.acquire));
    e.forget(&b);
    try std.testing.expectEqual(@as(usize, 0), counting.live);
}

test {
    _ = @import("encoder_pace_test.zig");
}

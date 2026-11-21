//! encoder.zig's gate tests: held, sliced, released; urgent encodes beside a parked job; the budget.
const std = @import("std");
const lanes = @import("lanes");
const enc = @import("encoder.zig");
const et = @import("encoder_test.zig");
const Encoder = enc.Encoder;
const Image = lanes.stream.Image;
const Watchdog = @import("encoder_gate.zig").Watchdog;
const Fake = et.Fake;
const Woken = et.Woken;
const sleepMs = et.sleepMs;
const jobsLeft = et.jobsLeft;
const jobRowsCapacity = et.jobRowsCapacity;
const waitReady = et.waitReady;
const digestOf = et.digestOf;
const io = std.testing.io;

/// The budget's charges are the jobs' row capacities, and never more than the budget plus the stated exceptions.
fn chargesMatch(e: *Encoder) !void {
    e.lock();
    defer e.unlock();
    var sum: usize = 0;
    for (e.jobs.items) |j| sum += j.rows.capacity;
    try std.testing.expectEqual(sum, e.charged);
}

test "a held tower runs no block outside a slice and one at least in each; released, it runs free" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    try e.start();
    defer e.stop();
    const h = e.hook();
    h.hold.?(h.ctx);
    var woken: Woken = .{};
    const a = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1, 3 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2, 3 } } };
    try e.submit(&a, 0, woken.wake());
    waitParkedAtBegin(&e);
    try blocksStay(&fake, 20); // parked before its first picture
    try std.testing.expectEqual(@as(u32, 0), fake.blocks.load(.acquire));
    // a 1 ns slice passes the gate once: a block, or the last block's tail (the picture's rows) at a picture's end
    for ([_]u32{ 1, 2, 3, 3, 4, 5 }) |n| {
        h.slice.?(h.ctx, 1);
        try std.testing.expect(!fake.in_block.load(.acquire)); // returns with the stream idle
        try std.testing.expectEqual(n, fake.blocks.load(.acquire));
        try blocksStay(&fake, 10);
    }
    try std.testing.expectEqual(@as(u32, 1), fake.encodes.load(.acquire)); // the fourth slice ended picture 1
    while (!e.ready(&a)) h.slice.?(h.ctx, 1_000_000_000); // long slices run the rest
    try std.testing.expectEqual(@as(u32, 6), fake.blocks.load(.acquire));
    h.slice.?(h.ctx, 1_000_000_000); // nothing in flight: at once
    h.release.?(h.ctx);
    const b = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 3, 4 } }};
    try e.submit(&b, 0, woken.wake());
    waitReady(&e, &b); // free: no slice needed
    try std.testing.expectEqual(@as(u32, 10), fake.blocks.load(.acquire));
    e.forget(&a);
    e.forget(&b);
}

test "a picture that fails while hold or slice waits for it wakes the waiter" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{ .block_ms = 30 };
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    try e.start();
    defer e.stop();
    const h = e.hook();
    var woken: Woken = .{};
    // hold waits for the block in flight, which fails
    fake.fail_v.store(1, .release);
    const a = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1, 3 } }};
    try e.submit(&a, 0, woken.wake());
    while (!fake.in_block.load(.acquire)) sleepMs(1);
    h.hold.?(h.ctx); // returns once that block failed and its picture ended
    try std.testing.expect(!fake.in_block.load(.acquire));
    waitReady(&e, &a);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectError(error.DeviceFault, e.take(&a, 0, gpa, &out));
    // a slice waits for its block, which fails
    fake.fail_v.store(2, .release);
    const b = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 2, 3 } }};
    try e.submit(&b, 0, woken.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1_000_000_000); // returns once its first block failed, long before the budget
    try std.testing.expect(!fake.in_block.load(.acquire));
    waitReady(&e, &b);
    try std.testing.expectError(error.DeviceFault, e.take(&b, 0, gpa, &out));
    e.forget(&a);
    e.forget(&b);
}

const Taker = struct {
    e: *Encoder,
    images: []const Image,
    out: std.ArrayList(u8) = .empty,
    tokens: ?usize = null,
    err: ?anyerror = null,
    returned: std.atomic.Value(bool) = .init(false),

    fn run(t: *Taker) void {
        if (t.e.take(t.images, 0, std.testing.allocator, &t.out)) |n| t.tokens = n else |x| t.err = x;
        t.returned.store(true, .release);
    }
};

fn blocksStay(f: *Fake, ms: i64) !void {
    const n = f.blocks.load(.acquire);
    sleepMs(ms);
    try std.testing.expectEqual(n, f.blocks.load(.acquire));
}

/// Waits until the encoder thread waits at a picture's begin (the gate held, no picture in flight).
fn waitParkedAtBegin(e: *Encoder) void {
    while (true) {
        e.gate.mutex.lockUncancelable(io);
        const at = e.gate.parked and !e.gate.in_picture;
        e.gate.mutex.unlock(io);
        if (at) return;
        sleepMs(1);
    }
}

const Scene = enum { plain, delayed_waiter, urgent_fails };

/// The round-stall case: A's picture skipped on a hint, B parked mid-picture at the held gate, A's pass asks, no cache.
fn parkedScene(scene: Scene) !void {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    e.options.rows_cache_bytes = 1 << 20; // on, and empty
    try e.start();
    defer e.stop();
    const h = e.hook();
    var wa: Woken = .{};
    var wb: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 5, 2 }, .digest = digestOf(5) }};
    try e.submit(&a, 4, wa.wake()); // the hint covers it: skipped, done at once
    try std.testing.expect(e.ready(&a));
    h.hold.?(h.ctx);
    const b = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1, 3 } }, .{ .at = 2, .tokens = 3, .bytes = &.{ 3, 2, 3 } } };
    try e.submit(&b, 0, wb.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1); // B's first block: then parked mid-picture
    try std.testing.expectEqual(@as(u32, 1), fake.blocks.load(.acquire));
    if (scene == .urgent_fails) {
        fake.fail_v.store(5, .release);
        fake.fail_block = 1;
    }
    const Delay = struct {
        e: *Encoder,
        f: *Fake,
        ok: bool = false,
        // the waiter is woken: before it returns, the encoder thread has tried to resume B (parked at its begin)
        fn call(ctx: *anyopaque) void {
            const d: *@This() = @ptrCast(@alignCast(ctx));
            waitParkedAtBegin(d.e);
            const n = d.f.blocks.load(.acquire);
            sleepMs(20);
            d.ok = n == d.f.blocks.load(.acquire);
        }
    };
    var delay: Delay = .{ .e = &e, .f = &fake };
    if (scene != .plain) e.woken_hook = .{ .ctx = &delay, .call = Delay.call };
    var taker: Taker = .{ .e = &e, .images = &a };
    defer taker.out.deinit(gpa);
    const t = try std.Thread.spawn(.{}, Taker.run, .{&taker});
    t.join(); // no slice meanwhile: the rush lets B drain its picture, then A runs
    if (scene == .urgent_fails) {
        try std.testing.expectEqual(@as(?anyerror, error.DeviceFault), taker.err);
    } else {
        try std.testing.expectEqual(@as(?usize, 2), taker.tokens);
        try std.testing.expect(taker.out.items.len == 8 and std.mem.allEqual(u8, taker.out.items, 5));
    }
    if (scene != .plain) try std.testing.expect(delay.ok);
    // B's first picture ended before A's began, and B's second had not begun
    const log = fake.log();
    try std.testing.expect(log.len >= 3);
    try std.testing.expectEqualSlices(u8, &.{ 1, 'b' }, &log[0]);
    try std.testing.expectEqualSlices(u8, &.{ 1, 'e' }, &log[1]);
    try std.testing.expectEqualSlices(u8, &.{ 5, 'b' }, &log[2]);
    for (log) |x| try std.testing.expect(x[0] != 2);
    // the gate binds B again: hold returns at once with the stream idle, and no block runs until a slice
    h.hold.?(h.ctx);
    try std.testing.expect(!fake.in_block.load(.acquire));
    try blocksStay(&fake, 30);
    const before = fake.blocks.load(.acquire);
    h.slice.?(h.ctx, 1);
    try std.testing.expectEqual(before + 1, fake.blocks.load(.acquire));
    while (!e.ready(&b)) h.slice.?(h.ctx, 1_000_000);
    // B encoded each picture once, with the rows of an undisturbed run, and was woken once
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    _ = try e.take(&b, 0, gpa, &out);
    _ = try e.take(&b, 1, gpa, &out);
    try std.testing.expect(out.items.len == 16 and std.mem.allEqual(u8, out.items[0..4], 1) and std.mem.allEqual(u8, out.items[4..], 2));
    var b_pictures: u32 = 0;
    for (fake.log()) |x| if (x[1] == 'b' and x[0] != 5) {
        b_pictures += 1;
    };
    try std.testing.expectEqual(@as(u32, 2), b_pictures);
    while (wb.n.load(.acquire) == 0) sleepMs(1);
    try std.testing.expectEqual(@as(u32, 1), wb.n.load(.acquire));
    h.release.?(h.ctx);
    e.forget(&a);
    e.forget(&b);
}

test "a prompt pass that asks for a skipped picture while another request is parked at the held gate gets it without a slice" {
    try parkedScene(.plain);
}

test "an urgent waiter woken late finds the gate held: the job it bypassed runs no block until the next slice" {
    try parkedScene(.delayed_waiter);
}

test "an urgent encode that fails hands the gate back the same way" {
    try parkedScene(.urgent_fails);
}

test "stop while an urgent waiter waits beside a parked job: the waiter gets Closed, the thread ends after its picture" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    try e.start();
    var stopped = false;
    defer if (!stopped) e.stop();
    const h = e.hook();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 5, 2 } }};
    try e.submit(&a, 4, woken.wake());
    h.hold.?(h.ctx);
    const b = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1, 3 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2, 3 } } };
    try e.submit(&b, 0, woken.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1);
    fake.hold_v.store(5, .release); // A's picture waits once it begins
    fake.hold.store(true, .release);
    var taker: Taker = .{ .e = &e, .images = &a };
    defer taker.out.deinit(gpa);
    const t = try std.Thread.spawn(.{}, Taker.run, .{&taker});
    while (fake.entered.load(.acquire) < 2) sleepMs(1); // B drained, A's picture in flight
    const stopper = try std.Thread.spawn(.{}, Encoder.stop, .{&e});
    t.join();
    try std.testing.expectEqual(@as(?anyerror, error.Closed), taker.err);
    fake.hold.store(false, .release);
    stopper.join();
    stopped = true;
}

test "forget and stop while a job is parked at the held gate return at once; the job is freed after its picture" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    try e.start();
    var stopped = false;
    defer if (!stopped) e.stop();
    const h = e.hook();
    var woken: Woken = .{};
    h.hold.?(h.ctx);
    const b = [_]Image{ .{ .at = 0, .tokens = 1, .bytes = &.{ 1, 1, 3 } }, .{ .at = 2, .tokens = 1, .bytes = &.{ 1, 2, 3 } } };
    try e.submit(&b, 0, woken.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    e.forget(&b); // a cancel ends in the same forget
    try std.testing.expect(std.Io.Clock.awake.now(io).toNanoseconds() - t0 < 50 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(usize, 1), jobsLeft(&e)); // its picture is in flight: the thread frees it
    while (jobsLeft(&e) > 0) h.slice.?(h.ctx, 1);
    try std.testing.expectEqual(@as(u32, 3), fake.blocks.load(.acquire)); // its first picture only
    try std.testing.expectEqual(@as(u32, 0), woken.n.load(.acquire));
    // stop while parked: the gate opens for good, the picture ends, the thread joins
    const c = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 3, 3 } }};
    try e.submit(&c, 0, woken.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1);
    e.stop();
    stopped = true;
    try std.testing.expectEqual(@as(u32, 6), fake.blocks.load(.acquire));
}

test "budget: queued jobs wait unreserved until rows are freed; a lone large job runs; an urgent encode passes; charges track the rows" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    e.options.rows_budget_bytes = 40; // 10 rows of 4 bytes
    try e.start();
    defer e.stop();
    var woken: Woken = .{};
    // more image requests than lanes, none admitted (nothing takes or forgets): two fit, the third waits
    const j1 = [_]Image{.{ .at = 0, .tokens = 4, .bytes = &.{ 4, 1 } }};
    const j2 = [_]Image{.{ .at = 0, .tokens = 4, .bytes = &.{ 4, 2 } }};
    const j3 = [_]Image{.{ .at = 0, .tokens = 4, .bytes = &.{ 4, 3 } }};
    const j4 = [_]Image{.{ .at = 0, .tokens = 1, .bytes = &.{ 1, 4 } }};
    try e.submit(&j1, 0, woken.wake());
    try e.submit(&j2, 0, woken.wake());
    try e.submit(&j3, 0, woken.wake());
    try e.submit(&j4, 0, woken.wake()); // small enough, but behind j3: in order
    waitReady(&e, &j2);
    sleepMs(20);
    try std.testing.expect(!e.ready(&j3) and !e.ready(&j4));
    try chargesMatch(&e);
    try std.testing.expectEqual(@as(usize, 32), e.charged);
    try std.testing.expectEqual(@as(usize, 0), jobRowsCapacity(&e, &j3)); // unreserved
    // a take copies and keeps the job's rows: still charged, j3 still waits
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    _ = try e.take(&j1, 0, gpa, &out);
    sleepMs(20);
    try std.testing.expect(!e.ready(&j3));
    try std.testing.expectEqual(@as(usize, 32), e.charged);
    // a cancel of the job waiting for the budget frees nothing it does not hold
    e.forget(&j3);
    try std.testing.expectEqual(@as(usize, 32), e.charged);
    try chargesMatch(&e);
    waitReady(&e, &j4); // j4 (16 + 4 more bytes fit: 36 <= 40) starts now that j3 is gone
    try std.testing.expectEqual(@as(usize, 36), e.charged);
    // the forget after the prompt pass frees j1's rows: a large job queued now waits until it is alone
    e.forget(&j1);
    try std.testing.expectEqual(@as(usize, 20), e.charged);
    const big = [_]Image{.{ .at = 0, .tokens = 20, .bytes = &.{ 20, 9 } }};
    try e.submit(&big, 0, woken.wake());
    sleepMs(20);
    try std.testing.expect(!e.ready(&big));
    e.forget(&j2);
    e.forget(&j4);
    waitReady(&e, &big); // alone: it runs past the budget
    try std.testing.expectEqual(@as(usize, 80), e.charged);
    try chargesMatch(&e);
    // an urgent encode ignores the budget: a skipped picture of a done job, the budget full
    const u = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 8 } }};
    try e.submit(&u, 2, woken.wake());
    try std.testing.expect(e.ready(&u));
    out.clearRetainingCapacity();
    try std.testing.expectEqual(@as(usize, 2), try e.take(&u, 0, gpa, &out));
    try std.testing.expectEqual(@as(usize, 88), e.charged);
    try chargesMatch(&e);
    e.forget(&big);
    e.forget(&u);
    try std.testing.expectEqual(@as(usize, 0), e.charged);
}

test "budget: a job that gave way keeps its charge while queued; cancelling it then frees exactly that" {
    const gpa = std.testing.allocator;
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var fake: Fake = .{};
    var e = Encoder.init(gpa, io, fake.tower(), 64);
    e.options.rows_budget_bytes = 40;
    try e.start();
    defer e.stop();
    const h = e.hook();
    var woken: Woken = .{};
    const a = [_]Image{.{ .at = 0, .tokens = 2, .bytes = &.{ 2, 5, 2 } }};
    try e.submit(&a, 4, woken.wake());
    h.hold.?(h.ctx);
    const b = [_]Image{ .{ .at = 0, .tokens = 4, .bytes = &.{ 4, 1, 3 } }, .{ .at = 5, .tokens = 4, .bytes = &.{ 4, 2, 3 } } };
    try e.submit(&b, 0, woken.wake());
    waitParkedAtBegin(&e);
    h.slice.?(h.ctx, 1); // B reserved its 8 rows and runs its first picture
    try std.testing.expectEqual(@as(usize, 32), e.charged);
    // an urgent picture: B gives way after its first picture and goes back to the queue with its rows
    fake.hold_v.store(5, .release);
    fake.hold.store(true, .release);
    var taker: Taker = .{ .e = &e, .images = &a };
    defer taker.out.deinit(gpa);
    const t = try std.Thread.spawn(.{}, Taker.run, .{&taker});
    while (fake.entered.load(.acquire) < 2) sleepMs(1); // A's picture in flight, B queued
    e.lock();
    const b_job = e.find(&b).?;
    const b_queued = b_job.state == .queued;
    const b_charge = b_job.charge;
    e.unlock();
    try std.testing.expect(b_queued);
    try std.testing.expectEqual(@as(usize, 32), b_charge);
    try chargesMatch(&e);
    try std.testing.expectEqual(@as(usize, 40), e.charged); // and A's 2 rows past nothing: 32 + 8
    fake.hold.store(false, .release);
    t.join();
    try std.testing.expectEqual(@as(?usize, 2), taker.tokens);
    // B, queued again with its rows: cancelled before it resumes, its whole charge goes
    waitParkedAtBegin(&e);
    e.forget(&b);
    while (jobsLeft(&e) > 1) h.slice.?(h.ctx, 1);
    try chargesMatch(&e);
    try std.testing.expectEqual(@as(usize, 8), e.charged);
    h.release.?(h.ctx);
    e.forget(&a);
    try std.testing.expectEqual(@as(usize, 0), e.charged);
}

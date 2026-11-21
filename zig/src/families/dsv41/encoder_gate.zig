//! The tower's gate (TF_DS_TOWER_SLICE_MS): it runs only in slices between rounds while streams decode.
const std = @import("std");

/// The encoder thread's way past `begin`: go, or give way to the urgent job.

pub const Pass = enum { go, give_way };

/// Encoder thread: begin, yield after each block, end. Engine thread: hold, slice, release. Rush: an urgent waiter.
pub const Gate = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init, // taken after the encoder's mutex, never held while taking another lock
    cond: std.Io.Condition = .init, // every change: the encoder thread and the engine thread wait on it
    state: enum { free, held, slicing } = .free,
    until: i96 = 0, // a slice's end (awake clock)
    owed: bool = false, // a slice's first block, run whatever the time
    rush: ?*const anyopaque = null, // the urgent job a prompt pass waits for (its encode passes; others give way)
    closed: bool = false, // the encoder stops: free for good
    in_picture: bool = false, // between `begin` and `end`
    parked: bool = false, // the encoder thread waits at `begin` or `yield` (its stream idle)
    paced: std.atomic.Value(bool) = .init(false), // held once: jobs change at a picture boundary for an urgent one

    fn lock(g: *Gate) void {
        g.mutex.lockUncancelable(g.io);
    }

    fn unlock(g: *Gate) void {
        g.mutex.unlock(g.io);
    }

    fn now(g: *const Gate) i96 {
        return std.Io.Clock.awake.now(g.io).toNanoseconds();
    }

    /// Whether the encoder thread may run its next block (the lock held).
    fn open(g: *const Gate) bool {
        return switch (g.state) {
            .free => true,
            .held => false,
            .slicing => g.owed or g.now() < g.until,
        };
    }

    /// Whether the encoder thread is in a picture or waits to begin one (the lock held).
    fn busy(g: *const Gate) bool {
        return g.in_picture or g.parked;
    }

    fn park(g: *Gate) void {
        g.parked = true;
        g.cond.broadcast(g.io);
        g.cond.waitUncancelable(g.io, &g.mutex);
        g.parked = false;
    }

    fn passed(g: *Gate) void {
        if (g.state == .slicing) g.owed = false;
    }

    /// Encoder thread, before a picture of job `who`: waits while held; `.give_way` when another job is urgent (paced).
    pub fn begin(g: *Gate, who: *const anyopaque) Pass {
        g.lock();
        defer g.unlock();
        while (true) {
            if (g.rush) |r| {
                if (r != who and g.paced.load(.acquire)) return .give_way;
                break;
            }
            if (g.closed or g.open()) break;
            g.park();
        }
        g.passed();
        g.in_picture = true;
        return .go;
    }

    /// Encoder thread, after a block (stream synchronized): waits while held, unless a rush or closed.
    pub fn yield(g: *Gate) void {
        g.lock();
        defer g.unlock();
        while (!(g.closed or g.rush != null or g.open())) g.park();
        g.passed();
    }

    /// Encoder thread, the picture's rows on the host or its encode failed (in a defer): wakes a `hold` or `slice`.
    pub fn end(g: *Gate) void {
        g.lock();
        defer g.unlock();
        g.in_picture = false;
        g.cond.broadcast(g.io);
    }

    /// Whether a block should synchronize its stream and yield (the gate held or slicing): a free tower is not paused.
    pub fn pausing(g: *Gate) bool {
        g.lock();
        defer g.unlock();
        return g.state != .free;
    }

    /// Engine thread: the tower runs only in `slice` from now; returns once the block in flight is done (stream idle).
    pub fn hold(g: *Gate) void {
        g.lock();
        defer g.unlock();
        g.paced.store(true, .release);
        if (g.state == .slicing) g.owed = false;
        g.state = .held;
        while (g.in_picture and !g.parked and !g.closed and g.rush == null) g.cond.waitUncancelable(g.io, &g.mutex);
    }

    /// Engine thread, held: one pass at least, then until `budget_ns`; returns held and idle (at once: no picture).
    pub fn slice(g: *Gate, budget_ns: u64) void {
        g.lock();
        defer g.unlock();
        if (g.state == .free or g.closed or !g.busy()) return;
        g.state = .slicing;
        g.until = g.now() + budget_ns;
        g.owed = true;
        g.cond.broadcast(g.io);
        while (g.busy() and !(g.parked and !g.owed) and !g.closed and g.rush == null) g.cond.waitUncancelable(g.io, &g.mutex);
        g.owed = false;
        g.state = .held;
    }

    /// Engine thread: the tower runs free (no stream decodes).
    pub fn release(g: *Gate) void {
        g.lock();
        defer g.unlock();
        g.state = .free;
        g.owed = false;
        g.cond.broadcast(g.io);
    }

    /// The encoder stops: a parked thread ends its picture free, and nothing waits on the gate again.
    pub fn close(g: *Gate) void {
        g.lock();
        defer g.unlock();
        g.closed = true;
        g.cond.broadcast(g.io);
    }

    /// A prompt pass waits for job `who` (the encoder's mutex held).
    pub fn rushFor(g: *Gate, who: *const anyopaque) void {
        g.lock();
        defer g.unlock();
        g.rush = who;
        g.cond.broadcast(g.io);
    }

    /// The urgent job ended (encoder thread, its mutex held, before the wake): stream idle, the gate binds all again.
    pub fn rushDone(g: *Gate) void {
        g.lock();
        defer g.unlock();
        std.debug.assert(!g.in_picture);
        g.rush = null;
        g.cond.broadcast(g.io);
    }
};

/// Tests: fails the test binary if still armed after 5 s, so a hang is a failure (encoder_test.zig uses it too).
pub const Watchdog = struct {
    done: std.atomic.Value(bool) = .init(false),
    thread: ?std.Thread = null,

    pub fn arm(w: *Watchdog) !void {
        w.thread = try std.Thread.spawn(.{}, watch, .{w});
    }

    fn watch(w: *Watchdog) void {
        var ms: u32 = 0;
        while (ms < 5000) : (ms += 5) {
            if (w.done.load(.acquire)) return;
            sleepMs(5);
        }
        @panic("encoder watchdog: no progress for 5 s (a hang)");
    }

    pub fn disarm(w: *Watchdog) void {
        w.done.store(true, .release);
        if (w.thread) |t| t.join();
    }
};

/// A worker that runs `blocks` blocks a picture through the gate, as the encoder thread does.
const Worker = struct {
    gate: *Gate,
    pictures: u32,
    blocks: u32,
    ran: std.atomic.Value(u32) = .init(0),
    in_block: std.atomic.Value(bool) = .init(false),
    gave_way: bool = false,
    who: u8 = 0,

    fn run(w: *Worker) void {
        for (0..w.pictures) |_| {
            if (w.gate.begin(&w.who) == .give_way) {
                w.gave_way = true;
                return;
            }
            defer w.gate.end();
            for (0..w.blocks) |_| {
                w.in_block.store(true, .release);
                std.Io.sleep(std.testing.io, .fromMilliseconds(2), .awake) catch {};
                _ = w.ran.fetchAdd(1, .acq_rel);
                w.in_block.store(false, .release);
                if (w.gate.pausing()) w.gate.yield();
            }
        }
    }
};

fn sleepMs(ms: i64) void {
    std.Io.sleep(std.testing.io, .fromMilliseconds(ms), .awake) catch {};
}

fn waitParked(g: *Gate) void {
    while (true) {
        g.lock();
        const parked = g.parked;
        g.unlock();
        if (parked) return;
        sleepMs(1);
    }
}

test "a held gate runs no block outside a slice, one block at least in a slice of 1 ns, and all once released" {
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var g: Gate = .{ .io = std.testing.io };
    g.hold();
    var w: Worker = .{ .gate = &g, .pictures = 2, .blocks = 3 };
    const t = try std.Thread.spawn(.{}, Worker.run, .{&w});
    defer t.join();
    waitParked(&g);
    sleepMs(20);
    try std.testing.expectEqual(@as(u32, 0), w.ran.load(.acquire)); // parked at begin
    for (1..4) |i| {
        g.slice(1);
        try std.testing.expect(!w.in_block.load(.acquire)); // returns idle
        try std.testing.expectEqual(@as(u32, @intCast(i)), w.ran.load(.acquire)); // exactly one block a 1 ns slice
        sleepMs(10);
        try std.testing.expectEqual(@as(u32, @intCast(i)), w.ran.load(.acquire));
    }
    g.hold();
    try std.testing.expect(!w.in_block.load(.acquire));
    g.release();
    while (w.ran.load(.acquire) < 6) sleepMs(1);
}

test "hold waits for the block in flight; slice and hold return at once with nothing in flight" {
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var g: Gate = .{ .io = std.testing.io };
    g.hold();
    g.slice(1_000_000_000); // nothing in flight: at once
    var w: Worker = .{ .gate = &g, .pictures = 1, .blocks = 50 };
    g.release();
    const t = try std.Thread.spawn(.{}, Worker.run, .{&w});
    while (w.ran.load(.acquire) < 2) sleepMs(1);
    g.hold();
    try std.testing.expect(!w.in_block.load(.acquire));
    const n = w.ran.load(.acquire);
    sleepMs(20);
    try std.testing.expectEqual(n, w.ran.load(.acquire));
    g.close(); // free for good: the picture ends
    t.join();
    try std.testing.expectEqual(@as(u32, 50), w.ran.load(.acquire));
    g.hold();
    g.slice(1);
}

test "a rush lets the picture in flight drain; another job gives way at begin; the urgent one passes" {
    var wd: Watchdog = .{};
    try wd.arm();
    defer wd.disarm();
    var g: Gate = .{ .io = std.testing.io };
    g.hold();
    g.release();
    var b: Worker = .{ .gate = &g, .pictures = 2, .blocks = 3 };
    g.hold(); // paced from here
    const tb = try std.Thread.spawn(.{}, Worker.run, .{&b});
    waitParked(&g);
    g.slice(1); // b's first block, then parked mid-picture
    try std.testing.expectEqual(@as(u32, 1), b.ran.load(.acquire));
    var a: Worker = .{ .gate = &g, .pictures = 1, .blocks = 2, .who = 1 };
    g.rushFor(&a.who);
    tb.join(); // its picture drained, then it gave way at its second
    try std.testing.expect(b.gave_way);
    try std.testing.expectEqual(@as(u32, 3), b.ran.load(.acquire));
    a.run(); // the urgent job runs held
    try std.testing.expectEqual(@as(u32, 2), a.ran.load(.acquire));
    g.rushDone();
    var again: Worker = .{ .gate = &g, .pictures = 1, .blocks = 1 };
    const t2 = try std.Thread.spawn(.{}, Worker.run, .{&again});
    sleepMs(20);
    try std.testing.expectEqual(@as(u32, 0), again.ran.load(.acquire)); // held again for every other job
    g.release();
    t2.join();
}

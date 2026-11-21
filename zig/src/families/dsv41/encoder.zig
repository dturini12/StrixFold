//! Rank 0's picture encodes on a thread of their own (HIP); rows cache, gate and budget each off unless set (Options).
const std = @import("std");
const lanes = @import("lanes");
const RowsCache = @import("rows_cache.zig").RowsCache;
const Gate = @import("encoder_gate.zig").Gate;
const Pass = @import("encoder_gate.zig").Pass;
const Allocator = std.mem.Allocator;
const Image = lanes.stream.Image;
const Wake = lanes.stream.Wake;
const Digest = lanes.image_key.Digest;

/// The tower the thread drives: `bind` once on that thread, `rows` appends a picture's span rows, returns its tokens.
pub const Tower = struct {
    ctx: *anyopaque,
    bind: *const fn (ctx: *anyopaque) anyerror!void,
    // `pace` after each block: the tower's stream synchronized then the gate's yield while the gate pauses it
    rows: *const fn (ctx: *anyopaque, payload: []const u8, gpa: Allocator, out: *std.ArrayList(u8), pace: Pace) anyerror!usize,
    row_bytes: usize, // a span row's bytes: a job reserves its pictures' rows exactly before their encode
    // a picture's payload from its source (Image.source), in `gpa`; null: payloads only
    prepare: ?*const fn (ctx: *anyopaque, gpa: Allocator, source: []const u8) anyerror![]align(16) u8 = null,
};

/// The gate as the tower sees it between two blocks.
pub const Pace = struct {
    gate: *Gate,

    /// Whether the block just launched must end (stream synchronized) and `yield` be asked: the gate is not free.
    pub fn pausing(p: Pace) bool {
        return p.gate.pausing();
    }

    pub fn yield(p: Pace) void {
        p.gate.yield();
    }
};

const Pic = struct {
    data: []align(16) u8 = &.{}, // the payload to encode, or with `lazy` its source (empty: rows there, skipped, ready)
    lazy: bool = false,
    digest: ?Digest = null,
    want: u32 = 0, // its span as the request says (Image.tokens): a cached copy's key with the digest
    tokens: usize = 0, // its span as its encode returned it
    range: [2]usize = .{ 0, 0 }, // its rows in the job's `rows`
    skipped: bool = false, // a resume covers its span: encoded only when a prompt pass asks for it (take)
};

const Job = struct {
    key: usize, // the request's images.ptr
    pics: []Pic,
    rows: std.ArrayList(u8) = .empty,
    need: usize = 0, // the span rows of the pictures it encodes or copies (their tokens): `rows` is reserved exactly
    charge: usize = 0, // its rows' capacity, counted in Encoder.charged from allocation to free (the budget)
    state: enum { queued, running, done } = .queued,
    urgent: bool = false, // a prompt pass waits for it: the thread takes it first
    err: ?anyerror = null,
    gone: bool = false, // forgotten while it ran: the thread frees it
    wake: Wake,
    since: i96 = 0, // submitted (its log line's ms)
    counts: struct { encoded: u32 = 0, cached: u32 = 0, skipped: u32 = 0 } = .{},
    waited: bool = false, // its wait for the budget logged
};

/// Settings the engine reads at start (native.zig); the defaults are the encoder without them.
pub const Options = struct {
    rows_cache_bytes: usize = 0, // TF_DS_IMAGE_ROWS_CACHE_MIB: encoded rows kept by picture digest (0: none)
    rows_budget_bytes: usize = 0, // TF_DS_IMAGE_ROWS_BUDGET_MIB: rows the jobs hold together, allocation to free (0: none)
};

pub const Encoder = struct {
    gpa: Allocator,
    io: std.Io,
    tower: Tower,
    cap: usize, // a request's image rows at most (Model.image_rows): more fail at once, nothing encoded
    options: Options = .{},
    mutex: std.Io.Mutex = .init,
    cond: std.Io.Condition = .init,
    done: std.Io.Condition = .init, // a job finished (a prompt pass waits in take)
    jobs: std.ArrayList(*Job) = .empty,
    closing: bool = false,
    thread: ?std.Thread = null,
    gate: Gate,
    cache: RowsCache = .{ .cap = 0 }, // its cap from options at start
    charged: usize = 0, // the jobs' charges together
    woken_hook: ?Wake = null, // tests: an urgent waiter, woken, runs it before it returns (no lock held)

    pub fn init(gpa: Allocator, io: std.Io, tower: Tower, cap: usize) Encoder {
        return .{ .gpa = gpa, .io = io, .tower = tower, .cap = cap, .gate = .{ .io = io } };
    }

    pub fn start(e: *Encoder) !void {
        e.cache.cap = e.options.rows_cache_bytes;
        e.thread = try std.Thread.spawn(.{}, run, .{e});
    }

    /// Ends the thread once its current picture is done (the gate free for good), then frees every job.
    pub fn stop(e: *Encoder) void {
        e.lock();
        e.closing = true;
        e.gate.close();
        e.cond.broadcast(e.io);
        e.done.broadcast(e.io);
        e.unlock();
        if (e.thread) |t| t.join();
        e.thread = null;
        for (e.jobs.items) |j| e.free(j);
        e.jobs.deinit(e.gpa);
        e.cache.deinit(e.gpa);
    }

    /// The lane host's view (lanes.stream.Encode).
    pub fn hook(e: *Encoder) lanes.stream.Encode {
        return .{ .ctx = e, .start = startFn, .ready = readyFn, .forget = forgetFn, .hold = holdFn, .slice = sliceFn, .release = releaseFn };
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

    fn holdFn(ctx: *anyopaque) void {
        of(ctx).gate.hold();
    }

    fn sliceFn(ctx: *anyopaque, budget_ns: u64) void {
        of(ctx).gate.slice(budget_ns);
    }

    fn releaseFn(ctx: *anyopaque) void {
        of(ctx).gate.release();
    }

    pub fn lock(e: *Encoder) void {
        e.mutex.lockUncancelable(e.io);
    }

    pub fn unlock(e: *Encoder) void {
        e.mutex.unlock(e.io);
    }

    fn now(e: *const Encoder) i96 {
        return std.Io.Clock.awake.now(e.io).toNanoseconds();
    }

    /// Queues a request's pictures past `from` (a resume covers the rest), payloads or sources copied; cached rows now.
    pub fn submit(e: *Encoder, images: []const Image, from: u32, wake: Wake) !void {
        const j = try e.gpa.create(Job);
        j.* = .{ .key = @intFromPtr(images.ptr), .pics = &.{}, .wake = wake, .since = e.now() };
        errdefer e.free(j);
        j.pics = try e.gpa.alloc(Pic, images.len);
        @memset(j.pics, .{});
        var rows: usize = 0;
        for (images) |img| rows += img.tokens;
        if (rows > e.cap) {
            j.state = .done;
            j.err = error.TooManyImageRows;
        } else for (images, j.pics) |img, *p| {
            p.* = .{ .digest = img.digest, .want = img.tokens, .skipped = !img.rows and @as(usize, img.at) + img.tokens <= from };
            if (img.rows) continue;
            if (p.skipped) {
                j.counts.skipped += 1;
                continue;
            }
            try e.copyIn(p, img);
            j.need += img.tokens;
        }
        e.lock();
        defer e.unlock();
        try e.jobs.append(e.gpa, j);
        if (j.err != null) return;
        if (j.need > 0 and e.mayCopyNow(j)) e.copyCached(j) catch {}; // (else the thread copies them at its start)
        if (j.need == 0 or (j.rows.capacity > 0 and !e.left(j))) { // nothing to encode now
            j.state = .done;
            e.logJob(j);
            return;
        }
        e.cond.signal(e.io);
    }

    fn copyIn(e: *Encoder, p: *Pic, img: Image) !void {
        const lazy = img.bytes.len == 0 and img.source.len > 0;
        const src = if (lazy) img.source else img.bytes;
        p.data = try e.gpa.alignedAlloc(u8, .@"16", src.len);
        @memcpy(p.data, src);
        p.lazy = lazy;
    }

    /// Whether a job has a picture still to encode (the lock held).
    fn left(_: *const Encoder, j: *const Job) bool {
        for (j.pics) |p| if (p.data.len > 0) return true;
        return false;
    }

    /// At submit: a picture of `j` is cached and the budget lets `j` reserve before the jobs waiting for it (the lock).
    fn mayCopyNow(e: *Encoder, j: *Job) bool {
        if (e.cache.cap == 0) return false;
        const hit = for (j.pics) |p| {
            if (p.data.len > 0 and p.digest != null and e.cache.has(&p.digest.?, p.want)) break true;
        } else false;
        if (!hit or e.options.rows_budget_bytes == 0) return hit;
        for (e.jobs.items) |x| if (x != j and x.state == .queued and !x.urgent and !e.fits(x)) return false;
        return e.fits(j);
    }

    /// The job's exact reserve, counted in the budget's charges (the lock held).
    fn reserve(e: *Encoder, j: *Job) !void {
        try j.rows.ensureTotalCapacityPrecise(e.gpa, j.need * e.tower.row_bytes);
        e.recharge(j);
    }

    fn recharge(e: *Encoder, j: *Job) void {
        e.charged = e.charged - j.charge + j.rows.capacity;
        j.charge = j.rows.capacity;
    }

    /// Reserves the job's rows and copies those of its pictures the rows cache holds (the lock held).
    fn copyCached(e: *Encoder, j: *Job) !void {
        try e.reserve(j);
        if (e.cache.cap == 0) return;
        for (j.pics) |*p| if (p.data.len > 0) if (p.digest) |*d| {
            const begin = j.rows.items.len;
            if (!e.cache.copy(d, p.want, &j.rows)) continue;
            p.range = .{ begin, j.rows.items.len };
            p.tokens = p.want;
            e.gpa.free(p.data);
            p.data = &.{};
            j.counts.cached += 1;
        };
    }

    /// Whether job `j` may reserve now: no budget, reserved already, no other job holds rows, or room (the lock).
    fn fits(e: *const Encoder, j: *const Job) bool {
        const budget = e.options.rows_budget_bytes;
        const more = (j.need * e.tower.row_bytes) -| j.charge;
        return budget == 0 or more == 0 or e.charged == j.charge or e.charged + more <= budget;
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
        if (j.pics[k].skipped) try e.encodeNow(j, k, images[k]);
        const p = j.pics[k];
        try out.appendSlice(gpa, j.rows.items[p.range[0]..p.range[1]]);
        return p.tokens;
    }

    /// Picture `k` of a done job from the cache, else queued first and waited for, past gate and budget (lock held).
    fn encodeNow(e: *Encoder, j: *Job, k: usize, img: Image) !void {
        const p = &j.pics[k];
        j.need += img.tokens; // its rows reserved too, with those before
        errdefer j.need -= img.tokens;
        if (e.cache.cap > 0) if (img.digest) |*d| if (e.cache.has(d, img.tokens)) {
            try e.reserve(j);
            const begin = j.rows.items.len;
            if (e.cache.copy(d, img.tokens, &j.rows)) {
                p.* = .{ .digest = d.*, .want = img.tokens, .tokens = img.tokens, .range = .{ begin, j.rows.items.len } };
                j.counts.skipped -= 1;
                j.counts.cached += 1;
                return;
            }
        };
        try e.copyIn(p, img);
        p.skipped = false;
        j.counts.skipped -= 1;
        j.state = .queued;
        j.urgent = true;
        e.gate.rushFor(j);
        e.cond.signal(e.io);
        while (j.state != .done) {
            if (e.closing) return error.Closed;
            e.done.waitUncancelable(e.io, &e.mutex);
        }
        if (e.woken_hook) |h| {
            e.unlock(); // (only this thread changes a done job)
            h.call(h.ctx);
            e.lock();
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
        e.cond.signal(e.io); // its charge is back: a job waiting for the budget may start
    }

    pub fn find(e: *Encoder, images: []const Image) ?*Job {
        if (images.len == 0) return null;
        for (e.jobs.items) |j| if (j.key == @intFromPtr(images.ptr) and !j.gone) return j;
        return null;
    }

    fn remove(e: *Encoder, j: *Job) void {
        const i = std.mem.indexOfScalar(*Job, e.jobs.items, j) orelse return;
        _ = e.jobs.orderedRemove(i);
    }

    fn freePayloads(e: *Encoder, j: *Job) void {
        for (j.pics) |*p| {
            e.gpa.free(p.data);
            p.data = &.{};
        }
    }

    /// Frees a job and its charge (a charged job: the lock held, or the thread joined).
    fn free(e: *Encoder, j: *Job) void {
        if (j.charge > 0) e.charged -= j.charge; // (submit's error path frees an unpublished job, unlocked)
        e.freePayloads(j);
        e.gpa.free(j.pics);
        j.rows.deinit(e.gpa);
        e.gpa.destroy(j);
    }

    fn logJob(e: *const Encoder, j: *const Job) void {
        if (e.cache.cap == 0) return;
        const c = j.counts;
        std.log.info("encoder: pictures={d} encoded={d} cached={d} skipped={d} ms={d}", .{ j.pics.len, c.encoded, c.cached, c.skipped, @divTrunc(e.now() - j.since, std.time.ns_per_ms) });
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
            const copied = e.copyCached(j);
            e.unlock();
            var err: ?anyerror = null;
            var gave = false;
            if (bound) |_| {
                if (copied) |_| {
                    gave = e.encode(j) catch |x| blk: {
                        err = x;
                        break :blk false;
                    };
                } else |x| err = x;
            } else |x| err = x;
            e.lock();
            if (gave) {
                if (!j.gone and !e.closing) { // back to the queue with its rows and charge, unwoken; the urgent job next
                    j.state = .queued;
                    continue;
                }
                err = error.EncodeCancelled;
            }
            j.state = .done;
            j.err = err;
            if (j.urgent) e.gate.rushDone(); // its picture ended (rows synchronized): held again before the waiter wakes
            j.urgent = false;
            e.freePayloads(j);
            e.logJob(j);
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

    /// The queued job a prompt pass waits for, else the first queued if the budget lets it start (the lock held).
    fn next(e: *Encoder) ?*Job {
        var first: ?*Job = null;
        for (e.jobs.items) |x| if (x.state == .queued) {
            if (x.urgent) return x;
            if (first == null) first = x;
        };
        const j = first orelse return null;
        if (e.fits(j)) return j;
        if (!j.waited) std.log.info("encoder: a job of {d} MiB of rows waits for the budget ({d} of {d} MiB held)", .{ (j.need * e.tower.row_bytes) >> 20, e.charged >> 20, e.options.rows_budget_bytes >> 20 });
        j.waited = true;
        return null;
    }

    /// Encodes the job's pictures left; true when it gave way to an urgent job at a picture boundary (paced gate).
    fn encode(e: *Encoder, j: *Job) !bool {
        for (j.pics) |*p| if (p.data.len > 0) {
            switch (e.ask(j)) {
                .cancel => return error.EncodeCancelled,
                .give_way => return true,
                .go => {},
            }
            if (try e.picture(j, p) == .give_way) return true;
        };
        return false;
    }

    /// Before job `j`'s next picture: it was forgotten or the encoder stops, or (paced) an urgent job is queued.
    fn ask(e: *Encoder, j: *const Job) enum { go, cancel, give_way } {
        e.lock();
        defer e.unlock();
        if (j.gone or e.closing) return .cancel;
        if (e.gate.paced.load(.acquire)) for (e.jobs.items) |x| if (x != j and x.urgent and x.state == .queued) return .give_way;
        return .go;
    }

    /// One picture through the gate (prepared from its source first if sent so), its rows kept, its payload freed.
    fn picture(e: *Encoder, j: *Job, p: *Pic) !Pass {
        const payload = if (p.lazy) try e.prepared(p) else p.data;
        defer if (p.lazy) e.gpa.free(payload);
        {
            if (e.gate.begin(j) == .give_way) return .give_way;
            defer e.gate.end();
            if (e.ask(j) == .cancel) return error.EncodeCancelled; // forgotten while it waited to begin
            const begin = j.rows.items.len;
            p.tokens = try e.tower.rows(e.tower.ctx, payload, e.gpa, &j.rows, .{ .gate = &e.gate });
            p.range = .{ begin, j.rows.items.len };
        }
        e.lock();
        defer e.unlock();
        e.gpa.free(p.data);
        p.data = &.{};
        j.counts.encoded += 1;
        e.recharge(j); // (exact: no change unless a span differed from its request's)
        if (p.digest) |*d| if (p.tokens == p.want) e.cache.put(e.gpa, d, p.want, j.rows.items[p.range[0]..p.range[1]]);
        return .go;
    }

    /// A picture's payload prepared from its source on this thread, its digest the one the server took for it.
    fn prepared(e: *Encoder, p: *const Pic) ![]align(16) u8 {
        const prep = e.tower.prepare orelse return error.NoImagePrepare;
        const payload = try prep(e.tower.ctx, e.gpa, p.data);
        errdefer e.gpa.free(payload);
        const d = p.digest orelse return error.ImageChanged;
        if (!std.mem.eql(u8, &lanes.image_key.imageDigest(payload), &d)) return error.ImageChanged;
        return payload;
    }
};

test {
    _ = @import("encoder_test.zig");
    _ = @import("encoder_gate.zig");
    _ = @import("rows_cache.zig");
}

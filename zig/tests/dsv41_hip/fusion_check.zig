//! tf-dsv41-fusion-check: the launch-fusion bundle's one-node window (out/fusion-20261121/PLAN.md 6.2); see `usage`.

const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const tri_aux = dsv41.tri_aux;
const hip_load = dsv41.hip_load;
const cm = @import("fusion_check_common.zig");
const route = @import("fusion_check_route.zig");
const hc = @import("fusion_check_hc.zig");
const Env = cm.Env;
const Report = cm.Report;
const Msg = cm.Msg;
const Opts = cm.Opts;

const usage =
    \\usage: tf-dsv41-fusion-check --kit DIR [--steps score,route,hc] [--oracle FILE --oracle-sha HEX]
    \\         [--aux-object SHA] [--score-batches N] [--replays N] [--seed N] [--receipt PATH] [--dry-run]
    \\The launch-fusion window (out/fusion-20261121/PLAN.md 6.2) on one GPU, no peer, no model, through the family's
    \\own launch code and the candidate's own serve kit (Triton variants and native objects, as hip_load loads them).
    \\Old and new are compared bit for bit; any difference fails its step and names the first one:
    \\  score  T2's oracle _t_score (the kit's pinned Triton at num_warps 1, BLOCK 512, as _route_parts runs; a
    \\         standalone code object of the given sha256) vs the kit's aux.route_score (the HIP score helper as built,
    \\         in the object of aux.route_pick), on all 2^32 fp32 patterns in 256 batches of 2^24: every output word
    \\         (a NaN that differs only in its payload counts, and is reported apart).
    \\  route  tri_norm.route with TF_DS_AUX=hip (the kit's _route_parts) vs hip,route (aux.route_pick): rows 1-16,
    \\         t_route.py's four logit regimes, two biases, ties (equal rows, ±0 bias), ±inf and NaN chunk sums,
    \\         rows with fewer than six finite selection values, all-NaN rows, a degenerate bias, and an image span's
    \\         row offset: PICK and WTS with guard rows. Timed: a graph of 40 routes, rows 1, 3, 6, 16.
    \\  hc     tri_hc.hcPre2 as round.zig's mix calls it, tri_hc.inline_mix off (the round's sink: 4 launches posted,
    \\         3 unposted) vs on (no sink: _hc_mix_part, _hc_finish_s): rows 1-16, chains UPP and PUP so each site's
    \\         post and comb feed the next post; after every site h_out, x, PART, pre_out, post, comb.
    \\         Timed: a graph of a round's 80 sites (3 unposted), rows 1, 3, 6, 16.
    \\Every run is held to the launches it must make (strict-mode counts), so a silent fallback cannot pass as
    \\identity; strict mode (TF_GPU_STRICT unset or 1) is required. Inputs are synthetic and seeded.
    \\  --kit DIR          the candidate's serve kit (manifest.json + objects/)
    \\  --steps LIST       comma list of score, route, hc (default all three, in that order)
    \\  --oracle FILE      T2's _t_score code object (required by score); --oracle-sha its sha256 (required with it)
    \\  --aux-object SHA   the aux_route object T2-score-asm.md was made on: aux.route_pick and aux.route_score
    \\                     must both live in exactly that kit object
    \\  --score-batches N  batches of 2^24 inputs from pattern 0 (default 256 = all 2^32; fewer only for a smoke run)
    \\  --replays N        timed graph replays a row count (default 2000)
    \\  --seed N           the synthetic inputs' seed (default 1)
    \\  --receipt PATH     a JSON receipt of every check and timing
    \\  --dry-run          no GPU: arguments, manifest and object hashes, the oracle's hash and metadata, the plan
    \\Exit: 0 identical and route and hc each at least 0.2 ms a round faster at one row; 4 identical but a timing
    \\short of that; 1 a difference, a refused check or an error; 2 usage. Run with the engine stopped
    \\(out/fusion-20261121/window/).
    \\
;

fn parseSteps(spec: []const u8) !cm.Steps {
    var s: cm.Steps = .{};
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |item| {
        if (std.mem.eql(u8, item, "score")) s.score = true else if (std.mem.eql(u8, item, "route")) s.route = true else if (std.mem.eql(u8, item, "hc")) s.hc = true else return error.BadSteps;
    }
    return s;
}

fn isHex64(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) return false;
    return true;
}

fn parseArgs(args: []const []const u8) !Opts {
    var o: Opts = .{ .kit = "", .steps = .{ .score = true, .route = true, .hc = true } };
    var kit: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--dry-run")) {
            o.dry_run = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--help") or std.mem.eql(u8, a, "-h")) return error.Help;
        if (i + 1 >= args.len) return error.MissingValue;
        const v = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, a, "--kit")) kit = v else if (std.mem.eql(u8, a, "--steps")) o.steps = try parseSteps(v) else if (std.mem.eql(u8, a, "--oracle")) o.oracle = v else if (std.mem.eql(u8, a, "--oracle-sha")) o.oracle_sha = v else if (std.mem.eql(u8, a, "--aux-object")) o.aux_object = v else if (std.mem.eql(u8, a, "--score-batches")) o.score_batches = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--replays")) o.replays = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, a, "--seed")) o.seed = try std.fmt.parseInt(u64, v, 10) else if (std.mem.eql(u8, a, "--receipt")) o.receipt = v else return error.UnknownOption;
    }
    o.kit = kit orelse return error.NoKit;
    if (o.steps.score and (o.oracle == null or o.oracle_sha == null)) return error.ScoreNeedsOracle;
    if ((o.oracle == null) != (o.oracle_sha == null)) return error.OracleNeedsSha;
    if (o.oracle_sha) |s| if (!isHex64(s)) return error.BadSha;
    if (o.aux_object) |s| if (!isHex64(s)) return error.BadSha;
    if (o.score_batches == 0 or o.score_batches > cm.score_batches_all) return error.BadBatches;
    if (o.replays == 0) return error.BadReplays;
    return o;
}

// ---- score: all 2^32 fp32 inputs through the oracle and the HIP helper

const score_n: usize = 1 << 24;
/// The oracle's BLOCK (elements a program) and threads (num_warps 1, wave 32); the helper's block.
const oracle_block = 512;
const oracle_threads = 32;
const score_threads = 256;

/// The kit's aux.route_score: out[i] = the router score of in[i], i < n (abi.h tf_ds_aux_route_score_args).
const ScoreParams = extern struct { a: gpu.kernel_abi.tf_ds_aux_route_score_args };

/// T2's oracle: its sha256, the arch, one _t_score with Triton's parameters (L, OUT, n, 2 scratch) at num_warps 1.
const Oracle = struct {
    module: gpu.Module,
    f: gpu.Function,

    fn check(a: std.mem.Allocator, bytes: []const u8, want_sha: []const u8, arch: []const u8, msg: *Msg) !gpu.hsaco.Kernel {
        const sha = gpu.manifest.sha256Hex(bytes);
        msg.put("oracle sha256 {s}", .{&sha});
        if (!std.mem.eql(u8, &sha, want_sha)) {
            msg.put(", expected {s}", .{want_sha});
            return error.OracleHash;
        }
        try gpu.checkImage(bytes, arch);
        const co = try gpu.hsaco.parse(a, bytes);
        const k = co.find("_t_score") orelse {
            msg.put(", no _t_score kernel in it", .{});
            return error.OracleKernel;
        };
        const ex = k.explicit();
        const sizes = [_]u32{ 8, 8, 4, 8, 8 };
        var ok = ex.len == sizes.len and k.wave == 32 and k.max_flat_workgroup_size == oracle_threads and !k.dynamic_stack and k.scratch == 0;
        if (ok) for (ex, sizes) |x, sz| {
            ok = ok and x.size == sz;
        };
        msg.put(", _t_score: {d} explicit args, wave {d}, max workgroup {d}, LDS {d}, scratch {d}", .{ ex.len, k.wave, k.max_flat_workgroup_size, k.lds_static, k.scratch });
        if (!ok) {
            msg.put(" (expected L, OUT, n, 2 scratch pointers; wave 32; num_warps 1 = 32 threads; no scratch)", .{});
            return error.OracleKernel;
        }
        return k;
    }

    fn load(e: *const Env, bytes: []const u8) !Oracle {
        var m = try gpu.Module.load(e.d, bytes);
        errdefer m.unload();
        return .{ .module = m, .f = try m.function("_t_score") };
    }

    /// out = _t_score(in): grid cdiv(n, 512) x 32 threads, then 2 null scratch pointers (tri/common.py's jit_kernel).
    fn run(o: Oracle, e: *const Env, in: u64, out: u64, n: usize) !void {
        var a: gpu.Args = .{};
        a.add(in);
        a.add(out);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(u64, 0));
        a.add(@as(u64, 0));
        // the kit's strict check is off for this launch alone: the oracle is not a kit kernel
        const saved = gpu.launch.before;
        gpu.launch.before = null;
        defer gpu.launch.before = saved;
        try gpu.launch.launch(o.f, .{ .grid = .{ .x = @intCast((n + oracle_block - 1) / oracle_block) }, .block = .{ .x = oracle_threads } }, e.s, &a);
    }
};

/// Words of `x` and `y` that differ: the count, those only in a NaN payload, and the first one's input.
const Diff = struct { n: u64 = 0, nan_payload: u64 = 0, first: ?[3]u32 = null };

fn stepScore(e: *const Env, r: *Report, arch: []const u8) !void {
    var msg: Msg = .{};
    const o = e.o;
    var arena: std.heap.ArenaAllocator = .init(e.gpa);
    defer arena.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAllocOptions(e.io, o.oracle.?, arena.allocator(), .limited(64 << 20), .@"16", null);
    _ = Oracle.check(arena.allocator(), bytes, o.oracle_sha.?, arch, &msg) catch |err| {
        r.check("score.oracle", false, msg.text());
        return err;
    };
    var oracle = try Oracle.load(e, bytes);
    defer oracle.module.unload();
    r.check("score.oracle", true, msg.text());
    msg = .{};
    const aux_ok = cm.auxObjectOk(e.kit.m, &msg, o.aux_object);
    r.check("score.aux_object", aux_ok, msg.text());
    if (!aux_ok) return error.AuxObject;
    const hip = try gpu.Kernel("aux.route_score", ScoreParams).bind(e.kit, &.{});
    var din = try e.alloc(score_n * 4);
    defer din.free();
    var dor = try e.alloc(score_n * 4);
    defer dor.free();
    var dhp = try e.alloc(score_n * 4);
    defer dhp.free();
    const hin = try e.gpa.alloc(u32, score_n);
    defer e.gpa.free(hin);
    const hor = try e.gpa.alloc(u32, score_n);
    defer e.gpa.free(hor);
    const hhp = try e.gpa.alloc(u32, score_n);
    defer e.gpa.free(hhp);
    var d: Diff = .{};
    const before = cm.launchesOf(e.kit, "aux.route_score");
    const t0 = cm.nowNs();
    for (0..o.score_batches) |b| {
        for (hin, 0..) |*x, i| x.* = @intCast((b << 24) | i);
        try e.put(din, 0, hin);
        // different fills: an element either side leaves unwritten shows as a difference
        try e.fill(dor.ptr, 0xA5, score_n * 4);
        try e.fill(dhp.ptr, 0x5A, score_n * 4);
        try oracle.run(e, din.ptr, dor.ptr, score_n);
        const args: gpu.kernel_abi.tf_ds_aux_route_score_args = .{ .in = din.ptr, .out = dhp.ptr, .n = @intCast(score_n), .pad0 = 0 };
        try hip.launch(e.s, .{ .grid = .{ .x = @intCast(score_n / score_threads) }, .block = .{ .x = score_threads } }, .{ .a = args });
        try e.get(dor, 0, hor);
        try e.get(dhp, 0, hhp);
        if (std.mem.eql(u32, hor, hhp)) continue;
        for (hor, hhp, hin) |x, y, in| if (x != y) {
            d.n += 1;
            if (cm.isNan32(x) and cm.isNan32(y)) d.nan_payload += 1;
            if (d.first == null) d.first = .{ in, x, y };
        };
    }
    const ms = (cm.nowNs() - t0) / std.time.ns_per_ms;
    const ran = cm.launchesOf(e.kit, "aux.route_score") - before;
    const all = o.score_batches == cm.score_batches_all;
    msg = .{};
    msg.put("{d} inputs ({d} of 256 batches{s}) through _t_score and aux.route_score ({d} launches): {d} differ ({d} only in a NaN payload); {d} ms", .{ @as(u64, o.score_batches) * score_n, o.score_batches, if (all) ", every fp32 pattern" else ", NOT every pattern", ran, d.n, d.nan_payload, ms });
    if (d.first) |f| msg.put("; first input 0x{x:0>8} ({e}): oracle 0x{x:0>8}, HIP 0x{x:0>8}", .{ f[0], @as(f32, @bitCast(f[0])), f[1], f[2] });
    r.check("score.all_inputs", d.n == 0 and ran == o.score_batches and all, msg.text());
}

// ---- dry run (no GPU)

fn dryRun(gpa: std.mem.Allocator, io: std.Io, o: *const Opts, r: *Report) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var msg: Msg = .{};
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit, "manifest.json" }), a, .limited(64 << 20));
    const m = try gpu.manifest.Manifest.parse(a, text);
    var bad_objects: usize = 0;
    for (m.objects) |ob| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ o.kit, ob.file }), gpa, .limited(1 << 30));
        defer gpa.free(bytes);
        if (!std.mem.eql(u8, &gpu.manifest.sha256Hex(bytes), ob.sha256)) {
            if (bad_objects == 0) msg.put("{s} does not match its manifest sha256; ", .{ob.file});
            bad_objects += 1;
        }
    }
    msg.put("kit {s}: {d} objects ({d} differ from the manifest), {d} kernels, manifest sha256 {s}", .{ o.kit, m.objects.len, bad_objects, m.kernels.len, &gpu.manifest.sha256Hex(text) });
    r.check("dry.kit", bad_objects == 0, msg.text());
    if (o.steps.route or o.steps.score) {
        msg = .{};
        r.check("dry.aux_object", cm.auxObjectOk(m, &msg, o.aux_object), msg.text());
    }
    // the Triton functions the steps launch (strict mode resolves each launch's exact variant on the GPU)
    msg = .{};
    var missing: usize = 0;
    for ([_][]const u8{ "_route_parts", "_hc_mix_part", "_hc_finish_s", "_hc_post_only", "_hc_sinkhorn" }) |f| {
        var n: usize = 0;
        for (m.kernels) |k| if (k.triton) |tt| {
            n += @intFromBool(std.mem.eql(u8, tt.@"fn", f));
        };
        msg.put("{s} {d} variants; ", .{ f, n });
        missing += @intFromBool(n == 0);
    }
    r.check("dry.triton", missing == 0, msg.text());
    if (o.oracle) |path| {
        msg = .{};
        const bytes = try std.Io.Dir.cwd().readFileAllocOptions(io, path, a, .limited(64 << 20), .@"16", null);
        const ok = if (Oracle.check(a, bytes, o.oracle_sha.?, m.arch, &msg)) |_| true else |_| false;
        r.check("dry.oracle", ok, msg.text());
    }
    const n_cases = (try route.cases(a)).len;
    r.add("{{\"plan\": {{\"steps\": \"{s}{s}{s}\", \"score_inputs\": {d}, \"route_cases\": {d}, \"hc_runs\": {d}, \"timed_rows\": [1, 3, 6, 16], \"replays\": {d}, \"route_launches\": {d}, \"hc_sites\": {d}, \"seed\": {d}}}}}", .{ if (o.steps.score) "score " else "", if (o.steps.route) "route " else "", if (o.steps.hc) "hc" else "", @as(u64, o.score_batches) * score_n, n_cases, hc.runs, o.replays, route.launches, hc.round_sites, o.seed });
}

/// The GPU run: the kit (strict), its Triton set, both TF_DS_AUX selections, then the steps asked for.
fn gpuRun(gpa: std.mem.Allocator, io: std.Io, o: *const Opts, r: *Report, id: *Ident) !void {
    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try ctx.caps();
    id.arch = caps.archName();
    const kit = try gpu.Kit.open(gpa, io, &d, o.kit, id.arch, .{ .vision = false });
    defer kit.deinit();
    id.kit_sha = kit.manifest_sha256;
    id.strict = kit.strict;
    if (!kit.strict) return r.check("strict", false, "TF_GPU_STRICT=0: the check needs strict mode (exact Triton variants, launch counts)");
    var set = try gpu.aot.Set.fromKit(gpa, kit);
    defer set.deinit();
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    const env: Env = .{
        .gpa = std.heap.page_allocator,
        .io = io,
        .d = &d,
        .s = s,
        .kit = kit,
        .set = &set,
        .aux_off = try hip_load.aux(kit, try tri_aux.Which.parse("hip")),
        .aux_on = try hip_load.aux(kit, try tri_aux.Which.parse("hip,route")),
        .o = o,
    };
    std.debug.print("kit {s}: {d} objects, {d} kernels, manifest sha256 {s}, strict {}\n", .{ o.kit, kit.m.objects.len, kit.m.kernels.len, &kit.manifest_sha256, kit.strict });
    if (o.steps.score) stepScore(&env, r, id.arch) catch |err| r.check("score.error", false, @errorName(err));
    if (o.steps.route) route.step(&env, r) catch |err| r.check("route.error", false, @errorName(err));
    if (o.steps.hc) hc.step(&env, r) catch |err| r.check("hc.error", false, @errorName(err));
}

/// What the receipt records of the run's device and kit.
const Ident = struct { arch: []const u8 = "", kit_sha: [64]u8 = @splat('-'), strict: bool = false };

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(gpa);
    const o = parseArgs(args) catch |err| {
        if (err != error.Help) std.debug.print("tf-dsv41-fusion-check: {s}\n", .{@errorName(err)});
        std.debug.print("{s}", .{usage});
        return if (err == error.Help) 0 else 2;
    };
    var r: Report = .{ .gpa = gpa };
    var id: Ident = .{};
    if (o.dry_run) {
        dryRun(gpa, io, &o, &r) catch |err| r.check("dry.error", false, @errorName(err));
    } else {
        gpuRun(gpa, io, &o, &r, &id) catch |err| r.check("error", false, @errorName(err));
    }
    const rc: u8 = if (r.failed > 0) 1 else if (r.timing_short > 0) 4 else 0;
    std.debug.print("fusion-check: {d} passed, {d} failed, {d} timings short; exit {d}\n", .{ r.passed, r.failed, r.timing_short, rc });
    if (o.receipt) |p| {
        const text = try std.fmt.allocPrint(gpa, "{{\n  \"receipt\": \"tf-dsv41-fusion-check\",\n  \"schema\": 1,\n  \"dry_run\": {},\n  \"arch\": \"{s}\",\n  \"kit\": \"{s}\",\n  \"kit_manifest_sha256\": \"{s}\",\n  \"strict\": {},\n  \"seed\": {d},\n  \"passed\": {d},\n  \"failed\": {d},\n  \"timing_short\": {d},\n  \"exit\": {d},\n  \"lines\": [\n{s}\n  ]\n}}\n", .{ o.dry_run, id.arch, o.kit, &id.kit_sha, id.strict, o.seed, r.passed, r.failed, r.timing_short, rc, r.json.items });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = text });
    }
    return rc;
}

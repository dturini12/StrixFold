//! tf-dsv41-vision-check: the HIP vision tower (vit_hip.zig on the kit's vision.* kernels) against engineLab's host
//! reference, byte for byte, step by step.
//!   tf-dsv41-vision-check --golden DIR [--kit DIR --weights DIR] [--case NAME]... [--out DIR] [--prep-only]
//! DIR (--golden) holds one folder a case, as tools/vision-golden writes them from engineLab's encode_reference:
//! grid.json, rgb.u8 (the decoded picture), patches.bf16, cos.f32 / sin.f32 (the RoPE tables), embed.bf16,
//! block00..31.bf16, norm.bf16, rows.bf16 and span.bf16. For each case:
//!  - prep (no GPU): picture.zig's grid and patches of rgb.u8 equal grid.json's and patches.bf16;
//!  - the tower (with --kit / --weights): encode of patches.bf16 with a probe on every step (the host RoPE tables,
//!    the patch embed, all 32 blocks, the final norm, the aligner rows) and the span, each equal to its file (the
//!    first difference named: step, row, column, both values); then two encodes without the probe, timed, whose span
//!    rows must equal the probed run's (determinism). --out writes each case's span rows there (rows.bf16, span.bf16).
//! Exit 0 only when every check of every case passes. Run with the GPU to itself (strixfold-container, pair stopped).
const std = @import("std");
const gpu = @import("gpu");
const dsv41 = @import("dsv41");
const vit = dsv41.vit_hip;
const picture = dsv41.picture;
const hip_load = dsv41.hip_load;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

const Case = struct {
    name: []const u8,
    dir: []const u8,
    width: usize,
    height: usize,
    grid: picture.Grid,
};

fn readCase(a: std.mem.Allocator, io: std.Io, root: []const u8, name: []const u8) !Case {
    const dir = try std.fs.path.join(a, &.{ root, name });
    const text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "grid.json" }), a, .limited(1 << 16));
    const v = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const o = v.object;
    const f = struct {
        fn get(obj: std.json.ObjectMap, key: []const u8) !usize {
            const x = obj.get(key) orelse return error.BadGrid;
            return @intCast(x.integer);
        }
    };
    return .{ .name = name, .dir = dir, .width = try f.get(o, "width"), .height = try f.get(o, "height"), .grid = .{
        .best_h = try f.get(o, "best_h"),
        .best_w = try f.get(o, "best_w"),
        .n_vit_h = try f.get(o, "n_vit_h"),
        .n_vit_w = try f.get(o, "n_vit_w"),
        .n_llm_h = try f.get(o, "n_llm_h"),
        .n_llm_w = try f.get(o, "n_llm_w"),
    } };
}

fn readFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, name }), a, .limited(1 << 31));
}

/// The first difference of two arrays of `elem`-byte values laid out `cols` a row ("" when equal).
fn firstDiff(a: std.mem.Allocator, got: []const u8, want: []const u8, elem: usize, cols: usize) ![]const u8 {
    if (got.len != want.len) return std.fmt.allocPrint(a, "{d} bytes, wanted {d}", .{ got.len, want.len });
    const i = std.mem.indexOfDiff(u8, got, want) orelse return "";
    const e = i / elem;
    var differ: usize = 0;
    var k: usize = 0;
    while (k < got.len) : (k += elem) {
        if (!std.mem.eql(u8, got[k..][0..elem], want[k..][0..elem])) differ += 1;
    }
    const gv: u32 = if (elem == 2) std.mem.readInt(u16, got[e * 2 ..][0..2], .little) else std.mem.readInt(u32, got[e * 4 ..][0..4], .little);
    const wv: u32 = if (elem == 2) std.mem.readInt(u16, want[e * 2 ..][0..2], .little) else std.mem.readInt(u32, want[e * 4 ..][0..4], .little);
    return std.fmt.allocPrint(a, "first difference at row {d} column {d}: 0x{x} vs 0x{x}; {d} of {d} values differ", .{ e / cols, e % cols, gv, wv, differ, got.len / elem });
}

const Probed = struct {
    a: std.mem.Allocator,
    io: std.Io,
    d: *const gpu.Driver,
    s: gpu.Stream,
    case: *const Case,
    host: std.ArrayList(u8) = .empty,
    checked: usize = 0,
    failed: usize = 0,

    fn probe(ctx: *anyopaque, what: []const u8, index: usize, ptr: u64, bytes: usize) anyerror!void {
        const p: *Probed = @ptrCast(@alignCast(ctx));
        var name_buf: [32]u8 = undefined;
        const name = if (std.mem.eql(u8, what, "block"))
            try std.fmt.bufPrint(&name_buf, "block{d:0>2}.bf16", .{index})
        else if (std.mem.eql(u8, what, "cos") or std.mem.eql(u8, what, "sin"))
            try std.fmt.bufPrint(&name_buf, "{s}.f32", .{what})
        else
            try std.fmt.bufPrint(&name_buf, "{s}.bf16", .{what});
        try p.s.synchronize();
        try p.host.resize(p.a, bytes);
        try gpu.copyD2H(p.d, p.host.items, ptr);
        const want = try readFile(p.a, p.io, p.case.dir, name);
        defer p.a.free(want);
        const f32s = std.mem.endsWith(u8, name, ".f32");
        const cols: usize = if (f32s) 32 else if (std.mem.eql(u8, what, "rows")) vit.model_dim else vit.dim;
        const diff = try firstDiff(p.a, p.host.items, want, if (f32s) 4 else 2, cols);
        p.checked += 1;
        if (diff.len != 0) {
            p.failed += 1;
            std.debug.print("  FAIL {s} {s}: {s}\n", .{ p.case.name, name, diff });
        }
    }
};

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var golden: ?[]const u8 = null;
    var kit_dir: ?[]const u8 = null;
    var weights: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var prep_only = false;
    var cases: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--prep-only")) {
            prep_only = true;
            continue;
        }
        if (i + 1 >= args.len) break;
        const val = args[i + 1];
        i += 1;
        if (std.mem.eql(u8, arg, "--golden")) golden = val else if (std.mem.eql(u8, arg, "--kit")) kit_dir = val else if (std.mem.eql(u8, arg, "--weights")) weights = val else if (std.mem.eql(u8, arg, "--out")) out = val else if (std.mem.eql(u8, arg, "--case")) try cases.append(a, val);
    }
    if (golden == null or (!prep_only and (kit_dir == null or weights == null))) {
        std.debug.print("usage: tf-dsv41-vision-check --golden DIR [--kit DIR --weights DIR] [--case NAME]... [--out DIR] [--prep-only]\n", .{});
        return 2;
    }
    if (cases.items.len == 0) {
        var dir = try std.Io.Dir.cwd().openDir(io, golden.?, .{ .iterate = true });
        defer dir.close(io);
        var it = dir.iterate();
        while (try it.next(io)) |e| {
            if (e.kind != .directory) continue;
            const has = std.Io.Dir.cwd().access(io, try std.fs.path.join(a, &.{ golden.?, e.name, "grid.json" }), .{});
            if (has) |_| try cases.append(a, try a.dupe(u8, e.name)) else |_| {}
        }
        std.mem.sort([]const u8, cases.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
    }

    var failed: usize = 0;
    // prep: picture.zig on the decoded RGB
    for (cases.items) |name| {
        const c = try readCase(a, io, golden.?, name);
        const rgb_bytes = try readFile(a, io, c.dir, "rgb.u8");
        const want = try readFile(a, io, c.dir, "patches.bf16");
        var pic = try picture.fromRgb(a, .{ .w = c.width, .h = c.height, .px = rgb_bytes }, .{});
        const grid_ok = std.meta.eql(pic.grid, c.grid);
        const diff = try firstDiff(a, std.mem.sliceAsBytes(pic.patches), want, 2, vit.patch_in);
        const ok = grid_ok and diff.len == 0;
        if (!ok) failed += 1;
        std.debug.print("{s} prep {s}: {d}x{d} -> vit {d}x{d}, llm {d}x{d}{s}{s}\n", .{ if (ok) "PASS" else "FAIL", name, c.width, c.height, pic.grid.n_vit_h, pic.grid.n_vit_w, pic.grid.n_llm_h, pic.grid.n_llm_w, if (grid_ok) "" else " (grid differs)", diff });
        pic.deinit(a);
    }
    if (prep_only) return if (failed == 0) 0 else 1;

    var d = try gpu.Driver.open();
    defer d.close();
    var ctx = try gpu.Context.init(&d, 0);
    defer ctx.deinit();
    const caps = try ctx.caps();
    const kit = try gpu.Kit.open(a, io, &d, kit_dir.?, caps.archName(), .{});
    defer kit.deinit();
    var kernels = (try hip_load.vision(kit)) orelse {
        std.debug.print("the kit at {s} has no vision.* kernels\n", .{kit_dir.?});
        return 1;
    };
    var s = try gpu.Stream.init(&d, true);
    defer s.deinit();
    const t_open = nowNs();
    var tower = try vit.Tower.open(a, io, &d, s, weights.?, &kernels);
    defer tower.close();
    const cells = vit.maxCells(1024);
    try tower.reserve(cells * 9, cells);
    std.debug.print("kit {s} (manifest {s}); tower {d} MiB on the device (weights {d} MiB), opened in {d} ms\n", .{ kit_dir.?, &kit.manifest_sha256, tower.bytes() >> 20, tower.weight_bytes >> 20, (nowNs() - t_open) / std.time.ns_per_ms });
    var span = try gpu.DeviceBuffer.alloc(&d, 1024 * vit.model_dim * 2);
    defer span.free();

    for (cases.items) |name| {
        const c = try readCase(a, io, golden.?, name);
        const patches_bytes = try readFile(a, io, c.dir, "patches.bf16");
        const patches = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(patches_bytes)));
        var p: Probed = .{ .a = a, .io = io, .d = &d, .s = s, .case = &c };
        const rows_at = try tower.encode(c.grid, patches, .{ .ctx = &p, .f = Probed.probe });
        try tower.spanRows(c.grid, rows_at, span.ptr, 0);
        try s.synchronize();
        const t = c.grid.spanTokens();
        const got_span = try a.alloc(u8, t * vit.model_dim * 2);
        try gpu.copyD2H(&d, got_span, span.ptr);
        const want_span = try readFile(a, io, c.dir, "span.bf16");
        const span_diff = try firstDiff(a, got_span, want_span, 2, vit.model_dim);
        if (span_diff.len != 0) std.debug.print("  FAIL {s} span.bf16: {s}\n", .{ name, span_diff });
        // twice more without the probe: the time of an encode, and the same bytes
        var ms: [2]u64 = undefined;
        var same = true;
        const again = try a.alloc(u8, got_span.len);
        for (&ms) |*m| {
            const t0 = nowNs();
            const at = try tower.encode(c.grid, patches, null);
            try tower.spanRows(c.grid, at, span.ptr, 0);
            try s.synchronize();
            m.* = (nowNs() - t0) / std.time.ns_per_ms;
            try gpu.copyD2H(&d, again, span.ptr);
            same = same and std.mem.eql(u8, again, got_span);
        }
        if (!same) std.debug.print("  FAIL {s}: a later encode's span differs from the first\n", .{name});
        const ok = p.failed == 0 and span_diff.len == 0 and same and p.checked == 37;
        if (!ok) failed += 1;
        std.debug.print("{s} tower {s}: {d} patches, {d} cells, {d} span tokens; {d} of {d} steps equal, span {s}, repeat {s}; encode {d} / {d} ms\n", .{ if (ok) "PASS" else "FAIL", name, c.grid.patches(), c.grid.n_llm_h * c.grid.n_llm_w, t, p.checked - p.failed, p.checked, if (span_diff.len == 0) "equal" else "DIFFERS", if (same) "equal" else "DIFFERS", ms[0], ms[1] });
        if (out) |o| {
            const dir = try std.fs.path.join(a, &.{ o, name });
            try std.Io.Dir.cwd().createDirPath(io, dir);
            const rows_bytes = c.grid.n_llm_h * c.grid.n_llm_w * vit.model_dim * 2;
            const rows_host = try a.alloc(u8, rows_bytes);
            const at = try tower.encode(c.grid, patches, null);
            try s.synchronize();
            try gpu.copyD2H(&d, rows_host, at);
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "rows.bf16" }), .data = rows_host });
            try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ dir, "span.bf16" }), .data = got_span });
        }
    }
    std.debug.print("{s}: {d} cases, {d} failed\n", .{ if (failed == 0) "PASS" else "FAIL", cases.items.len, failed });
    return if (failed == 0) 0 else 1;
}

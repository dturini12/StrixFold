//! The vision tower and aligner on HIP (gfx1151): vit.zig's Tower (open / close / reserve / encode / spanRows, the same
//! probes) over the kit's vision.* kernels (zig/kernels/hip/dsv41/vision.hip), engineLab's declared arithmetic, so each
//! step's bytes are its host reference's (encode_reference) and its device tower's: the patch embed and every linear
//! one FP32 fma chain over k ascending per output (bf16 after the bias), RMSNorm a wave-order sum, attention a fused
//! online softmax over 64-key tiles, the 2D RoPE tables built here on the host (double precision, libc's pow / cos /
//! sin, as the reference builds them). Not torch's kernels: the rows are not CUDA's vit.zig bytes (cosine ~0.999 to
//! torch's FP32 tower). Rank 0 runs it; the span rows go into the prompt in place of the image tokens' embeddings.
const std = @import("std");
const gpu = @import("gpu");
const core = @import("core");
const st = core.safetensors;
const picture = @import("picture.zig");

pub const dim = 1024;
pub const heads = 16;
pub const head_dim = dim / heads;
pub const layers = 32;
pub const inter = 2816;
pub const patch_in = 3 * 14 * 14;
pub const model_dim = 5120;
pub const merge = 3;
pub const unfold_dim = dim * merge * merge;
const eps: f32 = 1e-6;
const theta: f64 = 10000.0;

/// vision.hip's kernels (hip_load.vision binds them from the kit).
pub const Kernels = struct {
    linear: gpu.Function,
    rms_norm: gpu.Function,
    attention: gpu.Function,
    rope: gpu.Function,
    add: gpu.Function,
    silu_mul: gpu.Function,
    gelu: gpu.Function,
    unfold: gpu.Function,
    span: gpu.Function,
};

/// The parameter lists vision.hip's kernels take, field for field in the order the launchers below add them.
pub const params = struct {
    pub const Linear = extern struct { x: u64, w: u64, bias: u64, y: u64, m: i32, o: i32, k: i32 };
    pub const RmsNorm = extern struct { x: u64, w: u64, out: u64, rows: i32, d: i32, eps: f32 };
    pub const Attention = extern struct { qkv: u64, attn: u64, n: i32, d: i32 };
    pub const Rope = extern struct { qkv: u64, cos: u64, sin: u64, n: i32, d: i32, heads: i32, hd: i32 };
    pub const Add = extern struct { x: u64, y: u64, n: i64 };
    pub const SiluMul = extern struct { gu: u64, mid: u64, n: i32, i: i32 };
    pub const Gelu = extern struct { v: u64, n: i64 };
    pub const Unfold = extern struct { h: u64, unf: u64, vit_w: i32, vit_h: i32, llm_w: i32, llm_h: i32, d: i32, m: i32 };
    pub const Span = extern struct { rows: u64, start: u64, newline: u64, end: u64, out: u64, llm_w: i32, llm_h: i32, d: i32 };
};

const Block = struct { norm1: u64, wqkv: u64, bqkv: u64, wo: u64, bo: u64, norm2: u64, w1: u64, w2: u64 };

/// One step's output a gate checks (null: no probe): patch embed, the RoPE tables, each block, the final norm, the
/// aligner rows (vit.zig's names).
pub const Probe = struct {
    ctx: *anyopaque,
    f: *const fn (ctx: *anyopaque, what: []const u8, index: usize, ptr: u64, bytes: usize) anyerror!void,

    fn call(p: ?Probe, what: []const u8, index: usize, ptr: u64, bytes: usize) !void {
        if (p) |x| try x.f(x.ctx, what, index, ptr, bytes);
    }
};

/// The largest picture a span of `max_tokens` holds: its aligner cells (n_llm_h * n_llm_w with n_llm_h * (n_llm_w + 1)
/// + 2 <= max_tokens) and patches (merge^2 a cell at most); 1021 and 9189 for 1024 (engineLab's max_cells).
pub fn maxCells(max_tokens: usize) usize {
    var best: usize = 0;
    var h: usize = 1;
    while (h + 2 <= max_tokens) : (h += 1) {
        const w1 = (max_tokens - 2) / h;
        if (w1 < 2) break;
        best = @max(best, h * (w1 - 1));
    }
    return best;
}

/// The checkpoint's (or the sidecar's) shards that hold the vision tensors, by model.safetensors.index.json's weight_map.
const Sources = struct {
    gpa: std.mem.Allocator,
    files: std.ArrayList(st.File) = .empty,
    paths: std.ArrayList([]const u8) = .empty,
    arena: std.heap.ArenaAllocator,

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Sources {
        var sources: Sources = .{ .gpa = gpa, .arena = .init(std.heap.page_allocator) };
        errdefer sources.close(io);
        const a = sources.arena.allocator();
        const index_path = try std.fs.path.join(a, &.{ dir, "model.safetensors.index.json" });
        const text = std.Io.Dir.cwd().readFileAlloc(io, index_path, a, .limited(1 << 26)) catch |err| return if (err == error.FileNotFound) error.NoVisionWeights else err;
        const map = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
        const wm = map.object.get("weight_map") orelse return error.NoWeightMap;
        var it = wm.object.iterator();
        while (it.next()) |kv| {
            const k = kv.key_ptr.*;
            if (!(std.mem.startsWith(u8, k, "vision.") or std.mem.startsWith(u8, k, "aligner.") or std.mem.startsWith(u8, k, "image_"))) continue;
            const f = kv.value_ptr.*.string;
            for (sources.paths.items) |p| {
                if (std.mem.endsWith(u8, p, f)) break;
            } else {
                const path = try std.fs.path.join(gpa, &.{ dir, f });
                try sources.paths.append(gpa, path);
                try sources.files.append(gpa, try st.File.open(gpa, io, path));
            }
        }
        return sources;
    }

    fn get(s: *const Sources, name: []const u8) ?st.Tensor {
        for (s.files.items) |*f| if (f.get(name)) |t| return t;
        return null;
    }

    fn close(s: *Sources, io: std.Io) void {
        for (s.files.items) |*f| f.close(io);
        s.files.deinit(s.gpa);
        for (s.paths.items) |p| s.gpa.free(p);
        s.paths.deinit(s.gpa);
        s.arena.deinit();
    }
};

/// Every tensor the tower reads, with its shape ([out, in] for matrices) and its place in the weights buffer.
const names = [_][]const u8{ "attn.wqkv.weight", "attn.wqkv.bias", "attn.wo.weight", "attn.wo.bias", "norm1.weight", "norm2.weight", "mlp.w1.weight", "mlp.w2.weight" };
const block_shapes = [_][2]usize{ .{ 3 * dim, dim }, .{ 3 * dim, 1 }, .{ dim, dim }, .{ dim, 1 }, .{ dim, 1 }, .{ dim, 1 }, .{ 2 * inter, dim }, .{ dim, inter } };
const other_names = [_][]const u8{ "vision.patch_embed.proj.weight", "vision.patch_embed.proj.bias", "vision.norm.weight", "aligner.w1.weight", "aligner.w1.bias", "aligner.w2.weight", "aligner.w2.bias", "image_start", "image_newline", "image_end" };
const other_shapes = [_][2]usize{ .{ dim, patch_in }, .{ dim, 1 }, .{ dim, 1 }, .{ model_dim, unfold_dim }, .{ model_dim, 1 }, .{ model_dim, model_dim }, .{ model_dim, 1 }, .{ model_dim, 1 }, .{ model_dim, 1 }, .{ model_dim, 1 } };

pub const Tower = struct {
    d: *const gpu.Driver,
    stream: gpu.Stream,
    k: *const Kernels,
    weights: gpu.DeviceBuffer,
    weight_bytes: usize,
    blocks: [layers]Block,
    patch_w: u64,
    patch_b: u64,
    norm: u64,
    al_w1: u64,
    al_b1: u64,
    al_w2: u64,
    al_b2: u64,
    image_start: u64,
    image_newline: u64,
    image_end: u64,
    // the workspace for the largest picture reserved (reserve)
    work: ?gpu.DeviceBuffer = null,
    work_n: usize = 0,
    work_l: usize = 0,
    // the host's RoPE tables of the last picture, [n, 32] each
    cos: std.ArrayList(f32) = .empty,
    sin: std.ArrayList(f32) = .empty,
    gpa: std.mem.Allocator,

    /// The tower's weights (bf16, the checkpoint's own: every name, dtype and shape checked) on the device.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, d: *const gpu.Driver, stream: gpu.Stream, weights_dir: []const u8, k: *const Kernels) !Tower {
        var sh = try Sources.open(gpa, io, weights_dir);
        defer sh.close(io);
        var keys: std.ArrayList([]const u8) = .empty;
        var shapes: std.ArrayList([2]usize) = .empty;
        defer {
            for (keys.items) |key| gpa.free(key);
            keys.deinit(gpa);
            shapes.deinit(gpa);
        }
        for (0..layers) |i| for (names, block_shapes) |nm, s| {
            try keys.append(gpa, try std.fmt.allocPrint(gpa, "vision.blocks.{d}.{s}", .{ i, nm }));
            try shapes.append(gpa, s);
        };
        for (other_names, other_shapes) |key, s| {
            try keys.append(gpa, try gpa.dupe(u8, key));
            try shapes.append(gpa, s);
        }
        var total: usize = 0;
        for (keys.items, shapes.items) |key, s| {
            const t = sh.get(key) orelse return error.MissingVisionTensor;
            if (t.dtype != .bf16 or t.bytes.len != s[0] * s[1] * 2) return error.BadVisionTensor;
            total += std.mem.alignForward(usize, t.bytes.len, 256);
        }
        var buf = try gpu.DeviceBuffer.alloc(d, total);
        errdefer buf.free();
        const ptrs = try gpa.alloc(u64, keys.items.len);
        defer gpa.free(ptrs);
        // each tensor through anonymous memory first (a copy straight from the file mapping crawls), one at a time:
        // the host holds at most the largest tensor (aligner.w1, 94 MB), never the tower's 926 MB
        var tmp: std.ArrayList(u8) = .empty;
        defer tmp.deinit(std.heap.page_allocator);
        var at: usize = 0;
        for (keys.items, ptrs) |key, *p| {
            const t = sh.get(key).?;
            try tmp.resize(std.heap.page_allocator, t.bytes.len);
            @memcpy(tmp.items, t.bytes);
            try buf.upload(at, tmp.items);
            p.* = buf.ptr + at;
            at += std.mem.alignForward(usize, t.bytes.len, 256);
        }
        var tw: Tower = .{ .d = d, .stream = stream, .k = k, .weights = buf, .weight_bytes = total, .blocks = undefined, .patch_w = 0, .patch_b = 0, .norm = 0, .al_w1 = 0, .al_b1 = 0, .al_w2 = 0, .al_b2 = 0, .image_start = 0, .image_newline = 0, .image_end = 0, .gpa = gpa };
        for (0..layers) |i| {
            const b = ptrs[i * names.len ..];
            tw.blocks[i] = .{ .wqkv = b[0], .bqkv = b[1], .wo = b[2], .bo = b[3], .norm1 = b[4], .norm2 = b[5], .w1 = b[6], .w2 = b[7] };
        }
        const r = ptrs[layers * names.len ..];
        tw.patch_w = r[0];
        tw.patch_b = r[1];
        tw.norm = r[2];
        tw.al_w1 = r[3];
        tw.al_b1 = r[4];
        tw.al_w2 = r[5];
        tw.al_b2 = r[6];
        tw.image_start = r[7];
        tw.image_newline = r[8];
        tw.image_end = r[9];
        return tw;
    }

    pub fn close(tw: *Tower) void {
        if (tw.work) |*w| w.free();
        tw.weights.free();
        tw.cos.deinit(tw.gpa);
        tw.sin.deinit(tw.gpa);
    }

    /// The workspace's parts for n patches and l cells (engineLab's buffers, each 256-byte aligned).
    const Work = struct {
        patches: u64, // bf16 [n, 588]
        x: u64, // bf16 [n, 1024]: the residual stream
        h: u64, // bf16 [n, 1024]: a norm's output
        qkv: u64, // bf16 [n, 3072]
        attn: u64, // bf16 [n, 1024]
        tmp: u64, // bf16 [n, 1024]: wo's and w2's output
        gu: u64, // bf16 [n, 5632]
        mid: u64, // bf16 [n, 2816]
        cos: u64,
        sin: u64, // fp32 [n, 32]
        unf: u64, // bf16 [l, 9216]
        a1: u64, // bf16 [l, 5120]
        rows: u64, // bf16 [l, 5120]: the aligner's output
        bytes: usize,

        fn plan(base: u64, n: usize, l: usize) Work {
            var at: usize = 0;
            const sizes = [_]usize{ n * patch_in * 2, n * dim * 2, n * dim * 2, n * 3 * dim * 2, n * dim * 2, n * dim * 2, n * 2 * inter * 2, n * inter * 2, n * 32 * 4, n * 32 * 4, l * unfold_dim * 2, l * model_dim * 2, l * model_dim * 2 };
            var p: [sizes.len]u64 = undefined;
            for (sizes, 0..) |s, i| {
                p[i] = base + at;
                at += std.mem.alignForward(usize, s, 256);
            }
            return .{ .patches = p[0], .x = p[1], .h = p[2], .qkv = p[3], .attn = p[4], .tmp = p[5], .gu = p[6], .mid = p[7], .cos = p[8], .sin = p[9], .unf = p[10], .a1 = p[11], .rows = p[12], .bytes = at };
        }
    };

    /// The workspace for the largest picture up front (n patches, l aligner rows), so serving allocates nothing.
    pub fn reserve(tw: *Tower, n: usize, l: usize) !void {
        if (tw.work != null and n <= tw.work_n and l <= tw.work_l) return;
        const need = Work.plan(0, n, l).bytes;
        if (tw.work) |*w| w.free();
        tw.work = null;
        tw.work = try gpu.DeviceBuffer.alloc(tw.d, need);
        tw.work_n = n;
        tw.work_l = l;
        try tw.cos.ensureTotalCapacity(tw.gpa, n * 32);
        try tw.sin.ensureTotalCapacity(tw.gpa, n * 32);
    }

    /// The device bytes the tower holds: its weights and its workspace.
    pub fn bytes(tw: *const Tower) usize {
        return tw.weight_bytes + if (tw.work) |w| w.len else 0;
    }

    /// Tower.encode of one picture (its patches on the host, bf16): the aligner's rows bf16 [n_llm_h * n_llm_w, 5120],
    /// left in the workspace (the returned address) until the next call. The picture must fit the reserved workspace.
    pub fn encode(tw: *Tower, g: picture.Grid, patches: []const u16, probe: ?Probe) !u64 {
        const n = g.patches();
        const l = g.n_llm_h * g.n_llm_w;
        if (tw.work == null or n > tw.work_n or l > tw.work_l) return error.PictureTooLarge;
        if (n == 0 or patches.len != n * patch_in or g.n_vit_h > merge * g.n_llm_h or g.n_vit_w > merge * g.n_llm_w) return error.BadImagePayload;
        const w = Work.plan(tw.work.?.ptr, n, l);
        const s = tw.stream;
        // the previous picture's work is done before its inputs are overwritten (blocking copies on the null stream)
        try s.synchronize();
        try ropeTables(g, &tw.cos, &tw.sin);
        try gpu.copyH2D(tw.d, w.patches, std.mem.sliceAsBytes(patches));
        try gpu.copyH2D(tw.d, w.cos, std.mem.sliceAsBytes(tw.cos.items));
        try gpu.copyH2D(tw.d, w.sin, std.mem.sliceAsBytes(tw.sin.items));
        try Probe.call(probe, "cos", 0, w.cos, n * 32 * 4);
        try Probe.call(probe, "sin", 0, w.sin, n * 32 * 4);
        try linear(tw.k, s, w.patches, tw.patch_w, tw.patch_b, w.x, n, dim, patch_in);
        try Probe.call(probe, "embed", 0, w.x, n * dim * 2);
        for (tw.blocks, 0..) |b, i| {
            try tw.block(b, w, n);
            try Probe.call(probe, "block", i, w.x, n * dim * 2);
        }
        try rmsNorm(tw.k, s, w.x, tw.norm, w.h, n);
        try Probe.call(probe, "norm", 0, w.h, n * dim * 2);
        // the aligner: 3x3 cells unfolded (zeros past the grid), w1, gelu, w2
        try unfold(tw.k, s, w.h, w.unf, g);
        try linear(tw.k, s, w.unf, tw.al_w1, tw.al_b1, w.a1, l, model_dim, unfold_dim);
        try gelu(tw.k, s, w.a1, l * model_dim);
        try linear(tw.k, s, w.a1, tw.al_w2, tw.al_b2, w.rows, l, model_dim, model_dim);
        try Probe.call(probe, "rows", 0, w.rows, l * model_dim * 2);
        return w.rows;
    }

    fn block(tw: *Tower, b: Block, w: Work, n: usize) !void {
        const s = tw.stream;
        const k = tw.k;
        try rmsNorm(k, s, w.x, b.norm1, w.h, n);
        try linear(k, s, w.h, b.wqkv, b.bqkv, w.qkv, n, 3 * dim, dim);
        try rope(k, s, w.qkv, w.cos, w.sin, n);
        try attention(k, s, w.qkv, w.attn, n);
        try linear(k, s, w.attn, b.wo, b.bo, w.tmp, n, dim, dim);
        try add(k, s, w.x, w.tmp, n * dim);
        try rmsNorm(k, s, w.x, b.norm2, w.h, n);
        try linear(k, s, w.h, b.w1, 0, w.gu, n, 2 * inter, dim);
        try siluMul(k, s, w.gu, w.mid, n);
        try linear(k, s, w.mid, b.w2, 0, w.tmp, n, dim, inter);
        try add(k, s, w.x, w.tmp, n * dim);
    }

    /// Tower.span_rows: the image span's rows bf16 [span tokens, 5120] into `out` (device), the delimiters' learned
    /// embeddings around the aligner's rows (encode's output at `rows`). `scratch` is unused on HIP.
    pub fn spanRows(tw: *Tower, g: picture.Grid, rows: u64, out: u64, scratch: u64) !void {
        _ = scratch;
        try span(tw.k, tw.stream, rows, tw.image_start, tw.image_newline, tw.image_end, out, g.n_llm_w, g.n_llm_h);
    }
};

/// y [m, o] bf16 = bf16(x [m, kk] . w [o, kk]^T (+ bias)), bias 0: none (a 128 x 128 output tile a block).
pub fn linear(k: *const Kernels, s: gpu.Stream, x: u64, w: u64, bias: u64, y: u64, m: usize, o: usize, kk: usize) !void {
    var a: gpu.Args = .{};
    a.add(x);
    a.add(w);
    a.add(bias);
    a.add(y);
    a.add(@as(i32, @intCast(m)));
    a.add(@as(i32, @intCast(o)));
    a.add(@as(i32, @intCast(kk)));
    try gpu.launch.launch(k.linear, .{ .grid = .{ .x = @intCast((o + 127) / 128), .y = @intCast((m + 127) / 128) }, .block = .{ .x = 256 } }, s, &a);
}

/// out [rows, 1024] = RMSNorm(x) * w (eps 1e-6), one wave a row.
pub fn rmsNorm(k: *const Kernels, s: gpu.Stream, x: u64, w: u64, out: u64, rows: usize) !void {
    var a: gpu.Args = .{};
    a.add(x);
    a.add(w);
    a.add(out);
    a.add(@as(i32, @intCast(rows)));
    a.add(@as(i32, dim));
    a.add(eps);
    try gpu.launch.launch(k.rms_norm, .{ .grid = .{ .x = @intCast((rows + 7) / 8) }, .block = .{ .x = 256 } }, s, &a);
}

fn blocksFor(count: usize, per: usize) u32 {
    return @intCast(@max(1, (count + per - 1) / per));
}

/// q and k of qkv [n, 3072] rotated in place by the picture's tables (cos, sin fp32 [n, 32]).
pub fn rope(k: *const Kernels, s: gpu.Stream, qkv: u64, cos_t: u64, sin_t: u64, n: usize) !void {
    var a: gpu.Args = .{};
    a.add(qkv);
    a.add(cos_t);
    a.add(sin_t);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, dim));
    a.add(@as(i32, heads));
    a.add(@as(i32, head_dim));
    try gpu.launch.launch(k.rope, .{ .grid = .{ .x = blocksFor(n * 2 * heads * (head_dim / 2), 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// attn [n, 1024] = softmax(q k^T / 8) v per head, bidirectional (a block per 64 rows and head).
pub fn attention(k: *const Kernels, s: gpu.Stream, qkv: u64, attn: u64, n: usize) !void {
    var a: gpu.Args = .{};
    a.add(qkv);
    a.add(attn);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, dim));
    try gpu.launch.launch(k.attention, .{ .grid = .{ .x = blocksFor(n, 64), .y = heads }, .block = .{ .x = 256 } }, s, &a);
}

/// mid [n, 2816] = silu(gate) * up of gu [n, 5632].
pub fn siluMul(k: *const Kernels, s: gpu.Stream, gu: u64, mid: u64, n: usize) !void {
    var a: gpu.Args = .{};
    a.add(gu);
    a.add(mid);
    a.add(@as(i32, @intCast(n)));
    a.add(@as(i32, inter));
    try gpu.launch.launch(k.silu_mul, .{ .grid = .{ .x = blocksFor(n * inter, 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn gelu(k: *const Kernels, s: gpu.Stream, v: u64, count: usize) !void {
    var a: gpu.Args = .{};
    a.add(v);
    a.add(@as(i64, @intCast(count)));
    try gpu.launch.launch(k.gelu, .{ .grid = .{ .x = blocksFor(count, 256) }, .block = .{ .x = 256 } }, s, &a);
}

/// unf [cells, 9216] = the 3x3 unfold of h [patches, 1024] over g's grids.
pub fn unfold(k: *const Kernels, s: gpu.Stream, h: u64, unf: u64, g: picture.Grid) !void {
    var a: gpu.Args = .{};
    a.add(h);
    a.add(unf);
    a.add(@as(i32, @intCast(g.n_vit_w)));
    a.add(@as(i32, @intCast(g.n_vit_h)));
    a.add(@as(i32, @intCast(g.n_llm_w)));
    a.add(@as(i32, @intCast(g.n_llm_h)));
    a.add(@as(i32, dim));
    a.add(@as(i32, merge));
    try gpu.launch.launch(k.unfold, .{ .grid = .{ .x = blocksFor(g.n_llm_h * g.n_llm_w * unfold_dim, 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn add(k: *const Kernels, s: gpu.Stream, x: u64, y: u64, count: usize) !void {
    var a: gpu.Args = .{};
    a.add(x);
    a.add(y);
    a.add(@as(i64, @intCast(count)));
    try gpu.launch.launch(k.add, .{ .grid = .{ .x = @intCast((count + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

pub fn span(k: *const Kernels, s: gpu.Stream, rows: u64, start: u64, newline: u64, end: u64, out: u64, llm_w: usize, llm_h: usize) !void {
    const t = llm_h * (llm_w + 1) + 2;
    var a: gpu.Args = .{};
    a.add(rows);
    a.add(start);
    a.add(newline);
    a.add(end);
    a.add(out);
    a.add(@as(i32, @intCast(llm_w)));
    a.add(@as(i32, @intCast(llm_h)));
    a.add(@as(i32, model_dim));
    try gpu.launch.launch(k.span, .{ .grid = .{ .x = @intCast((t * model_dim + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
}

extern "c" fn pow(x: f64, y: f64) f64;
extern "c" fn cos(x: f64) f64;
extern "c" fn sin(x: f64) f64;

/// engineLab's rope_tables: for patch r (row r / n_vit_w, column r % n_vit_w) and pair j of 32, angle = f32(row or
/// column) * inv[j % 16] in fp32 (rows for j < 16, columns after), inv[j] = f32(1 / pow(theta, 2 j / 32)) in double;
/// cos and sin of the angle in double (libc, as the reference calls std::cos / std::sin), rounded to fp32.
pub fn ropeTables(g: picture.Grid, cos_out: *std.ArrayList(f32), sin_out: *std.ArrayList(f32)) !void {
    const half = head_dim / 2;
    const quarter = half / 2;
    const n = g.patches();
    var inv: [quarter]f32 = undefined;
    for (&inv, 0..) |*v, j| v.* = @floatCast(1.0 / pow(theta, @as(f64, @floatFromInt(2 * j)) / @as(f64, @floatFromInt(half))));
    cos_out.clearRetainingCapacity();
    sin_out.clearRetainingCapacity();
    if (cos_out.capacity < n * half or sin_out.capacity < n * half) return error.PictureTooLarge;
    for (0..n) |r| {
        const row: f32 = @floatFromInt(r / g.n_vit_w);
        const col: f32 = @floatFromInt(r % g.n_vit_w);
        for (0..half) |j| {
            const angle: f32 = (if (j < quarter) row else col) * inv[j % quarter];
            cos_out.appendAssumeCapacity(@floatCast(cos(@floatCast(angle))));
            sin_out.appendAssumeCapacity(@floatCast(sin(@floatCast(angle))));
        }
    }
}

test "the largest picture a 1024-token span holds is engineLab's 1021 cells" {
    try std.testing.expectEqual(@as(usize, 1021), maxCells(1024));
}

test "rope tables: rows for the first 16 pairs, columns after, angle 0 at the origin" {
    const a = std.testing.allocator;
    var c: std.ArrayList(f32) = .empty;
    defer c.deinit(a);
    var s: std.ArrayList(f32) = .empty;
    defer s.deinit(a);
    try c.ensureTotalCapacity(a, 6 * 32);
    try s.ensureTotalCapacity(a, 6 * 32);
    const g: picture.Grid = .{ .best_h = 28, .best_w = 42, .n_vit_h = 2, .n_vit_w = 3, .n_llm_h = 1, .n_llm_w = 1 };
    try ropeTables(g, &c, &s);
    try std.testing.expectEqual(@as(usize, 6 * 32), c.items.len);
    for (0..32) |j| try std.testing.expectEqual(@as(f32, 1), c.items[j]); // patch 0: row 0, column 0
    // patch 4: row 1, column 1; pair 0 has inv 1 -> cos(1)
    try std.testing.expectEqual(@as(f32, @floatCast(@cos(@as(f64, 1)))), c.items[4 * 32]);
    try std.testing.expectEqual(c.items[4 * 32], c.items[4 * 32 + 16]);
    // patch 2: row 0, column 2: rows' pairs are 1, columns' are cos(2 inv)
    try std.testing.expectEqual(@as(f32, 1), c.items[2 * 32 + 3]);
    try std.testing.expect(c.items[2 * 32 + 16] != 1);
}

//! The family's kernel sets from the AMD kit (HIP builds only: model.zig's HIP branch; a CUDA build never references
//! this file). The helpers are glue.* (w-arch, glue.hip) and aux.* (w-tri, aux.hip), the EXL3 kernels exl3.* (w-exl3)
//! in the kit's registry; each is bound with the parameter list its launcher packs (`params`: ops.zig, exact.zig,
//! exl3_prefill.zig, exl3_linear.zig, exl3_experts.zig), so a kernel whose compiled arguments differ from that packing
//! refuses the model's open (AbiMismatch naming it), and in strict mode every launch is checked again against its
//! metadata (zig/src/gpu/hip/kit.zig). A host test runs every launcher and holds its packing to these lists. The kit
//! owns every module: the sets' unload leaves them.
//!
//! The EXL3 kernels keep the CUDA kernels' parameter lists (the family's launchers are unchanged). Specializations are
//! kit entries of one id told apart by their spec: exl3.unpack {k2, cb}, exl3.glinear {k2, cb, warps},
//! exl3.experts.grouped_mma and grouped_rows {cb, lo, hi}, exl3.experts.grouped_cp {cb, lo, hi, epi}; cb is
//! exl3_linear.codebook_mul1 and lo/hi a K2 range as exl3_experts.range() reads it.
const std = @import("std");
const gpu = @import("gpu");
const ops_mod = @import("ops.zig");
const exact_mod = @import("exact.zig");
const exl3_prefill = @import("exl3_prefill.zig");
const exl3_linear = @import("exl3_linear.zig");
const exl3_experts = @import("exl3_experts.zig");
const weights = @import("weights.zig");
const tri_aux = @import("tri_aux.zig");
const dense_mod = @import("dense.zig");
const vit_hip = @import("vit_hip.zig");

/// The parameter lists, field for field in the order ops.zig and exact.zig add them (u64: a device address or a u64).
pub const params = struct {
    pub const Convert = extern struct { in: u64, out: u64, count: u64 };
    pub const Add2Bf16 = extern struct { g0: u64, g1: u64, out: u64, count: u64 };
    pub const TopkI64 = extern struct { keys: u64, ks: i64, n: i32, k: i32, top: u64 };
    pub const TopkIndices = extern struct { score: u64, ss: i64, n: i32, k: i32, vis: u64, out: u64 };
    pub const ScatterRows = extern struct { src: u64, src_ld: i64, idx: u64, dst: u64, dst_ld: i64, bytes: i64, rows: i32 };
    pub const CandFast = extern struct { score: u64, ss: i64, nb: i32, bsize: i32, vis: u64, out: u64, os: i64 };
    pub const IotaVis = extern struct { vis: u64, out: u64, nb: i32 };
    pub const BlockMax = extern struct { score: u64, ss: i64, width: i32, bsize: i32, vis: u64, out: u64, os: i64, nb: i32 };
    pub const PoolPick = extern struct { bmax: u64, bs: i64, nb: i32, idx: u64, k: i32, mask: u64, ms: i64, cblk: u64 };
    pub const ApplyPool = extern struct { score: u64, ss: i64, width: i32, bsize: i32, mask: u64, ms: i64 };
    pub const StridedCopy = extern struct { src: u64, dst: u64, outer: u64, middle: u64, inner: u64, src_outer: u64, src_middle: u64, dst_outer: u64, dst_middle: u64 };
    pub const GatherRows = extern struct { src: u64, dst: u64, idx: u64, rows: u64, src_rows: u64, bytes: u64, invalid: u64 };
    pub const CopyRows16 = extern struct { src: u64, src_ld: u64, dst: u64, dst_ld: u64, n16: u64 };
    pub const ArgmaxRows = extern struct { logits: u64, ld: u64, n: i32, out: u64 };
    pub const Clock = extern struct { buf: u64, tag: u32, reset: u32, cap: u32, ns_q32: u64 };
    pub const L2Prefetch = extern struct { pf: ops_mod.Prefetch, delay_ns: i64, wave_ns: i64 };
    pub const RmsNorm = extern struct { x: u64, ldx: i64, w: u64, out: u64, ldo: i64, rows: i32, d: i32, factor: f32, eps: f32 };
    pub const Compress2 = extern struct { kv: u64, score: u64, out: u64, g: i32, c: i32 };
    pub const Rope = extern struct { x: u64, ldx: i64, off: i32, cos: u64, sin: u64, pos: u64, n: i32, half: i32, inverse: i32 };
    pub const Bf16Scale = extern struct { in: u64, out: u64, count: u64, scale: f32 };
    pub const HcMean4 = extern struct { h: u64, out: u64, os: i64, rows: i32, d: i32 };
    // w-tri's kernels for Triton launches (tri_aux.zig): one by-value struct each
    pub const AuxAttn = extern struct { a: tri_aux.AttnArgs };
    pub const AuxIndex = extern struct { a: tri_aux.IndexArgs };
    pub const AuxHcMix = extern struct { a: tri_aux.HcMixArgs };
    pub const AuxGate = extern struct { a: tri_aux.GateArgs };
    pub const AuxRopeHeads = extern struct { a: tri_aux.RopeHeadsArgs };

    // EXL3: exl3_prefill.zig
    pub const Exl3RotIn = extern struct { x: u64, x_dtype: i32, ldx: i64, suh: u64, xh: u64, k: i32 };
    pub const Exl3Unpack = extern struct { words: u64, w: u64, n: i32, stride_k: i64, stride_nb: i64 };
    pub const Exl3Gemm = extern struct { xh: u64, w: u64, stride_k: i64, stride_nb: i64, svh: u64, out: u64, o_stride: i64, m: i32, k: i32, n: i32, out_f32: i32 };
    // exl3_linear.zig: one by-value struct each (RArgs 392 B, GArgs 1,424 B)
    pub const Exl3RotMany = extern struct { a: exl3_linear.RArgs };
    pub const Exl3GLinear = extern struct { a: exl3_linear.GArgs };
    // exl3_experts.zig (a/b: gate and up, or down twice)
    pub const GroupCount = extern struct { pick: u64, counts: u64, p: i32 };
    // dense.zig (p-par, dense.hip)
    pub const DenseGemv = extern struct { w: u64, x: u64, y: u64, k: i32, n: i32, rows: i32, ldx: i64, ldy: i64, accumulate: i32 };
    pub const DenseSwiglu = extern struct { g: u64, u: u64, ld: i64, h: u64, ldh: i64, rows: i32, width: i32, limit: f32 };
    pub const GroupPlace = extern struct { pick: u64, counts: u64, ids: u64, count: u64, members: u64, p: i32, slots: i32, e: i32, maxm: i32 };
    pub const WorkList = extern struct { counts: u64, ids: u64, count: u64, maxu: i32, work_gu: u64, rows_gu: i32, n_gu: i32, work_d: u64, rows_d: i32, n_d: i32, skip_e: i32 };
    pub const ExpertsRotIn = extern struct { x: u64, x_stride: i32, pick: u64, suh_g: u64, suh_u: u64, xg: u64, xu: u64, d: i32, slots: i32, e: i32 };
    pub const GroupedMma = extern struct { xa: u64, xb: u64, wa: u64, wb: u64, ka: u64, kb: u64, ids: u64, count: u64, members: u64, z: u64, k: i32, n: i32, p: i32, splits: i32, warps: i32, maxm: i32, slots: i32, n_work: i32, work: u64 };
    pub const GateupEpilogue = extern struct { z: u64, pick: u64, svh_g: u64, svh_u: u64, suh_d: u64, xd: u64, p: i32, i: i32, splits: i32, e: i32, limit: f32, act: i32 };
    pub const GroupedRows = extern struct { xa: u64, xb: u64, wa: u64, wb: u64, ka: u64, kb: u64, ids: u64, count: u64, members: u64, z: u64, k: i32, n: i32, p: i32, splits: i32, maxm: i32, slots: i32, n_work: i32, work: u64 };
    pub const DownCombine = extern struct { z: u64, pick: u64, svh_d: u64, y: u64, wts: u64, add: u64, out: u64, p: i32, d: i32, splits: i32, e: i32, slots: i32, store_y: i32 };
    pub const DecodePrep = extern struct { x: u64, x_stride: i32, pick: u64, suh_g: u64, suh_u: u64, xg: u64, xu: u64, d: i32, slots: i32, e: i32, r: i32, ids: u64, count: u64, members: u64, maxm: i32, wts: u64, y: u64, add: u64, out: u64, d_out: i32, store_y: i32, epoch: u64 };
    pub const GroupedCp = extern struct { xa: u64, xb: u64, wa: u64, wb: u64, ka: u64, kb: u64, ids: u64, count: u64, members: u64, z: u64, k: i32, n: i32, p: i32, splits: i32, maxm: i32, slots: i32, epi: exl3_experts.DecodeEpi };
    // exl3_experts.zig pfPlan: every exl3.pf_experts kernel takes the plan's one by-value struct (232 B)
    pub const PfExperts = extern struct { a: exl3_experts.PfArgs };
};

/// The EXL3 by-value arguments have one definition for hipcc and Zig: abi.h's structs. The family's Zig structs must
/// lay every field where abi.h does (and be as large), or this file does not compile.
fn sameLayout(comptime Z: type, comptime C: type) void {
    if (@sizeOf(Z) != @sizeOf(C)) @compileError(@typeName(Z) ++ " and abi.h's " ++ @typeName(C) ++ " differ in size");
    for (@typeInfo(Z).@"struct".field_names) |name| {
        if (@offsetOf(Z, name) != @offsetOf(C, name)) @compileError(@typeName(Z) ++ "." ++ name ++ " is not where abi.h puts it");
    }
}

comptime {
    const C = gpu.kernel_abi;
    sameLayout(exl3_linear.GLayer, C.tf_ds_exl3_glayer);
    sameLayout(exl3_linear.GArgs, C.tf_ds_exl3_gargs);
    sameLayout(exl3_linear.RLayer, C.tf_ds_exl3_rlayer);
    sameLayout(exl3_linear.RArgs, C.tf_ds_exl3_rargs);
    sameLayout(exl3_experts.DecodeEpi, C.tf_ds_exl3_decode_epi);
    sameLayout(exl3_experts.PfArgs, C.tf_ds_exl3_pf_args);
    sameLayout(ops_mod.Prefetch, C.tf_ds_prefetch);
    sameLayout(tri_aux.AttnArgs, C.tf_ds_aux_attn_args);
    sameLayout(tri_aux.IndexArgs, C.tf_ds_aux_index_args);
    sameLayout(tri_aux.HcMixArgs, C.tf_ds_aux_hc_mix_args);
    if (C.TF_DS_EXL3_GMAX != exl3_linear.gmax) @compileError("abi.h TF_DS_EXL3_GMAX is not exl3_linear.gmax");
}

fn bind(kit: *const gpu.Kit, comptime id: []const u8, comptime P: type) !gpu.Function {
    return (try gpu.Kernel(id, P).bind(kit, &.{})).f;
}

/// A kernel the kit may leave out (its feature is off): null when absent, AbiMismatch when present but different.
fn optional(kit: *const gpu.Kit, comptime id: []const u8, comptime P: type) !?gpu.Function {
    if (kit.find(id, &.{}) == null) return null;
    return try bind(kit, id, P);
}

/// dense.hip's kernels (the dense projections of a checkpoint that has them): null when the kit has neither, an
/// error when it has only one of them.
pub fn dense(kit: *const gpu.Kit) !?dense_mod.Kernels {
    const gemv = try optional(kit, "dense.gemv", params.DenseGemv);
    const swiglu = try optional(kit, "dense.swiglu", params.DenseSwiglu);
    if (gemv == null and swiglu == null) return null;
    return .{ .gemv = gemv orelse return error.MissingKernel, .swiglu = swiglu orelse return error.MissingKernel };
}

/// vision.hip's kernels (the vision tower, for MODALITIES=text_image): null when the kit has none of them, an error
/// when it has only some.
pub fn vision(kit: *const gpu.Kit) !?vit_hip.Kernels {
    const vp = vit_hip.params;
    const k: vit_hip.Kernels = .{
        .linear = try optional(kit, "vision.linear", vp.Linear) orelse return if (kit.find("vision.rms_norm", &.{}) == null) null else error.MissingKernel,
        .rms_norm = try optional(kit, "vision.rms_norm", vp.RmsNorm) orelse return error.MissingKernel,
        .attention = try optional(kit, "vision.attention", vp.Attention) orelse return error.MissingKernel,
        .rope = try optional(kit, "vision.rope", vp.Rope) orelse return error.MissingKernel,
        .add = try optional(kit, "vision.add", vp.Add) orelse return error.MissingKernel,
        .silu_mul = try optional(kit, "vision.silu_mul", vp.SiluMul) orelse return error.MissingKernel,
        .gelu = try optional(kit, "vision.gelu", vp.Gelu) orelse return error.MissingKernel,
        .unfold = try optional(kit, "vision.unfold", vp.Unfold) orelse return error.MissingKernel,
        .span = try optional(kit, "vision.span", vp.Span) orelse return error.MissingKernel,
    };
    return k;
}

/// Nanoseconds a tick of a `hz` clock in 32.32 fixed point (glue.clock's ns_q32), rounded to nearest.
pub fn nsPerTickQ32(hz: u64) u64 {
    return @intCast(((@as(u128, 1_000_000_000) << 32) + hz / 2) / hz);
}

/// The helpers; `wall_clock_hz` is the device's constant clock (caps), which glue.clock turns into nanoseconds.
pub fn ops(kit: *const gpu.Kit, wall_clock_hz: u64) !ops_mod.Ops {
    const p = params;
    return .{
        .pointwise = undefined,
        .movement = undefined,
        .family = undefined,
        .kit_owned = true,
        .to_f32 = try bind(kit, "glue.bf16_to_f32", p.Convert),
        .f16_to_f32 = try bind(kit, "glue.f16_to_f32", p.Convert),
        .add2_bf16 = try bind(kit, "glue.add2_bf16", p.Add2Bf16),
        .topk_i64 = try bind(kit, "aux.topk_i64", p.TopkI64),
        .topk_indices = try bind(kit, "aux.topk_indices", p.TopkIndices),
        .scatter = try bind(kit, "glue.scatter_rows", p.ScatterRows),
        .cand_fast = try bind(kit, "aux.cand_fast", p.CandFast),
        .iota_vis = try bind(kit, "aux.iota_vis", p.IotaVis),
        .l2_prefetch = try optional(kit, "exl3.l2_prefetch", p.L2Prefetch), // the MALL prefetch (exl3.hip)
        .block_max = try bind(kit, "aux.block_max", p.BlockMax),
        .pool_pick = try bind(kit, "aux.pool_pick", p.PoolPick),
        .apply_pool = try bind(kit, "aux.apply_pool", p.ApplyPool),
        .to_bf16 = null, // loaded by the CUDA set, never launched by the family
        .strided = try bind(kit, "glue.strided_copy", p.StridedCopy),
        .gather = try bind(kit, "glue.gather_rows", p.GatherRows),
        .copy16 = try bind(kit, "glue.copy_rows16", p.CopyRows16),
        .argmax_rows = try bind(kit, "glue.argmax_rows", p.ArgmaxRows),
        .gpu_clock = try optional(kit, "glue.clock", p.Clock), // profiles only
        .clock_ns_q32 = nsPerTickQ32(wall_clock_hz),
    };
}

pub fn exact(kit: *const gpu.Kit) !exact_mod.Exact {
    const p = params;
    return .{
        .module = undefined,
        .kit_owned = true,
        .k_rms_norm = try bind(kit, "aux.rms_norm", p.RmsNorm),
        .k_compress2 = try bind(kit, "aux.compress2", p.Compress2),
        .k_rope = try bind(kit, "aux.rope", p.Rope),
        .k_bf16_scale = try bind(kit, "aux.bf16_scale", p.Bf16Scale),
        .k_hc_mean4 = try bind(kit, "aux.hc_mean4", p.HcMean4),
    };
}

/// w-tri's kernels for the launches `which` names (TF_DS_AUX); the others stay null (Triton, or hipBLAS for the gates).
/// Each is the kit entry of its compiled shape (registry.json's spec: tri_aux.zig's constants).
pub fn aux(kit: *const gpu.Kit, which: tri_aux.Which) !tri_aux.Aux {
    const p = params;
    return .{
        .attn = if (which.attn) (try gpu.Kernel("aux.sparse_attn", p.AuxAttn).bind(kit, &aux_specs.attn)).f else null,
        .index = if (which.index) (try gpu.Kernel("aux.index_rows", p.AuxIndex).bind(kit, &aux_specs.index)).f else null,
        .hc_mix = if (which.hc) (try gpu.Kernel("aux.hc_mix_pf", p.AuxHcMix).bind(kit, &aux_specs.hc)).f else null,
        .gate = if (which.gate) (try gpu.Kernel("aux.gate_logits", p.AuxGate).bind(kit, &aux_specs.gate)).f else null,
        .gate32 = if (which.idxw) (try gpu.Kernel("aux.gate_logits", p.AuxGate).bind(kit, &aux_specs.gate32)).f else null,
        .rope = if (which.rope) (try gpu.Kernel("aux.rope_heads", p.AuxRopeHeads).bind(kit, &aux_specs.rope)).f else null,
    };
}

/// The aux kernels' registry specs.
const aux_specs = struct {
    const attn = [_]gpu.kit.SpecItem{ .{ .key = "heads", .value = tri_aux.attn_heads }, .{ .key = "hd", .value = tri_aux.attn_hd } };
    const index = [_]gpu.kit.SpecItem{ .{ .key = "heads", .value = tri_aux.index_heads }, .{ .key = "dim", .value = tri_aux.index_dim } };
    const hc = [_]gpu.kit.SpecItem{ .{ .key = "d", .value = tri_aux.hc_d }, .{ .key = "nb", .value = tri_aux.hc_nb } };
    const gate = [_]gpu.kit.SpecItem{ .{ .key = "d", .value = tri_aux.gate_d }, .{ .key = "bn", .value = tri_aux.gate_bn } };
    const gate32 = [_]gpu.kit.SpecItem{ .{ .key = "d", .value = tri_aux.gate_d }, .{ .key = "bn", .value = tri_aux.gate32_ne } };
    const rope = [_]gpu.kit.SpecItem{};
};

test "the aux kernels' specs are registry.json's" {
    const reg = @embedFile("dsv41_registry");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), reg, .{});
    const want = [_]struct { []const u8, []const gpu.kit.SpecItem, i64 }{
        .{ "aux.sparse_attn", &aux_specs.attn, 256 },  .{ "aux.index_rows", &aux_specs.index, 256 },    .{ "aux.hc_mix_pf", &aux_specs.hc, 256 },
        .{ "aux.gate_logits", &aux_specs.gate, 256 }, .{ "aux.gate_logits", &aux_specs.gate32, tri_aux.gate32_block }, .{ "aux.rope_heads", &aux_specs.rope, 256 },
    };
    for (want) |w| {
        // exactly one entry of the id has this spec (aux.gate_logits has two builds), with the block the launch uses
        var found: usize = 0;
        for (v.object.get("sources").?.array.items) |src| for (src.object.get("kernels").?.array.items) |k| {
            if (!std.mem.eql(u8, k.object.get("id").?.string, w[0])) continue;
            const spec = k.object.get("spec");
            var same = (if (spec) |sp| sp.object.count() else 0) == w[1].len;
            for (w[1]) |item| {
                const got = if (spec) |sp| sp.object.get(item.key) else null;
                same = same and got != null and got.?.integer == item.value;
            }
            if (!same) continue;
            try std.testing.expectEqual(w[2], k.object.get("block").?.array.items[0].integer);
            found += 1;
        };
        try std.testing.expectEqual(@as(usize, 1), found);
    }
}

/// The EXL3 specializations a rank's weights reach, against its kernel sets: every main-model dense linear (and the
/// head) takes glinear (decode windows, prompt chunks up to 128 rows) and unpack (longer chunks); the drafter's take
/// glinear; every expert set (main and drafter) the prompt GEMMs (grouped_mma for gate / up, grouped_rows for down) and
/// the decode ones (grouped_cp, both epilogues), for its K2 ranges. A missing one refuses the open here, each named
/// once, instead of failing its first launch mid-request: the kit's `required` lists hold ids, not specializations.
pub fn coverage(w: *const weights.Weights, pf: *const exl3_prefill.Kernels, lg: *const exl3_linear.Kernels, xk: *const exl3_experts.Kernels) !void {
    if (scan(w, pf, lg, xk, true).missing > 0) return error.MissingKernel;
}

fn scan(w: *const weights.Weights, pf: *const exl3_prefill.Kernels, lg: *const exl3_linear.Kernels, xk: *const exl3_experts.Kernels, log: bool) Coverage {
    var c: Coverage = .{ .pf = pf, .lg = lg, .xk = xk, .log = log };
    for (w.layers) |*lay| c.layer(lay, true);
    c.dense(w.head, true);
    if (w.dspark) |ds| {
        c.dense(ds.main_proj, false);
        for (ds.blocks) |*lay| c.layer(lay, false);
    }
    return c;
}

const Coverage = struct {
    pf: *const exl3_prefill.Kernels,
    lg: *const exl3_linear.Kernels,
    xk: *const exl3_experts.Kernels,
    log: bool,
    missing: usize = 0,
    named: [64]u32 = undefined, // the missing specializations already logged (a key each)
    n_named: usize = 0,

    fn miss(c: *Coverage, key: u32, comptime fmt: []const u8, args: anytype) void {
        c.missing += 1;
        for (c.named[0..c.n_named]) |k| if (k == key) return;
        if (c.n_named < c.named.len) {
            c.named[c.n_named] = key;
            c.n_named += 1;
        }
        if (c.log) std.log.err("the kit has no " ++ fmt ++ ", which this checkpoint's weights reach", args);
    }

    fn dense(c: *Coverage, l: weights.Linear, prompt: bool) void {
        if (l.dense != 0) return; // a dense projection: dense.zig's (model.zig checks its kernels)
        const cb = exl3_linear.codebook_mul1;
        if (l.k2 > 16) return c.miss(0xff00 | l.k2, "EXL3 kernel for k2 {d} (above 16)", .{l.k2});
        const wk = exl3_linear.plan(l.k, if (l.plan_n != 0) l.plan_n else l.n).wk;
        if (c.lg.glinear[l.k2][cb][if (wk == 8) 1 else 0] == null) c.miss(0x1000 | l.k2 << 4 | wk, "exl3.glinear {{k2 {d}, cb {d}, warps {d}}}", .{ l.k2, cb, wk });
        if (prompt and c.pf.unpack[l.k2][cb] == null) c.miss(0x2000 | l.k2, "exl3.unpack {{k2 {d}, cb {d}}}", .{ l.k2, cb });
    }

    fn experts(c: *Coverage, ex: weights.Experts) void {
        const gu = exl3_experts.range(ex.k2_gu[0], ex.k2_gu[1]);
        const d = exl3_experts.range(ex.k2_d[0], ex.k2_d[1]);
        if (c.xk.mma[gu] == null) c.miss(0x3000 | @as(u32, @intCast(gu)), "exl3.experts.grouped_mma for K2 {d}..{d}", .{ ex.k2_gu[0], ex.k2_gu[1] });
        if (c.xk.rows[d] == null) c.miss(0x4000 | @as(u32, @intCast(d)), "exl3.experts.grouped_rows for K2 {d}..{d}", .{ ex.k2_d[0], ex.k2_d[1] });
        if (c.xk.cp[gu][0] == null) c.miss(0x5000 | @as(u32, @intCast(gu)), "exl3.experts.grouped_cp epi 1 for K2 {d}..{d}", .{ ex.k2_gu[0], ex.k2_gu[1] });
        if (c.xk.cp[d][1] == null) c.miss(0x6000 | @as(u32, @intCast(d)), "exl3.experts.grouped_cp epi 2 for K2 {d}..{d}", .{ ex.k2_d[0], ex.k2_d[1] });
    }

    fn layer(c: *Coverage, lay: *const weights.Layer, prompt: bool) void {
        for ([_]weights.Linear{ lay.wq_a, lay.wq_b, lay.wkv, lay.wo_b }) |l| c.dense(l, prompt);
        for (lay.wo_a[0..lay.groups]) |l| c.dense(l, prompt);
        for ([_]?weights.Linear{ lay.comp_wkv, lay.comp_wgate, lay.idx_wq_b, lay.idx_wk, lay.engram_wkv }) |o| if (o) |l| c.dense(l, prompt);
        if (lay.shared) |sh| {
            c.dense(sh.w1, prompt);
            c.dense(sh.w3, prompt);
        }
        c.experts(lay.experts);
    }
};

/// A spec value of kit entry `e` (a specialized kernel's template argument), or a refusal naming it.
fn specValue(e: *const gpu.kit.Entry, key: []const u8, max: i64) !usize {
    const v = e.k.spec.map.get(key) orelse {
        std.log.err("kit entry {s} ({s}) has no spec {s}", .{ e.k.id, e.k.symbol, key });
        return error.KitManifest;
    };
    if (v < 0 or v > max) {
        std.log.err("kit entry {s} ({s}): spec {s} = {d} is outside 0 .. {d}", .{ e.k.id, e.k.symbol, key, v, max });
        return error.KitManifest;
    }
    return @intCast(v);
}

fn bindAt(kit: *const gpu.Kit, i: usize, comptime id: []const u8, comptime P: type) !gpu.Function {
    return (try gpu.Kernel(id, P).bindIndex(kit, i)).f;
}

/// The prompt chunks' dense EXL3 path: exl3.rot_in, exl3.unpack by {k2, cb}, and (optional: matmul falls back to
/// unpack and the Triton GEMM without it) exl3.gemm by {k2, cb 2, bm 128 | 256}.
pub fn exl3Prefill(kit: *const gpu.Kit) !exl3_prefill.Kernels {
    var k: exl3_prefill.Kernels = .{ .module = undefined, .rot_in = try bind(kit, "exl3.rot_in", params.Exl3RotIn), .kit_owned = true };
    for (kit.entries, 0..) |*e, i| {
        if (std.mem.eql(u8, e.k.id, "exl3.unpack")) {
            k.unpack[try specValue(e, "k2", 16)][try specValue(e, "cb", 2)] = try bindAt(kit, i, "exl3.unpack", params.Exl3Unpack);
        } else if (std.mem.eql(u8, e.k.id, "exl3.gemm")) {
            if (try specValue(e, "cb", 2) != exl3_linear.codebook_mul1) continue;
            const bm = try specValue(e, "bm", 256);
            if (bm != 128 and bm != 256) return error.KitManifest;
            k.gemm[try specValue(e, "k2", 16)][if (bm == 256) 1 else 0] = try bindAt(kit, i, "exl3.gemm", params.Exl3Gemm);
        }
    }
    return k;
}

/// The dense decode path: exl3.rot_many, and exl3.glinear by {k2, cb, warps 4 | 8}.
pub fn exl3Linear(kit: *const gpu.Kit) !exl3_linear.Kernels {
    var k: exl3_linear.Kernels = .{ .module = undefined, .rot_many = try bind(kit, "exl3.rot_many", params.Exl3RotMany), .kit_owned = true };
    for (kit.entries, 0..) |*e, i| if (std.mem.eql(u8, e.k.id, "exl3.glinear")) {
        const warps = try specValue(e, "warps", 8);
        if (warps != 4 and warps != 8) return error.KitManifest;
        k.glinear[try specValue(e, "k2", 16)][try specValue(e, "cb", 2)][if (warps == 8) 1 else 0] = try bindAt(kit, i, "exl3.glinear", params.Exl3GLinear);
    };
    return k;
}

/// The routed experts: the seven plain kernels, and the grouped GEMMs by K2 range (exl3_experts.range of {lo, hi}) for
/// the codebook in use.
pub fn exl3Experts(kit: *const gpu.Kit) !exl3_experts.Kernels {
    const p = params;
    var k: exl3_experts.Kernels = .{
        .base = undefined,
        .cb = undefined,
        .kit_owned = true,
        .group_count = try bind(kit, "exl3.experts.group_count", p.GroupCount),
        .group_place = try bind(kit, "exl3.experts.group_place", p.GroupPlace),
        .work_list = try bind(kit, "exl3.experts.work_list", p.WorkList),
        .rot_in = try bind(kit, "exl3.experts.rot_in", p.ExpertsRotIn),
        .gateup_epilogue = try bind(kit, "exl3.experts.gateup_epilogue", p.GateupEpilogue),
        .down_combine = try bind(kit, "exl3.experts.down_combine", p.DownCombine),
        .decode_prep = try bind(kit, "exl3.experts.decode_prep", p.DecodePrep),
    };
    for (kit.entries, 0..) |*e, i| {
        const grouped = std.mem.startsWith(u8, e.k.id, "exl3.experts.grouped_");
        if (!grouped or try specValue(e, "cb", 2) != exl3_linear.codebook_mul1) continue;
        const r = exl3_experts.range(@intCast(try specValue(e, "lo", 16)), @intCast(try specValue(e, "hi", 16)));
        if (std.mem.eql(u8, e.k.id, "exl3.experts.grouped_mma")) {
            k.mma[r] = try bindAt(kit, i, "exl3.experts.grouped_mma", p.GroupedMma);
        } else if (std.mem.eql(u8, e.k.id, "exl3.experts.grouped_rows")) {
            k.rows[r] = try bindAt(kit, i, "exl3.experts.grouped_rows", p.GroupedRows);
        } else if (std.mem.eql(u8, e.k.id, "exl3.experts.grouped_cp")) {
            const epi = try specValue(e, "epi", 2);
            if (epi < 1) return error.KitManifest;
            k.cp[r][epi - 1] = try bindAt(kit, i, "exl3.experts.grouped_cp", p.GroupedCp);
        }
    }
    k.pf = try pfExperts(kit);
    return k;
}

/// The prompt experts of the AMD kit (exl3.pf_experts.*: count, place, rot, combine, and gu / down by {k2, nm 64 | 128});
/// null when the kit has none of them (exl3_experts.prompt then takes the grouped kernels).
fn pfExperts(kit: *const gpu.Kit) !?exl3_experts.Pf {
    const p = params;
    var k: exl3_experts.Pf = .{
        .count = try optional(kit, "exl3.pf_experts.count", p.PfExperts) orelse return null,
        .place = try bind(kit, "exl3.pf_experts.place", p.PfExperts),
        .rot = try bind(kit, "exl3.pf_experts.rot", p.PfExperts),
        .combine = try bind(kit, "exl3.pf_experts.combine", p.PfExperts),
    };
    for (kit.entries, 0..) |*e, i| {
        const gu = std.mem.eql(u8, e.k.id, "exl3.pf_experts.gu");
        if (!gu and !std.mem.eql(u8, e.k.id, "exl3.pf_experts.down")) continue;
        if (try specValue(e, "cb", 2) != exl3_linear.codebook_mul1) continue;
        const nm = try specValue(e, "nm", 128);
        if (nm != 64 and nm != 128) return error.KitManifest;
        const k2 = try specValue(e, "k2", 16);
        if (gu) {
            k.gu[nm / 128][k2] = try bindAt(kit, i, "exl3.pf_experts.gu", p.PfExperts);
        } else k.down[nm / 128][k2] = try bindAt(kit, i, "exl3.pf_experts.down", p.PfExperts);
    }
    return k;
}

test "a 99.81 MHz wall clock is 10.019 ns a tick in 32.32" {
    const q = nsPerTickQ32(99_810_000);
    // 1e9 / 99.81e6 = 10.01903617...
    try std.testing.expectEqual(@as(u64, 10), q >> 32);
    // rounded to nearest: within half a unit of 2^-32 ns (1.2e-11 relative)
    try std.testing.expectApproxEqRel(@as(f64, 1e9 / 99.81e6), @as(f64, @floatFromInt(q)) / 4294967296.0, 2e-11);
    // 100 MHz is exactly 10 ns
    try std.testing.expectEqual(@as(u64, 10) << 32, nsPerTickQ32(100_000_000));
}

/// The launches the packing tests saw, in order (a fake runtime accepts each one after launch.before records it).
const seen = struct {
    const Launch = struct { handle: usize, count: usize, offsets: [32]u16, sizes: [32]u16 };
    var launches: [16]Launch = undefined;
    var n: usize = 0;
};

fn record(f: gpu.Function, _: gpu.Config, args: *const gpu.Args) gpu.Error!void {
    const l = &seen.launches[seen.n];
    seen.n += 1;
    l.handle = @intFromPtr(f.handle);
    l.count = args.count;
    @memcpy(l.offsets[0..args.count], args.offsets[0..args.count]);
    @memcpy(l.sizes[0..args.count], args.sizes[0..args.count]);
}

fn accept(_: gpu.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: gpu.abi.Stream, _: gpu.abi.Params, _: gpu.abi.Params) callconv(.c) gpu.abi.Result {
    return 0;
}

/// Launch `i` of the last call was `f`, packed exactly as `P` declares.
fn expectLaunch(i: usize, comptime P: type, f: gpu.Function) !void {
    const t = comptime gpu.kernelTable.table(P);
    const l = seen.launches[i];
    try std.testing.expectEqual(@intFromPtr(f.handle), l.handle);
    try std.testing.expectEqual(t.len, l.count);
    for (t, 0..) |q, j| {
        try std.testing.expectEqual(q.offset, l.offsets[j]);
        try std.testing.expectEqual(q.size, l.sizes[j]);
    }
}

/// One launcher call, which must have made exactly the launches `want` lists (function, parameter list) in order (the
/// call is evaluated as the argument, before this runs; the record starts empty for the next one).
fn expectPacked(launched: anytype, want: anytype) !void {
    defer seen.n = 0;
    try launched;
    try std.testing.expectEqual(want.len, seen.n);
    inline for (want, 0..) |w, i| try expectLaunch(i, w[1], w[0]);
}

var fake_driver: gpu.Driver = undefined;

/// A launcher's result without its value (hcMixPf returns the streams it posted).
fn dropValue(r: anytype) !void {
    _ = try r;
}

/// A function handle the fake runtime launches (distinct per call).
fn fake(next: *usize) gpu.Function {
    next.* += 0x10;
    return .{ .d = &fake_driver, .handle = @ptrFromInt(next.*) };
}

test "every launcher packs exactly the parameter list its kernel is bound with" {
    const saved = gpu.launch.before;
    defer gpu.launch.before = saved;
    gpu.launch.before = record;
    fake_driver.api.hipModuleLaunchKernel = accept;
    var n: usize = 0x1000;
    const o: ops_mod.Ops = .{
        .pointwise = undefined, .movement = undefined, .family = undefined, .kit_owned = true,
        .to_f32 = fake(&n), .f16_to_f32 = fake(&n), .add2_bf16 = fake(&n), .topk_i64 = fake(&n), .topk_indices = fake(&n),
        .scatter = fake(&n), .cand_fast = fake(&n), .iota_vis = fake(&n), .l2_prefetch = null, .block_max = fake(&n),
        .pool_pick = fake(&n), .apply_pool = fake(&n), .to_bf16 = null, .strided = fake(&n), .gather = fake(&n),
        .copy16 = fake(&n), .argmax_rows = fake(&n), .gpu_clock = fake(&n), .clock_ns_q32 = 1,
    };
    const x: exact_mod.Exact = .{
        .module = undefined, .kit_owned = true, .k_rms_norm = fake(&n), .k_compress2 = fake(&n), .k_rope = fake(&n),
        .k_bf16_scale = fake(&n), .k_hc_mean4 = fake(&n),
    };
    const s: gpu.Stream = .{ .d = &fake_driver, .handle = @ptrFromInt(0x20) };
    const p = params;
    try expectPacked(o.toF32(s, 16, 32, 100), .{.{ o.to_f32, p.Convert }});
    try expectPacked(o.f16ToF32(s, 16, 32, 100), .{.{ o.f16_to_f32, p.Convert }});
    try expectPacked(o.add2Bf16(s, 16, 32, 48, 100), .{.{ o.add2_bf16, p.Add2Bf16 }});
    try expectPacked(o.topkI64(s, 16, 4096, 2, 4096, 512, 32), .{.{ o.topk_i64, p.TopkI64 }});
    try expectPacked(o.topkIndices(s, 16, 4096, 2, 4096, 2048, 32, 48), .{.{ o.topk_indices, p.TopkIndices }});
    try expectPacked(o.scatterRows(s, 16, 1024, 32, 48, 1024, 1024, 4), .{.{ o.scatter, p.ScatterRows }});
    try expectPacked(o.candFast(s, 16, 4096, 32, 128, 32, 48, 64, 2), .{.{ o.cand_fast, p.CandFast }});
    try expectPacked(o.iotaVis(s, 16, 32, 64, 2), .{.{ o.iota_vis, p.IotaVis }});
    try expectPacked(o.blockMax(s, 16, 4096, 4096, 128, 32, 48, 64, 2), .{.{ o.block_max, p.BlockMax }});
    try expectPacked(o.poolPick(s, 16, 64, 32, 32, 8, 48, 64, 80, 2), .{.{ o.pool_pick, p.PoolPick }});
    try expectPacked(o.applyPool(s, 16, 4096, 4096, 128, 32, 64, 2), .{.{ o.apply_pool, p.ApplyPool }});
    try expectPacked(o.copyRows(s, 16, 1024, 32, 2048, 1024, 4), .{.{ o.copy16, p.CopyRows16 }});
    try expectPacked(o.copyRows(s, 16, 1000, 32, 2000, 1000, 4), .{.{ o.strided, p.StridedCopy }});
    try expectPacked(o.argmaxRows(s, 16, 129280, 129280, 2, 32), .{.{ o.argmax_rows, p.ArgmaxRows }});
    try expectPacked(o.gpuClock(s, 16, 7, true, 512), .{.{ o.gpu_clock.?, p.Clock }});
    try expectPacked(o.gatherRows(s, 16, 100, 32, 48, 512, 4, 64), .{.{ o.gather, p.GatherRows }});
    try expectPacked(x.rmsNorm(s, 16, 5120, 32, 48, 5120, 4, 5120, 1e-6), .{.{ x.k_rms_norm, p.RmsNorm }});
    try expectPacked(x.compress2(s, 16, 32, 48, 4, 512), .{.{ x.k_compress2, p.Compress2 }});
    try expectPacked(x.rope(s, 16, 576, 512, 32, 48, 64, 4, 32, false), .{.{ x.k_rope, p.Rope }});
    try expectPacked(x.bf16Scale(s, 16, 32, 128, 0.0156), .{.{ x.k_bf16_scale, p.Bf16Scale }});
    try expectPacked(x.hcMean4(s, 16, 32, 5120, 4, 5120), .{.{ x.k_hc_mean4, p.HcMean4 }});

    // w-tri's kernels for Triton launches, through the Triton wrappers that hand them over (prompt-chunk shapes)
    const ax: tri_aux.Aux = .{ .attn = fake(&n), .index = fake(&n), .hc_mix = fake(&n), .gate = fake(&n), .gate32 = fake(&n), .rope = fake(&n) };
    const t: @import("tri.zig").Tri = .{ .stream = s, .aux = &ax };
    const attn = @import("tri_attn.zig");
    try expectPacked(attn.sparseAttn(t, .{ .q = 16, .out = 32, .rows = 2048, .h = 32, .hd = 512, .sink = 48, .wsrc = 64, .wsrc_rows = 2048, .wlo = 80, .ring = false, .comp = .{ .fp4 = .{ .codes = 96, .scales = 112 } }, .idx = 128, .n_idx = 512, .pos = 144, .scale = 0.044194173, .window = 128 }), .{.{ ax.attn.?, p.AuxAttn }});
    try expectPacked(@import("tri_index.zig").indexScore(t, 16, .{ .fp4 = .{ .codes = 32, .scales = 48 } }, 64, 80, 8192, 96, null, true, 112, false, 2048, 32, 128), .{.{ ax.index.?, p.AuxIndex }});
    try expectPacked(dropValue(@import("tri_hc.zig").hcMixPf(t, 16, 32, 48, .{ .gathered = 64, .world = 2, .h_out = 80 }, 96, 112, 2048, 5120)), .{.{ ax.hc_mix.?, p.AuxHcMix }});
    const norm = @import("tri_norm.zig");
    try expectPacked(dropValue(norm.gateLogits(t, 16, 32, 48, 2048, 5120, 384)), .{.{ ax.gate.?, p.AuxGate }});
    try expectPacked(dropValue(norm.gateLogits(t, 16, 32, 48, 2048, 5120, 32)), .{.{ ax.gate32.?, p.AuxGate }});
    try expectPacked(norm.ropeHeads(t, 16, 32, 48, 64, 64, false, 2048, 32, 512), .{.{ ax.rope.?, p.AuxRopeHeads }});

    // dense.hip, on a dense pack's shapes (wo_b's rank slice [5120, 4096]; the shared expert's 1152)
    const dk: dense_mod.Kernels = .{ .gemv = fake(&n), .swiglu = fake(&n) };
    const dl: @import("weights.zig").Linear = .{ .words = 0, .suh = 0, .svh = 0, .k = 4096, .n = 5120, .k2 = 0, .dense = 0x10000 };
    try expectPacked(dense_mod.gemv(&dk, s, dl, 0x1000, 4096, 0x2000, 5120, 6, true), .{.{ dk.gemv, p.DenseGemv }});
    try expectPacked(dense_mod.swiglu(&dk, s, 0x1000, 0x2000, 1152, 0x3000, 1152, 6, 1152, 10.0), .{.{ dk.swiglu, p.DenseSwiglu }});

    // vision.hip, on a 1920x1080 picture's shapes (8418 patches, 968 cells) through vit_hip.zig's launchers
    const vk: vit_hip.Kernels = .{ .linear = fake(&n), .rms_norm = fake(&n), .attention = fake(&n), .rope = fake(&n), .add = fake(&n), .silu_mul = fake(&n), .gelu = fake(&n), .unfold = fake(&n), .span = fake(&n) };
    const vp = vit_hip.params;
    try expectPacked(vit_hip.linear(&vk, s, 0x1000, 0x2000, 0x3000, 0x4000, 8418, 3072, 1024), .{.{ vk.linear, vp.Linear }});
    try expectPacked(vit_hip.rmsNorm(&vk, s, 0x1000, 0x2000, 0x3000, 8418), .{.{ vk.rms_norm, vp.RmsNorm }});
    try expectPacked(vit_hip.add(&vk, s, 0x1000, 0x2000, 8418 * 1024), .{.{ vk.add, vp.Add }});
    try expectPacked(vit_hip.rope(&vk, s, 0x1000, 0x2000, 0x3000, 8418), .{.{ vk.rope, vp.Rope }});
    try expectPacked(vit_hip.attention(&vk, s, 0x1000, 0x2000, 8418), .{.{ vk.attention, vp.Attention }});
    try expectPacked(vit_hip.siluMul(&vk, s, 0x1000, 0x2000, 8418), .{.{ vk.silu_mul, vp.SiluMul }});
    try expectPacked(vit_hip.gelu(&vk, s, 0x1000, 968 * 5120), .{.{ vk.gelu, vp.Gelu }});
    try expectPacked(vit_hip.unfold(&vk, s, 0x1000, 0x2000, .{ .best_h = 1078, .best_w = 1918, .n_vit_h = 77, .n_vit_w = 137, .n_llm_h = 26, .n_llm_w = 46 }), .{.{ vk.unfold, vp.Unfold }});
    try expectPacked(vit_hip.span(&vk, s, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 44, 22), .{.{ vk.span, vp.Span }});

    // EXL3, on the checkpoint's shapes (hidden 5120; a rank's expert width 1152; 384 routed + the shared expert)
    const cb = exl3_linear.codebook_mul1;
    var pf: exl3_prefill.Kernels = .{ .module = undefined, .rot_in = fake(&n), .kit_owned = true };
    pf.unpack[6][cb] = fake(&n);
    const lin: @import("weights.zig").Linear = .{ .words = 0x10000, .suh = 0x20000, .svh = 0x30000, .k = 5120, .n = 1536, .k2 = 6 };
    try expectPacked(exl3_prefill.rotIn(&pf, s, 0x1000, .bf16, 5120, 0x2000, 0x3000, 64, 5120), .{.{ pf.rot_in, p.Exl3RotIn }});
    try expectPacked(exl3_prefill.unpack(&pf, s, lin, 0x4000), .{.{ pf.unpack[6][cb].?, p.Exl3Unpack }});
    pf.gemm[6] = .{ fake(&n), fake(&n) };
    // 640 rows of 1,536 columns: 12 strips x 3 tiles of 256 rows < 120 programs, and those would cover 768 rows, the
    // 128-row tile
    try expectPacked(exl3_prefill.gemm(&pf, s, lin, 0x2000, 0x5000, .bf16, 1536, 640), .{.{ pf.gemm[6][0].?, p.Exl3Gemm }});
    try std.testing.expectEqual(@as(usize, 1), exl3_prefill.gemmTile(16384, 2048));
    // under 120 programs: the 256-row tile where it covers the rows 128-row tiles do (from 512 rows)
    try std.testing.expectEqual(@as(usize, 1), exl3_prefill.gemmTile(1280, 1024));
    try std.testing.expectEqual(@as(usize, 1), exl3_prefill.gemmTile(512, 714));
    try std.testing.expectEqual(@as(usize, 0), exl3_prefill.gemmTile(1280, 640));
    try std.testing.expectEqual(@as(usize, 0), exl3_prefill.gemmTile(1280, 384));
    try std.testing.expectEqual(@as(usize, 0), exl3_prefill.gemmTile(1024, 128));
    var lg: exl3_linear.Kernels = .{ .module = undefined, .rot_many = fake(&n), .kit_owned = true };
    for (&lg.glinear[6][cb]) |*g| g.* = fake(&n);
    const call: exl3_linear.Call = .{ .layer = lin, .x = 0x1000, .ldx = 5120, .x_dtype = .bf16, .xh = 0x2000, .y = 0x3000, .ldy = 1536, .y_dtype = .bf16, .counters = 0x4000 };
    try expectPacked(exl3_linear.rotMany(&lg, s, &.{call}, 4, false), .{.{ lg.rot_many, p.Exl3RotMany }});
    try exl3_linear.glinear(&lg, s, &.{call}, 4, 0x5000, false, false);
    try std.testing.expectEqual(@as(usize, 1), seen.n);
    try std.testing.expect(seen.launches[0].handle == @intFromPtr(lg.glinear[6][cb][0].?.handle) or seen.launches[0].handle == @intFromPtr(lg.glinear[6][cb][1].?.handle));
    try expectLaunch(0, p.Exl3GLinear, .{ .d = &fake_driver, .handle = @ptrFromInt(seen.launches[0].handle) });
    seen.n = 0;

    var xk: exl3_experts.Kernels = .{
        .base = undefined, .cb = undefined, .kit_owned = true, .group_count = fake(&n), .group_place = fake(&n), .work_list = fake(&n),
        .rot_in = fake(&n), .gateup_epilogue = fake(&n), .down_combine = fake(&n), .decode_prep = fake(&n),
    };
    for (0..3) |r| {
        xk.mma[r] = fake(&n);
        xk.rows[r] = fake(&n);
        xk.cp[r] = .{ fake(&n), fake(&n) };
    }
    const ex: @import("weights.zig").Experts = .{
        .trellis = 0x100000, .gate_ptr = 0x110000, .up_ptr = 0x120000, .down_ptr = 0x130000, .gate_k2 = 0x140000, .up_k2 = 0x150000,
        .down_k2 = 0x160000, .suh_g = 0x170000, .suh_u = 0x180000, .svh_g = 0x190000, .svh_u = 0x1a0000, .suh_d = 0x1b0000,
        .svh_d = 0x1c0000, .count = 385, .dims = 5120, .width = 1152, .k2_gu = .{ 2, 3 }, .k2_d = .{ 2, 3 }, .down_k = 1152, .down_n = 5120,
    };
    const sc: exl3_experts.Scratch = .{ .rows = 2048, .slots = 7, .xg = 0x200000, .xu = 0x210000, .xd = 0x220000, .z = 0x230000, .no_y = 0x240000, .ids = 0x250000, .count = 0x260000, .counts = 0x270000, .members = 0x280000, .work_gu = 0x290000, .work_d = 0x2a0000, .order = 0x2b0000, .jobs = 0x2c0000, .meta = 0x2d0000 };
    const rg = exl3_experts.range(2, 3);
    try expectPacked(exl3_experts.prompt(&xk, s, ex, sc, 0x1000, 5120, 0x2000, 0x3000, 0x4000, 64, 10.0), .{
        .{ xk.group_count, p.GroupCount },         .{ xk.group_place, p.GroupPlace }, .{ xk.work_list, p.WorkList },
        .{ xk.rot_in, p.ExpertsRotIn },            .{ xk.mma[rg].?, p.GroupedMma },   .{ xk.gateup_epilogue, p.GateupEpilogue },
        .{ xk.rows[rg].?, p.GroupedRows },         .{ xk.down_combine, p.DownCombine },
    });
    // with the kit's prompt experts (routed K2 4, the shared expert's 6): count, place, rot, gu and down for each K2, combine
    var pfk: exl3_experts.Pf = .{ .count = fake(&n), .place = fake(&n), .rot = fake(&n), .combine = fake(&n) };
    for ([_]usize{ 4, 6 }) |k2| {
        pfk.gu[0][k2] = fake(&n);
        pfk.down[0][k2] = fake(&n);
    }
    xk.pf = pfk;
    var exs = ex;
    exs.k2_masks = .{ 1 << 4 | 1 << 6, 1 << 4 | 1 << 6 };
    const P = p.PfExperts;
    try expectPacked(exl3_experts.prompt(&xk, s, exs, sc, 0x1000, 5120, 0x2000, 0x3000, 0x4000, 64, 10.0), .{
        .{ pfk.count, P },          .{ pfk.place, P },          .{ pfk.rot, P },     .{ pfk.gu[0][4].?, P }, .{ pfk.gu[0][6].?, P },
        .{ pfk.down[0][4].?, P },   .{ pfk.down[0][6].?, P },   .{ pfk.combine, P },
    });
    xk.pf = null;
    const dsc: exl3_experts.DecodeScratch = .{ .rows = 16, .slots = 7, .xg = 0x200000, .xu = 0x210000, .xd = 0x220000, .z = 0x230000, .y = 0x240000, .cnt_gu = 0x250000, .cnt_d = 0x260000, .epoch = 0x270000, .ready = 0x280000, .ready_cnt = 0x290000, .ids = 0x2a0000, .count = 0x2b0000, .members = 0x2c0000 };
    try expectPacked(exl3_experts.decode(&xk, s, ex, dsc, 0x1000, 5120, 0x2000, 0x3000, 0x4000, 4, 10.0), .{
        .{ xk.decode_prep, p.DecodePrep }, .{ xk.cp[rg][0].?, p.GroupedCp }, .{ xk.cp[rg][1].?, p.GroupedCp },
    });
}

test "coverage names each missing specialization the weights reach, once" {
    const cb = exl3_linear.codebook_mul1;
    var n: usize = 0x1000;
    var pf: exl3_prefill.Kernels = .{ .module = undefined, .rot_in = fake(&n), .kit_owned = true };
    var lg: exl3_linear.Kernels = .{ .module = undefined, .rot_many = fake(&n), .kit_owned = true };
    var xk: exl3_experts.Kernels = .{
        .base = undefined, .cb = undefined, .kit_owned = true, .group_count = fake(&n), .group_place = fake(&n), .work_list = fake(&n),
        .rot_in = fake(&n), .gateup_epilogue = fake(&n), .down_combine = fake(&n), .decode_prep = fake(&n),
    };
    // the checkpoint's dense K2 {8, 10, 12, 16} at both warp counts, the experts' two ranges
    for ([_]usize{ 8, 10, 12, 16 }) |k2| {
        pf.unpack[k2][cb] = fake(&n);
        lg.glinear[k2][cb] = .{ fake(&n), fake(&n) };
    }
    for ([_]usize{ exl3_experts.range(2, 10), exl3_experts.range(8, 8) }) |r| {
        xk.mma[r] = fake(&n);
        xk.rows[r] = fake(&n);
        xk.cp[r] = .{ fake(&n), fake(&n) };
    }
    const lin = struct {
        fn f(k2: u32) weights.Linear {
            return .{ .words = 1, .suh = 1, .svh = 1, .k = 5120, .n = 1536, .k2 = k2 };
        }
    }.f;
    const ex: weights.Experts = .{
        .trellis = 1, .gate_ptr = 1, .up_ptr = 1, .down_ptr = 1, .gate_k2 = 1, .up_k2 = 1, .down_k2 = 1, .suh_g = 1, .suh_u = 1, .svh_g = 1,
        .svh_u = 1, .suh_d = 1, .svh_d = 1, .count = 385, .dims = 5120, .width = 1152, .k2_gu = .{ 4, 6 }, .k2_d = .{ 4, 6 }, .down_k = 1152, .down_n = 5120,
    };
    var layers = [_]weights.Layer{.{ .idx = 0, .ratio = 4, .wq_a = lin(8), .wq_b = lin(10), .wkv = lin(12), .wo_a = .{ lin(16), lin(16), lin(16), lin(16) }, .wo_b = lin(8), .experts = ex }};
    var w: weights.Weights = .{ .gpa = std.testing.allocator, .head = lin(16), .layers = &layers };
    try coverage(&w, &pf, &lg, &xk);
    // a K2 the kit lacks on two linears: four misses (glinear and unpack each), two specializations named
    layers[0].wq_b = lin(6);
    layers[0].wkv = lin(6);
    var c = scan(&w, &pf, &lg, &xk, false);
    try std.testing.expectEqual(@as(usize, 4), c.missing);
    try std.testing.expectEqual(@as(usize, 2), c.n_named);
    // the decode down epilogue of the routed experts' range gone: one miss
    layers[0].wq_b = lin(10);
    layers[0].wkv = lin(12);
    xk.cp[exl3_experts.range(2, 10)][1] = null;
    c = scan(&w, &pf, &lg, &xk, false);
    try std.testing.expectEqual(@as(usize, 1), c.missing);
    // the drafter's blocks take no prompt GEMM of their dense linears: unpack may be absent for their K2
    xk.cp[exl3_experts.range(2, 10)][1] = fake(&n);
    var blocks = [_]weights.Layer{layers[0]};
    blocks[0].wq_a = lin(6);
    w.dspark = .{ .blocks = &blocks, .main_proj = lin(8), .main_norm = 0, .norm = 0, .markov_embed = 0, .markov_head = 0, .conf = 0 };
    lg.glinear[6][cb] = .{ fake(&n), fake(&n) };
    try coverage(&w, &pf, &lg, &xk);
}

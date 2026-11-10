//! The dsv41 family built for HIP (zig build test-dsv41-hip-host): a HIP rank's whole serving path is analyzed from its
//! entry points (native.open, which builds the lane backend over every model primitive, its refusals and weight check),
//! so anything it reaches must compile without CUDA, NCCL or verbs (its vision tower is vit_hip.zig on the AMD kit, no
//! cubin, cuBLAS or torch kernel); then the tests of the files it uses run on the host (no GPU work).
const std = @import("std");
const gpu = @import("gpu");
const native = @import("native.zig");
const model = @import("model.zig");

test "a HIP rank's serving path, vision included, compiles without CUDA, NCCL or verbs RDMA" {
    comptime std.debug.assert(gpu.backend == .hip and gpu.features.vision and !gpu.features.verbs_rdma and !gpu.features.world4);
    _ = &native.open;
    _ = &native.explain;
    _ = &native.weightBytes;
    _ = &native.modalities;
    _ = &model.Model.open;
    _ = &model.Model.close;
}

test "the text capability stays a HIP rank's default; text_image is asked for" {
    try std.testing.expectEqual(model.Modalities.text, model.default_modalities);
    try std.testing.expectEqual(model.Modalities.text, try native.modalities(.{ .context = 8192, .drafts = true }));
    try std.testing.expectEqual(model.Modalities.text_image, try native.modalities(.{ .context = 8192, .drafts = true, .modalities = .text_image }));
    try std.testing.expect(std.mem.indexOf(u8, native.explain(null, error.NoVisionWeights).?, "TF_DS_VISION_DIR") != null);
    try std.testing.expect(std.mem.indexOf(u8, native.explain(null, error.NoVisionKit).?, "gate_bias_vl") != null);
    try std.testing.expect(std.mem.indexOf(u8, native.explain(null, error.NoCollective).?, "w-link") != null);
}

test {
    // the host tests of the files a HIP rank uses
    _ = @import("coll.zig");
    _ = @import("model.zig");
    _ = @import("tcp_wire.zig");
    _ = @import("tcp_coll.zig");
    _ = @import("tcp_proxy.zig");
    _ = @import("fan.zig");
    _ = @import("config.zig");
    _ = @import("plan.zig");
    _ = @import("rank_cache.zig");
    _ = @import("checkpoint.zig");
    _ = @import("engram.zig");
    _ = @import("engram_io.zig");
    _ = @import("engram_aio.zig");
    _ = @import("sampling.zig");
    _ = @import("text.zig");
    _ = @import("markov_tokens.zig");
    _ = @import("tri.zig");
    _ = @import("tri_basic.zig");
    _ = @import("tri_hc.zig");
    _ = @import("tri_norm.zig");
    _ = @import("tri_index.zig");
    _ = @import("tri_attn.zig");
    _ = @import("tri_aux.zig");
    _ = @import("tri_markov.zig");
    _ = @import("ops.zig");
    _ = @import("exact.zig");
    _ = @import("weights.zig");
    _ = @import("draft.zig");
    _ = @import("round.zig");
    _ = @import("prompt.zig");
    _ = @import("lanes.zig");
    _ = @import("disk.zig");
    _ = @import("native.zig");
    _ = @import("hipblas.zig");
    _ = @import("hip_load.zig");
    _ = @import("vision.zig");
    _ = @import("vit_hip.zig");
    _ = @import("picture.zig");
    _ = @import("pil.zig");
}

//! The `gpu` module of a HIP build: the HIP runtime (hip/root.zig) under the backend-neutral names the dsv41 family
//! uses (interface.zig), plus what replaces the family's direct driver-table calls. No CUDA code is reachable from here.

const rt = @import("hip/root.zig");

pub const Backend = enum { cuda, hip };
pub const backend: Backend = .hip;

/// What the family may ask of this backend. CUDA-only launch features are absent; verbs RDMA and the four-node 2D split
/// are not implemented on HIP (ADAPTER.md sections 2 and 8). Vision: the tower on the AMD kit's vision.* kernels
/// (vit_hip.zig), opened only when a rank serves text_image (the HIP default stays text).
pub const features = .{ .pdl = false, .clusters = false, .vision = true, .verbs_rdma = false, .l2_prefetch = false, .world4 = false };

pub const abi = rt.abi;
pub const Driver = rt.Driver;
pub const Error = rt.Error;
pub const Context = rt.Context;
pub const Caps = rt.Caps;
pub const caps = rt.caps;
pub const DeviceBuffer = rt.DeviceBuffer;
pub const DeviceKind = rt.DeviceKind;
pub const HostBuffer = rt.HostBuffer;
pub const HostFlags = rt.HostFlags;
pub const Usage = rt.Usage;
pub const usage = rt.usage;
pub const Stream = rt.Stream;
pub const Event = rt.Event;
pub const Module = rt.Module;
pub const Function = rt.Function;
pub const code_object = rt.code_object;
pub const checkImage = rt.checkImage;
pub const launch = rt.launch;
pub const Args = rt.Args;
pub const Config = rt.Config;
pub const Dim3 = rt.Dim3;
pub const graph = rt.graph;
pub const health = rt.health;
pub const markFault = rt.markFault;
pub const abiChecked = rt.abiChecked;

pub const copyH2D = rt.copyH2D;
pub const copyD2H = rt.copyD2H;
pub const copyD2DAsync = rt.copyD2DAsync;
pub const fill8Async = rt.fill8Async;
pub const fill32Async = rt.fill32Async;

/// Pinned host bytes to `dst` in `stream`'s order; they must stay unchanged until the stream passes the copy.
pub fn uploadPinned(d: *const Driver, stream: Stream, dst: u64, pinned: []const u8) Error!void {
    if (pinned.len == 0) return;
    try d.check(d.api.hipMemcpyHtoDAsync(abi.ptr(dst), @constCast(pinned.ptr), pinned.len, stream.handle), "hipMemcpyHtoDAsync");
}

pub fn streamCapturing(s: Stream) Error!bool {
    return s.capturing();
}

pub fn synchronizeFor(s: Stream, ns: u64) Error!void {
    return s.synchronizeFor(ns);
}

/// Pageable host bytes in a stream's order through a pinned ring (staging.zig).
pub const Staging = @import("staging.zig").Staging(@This());

/// The AMD kit (manifest-verified code objects), typed kernels bound from it and its Triton variant set.
pub const kit = rt.kit;
pub const Kit = rt.kit.Kit;
pub const Kernel = rt.kit.Kernel;
pub const aot = rt.aot;
pub const manifest = @import("manifest.zig");
pub const hsaco = @import("hsaco.zig");
/// Parameter tables of typed kernels (kernel.zig), for tests and tools.
pub const kernelTable = @import("kernel.zig");
/// zig/kernels/hip/dsv41/abi.h translated: TF_DS_HIP_ABI_VERSION and the by-value kernel structs.
pub const kernel_abi = @import("dsv41_abi");

test {
    _ = rt;
    _ = @import("manifest.zig");
    _ = @import("hsaco.zig");
    _ = @import("msgpack.zig");
    _ = @import("kernel.zig");
    _ = @import("staging.zig");
    @import("interface.zig").check(@This());
}

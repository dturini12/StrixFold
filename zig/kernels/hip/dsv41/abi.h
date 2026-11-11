/* The dsv41 HIP kernel ABI (tools/dsv41-hip/ADAPTER.md section 6): the version the kit and the binary must share, and
 * every struct a kernel takes by value. Included by every .hip file of the kit and translated for Zig by the build, so
 * each struct has one definition. Rules: fixed-width types only (uint64_t for device addresses), no bool, enum, long,
 * pointer or bitfield members, natural alignment, explicit padding, and a size + every-offset assertion per struct.
 * Bump TF_DS_HIP_ABI_VERSION on any change to a struct, a kernel's parameter list or its documented semantics. */
#ifndef TF_DS_HIP_ABI_H
#define TF_DS_HIP_ABI_H

#include <stddef.h>
#include <stdint.h>

#define TF_DS_HIP_ABI_VERSION 1

#ifdef __cplusplus
#define TF_DS_ASSERT(c, m) static_assert(c, m)
#else
#define TF_DS_ASSERT(c, m) _Static_assert(c, m)
#endif

/* ABI test (tf-dsv41-hip-abi-test): one of each member width, with padding the layout must reproduce. */
typedef struct {
    uint64_t addr;
    int32_t i32;
    float f32;
    uint64_t words[3];
    int64_t i64;
    uint16_t u16;
    uint8_t u8;
    uint8_t pad0;
    uint32_t u32;
    double f64;
} tf_ds_abi_mixed;
TF_DS_ASSERT(sizeof(tf_ds_abi_mixed) == 64, "tf_ds_abi_mixed size");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, addr) == 0, "addr");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, i32) == 8, "i32");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, f32) == 12, "f32");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, words) == 16, "words");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, i64) == 40, "i64");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, u16) == 48, "u16");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, u8) == 50, "u8");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, pad0) == 51, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, u32) == 52, "u32");
TF_DS_ASSERT(offsetof(tf_ds_abi_mixed, f64) == 56, "f64");

/* ABI test: a kernarg as large as EXL3's grouped-linear arguments today (GArgs, 1424 bytes). */
typedef struct {
    uint64_t w[176];
    int32_t n;
    int32_t m;
    int32_t discard;
    int32_t pad0;
} tf_ds_abi_big;
TF_DS_ASSERT(sizeof(tf_ds_abi_big) == 1424, "tf_ds_abi_big size");
TF_DS_ASSERT(offsetof(tf_ds_abi_big, n) == 1408, "n");
TF_DS_ASSERT(offsetof(tf_ds_abi_big, m) == 1412, "m");
TF_DS_ASSERT(offsetof(tf_ds_abi_big, discard) == 1416, "discard");
TF_DS_ASSERT(offsetof(tf_ds_abi_big, pad0) == 1420, "pad0");

/* EXL3 (w-exl3's kernels): the by-value arguments of the CUDA kernels the family's launchers pack, field for field
 * (exl3_linear.zig GLayer / GArgs / RLayer / RArgs, exl3_experts.zig DecodeEpi; hip_load.zig checks those Zig structs
 * against these at compile time). The padding is explicit here and implicit (the same bytes) in Zig. */
#define TF_DS_EXL3_GMAX 8

/* one layer of glinear's group (linear_grouped.cu GLayer) */
typedef struct {
    uint64_t xh;
    uint64_t T;
    uint64_t svh;
    uint64_t bias;
    uint64_t y;
    uint64_t Z;
    uint64_t counters;
    int64_t ldx;
    int64_t ldy;
    int64_t stride_k;
    int64_t stride_nb;
    int32_t y_dtype;
    int32_t K;
    int32_t N;
    int32_t SK;
    int32_t first;
    int32_t pad0;
    uint64_t rsuh;
    uint64_t rxh;
    int64_t ldr;
    int32_t roff;
    int32_t pad1;
    uint64_t rcos;
    uint64_t rsin;
    uint64_t rpos;
    int32_t rhd;
    int32_t rrd;
} tf_ds_exl3_glayer;
TF_DS_ASSERT(sizeof(tf_ds_exl3_glayer) == 176, "tf_ds_exl3_glayer size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, xh) == 0, "xh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, T) == 8, "T");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, svh) == 16, "svh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, bias) == 24, "bias");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, y) == 32, "y");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, Z) == 40, "Z");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, counters) == 48, "counters");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldx) == 56, "ldx");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldy) == 64, "ldy");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, stride_k) == 72, "stride_k");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, stride_nb) == 80, "stride_nb");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, y_dtype) == 88, "y_dtype");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, K) == 92, "K");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, N) == 96, "N");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, SK) == 100, "SK");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, first) == 104, "first");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, pad0) == 108, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rsuh) == 112, "rsuh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rxh) == 120, "rxh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, ldr) == 128, "ldr");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, roff) == 136, "roff");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, pad1) == 140, "pad1");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rcos) == 144, "rcos");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rsin) == 152, "rsin");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rpos) == 160, "rpos");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rhd) == 168, "rhd");
TF_DS_ASSERT(offsetof(tf_ds_exl3_glayer, rrd) == 172, "rrd");

/* exl3.glinear's one parameter (GArgs) */
typedef struct {
    tf_ds_exl3_glayer l[TF_DS_EXL3_GMAX];
    int32_t n;
    int32_t M;
    int32_t discard;
    int32_t pad0;
} tf_ds_exl3_gargs;
TF_DS_ASSERT(sizeof(tf_ds_exl3_gargs) == 1424, "tf_ds_exl3_gargs size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, n) == 1408, "n");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, M) == 1412, "M");
TF_DS_ASSERT(offsetof(tf_ds_exl3_gargs, discard) == 1416, "discard");

/* one layer of rot_many (RLayer) */
typedef struct {
    uint64_t x;
    uint64_t suh;
    uint64_t xh;
    int64_t ldx;
    int32_t x_dtype;
    int32_t K;
    int32_t first;
    int32_t pad0;
} tf_ds_exl3_rlayer;
TF_DS_ASSERT(sizeof(tf_ds_exl3_rlayer) == 48, "tf_ds_exl3_rlayer size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, x) == 0, "x");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, suh) == 8, "suh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, xh) == 16, "xh");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, ldx) == 24, "ldx");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, x_dtype) == 32, "x_dtype");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, K) == 36, "K");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, first) == 40, "first");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rlayer, pad0) == 44, "pad0");

/* exl3.rot_many's one parameter (RArgs) */
typedef struct {
    tf_ds_exl3_rlayer l[TF_DS_EXL3_GMAX];
    int32_t n;
    int32_t M;
} tf_ds_exl3_rargs;
TF_DS_ASSERT(sizeof(tf_ds_exl3_rargs) == 392, "tf_ds_exl3_rargs size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rargs, n) == 384, "n");
TF_DS_ASSERT(offsetof(tf_ds_exl3_rargs, M) == 388, "M");

/* exl3.experts.grouped_cp's epilogue (experts_grouped.cuh DecodeEpi, its defaults: act_mode 1, store_y 1) */
typedef struct {
    uint64_t pick;
    int32_t E;
    int32_t pad0;
    uint64_t svh_g;
    uint64_t svh_u;
    uint64_t suh_d;
    uint64_t xd;
    float limit;
    int32_t act_mode;
    uint64_t svh_d;
    uint64_t y;
    uint64_t wts;
    uint64_t add;
    uint64_t out;
    int32_t store_y;
    int32_t pad1;
    uint64_t cnt;
    uint64_t ready;
    uint64_t ready_cnt;
    uint64_t epoch;
    int32_t discard;
    int32_t pad2;
} tf_ds_exl3_decode_epi;
TF_DS_ASSERT(sizeof(tf_ds_exl3_decode_epi) == 144, "tf_ds_exl3_decode_epi size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, pick) == 0, "pick");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, E) == 8, "E");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, pad0) == 12, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, svh_g) == 16, "svh_g");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, svh_u) == 24, "svh_u");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, suh_d) == 32, "suh_d");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, xd) == 40, "xd");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, limit) == 48, "limit");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, act_mode) == 52, "act_mode");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, svh_d) == 56, "svh_d");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, y) == 64, "y");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, wts) == 72, "wts");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, add) == 80, "add");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, out) == 88, "out");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, store_y) == 96, "store_y");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, pad1) == 100, "pad1");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, cnt) == 104, "cnt");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, ready) == 112, "ready");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, ready_cnt) == 120, "ready_cnt");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, epoch) == 128, "epoch");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, discard) == 136, "discard");
TF_DS_ASSERT(offsetof(tf_ds_exl3_decode_epi, pad2) == 140, "pad2");

/* exl3.pf_experts.* (w-pf's prompt experts, exl3_pf.hip; exl3_experts.zig PfArgs and pfPlan; w-pf's TfDsExl3PfMoeArgs):
 * the experts of one prompt chunk, the shared expert the table's last. Tables as weights.zig Experts: pointer tables
 * int64 [E] to checkpoint-layout trellises, K2 tables int32 [E] (gate's equal up's), suh_g, suh_u [E, d], svh_g,
 * svh_u, suh_d [E, i], svh_d [E, d] fp16 stacked; x [r, ldx] (x_fmt 0 fp16, 1 bf16); ids int32, wts fp32 [r, slots];
 * scratch xg, xu fp16 [r * slots, d], xd fp16 [r * slots, i], z fp32 [r * slots, d], order int32 [r * slots], jobs
 * int32 [3 * maxjobs], meta int32 [2 * 1025]; out fp32 [r, ldo] = (out if accumulate, else 0) then the fma chain over
 * each row's slots in slot order of wts * the slot's output. A gu / down kernel (_k<K2>) runs the jobs of the experts
 * whose gate (down) K2 is its own. */
typedef struct {
    uint64_t gate_ptr, up_ptr, down_ptr;
    uint64_t gate_k2, up_k2, down_k2;
    uint64_t suh_g, suh_u, svh_g, svh_u, suh_d, svh_d;
    uint64_t x, ids, wts;
    uint64_t xg, xu, xd, z, order, jobs, meta;
    uint64_t out;
    uint32_t r, slots, e, d, i, ldx, ldo, maxjobs, nmax, accumulate;
    float limit;
    uint32_t x_fmt;
} tf_ds_exl3_pf_args;
TF_DS_ASSERT(sizeof(tf_ds_exl3_pf_args) == 232, "tf_ds_exl3_pf_args size");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, gate_k2) == 24, "gate_k2");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, out) == 176, "out");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, r) == 184, "r");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, accumulate) == 220, "accumulate");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, limit) == 224, "limit");
TF_DS_ASSERT(offsetof(tf_ds_exl3_pf_args, x_fmt) == 228, "x_fmt");


/* w-tri: sparse attention with a sink (aux_attn.hip, tf_ds_aux_sparse_attn_h32_d512). Device addresses: q bf16
 * [rows, 32, 512]; wsrc bf16 window rows; wlo int64 [1] (a linear window's first position); comp bf16 [N, 512] or
 * FP4 codes u8 [N, 256] with csc E4M3 scales u8 [N, 32]; idx int64 [rows, n_idx] (-1 none); pos int64 [rows]; pm, pl,
 * po fp32: the split partials in the layout of the Triton _sparse_attn_part (merged by the Triton merge kernels), or
 * out bf16 [rows, 32, 512] (one pass); sink fp32 [32]; rbase / cbase int64 [rows] (per-row ring / compressed bases).
 * flags: 1 ring, 2 compressed keys, 4 packed FP4, 8 one pass, 16 per-row bases. Grid (rows, 1, splits), block 256. */
typedef struct {
    uint64_t q, wsrc, wlo, comp, csc, idx, pos, pm, pl, po, sink, out, rbase, cbase;
    int32_t ring_size, n_idx, rows, heads;
    int32_t win, splits, flags, pad0;
    float scale;
    int32_t pad1;
} tf_ds_aux_attn_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_attn_args) == 152, "tf_ds_aux_attn_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, q) == 0, "q");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, wsrc) == 8, "wsrc");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, wlo) == 16, "wlo");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, comp) == 24, "comp");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, csc) == 32, "csc");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, idx) == 40, "idx");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, pos) == 48, "pos");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, pm) == 56, "pm");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, pl) == 64, "pl");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, po) == 72, "po");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, sink) == 80, "sink");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, out) == 88, "out");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, rbase) == 96, "rbase");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, cbase) == 104, "cbase");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, ring_size) == 112, "ring_size");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, n_idx) == 116, "n_idx");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, rows) == 120, "rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, heads) == 124, "heads");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, win) == 128, "win");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, splits) == 132, "splits");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, flags) == 136, "flags");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, pad0) == 140, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, scale) == 144, "scale");
TF_DS_ASSERT(offsetof(tf_ds_aux_attn_args, pad1) == 148, "pad1");

/* w-tri: the indexer's scores of a prompt chunk (aux_index.hip, tf_ds_aux_index_rows_h32_d128). Device addresses: q
 * bf16 [rows, 32, 128]; k FP4 codes u8 [>= n, 64] with ks E8M0 scales u8 [>= n, 4]; w bf16 [rows, 32]; vis int64
 * [rows]; out fp32 [rows, n] scores or (flags bit 0) int64 keys; tmax int64 [rows, nt] (flags bit 1, with keys), nt =
 * ceil(n / 64). Grid (ceil(rows / 8), splits, 1), block 256: grid y splits the key tiles into consecutive ranges
 * (any count; about 240 workgroups in all fill the GPU). */
typedef struct {
    uint64_t q, k, ks, w, vis, out, tmax;
    int32_t rows, n, nt, flags;
} tf_ds_aux_index_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_index_args) == 72, "tf_ds_aux_index_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, q) == 0, "q");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, k) == 8, "k");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, ks) == 16, "ks");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, w) == 24, "w");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, vis) == 32, "vis");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, out) == 40, "out");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, tmax) == 48, "tmax");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, rows) == 56, "rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, n) == 60, "n");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, nt) == 64, "nt");
TF_DS_ASSERT(offsetof(tf_ds_aux_index_args, flags) == 68, "flags");

/* w-tri: the mHC posted streams and mixes of a prompt chunk (aux_hc.hip, tf_ds_aux_hc_mix_pf_d5120). Device addresses:
 * x bf16 [rows, 4, 5120]; xo bf16 [rows, 4, 5120] (flags bit 0, posted); g fp32 [world, rows, 5120], rank q at
 * g + q rs elements; post fp32 [rows, 4]; comb fp32 [rows, 4, 4]; fn fp32 [24, 20480]; part fp32 [rows, 40, 32]
 * (slots 0..24 written). world 1 or 2. Grid (10 ceil(rows / 32), 1, 1), block 256. */
typedef struct {
    uint64_t x, xo, g, post, comb, fn, part;
    int64_t rs;
    int32_t rows, world, flags, pad0;
} tf_ds_aux_hc_mix_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_hc_mix_args) == 80, "tf_ds_aux_hc_mix_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, x) == 0, "x");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, xo) == 8, "xo");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, g) == 16, "g");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, post) == 24, "post");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, comb) == 32, "comb");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, fn) == 40, "fn");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, part) == 48, "part");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, rs) == 56, "rs");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, rows) == 64, "rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, world) == 68, "world");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, flags) == 72, "flags");
TF_DS_ASSERT(offsetof(tf_ds_aux_hc_mix_args, pad0) == 76, "pad0");

/* w-link: tcp_proxy's mapped coherent slab header (coll.hip; tcp_proxy.zig Shared/Slot). One slot per gather in flight:
 * ready/done/consumed are generations (the gather's sequence number), bytes and site describe the gather for the host
 * proxy thread. seq counts staged gathers (the stage kernel's last block advances it); staged/collected count the
 * blocks of the stage/collect in flight; abort non-zero ends every device wait. At most 16 slots. */
typedef struct {
    uint64_t ready, done, consumed, bytes, site, pad[3];
} tf_ds_coll_slot;
TF_DS_ASSERT(sizeof(tf_ds_coll_slot) == 64, "tf_ds_coll_slot size");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, ready) == 0, "ready");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, done) == 8, "done");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, consumed) == 16, "consumed");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, bytes) == 24, "bytes");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, site) == 32, "site");
TF_DS_ASSERT(offsetof(tf_ds_coll_slot, pad) == 40, "pad");

#define TF_DS_COLL_MAX_SLOTS 16
typedef struct {
    uint32_t abort, pad0;
    uint64_t seq;
    uint32_t staged, collected;
    uint64_t pad1[5];
    tf_ds_coll_slot slot[TF_DS_COLL_MAX_SLOTS];
} tf_ds_coll_shared;
TF_DS_ASSERT(sizeof(tf_ds_coll_shared) == 64 + TF_DS_COLL_MAX_SLOTS * 64, "tf_ds_coll_shared size");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, abort) == 0, "abort");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, pad0) == 4, "pad0");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, seq) == 8, "seq");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, staged) == 16, "staged");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, collected) == 20, "collected");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, pad1) == 24, "pad1");
TF_DS_ASSERT(offsetof(tf_ds_coll_shared, slot) == 64, "slot");

/* w-tri: the MoE router gate's logits of a prompt chunk (aux_gate.hip, tf_ds_aux_gate_logits_d5120). Device addresses:
 * x bf16 [rows, 5120]; w fp16 [ne, 5120] (the weight as stored); out fp32 [rows, ne] = x . w^T; flags 0. Two builds:
 * tf_ds_aux_gate_logits_d5120 for ne a multiple of 128 (384 routed experts, the drafter's 128): grid (ceil(rows / 64),
 * ne / 128, 1), block 256; tf_ds_aux_gate_logits_n32_d5120 for ne = 32 (the indexer's weights_proj): grid
 * (ceil(rows / 32), 1, 1), block 128. */
typedef struct {
    uint64_t x, w, out;
    int32_t rows, ne, flags, pad0;
} tf_ds_aux_gate_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_gate_args) == 40, "tf_ds_aux_gate_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, x) == 0, "x");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, w) == 8, "w");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, out) == 16, "out");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, rows) == 24, "rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, ne) == 28, "ne");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, flags) == 32, "flags");
TF_DS_ASSERT(offsetof(tf_ds_aux_gate_args, pad0) == 36, "pad0");

/* w-tri: the attention heads' RoPE (aux_rope.hip, tf_ds_aux_rope_heads), _rope_heads' bits. Device addresses: x bf16
 * [rows, h, hd] (rotated in place: the last rd dims of each head as pairs); cos, sin fp32 [positions, rd / 2]; pos
 * int64 [rows]. rd a multiple of 8 (64); flags bit 0: inverse. Grid (ceil(rows h rd / 8 / 256), 1, 1), block 256. */
typedef struct {
    uint64_t x, cos, sin, pos;
    int32_t rows, h, hd, rd, flags, pad0;
} tf_ds_aux_rope_heads_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_rope_heads_args) == 56, "tf_ds_aux_rope_heads_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, x) == 0, "x");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, cos) == 8, "cos");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, sin) == 16, "sin");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, pos) == 24, "pos");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, rows) == 32, "rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, h) == 36, "h");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, hd) == 40, "hd");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, rd) == 44, "rd");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, flags) == 48, "flags");
TF_DS_ASSERT(offsetof(tf_ds_aux_rope_heads_args, pad0) == 52, "pad0");

/* exl3.l2_prefetch's first parameter (ops.zig Prefetch, the CUDA kernel's TfDsPrefetch): up to 16 device ranges, each
 * its first byte and length; the kernel's other two parameters are int64 delay_ns and wave_ns. */
typedef struct {
    uint64_t ptr[16];
    uint64_t bytes[16];
    int32_t n;
    int32_t pad0;
} tf_ds_prefetch;
TF_DS_ASSERT(sizeof(tf_ds_prefetch) == 264, "tf_ds_prefetch size");
TF_DS_ASSERT(offsetof(tf_ds_prefetch, ptr) == 0, "ptr");
TF_DS_ASSERT(offsetof(tf_ds_prefetch, bytes) == 128, "bytes");
TF_DS_ASSERT(offsetof(tf_ds_prefetch, n) == 256, "n");
TF_DS_ASSERT(offsetof(tf_ds_prefetch, pad0) == 260, "pad0");

/* The drafter's sampled Markov step (aux_markov.hip, tf_ds_aux_markov_race). A row's key: its stream's sampling seed,
 * the absolute position its first draft is drawn at (the kernel adds its step) and 1 / the draft temperature (0: a
 * greedy row, no draw). */
typedef struct {
    uint64_t seed, position;
    float inv_t;
    uint32_t pad0;
} tf_ds_markov_key;
TF_DS_ASSERT(sizeof(tf_ds_markov_key) == 24, "tf_ds_markov_key size");
TF_DS_ASSERT(offsetof(tf_ds_markov_key, seed) == 0, "seed");
TF_DS_ASSERT(offsetof(tf_ds_markov_key, position) == 8, "position");
TF_DS_ASSERT(offsetof(tf_ds_markov_key, inv_t) == 16, "inv_t");
TF_DS_ASSERT(offsetof(tf_ds_markov_key, pad0) == 20, "pad0");

/* Device addresses: lg fp32 (row r's n_cols logits at lg + 4 r l_row: this rank's columns of the step's block row);
 * tok int64 (row r's previous token at tok[r t_stride]); slot int32 [vocabulary] (a token's cached bias row, -1
 * none); cache fp16 [cached rows, c_row]; stage fp16 [rows, s_row] (the bias rows of tokens without a slot); keys
 * tf_ds_markov_key [rows]; send fp32 [rows, 4] (value, token id, 0, 0). id0: the token id of column 0. Grid
 * (n_rows, 1, 1), block 1024. */
typedef struct {
    uint64_t lg, tok, slot, cache, stage, keys, send;
    int64_t l_row, t_stride, c_row, s_row;
    int32_t n_rows, n_cols, id0, step;
} tf_ds_aux_markov_race_args;
TF_DS_ASSERT(sizeof(tf_ds_aux_markov_race_args) == 104, "tf_ds_aux_markov_race_args size");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, lg) == 0, "lg");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, tok) == 8, "tok");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, slot) == 16, "slot");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, cache) == 24, "cache");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, stage) == 32, "stage");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, keys) == 40, "keys");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, send) == 48, "send");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, l_row) == 56, "l_row");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, t_stride) == 64, "t_stride");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, c_row) == 72, "c_row");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, s_row) == 80, "s_row");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, n_rows) == 88, "n_rows");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, n_cols) == 92, "n_cols");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, id0) == 96, "id0");
TF_DS_ASSERT(offsetof(tf_ds_aux_markov_race_args, step) == 100, "step");

#endif

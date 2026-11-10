#pragma clang fp contract(off)
/* The DeepSeek-V4.1 vision tower and aligner on gfx1151 (registered in registry.json as group "vision"; vit_hip.zig
 * launches them, hip_load.zig binds them; only a rank serving MODALITIES=text_image needs them).
 *
 * Adapted from engineLab's HIP tower (src/hip_ds41_vision.hip, include/strix/ds41_vision_math.hpp and the
 * declared_exp / declared_sqrt of include/strix/ds41_moe.hpp, branch pr/ds41-vision at 481141258, Apache-2.0, by the
 * same owner), arithmetic unchanged, so every step's bytes equal engineLab's host reference (encode_reference) and its
 * device tower: BF16 weights and activations, every FP32 operation a single IEEE operation in a declared order, no
 * contraction (the pragma above and the kit's -ffp-contract=off), no fast math, no library transcendental:
 *  - linear: y(m, o) = bf16(acc + bf16 bias) with acc the FP32 fma chain over k ascending from +0 (tiles of 16 k in
 *    order; tails padded with zeros, which leave the accumulator unchanged: it starts at +0 and never becomes -0);
 *  - RMSNorm: one wave per row; lane l sums fma(x, x) over k = l, l + 32, ...; the lanes combine by the butterfly
 *    v + v[l ^ d], d = 16 .. 1; out = bf16(w * (x * (1 / sqrt(ss / D + eps)))) with a correctly rounded sqrt;
 *  - attention: per (64 query rows, head) an online softmax over key tiles of 64 (declared_exp, 16-lane reductions),
 *    scores = (fma chain over d) * 0.125, output bf16(o / l);
 *  - rotary (rotate-half, tables built on the host in double precision), residual add, silu * up, erf-GELU (the
 *    fdlibm/musl erff approximations with declared_exp), the 3 x 3 unfold and the span: their host forms are the same
 *    expressions.
 * declared_erf follows fdlibm/musl erff (fdlibm: Copyright (C) 1993 by Sun Microsystems, Inc.; permission to use,
 * copy, modify and distribute this software is freely granted, provided that this notice is preserved). */
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "abi.h"

#define TF_DS_VISION_THREADS 256

static __device__ __forceinline__ uint16_t tf_ds_v_bf16_round(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16u) | 0x40u);
    u += 0x7FFFu + ((u >> 16u) & 1u);
    return (uint16_t)(u >> 16u);
}
static __device__ __forceinline__ float tf_ds_v_bf16_float(uint16_t b) { return __uint_as_float((uint32_t)b << 16u); }

static __device__ __forceinline__ float tf_ds_v_pow2(int k) { return __uint_as_float((uint32_t)(k + 127) << 23u); }

// exp(x) from basic IEEE operations: Cody-Waite reduction by ln 2, degree-7 Taylor polynomial (Horner), two-step
// power-of-two scaling; below -103 0, above 88 +inf.
static __device__ __forceinline__ float tf_ds_v_exp(float x) {
    if (!(x == x)) return x;
    if (x > 88.0f) return __builtin_huge_valf();
    if (x < -103.0f) return 0.0f;
    const float n = __builtin_rintf(x * 1.44269504f);
    const float r = (x - n * 0.693359375f) - n * -2.12194440e-4f;
    float p = 1.98412698e-4f;
    p = p * r + 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 0.5f;
    p = p * r + 1.0f;
    p = p * r + 1.0f;
    const int k = (int)n, k1 = k / 2, k2 = k - k1;
    return (p * tf_ds_v_pow2(k1)) * tf_ds_v_pow2(k2);
}

// Correctly rounded float square root: the float rounding of the correctly rounded double root.
static __device__ __forceinline__ float tf_ds_v_sqrt(float x) { return (float)__dsqrt_rn((double)x); }

static __device__ __forceinline__ float tf_ds_v_erf(float x) {
    const uint32_t bits = __float_as_uint(x);
    const bool negative = (bits >> 31u) != 0u;
    const uint32_t ix = bits & 0x7FFFFFFFu;
    if (ix > 0x7F800000u) return x;
    if (ix == 0x7F800000u) return negative ? -1.0f : 1.0f;
    if (ix < 0x3F580000u) {  // |x| < 0.84375
        if (ix < 0x31800000u) return 0.125f * (8.0f * x + 1.0270333290e+00f * x);
        const float z = x * x;
        const float r = 1.2837916613e-01f + z * (-3.2504209876e-01f + z * (-2.8481749818e-02f + z * (-5.7702702470e-03f + z * -2.3763017452e-05f)));
        const float s = 1.0f + z * (3.9791721106e-01f + z * (6.5022252500e-02f + z * (5.0813062117e-03f + z * (1.3249473704e-04f + z * -3.9602282413e-06f))));
        const float y = r / s;
        return x + x * y;
    }
    float y;
    const float ax = negative ? -x : x;
    if (ix < 0x3FA00000u) {  // |x| < 1.25
        const float s = ax - 1.0f;
        const float P = -2.3621185683e-03f + s * (4.1485610604e-01f + s * (-3.7220788002e-01f + s * (3.1834661961e-01f + s * (-1.1089469492e-01f + s * (3.5478305072e-02f + s * -2.1663755178e-03f)))));
        const float Q = 1.0f + s * (1.0642088205e-01f + s * (5.4039794207e-01f + s * (7.1828655899e-02f + s * (1.2617121637e-01f + s * (1.3637083583e-02f + s * 1.1984500103e-02f)))));
        y = 8.4506291151e-01f + P / Q;
    } else if (ix < 0x40C00000u) {  // |x| < 6
        const float s = 1.0f / (ax * ax);
        float R, S;
        if (ix < 0x4036DB6Du) {  // |x| < 1/0.35
            R = -9.8649440333e-03f + s * (-6.9385856390e-01f + s * (-1.0558626175e+01f + s * (-6.2375331879e+01f + s * (-1.6239666748e+02f + s * (-1.8460508728e+02f + s * (-8.1287437439e+01f + s * -9.8143291473e+00f))))));
            S = 1.0f + s * (1.9651271820e+01f + s * (1.3765776062e+02f + s * (4.3456588745e+02f + s * (6.4538726807e+02f + s * (4.2900814819e+02f + s * (1.0863500214e+02f + s * (6.5702495575e+00f + s * -6.0424413532e-02f)))))));
        } else {
            R = -9.8649431020e-03f + s * (-7.9928326607e-01f + s * (-1.7757955551e+01f + s * (-1.6063638306e+02f + s * (-6.3756646729e+02f + s * (-1.0250950928e+03f + s * -4.8351919556e+02f)))));
            S = 1.0f + s * (3.0338060379e+01f + s * (3.2579251099e+02f + s * (1.5367296143e+03f + s * (3.1998581543e+03f + s * (2.5530502930e+03f + s * (4.7452853394e+02f + s * -2.2440952301e+01f))))));
        }
        const float z = __uint_as_float(__float_as_uint(ax) & 0xFFFFE000u);
        const float erfc = tf_ds_v_exp(-z * z - 0.5625f) * tf_ds_v_exp((z - ax) * (z + ax) + R / S) / ax;
        y = 1.0f - erfc;
    } else {
        y = 1.0f;
    }
    return negative ? -y : y;
}

// GELU (exact form, torch's CUDA order): (x * 0.5) * (1 + erf(x * sqrt(1/2))).
static __device__ __forceinline__ float tf_ds_v_gelu(float x) { return (x * 0.5f) * (1.0f + tf_ds_v_erf(x * 0.70710678118654752f)); }
// bf16(bf16(silu(gate)) * up), silu(g) = g / (1 + exp(-g)).
static __device__ __forceinline__ uint16_t tf_ds_v_silu_mul(uint16_t gate, uint16_t up) {
    const float g = tf_ds_v_bf16_float(gate);
    return tf_ds_v_bf16_round(tf_ds_v_bf16_float(tf_ds_v_bf16_round(g / (1.0f + tf_ds_v_exp(-g)))) * tf_ds_v_bf16_float(up));
}

/* ---- linear: y [M, O] bf16 = bf16(x [M, K] bf16 . w [O, K]^T (+ bias [O] bf16)), all rows contiguous ---- */
#define TF_DS_VISION_BM 128
#define TF_DS_VISION_BO 128
#define TF_DS_VISION_BK 16
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_linear(const uint16_t* x, const uint16_t* w, const uint16_t* bias, uint16_t* y, int32_t M, int32_t O, int32_t K) {
    __shared__ float As[TF_DS_VISION_BK][TF_DS_VISION_BM + 4];
    __shared__ float Bs[TF_DS_VISION_BK][TF_DS_VISION_BO + 4];
    const int tx = (int)(threadIdx.x % 16u), ty = (int)(threadIdx.x / 16u);
    const int o0 = (int)blockIdx.x * TF_DS_VISION_BO, m0 = (int)blockIdx.y * TF_DS_VISION_BM;
    float acc[8][8];
    for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TF_DS_VISION_BK) {
        for (int e = (int)threadIdx.x; e < TF_DS_VISION_BM * TF_DS_VISION_BK; e += TF_DS_VISION_THREADS) {
            const int mi = e / TF_DS_VISION_BK, kk = e % TF_DS_VISION_BK, m = m0 + mi, k = k0 + kk;
            As[kk][mi] = (m < M && k < K) ? tf_ds_v_bf16_float(x[(int64_t)m * K + k]) : 0.0f;
        }
        for (int e = (int)threadIdx.x; e < TF_DS_VISION_BO * TF_DS_VISION_BK; e += TF_DS_VISION_THREADS) {
            const int oi = e / TF_DS_VISION_BK, kk = e % TF_DS_VISION_BK, o = o0 + oi, k = k0 + kk;
            Bs[kk][oi] = (o < O && k < K) ? tf_ds_v_bf16_float(w[(int64_t)o * K + k]) : 0.0f;
        }
        __syncthreads();
        for (int kk = 0; kk < TF_DS_VISION_BK; ++kk) {
            float av[8], bv[8];
            for (int i = 0; i < 8; ++i) av[i] = As[kk][ty + 16 * i];
            for (int j = 0; j < 8; ++j) bv[j] = Bs[kk][tx + 16 * j];
            for (int i = 0; i < 8; ++i)
                for (int j = 0; j < 8; ++j) acc[i][j] = __builtin_fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 8; ++i) {
        const int m = m0 + ty + 16 * i;
        if (m >= M) continue;
        for (int j = 0; j < 8; ++j) {
            const int o = o0 + tx + 16 * j;
            if (o >= O) continue;
            const float v = bias ? acc[i][j] + tf_ds_v_bf16_float(bias[o]) : acc[i][j];
            y[(int64_t)m * O + o] = tf_ds_v_bf16_round(v);
        }
    }
}

static __device__ __forceinline__ float tf_ds_v_wave_sum(float v) {
    for (int d = 16; d != 0; d /= 2) v = v + __shfl_xor(v, d, 32);
    return v;
}

/* ---- RMSNorm: out [rows, D] = bf16(w * (x * r)), one wave per row (8 rows a block) ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_rms_norm(const uint16_t* x, const uint16_t* w, uint16_t* out, int32_t rows, int32_t D, float eps) {
    const int row = (int)(blockIdx.x * (TF_DS_VISION_THREADS / 32u) + threadIdx.x / 32u), lane = (int)(threadIdx.x % 32u);
    if (row >= rows) return;
    const uint16_t* xr = x + (int64_t)row * D;
    float part = 0.0f;
    for (int k = lane; k < D; k += 32) {
        const float v = tf_ds_v_bf16_float(xr[k]);
        part = __builtin_fmaf(v, v, part);
    }
    const float r = 1.0f / tf_ds_v_sqrt(tf_ds_v_wave_sum(part) / (float)D + eps);
    for (int k = lane; k < D; k += 32)
        out[(int64_t)row * D + k] = tf_ds_v_bf16_round(tf_ds_v_bf16_float(w[k]) * (tf_ds_v_bf16_float(xr[k]) * r));
}

/* ---- attention: attn [N, D] bf16 of qkv [N, 3D] bf16 (q | k | v, 16 heads of 64), one block per (64 rows, head) ----
 * Thread t = 16 tr + tc owns rows 4 tr .. 4 tr + 3 with keys 4 tc .. 4 tc + 3 (scores) and dims 4 tc .. 4 tc + 3
 * (output); a row's running max, rescale and sum live identically on the 16 lanes of a half wave (xor 8, 4, 2, 1). */
#define TF_DS_VISION_QT 64
#define TF_DS_VISION_KB 64
#define TF_DS_VISION_HD 64
static __device__ __forceinline__ float tf_ds_v_half_max(float v) {
    for (int d = 8; d != 0; d /= 2) v = fmaxf(v, __shfl_xor(v, d, 32));
    return v;
}
static __device__ __forceinline__ float tf_ds_v_half_sum(float v) {
    for (int d = 8; d != 0; d /= 2) v = v + __shfl_xor(v, d, 32);
    return v;
}
static __device__ __forceinline__ void tf_ds_v_unpack(uint2 w, float* f) {
    f[0] = __uint_as_float(w.x << 16u);
    f[1] = __uint_as_float(w.x & 0xFFFF0000u);
    f[2] = __uint_as_float(w.y << 16u);
    f[3] = __uint_as_float(w.y & 0xFFFF0000u);
}
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_attention(const uint16_t* qkv, uint16_t* attn, int32_t N, int32_t D) {
    constexpr int kS = 68, kQT = TF_DS_VISION_QT, kKB = TF_DS_VISION_KB, kHD = TF_DS_VISION_HD, kThreads = TF_DS_VISION_THREADS;
    __shared__ __attribute__((aligned(16))) float Qs[kHD][kS];  // [d][row]
    __shared__ __attribute__((aligned(16))) float KP[kHD][kS];  // [d][key] for the scores, then [key][row] for P
    __shared__ __attribute__((aligned(16))) float Vs[kKB][kS];  // [key][d]
    const int hh = (int)blockIdx.y, q0 = (int)blockIdx.x * kQT;
    const int t = (int)threadIdx.x, tr = t / 16, tc = t % 16;
    const int64_t row3 = 3LL * D;
    for (int e = t; e < kQT * kHD / 4; e += kThreads) {
        const int r = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
        float f[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (q0 + r < N) tf_ds_v_unpack(*reinterpret_cast<const uint2*>(qkv + (q0 + r) * row3 + hh * kHD + dq), f);
        for (int i = 0; i < 4; ++i) Qs[dq + i][r] = f[i];
    }
    float o[4][4], m[4], lsum[4];
    for (int a = 0; a < 4; ++a) {
        m[a] = -__builtin_huge_valf();
        lsum[a] = 0.0f;
        for (int b = 0; b < 4; ++b) o[a][b] = 0.0f;
    }
    // the next key tile's K and V are fetched into registers while the current tile is computed
    constexpr int kLoads = kKB * kHD / 4 / kThreads;
    static_assert(kKB * kHD / 4 % kThreads == 0, "tile load split");
    uint2 nk[kLoads], nv[kLoads];
    auto fetch = [&](int jstart) {
        const int kc = min(kKB, N - jstart);
        for (int u = 0; u < kLoads; ++u) {
            const int e = t + u * kThreads, jj = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
            nk[u] = make_uint2(0u, 0u);
            nv[u] = make_uint2(0u, 0u);
            if (jj < kc) {
                const uint16_t* src = qkv + (jstart + jj) * row3 + hh * kHD + dq;
                nk[u] = *reinterpret_cast<const uint2*>(src + D);
                nv[u] = *reinterpret_cast<const uint2*>(src + 2LL * D);
            }
        }
    };
    fetch(0);
    for (int j0 = 0; j0 < N; j0 += kKB) {
        const int kn = min(kKB, N - j0);
        __syncthreads();  // the previous block's P and V reads are done
        for (int u = 0; u < kLoads; ++u) {
            const int e = t + u * kThreads, jj = e / (kHD / 4), dq = (e % (kHD / 4)) * 4;
            float fk[4], fv[4];
            tf_ds_v_unpack(nk[u], fk);
            tf_ds_v_unpack(nv[u], fv);
            for (int i = 0; i < 4; ++i) KP[dq + i][jj] = fk[i];
            *reinterpret_cast<float4*>(&Vs[jj][dq]) = make_float4(fv[0], fv[1], fv[2], fv[3]);
        }
        __syncthreads();
        if (j0 + kKB < N) fetch(j0 + kKB);
        // scores: s = (sum over d ascending of fma(q, k)) * 0.125
        float acc[4][4];
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) acc[a][b] = 0.0f;
#pragma unroll 8
        for (int d = 0; d < kHD; ++d) {
            const float4 qv = *reinterpret_cast<const float4*>(&Qs[d][4 * tr]);
            const float4 kv = *reinterpret_cast<const float4*>(&KP[d][4 * tc]);
            const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, kb[4] = {kv.x, kv.y, kv.z, kv.w};
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) acc[a][b] = __builtin_fmaf(qa[a], kb[b], acc[a][b]);
        }
        __syncthreads();  // every thread is done with the K tile; KP becomes P
        // per row: block max over valid keys, m' = max(m, max), a = exp(m - m'), p = exp(s - m'), l = l * a + block sum
        float al[4];
        for (int a = 0; a < 4; ++a) {
            float s[4], mb = -__builtin_huge_valf();
            for (int b = 0; b < 4; ++b) {
                s[b] = acc[a][b] * 0.125f;
                if (4 * tc + b < kn) mb = fmaxf(mb, s[b]);
            }
            mb = tf_ds_v_half_max(mb);
            const float mn = fmaxf(m[a], mb);
            al[a] = tf_ds_v_exp(m[a] - mn);
            m[a] = mn;
            float p[4];
            for (int b = 0; b < 4; ++b) p[b] = 4 * tc + b < kn ? tf_ds_v_exp(s[b] - mn) : 0.0f;
            const float g = tf_ds_v_half_sum(((p[0] + p[1]) + p[2]) + p[3]);
            lsum[a] = lsum[a] * al[a];
            lsum[a] = lsum[a] + g;
            for (int b = 0; b < 4; ++b) KP[4 * tc + b][4 * tr + a] = p[b];
        }
        __syncthreads();
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b) o[a][b] = o[a][b] * al[a];
#pragma unroll 8
        for (int jj = 0; jj < kn; ++jj) {
            const float4 pv = *reinterpret_cast<const float4*>(&KP[jj][4 * tr]);
            const float4 vv = *reinterpret_cast<const float4*>(&Vs[jj][4 * tc]);
            const float pa[4] = {pv.x, pv.y, pv.z, pv.w}, vb[4] = {vv.x, vv.y, vv.z, vv.w};
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) o[a][b] = __builtin_fmaf(pa[a], vb[b], o[a][b]);
        }
    }
    for (int a = 0; a < 4; ++a) {
        const int i = q0 + 4 * tr + a;
        if (i >= N) continue;
        for (int b = 0; b < 4; ++b) attn[(int64_t)i * D + hh * kHD + 4 * tc + b] = tf_ds_v_bf16_round(o[a][b] / lsum[a]);
    }
}

/* ---- rotary on q and k in place (rotate-half within each head): one thread per (row, q|k, head, pair) ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_rope(uint16_t* qkv, const float* cosv, const float* sinv, int32_t N, int32_t D, int32_t heads, int32_t hd) {
    const int half = hd / 2;
    const int64_t idx = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    const int64_t total = (int64_t)N * 2 * heads * half;
    if (idx >= total) return;
    const int j = (int)(idx % half), hh = (int)((idx / half) % heads), part = (int)((idx / half / heads) % 2);
    const int n = (int)(idx / half / heads / 2);
    uint16_t* v = qkv + (int64_t)n * 3 * D + (int64_t)part * D + (int64_t)hh * hd;
    const float x1 = tf_ds_v_bf16_float(v[j]), x2 = tf_ds_v_bf16_float(v[j + half]);
    const float c = cosv[(int64_t)n * half + j], s = sinv[(int64_t)n * half + j];
    const float a = x1 * c, b = x2 * s, d = x2 * c, e = x1 * s;
    v[j] = tf_ds_v_bf16_round(a - b);
    v[j + half] = tf_ds_v_bf16_round(d + e);
}

/* ---- x = bf16(x + y) ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS) tf_ds_vision_add(uint16_t* x, const uint16_t* y, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    if (i < n) x[i] = tf_ds_v_bf16_round(tf_ds_v_bf16_float(x[i]) + tf_ds_v_bf16_float(y[i]));
}

/* ---- mid [N, I] = silu(gate) * up of gu [N, 2I] (gate first) ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS) tf_ds_vision_silu_mul(const uint16_t* gu, uint16_t* mid, int32_t N, int32_t I) {
    const int64_t i = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    if (i >= (int64_t)N * I) return;
    const int64_t n = i / I, k = i % I;
    mid[i] = tf_ds_v_silu_mul(gu[n * 2 * I + k], gu[n * 2 * I + I + k]);
}

/* ---- v = bf16(gelu(v)) ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS) tf_ds_vision_gelu(uint16_t* v, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    if (i < n) v[i] = tf_ds_v_bf16_round(tf_ds_v_gelu(tf_ds_v_bf16_float(v[i])));
}

/* ---- M x M unfold: feature ch * M * M + ki * M + kj of cell (lh, lw) = patch (M lh + ki, M lw + kj), 0 past the grid;
 * unf [llm_h * llm_w, D * M * M] row-major ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_unfold(const uint16_t* h, uint16_t* unf, int32_t vit_w, int32_t vit_h, int32_t llm_w, int32_t llm_h, int32_t D, int32_t M) {
    const int64_t F = (int64_t)D * M * M;
    const int64_t i = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    if (i >= (int64_t)llm_w * llm_h * F) return;
    const int64_t cell = i / F, f = i % F;
    const int ch = (int)(f / (M * M)), ki = (int)((f / M) % M), kj = (int)(f % M);
    const int lh = (int)(cell / llm_w), lw = (int)(cell % llm_w);
    const int py = lh * M + ki, px = lw * M + kj;
    unf[i] = (py < vit_h && px < vit_w) ? h[((int64_t)py * vit_w + px) * D + ch] : (uint16_t)0;
}

/* ---- the span [llm_h * (llm_w + 1) + 2, D]: START, (each row of cells, NEWLINE), END ---- */
extern "C" __global__ void __launch_bounds__(TF_DS_VISION_THREADS)
tf_ds_vision_span(const uint16_t* rows, const uint16_t* start, const uint16_t* newline, const uint16_t* end, uint16_t* out, int32_t llm_w,
                  int32_t llm_h, int32_t D) {
    const int span = llm_h * (llm_w + 1) + 2;
    const int64_t i = (int64_t)blockIdx.x * TF_DS_VISION_THREADS + threadIdx.x;
    if (i >= (int64_t)span * D) return;
    const int t = (int)(i / D), c = (int)(i % D);
    uint16_t v;
    if (t == 0) v = start[c];
    else if (t == span - 1) v = end[c];
    else {
        const int k = t - 1, row = k / (llm_w + 1), col = k % (llm_w + 1);
        v = col == llm_w ? newline[c] : rows[((int64_t)row * llm_w + col) * D + c];
    }
    out[i] = v;
}

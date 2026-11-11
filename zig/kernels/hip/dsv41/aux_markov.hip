/* The drafter's Markov step for a sampled stream on gfx1151, in place of the Triton _score and _finish (kernels.py at
 * 222a9c0b: markov.py's _score, _finish). ADAPTER.md section 6: group "aux", one by-value struct
 * (tf_ds_aux_markov_race_args, abi.h), registered in registry.json.
 *
 * Why: a sampled request's verify row draws its token as the best of logit / temperature + g(seed, position, token)
 * over the target's top_k (sampling.zig: g = -log(-log(u)), u a splitmix64 hash of the key), while the Triton step
 * drafts the drafter's single best token whatever the request's sampling is. A greedy draft lands with the target's
 * probability of that one token. Here the drafter draws with the same g the verify row will use, on its own scores
 * over a draft temperature, so where the drafter's distribution is near the target's both pick the same token.
 * The drafts stay proposals: the verify rows draw as before, so the reply's tokens do not depend on this kernel.
 *
 * What it computes, for each row r (a stream's step) of this rank's n_cols vocabulary columns: score[c] =
 * lg[r, c] + fp32(bias[c]) (one fp32 add, as _score; the bias the previous token's cached row, else its staging
 * row), v[c] = score[c] * inv_t + g(seed, position + step, id0 + c) when the row's key has inv_t > 0, else score[c]
 * (a greedy stream in a pass with sampled ones: _score's pick); the best v in _score's order (a NaN above
 * everything, then the larger, ties to the lower token id: so a greedy row's pick is _score's and _finish's bit for
 * bit) as (value, token id, 0, 0) fp32 into send[r], what _finish hands the ranks' gather and _pick reads.
 * g is fp32: the host's u (the double nearest ((bits >> 11) + 0.5) 2^-53), -log(u) from u below a half and from
 * 1 - u above it, within a few fp32 ulps of the host's double (a difference decides only near ties).
 * Work: a row a block (1024 threads, 32 waves), a thread every 1024th column. */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "abi.h"

#define TF_GLOBAL __attribute__((address_space(1)))

namespace {
template <class T> __device__ inline TF_GLOBAL T* gp(uint64_t a) { return (TF_GLOBAL T*)a; }

// sampling.zig's mix (splitmix64's finalizer)
__device__ inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// -log(-log(u)) of the key bits' uniform (sampling.zig's uniform: the double nearest ((bits >> 11) + 0.5) 2^-53).
// Below a half that double is exact; from a half on it is (a + (a & 1)) 2^-53 (a tie, to even), and -log(u) comes
// from 1 - u. (tri_markov.raceGumbel is this function on the host.)
__device__ inline float gumbel(uint64_t bits) {
    const uint64_t a = bits >> 11;
    float e;                                                   // -log(u)
    if (a < (1ull << 52)) e = -logf(((float)a + 0.5f) * 0x1p-53f);
    else e = -log1pf(-((float)((1ull << 53) - a - (a & 1ull)) * 0x1p-53f));
    return -logf(e);
}

// _score's order (kernels.py _better, torch.argmax's): a NaN above everything, then the larger value, ties (and a NaN
// against a NaN) to the lower token
__device__ inline bool better(float v, int i, float bv, int bi) {
    const bool n1 = v != v, n2 = bv != bv;
    return (n1 && (!n2 || i < bi)) || (!n1 && !n2 && (v > bv || (v == bv && i < bi)));
}
}  // namespace

extern "C" __global__ __launch_bounds__(1024) void tf_ds_aux_markov_race(tf_ds_aux_markov_race_args a) {
    const int r = blockIdx.x;
    if (r >= a.n_rows) return;
    const float ninf = -__int_as_float(0x7f800000);
    const int64_t tok = gp<const int64_t>(a.tok)[(int64_t)r * a.t_stride];
    const int32_t slot = gp<const int32_t>(a.slot)[tok];
    const TF_GLOBAL uint16_t* bias =
        slot >= 0 ? gp<const uint16_t>(a.cache) + (int64_t)slot * a.c_row : gp<const uint16_t>(a.stage) + (int64_t)r * a.s_row;
    const TF_GLOBAL float* lg = gp<const float>(a.lg) + (int64_t)r * a.l_row;
    const TF_GLOBAL tf_ds_markov_key* kp = gp<const tf_ds_markov_key>(a.keys) + r;
    const float inv_t = kp->inv_t;
    const bool sampled = inv_t > 0.0f;
    const uint64_t h = mix64(mix64(kp->seed + 0x9E3779B97F4A7C15ull) ^ ((kp->position + (uint64_t)a.step) * 0xD1B54A32D192ED03ull));
    float bv = ninf;
    int bi = 0x7fffffff;
    for (int c = threadIdx.x; c < a.n_cols; c += blockDim.x) {
        const float score = lg[c] + __half2float(__ushort_as_half(bias[c]));
        const int id = a.id0 + c;
        const float v = sampled ? score * inv_t + gumbel(mix64(h ^ (uint64_t)id)) : score;
        if (better(v, id, bv, bi)) {
            bv = v;
            bi = id;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down(bv, off, 32);
        const int oi = __shfl_down(bi, off, 32);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sv[warp] = bv;
        si[warp] = bi;
    }
    __syncthreads();
    if (warp != 0) return;
    const int nw = (blockDim.x + 31) >> 5;
    bv = lane < nw ? sv[lane] : ninf;
    bi = lane < nw ? si[lane] : 0x7fffffff;
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down(bv, off, 32);
        const int oi = __shfl_down(bi, off, 32);
        if (better(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        TF_GLOBAL float* send = gp<float>(a.send) + 4 * (int64_t)r;
        send[0] = bv;
        send[1] = (float)(bi == 0x7fffffff ? a.id0 : bi);     // (no columns: this rank's first token, at -inf)
        send[2] = 0.0f;
        send[3] = 0.0f;
    }
}

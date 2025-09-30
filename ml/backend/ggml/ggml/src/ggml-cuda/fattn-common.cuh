#pragma once

#include "common.cuh"
#include "convert.cuh"
#include "vecdotq.cuh"

#include <cstdint>

#define FATTN_KQ_STRIDE       256
#define HALF_MAX_HALF         __float2half(65504.0f/2) // Use neg. of this instead of -INFINITY to initialize KQ max vals to avoid NaN upon subtraction.
#define SOFTMAX_FTZ_THRESHOLD -20.0f                   // Softmax exp. of values smaller than this are flushed to zero to avoid NaNs.

typedef void (* fattn_kernel_t)(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const int32_t ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33);

typedef half (*vec_dot_KQ_f16_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);
typedef float (*vec_dot_KQ_f32_t)(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds);

template<typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_0 * K_q4_0 = (const block_q4_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_0;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_b2(K_q4_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q4_0[ib].d) * Q_ds[k_KQ_0/warp_size];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2) /* *8/QI8_1 == 1 */);
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q4_0[ib].d) * (sumi*Q_ds[k_KQ_0/warp_size].x - (8/QI8_1)*Q_ds[k_KQ_0/warp_size].y));
        }
    }

    return sum;
}

template<typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q4_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q4_1 * K_q4_1 = (const block_q4_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI4_1;
        const int shift = k_KQ & (QI8_1/2);

        const int v = (get_int_b4(K_q4_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int u = Q_q8[k_KQ_0/warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d4d8_m4s8 = K_q4_1[ib].dm * Q_ds[k_KQ_0/warp_size];
            const half2 sumid4d8_m4s8scaled = d4d8_m4s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid4d8_m4s8scaled) + __high2half(sumid4d8_m4s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid4d8   =  __low2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/warp_size].x * sumi;
            const float m4s8scaled = __high2float(K_q4_1[ib].dm)*Q_ds[k_KQ_0/warp_size].y / QI8_1;

            sum += (T) (sumid4d8 + m4s8scaled);
        }
    }

    return sum;
}

template<typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_0 * K_q5_0 = (const block_q5_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_0;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_b2(K_q5_0[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_0[ib].qh, 0) >> (iqs8 * QI5_0);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 sum2 = __half2half2(K_q5_0[ib].d) * Q_ds[k_KQ_0/warp_size];
            sum += (T) (((half) sumi)*__low2half(sum2) - __high2half(sum2)*__float2half(2.0f)) /* *16/QI8_1 == 2 */;
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            sum += (T) (__half2float(K_q5_0[ib].d) * (sumi*Q_ds[k_KQ_0/warp_size].x - (16/QI8_1)*Q_ds[k_KQ_0/warp_size].y));
        }
    }

    return sum;
}

template<typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q5_1(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q5_1 * K_q5_1 = (const block_q5_1 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib    = k_KQ /  QI8_1;
        const int iqs4  = k_KQ %  QI5_1;
        const int iqs8  = k_KQ %  QI8_1;
        const int shift = k_KQ & (QI8_1/2);

        int v = (get_int_b2(K_q5_1[ib].qs, iqs4) >> shift) & 0x0F0F0F0F;
        const int vh = get_int_b2(K_q5_1[ib].qh, 0) >> (iqs8 * QI5_1);
        v |= (vh <<  4) & 0x00000010; // 0 ->  4
        v |= (vh << 11) & 0x00001000; // 1 -> 12
        v |= (vh << 18) & 0x00100000; // 2 -> 20
        v |= (vh << 25) & 0x10000000; // 3 -> 28

        const int u = Q_q8[k_KQ_0/warp_size];

        const int sumi = ggml_cuda_dp4a(v, u, 0);

#ifdef FP16_AVAILABLE
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;

            const half2 d5d8_m5s8 = K_q5_1[ib].dm * Q_ds[k_KQ_0/warp_size];
            const half2 sumid5d8_m5s8scaled = d5d8_m5s8 * make_half2(sumi, 1.0f/QI8_1);
            sum += (T) (__low2half(sumid5d8_m5s8scaled) + __high2half(sumid5d8_m5s8scaled));
        } else
#endif // FP16_AVAILABLE
        {
            const float2 * Q_ds = (const float2 *) Q_ds_v;

            const float sumid5d8   =  __low2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/warp_size].x * sumi;
            const float m5s8scaled = __high2float(K_q5_1[ib].dm)*Q_ds[k_KQ_0/warp_size].y / QI8_1;

            sum += (T) (sumid5d8 + m5s8scaled);
        }
    }

    return sum;
}

template <typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_q8_0(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8, const void * __restrict__ Q_ds_v) {

    const block_q8_0 * K_q8_0 = (const block_q8_0 *) K_c;
    GGML_UNUSED(Q_v);

    T sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < int(D/sizeof(int)); k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const int ib  = k_KQ / QI8_0;
        const int iqs = k_KQ % QI8_0;

        const int v = get_int_b2(K_q8_0[ib].qs, iqs);

        T Q_d;
        if (std::is_same<T, half>::value) {
            const half2  * Q_ds = (const half2  *) Q_ds_v;
            Q_d = __low2half(Q_ds[k_KQ_0/warp_size]);
        } else {
            const float2 * Q_ds = (const float2 *) Q_ds_v;
            Q_d = Q_ds[k_KQ_0/warp_size].x;
        }

        sum += vec_dot_q8_0_q8_1_impl<T, 1>(&v, &Q_q8[k_KQ_0/warp_size], K_q8_0[ib].d, Q_d);
    }

    return sum;
}

template <typename T, int D, int warp_size>
static __device__ __forceinline__ T vec_dot_fattn_vec_KQ_f16(
    const char * __restrict__ K_c, const void * __restrict__ Q_v, const int * __restrict__ Q_q8 , const void * __restrict__ Q_ds_v) {

    const half2 * K_h2 = (const half2 *) K_c;
    GGML_UNUSED(Q_q8);
    GGML_UNUSED(Q_ds_v);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        const half2 * Q_h2 = (const half2 *) Q_v;

        half2 sum2 = make_half2(0.0f, 0.0f);

#pragma unroll
        for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += warp_size) {
            const int k_KQ = k_KQ_0 + threadIdx.x;

            const half2 K_ik = K_h2[k_KQ];
            sum2 += K_ik * Q_h2[k_KQ_0/warp_size];
        }

        return __low2half(sum2) + __high2half(sum2);
    }
#endif // FP16_AVAILABLE

    const float2 * Q_f2 = (const float2 *) Q_v;

    float sum = 0.0f;

#pragma unroll
    for (int k_KQ_0 = 0; k_KQ_0 < D/2; k_KQ_0 += warp_size) {
        const int k_KQ = k_KQ_0 + threadIdx.x;

        const half2 K_ik = K_h2[k_KQ];
        sum +=  __low2float(K_ik) * Q_f2[k_KQ_0/warp_size].x;
        sum += __high2float(K_ik) * Q_f2[k_KQ_0/warp_size].y;
    }

    return sum;
}

template <typename Tds>
static __device__ __forceinline__ void quantize_q8_1_to_shared(
    const float * __restrict__ x, const float scale, int * __restrict__ yq32, void * __restrict__ yds) {

    float vals[sizeof(int)] = {0.0f};
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
        vals[l] = scale * x[4*threadIdx.x + l];
    }

    float amax = fabsf(vals[0]);
    float sum  = vals[0];
#pragma unroll
    for (int l = 1; l < int(sizeof(int)); ++l) {
        amax = fmaxf(amax, fabsf(vals[l]));
        sum += vals[l];
    }
#pragma unroll
    for (int mask = QI8_1/2; mask > 0; mask >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, 32));
        sum +=             __shfl_xor_sync(0xFFFFFFFF, sum,  mask, 32);
    }

    const float d = amax / 127;
    int q32 = 0;
    int8_t * q8 = (int8_t *) &q32;

    if (d != 0.0f) {
#pragma unroll
        for (int l = 0; l < int(sizeof(int)); ++l) {
            q8[l] = roundf(vals[l] / d);
        }
    }

    yq32[threadIdx.x] = q32;
    if (threadIdx.x % QI8_1 == 0) {
        if (std::is_same<Tds, half2>::value) {
            ((half2  *) yds)[threadIdx.x/QI8_1] =  make_half2(d, sum);
        } else {
            ((float2 *) yds)[threadIdx.x/QI8_1] = make_float2(d, sum);
        }
    }
}

typedef half  (*dequantize_1_f16_t)(const void *, const int64_t);
typedef float (*dequantize_1_f32_t)(const void *, const int64_t);

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_0(const void * __restrict__ vx, const int64_t i) {
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const int64_t ib    =  i          /  QK4_0;
    const int     iqs   =  i          % (QK4_0/2);
    const int     shift = (i % QK4_0) / (QK4_0/2);

    const T   d  = x[ib].d;
    const int q0 = x[ib].qs[iqs];
    const int q  = ((q0 >> (4*shift)) & 0x0F) - 8;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q4_1(const void * __restrict__ vx, const int64_t i) {
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const int64_t ib    =  i          /  QK4_1;
    const int     iqs   =  i          % (QK4_1/2);
    const int     shift = (i % QK4_1) / (QK4_1/2);

    const half2 dm = x[ib].dm;
    const int   q0 = x[ib].qs[iqs];
    const int   q  = ((q0 >> (4*shift)) & 0x0F);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_0(const void * __restrict__ vx, const int64_t i) {
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const int64_t ib    =  i          /  QK5_0;
    const int     idq   =  i          %  QK5_0;
    const int     iqs   =  i          % (QK5_0/2);
    const int     shift = (i % QK5_0) / (QK5_0/2);

    const T   d   = x[ib].d;
    const int ql0 = x[ib].qs[iqs];
    const int qh0 = get_int_b2(x[ib].qh, 0);
    const int ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int qh  = ((qh0 >> idq) << 4) & 0x10;
    const int q   = (ql | qh) - 16;

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q5_1(const void * __restrict__ vx, const int64_t i) {
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const int64_t ib    =  i          /  QK5_1;
    const int     idq   =  i          %  QK5_1;
    const int     iqs   =  i          % (QK5_1/2);
    const int     shift = (i % QK5_1) / (QK5_1/2);

    const half2 dm  = x[ib].dm;
    const int   ql0 = x[ib].qs[iqs];
    const int   qh0 = get_int_b4(x[ib].qh, 0);
    const int   ql  = ((ql0 >> (4*shift)) & 0x0F);
    const int   qh  = ((qh0 >> idq) << 4) & 0x10;
    const int   q   = (ql | qh);

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return __low2half(dm)*((half) q) + __high2half(dm);
    }
#endif // FP16_AVAILABLE

    return __low2float(dm)*((float) q) + __high2float(dm);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_q8_0(const void * __restrict__ vx, const int64_t i) {
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const int64_t ib  = i / QK8_0;
    const int     iqs = i % QK8_0;

    const T   d = x[ib].d;
    const int q = x[ib].qs[iqs];

#ifdef FP16_AVAILABLE
    if (std::is_same<T, half>::value) {
        return ((half) d)*((half) q);
    }
#endif // FP16_AVAILABLE

    return ((float) d)*((float) q);
}

template <typename T>
static __device__ __forceinline__ T dequantize_1_f16(const void * __restrict__ vx, const int64_t i) {
    const half * x = (const half *) vx;

    return x[i];
}

template <int D, int warp_size = WARP_SIZE>
constexpr __device__ vec_dot_KQ_f16_t get_vec_dot_KQ_f16(ggml_type type_K) {
    return type_K == GGML_TYPE_Q4_0 ? vec_dot_fattn_vec_KQ_q4_0<half, D, warp_size> :
        type_K == GGML_TYPE_Q4_1 ? vec_dot_fattn_vec_KQ_q4_1<half, D, warp_size> :
        type_K == GGML_TYPE_Q5_0 ? vec_dot_fattn_vec_KQ_q5_0<half, D, warp_size> :
        type_K == GGML_TYPE_Q5_1 ? vec_dot_fattn_vec_KQ_q5_1<half, D, warp_size> :
        type_K == GGML_TYPE_Q8_0 ? vec_dot_fattn_vec_KQ_q8_0<half, D, warp_size> :
        type_K == GGML_TYPE_F16 ? vec_dot_fattn_vec_KQ_f16<half, D, warp_size> :
        nullptr;
}

template <int D, int warp_size = WARP_SIZE>
constexpr __device__ vec_dot_KQ_f32_t get_vec_dot_KQ_f32(ggml_type type_K) {
    return type_K == GGML_TYPE_Q4_0 ? vec_dot_fattn_vec_KQ_q4_0<float, D, warp_size> :
        type_K == GGML_TYPE_Q4_1 ? vec_dot_fattn_vec_KQ_q4_1<float, D, warp_size> :
        type_K == GGML_TYPE_Q5_0 ? vec_dot_fattn_vec_KQ_q5_0<float, D, warp_size> :
        type_K == GGML_TYPE_Q5_1 ? vec_dot_fattn_vec_KQ_q5_1<float, D, warp_size> :
        type_K == GGML_TYPE_Q8_0 ? vec_dot_fattn_vec_KQ_q8_0<float, D, warp_size> :
        type_K == GGML_TYPE_F16 ? vec_dot_fattn_vec_KQ_f16<float, D, warp_size> :
        nullptr;
}

constexpr __device__ dequantize_1_f16_t get_dequantize_1_f16(ggml_type type_V) {
    return type_V == GGML_TYPE_Q4_0 ? dequantize_1_q4_0<half> :
        type_V == GGML_TYPE_Q4_1 ? dequantize_1_q4_1<half> :
        type_V == GGML_TYPE_Q5_0 ? dequantize_1_q5_0<half> :
        type_V == GGML_TYPE_Q5_1 ? dequantize_1_q5_1<half> :
        type_V == GGML_TYPE_Q8_0 ? dequantize_1_q8_0<half> :
        type_V == GGML_TYPE_F16 ? dequantize_1_f16<half> :
        nullptr;
}

constexpr __device__ dequantize_1_f32_t get_dequantize_1_f32(ggml_type type_V) {
    return type_V == GGML_TYPE_Q4_0 ? dequantize_1_q4_0<float> :
        type_V == GGML_TYPE_Q4_1 ? dequantize_1_q4_1<float> :
        type_V == GGML_TYPE_Q5_0 ? dequantize_1_q5_0<float> :
        type_V == GGML_TYPE_Q5_1 ? dequantize_1_q5_1<float> :
        type_V == GGML_TYPE_Q8_0 ? dequantize_1_q8_0<float> :
        type_V == GGML_TYPE_F16 ? dequantize_1_f16<float> :
        nullptr;
}

template <int ncols1>
__launch_bounds__(FATTN_KQ_STRIDE/2, 1)
static __global__ void flash_attn_mask_to_KV_max(
        const half2 * __restrict__ mask, int * __restrict__ KV_max, const int ne30, const int s31, const int s33) {
    const int ne31     = gridDim.x;
    const int tid      = threadIdx.x;
    const int sequence = blockIdx.y;
    const int jt       = blockIdx.x;

    mask += sequence*s33 + jt*ncols1*s31;

    __shared__ int buf_iw[WARP_SIZE];
    if (tid < WARP_SIZE) {
        buf_iw[tid] = 1;
    }
    __syncthreads();

    int KV_max_sj = (ne30 - 1) * FATTN_KQ_STRIDE;
    for (; KV_max_sj >= 0; KV_max_sj -= FATTN_KQ_STRIDE) {
        int all_inf = 1;

#pragma unroll
        for (int j = 0; j < ncols1; ++j) {
            const float2 tmp = __half22float2(mask[j*s31 + KV_max_sj/2 + tid]);
            all_inf = all_inf && int(isinf(tmp.x)) && int(isinf(tmp.y));
        }

        all_inf = warp_reduce_all(all_inf);
        if (tid % WARP_SIZE == 0) {
            buf_iw[tid / WARP_SIZE] = all_inf;
        }
        __syncthreads();
        all_inf = buf_iw[tid % WARP_SIZE];
        __syncthreads();
        all_inf = warp_reduce_all(all_inf);

        if (!all_inf) {
            KV_max_sj += FATTN_KQ_STRIDE;
            break;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    KV_max[sequence*ne31 + jt] = KV_max_sj;
}

template<int D, int ncols1, int ncols2> // D == head size
__launch_bounds__(D, 1)
static __global__ void flash_attn_stream_k_fixup(
        float * __restrict__ dst, const float2 * __restrict__ dst_fixup, const int ne01, const int ne02, const int ne03, const int ne11) {
    constexpr int ncols = ncols1*ncols2;

    const int bidx0 = blockIdx.x;
    const int j     = blockIdx.y;
    const int c     = blockIdx.z;
    const int jc    = j*ncols2 + c;
    const int tid   = threadIdx.x;

    const float * dst_fixup_data = ((const float *) dst_fixup) + gridDim.x*(2*2*ncols);

    const int iter_k = ne11 / FATTN_KQ_STRIDE;
    const int iter_j = (ne01 + (ncols1 - 1)) / ncols1;

    const int kbc0      = (bidx0 + 0)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
    const int kbc0_stop = (bidx0 + 1)*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;

    const bool did_not_have_any_data   = kbc0 == kbc0_stop;
    const bool wrote_beginning_of_tile = kbc0 % iter_k == 0;
    const bool did_not_write_last      = kbc0/iter_k == kbc0_stop/iter_k && kbc0_stop % iter_k != 0;
    if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
        return;
    }

    const int sequence = kbc0 / (iter_k*iter_j*(ne02/ncols2));
    const int head = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence) / (iter_k*iter_j);
    const int jt = (kbc0 - iter_k*iter_j*(ne02/ncols2)*sequence - iter_k*iter_j*head) / iter_k; // j index of current tile.

    if (jt*ncols1 + j >= ne01) {
        return;
    }

    dst += sequence*ne02*ne01*D + jt*ne02*(ncols1*D) + head*(ncols2*D) + (j*ne02 + c)*D + tid;

    // Load the partial result that needs a fixup:
    float dst_val = 0.0f;
    float max_val = 0.0f;
    float rowsum  = 0.0f;
    {
        dst_val = *dst;

        const float2 tmp = dst_fixup[bidx0*ncols + jc];
        max_val = tmp.x;
        rowsum  = tmp.y;
    }

    // Iterate over previous blocks and compute the combined results.
    // All CUDA blocks that get here must have a previous block that needs a fixup.
    int bidx = bidx0 - 1;
    int kbc_stop = kbc0;
    while(true) {
        const int kbc = bidx*(iter_k*iter_j*(ne02/ncols2)*ne03) / gridDim.x;
        if (kbc == kbc_stop) { // Did not have any data.
            bidx--;
            kbc_stop = kbc;
            continue;
        }

        const float dst_add = dst_fixup_data[bidx*ncols*D + jc*D + tid];

        const float2 tmp = dst_fixup[(gridDim.x + bidx)*ncols + jc];

        // Scale the current and new value accumulators depending on the max. values.
        const float max_val_new = fmaxf(max_val, tmp.x);

        const float diff_val = max_val - max_val_new;
        const float diff_add = tmp.x   - max_val_new;

        const float scale_val = diff_val >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_val) : 0.0f;
        const float scale_add = diff_add >= SOFTMAX_FTZ_THRESHOLD ? expf(diff_add) : 0.0f;

        dst_val = scale_val*dst_val + scale_add*dst_add;
        rowsum  = scale_val*rowsum  + scale_add*tmp.y;

        max_val = max_val_new;

        // If this block started in a previous tile we are done and don't need to combine additional partial results.
        if (kbc % iter_k == 0 || kbc/iter_k < kbc0/iter_k) {
            break;
        }
        bidx--;
        kbc_stop = kbc;
    }

    // Write back final result:
    *dst = dst_val / rowsum;
}

template<int D> // D == head size
#if !defined(GGML_USE_HIP)
__launch_bounds__(D, 1)
#endif // !(defined(GGML_USE_HIP)
static __global__ void flash_attn_combine_results(
        const float  * __restrict__ VKQ_parts,
        const float2 * __restrict__ VKQ_meta,
        float * __restrict__ dst,
        const int parallel_blocks) {
    // Dimension 0: threadIdx.x
    // Dimension 1: blockIdx.x
    // Dimension 2: blockIdx.y
    // Dimension 3: blockIdx.z
    // Memory layout is permuted with [0, 2, 1, 3]

    const int ne01 = gridDim.x;
    const int ne02 = gridDim.y;

    const int col      = blockIdx.x;
    const int head     = blockIdx.y;
    const int sequence = blockIdx.z;

    const int j_dst_unrolled = (sequence*ne01 + col)*ne02 + head;

    VKQ_parts += j_dst_unrolled * parallel_blocks*D;
    VKQ_meta  += j_dst_unrolled * parallel_blocks;
    dst       += j_dst_unrolled *                 D;

    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    for (int i = tid; i < 2*parallel_blocks; i += D) {
        ((float *) meta)[i] = ((const float *)VKQ_meta) [i];
    }

    __syncthreads();

    float kqmax = meta[0].x;
    for (int l = 1; l < parallel_blocks; ++l) {
        kqmax = max(kqmax, meta[l].x);
    }

    float VKQ_numerator   = 0.0f;
    float VKQ_denominator = 0.0f;
    for (int l = 0; l < parallel_blocks; ++l) {
        const float diff = meta[l].x - kqmax;
        float KQ_max_scale = expf(diff);
        const uint32_t ftz_mask = 0xFFFFFFFF * (diff > SOFTMAX_FTZ_THRESHOLD);
        *((uint32_t *) &KQ_max_scale) &= ftz_mask;

        VKQ_numerator   += KQ_max_scale * VKQ_parts[l*D + tid];
        VKQ_denominator += KQ_max_scale * meta[l].y;
    }

    dst[tid] = VKQ_numerator / VKQ_denominator;
}

[[noreturn]]
static void on_no_fattn_vec_case(const int D) {
    if (D == 64) {
        fprintf(stderr, "Unsupported KV type combination for head_size 64.\n");
        fprintf(stderr, "By default only f16 KV cache is supported.\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for V cache quantization support.\n");
        GGML_ABORT("fatal error");
    } else if (D == 128) {
        fprintf(stderr, "Unsupported KV type combination for head_size 128.\n");
        fprintf(stderr, "Supported combinations:\n");
        fprintf(stderr, "  - K == q4_0, V == q4_0,  4.50 BPV\n");
        fprintf(stderr, "  - K == q8_0, V == q8_0,  8.50 BPV\n");
        fprintf(stderr, "  - K == f16,  V == f16,  16.00 BPV\n");
        fprintf(stderr, "Compile with GGML_CUDA_FA_ALL_QUANTS for all combinations of q4_0, q4_1, q5_0, q5_1, q8_0, and f16.\n");
        GGML_ABORT("fatal error");
    } else {
        fprintf(stderr, "Unsupported KV type combination for head_size %d.\n", D);
        fprintf(stderr, "Only f16 is supported.\n");
        GGML_ABORT("fatal error");
    }
}

template <int DV, int ncols1, int ncols2>
void launch_fattn(
    ggml_backend_cuda_context & ctx, ggml_tensor * dst, fattn_kernel_t fattn_kernel, const int nwarps, const size_t nbytes_shared,
    const int KQ_row_granularity, const bool need_f16_K, const bool need_f16_V, const bool stream_k, const int warp_size = WARP_SIZE
) {
    constexpr int ncols = ncols1 * ncols2;

    const bool is_mla = DV == 512; // TODO better parameterization

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    GGML_ASSERT(V || is_mla);

    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    GGML_ASSERT(      Q->nb[0] == ggml_element_size(Q));
    GGML_ASSERT(      K->nb[0] == ggml_element_size(K));
    GGML_ASSERT(!V || V->nb[0] == ggml_element_size(V));

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);
    GGML_ASSERT(!mask || mask->ne[1] >= GGML_PAD(Q->ne[1], 16) &&
        "the Flash-Attention CUDA kernel requires the mask to be padded to 16 and at least n_queries big");

    GGML_ASSERT(K->ne[1] % FATTN_KQ_STRIDE == 0 && "Incorrect KV cache padding.");

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t main_stream = ctx.stream();
    const int id  = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[id].cc;
    const int nsm = ggml_cuda_info().devices[id].nsm;

    ggml_cuda_pool_alloc<half>   K_f16(pool);
    ggml_cuda_pool_alloc<half>   V_f16(pool);
    ggml_cuda_pool_alloc<int>    KV_max(pool);
    ggml_cuda_pool_alloc<float>  dst_tmp(pool);
    ggml_cuda_pool_alloc<float2> dst_tmp_meta(pool);

    const char * K_data = (const char *) K->data;
    size_t nb11 = K->nb[1];
    size_t nb12 = K->nb[2];
    size_t nb13 = K->nb[3];

    const char * V_data = V ? (const char *) V->data : nullptr;
    size_t nb21 = V ? V->nb[1] : nb11;
    size_t nb22 = V ? V->nb[2] : nb12;
    size_t nb23 = V ? V->nb[3] : nb13;

    if (need_f16_K && K->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(K->type);
        const size_t ts = ggml_type_size(K->type);

        K_f16.alloc(ggml_nelements(K));
        if (ggml_is_contiguously_allocated(K)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(K->type);
            to_fp16(K_data, K_f16.ptr, ggml_nelements(K), main_stream);

            nb11 = nb11*bs*sizeof(half)/ts;
            nb12 = nb12*bs*sizeof(half)/ts;
            nb13 = nb13*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(K->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(K->type);
            const int64_t s01 = nb11 / ts;
            const int64_t s02 = nb12 / ts;
            const int64_t s03 = nb13 / ts;
            to_fp16(K_data, K_f16.ptr, K->ne[0], K->ne[1], K->ne[2], K->ne[3], s01, s02, s03, main_stream);

            nb11 = K->ne[0] * sizeof(half);
            nb12 = K->ne[1] * nb11;
            nb13 = K->ne[2] * nb12;
        }
        K_data = (char *) K_f16.ptr;
    }

    if (V && need_f16_V && V->type != GGML_TYPE_F16) {
        const size_t bs = ggml_blck_size(V->type);
        const size_t ts = ggml_type_size(V->type);

        V_f16.alloc(ggml_nelements(V));
        if (ggml_is_contiguously_allocated(V)) {
            to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(V->type);
            to_fp16(V_data, V_f16.ptr, ggml_nelements(V), main_stream);
            V_data = (char *) V_f16.ptr;

            nb21 = nb21*bs*sizeof(half)/ts;
            nb22 = nb22*bs*sizeof(half)/ts;
            nb23 = nb23*bs*sizeof(half)/ts;
        } else {
            GGML_ASSERT(V->nb[0] == ts);
            to_fp16_nc_cuda_t to_fp16 = ggml_get_to_fp16_nc_cuda(V->type);
            const int64_t s01 = nb21 / ts;
            const int64_t s02 = nb22 / ts;
            const int64_t s03 = nb23 / ts;
            to_fp16(V_data, V_f16.ptr, V->ne[0], V->ne[1], V->ne[2], V->ne[3], s01, s02, s03, main_stream);

            nb21 = V->ne[0] * sizeof(half);
            nb22 = V->ne[1] * nb21;
            nb23 = V->ne[2] * nb22;
        }
        V_data = (char *) V_f16.ptr;
    }

    const int ntiles_x = ((Q->ne[1] + ncols1 - 1) / ncols1);
    const int ntiles_total = ntiles_x * (Q->ne[2] / ncols2) * Q->ne[3];

    // Optional optimization where the mask is scanned to determine whether part of the calculation can be skipped.
    // Only worth the overhead if there is at lease one FATTN_KQ_STRIDE x FATTN_KQ_STRIDE square to be skipped or
    //     multiple sequences of possibly different lengths.
    if (mask && (Q->ne[1] >= 1024 || Q->ne[3] > 1)) {
        const int s31 = mask->nb[1] / sizeof(half2);
        const int s33 = mask->nb[3] / sizeof(half2);

        const dim3 blocks_num_KV_max(ntiles_x, Q->ne[3], 1);
        const dim3 block_dim_KV_max(FATTN_KQ_STRIDE/2, 1, 1);

        const int ne_KV_max = blocks_num_KV_max.x*blocks_num_KV_max.y;
        const int iter_k = K->ne[1] / FATTN_KQ_STRIDE;

        KV_max.alloc(ne_KV_max);
        flash_attn_mask_to_KV_max<ncols1><<<blocks_num_KV_max, block_dim_KV_max, 0, main_stream>>>
            ((const half2 *) mask->data, KV_max.ptr, iter_k, s31, s33);
        CUDA_CHECK(cudaGetLastError());
    }

    int parallel_blocks = 1;

    const dim3 block_dim(warp_size, nwarps, 1);
    int max_blocks_per_sm = 1; // Max. number of active blocks limited by occupancy.
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, fattn_kernel, block_dim.x * block_dim.y * block_dim.z, nbytes_shared));

    dim3 blocks_num;
    if (stream_k) {
        // For short contexts it can be faster to have the SMs work on whole tiles because this lets us skip the fixup.
        const int max_blocks = max_blocks_per_sm*nsm;
        const int tiles_nwaves = (ntiles_total + max_blocks - 1) / max_blocks;
        const int tiles_efficiency_percent = 100 * ntiles_total / (max_blocks*tiles_nwaves);

        const int nblocks_stream_k = max_blocks;

        const bool use_stream_k = cc >= GGML_CUDA_CC_ADA_LOVELACE || tiles_efficiency_percent < 75;

        blocks_num.x = use_stream_k ? nblocks_stream_k : ntiles_total;
        blocks_num.y = 1;
        blocks_num.z = 1;

        dst_tmp_meta.alloc(blocks_num.x*ncols * (2*2 + DV) * sizeof(float));
    } else {
        GGML_ASSERT(K->ne[1] % KQ_row_granularity == 0);
        const int ntiles_KQ = K->ne[1] / KQ_row_granularity; // Max. number of parallel blocks limited by tensor size.

        // parallel_blocks should be at least large enough to achieve max. occupancy for a single wave:
        parallel_blocks = std::max((nsm * max_blocks_per_sm) / ntiles_total, 1);

        // parallel_blocks must not be larger than what the tensor size allows:
        parallel_blocks = std::min(parallel_blocks, ntiles_KQ);

        // If ntiles_total % blocks_per_wave != 0 then some efficiency is lost due to tail effects.
        // Test whether parallel_blocks can be set to a higher value for better efficiency.
        const int blocks_per_wave = nsm * max_blocks_per_sm;
        int nwaves_best = 0;
        int efficiency_percent_best = 0;
        for (int parallel_blocks_test = parallel_blocks; parallel_blocks_test <= ntiles_KQ; ++parallel_blocks_test) {
            const int nblocks_total = ntiles_total * parallel_blocks_test;
            const int nwaves = (nblocks_total + blocks_per_wave - 1) / blocks_per_wave;
            const int efficiency_percent = 100 * nblocks_total / (nwaves*blocks_per_wave);

            // Stop trying configurations with more waves if we already have good efficiency to avoid excessive overhead.
            if (efficiency_percent_best >= 90 && nwaves > nwaves_best) {
                break;
            }

            if (efficiency_percent > efficiency_percent_best) {
                nwaves_best = nwaves;
                efficiency_percent_best = efficiency_percent;
                parallel_blocks = parallel_blocks_test;
            }
        }

        blocks_num.x = ntiles_x;
        blocks_num.y = parallel_blocks;
        blocks_num.z = Q->ne[2]*Q->ne[3];

        if (parallel_blocks > 1) {
            dst_tmp.alloc(parallel_blocks*ggml_nelements(KQV));
            dst_tmp_meta.alloc(parallel_blocks*ggml_nrows(KQV));
        }
    }

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;

    memcpy(&scale,         (const float *) KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(block_dim.x % warp_size == 0);
    fattn_kernel<<<blocks_num, block_dim, nbytes_shared, main_stream>>>(
        (const char *) Q->data,
        K_data,
        V_data,
        mask ? ((const char *) mask->data) : nullptr,
        sinks ? ((const char *) sinks->data) : nullptr,
        KV_max.ptr,
        !stream_k && parallel_blocks > 1 ? dst_tmp.ptr : (float *) KQV->data, dst_tmp_meta.ptr,
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], Q->nb[1], Q->nb[2], Q->nb[3],
        K->ne[0], K->ne[1], K->ne[2], K->ne[3], nb11, nb12, nb13,
        nb21, nb22, nb23,
        mask ? mask->ne[1] : 0, mask ? mask->ne[2] : 0, mask ? mask->ne[3] : 0,
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0
    );
    CUDA_CHECK(cudaGetLastError());

    if (stream_k) {
        if (ntiles_total % blocks_num.x != 0) { // Fixup is only needed if the SMs work on fractional tiles.
            const dim3 block_dim_combine(DV, 1, 1);
            const dim3 blocks_num_combine = {blocks_num.x, ncols1, ncols2};

            flash_attn_stream_k_fixup<DV, ncols1, ncols2>
                <<<blocks_num_combine, block_dim_combine, 0, main_stream>>>
                ((float *) KQV->data, dst_tmp_meta.ptr, Q->ne[1], Q->ne[2], Q->ne[3], K->ne[1]);
        }
    } else if (parallel_blocks > 1) {
        const dim3 block_dim_combine(DV, 1, 1);
        const dim3 blocks_num_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const size_t nbytes_shared_combine = parallel_blocks*sizeof(float2);

        flash_attn_combine_results<DV>
            <<<blocks_num_combine, block_dim_combine, nbytes_shared_combine, main_stream>>>
            (dst_tmp.ptr, dst_tmp_meta.ptr, (float *) KQV->data, parallel_blocks);
    }
    CUDA_CHECK(cudaGetLastError());
}

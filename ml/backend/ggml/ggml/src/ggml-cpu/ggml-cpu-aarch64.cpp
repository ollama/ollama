#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu-traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cfloat>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#include "ggml-cpu-aarch64.h"

// TODO: move to include file?
template <int K> constexpr int QK_0() {
    if constexpr (K == 4) {
        return QK4_0;
    }
    if constexpr (K == 8) {
        return QK8_0;
    }
    return -1;
}

template <int K, int N> struct block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    int8_t    qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

// control size
static_assert(sizeof(block<4, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 2, "wrong block<4,4> size/padding");
static_assert(sizeof(block<4, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<4,8> size/padding");
static_assert(sizeof(block<8, 4>) == 4 * sizeof(ggml_half) + QK8_0 * 4, "wrong block<8,4> size/padding");
static_assert(sizeof(block<8, 8>) == 8 * sizeof(ggml_half) + QK8_0 * 8, "wrong block<8,8> size/padding");

using block_q4_0x4 = block<4, 4>;
using block_q4_0x8 = block<4, 8>;
using block_q8_0x4 = block<8, 4>;
using block_q8_0x8 = block<8, 8>;

struct block_iq4_nlx4 {
    ggml_half d[4];            // deltas for 4 iq4_nl blocks
    uint8_t   qs[QK4_NL * 2];  // nibbles / quants for 4 iq4_nl blocks
};

static_assert(sizeof(block_iq4_nlx4) == 4 * sizeof(ggml_half) + QK4_NL * 2, "wrong iq4_nlx4 block size/padding");

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#elif defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define UNUSED GGML_UNUSED

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//
#if defined(__AVX__)
#if defined(__F16C__)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     _mm512_cvtph_ps(_mm256_set_m128i(_mm_loadu_si128((const __m128i *)(y)), _mm_loadu_si128((const __m128i *)(x))))
#define GGML_F32Cx16_REPEAT_LOAD(x)  _mm512_cvtph_ps(_mm256_set_m128i(x, x))
#endif
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     _mm256_cvtph_ps(_mm_shuffle_epi32(_mm_maskload_epi32((int const*)(x), loadMask), 68))
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     _mm256_cvtph_ps(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask))
#else
#if defined(__AVX512F__)
static inline __m512 __avx512_f32cx8x2_load(ggml_fp16_t *x, ggml_fp16_t *y) {
    float tmp[16];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    for (int i = 0; i < 8; i++) {
        tmp[i + 8] = GGML_FP16_TO_FP32(y[i]);
    }

    return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
    float tmp[16];
    uint16_t tmphalf[8];
    _mm_storeu_si128((__m128i*)tmphalf, x);

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 4] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 8] = GGML_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 12] = GGML_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
        tmp[i + 4] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];

    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask));
    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm256_loadu_ps(tmp);
}

#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_REPEAT_LOAD(x, loadMask)     __avx_repeat_f32cx8_load(x)
#define GGML_F32Cx8_REARRANGE_LOAD(x, arrangeMask)     __avx_rearranged_f32cx8_load(x, arrangeMask)
#if defined(__AVX512F__)
#define GGML_F32Cx8x2_LOAD(x, y)     __avx512_f32cx8x2_load(x, y)
#define GGML_F32Cx16_REPEAT_LOAD(x)  __avx512_repeat_f32cx16_load(x)
#endif
#endif
#endif


#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector
static inline __m512i sum_i16_pairs_int_32x16(const __m512i x) {
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_madd_epi16(ones, x);
}

static inline __m512i mul_sum_us8_pairs_int32x16(const __m512i ax, const __m512i sy) {
#if defined(__AVX512VNNI__)
    const __m512i zero = _mm512_setzero_si512();
    return _mm512_dpbusd_epi32(zero, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return sum_i16_pairs_int_32x16(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector
static inline __m512i mul_sum_i8_pairs_int32x16(const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Get absolute values of x vectors
    const __m512i ax = _mm512_abs_epi8(x);
    // Sign the values of the y vectors
    __mmask64 blt0 = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
    return mul_sum_us8_pairs_int32x16(ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector
static inline __m256i sum_i16_pairs_int32x8(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_madd_epi16(ones, x);
}

static inline __m256i mul_sum_us8_pairs_int32x8(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbusd_epi32(zero, ax, sy);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbusd_avx_epi32(zero, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_int32x8(dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector
static inline __m256i mul_sum_i8_pairs_int32x8(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    return _mm256_dpbssd_epi32(zero, x, y);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_int32x8(ax, sy);
#endif
}
#endif

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

static void quantize_q8_0_4x4(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 8; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 3] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 11] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[16 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[16 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[16 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[16 * j + 15] = vgetq_lane_s32(vi, 3);
        }
    }
#else
    // scalar
    const int blck_size_interleave = 4;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

static void quantize_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__ARM_NEON)
    float32x4_t srcv[4][8];
    float id[4];

    for (int i = 0; i < nb; i++) {
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int row_iter = 0; row_iter < 4; row_iter++) {
            for (int j = 0; j < 8; j++) srcv[row_iter][j] = vld1q_f32(x + row_iter * k + i * 32 + 4 * j);
            for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[row_iter][j]);

            for (int j = 0; j < 4; j++) amaxv[2 * j] = vmaxq_f32(asrcv[2 * j], asrcv[2 * j + 1]);
            for (int j = 0; j < 2; j++) amaxv[4 * j] = vmaxq_f32(amaxv[4 * j], amaxv[4 * j + 2]);
            for (int j = 0; j < 1; j++) amaxv[8 * j] = vmaxq_f32(amaxv[8 * j], amaxv[8 * j + 4]);

            const float amax = vmaxvq_f32(amaxv[0]);

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < 4; j++) {
            float32x4_t v = vmulq_n_f32(srcv[0][2 * j], id[0]);
            int32x4_t vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 3] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[0][2 * j + 1], id[0]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 4] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 5] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 6] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 7] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[1][2 * j], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 8] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 9] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 10] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 11] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[1][2 * j + 1], id[1]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 12] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 13] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 14] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 15] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[2][2 * j], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 16] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 17] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 18] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 19] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[2][2 * j + 1], id[2]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 20] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 21] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 22] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 23] = vgetq_lane_s32(vi, 3);

            v = vmulq_n_f32(srcv[3][2 * j], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 24] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 25] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 26] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 27] = vgetq_lane_s32(vi, 3);
            v = vmulq_n_f32(srcv[3][2 * j + 1], id[3]);
            vi = vcvtnq_s32_f32(v);
            y[i].qs[32 * j + 28] = vgetq_lane_s32(vi, 0);
            y[i].qs[32 * j + 29] = vgetq_lane_s32(vi, 1);
            y[i].qs[32 * j + 30] = vgetq_lane_s32(vi, 2);
            y[i].qs[32 * j + 31] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    float id[4];
    __m256 srcv[4][4];
    __m256 idvec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 32 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 32 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
            maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            // Divided by 127.f to mirror results in quantize_row_q8_0
            const float d = maxScalar  / 127.f;
            id[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f; //d ? 1.0f / d : 0.0f;

            // Store the scale for the individual block
            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);

            // Store the values in blocks of eight values - Aim is to use these later for block interleaving
            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;
            idvec[row_iter] = _mm256_set1_ps(id[row_iter]);
        }

        // The loop iterates four times - The aim is to get 4 corresponding chunks of eight bytes from the original weight blocks that are interleaved
        for (int j = 0; j < 4; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], idvec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], idvec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], idvec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], idvec[3]);

            // Round to nearest integer
            v0 = _mm256_round_ps( v0, _MM_ROUND_NEAREST );
            v1 = _mm256_round_ps( v1, _MM_ROUND_NEAREST );
            v2 = _mm256_round_ps( v2, _MM_ROUND_NEAREST );
            v3 = _mm256_round_ps( v3, _MM_ROUND_NEAREST );

            // Convert floats to integers
            __m256i i0 = _mm256_cvtps_epi32( v0 );
            __m256i i1 = _mm256_cvtps_epi32( v1 );
            __m256i i2 = _mm256_cvtps_epi32( v2 );
            __m256i i3 = _mm256_cvtps_epi32( v3 );

#if defined(__AVX2__)
            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
#else
            // Since we don't have in AVX some necessary functions,
            // we split the registers in half and call AVX2 analogs from SSE
            __m128i ni0 = _mm256_castsi256_si128( i0 );
            __m128i ni1 = _mm256_extractf128_si256( i0, 1);
            __m128i ni2 = _mm256_castsi256_si128( i1 );
            __m128i ni3 = _mm256_extractf128_si256( i1, 1);
            __m128i ni4 = _mm256_castsi256_si128( i2 );
            __m128i ni5 = _mm256_extractf128_si256( i2, 1);
            __m128i ni6 = _mm256_castsi256_si128( i3 );
            __m128i ni7 = _mm256_extractf128_si256( i3, 1);

            // Convert int32 to int16
            ni0 = _mm_packs_epi32( ni0, ni1 );
            ni2 = _mm_packs_epi32( ni2, ni3 );
            ni4 = _mm_packs_epi32( ni4, ni5 );
            ni6 = _mm_packs_epi32( ni6, ni7 );
            // Convert int16 to int8
            ni0 = _mm_packs_epi16( ni0, ni2 );
            ni4 = _mm_packs_epi16( ni4, ni6 );
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j), ni0);
            _mm_storeu_si128((__m128i *)(y[i].qs + 32 * j + 16), ni4);
#endif
        }
    }
#else
    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK8_0];
    float id[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max

            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[row_iter][j]));
            }

            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[row_iter] = GGML_FP32_TO_FP16(d);
        }

        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);

            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
#endif
}

static void quantize_mat_q8_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t nrow, int64_t n_per_row, int64_t blck_size_interleave) {
    assert(nrow == 4);
    UNUSED(nrow);
    if (blck_size_interleave == 4) {
        quantize_q8_0_4x4(x, vy, n_per_row);
    } else if (blck_size_interleave == 8) {
        quantize_q8_0_4x8(x, vy, n_per_row);
    } else {
        assert(false);
    }
}

static void ggml_gemv_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

        for (int c = 0; c < nc; c += ncols_interleaved) {
            const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
            float32x4_t acc = vdupq_n_f32(0);
            for (int b = 0; b < nb; b++) {
                int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
                int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
                int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
                int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
                float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

                int8x16_t a0 = vld1q_s8(a_ptr->qs);
                int8x16_t a1 = vld1q_s8(a_ptr->qs + qk/2);
                float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

                int32x4_t ret = vdupq_n_s32(0);

                ret = vdotq_laneq_s32(ret, b0 << 4, a0, 0);
                ret = vdotq_laneq_s32(ret, b1 << 4, a0, 1);
                ret = vdotq_laneq_s32(ret, b2 << 4, a0, 2);
                ret = vdotq_laneq_s32(ret, b3 << 4, a0, 3);

                ret = vdotq_laneq_s32(ret, b0 & 0xf0U, a1, 0);
                ret = vdotq_laneq_s32(ret, b1 & 0xf0U, a1, 1);
                ret = vdotq_laneq_s32(ret, b2 & 0xf0U, a1, 2);
                ret = vdotq_laneq_s32(ret, b3 & 0xf0U, a1, 3);

                acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                                vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
                a_ptr++;
                b_ptr++;
            }
            vst1q_f32(s, acc);
            s += ncols_interleaved;
        }
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

static void ggml_gemv_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx;

        for (int c = 0; c < nc; c += ncols_interleaved) {
            const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
            float32x4_t acc = vdupq_n_f32(0);
            for (int b = 0; b < nb; b++) {
                int8x16_t b0 = vld1q_s8((const int8_t *) b_ptr->qs);
                int8x16_t b1 = vld1q_s8((const int8_t *) b_ptr->qs + 16);
                int8x16_t b2 = vld1q_s8((const int8_t *) b_ptr->qs + 32);
                int8x16_t b3 = vld1q_s8((const int8_t *) b_ptr->qs + 48);
                float16x4_t bd = vld1_f16((const __fp16 *) b_ptr->d);

                int8x16_t a0 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs);
                int8x16_t a1 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 1);
                int8x16_t a2 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 2);
                int8x16_t a3 = (int8x16_t) vld1q_dup_s64((const int64_t *) a_ptr->qs + 3);
                float16x4_t ad = vld1_dup_f16((const __fp16 *) &a_ptr->d);

                int32x4_t ret0 = vdupq_n_s32(0);
                int32x4_t ret1 = vdupq_n_s32(0);

                ret0 = vdotq_s32(ret0, b0 << 4, a0);
                ret1 = vdotq_s32(ret1, b1 << 4, a0);
                ret0 = vdotq_s32(ret0, b2 << 4, a1);
                ret1 = vdotq_s32(ret1, b3 << 4, a1);

                ret0 = vdotq_s32(ret0, b0 & 0xf0U, a2);
                ret1 = vdotq_s32(ret1, b1 & 0xf0U, a2);
                ret0 = vdotq_s32(ret0, b2 & 0xf0U, a3);
                ret1 = vdotq_s32(ret1, b3 & 0xf0U, a3);

                int32x4_t ret = vpaddq_s32(ret0, ret1);

                acc = vfmaq_f32(acc, vcvtq_n_f32_s32(ret, 4),
                        vmulq_f32(vcvt_f32_f16(ad), vcvt_f32_f16(bd)));
                a_ptr++;
                b_ptr++;
            }
            vst1q_f32(s, acc);
            s += ncols_interleaved;
        }
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    float sumf[4];
    int sumi;

    const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        for (int l = 0; l < nb; l++) {
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

static void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE)
    if (ggml_cpu_has_sve() && ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;

        __asm__ __volatile__(
            "ptrue p0.b\n"
            "add %x[b_ptr], %x[b_ptr], #0x10\n"
            "1:"  // Column loop
            "add x22, %x[a_ptr], #0x2\n"
            "mov z31.b, #0x0\n"
            "mov x21, %x[nb]\n"
            "2:"  // Block loop
            "ld1b { z30.b }, p0/Z, [%x[b_ptr]]\n"
            "ld1b { z29.b }, p0/Z, [%x[b_ptr], #1, MUL VL]\n"
            "mov z28.s, #0x0\n"
            "mov z27.s, #0x0\n"
            "ld1rd { z26.d }, p0/Z, [x22]\n"
            "ld1b { z25.b }, p0/Z, [%x[b_ptr], #2, MUL VL]\n"
            "sub x20, x22, #0x2\n"
            "sub x21, x21, #0x1\n"
            "ld1b { z24.b }, p0/Z, [%x[b_ptr], #3, MUL VL]\n"
            "ld1rd { z23.d }, p0/Z, [x22, #8]\n"
            "lsl z22.b, z30.b, #0x4\n"
            "lsl z16.b, z29.b, #0x4\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1rd { z21.d }, p0/Z, [x22, #16]\n"
            "ld1rd { z20.d }, p0/Z, [x22, #24]\n"
            "lsl z19.b, z25.b, #0x4\n"
            "and z25.b, z25.b, #0xf0\n"
            "ld1rh { z17.h }, p0/Z, [x20]\n"
            "ld1h { z18.s }, p0/Z, [%x[b_ptr], #-1, MUL VL]\n"
            "sdot z28.s, z22.b, z26.b\n"
            "sdot z27.s, z16.b, z26.b\n"
            "lsl z16.b, z24.b, #0x4\n"
            "add x22, x22, #0x22\n"
            "and z24.b, z24.b, #0xf0\n"
            "add %x[b_ptr], %x[b_ptr], #0x90\n"
            "fcvt z17.s, p0/m, z17.h\n"
            "fcvt z18.s, p0/m, z18.h\n"
            "sdot z28.s, z19.b, z23.b\n"
            "sdot z27.s, z16.b, z23.b\n"
            "fmul z18.s, z18.s, z17.s\n"
            "sdot z28.s, z30.b, z21.b\n"
            "sdot z27.s, z29.b, z21.b\n"
            "sdot z28.s, z25.b, z20.b\n"
            "sdot z27.s, z24.b, z20.b\n"
            "uzp1 z17.s, z28.s, z27.s\n"
            "uzp2 z16.s, z28.s, z27.s\n"
            "add z17.s, z17.s, z16.s\n"
            "asr z17.s, z17.s, #0x4\n"
            "scvtf z17.s, p0/m, z17.s\n"
            "fmla z31.s, p0/M, z17.s, z18.s\n"
            "cbnz x21, 2b\n"
            "sub %x[nc], %x[nc], #0x8\n"
            "st1w { z31.s }, p0, [%x[res_ptr]]\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "cbnz %x[nc], 1b\n"
            : [b_ptr] "+&r" (b_ptr), [res_ptr] "+&r" (res_ptr), [nc] "+&r" (nc)
            : [a_ptr] "r" (a_ptr), [nb] "r" (nb)
            : "memory", "p0", "x20", "x21", "x22", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE)
#elif defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    __m128i changemask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Permute mask used for easier vector processing at later stages
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK4_0;

    const block_q4_0x8 * b_ptr_start = (const block_q4_0x8 *)vx;
    const block_q8_0 * a_ptr_start = (const block_q8_0 *)vy;

    // Process Q8_0 blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_0 format
        const block_q8_0 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q4_0x8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q4_0x8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulator
            __m256 acc_row = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {
                // Load 8 blocks of Q4_0 interleaved as 8 bytes (B0 - B7)
                const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 1);
                const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 2);
                const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs) + 3);

                // 4-bit -> 8-bit - Sign is maintained
                const __m256i rhs_vec_0123_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_0, m4b)); // B0(0-7) B1(0-7) B2(0-7) B3(0-7)
                const __m256i rhs_vec_4567_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_0, m4b)); // B4(0-7) B5(0-7) B6(0-7) B7(0-7)
                const __m256i rhs_vec_0123_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_0123_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)
                const __m256i rhs_vec_4567_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_vec_4567_1, m4b)); // B0(8-15) B1(8-15) B2(8-15) B3(8-15)

                const __m256i rhs_vec_0123_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b)); // B0(16-23) B1(16-23) B2(16-23) B3(16-23)
                const __m256i rhs_vec_4567_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b)); // B4(16-23) B5(16-23) B6(16-23) B7(16-23)
                const __m256i rhs_vec_0123_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b)); // B0(24-31) B1(24-31) B2(24-31) B3(24-31)
                const __m256i rhs_vec_4567_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b)); // B4(24-31) B5(24-31) B6(24-31) B7(24-31)

                // Load the scale values for the 8 blocks interleaved in block_q4_0x8
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, changemask);

                // Load and convert to FP32 scale from block_q8_0
                const __m256 row_scale_f32 = _mm256_set1_ps(GGML_FP16_TO_FP32(a_ptr[b].d));

                // Load the block values in block_q8_0 in batches of 16 bytes and replicate the same across 256 bit vector
                __m256i lhs_vec_0 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)a_ptr[b].qs));
                __m256i lhs_vec_1 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16)));

                lhs_vec_0 = _mm256_permute2f128_si256(lhs_vec_0, lhs_vec_0, 0); // A0 (0-15) A0(0-15)
                lhs_vec_1 = _mm256_permute2f128_si256(lhs_vec_1, lhs_vec_1, 0); // A0 (16-31) A0(16-31))

                __m256i iacc = _mm256_setzero_si256();

                // Dot product done within 32 bit lanes and accumulated in the same vector
                // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                // ...........................................................................
                // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)

                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(rhs_vec_0123_0 ,_mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0)));
                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177) ,rhs_vec_4567_0, 170), _mm256_shuffle_epi32(lhs_vec_0, 85)));

                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(rhs_vec_0123_1 ,_mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170)));
                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177) ,rhs_vec_4567_1, 170), _mm256_shuffle_epi32(lhs_vec_0, 255)));

                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(rhs_vec_0123_2 ,_mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0)));
                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177) ,rhs_vec_4567_2, 170), _mm256_shuffle_epi32(lhs_vec_1, 85)));

                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(rhs_vec_0123_3 ,_mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170)));
                iacc = _mm256_add_epi32(iacc, mul_sum_i8_pairs_int32x8(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177) ,rhs_vec_4567_3, 170), _mm256_shuffle_epi32(lhs_vec_1, 255)));

                // Accumulated values multipled with appropriate scales
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
        }
    }
    return;
#elif defined(__riscv_v_intrinsic)
    if (__riscv_vlenb() >= QK4_0) {
        const size_t vl = QK4_0;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

            vfloat32m1_t sumf = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
            for (int l = 0; l < nb; l++) {
                const int64_t a0 = *(const int64_t *)&a_ptr[l].qs[0];
                const int64_t a1 = *(const int64_t *)&a_ptr[l].qs[8];
                const int64_t a2 = *(const int64_t *)&a_ptr[l].qs[16];
                const int64_t a3 = *(const int64_t *)&a_ptr[l].qs[24];
                __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a0, vl / 4));
                const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a1, vl / 4));
                const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a2, vl / 4));
                const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(a3, vl / 4));

                const vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
                const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                const vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                const vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                const vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                const vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                const vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_hi_m));
                const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                // vector version needs Zvfhmin extension
                const float a_scale = GGML_FP16_TO_FP32(a_ptr[l].d);
                const float b_scales[8] = {
                    GGML_FP16_TO_FP32(b_ptr[l].d[0]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[1]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[2]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[3]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[4]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[5]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[6]),
                    GGML_FP16_TO_FP32(b_ptr[l].d[7])
                };
                const vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);
                const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scale, vl / 4);
                sumf = __riscv_vfmacc_vv_f32m1(sumf, tmp1, b_scales_vec, vl / 4);
            }
            __riscv_vse32_v_f32m1(s + x * ncols_interleaved, sumf, vl / 4);
        }
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    {
        float sumf[8];
        int sumi;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);

            for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi = 0;
                        for (int i = 0; i < blocklen; ++i) {
                            const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                            const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                        }
                        sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                    }
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

static void ggml_gemv_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
        const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);
        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        float * res_ptr = s;

        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            float32x4_t sumf = vdupq_n_f32(0);
            for (int l = 0; l < nb; l++) {
                uint8x16_t b_0 = vld1q_u8(b_ptr[l].qs + 0);
                uint8x16_t b_1 = vld1q_u8(b_ptr[l].qs + 16);
                uint8x16_t b_2 = vld1q_u8(b_ptr[l].qs + 32);
                uint8x16_t b_3 = vld1q_u8(b_ptr[l].qs + 48);

                int8x16_t b_0_hi = vqtbl1q_s8(kvalues, b_0 >> 4);
                int8x16_t b_0_lo = vqtbl1q_s8(kvalues, b_0 & 0x0F);
                int8x16_t b_1_hi = vqtbl1q_s8(kvalues, b_1 >> 4);
                int8x16_t b_1_lo = vqtbl1q_s8(kvalues, b_1 & 0x0F);
                int8x16_t b_2_hi = vqtbl1q_s8(kvalues, b_2 >> 4);
                int8x16_t b_2_lo = vqtbl1q_s8(kvalues, b_2 & 0x0F);
                int8x16_t b_3_hi = vqtbl1q_s8(kvalues, b_3 >> 4);
                int8x16_t b_3_lo = vqtbl1q_s8(kvalues, b_3 & 0x0F);

                int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 0);
                int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16);

                int32x4_t sumi = vdupq_n_s32(0);
                sumi = vdotq_laneq_s32(sumi, b_0_lo, a_0, 0);
                sumi = vdotq_laneq_s32(sumi, b_0_hi, a_1, 0);
                sumi = vdotq_laneq_s32(sumi, b_1_lo, a_0, 1);
                sumi = vdotq_laneq_s32(sumi, b_1_hi, a_1, 1);
                sumi = vdotq_laneq_s32(sumi, b_2_lo, a_0, 2);
                sumi = vdotq_laneq_s32(sumi, b_2_hi, a_1, 2);
                sumi = vdotq_laneq_s32(sumi, b_3_lo, a_0, 3);
                sumi = vdotq_laneq_s32(sumi, b_3_hi, a_1, 3);

                float32x4_t a_d = vcvt_f32_f16(vld1_dup_f16((const float16_t *)&a_ptr[l].d));
                float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));
                float32x4_t d = a_d * b_d;

                sumf = vmlaq_f32(sumf, d, vcvtq_f32_s32(sumi));
            }

            vst1q_f32(res_ptr + x * 4, sumf);
        }
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    {
        float sumf[4];
        int sumi;

        const block_q8_0 * a_ptr = (const block_q8_0 *) vy;
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

            for (int j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int j = 0; j < ncols_interleaved; j++) {
                        sumi = 0;
                        for (int i = 0; i < blocklen; ++i) {
                            const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                            const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                            sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                        }
                        sumf[j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d);
                    }
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

static void ggml_gemm_q4_0_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__(
            "mov x10, %x[nr]\n"
            "mov x9, #0x88\n"
            "cmp x10, #0x10\n"
            "mul x9, %x[nb], x9\n"
            "blt 4f\n"
            "1:"  // Row loop
            "add x28, %x[b_ptr], #0x8\n"
            "mov x27, %x[nc]\n"
            "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
            "2:"  // Column loop
            "add x25, %x[a_ptr], #0x8\n"
            "movi v15.16b, #0x0\n"
            "movi v19.16b, #0x0\n"
            "mov x24, %x[nb]\n"
            "add x23, x25, x9\n"
            "movi v18.16b, #0x0\n"
            "movi v14.16b, #0x0\n"
            "add x22, x23, x9\n"
            "movi v11.16b, #0x0\n"
            "movi v13.16b, #0x0\n"
            "add x21, x22, x9\n"
            "movi v23.16b, #0x0\n"
            "movi v16.16b, #0x0\n"
            "movi v25.16b, #0x0\n"
            "movi v7.16b, #0x0\n"
            "movi v0.16b, #0x0\n"
            "movi v4.16b, #0x0\n"
            "movi v5.16b, #0x0\n"
            "movi v21.16b, #0x0\n"
            "movi v8.16b, #0x0\n"
            "movi v1.16b, #0x0\n"
            "3:"  // Block loop
            "ldr q3, [x28, #0x0]\n"
            "ldr q31, [x25, #0x0]\n"
            "movi v28.16b, #0x4\n"
            "movi v10.4s, #0x0\n"
            "ldr q22, [x28, #0x10]\n"
            "ldr q6, [x25, #0x10]\n"
            "movi v29.4s, #0x0\n"
            "movi v9.4s, #0x0\n"
            "ldr q27, [x28, #0x20]\n"
            "ldr q30, [x28, #0x30]\n"
            "movi v20.4s, #0x0\n"
            "movi v24.16b, #0xf0\n"
            "ldr d2, [x25, #-0x8]\n"
            "ldr d26, [x23, #-0x8]\n"
            "sshl v12.16b, v3.16b, v28.16b\n"
            "sub x20, x28, #0x8\n"
            "ldr d17, [x20, #0x0]\n"
            "and v3.16b, v3.16b, v24.16b\n"
            "subs x24, x24, #0x1\n"
            "add x28, x28, #0x48\n"
            ".inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]\n"
            ".inst 0x4fbfe19d  // sdot v29.4s, v12.16b, v31.4b[1]\n"
            ".inst 0x4f9fe989  // sdot v9.4s, v12.16b, v31.4b[2]\n"
            ".inst 0x4fbfe994  // sdot v20.4s, v12.16b, v31.4b[3]\n"
            "sshl v31.16b, v22.16b, v28.16b\n"
            "and v22.16b, v22.16b, v24.16b\n"
            "fcvtl v17.4s, v17.4h\n"
            "fcvtl v2.4s, v2.4h\n"
            "fcvtl v26.4s, v26.4h\n"
            ".inst 0x4f86e3ea  // sdot v10.4s, v31.16b, v6.4b[0]\n"
            ".inst 0x4fa6e3fd  // sdot v29.4s, v31.16b, v6.4b[1]\n"
            ".inst 0x4f86ebe9  // sdot v9.4s, v31.16b, v6.4b[2]\n"
            ".inst 0x4fa6ebf4  // sdot v20.4s, v31.16b, v6.4b[3]\n"
            "sshl v6.16b, v27.16b, v28.16b\n"
            "sshl v28.16b, v30.16b, v28.16b\n"
            "and v27.16b, v27.16b, v24.16b\n"
            "and v30.16b, v30.16b, v24.16b\n"
            "ldr q24, [x25, #0x20]\n"
            ".inst 0x4f98e0ca  // sdot v10.4s, v6.16b, v24.4b[0]\n"
            ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
            ".inst 0x4f98e8c9  // sdot v9.4s, v6.16b, v24.4b[2]\n"
            ".inst 0x4fb8e8d4  // sdot v20.4s, v6.16b, v24.4b[3]\n"
            "ldr q24, [x25, #0x30]\n"
            ".inst 0x4f98e38a  // sdot v10.4s, v28.16b, v24.4b[0]\n"
            ".inst 0x4fb8e39d  // sdot v29.4s, v28.16b, v24.4b[1]\n"
            ".inst 0x4f98eb89  // sdot v9.4s, v28.16b, v24.4b[2]\n"
            ".inst 0x4fb8eb94  // sdot v20.4s, v28.16b, v24.4b[3]\n"
            "ldr q24, [x25, #0x40]\n"
            ".inst 0x4f98e06a  // sdot v10.4s, v3.16b, v24.4b[0]\n"
            ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
            ".inst 0x4f98e869  // sdot v9.4s, v3.16b, v24.4b[2]\n"
            ".inst 0x4fb8e874  // sdot v20.4s, v3.16b, v24.4b[3]\n"
            "ldr q24, [x25, #0x50]\n"
            ".inst 0x4f98e2ca  // sdot v10.4s, v22.16b, v24.4b[0]\n"
            ".inst 0x4fb8e2dd  // sdot v29.4s, v22.16b, v24.4b[1]\n"
            ".inst 0x4f98eac9  // sdot v9.4s, v22.16b, v24.4b[2]\n"
            ".inst 0x4fb8ead4  // sdot v20.4s, v22.16b, v24.4b[3]\n"
            "ldr q24, [x25, #0x60]\n"
            ".inst 0x4f98e36a  // sdot v10.4s, v27.16b, v24.4b[0]\n"
            ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
            ".inst 0x4f98eb69  // sdot v9.4s, v27.16b, v24.4b[2]\n"
            ".inst 0x4fb8eb74  // sdot v20.4s, v27.16b, v24.4b[3]\n"
            "ldr q24, [x25, #0x70]\n"
            "add x25, x25, #0x88\n"
            ".inst 0x4f98e3ca  // sdot v10.4s, v30.16b, v24.4b[0]\n"
            ".inst 0x4fb8e3dd  // sdot v29.4s, v30.16b, v24.4b[1]\n"
            ".inst 0x4f98ebc9  // sdot v9.4s, v30.16b, v24.4b[2]\n"
            ".inst 0x4fb8ebd4  // sdot v20.4s, v30.16b, v24.4b[3]\n"
            "fmul v24.4s, v17.4s, v2.s[0]\n"
            "scvtf v10.4s, v10.4s, #0x4\n"
            "scvtf v29.4s, v29.4s, #0x4\n"
            "scvtf v9.4s, v9.4s, #0x4\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "fmla v15.4s, v10.4s, v24.4s\n"
            "ldr q24, [x23, #0x0]\n"
            "fmul v10.4s, v17.4s, v2.s[1]\n"
            "fmla v19.4s, v29.4s, v10.4s\n"
            "ldr q10, [x23, #0x10]\n"
            "fmul v29.4s, v17.4s, v2.s[2]\n"
            "fmul v2.4s, v17.4s, v2.s[3]\n"
            "fmla v18.4s, v9.4s, v29.4s\n"
            "movi v9.4s, #0x0\n"
            "movi v29.4s, #0x0\n"
            ".inst 0x4f98e189  // sdot v9.4s, v12.16b, v24.4b[0]\n"
            ".inst 0x4fb8e19d  // sdot v29.4s, v12.16b, v24.4b[1]\n"
            "fmla v14.4s, v20.4s, v2.4s\n"
            "movi v20.4s, #0x0\n"
            "movi v2.4s, #0x0\n"
            ".inst 0x4f98e994  // sdot v20.4s, v12.16b, v24.4b[2]\n"
            ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
            "ldr q24, [x23, #0x20]\n"
            ".inst 0x4f8ae3e9  // sdot v9.4s, v31.16b, v10.4b[0]\n"
            ".inst 0x4faae3fd  // sdot v29.4s, v31.16b, v10.4b[1]\n"
            ".inst 0x4f8aebf4  // sdot v20.4s, v31.16b, v10.4b[2]\n"
            ".inst 0x4faaebe2  // sdot v2.4s, v31.16b, v10.4b[3]\n"
            "ldr q10, [x23, #0x30]\n"
            ".inst 0x4f98e0c9  // sdot v9.4s, v6.16b, v24.4b[0]\n"
            ".inst 0x4fb8e0dd  // sdot v29.4s, v6.16b, v24.4b[1]\n"
            ".inst 0x4f98e8d4  // sdot v20.4s, v6.16b, v24.4b[2]\n"
            ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
            "ldr q24, [x23, #0x40]\n"
            ".inst 0x4f8ae389  // sdot v9.4s, v28.16b, v10.4b[0]\n"
            ".inst 0x4faae39d  // sdot v29.4s, v28.16b, v10.4b[1]\n"
            ".inst 0x4f8aeb94  // sdot v20.4s, v28.16b, v10.4b[2]\n"
            ".inst 0x4faaeb82  // sdot v2.4s, v28.16b, v10.4b[3]\n"
            "ldr q10, [x23, #0x50]\n"
            ".inst 0x4f98e069  // sdot v9.4s, v3.16b, v24.4b[0]\n"
            ".inst 0x4fb8e07d  // sdot v29.4s, v3.16b, v24.4b[1]\n"
            ".inst 0x4f98e874  // sdot v20.4s, v3.16b, v24.4b[2]\n"
            ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
            "ldr q24, [x23, #0x60]\n"
            ".inst 0x4f8ae2c9  // sdot v9.4s, v22.16b, v10.4b[0]\n"
            ".inst 0x4faae2dd  // sdot v29.4s, v22.16b, v10.4b[1]\n"
            ".inst 0x4f8aead4  // sdot v20.4s, v22.16b, v10.4b[2]\n"
            ".inst 0x4faaeac2  // sdot v2.4s, v22.16b, v10.4b[3]\n"
            "ldr q10, [x23, #0x70]\n"
            "add x23, x23, #0x88\n"
            ".inst 0x4f98e369  // sdot v9.4s, v27.16b, v24.4b[0]\n"
            ".inst 0x4fb8e37d  // sdot v29.4s, v27.16b, v24.4b[1]\n"
            ".inst 0x4f98eb74  // sdot v20.4s, v27.16b, v24.4b[2]\n"
            ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
            "ldr q24, [x22, #0x0]\n"
            ".inst 0x4f8ae3c9  // sdot v9.4s, v30.16b, v10.4b[0]\n"
            ".inst 0x4faae3dd  // sdot v29.4s, v30.16b, v10.4b[1]\n"
            ".inst 0x4f8aebd4  // sdot v20.4s, v30.16b, v10.4b[2]\n"
            ".inst 0x4faaebc2  // sdot v2.4s, v30.16b, v10.4b[3]\n"
            "fmul v10.4s, v17.4s, v26.s[0]\n"
            "scvtf v9.4s, v9.4s, #0x4\n"
            "scvtf v29.4s, v29.4s, #0x4\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "scvtf v2.4s, v2.4s, #0x4\n"
            "fmla v11.4s, v9.4s, v10.4s\n"
            "ldr q9, [x22, #0x10]\n"
            "fmul v10.4s, v17.4s, v26.s[1]\n"
            "fmla v13.4s, v29.4s, v10.4s\n"
            "ldr d29, [x22, #-0x8]\n"
            "fmul v10.4s, v17.4s, v26.s[2]\n"
            "fmul v26.4s, v17.4s, v26.s[3]\n"
            "fcvtl v29.4s, v29.4h\n"
            "fmla v23.4s, v20.4s, v10.4s\n"
            "movi v20.4s, #0x0\n"
            "movi v10.4s, #0x0\n"
            "fmla v16.4s, v2.4s, v26.4s\n"
            "movi v26.4s, #0x0\n"
            "movi v2.4s, #0x0\n"
            ".inst 0x4f98e194  // sdot v20.4s, v12.16b, v24.4b[0]\n"
            ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
            ".inst 0x4f98e99a  // sdot v26.4s, v12.16b, v24.4b[2]\n"
            ".inst 0x4fb8e982  // sdot v2.4s, v12.16b, v24.4b[3]\n"
            "ldr q24, [x22, #0x20]\n"
            ".inst 0x4f89e3f4  // sdot v20.4s, v31.16b, v9.4b[0]\n"
            ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
            ".inst 0x4f89ebfa  // sdot v26.4s, v31.16b, v9.4b[2]\n"
            ".inst 0x4fa9ebe2  // sdot v2.4s, v31.16b, v9.4b[3]\n"
            "ldr q9, [x22, #0x30]\n"
            ".inst 0x4f98e0d4  // sdot v20.4s, v6.16b, v24.4b[0]\n"
            ".inst 0x4fb8e0ca  // sdot v10.4s, v6.16b, v24.4b[1]\n"
            ".inst 0x4f98e8da  // sdot v26.4s, v6.16b, v24.4b[2]\n"
            ".inst 0x4fb8e8c2  // sdot v2.4s, v6.16b, v24.4b[3]\n"
            "ldr q24, [x22, #0x40]\n"
            ".inst 0x4f89e394  // sdot v20.4s, v28.16b, v9.4b[0]\n"
            ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
            ".inst 0x4f89eb9a  // sdot v26.4s, v28.16b, v9.4b[2]\n"
            ".inst 0x4fa9eb82  // sdot v2.4s, v28.16b, v9.4b[3]\n"
            "ldr q9, [x22, #0x50]\n"
            ".inst 0x4f98e074  // sdot v20.4s, v3.16b, v24.4b[0]\n"
            ".inst 0x4fb8e06a  // sdot v10.4s, v3.16b, v24.4b[1]\n"
            ".inst 0x4f98e87a  // sdot v26.4s, v3.16b, v24.4b[2]\n"
            ".inst 0x4fb8e862  // sdot v2.4s, v3.16b, v24.4b[3]\n"
            "ldr q24, [x22, #0x60]\n"
            ".inst 0x4f89e2d4  // sdot v20.4s, v22.16b, v9.4b[0]\n"
            ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
            ".inst 0x4f89eada  // sdot v26.4s, v22.16b, v9.4b[2]\n"
            ".inst 0x4fa9eac2  // sdot v2.4s, v22.16b, v9.4b[3]\n"
            "ldr q9, [x22, #0x70]\n"
            "add x22, x22, #0x88\n"
            ".inst 0x4f98e374  // sdot v20.4s, v27.16b, v24.4b[0]\n"
            ".inst 0x4fb8e36a  // sdot v10.4s, v27.16b, v24.4b[1]\n"
            ".inst 0x4f98eb7a  // sdot v26.4s, v27.16b, v24.4b[2]\n"
            ".inst 0x4fb8eb62  // sdot v2.4s, v27.16b, v24.4b[3]\n"
            "ldr q24, [x21, #0x0]\n"
            ".inst 0x4f89e3d4  // sdot v20.4s, v30.16b, v9.4b[0]\n"
            ".inst 0x4fa9e3ca  // sdot v10.4s, v30.16b, v9.4b[1]\n"
            ".inst 0x4f89ebda  // sdot v26.4s, v30.16b, v9.4b[2]\n"
            ".inst 0x4fa9ebc2  // sdot v2.4s, v30.16b, v9.4b[3]\n"
            "fmul v9.4s, v17.4s, v29.s[0]\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "scvtf v10.4s, v10.4s, #0x4\n"
            "scvtf v26.4s, v26.4s, #0x4\n"
            "scvtf v2.4s, v2.4s, #0x4\n"
            "fmla v25.4s, v20.4s, v9.4s\n"
            "ldr q9, [x21, #0x10]\n"
            "fmul v20.4s, v17.4s, v29.s[1]\n"
            "fmla v7.4s, v10.4s, v20.4s\n"
            "ldr d20, [x21, #-0x8]\n"
            "fmul v10.4s, v17.4s, v29.s[2]\n"
            "fmul v29.4s, v17.4s, v29.s[3]\n"
            "fcvtl v20.4s, v20.4h\n"
            "fmla v0.4s, v26.4s, v10.4s\n"
            "movi v26.4s, #0x0\n"
            "movi v10.4s, #0x0\n"
            "fmla v4.4s, v2.4s, v29.4s\n"
            "movi v2.4s, #0x0\n"
            "movi v29.4s, #0x0\n"
            ".inst 0x4f98e19a  // sdot v26.4s, v12.16b, v24.4b[0]\n"
            ".inst 0x4fb8e18a  // sdot v10.4s, v12.16b, v24.4b[1]\n"
            ".inst 0x4f98e982  // sdot v2.4s, v12.16b, v24.4b[2]\n"
            ".inst 0x4fb8e99d  // sdot v29.4s, v12.16b, v24.4b[3]\n"
            "ldr q12, [x21, #0x20]\n"
            "fmul v24.4s, v17.4s, v20.s[0]\n"
            ".inst 0x4f89e3fa  // sdot v26.4s, v31.16b, v9.4b[0]\n"
            ".inst 0x4fa9e3ea  // sdot v10.4s, v31.16b, v9.4b[1]\n"
            ".inst 0x4f89ebe2  // sdot v2.4s, v31.16b, v9.4b[2]\n"
            ".inst 0x4fa9ebfd  // sdot v29.4s, v31.16b, v9.4b[3]\n"
            "ldr q9, [x21, #0x30]\n"
            "fmul v31.4s, v17.4s, v20.s[1]\n"
            ".inst 0x4f8ce0da  // sdot v26.4s, v6.16b, v12.4b[0]\n"
            ".inst 0x4face0ca  // sdot v10.4s, v6.16b, v12.4b[1]\n"
            ".inst 0x4f8ce8c2  // sdot v2.4s, v6.16b, v12.4b[2]\n"
            ".inst 0x4face8dd  // sdot v29.4s, v6.16b, v12.4b[3]\n"
            "ldr q12, [x21, #0x40]\n"
            "fmul v6.4s, v17.4s, v20.s[2]\n"
            "fmul v20.4s, v17.4s, v20.s[3]\n"
            ".inst 0x4f89e39a  // sdot v26.4s, v28.16b, v9.4b[0]\n"
            ".inst 0x4fa9e38a  // sdot v10.4s, v28.16b, v9.4b[1]\n"
            ".inst 0x4f89eb82  // sdot v2.4s, v28.16b, v9.4b[2]\n"
            ".inst 0x4fa9eb9d  // sdot v29.4s, v28.16b, v9.4b[3]\n"
            "ldr q9, [x21, #0x50]\n"
            ".inst 0x4f8ce07a  // sdot v26.4s, v3.16b, v12.4b[0]\n"
            ".inst 0x4face06a  // sdot v10.4s, v3.16b, v12.4b[1]\n"
            ".inst 0x4f8ce862  // sdot v2.4s, v3.16b, v12.4b[2]\n"
            ".inst 0x4face87d  // sdot v29.4s, v3.16b, v12.4b[3]\n"
            "ldr q12, [x21, #0x60]\n"
            ".inst 0x4f89e2da  // sdot v26.4s, v22.16b, v9.4b[0]\n"
            ".inst 0x4fa9e2ca  // sdot v10.4s, v22.16b, v9.4b[1]\n"
            ".inst 0x4f89eac2  // sdot v2.4s, v22.16b, v9.4b[2]\n"
            ".inst 0x4fa9eadd  // sdot v29.4s, v22.16b, v9.4b[3]\n"
            "ldr q17, [x21, #0x70]\n"
            "add x21, x21, #0x88\n"
            ".inst 0x4f8ce37a  // sdot v26.4s, v27.16b, v12.4b[0]\n"
            ".inst 0x4face36a  // sdot v10.4s, v27.16b, v12.4b[1]\n"
            ".inst 0x4f8ceb62  // sdot v2.4s, v27.16b, v12.4b[2]\n"
            ".inst 0x4faceb7d  // sdot v29.4s, v27.16b, v12.4b[3]\n"
            ".inst 0x4f91e3da  // sdot v26.4s, v30.16b, v17.4b[0]\n"
            ".inst 0x4fb1e3ca  // sdot v10.4s, v30.16b, v17.4b[1]\n"
            ".inst 0x4f91ebc2  // sdot v2.4s, v30.16b, v17.4b[2]\n"
            ".inst 0x4fb1ebdd  // sdot v29.4s, v30.16b, v17.4b[3]\n"
            "scvtf v26.4s, v26.4s, #0x4\n"
            "scvtf v10.4s, v10.4s, #0x4\n"
            "fmla v5.4s, v26.4s, v24.4s\n"
            "scvtf v2.4s, v2.4s, #0x4\n"
            "scvtf v29.4s, v29.4s, #0x4\n"
            "fmla v21.4s, v10.4s, v31.4s\n"
            "fmla v8.4s, v2.4s, v6.4s\n"
            "fmla v1.4s, v29.4s, v20.4s\n"
            "bgt 3b\n"
            "mov x20, %x[res_ptr]\n"
            "subs x27, x27, #0x4\n"
            "add %x[res_ptr], %x[res_ptr], #0x10\n"
            "str q15, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q19, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q18, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q14, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q11, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q13, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q23, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q16, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q25, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q7, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q0, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q4, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q5, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q21, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q8, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q1, [x20, #0x0]\n"
            "bne 2b\n"
            "mov x20, #0x4\n"
            "sub x10, x10, #0x10\n"
            "cmp x10, #0x10\n"
            "mov %x[res_ptr], x26\n"
            "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
            "bge 1b\n"
            "4:"  // Row loop skip
            "cbz x10, 9f\n"
            "5:"  // Row tail: Row loop
            "add x24, %x[b_ptr], #0x8\n"
            "mov x23, %x[nc]\n"
            "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
            "6:"  // Row tail: Column loop
            "movi v15.16b, #0x0\n"
            "movi v19.16b, #0x0\n"
            "add x25, %x[a_ptr], #0x8\n"
            "mov x21, %x[nb]\n"
            "movi v18.16b, #0x0\n"
            "movi v14.16b, #0x0\n"
            "7:"  // Row tail: Block loop
            "ldr q7, [x24, #0x0]\n"
            "ldr q5, [x25, #0x0]\n"
            "movi v9.16b, #0x4\n"
            "movi v4.4s, #0x0\n"
            "ldr q3, [x24, #0x10]\n"
            "ldr q2, [x25, #0x10]\n"
            "movi v1.4s, #0x0\n"
            "movi v0.4s, #0x0\n"
            "ldr q13, [x24, #0x20]\n"
            "ldr q31, [x25, #0x20]\n"
            "movi v30.4s, #0x0\n"
            "movi v29.16b, #0xf0\n"
            "ldr q28, [x24, #0x30]\n"
            "ldr q27, [x25, #0x30]\n"
            "sshl v20.16b, v7.16b, v9.16b\n"
            "sub x20, x24, #0x8\n"
            "ldr q26, [x25, #0x40]\n"
            "ldr q25, [x25, #0x50]\n"
            "sshl v17.16b, v3.16b, v9.16b\n"
            "and v7.16b, v7.16b, v29.16b\n"
            "ldr q24, [x25, #0x60]\n"
            "ldr q16, [x25, #0x70]\n"
            "sshl v22.16b, v13.16b, v9.16b\n"
            "and v3.16b, v3.16b, v29.16b\n"
            "ldr d21, [x20, #0x0]\n"
            "ldr d12, [x25, #-0x8]\n"
            ".inst 0x4f85e284  // sdot v4.4s, v20.16b, v5.4b[0]\n"
            ".inst 0x4fa5e281  // sdot v1.4s, v20.16b, v5.4b[1]\n"
            ".inst 0x4f85ea80  // sdot v0.4s, v20.16b, v5.4b[2]\n"
            ".inst 0x4fa5ea9e  // sdot v30.4s, v20.16b, v5.4b[3]\n"
            "sshl v9.16b, v28.16b, v9.16b\n"
            "subs x21, x21, #0x1\n"
            "and v13.16b, v13.16b, v29.16b\n"
            "and v28.16b, v28.16b, v29.16b\n"
            "add x25, x25, #0x88\n"
            "add x24, x24, #0x48\n"
            "fcvtl v21.4s, v21.4h\n"
            "fcvtl v12.4s, v12.4h\n"
            ".inst 0x4f82e224  // sdot v4.4s, v17.16b, v2.4b[0]\n"
            ".inst 0x4fa2e221  // sdot v1.4s, v17.16b, v2.4b[1]\n"
            ".inst 0x4f82ea20  // sdot v0.4s, v17.16b, v2.4b[2]\n"
            ".inst 0x4fa2ea3e  // sdot v30.4s, v17.16b, v2.4b[3]\n"
            "fmul v11.4s, v21.4s, v12.s[0]\n"
            "fmul v23.4s, v21.4s, v12.s[1]\n"
            "fmul v17.4s, v21.4s, v12.s[2]\n"
            ".inst 0x4f9fe2c4  // sdot v4.4s, v22.16b, v31.4b[0]\n"
            "fmul v6.4s, v21.4s, v12.s[3]\n"
            ".inst 0x4fbfe2c1  // sdot v1.4s, v22.16b, v31.4b[1]\n"
            ".inst 0x4f9feac0  // sdot v0.4s, v22.16b, v31.4b[2]\n"
            ".inst 0x4fbfeade  // sdot v30.4s, v22.16b, v31.4b[3]\n"
            ".inst 0x4f9be124  // sdot v4.4s, v9.16b, v27.4b[0]\n"
            ".inst 0x4fbbe121  // sdot v1.4s, v9.16b, v27.4b[1]\n"
            ".inst 0x4f9be920  // sdot v0.4s, v9.16b, v27.4b[2]\n"
            ".inst 0x4fbbe93e  // sdot v30.4s, v9.16b, v27.4b[3]\n"
            ".inst 0x4f9ae0e4  // sdot v4.4s, v7.16b, v26.4b[0]\n"
            ".inst 0x4fbae0e1  // sdot v1.4s, v7.16b, v26.4b[1]\n"
            ".inst 0x4f9ae8e0  // sdot v0.4s, v7.16b, v26.4b[2]\n"
            ".inst 0x4fbae8fe  // sdot v30.4s, v7.16b, v26.4b[3]\n"
            ".inst 0x4f99e064  // sdot v4.4s, v3.16b, v25.4b[0]\n"
            ".inst 0x4fb9e061  // sdot v1.4s, v3.16b, v25.4b[1]\n"
            ".inst 0x4f99e860  // sdot v0.4s, v3.16b, v25.4b[2]\n"
            ".inst 0x4fb9e87e  // sdot v30.4s, v3.16b, v25.4b[3]\n"
            ".inst 0x4f98e1a4  // sdot v4.4s, v13.16b, v24.4b[0]\n"
            ".inst 0x4fb8e1a1  // sdot v1.4s, v13.16b, v24.4b[1]\n"
            ".inst 0x4f98e9a0  // sdot v0.4s, v13.16b, v24.4b[2]\n"
            ".inst 0x4fb8e9be  // sdot v30.4s, v13.16b, v24.4b[3]\n"
            ".inst 0x4f90e384  // sdot v4.4s, v28.16b, v16.4b[0]\n"
            ".inst 0x4fb0e381  // sdot v1.4s, v28.16b, v16.4b[1]\n"
            ".inst 0x4f90eb80  // sdot v0.4s, v28.16b, v16.4b[2]\n"
            ".inst 0x4fb0eb9e  // sdot v30.4s, v28.16b, v16.4b[3]\n"
            "scvtf v4.4s, v4.4s, #0x4\n"
            "scvtf v1.4s, v1.4s, #0x4\n"
            "scvtf v0.4s, v0.4s, #0x4\n"
            "fmla v15.4s, v4.4s, v11.4s\n"
            "scvtf v30.4s, v30.4s, #0x4\n"
            "fmla v19.4s, v1.4s, v23.4s\n"
            "fmla v18.4s, v0.4s, v17.4s\n"
            "fmla v14.4s, v30.4s, v6.4s\n"
            "bgt 7b\n"
            "mov x20, %x[res_ptr]\n"
            "cmp x10, #0x1\n"
            "str q15, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x10, #0x2\n"
            "str q19, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x10, #0x3\n"
            "str q18, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "str q14, [x20, #0x0]\n"
            "8:"  // Row tail: Accumulator store skip
            "subs x23, x23, #0x4\n"
            "add %x[res_ptr], %x[res_ptr], #0x10\n"
            "bne 6b\n"
            "subs x10, x10, #0x4\n"
            "add %x[a_ptr], %x[a_ptr], x9\n"
            "mov %x[res_ptr], x22\n"
            "bgt 5b\n"
            "9:"  // Row tail: Row loop skip
            : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
            : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
        );
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

static void ggml_gemm_q4_0_4x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__(
            "mov x10, %x[nr]\n"
            "mov x9, #0x88\n"
            "cmp x10, #0x10\n"
            "mul x9, %x[nb], x9\n"
            "blt 4f\n"
            "1:"  // Row loop
            "add x28, %x[b_ptr], #0x8\n"
            "mov x27, %x[nc]\n"
            "add x26, %x[res_ptr], %x[res_stride], LSL #4\n"
            "2:"  // Column loop
            "add x25, %x[a_ptr], #0x8\n"
            "movi v2.16b, #0x0\n"
            "movi v10.16b, #0x0\n"
            "mov x24, %x[nb]\n"
            "add x23, x25, x9\n"
            "movi v12.16b, #0x0\n"
            "movi v28.16b, #0x0\n"
            "add x22, x23, x9\n"
            "movi v11.16b, #0x0\n"
            "movi v13.16b, #0x0\n"
            "add x21, x22, x9\n"
            "movi v22.16b, #0x0\n"
            "movi v23.16b, #0x0\n"
            "movi v25.16b, #0x0\n"
            "movi v5.16b, #0x0\n"
            "movi v7.16b, #0x0\n"
            "movi v4.16b, #0x0\n"
            "movi v6.16b, #0x0\n"
            "movi v30.16b, #0x0\n"
            "movi v24.16b, #0x0\n"
            "movi v14.16b, #0x0\n"
            "3:"  // Block loop
            "ldr q21, [x28, #0x0]\n"
            "ldr q16, [x28, #0x10]\n"
            "movi v1.16b, #0x4\n"
            "movi v19.4s, #0x0\n"
            "ldr q27, [x25, #0x0]\n"
            "ldr q15, [x25, #0x10]\n"
            "movi v26.4s, #0x0\n"
            "movi v18.4s, #0x0\n"
            "ldr q29, [x28, #0x20]\n"
            "ldr q3, [x28, #0x30]\n"
            "movi v17.4s, #0x0\n"
            "movi v0.16b, #0xf0\n"
            "ldr d20, [x25, #-0x8]\n"
            "ldr d9, [x23, #-0x8]\n"
            "sshl v8.16b, v21.16b, v1.16b\n"
            "sshl v31.16b, v16.16b, v1.16b\n"
            "and v21.16b, v21.16b, v0.16b\n"
            "and v16.16b, v16.16b, v0.16b\n"
            "sub x20, x28, #0x8\n"
            "subs x24, x24, #0x1\n"
            "add x28, x28, #0x48\n"
            ".inst 0x4e88a773  // smmla v19.4s, v27.16b, v8.16b\n"
            ".inst 0x4e9fa77a  // smmla v26.4s, v27.16b, v31.16b\n"
            "ldr q27, [x25, #0x20]\n"
            ".inst 0x4e88a5f2  // smmla v18.4s, v15.16b, v8.16b\n"
            ".inst 0x4e9fa5f1  // smmla v17.4s, v15.16b, v31.16b\n"
            "sshl v15.16b, v29.16b, v1.16b\n"
            "sshl v1.16b, v3.16b, v1.16b\n"
            "and v29.16b, v29.16b, v0.16b\n"
            "and v3.16b, v3.16b, v0.16b\n"
            "ldr q0, [x25, #0x30]\n"
            "fcvtl v20.4s, v20.4h\n"
            ".inst 0x4e8fa773  // smmla v19.4s, v27.16b, v15.16b\n"
            "fcvtl v9.4s, v9.4h\n"
            ".inst 0x4e81a77a  // smmla v26.4s, v27.16b, v1.16b\n"
            "ldr q27, [x25, #0x40]\n"
            ".inst 0x4e8fa412  // smmla v18.4s, v0.16b, v15.16b\n"
            ".inst 0x4e81a411  // smmla v17.4s, v0.16b, v1.16b\n"
            "ldr q0, [x25, #0x50]\n"
            ".inst 0x4e95a773  // smmla v19.4s, v27.16b, v21.16b\n"
            ".inst 0x4e90a77a  // smmla v26.4s, v27.16b, v16.16b\n"
            "ldr q27, [x25, #0x60]\n"
            ".inst 0x4e95a412  // smmla v18.4s, v0.16b, v21.16b\n"
            ".inst 0x4e90a411  // smmla v17.4s, v0.16b, v16.16b\n"
            "ldr q0, [x25, #0x70]\n"
            "add x25, x25, #0x88\n"
            ".inst 0x4e9da773  // smmla v19.4s, v27.16b, v29.16b\n"
            ".inst 0x4e83a77a  // smmla v26.4s, v27.16b, v3.16b\n"
            "ldr d27, [x20, #0x0]\n"
            ".inst 0x4e9da412  // smmla v18.4s, v0.16b, v29.16b\n"
            ".inst 0x4e83a411  // smmla v17.4s, v0.16b, v3.16b\n"
            "fcvtl v27.4s, v27.4h\n"
            "uzp1 v0.2d, v19.2d, v26.2d\n"
            "uzp2 v26.2d, v19.2d, v26.2d\n"
            "fmul v19.4s, v27.4s, v20.s[0]\n"
            "scvtf v0.4s, v0.4s, #0x4\n"
            "scvtf v26.4s, v26.4s, #0x4\n"
            "fmla v2.4s, v0.4s, v19.4s\n"
            "ldr q19, [x23, #0x0]\n"
            "uzp1 v0.2d, v18.2d, v17.2d\n"
            "uzp2 v18.2d, v18.2d, v17.2d\n"
            "fmul v17.4s, v27.4s, v20.s[1]\n"
            "scvtf v0.4s, v0.4s, #0x4\n"
            "scvtf v18.4s, v18.4s, #0x4\n"
            "fmla v10.4s, v26.4s, v17.4s\n"
            "ldr q17, [x23, #0x10]\n"
            "fmul v26.4s, v27.4s, v20.s[2]\n"
            "fmul v20.4s, v27.4s, v20.s[3]\n"
            "fmla v12.4s, v0.4s, v26.4s\n"
            "ldr d0, [x22, #-0x8]\n"
            "ldr d26, [x21, #-0x8]\n"
            "fcvtl v0.4s, v0.4h\n"
            "fmla v28.4s, v18.4s, v20.4s\n"
            "movi v20.4s, #0x0\n"
            "movi v18.4s, #0x0\n"
            ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
            ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
            "ldr q19, [x23, #0x20]\n"
            "fcvtl v26.4s, v26.4h\n"
            ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
            ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
            "ldr q19, [x23, #0x40]\n"
            ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
            ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
            "ldr q19, [x23, #0x60]\n"
            ".inst 0x4e9da674  // smmla v20.4s, v19.16b, v29.16b\n"
            ".inst 0x4e83a672  // smmla v18.4s, v19.16b, v3.16b\n"
            "uzp1 v19.2d, v20.2d, v18.2d\n"
            "scvtf v19.4s, v19.4s, #0x4\n"
            "uzp2 v20.2d, v20.2d, v18.2d\n"
            "fmul v18.4s, v27.4s, v9.s[0]\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "fmla v11.4s, v19.4s, v18.4s\n"
            "ldr q18, [x22, #0x0]\n"
            "fmul v19.4s, v27.4s, v9.s[1]\n"
            "fmla v13.4s, v20.4s, v19.4s\n"
            "movi v19.4s, #0x0\n"
            "movi v20.4s, #0x0\n"
            ".inst 0x4e88a633  // smmla v19.4s, v17.16b, v8.16b\n"
            ".inst 0x4e9fa634  // smmla v20.4s, v17.16b, v31.16b\n"
            "ldr q17, [x23, #0x30]\n"
            ".inst 0x4e8fa633  // smmla v19.4s, v17.16b, v15.16b\n"
            ".inst 0x4e81a634  // smmla v20.4s, v17.16b, v1.16b\n"
            "ldr q17, [x23, #0x50]\n"
            ".inst 0x4e95a633  // smmla v19.4s, v17.16b, v21.16b\n"
            ".inst 0x4e90a634  // smmla v20.4s, v17.16b, v16.16b\n"
            "ldr q17, [x23, #0x70]\n"
            "add x23, x23, #0x88\n"
            ".inst 0x4e9da633  // smmla v19.4s, v17.16b, v29.16b\n"
            ".inst 0x4e83a634  // smmla v20.4s, v17.16b, v3.16b\n"
            "uzp1 v17.2d, v19.2d, v20.2d\n"
            "scvtf v17.4s, v17.4s, #0x4\n"
            "uzp2 v20.2d, v19.2d, v20.2d\n"
            "fmul v19.4s, v27.4s, v9.s[2]\n"
            "fmul v9.4s, v27.4s, v9.s[3]\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "fmla v22.4s, v17.4s, v19.4s\n"
            "ldr q17, [x22, #0x10]\n"
            "movi v19.4s, #0x0\n"
            ".inst 0x4e88a653  // smmla v19.4s, v18.16b, v8.16b\n"
            "fmla v23.4s, v20.4s, v9.4s\n"
            "movi v20.4s, #0x0\n"
            "movi v9.4s, #0x0\n"
            ".inst 0x4e9fa654  // smmla v20.4s, v18.16b, v31.16b\n"
            "ldr q18, [x22, #0x20]\n"
            ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
            ".inst 0x4e8fa653  // smmla v19.4s, v18.16b, v15.16b\n"
            ".inst 0x4e81a654  // smmla v20.4s, v18.16b, v1.16b\n"
            "ldr q18, [x22, #0x40]\n"
            ".inst 0x4e95a653  // smmla v19.4s, v18.16b, v21.16b\n"
            ".inst 0x4e90a654  // smmla v20.4s, v18.16b, v16.16b\n"
            "ldr q18, [x22, #0x60]\n"
            ".inst 0x4e9da653  // smmla v19.4s, v18.16b, v29.16b\n"
            ".inst 0x4e83a654  // smmla v20.4s, v18.16b, v3.16b\n"
            "movi v18.4s, #0x0\n"
            ".inst 0x4e9fa632  // smmla v18.4s, v17.16b, v31.16b\n"
            "ldr q17, [x22, #0x30]\n"
            ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
            ".inst 0x4e81a632  // smmla v18.4s, v17.16b, v1.16b\n"
            "ldr q17, [x22, #0x50]\n"
            ".inst 0x4e95a629  // smmla v9.4s, v17.16b, v21.16b\n"
            ".inst 0x4e90a632  // smmla v18.4s, v17.16b, v16.16b\n"
            "ldr q17, [x22, #0x70]\n"
            "add x22, x22, #0x88\n"
            ".inst 0x4e9da629  // smmla v9.4s, v17.16b, v29.16b\n"
            ".inst 0x4e83a632  // smmla v18.4s, v17.16b, v3.16b\n"
            "uzp1 v17.2d, v19.2d, v20.2d\n"
            "uzp2 v20.2d, v19.2d, v20.2d\n"
            "fmul v19.4s, v27.4s, v0.s[0]\n"
            "scvtf v17.4s, v17.4s, #0x4\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "fmla v25.4s, v17.4s, v19.4s\n"
            "ldr q19, [x21, #0x0]\n"
            "fmul v17.4s, v27.4s, v0.s[1]\n"
            "fmla v5.4s, v20.4s, v17.4s\n"
            "ldr q17, [x21, #0x10]\n"
            "uzp1 v20.2d, v9.2d, v18.2d\n"
            "uzp2 v9.2d, v9.2d, v18.2d\n"
            "fmul v18.4s, v27.4s, v0.s[2]\n"
            "fmul v0.4s, v27.4s, v0.s[3]\n"
            "scvtf v20.4s, v20.4s, #0x4\n"
            "scvtf v9.4s, v9.4s, #0x4\n"
            "fmla v7.4s, v20.4s, v18.4s\n"
            "movi v20.4s, #0x0\n"
            "movi v18.4s, #0x0\n"
            ".inst 0x4e88a674  // smmla v20.4s, v19.16b, v8.16b\n"
            ".inst 0x4e9fa672  // smmla v18.4s, v19.16b, v31.16b\n"
            "ldr q19, [x21, #0x20]\n"
            "fmla v4.4s, v9.4s, v0.4s\n"
            "movi v9.4s, #0x0\n"
            "movi v0.4s, #0x0\n"
            ".inst 0x4e88a629  // smmla v9.4s, v17.16b, v8.16b\n"
            "fmul v8.4s, v27.4s, v26.s[0]\n"
            ".inst 0x4e9fa620  // smmla v0.4s, v17.16b, v31.16b\n"
            "ldr q17, [x21, #0x30]\n"
            ".inst 0x4e8fa674  // smmla v20.4s, v19.16b, v15.16b\n"
            "fmul v31.4s, v27.4s, v26.s[1]\n"
            ".inst 0x4e81a672  // smmla v18.4s, v19.16b, v1.16b\n"
            "ldr q19, [x21, #0x40]\n"
            ".inst 0x4e8fa629  // smmla v9.4s, v17.16b, v15.16b\n"
            "fmul v15.4s, v27.4s, v26.s[2]\n"
            "fmul v27.4s, v27.4s, v26.s[3]\n"
            ".inst 0x4e81a620  // smmla v0.4s, v17.16b, v1.16b\n"
            "ldr q1, [x21, #0x50]\n"
            ".inst 0x4e95a674  // smmla v20.4s, v19.16b, v21.16b\n"
            ".inst 0x4e90a672  // smmla v18.4s, v19.16b, v16.16b\n"
            "ldr q26, [x21, #0x60]\n"
            ".inst 0x4e95a429  // smmla v9.4s, v1.16b, v21.16b\n"
            ".inst 0x4e90a420  // smmla v0.4s, v1.16b, v16.16b\n"
            "ldr q21, [x21, #0x70]\n"
            "add x21, x21, #0x88\n"
            ".inst 0x4e9da754  // smmla v20.4s, v26.16b, v29.16b\n"
            ".inst 0x4e83a752  // smmla v18.4s, v26.16b, v3.16b\n"
            ".inst 0x4e9da6a9  // smmla v9.4s, v21.16b, v29.16b\n"
            ".inst 0x4e83a6a0  // smmla v0.4s, v21.16b, v3.16b\n"
            "uzp1 v29.2d, v20.2d, v18.2d\n"
            "uzp2 v21.2d, v20.2d, v18.2d\n"
            "scvtf v29.4s, v29.4s, #0x4\n"
            "uzp1 v18.2d, v9.2d, v0.2d\n"
            "uzp2 v16.2d, v9.2d, v0.2d\n"
            "scvtf v21.4s, v21.4s, #0x4\n"
            "fmla v6.4s, v29.4s, v8.4s\n"
            "scvtf v18.4s, v18.4s, #0x4\n"
            "scvtf v16.4s, v16.4s, #0x4\n"
            "fmla v30.4s, v21.4s, v31.4s\n"
            "fmla v24.4s, v18.4s, v15.4s\n"
            "fmla v14.4s, v16.4s, v27.4s\n"
            "bgt 3b\n"
            "mov x20, %x[res_ptr]\n"
            "subs x27, x27, #0x4\n"
            "add %x[res_ptr], %x[res_ptr], #0x10\n"
            "str q2, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q10, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q12, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q28, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q11, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q13, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q22, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q23, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q25, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q5, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q7, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q4, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q6, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q30, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q24, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "str q14, [x20, #0x0]\n"
            "bne 2b\n"
            "mov x20, #0x4\n"
            "sub x10, x10, #0x10\n"
            "cmp x10, #0x10\n"
            "mov %x[res_ptr], x26\n"
            "madd %x[a_ptr], x20, x9, %x[a_ptr]\n"
            "bge 1b\n"
            "4:"  // Row loop skip
            "cbz x10, 9f\n"
            "5:"  // Row tail: Row loop
            "add x24, %x[b_ptr], #0x8\n"
            "mov x23, %x[nc]\n"
            "add x22, %x[res_ptr], %x[res_stride], LSL #2\n"
            "6:"  // Row tail: Column loop
            "movi v2.16b, #0x0\n"
            "movi v10.16b, #0x0\n"
            "add x25, %x[a_ptr], #0x8\n"
            "mov x21, %x[nb]\n"
            "movi v12.16b, #0x0\n"
            "movi v28.16b, #0x0\n"
            "7:"  // Row tail: Block loop
            "ldr q6, [x24, #0x0]\n"
            "ldr q5, [x24, #0x10]\n"
            "movi v17.16b, #0x4\n"
            "movi v8.4s, #0x0\n"
            "ldr q4, [x25, #0x0]\n"
            "ldr q13, [x25, #0x10]\n"
            "movi v27.4s, #0x0\n"
            "movi v0.4s, #0x0\n"
            "ldr q31, [x24, #0x20]\n"
            "ldr q14, [x24, #0x30]\n"
            "movi v29.4s, #0x0\n"
            "movi v22.16b, #0xf0\n"
            "ldr q11, [x25, #0x20]\n"
            "ldr q23, [x25, #0x30]\n"
            "sshl v21.16b, v6.16b, v17.16b\n"
            "sshl v16.16b, v5.16b, v17.16b\n"
            "ldr q20, [x25, #0x40]\n"
            "ldr q26, [x25, #0x50]\n"
            "and v6.16b, v6.16b, v22.16b\n"
            "and v5.16b, v5.16b, v22.16b\n"
            "ldr q25, [x25, #0x60]\n"
            "ldr q3, [x25, #0x70]\n"
            "sshl v19.16b, v31.16b, v17.16b\n"
            "sshl v18.16b, v14.16b, v17.16b\n"
            "ldr d17, [x25, #-0x8]\n"
            ".inst 0x4e95a488  // smmla v8.4s, v4.16b, v21.16b\n"
            ".inst 0x4e90a49b  // smmla v27.4s, v4.16b, v16.16b\n"
            "and v31.16b, v31.16b, v22.16b\n"
            ".inst 0x4e95a5a0  // smmla v0.4s, v13.16b, v21.16b\n"
            ".inst 0x4e90a5bd  // smmla v29.4s, v13.16b, v16.16b\n"
            "and v14.16b, v14.16b, v22.16b\n"
            "sub x20, x24, #0x8\n"
            "ldr d16, [x20, #0x0]\n"
            "subs x21, x21, #0x1\n"
            "add x25, x25, #0x88\n"
            "fcvtl v17.4s, v17.4h\n"
            "add x24, x24, #0x48\n"
            ".inst 0x4e93a568  // smmla v8.4s, v11.16b, v19.16b\n"
            ".inst 0x4e92a57b  // smmla v27.4s, v11.16b, v18.16b\n"
            ".inst 0x4e93a6e0  // smmla v0.4s, v23.16b, v19.16b\n"
            ".inst 0x4e92a6fd  // smmla v29.4s, v23.16b, v18.16b\n"
            "fcvtl v16.4s, v16.4h\n"
            ".inst 0x4e86a688  // smmla v8.4s, v20.16b, v6.16b\n"
            ".inst 0x4e85a69b  // smmla v27.4s, v20.16b, v5.16b\n"
            "fmul v23.4s, v16.4s, v17.s[0]\n"
            "fmul v21.4s, v16.4s, v17.s[1]\n"
            "fmul v1.4s, v16.4s, v17.s[2]\n"
            "fmul v20.4s, v16.4s, v17.s[3]\n"
            ".inst 0x4e86a740  // smmla v0.4s, v26.16b, v6.16b\n"
            ".inst 0x4e85a75d  // smmla v29.4s, v26.16b, v5.16b\n"
            ".inst 0x4e9fa728  // smmla v8.4s, v25.16b, v31.16b\n"
            ".inst 0x4e8ea73b  // smmla v27.4s, v25.16b, v14.16b\n"
            ".inst 0x4e9fa460  // smmla v0.4s, v3.16b, v31.16b\n"
            ".inst 0x4e8ea47d  // smmla v29.4s, v3.16b, v14.16b\n"
            "uzp1 v19.2d, v8.2d, v27.2d\n"
            "uzp2 v18.2d, v8.2d, v27.2d\n"
            "scvtf v19.4s, v19.4s, #0x4\n"
            "uzp1 v17.2d, v0.2d, v29.2d\n"
            "uzp2 v16.2d, v0.2d, v29.2d\n"
            "scvtf v18.4s, v18.4s, #0x4\n"
            "fmla v2.4s, v19.4s, v23.4s\n"
            "scvtf v17.4s, v17.4s, #0x4\n"
            "scvtf v16.4s, v16.4s, #0x4\n"
            "fmla v10.4s, v18.4s, v21.4s\n"
            "fmla v12.4s, v17.4s, v1.4s\n"
            "fmla v28.4s, v16.4s, v20.4s\n"
            "bgt 7b\n"
            "mov x20, %x[res_ptr]\n"
            "cmp x10, #0x1\n"
            "str q2, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x10, #0x2\n"
            "str q10, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x10, #0x3\n"
            "str q12, [x20, #0x0]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "str q28, [x20, #0x0]\n"
            "8:"  // Row tail: Accumulator store skip
            "subs x23, x23, #0x4\n"
            "add %x[res_ptr], %x[res_ptr], #0x10\n"
            "bne 6b\n"
            "subs x10, x10, #0x4\n"
            "add %x[a_ptr], %x[a_ptr], x9\n"
            "mov %x[res_ptr], x22\n"
            "bgt 5b\n"
            "9:"  // Row tail: Row loop skip
            : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
            : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
            : "cc", "memory", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31", "x9", "x10", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28"
        );
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_MATMUL_INT8)
    float sumf[4][4];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x4 * b_ptr = (const block_q4_0x4 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

static void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
    if (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0) {
        const void * b_ptr = vx;
        const void * a_ptr = vy;
        float * res_ptr = s;
        size_t res_stride = bs * sizeof(float);

        __asm__ __volatile__(
            "mov x20, #0x4\n"
            "mov x13, %x[nr]\n"
            "mov z28.s, #-0x4\n"
            "mov x12, #0x88\n"
            "ptrue p1.b\n"
            "whilelt p0.s, XZR, x20\n"
            "cmp x13, #0x10\n"
            "mul x12, %x[nb], x12\n"
            "blt 4f\n"
            "1:"  // Row loop
            "add x11, %x[b_ptr], #0x10\n"
            "mov x10, %x[nc]\n"
            "add x9, %x[res_ptr], %x[res_stride], LSL #4\n"
            "2:"  // Column loop
            "add x28, %x[a_ptr], #0x8\n"
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "mov x27, %x[nb]\n"
            "add x26, x28, x12\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "add x25, x26, x12\n"
            "mov z13.b, #0x0\n"
            "mov z1.b, #0x0\n"
            "add x24, x25, x12\n"
            "mov z20.b, #0x0\n"
            "mov z25.b, #0x0\n"
            "mov z11.b, #0x0\n"
            "mov z16.b, #0x0\n"
            "mov z19.b, #0x0\n"
            "mov z26.b, #0x0\n"
            "mov z8.b, #0x0\n"
            "mov z29.b, #0x0\n"
            "mov z27.b, #0x0\n"
            "mov z10.b, #0x0\n"
            "3:"  // Block loop
            "ld1b { z30.b }, p1/Z, [x11]\n"
            "ld1b { z21.b }, p1/Z, [x11, #1, MUL VL]\n"
            "mov z18.s, #0x0\n"
            "mov z7.s, #0x0\n"
            "ld1rqb { z3.b }, p1/Z, [x28]\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #16]\n"
            "mov z9.s, #0x0\n"
            "mov z22.s, #0x0\n"
            "ld1b { z4.b }, p1/Z, [x11, #2, MUL VL]\n"
            "ld1b { z17.b }, p1/Z, [x11, #3, MUL VL]\n"
            "sub x20, x11, #0x10\n"
            "sub x23, x28, #0x8\n"
            "lsl z31.b, z30.b, #0x4\n"
            "lsl z6.b, z21.b, #0x4\n"
            "ld1h { z23.s }, p1/Z, [x20]\n"
            "sub x22, x26, #0x8\n"
            "and z30.b, z30.b, #0xf0\n"
            "and z21.b, z21.b, #0xf0\n"
            "sub x21, x25, #0x8\n"
            "sub x20, x24, #0x8\n"
            "lsl z14.b, z4.b, #0x4\n"
            "lsl z2.b, z17.b, #0x4\n"
            "subs x27, x27, #0x1\n"
            "add x11, x11, #0x90\n"
            ".inst 0x451f9872  // smmla z18.s, z3.b, z31.b\n"
            ".inst 0x45069867  // smmla z7.s, z3.b, z6.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #32]\n"
            "and z4.b, z4.b, #0xf0\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #48]\n"
            "and z17.b, z17.b, #0xf0\n"
            "fcvt z23.s, p1/m, z23.h\n"
            ".inst 0x450e9872  // smmla z18.s, z3.b, z14.b\n"
            ".inst 0x45029867  // smmla z7.s, z3.b, z2.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #64]\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #80]\n"
            "fscale z23.s, p1/m, z23.s, z28.s\n"
            ".inst 0x451e9872  // smmla z18.s, z3.b, z30.b\n"
            ".inst 0x45159867  // smmla z7.s, z3.b, z21.b\n"
            "ld1rqb { z3.b }, p1/Z, [x28, #96]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x28, #112]\n"
            "add x28, x28, #0x88\n"
            ".inst 0x45049872  // smmla z18.s, z3.b, z4.b\n"
            ".inst 0x45119867  // smmla z7.s, z3.b, z17.b\n"
            "ld1h { z3.s }, p0/Z, [x23]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "fcvt z3.s, p1/m, z3.h\n"
            "uzp1 z5.d, z18.d, z7.d\n"
            "uzp2 z18.d, z18.d, z7.d\n"
            "mov z3.q, z3.q[0]\n"
            "uzp1 z7.d, z9.d, z22.d\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z3.s[0]\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z24.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z5.b }, p1/Z, [x26]\n"
            "fmul z9.s, z23.s, z3.s[1]\n"
            "fmla z15.s, p1/M, z18.s, z9.s\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #16]\n"
            "fmul z9.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "fmla z12.s, p1/M, z7.s, z9.s\n"
            "mov z9.s, #0x0\n"
            "ld1h { z7.s }, p0/Z, [x22]\n"
            ".inst 0x451f98a9  // smmla z9.s, z5.b, z31.b\n"
            "fmla z0.s, p1/M, z22.s, z3.s\n"
            "mov z22.s, #0x0\n"
            "ld1h { z3.s }, p0/Z, [x21]\n"
            ".inst 0x450698b6  // smmla z22.s, z5.b, z6.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #32]\n"
            "fcvt z7.s, p1/m, z7.h\n"
            "fcvt z3.s, p1/m, z3.h\n"
            ".inst 0x450e98a9  // smmla z9.s, z5.b, z14.b\n"
            ".inst 0x450298b6  // smmla z22.s, z5.b, z2.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #64]\n"
            "mov z7.q, z7.q[0]\n"
            "mov z3.q, z3.q[0]\n"
            ".inst 0x451e98a9  // smmla z9.s, z5.b, z30.b\n"
            ".inst 0x451598b6  // smmla z22.s, z5.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x26, #96]\n"
            ".inst 0x450498a9  // smmla z9.s, z5.b, z4.b\n"
            ".inst 0x451198b6  // smmla z22.s, z5.b, z17.b\n"
            "uzp1 z5.d, z9.d, z22.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "uzp2 z22.d, z9.d, z22.d\n"
            "fmul z9.s, z23.s, z7.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z13.s, p1/M, z5.s, z9.s\n"
            "ld1rqb { z9.b }, p1/Z, [x25]\n"
            "fmul z5.s, z23.s, z7.s[1]\n"
            "fmla z1.s, p1/M, z22.s, z5.s\n"
            "mov z5.s, #0x0\n"
            "mov z22.s, #0x0\n"
            ".inst 0x451f9a45  // smmla z5.s, z18.b, z31.b\n"
            ".inst 0x45069a56  // smmla z22.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #48]\n"
            ".inst 0x450e9a45  // smmla z5.s, z18.b, z14.b\n"
            ".inst 0x45029a56  // smmla z22.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #80]\n"
            ".inst 0x451e9a45  // smmla z5.s, z18.b, z30.b\n"
            ".inst 0x45159a56  // smmla z22.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x26, #112]\n"
            "add x26, x26, #0x88\n"
            ".inst 0x45049a45  // smmla z5.s, z18.b, z4.b\n"
            ".inst 0x45119a56  // smmla z22.s, z18.b, z17.b\n"
            "uzp1 z18.d, z5.d, z22.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z22.d, z5.d, z22.d\n"
            "fmul z5.s, z23.s, z7.s[2]\n"
            "fmul z7.s, z23.s, z7.s[3]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z20.s, p1/M, z18.s, z5.s\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #16]\n"
            "ld1h { z5.s }, p0/Z, [x20]\n"
            "fcvt z5.s, p1/m, z5.h\n"
            "fmla z25.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9936  // smmla z22.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #32]\n"
            "mov z5.q, z5.q[0]\n"
            ".inst 0x450e9936  // smmla z22.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #64]\n"
            ".inst 0x451e9936  // smmla z22.s, z9.b, z30.b\n"
            ".inst 0x45159927  // smmla z7.s, z9.b, z21.b\n"
            "ld1rqb { z9.b }, p1/Z, [x25, #96]\n"
            ".inst 0x45049936  // smmla z22.s, z9.b, z4.b\n"
            ".inst 0x45119927  // smmla z7.s, z9.b, z17.b\n"
            "uzp1 z9.d, z22.d, z7.d\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "uzp2 z22.d, z22.d, z7.d\n"
            "fmul z7.s, z23.s, z3.s[0]\n"
            "scvtf z22.s, p1/m, z22.s\n"
            "fmla z11.s, p1/M, z9.s, z7.s\n"
            "ld1rqb { z9.b }, p1/Z, [x24]\n"
            "fmul z7.s, z23.s, z3.s[1]\n"
            "fmla z16.s, p1/M, z22.s, z7.s\n"
            "mov z22.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9a56  // smmla z22.s, z18.b, z31.b\n"
            ".inst 0x45069a47  // smmla z7.s, z18.b, z6.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #48]\n"
            ".inst 0x450e9a56  // smmla z22.s, z18.b, z14.b\n"
            ".inst 0x45029a47  // smmla z7.s, z18.b, z2.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #80]\n"
            ".inst 0x451e9a56  // smmla z22.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x25, #112]\n"
            "add x25, x25, #0x88\n"
            ".inst 0x45049a56  // smmla z22.s, z18.b, z4.b\n"
            ".inst 0x45119a47  // smmla z7.s, z18.b, z17.b\n"
            "uzp1 z18.d, z22.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp2 z7.d, z22.d, z7.d\n"
            "fmul z22.s, z23.s, z3.s[2]\n"
            "fmul z3.s, z23.s, z3.s[3]\n"
            "scvtf z7.s, p1/m, z7.s\n"
            "fmla z19.s, p1/M, z18.s, z22.s\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #16]\n"
            "fmul z22.s, z23.s, z5.s[0]\n"
            "fmla z26.s, p1/M, z7.s, z3.s\n"
            "mov z3.s, #0x0\n"
            "mov z7.s, #0x0\n"
            ".inst 0x451f9923  // smmla z3.s, z9.b, z31.b\n"
            ".inst 0x45069927  // smmla z7.s, z9.b, z6.b\n"
            "ld1rqb { z9.b }, p1/Z, [x24, #32]\n"
            ".inst 0x450e9923  // smmla z3.s, z9.b, z14.b\n"
            ".inst 0x45029927  // smmla z7.s, z9.b, z2.b\n"
            "mov z9.s, #0x0\n"
            ".inst 0x451f9a49  // smmla z9.s, z18.b, z31.b\n"
            "mov z31.s, #0x0\n"
            ".inst 0x45069a5f  // smmla z31.s, z18.b, z6.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #48]\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #64]\n"
            ".inst 0x450e98c9  // smmla z9.s, z6.b, z14.b\n"
            "fmul z14.s, z23.s, z5.s[1]\n"
            ".inst 0x450298df  // smmla z31.s, z6.b, z2.b\n"
            "ld1rqb { z6.b }, p1/Z, [x24, #80]\n"
            "fmul z2.s, z23.s, z5.s[2]\n"
            "fmul z23.s, z23.s, z5.s[3]\n"
            ".inst 0x451e9a43  // smmla z3.s, z18.b, z30.b\n"
            ".inst 0x45159a47  // smmla z7.s, z18.b, z21.b\n"
            "ld1rqb { z5.b }, p1/Z, [x24, #96]\n"
            ".inst 0x451e98c9  // smmla z9.s, z6.b, z30.b\n"
            ".inst 0x451598df  // smmla z31.s, z6.b, z21.b\n"
            "ld1rqb { z18.b }, p1/Z, [x24, #112]\n"
            "add x24, x24, #0x88\n"
            ".inst 0x450498a3  // smmla z3.s, z5.b, z4.b\n"
            ".inst 0x451198a7  // smmla z7.s, z5.b, z17.b\n"
            ".inst 0x45049a49  // smmla z9.s, z18.b, z4.b\n"
            ".inst 0x45119a5f  // smmla z31.s, z18.b, z17.b\n"
            "uzp1 z18.d, z3.d, z7.d\n"
            "uzp2 z5.d, z3.d, z7.d\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "uzp1 z6.d, z9.d, z31.d\n"
            "uzp2 z9.d, z9.d, z31.d\n"
            "scvtf z5.s, p1/m, z5.s\n"
            "fmla z8.s, p1/M, z18.s, z22.s\n"
            "scvtf z6.s, p1/m, z6.s\n"
            "scvtf z9.s, p1/m, z9.s\n"
            "fmla z29.s, p1/M, z5.s, z14.s\n"
            "fmla z27.s, p1/M, z6.s, z2.s\n"
            "fmla z10.s, p1/M, z9.s, z23.s\n"
            "bgt 3b\n"
            "mov x20, %x[res_ptr]\n"
            "subs x10, x10, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z0.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z13.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z1.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z20.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z25.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z11.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z16.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z19.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z26.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z8.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z29.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z27.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "st1w { z10.s }, p1, [x20]\n"
            "bne 2b\n"
            "mov x20, #0x4\n"
            "sub x13, x13, #0x10\n"
            "cmp x13, #0x10\n"
            "mov %x[res_ptr], x9\n"
            "madd %x[a_ptr], x20, x12, %x[a_ptr]\n"
            "bge 1b\n"
            "4:"  // Row loop skip
            "cbz x13, 9f\n"
            "5:"  // Row tail: Row loop
            "add x25, %x[b_ptr], #0x10\n"
            "mov x24, %x[nc]\n"
            "add x23, %x[res_ptr], %x[res_stride], LSL #2\n"
            "6:"  // Row tail: Column loop
            "mov z24.b, #0x0\n"
            "mov z15.b, #0x0\n"
            "add x28, %x[a_ptr], #0x8\n"
            "mov x22, %x[nb]\n"
            "mov z12.b, #0x0\n"
            "mov z0.b, #0x0\n"
            "7:"  // Row tail: Block loop
            "ld1b { z3.b }, p1/Z, [x25]\n"
            "ld1b { z6.b }, p1/Z, [x25, #1, MUL VL]\n"
            "mov z2.s, #0x0\n"
            "mov z25.s, #0x0\n"
            "ld1rqb { z26.b }, p1/Z, [x28]\n"
            "ld1rqb { z21.b }, p1/Z, [x28, #16]\n"
            "mov z27.s, #0x0\n"
            "mov z19.s, #0x0\n"
            "ld1b { z29.b }, p1/Z, [x25, #2, MUL VL]\n"
            "ld1b { z16.b }, p1/Z, [x25, #3, MUL VL]\n"
            "sub x21, x25, #0x10\n"
            "sub x20, x28, #0x8\n"
            "lsl z20.b, z3.b, #0x4\n"
            "lsl z4.b, z6.b, #0x4\n"
            "ld1rqb { z10.b }, p1/Z, [x28, #32]\n"
            "ld1rqb { z23.b }, p1/Z, [x28, #48]\n"
            "and z3.b, z3.b, #0xf0\n"
            "and z6.b, z6.b, #0xf0\n"
            "ld1rqb { z11.b }, p1/Z, [x28, #64]\n"
            "ld1rqb { z7.b }, p1/Z, [x28, #80]\n"
            "lsl z8.b, z29.b, #0x4\n"
            "lsl z14.b, z16.b, #0x4\n"
            "ld1rqb { z18.b }, p1/Z, [x28, #96]\n"
            "ld1rqb { z30.b }, p1/Z, [x28, #112]\n"
            ".inst 0x45149b42  // smmla z2.s, z26.b, z20.b\n"
            ".inst 0x45049b59  // smmla z25.s, z26.b, z4.b\n"
            "and z29.b, z29.b, #0xf0\n"
            "ld1h { z17.s }, p1/Z, [x21]\n"
            ".inst 0x45149abb  // smmla z27.s, z21.b, z20.b\n"
            ".inst 0x45049ab3  // smmla z19.s, z21.b, z4.b\n"
            "and z16.b, z16.b, #0xf0\n"
            "ld1h { z4.s }, p0/Z, [x20]\n"
            "subs x22, x22, #0x1\n"
            "add x28, x28, #0x88\n"
            "fcvt z17.s, p1/m, z17.h\n"
            "add x25, x25, #0x90\n"
            ".inst 0x45089942  // smmla z2.s, z10.b, z8.b\n"
            ".inst 0x450e9959  // smmla z25.s, z10.b, z14.b\n"
            "fcvt z4.s, p1/m, z4.h\n"
            ".inst 0x45089afb  // smmla z27.s, z23.b, z8.b\n"
            ".inst 0x450e9af3  // smmla z19.s, z23.b, z14.b\n"
            "fscale z17.s, p1/m, z17.s, z28.s\n"
            "mov z4.q, z4.q[0]\n"
            ".inst 0x45039962  // smmla z2.s, z11.b, z3.b\n"
            ".inst 0x45069979  // smmla z25.s, z11.b, z6.b\n"
            "fmul z23.s, z17.s, z4.s[0]\n"
            "fmul z9.s, z17.s, z4.s[1]\n"
            "fmul z21.s, z17.s, z4.s[2]\n"
            "fmul z4.s, z17.s, z4.s[3]\n"
            ".inst 0x450398fb  // smmla z27.s, z7.b, z3.b\n"
            ".inst 0x450698f3  // smmla z19.s, z7.b, z6.b\n"
            ".inst 0x451d9a42  // smmla z2.s, z18.b, z29.b\n"
            ".inst 0x45109a59  // smmla z25.s, z18.b, z16.b\n"
            ".inst 0x451d9bdb  // smmla z27.s, z30.b, z29.b\n"
            ".inst 0x45109bd3  // smmla z19.s, z30.b, z16.b\n"
            "uzp1 z31.d, z2.d, z25.d\n"
            "uzp2 z13.d, z2.d, z25.d\n"
            "scvtf z31.s, p1/m, z31.s\n"
            "uzp1 z17.d, z27.d, z19.d\n"
            "uzp2 z18.d, z27.d, z19.d\n"
            "scvtf z13.s, p1/m, z13.s\n"
            "fmla z24.s, p1/M, z31.s, z23.s\n"
            "scvtf z17.s, p1/m, z17.s\n"
            "scvtf z18.s, p1/m, z18.s\n"
            "fmla z15.s, p1/M, z13.s, z9.s\n"
            "fmla z12.s, p1/M, z17.s, z21.s\n"
            "fmla z0.s, p1/M, z18.s, z4.s\n"
            "bgt 7b\n"
            "mov x20, %x[res_ptr]\n"
            "cmp x13, #0x1\n"
            "st1w { z24.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x2\n"
            "st1w { z15.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "cmp x13, #0x3\n"
            "st1w { z12.s }, p1, [x20]\n"
            "add x20, x20, %x[res_stride]\n"
            "ble 8f\n"
            "st1w { z0.s }, p1, [x20]\n"
            "8:"  // Row tail: Accumulator store skip
            "subs x24, x24, #0x8\n"
            "add %x[res_ptr], %x[res_ptr], #0x20\n"
            "bne 6b\n"
            "subs x13, x13, #0x4\n"
            "add %x[a_ptr], %x[a_ptr], x12\n"
            "mov %x[res_ptr], x23\n"
            "bgt 5b\n"
            "9:"  // Row tail: Row loop skip
            : [a_ptr] "+&r" (a_ptr), [res_ptr] "+&r" (res_ptr)
            : [b_ptr] "r" (b_ptr), [nr] "r" (nr), [nb] "r" (nb), [res_stride] "r" (res_stride), [nc] "r" (nc)
            : "cc", "memory", "p0", "p1", "x9", "x10", "x11", "x12", "x13", "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "z8", "z9", "z10", "z11", "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24", "z25", "z26", "z27", "z28", "z29", "z30", "z31"
        );
        return;
    }
#endif // #if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)
#elif defined(__AVX2__) || defined(__AVX512F__)
    {
        const block_q4_0x8 * b_ptr_start = (const block_q4_0x8 *)vx;
        const block_q8_0x4 * a_ptr_start = (const block_q8_0x4 *)vy;
        int64_t b_nb = n / QK4_0;
        int64_t y = 0;
        // Mask to mask out nibbles from packed bytes
        const __m256i m4b = _mm256_set1_epi8(0x0F);
        const __m128i loadMask = _mm_blend_epi32(_mm_setzero_si128(), _mm_set1_epi32(0xFFFFFFFF), 3);
        // Lookup table to convert signed nibbles to signed bytes
        __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
        signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
        // Permute mask used for easier vector processing at later stages
        __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
        int64_t xstart = 0;
        int anr = nr - nr%16; // Used to align nr with boundary of 16
    #ifdef __AVX512F__
        int anc = nc - nc%16; // Used to align nc with boundary of 16
        // Mask to mask out nibbles from packed bytes expanded to 512 bit length
        const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
        // Lookup table to convert signed nibbles to signed bytes expanded to 512 bit length
        __m512i signextendlutexpanded = _mm512_inserti32x8(_mm512_castsi256_si512(signextendlut), signextendlut, 1);

        // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation
        for (; y < anr / 4; y += 4) {

            const block_q8_0x4 * a_ptrs[4];

            a_ptrs[0] = a_ptr_start + (y * nb);
            for (int i = 0; i < 3; ++i) {
                a_ptrs[i + 1] = a_ptrs[i] + nb;
            }

            // Take group of two block_q4_0x8 structures at each pass of the loop and perform dot product operation
            for (int64_t x = 0; x < anc / 8; x += 2) {

                const block_q4_0x8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
                const block_q4_0x8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

                // Master FP accumulators
                __m512 acc_rows[16];
                for (int i = 0; i < 16; i++) {
                    acc_rows[i] = _mm512_setzero_ps();
                }

                for (int64_t b = 0; b < nb; b++) {
                    // Load the sixteen block_q4_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                    // Save the values in the following vectors in the formats B0B1B4B5B8B9BCBD, B2B3B6B7BABBBEBF for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    // 4-bit -> 8-bit - Sign is maintained
                    const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                    const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                    const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                    const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                    const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                    const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                    const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                    const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                    const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                    const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                    const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                    const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                    const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                    const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                    const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                    // Shuffle pattern two - right side input

                    const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                    const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                    const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                    const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                    const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                    const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                    const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                    const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)

                    // Scale values - Load the weight scale values of two block_q4_0x8
                    const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                    // Process LHS in pairs of rows
                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q4_0 quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                        __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                        __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                        __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                        __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                        __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                        __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                        __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                        __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                        __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                        __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                        __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                        __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                        __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                        __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                        __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                        __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                        __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                        __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                        __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                        // Shuffle pattern one - left side input

                        const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                        const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                        const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                        const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                        const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                        const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                        const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                        const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                        // Shuffle pattern two - left side input

                        const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                        const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                        const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                        const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                        const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                        const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                        const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                        const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        // Resembles MMLAs into 2x2 matrices in ARM Version
                        __m512i iacc_mat_00_sp1 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1));
                        __m512i iacc_mat_01_sp1 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1));
                        __m512i iacc_mat_10_sp1 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1));
                        __m512i iacc_mat_11_sp1 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1));
                        __m512i iacc_mat_00_sp2 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2));
                        __m512i iacc_mat_01_sp2 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2));
                        __m512i iacc_mat_10_sp2 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2));
                        __m512i iacc_mat_11_sp2 =
                            _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                        __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                        __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                        __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                        // Straighten out to make 4 row vectors
                        __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                        __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                        // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                        const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptrs[rp][b].d), loadMask), 68);
                        const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                        // Multiply with appropiate scales and accumulate
                        acc_rows[rp * 4]     = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);
                    }
                }

                // Store the accumulated values
                for (int i = 0; i < 16; i++) {
                    _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
                }
            }
        }
        // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
        for (; y < nr / 4; y ++) {

            const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

            // Take group of two block_q4_0x8 structures at each pass of the loop and perform dot product operation
            for (int64_t x = 0; x < anc / 8; x += 2) {

                const block_q4_0x8 * b_ptr_0 = b_ptr_start + ((x)     * b_nb);
                const block_q4_0x8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

                // Master FP accumulators
                __m512 acc_rows[4];
                for (int i = 0; i < 4; i++) {
                    acc_rows[i] = _mm512_setzero_ps();
                }

                for (int64_t b = 0; b < nb; b++) {
                    // Load the sixteen block_q4_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....BE,BF
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 32));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 64));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_0[b].qs + 96));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 32));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 64));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i *)(b_ptr_1[b].qs + 96));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of valuess
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    // 4-bit -> 8-bit - Sign is maintained
                    const __m512i rhs_mat_014589CD_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7) B8(0-7) B9(0-7) BC(0-7) BD(0-7)
                    const __m512i rhs_mat_2367ABEF_0 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7) BA(0-7) BB(0-7) BE(0-7) BF(0-7)

                    const __m512i rhs_mat_014589CD_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15) B8(8-15) B9(8-15) BC(8-15) BD(8-15)
                    const __m512i rhs_mat_2367ABEF_1 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15) BA(8-15) BB(8-15) BE(8-15) BF(8-15)

                    const __m512i rhs_mat_014589CD_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23) B8(16-23) B9(16-23) BC(16-23) BD(16-23)
                    const __m512i rhs_mat_2367ABEF_2 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23) BA(16-23) BB(16-23) BE(16-23) BF(16-23)

                    const __m512i rhs_mat_014589CD_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31) B8(24-31) B9(24-31) BC(24-31) BD(24-31)
                    const __m512i rhs_mat_2367ABEF_3 = _mm512_shuffle_epi8(signextendlutexpanded, _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31) BA(24-31) BB(24-31) BE(24-31) BF(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_0_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)136); //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3) B8(0-3) B9(0-3) B8(0-3) B9(0-3) BC(0-3) BD(0-3) BC(0-3) BD(0-3)
                    const __m512i rhs_mat_2367ABEF_0_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)136); //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3) BA(0-3) BB(0-3) BA(0-3) BB(0-3) BE(0-3) BF(0-3) BE(0-3) BF(0-3)

                    const __m512i rhs_mat_014589CD_1_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)136); //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11) B8(8-11) B9(8-11) B8(8-11) B9(8-11) BC(8-11) BD(8-11) BC(8-11) BD(8-11)
                    const __m512i rhs_mat_2367ABEF_1_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)136); //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11) BA(8-11) BB(8-11) BA(8-11) BB(8-11) BE(8-11) BF(8-11) BE(8-11) BF(8-11)

                    const __m512i rhs_mat_014589CD_2_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)136); //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19) B8(16-19) B9(16-19) B8(16-19) B9(16-19) BC(16-19) BD(16-19) BC(16-19) BD(16-19)
                    const __m512i rhs_mat_2367ABEF_2_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)136); //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19) BA(16-19) BB(16-19) BA(16-19) BB(16-19) BE(16-19) BF(16-19) BE(16-19) BF(16-19)

                    const __m512i rhs_mat_014589CD_3_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)136); //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27) B8(24-27) B9(24-27) B8(24-27) B9(24-27) BC(24-27) BD(24-27) BC(24-27) BD(24-27)
                    const __m512i rhs_mat_2367ABEF_3_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)136); //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27) BA(24-27) BB(24-27) BA(24-27) BB(24-27) BE(24-27) BF(24-27) BE(24-27) BF(24-27)

                    // Shuffle pattern two - right side input

                    const __m512i rhs_mat_014589CD_0_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_0, (_MM_PERM_ENUM)221); //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7) B8(4-7) B9(4-7) B8(4-7) B9(4-7) BC(4-7) BD(4-7) BC(4-7) BD(4-7)
                    const __m512i rhs_mat_2367ABEF_0_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_0, (_MM_PERM_ENUM)221); //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7) BA(4-7) BB(4-7) BA(4-7) BB(4-7) BE(4-7) BF(4-7) BE(4-7) BF(4-7)

                    const __m512i rhs_mat_014589CD_1_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_1, (_MM_PERM_ENUM)221); //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15) B8(12-15) B9(12-15) B8(12-15) B9(12-15) BC(12-15) BD(12-15) BC(12-15) BD(12-15)
                    const __m512i rhs_mat_2367ABEF_1_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_1, (_MM_PERM_ENUM)221); //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15) BA(12-15) BB(12-15) BA(12-15) BB(12-15) BE(12-15) BF(12-15) BE(12-15) BF(12-15)

                    const __m512i rhs_mat_014589CD_2_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_2, (_MM_PERM_ENUM)221); //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23) B8(20-23) B9(20-23) B8(20-23) B9(20-23) BC(20-23) BD(20-23) BC(20-23) BD(20-23)
                    const __m512i rhs_mat_2367ABEF_2_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_2, (_MM_PERM_ENUM)221); //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23) BA(20-23) BB(20-23) BA(20-23) BB(20-23) BE(20-23) BF(20-23) BE(20-23) BF(20-23)

                    const __m512i rhs_mat_014589CD_3_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_3, (_MM_PERM_ENUM)221); //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31) B8(28-31) B9(28-31) B8(28-31) B9(28-31) BC(28-31) BD(28-31) BC(28-31) BD(28-31)
                    const __m512i rhs_mat_2367ABEF_3_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_3, (_MM_PERM_ENUM)221); //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31) BA(28-31) BB(28-31) BA(28-31) BB(28-31) BE(28-31) BF(28-31) BE(28-31) BF(28-31)


                    // Scale values - Load the weight scale values of two block_q4_0x8
                    const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                    // Load the four block_q4_0 quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                    __m256i lhs_mat_ymm_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                    __m256i lhs_mat_ymm_01_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 0);
                    __m256i lhs_mat_ymm_23_0 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_0, lhs_mat_ymm_0123_0, 17);
                    __m256i lhs_mat_ymm_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                    __m256i lhs_mat_ymm_01_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 0);
                    __m256i lhs_mat_ymm_23_1 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_1, lhs_mat_ymm_0123_1, 17);
                    __m256i lhs_mat_ymm_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                    __m256i lhs_mat_ymm_01_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 0);
                    __m256i lhs_mat_ymm_23_2 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_2, lhs_mat_ymm_0123_2, 17);
                    __m256i lhs_mat_ymm_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                    __m256i lhs_mat_ymm_01_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 0);
                    __m256i lhs_mat_ymm_23_3 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_3, lhs_mat_ymm_0123_3, 17);

                    __m512i lhs_mat_01_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_0), lhs_mat_ymm_01_0, 1);
                    __m512i lhs_mat_23_0 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_0), lhs_mat_ymm_23_0, 1);
                    __m512i lhs_mat_01_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_1), lhs_mat_ymm_01_1, 1);
                    __m512i lhs_mat_23_1 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_1), lhs_mat_ymm_23_1, 1);
                    __m512i lhs_mat_01_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_2), lhs_mat_ymm_01_2, 1);
                    __m512i lhs_mat_23_2 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_2), lhs_mat_ymm_23_2, 1);
                    __m512i lhs_mat_01_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_3), lhs_mat_ymm_01_3, 1);
                    __m512i lhs_mat_23_3 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_3), lhs_mat_ymm_23_3, 1);

                    // Shuffle pattern one - left side input

                    const __m512i lhs_mat_01_0_sp1 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m512i lhs_mat_23_0_sp1 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m512i lhs_mat_01_1_sp1 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m512i lhs_mat_23_1_sp1 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m512i lhs_mat_01_2_sp1 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m512i lhs_mat_23_2_sp1 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m512i lhs_mat_01_3_sp1 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m512i lhs_mat_23_3_sp1 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input

                    const __m512i lhs_mat_01_0_sp2 = _mm512_shuffle_epi32(lhs_mat_01_0, (_MM_PERM_ENUM)245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m512i lhs_mat_23_0_sp2 = _mm512_shuffle_epi32(lhs_mat_23_0, (_MM_PERM_ENUM)245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m512i lhs_mat_01_1_sp2 = _mm512_shuffle_epi32(lhs_mat_01_1, (_MM_PERM_ENUM)245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m512i lhs_mat_23_1_sp2 = _mm512_shuffle_epi32(lhs_mat_23_1, (_MM_PERM_ENUM)245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m512i lhs_mat_01_2_sp2 = _mm512_shuffle_epi32(lhs_mat_01_2, (_MM_PERM_ENUM)245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m512i lhs_mat_23_2_sp2 = _mm512_shuffle_epi32(lhs_mat_23_2, (_MM_PERM_ENUM)245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m512i lhs_mat_01_3_sp2 = _mm512_shuffle_epi32(lhs_mat_01_3, (_MM_PERM_ENUM)245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m512i lhs_mat_23_3_sp2 = _mm512_shuffle_epi32(lhs_mat_23_3, (_MM_PERM_ENUM)245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    __m512i iacc_mat_00_sp1 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1));
                    __m512i iacc_mat_01_sp1 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1));
                    __m512i iacc_mat_10_sp1 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1));
                    __m512i iacc_mat_11_sp1 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1));
                    __m512i iacc_mat_00_sp2 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2));
                    __m512i iacc_mat_01_sp2 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2));
                    __m512i iacc_mat_10_sp2 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2));
                    __m512i iacc_mat_11_sp2 =
                        _mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(mul_sum_i8_pairs_int32x16(lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), mul_sum_i8_pairs_int32x16(lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2)), mul_sum_i8_pairs_int32x16(lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00 = _mm512_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m512i iacc_mat_01 = _mm512_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m512i iacc_mat_10 = _mm512_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m512i iacc_mat_11 = _mm512_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                    // Straighten out to make 4 row vectors
                    __m512i iacc_row_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00, _mm512_shuffle_epi32(iacc_mat_01, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00, (_MM_PERM_ENUM)78), iacc_mat_01);
                    __m512i iacc_row_2 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10, _mm512_shuffle_epi32(iacc_mat_11, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10, (_MM_PERM_ENUM)78), iacc_mat_11);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m128i row_scale_f16 = _mm_shuffle_epi32(_mm_maskload_epi32((int const*)(a_ptr[b].d), loadMask), 68);
                    const __m512 row_scale_f32 = GGML_F32Cx16_REPEAT_LOAD(row_scale_f16);

                    // Multiply with appropiate scales and accumulate
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)),   acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)),  acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
                }

                // Store the accumulated values
                for (int i = 0; i < 4; i++) {
                    _mm512_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
                }
            }
        }
        if (anc != nc) {
            xstart = anc/8;
            y = 0;
        }
    #endif // __AVX512F__

        // Take group of four block_q8_0x4 structures at each pass of the loop and perform dot product operation

        for (; y < anr / 4; y += 4) {
            const block_q8_0x4 * a_ptrs[4];

            a_ptrs[0] = a_ptr_start + (y * nb);
            for (int i = 0; i < 3; ++i) {
                a_ptrs[i + 1] = a_ptrs[i] + nb;
            }

            // Take group of eight block_q4_0x8 structures at each pass of the loop and perform dot product operation
            for (int64_t x = xstart; x < nc / 8; x++) {

                const block_q4_0x8 * b_ptr = b_ptr_start + (x * b_nb);

                // Master FP accumulators
                __m256 acc_rows[16];
                for (int i = 0; i < 16; i++) {
                    acc_rows[i] = _mm256_setzero_ps();
                }

                for (int64_t b = 0; b < nb; b++) {
                    // Load the eight block_q4_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    // 4-bit -> 8-bit - Sign is maintained
                    const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b)); //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                    const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b)); //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                    const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b)); //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                    const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b)); //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                    const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b)); //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                    const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b)); //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                    const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b)); //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                    const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b)); //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                    const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                    const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                    const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                    const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                    const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                    const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                    const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                    // Shuffle pattern two - right side input

                    const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                    const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                    const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                    const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                    const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                    const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                    const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                    const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                    // Scale values - Load the wight scale values of block_q4_0x8
                    const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                    // Process LHS in groups of four
                    for (int rp = 0; rp < 4; rp++) {
                        // Load the four block_q4_0 quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs)));
                        __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                        __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                        __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 32)));
                        __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                        __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                        __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 64)));
                        __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                        __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                        __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptrs[rp][b].qs + 96)));
                        __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                        __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                        const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                        const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                        const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                        const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                        const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                        const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                        const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                        // Shuffle pattern two - left side input
                        const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                        const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                        const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                        const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                        const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                        const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                        const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                        const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        // Resembles MMLAs into 2x2 matrices in ARM Version
                        __m256i iacc_mat_00_sp1 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1));
                        __m256i iacc_mat_01_sp1 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1));
                        __m256i iacc_mat_10_sp1 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1));
                        __m256i iacc_mat_11_sp1 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1));
                        __m256i iacc_mat_00_sp2 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2));
                        __m256i iacc_mat_01_sp2 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2));
                        __m256i iacc_mat_10_sp2 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2));
                        __m256i iacc_mat_11_sp2 =
                            _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                        __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                        __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                        __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);

                        // Straighten out to make 4 row vectors
                        __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                        __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                        __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                        __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                        // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                        const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                        // Multiply with appropiate scales and accumulate
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32,  255)), acc_rows[rp * 4 + 3]);
                    }
                }

                // Store the accumulated values
                for (int i = 0; i < 16; i++) {
                    _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
                }
            }
        }

        // Take a block_q8_0x4 structures at each pass of the loop and perform dot product operation
        for (; y < nr / 4; y ++) {

            const block_q8_0x4 * a_ptr = a_ptr_start + (y * nb);

            // Load the eight block_q4_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
            for (int64_t x = xstart; x < nc / 8; x++) {

                const block_q4_0x8 * b_ptr = b_ptr_start + (x * b_nb);

                // Master FP accumulators
                __m256 acc_rows[4];
                for (int i = 0; i < 4; i++) {
                    acc_rows[i] = _mm256_setzero_ps();
                }

                for (int64_t b = 0; b < nb; b++) {
                    // Load the eight block_q8_0 quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 32));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 64));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i *)(b_ptr[b].qs + 96));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of valuess
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);

                    // 4-bit -> 8-bit - Sign is maintained
                    const __m256i rhs_mat_0145_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_0, m4b));  //B0(0-7) B1(0-7) B4(0-7) B5(0-7)
                    const __m256i rhs_mat_2367_0 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_0, m4b));  //B2(0-7) B3(0-7) B6(0-7) B7(0-7)

                    const __m256i rhs_mat_0145_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_0145_1, m4b));  //B0(8-15) B1(8-15) B4(8-15) B5(8-15)
                    const __m256i rhs_mat_2367_1 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(rhs_raw_mat_2367_1, m4b));  //B2(8-15) B3(8-15) B6(8-15) B7(8-15)

                    const __m256i rhs_mat_0145_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b));  //B0(16-23) B1(16-23) B4(16-23) B5(16-23)
                    const __m256i rhs_mat_2367_2 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b));  //B2(16-23) B3(16-23) B6(16-23) B7(16-23)

                    const __m256i rhs_mat_0145_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b));  //B0(24-31) B1(24-31) B4(24-31) B5(24-31)
                    const __m256i rhs_mat_2367_3 = _mm256_shuffle_epi8(signextendlut, _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b));  //B2(24-31) B3(24-31) B6(24-31) B7(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_0_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_0, 136);  //B0(0-3) B1(0-3) B0(0-3) B1(0-3) B4(0-3) B5(0-3) B4(0-3) B5(0-3)
                    const __m256i rhs_mat_2367_0_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_0, 136);  //B2(0-3) B3(0-3) B2(0-3) B3(0-3) B6(0-3) B7(0-3) B6(0-3) B7(0-3)

                    const __m256i rhs_mat_0145_1_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_1, 136);  //B0(8-11) B1(8-11) B0(8-11) B1(8-11) B4(8-11) B5(8-11) B4(8-11) B5(8-11)
                    const __m256i rhs_mat_2367_1_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_1, 136);  //B2(8-11) B3(8-11) B2(8-11) B3(8-11) B6(8-11) B7(8-11) B6(8-11) B7(8-11)

                    const __m256i rhs_mat_0145_2_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_2, 136);  //B0(16-19) B1(16-19) B0(16-19) B1(16-19) B4(16-19) B5(16-19) B4(16-19) B5(16-19)
                    const __m256i rhs_mat_2367_2_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_2, 136);  //B2(16-19) B3(16-19) B2(16-19) B3(16-19) B6(16-19) B7(16-19) B6(16-19) B7(16-19)

                    const __m256i rhs_mat_0145_3_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_3, 136);  //B0(24-27) B1(24-27) B0(24-27) B1(24-27) B4(24-27) B5(24-27) B4(24-27) B5(24-27)
                    const __m256i rhs_mat_2367_3_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_3, 136);  //B2(24-27) B3(24-27) B2(24-27) B3(24-27) B6(24-27) B7(24-27) B6(24-27) B7(24-27)

                    // Shuffle pattern two - right side input

                    const __m256i rhs_mat_0145_0_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_0, 221);  //B0(4-7) B1(4-7) B0(4-7) B1(4-7) B4(4-7) B5(4-7) B4(4-7) B5(4-7)
                    const __m256i rhs_mat_2367_0_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_0, 221);  //B2(4-7) B3(4-7) B2(4-7) B3(4-7) B6(4-7) B7(4-7) B6(4-7) B7(4-7)

                    const __m256i rhs_mat_0145_1_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_1, 221);  //B0(12-15) B1(12-15) B0(12-15) B1(12-15) B4(12-15) B5(12-15) B4(12-15) B5(12-15)
                    const __m256i rhs_mat_2367_1_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_1, 221);  //B2(12-15) B3(12-15) B2(12-15) B3(12-15) B6(12-15) B7(12-15) B6(12-15) B7(12-15)

                    const __m256i rhs_mat_0145_2_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_2, 221);  //B0(20-23) B1(20-23) B0(20-23) B1(20-23) B4(20-23) B5(20-23) B4(20-23) B5(20-23)
                    const __m256i rhs_mat_2367_2_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_2, 221);  //B2(20-23) B3(20-23) B2(20-23) B3(20-23) B6(20-23) B7(20-23) B6(20-23) B7(20-23)

                    const __m256i rhs_mat_0145_3_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_3, 221);  //B0(28-31) B1(28-31) B0(28-31) B1(28-31) B4(28-31) B5(28-31) B4(28-31) B5(28-31)
                    const __m256i rhs_mat_2367_3_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_3, 221);  //B2(28-31) B3(28-31) B2(28-31) B3(28-31) B6(28-31) B7(28-31) B6(28-31) B7(28-31)

                    // Scale values - Load the wight scale values of block_q4_0x8
                    const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                    // Load the four block_q4_0 quantized values interleaved with each other in chunks of eight - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_0 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs)));
                    __m256i lhs_mat_01_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 0);
                    __m256i lhs_mat_23_0 = _mm256_permute2f128_si256(lhs_mat_0123_0, lhs_mat_0123_0, 17);
                    __m256i lhs_mat_0123_1 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 32)));
                    __m256i lhs_mat_01_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 0);
                    __m256i lhs_mat_23_1 = _mm256_permute2f128_si256(lhs_mat_0123_1, lhs_mat_0123_1, 17);
                    __m256i lhs_mat_0123_2 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 64)));
                    __m256i lhs_mat_01_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 0);
                    __m256i lhs_mat_23_2 = _mm256_permute2f128_si256(lhs_mat_0123_2, lhs_mat_0123_2, 17);
                    __m256i lhs_mat_0123_3 = _mm256_loadu_si256((const __m256i *)((a_ptr[b].qs + 96)));
                    __m256i lhs_mat_01_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 0);
                    __m256i lhs_mat_23_3 = _mm256_permute2f128_si256(lhs_mat_0123_3, lhs_mat_0123_3, 17);

                    // Shuffle pattern one - left side input

                    const __m256i lhs_mat_01_0_sp1 = _mm256_shuffle_epi32(lhs_mat_01_0, 160);  //A0(0-3) A0(0-3) A1(0-3) A1(0-3) A0(0-3) A0(0-3) A1(0-3) A1(0-3)
                    const __m256i lhs_mat_23_0_sp1 = _mm256_shuffle_epi32(lhs_mat_23_0, 160);  //A2(0-3) A2(0-3) A3(0-3) A3(0-3) A2(0-3) A2(0-3) A3(0-3) A3(0-3)

                    const __m256i lhs_mat_01_1_sp1 = _mm256_shuffle_epi32(lhs_mat_01_1, 160);  //A0(8-11) A0(8-11) A1(8-11) A1(8-11) A0(8-11) A0(8-11) A1(8-11) A1(8-11)
                    const __m256i lhs_mat_23_1_sp1 = _mm256_shuffle_epi32(lhs_mat_23_1, 160);  //A2(8-11) A2(8-11) A3(8-11) A3(8-11) A2(8-11) A2(8-11) A3(8-11) A3(8-11)

                    const __m256i lhs_mat_01_2_sp1 = _mm256_shuffle_epi32(lhs_mat_01_2, 160);  //A0(16-19) A0(16-19) A1(16-19) A1(16-19) A0(16-19) A0(16-19) A1(16-19) A1(16-19)
                    const __m256i lhs_mat_23_2_sp1 = _mm256_shuffle_epi32(lhs_mat_23_2, 160);  //A2(16-19) A2(16-19) A3(16-19) A3(16-19) A2(16-19) A2(16-19) A3(16-19) A3(16-19)

                    const __m256i lhs_mat_01_3_sp1 = _mm256_shuffle_epi32(lhs_mat_01_3, 160);  //A0(24-27) A0(24-27) A1(24-27) A1(24-27) A0(24-27) A0(24-27) A1(24-27) A1(24-27)
                    const __m256i lhs_mat_23_3_sp1 = _mm256_shuffle_epi32(lhs_mat_23_3, 160);  //A2(24-27) A2(24-27) A3(24-27) A3(24-27) A2(24-27) A2(24-27) A3(24-27) A3(24-27)

                    // Shuffle pattern two - left side input

                    const __m256i lhs_mat_01_0_sp2 = _mm256_shuffle_epi32(lhs_mat_01_0, 245);  //A0(4-7) A0(4-7) A1(4-7) A1(4-7) A0(4-7) A0(4-7) A1(4-7) A1(4-7)
                    const __m256i lhs_mat_23_0_sp2 = _mm256_shuffle_epi32(lhs_mat_23_0, 245);  //A2(4-7) A2(4-7) A3(4-7) A3(4-7) A2(4-7) A2(4-7) A3(4-7) A3(4-7)

                    const __m256i lhs_mat_01_1_sp2 = _mm256_shuffle_epi32(lhs_mat_01_1, 245);  //A0(12-15) A0(12-15) A1(12-15) A1(12-15) A0(12-15) A0(12-15) A1(12-15) A1(12-15)
                    const __m256i lhs_mat_23_1_sp2 = _mm256_shuffle_epi32(lhs_mat_23_1, 245);  //A2(12-15) A2(12-15) A3(12-15) A3(12-15) A2(12-15) A2(12-15) A3(12-15) A3(12-15)

                    const __m256i lhs_mat_01_2_sp2 = _mm256_shuffle_epi32(lhs_mat_01_2, 245);  //A0(20-23) A0(20-23) A1(20-23) A1(20-23) A0(20-23) A0(20-23) A1(20-23) A1(20-23)
                    const __m256i lhs_mat_23_2_sp2 = _mm256_shuffle_epi32(lhs_mat_23_2, 245);  //A2(20-23) A2(20-23) A3(20-23) A3(20-23) A2(20-23) A2(20-23) A3(20-23) A3(20-23)

                    const __m256i lhs_mat_01_3_sp2 = _mm256_shuffle_epi32(lhs_mat_01_3, 245);  //A0(28-31) A0(28-31) A1(28-31) A1(28-31) A0(28-31) A0(28-31) A1(28-31) A1(28-31)
                    const __m256i lhs_mat_23_3_sp2 = _mm256_shuffle_epi32(lhs_mat_23_3, 245);  //A2(28-31) A2(28-31) A3(28-31) A3(28-31) A2(28-31) A2(28-31) A3(28-31) A3(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    // Resembles MMLAs into 2x2 matrices in ARM Version
                    __m256i iacc_mat_00_sp1 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1));
                    __m256i iacc_mat_01_sp1 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1));
                    __m256i iacc_mat_10_sp1 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1));
                    __m256i iacc_mat_11_sp1 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1));
                    __m256i iacc_mat_00_sp2 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2));
                    __m256i iacc_mat_01_sp2 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2));
                    __m256i iacc_mat_10_sp2 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2));
                    __m256i iacc_mat_11_sp2 =
                        _mm256_add_epi32(_mm256_add_epi32(_mm256_add_epi32(mul_sum_i8_pairs_int32x8(lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), mul_sum_i8_pairs_int32x8(lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2)), mul_sum_i8_pairs_int32x8(lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00 = _mm256_add_epi32(iacc_mat_00_sp1, iacc_mat_00_sp2);
                    __m256i iacc_mat_01 = _mm256_add_epi32(iacc_mat_01_sp1, iacc_mat_01_sp2);
                    __m256i iacc_mat_10 = _mm256_add_epi32(iacc_mat_10_sp1, iacc_mat_10_sp2);
                    __m256i iacc_mat_11 = _mm256_add_epi32(iacc_mat_11_sp1, iacc_mat_11_sp2);


                    // Straighten out to make 4 row vectors
                    __m256i iacc_row_0 = _mm256_blend_epi32(iacc_mat_00, _mm256_shuffle_epi32(iacc_mat_01, 78), 204);
                    __m256i iacc_row_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00, 78), iacc_mat_01, 204);
                    __m256i iacc_row_2 = _mm256_blend_epi32(iacc_mat_10, _mm256_shuffle_epi32(iacc_mat_11, 78), 204);
                    __m256i iacc_row_3 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10, 78), iacc_mat_11, 204);

                    // Load the scale(d) values for all the 4 Q8_0 blocks and repeat it across lanes
                    const __m256 row_scale_f32 = GGML_F32Cx8_REPEAT_LOAD(a_ptr[b].d, loadMask);

                    // Multiply with appropiate scales and accumulate
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);
                }

                // Store the accumulated values
                for (int i = 0; i < 4; i++) {
                    _mm256_storeu_ps((float *)(s + ((y * 4 + i) * bs + x * 8)), acc_rows[i]);
                }
            }
        }
        return;
    }
#elif defined(__riscv_v_intrinsic)
    if (__riscv_vlenb() >= QK4_0) {
        const size_t vl = QK4_0;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
                vfloat32m1_t sumf0 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf1 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf2 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                vfloat32m1_t sumf3 = __riscv_vfmv_v_f_f32m1(0.0, vl / 4);
                for (int l = 0; l < nb; l++) {
                    const vint8m4_t rhs_raw_vec = __riscv_vle8_v_i8m4((const int8_t *)b_ptr[l].qs, vl * 4);
                    const vint8m4_t rhs_vec_lo = __riscv_vsra_vx_i8m4(__riscv_vsll_vx_i8m4(rhs_raw_vec, 4, vl * 4), 4, vl * 4);
                    const vint8m4_t rhs_vec_hi = __riscv_vsra_vx_i8m4(rhs_raw_vec, 4, vl * 4);
                    const vint8m2_t rhs_vec_lo_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 0);
                    const vint8m2_t rhs_vec_lo_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_lo, 1);
                    const vint8m2_t rhs_vec_hi_0 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 0);
                    const vint8m2_t rhs_vec_hi_1 = __riscv_vget_v_i8m4_i8m2(rhs_vec_hi, 1);

                    // vector version needs Zvfhmin extension
                    const float a_scales[4] = {
                        GGML_FP16_TO_FP32(a_ptr[l].d[0]),
                        GGML_FP16_TO_FP32(a_ptr[l].d[1]),
                        GGML_FP16_TO_FP32(a_ptr[l].d[2]),
                        GGML_FP16_TO_FP32(a_ptr[l].d[3])
                    };
                    const float b_scales[8] = {
                        GGML_FP16_TO_FP32(b_ptr[l].d[0]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[1]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[2]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[3]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[4]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[5]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[6]),
                        GGML_FP16_TO_FP32(b_ptr[l].d[7])
                    };
                    const vfloat32m1_t b_scales_vec = __riscv_vle32_v_f32m1(b_scales, vl / 4);

                    const int64_t A0 = *(const int64_t *)&a_ptr[l].qs[0];
                    const int64_t A4 = *(const int64_t *)&a_ptr[l].qs[32];
                    const int64_t A8 = *(const int64_t *)&a_ptr[l].qs[64];
                    const int64_t Ac = *(const int64_t *)&a_ptr[l].qs[96];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l0;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A0, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A4, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A8, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ac, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l0 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l0));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[0], vl / 4);
                        sumf0 = __riscv_vfmacc_vv_f32m1(sumf0, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A1 = *(const int64_t *)&a_ptr[l].qs[8];
                    const int64_t A5 = *(const int64_t *)&a_ptr[l].qs[40];
                    const int64_t A9 = *(const int64_t *)&a_ptr[l].qs[72];
                    const int64_t Ad = *(const int64_t *)&a_ptr[l].qs[104];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l1;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A1, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A5, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A9, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ad, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l1 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l1));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[1], vl / 4);
                        sumf1 = __riscv_vfmacc_vv_f32m1(sumf1, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A2 = *(const int64_t *)&a_ptr[l].qs[16];
                    const int64_t A6 = *(const int64_t *)&a_ptr[l].qs[48];
                    const int64_t Aa = *(const int64_t *)&a_ptr[l].qs[80];
                    const int64_t Ae = *(const int64_t *)&a_ptr[l].qs[112];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l2;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A2, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A6, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Aa, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ae, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l2 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l2));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[2], vl / 4);
                        sumf2 = __riscv_vfmacc_vv_f32m1(sumf2, tmp1, b_scales_vec, vl / 4);
                    }

                    const int64_t A3 = *(const int64_t *)&a_ptr[l].qs[24];
                    const int64_t A7 = *(const int64_t *)&a_ptr[l].qs[56];
                    const int64_t Ab = *(const int64_t *)&a_ptr[l].qs[88];
                    const int64_t Af = *(const int64_t *)&a_ptr[l].qs[120];
                    __asm__ __volatile__("" ::: "memory"); // prevent gcc from emitting fused vlse64, violating alignment
                    vint16m4_t sumi_l3;
                    {
                        const vint8m2_t lhs_0_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A3, vl / 4));
                        const vint8m2_t lhs_1_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(A7, vl / 4));
                        const vint8m2_t lhs_2_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Ab, vl / 4));
                        const vint8m2_t lhs_3_8 =__riscv_vreinterpret_v_i64m2_i8m2(__riscv_vmv_v_x_i64m2(Af, vl / 4));
                        const vint16m4_t sumi_lo_0 = __riscv_vwmul_vv_i16m4(rhs_vec_lo_0, lhs_0_8, vl * 2);
                        const vint16m4_t sumi_lo_1 = __riscv_vwmacc_vv_i16m4(sumi_lo_0, rhs_vec_lo_1, lhs_1_8, vl * 2);
                        const vint16m4_t sumi_hi_0 = __riscv_vwmacc_vv_i16m4(sumi_lo_1, rhs_vec_hi_0, lhs_2_8, vl * 2);
                        const vint16m4_t sumi_hi_m = __riscv_vwmacc_vv_i16m4(sumi_hi_0, rhs_vec_hi_1, lhs_3_8, vl * 2);

                        sumi_l3 = sumi_hi_m;
                    }

                    {
                        const vuint32m4_t sumi_i32 = __riscv_vreinterpret_v_i32m4_u32m4(__riscv_vreinterpret_v_i16m4_i32m4(sumi_l3));
                        const vuint16m2_t sumi_h2_0 = __riscv_vnsrl_wx_u16m2(sumi_i32, 0, vl);
                        const vuint16m2_t sumi_h2_1 = __riscv_vnsrl_wx_u16m2(sumi_i32, 16, vl);
                        const vuint16m2_t sumi_h2 = __riscv_vadd_vv_u16m2(sumi_h2_0, sumi_h2_1, vl);
                        const vuint32m2_t sumi_h2_i32 = __riscv_vreinterpret_v_u16m2_u32m2(sumi_h2);
                        const vuint16m1_t sumi_h4_0 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 0, vl / 2);
                        const vuint16m1_t sumi_h4_1 = __riscv_vnsrl_wx_u16m1(sumi_h2_i32, 16, vl / 2);
                        const vuint16m1_t sumi_h4 = __riscv_vadd_vv_u16m1(sumi_h4_0, sumi_h4_1, vl / 2);
                        const vuint32m1_t sumi_h4_i32 = __riscv_vreinterpret_v_u16m1_u32m1(sumi_h4);
                        const vint16mf2_t sumi_h8_0 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 0, vl / 4));
                        const vint16mf2_t sumi_h8_1 = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vnsrl_wx_u16mf2(sumi_h4_i32, 16, vl / 4));
                        const vint32m1_t sumi_h8 = __riscv_vwadd_vv_i32m1(sumi_h8_0, sumi_h8_1, vl / 4);
                        const vfloat32m1_t facc = __riscv_vfcvt_f_x_v_f32m1(sumi_h8, vl / 4);

                        const vfloat32m1_t tmp1 = __riscv_vfmul_vf_f32m1(facc, a_scales[3], vl / 4);
                        sumf3 = __riscv_vfmacc_vv_f32m1(sumf3, tmp1, b_scales_vec, vl / 4);
                    }
                }
                __riscv_vse32_v_f32m1(&s[(y * 4 + 0) * bs + x * ncols_interleaved], sumf0, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 1) * bs + x * ncols_interleaved], sumf1, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 2) * bs + x * ncols_interleaved], sumf2, vl / 4);
                __riscv_vse32_v_f32m1(&s[(y * 4 + 3) * bs + x * ncols_interleaved], sumf3, vl / 4);
            }
        }

        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__)
    float sumf[4][8];
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x8 * b_ptr = (const block_q4_0x8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

static void ggml_gemm_iq4_nl_4x4_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 4;
    const int blocklen = 4;

    assert (n % qk == 0);
    assert (nr % 4 == 0);
    assert (nc % ncols_interleaved == 0);

    UNUSED(s);
    UNUSED(bs);
    UNUSED(vx);
    UNUSED(vy);
    UNUSED(nr);
    UNUSED(nc);
    UNUSED(nb);
    UNUSED(ncols_interleaved);
    UNUSED(blocklen);

#if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
    if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
        const int8x16_t kvalues = vld1q_s8(kvalues_iq4nl);

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);

                float32x4_t sumf[4];
                for (int m = 0; m < 4; m++) {
                    sumf[m] = vdupq_n_f32(0);
                }

                for (int l = 0; l < nb; l++) {
                    float32x4_t a_d = vcvt_f32_f16(vld1_f16((const float16_t *)a_ptr[l].d));
                    float32x4_t b_d = vcvt_f32_f16(vld1_f16((const float16_t *)b_ptr[l].d));

                    int32x4_t sumi_0 = vdupq_n_s32(0);
                    int32x4_t sumi_1 = vdupq_n_s32(0);
                    int32x4_t sumi_2 = vdupq_n_s32(0);
                    int32x4_t sumi_3 = vdupq_n_s32(0);

                    for (int k = 0; k < 4; k++) {
                        int8x16_t a_0 = vld1q_s8(a_ptr[l].qs + 16 * k + 0);
                        int8x16_t a_1 = vld1q_s8(a_ptr[l].qs + 16 * k + 64);

                        uint8x16_t b = vld1q_u8(b_ptr[l].qs + 16 * k);
                        int8x16_t b_hi = vqtbl1q_s8(kvalues, b >> 4);
                        int8x16_t b_lo = vqtbl1q_s8(kvalues, b & 0xF);

                        sumi_0 = vdotq_laneq_s32(sumi_0, b_lo, a_0, 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b_lo, a_0, 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b_lo, a_0, 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b_lo, a_0, 3);
                        sumi_0 = vdotq_laneq_s32(sumi_0, b_hi, a_1, 0);
                        sumi_1 = vdotq_laneq_s32(sumi_1, b_hi, a_1, 1);
                        sumi_2 = vdotq_laneq_s32(sumi_2, b_hi, a_1, 2);
                        sumi_3 = vdotq_laneq_s32(sumi_3, b_hi, a_1, 3);
                    }

                    sumf[0] = vmlaq_f32(sumf[0], vmulq_laneq_f32(b_d, a_d, 0), vcvtq_f32_s32(sumi_0));
                    sumf[1] = vmlaq_f32(sumf[1], vmulq_laneq_f32(b_d, a_d, 1), vcvtq_f32_s32(sumi_1));
                    sumf[2] = vmlaq_f32(sumf[2], vmulq_laneq_f32(b_d, a_d, 2), vcvtq_f32_s32(sumi_2));
                    sumf[3] = vmlaq_f32(sumf[3], vmulq_laneq_f32(b_d, a_d, 3), vcvtq_f32_s32(sumi_3));
                }

                for (int m = 0; m < 4; m++) {
                    vst1q_f32(s + (y * 4 + m) * bs + x * 4, sumf[m]);
                }
            }
        }
        return;
    }
#endif // #if ! ((defined(_MSC_VER)) && ! defined(__clang__)) && defined(__aarch64__) && defined(__ARM_NEON)
    {
        float sumf[4][4];
        int sumi;

        for (int y = 0; y < nr / 4; y++) {
            const block_q8_0x4 * a_ptr = (const block_q8_0x4 *) vy + (y * nb);
            for (int x = 0; x < nc / ncols_interleaved; x++) {
                const block_iq4_nlx4 * b_ptr = (const block_iq4_nlx4 *) vx + (x * nb);
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                for (int l = 0; l < nb; l++) {
                    for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                        for (int m = 0; m < 4; m++) {
                            for (int j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                for (int i = 0; i < blocklen; ++i) {
                                    const int v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    const int v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_FP16_TO_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                for (int m = 0; m < 4; m++) {
                    for (int j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

static block_q4_0x4 make_block_q4_0x4(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 2 / blck_size_interleave;

    if (blck_size_interleave == 8) {
        const uint64_t xor_mask = 0x8888888888888888ULL;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
        }
    } else if (blck_size_interleave == 4) {
        const uint32_t xor_mask = 0x88888888;
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            uint32_t elems;
            memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint32_t));
            elems ^= xor_mask;
            memcpy(&out.qs[dst_offset], &elems, sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
static block_q4_0x8 make_block_q4_0x8(block_q4_0 * in, unsigned int blck_size_interleave) {
    block_q4_0x8 out;

    for (int i = 0; i < 8; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_0 * 4 / blck_size_interleave;
    const uint64_t xor_mask = 0x8888888888888888ULL;

    for (int i = 0; i < end; ++i) {
        int src_id = i % 8;
        int src_offset = (i / 8) * blck_size_interleave;
        int dst_offset = i * blck_size_interleave;

        uint64_t elems;
        memcpy(&elems, &in[src_id].qs[src_offset], sizeof(uint64_t));
        elems ^= xor_mask;
        memcpy(&out.qs[dst_offset], &elems, sizeof(uint64_t));
    }

    return out;
}

static int repack_q4_0_to_q4_0_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    constexpr int nrows_interleaved = 4;

    block_q4_0x4 * dst = (block_q4_0x4 *)t->data;
    const block_q4_0 * src = (const block_q4_0 *)data;
    block_q4_0 dst_tmp[4];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static int repack_q4_0_to_q4_0_8_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 8);
    constexpr int nrows_interleaved = 8;

    block_q4_0x8 * dst = (block_q4_0x8*)t->data;
    const block_q4_0 * src = (const block_q4_0*) data;
    block_q4_0 dst_tmp[8];
    int nrow = ggml_nrows(t);
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_q4_0));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x8(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

static block_iq4_nlx4 make_block_iq4_nlx4(block_iq4_nl * in, unsigned int blck_size_interleave) {
    block_iq4_nlx4 out;

    for (int i = 0; i < 4; i++) {
        out.d[i] = in[i].d;
    }

    const int end = QK4_NL * 2 / blck_size_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    if (blck_size_interleave == 4) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 4;
            int src_offset = (i / 4) * blck_size_interleave;
            int dst_offset = i * blck_size_interleave;

            memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint32_t));
        }
    } else {
        GGML_ASSERT(false);
    }

    return out;
}

static int repack_iq4_nl_to_iq4_nl_4_bl(struct ggml_tensor * t, int interleave_block, const void * GGML_RESTRICT data, size_t data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    //GGML_ASSERT(interleave_block == 4 || interleave_block == 8);
    GGML_ASSERT(interleave_block == 4);

    block_iq4_nlx4 * dst = (block_iq4_nlx4 *)t->data;
    const block_iq4_nl * src = (const block_iq4_nl *)data;
    block_iq4_nl dst_tmp[4];
    int nrow = ggml_nrows(t);
    int nrows_interleaved = 4;
    int nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * sizeof(block_iq4_nl));

    if (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        return -1;
    }

    for (int b = 0; b < nrow; b += nrows_interleaved) {
        for (int64_t x = 0; x < nblocks; x++) {
            for (int i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_iq4_nlx4(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    return 0;

    GGML_UNUSED(data_size);
}

namespace ggml::cpu::aarch64 {
// repack
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(struct ggml_tensor *, const void *, size_t);

// TODO: generalise.
template <> int repack<block_q4_0, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 4, data, data_size);
}

template <> int repack<block_q4_0, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_4_bl(t, 8, data, data_size);
}

template <> int repack<block_q4_0, 8, 8>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_q4_0_to_q4_0_8_bl(t, 8, data, data_size);
}

template <> int repack<block_iq4_nl, 4, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
    return repack_iq4_nl_to_iq4_nl_4_bl(t, 4, data, data_size);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

// gemv
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
void gemv(int, float *, size_t, const void *, const void *, int, int);

template <> void gemv<block_q4_0, 4, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemv<block_q4_0, 8, 8>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemv<block_iq4_nl, 4, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

// gemm
template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
void gemm(int, float *, size_t, const void *, const void *, int, int);

template <> void gemm<block_q4_0, 4, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_0, 8, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <> void gemm<block_q4_0, 8, 8>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

template <>
void gemm<block_iq4_nl, 4, 4>(int n, float * s, size_t bs, const void * vx, const void * vy, int nr, int nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

class tensor_traits_base : public ggml::cpu::tensor_traits {
  public:
    virtual int repack(struct ggml_tensor * t, const void * data, size_t data_size) = 0;
};

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> class tensor_traits : public tensor_traits_base {

    bool work_size(int /* n_threads */, const struct ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        switch (op->op) {
        case GGML_OP_MUL_MAT:
            size = ggml_row_size(GGML_TYPE_Q8_0, ggml_nelements(op->src[1]));
            return true;
        case GGML_OP_MUL_MAT_ID:
            size = ggml_row_size(GGML_TYPE_Q8_0, ggml_nelements(op->src[1]));
            size = GGML_PAD(size, sizeof(int64_t));  // + padding for next bloc.
            size += sizeof(int64_t) * (1+op->src[0]->ne[2]) * op->src[1]->ne[2];
            return true;
        default:
            // GGML_ABORT("fatal error");
            break;
        }
        return false;
    }

    bool compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) override {
        switch (op->op) {
        case GGML_OP_MUL_MAT:
            forward_mul_mat(params, op);
            return true;
        case GGML_OP_MUL_MAT_ID:
            forward_mul_mat_id(params, op);
            return true;
        default:
            // GGML_ABORT("fatal error");
            break;
        }
        return false;
    }

    void forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
        // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

        char *       wdata = static_cast<char *>(params->wdata);
        const size_t nbw1  = ggml_row_size(GGML_TYPE_Q8_0, ne10);

        assert(params->wsize >= nbw1 * ne11);

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;

        int64_t i11_processed = 0;
        for (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
            quantize_mat_q8_0((float *) ((char *) src1->data + i11 * nb11), (void *) (wdata + i11 * nbw1), 4, ne10,
                              INTER_SIZE);
        }
        i11_processed = ne11 - ne11 % 4;
        for (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
            from_float((float *) ((char *) src1->data + i11 * nb11), (void *) (wdata + i11 * nbw1), ne10);
        }

        ggml_barrier(params->threadpool);

        const void * src1_wdata      = params->wdata;
        const size_t src1_col_stride = ggml_row_size(GGML_TYPE_Q8_0, ne10);
        int64_t      src0_start      = (ith * ne01) / nth;
        int64_t      src0_end        = ((ith + 1) * ne01) / nth;
        src0_start = (src0_start % NB_COLS) ? src0_start + NB_COLS - (src0_start % NB_COLS) : src0_start;
        src0_end   = (src0_end % NB_COLS) ? src0_end + NB_COLS - (src0_end % NB_COLS) : src0_end;
        if (src0_start >= src0_end) {
            return;
        }

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        if (ne11 > 3) {
            gemm<BLOC_TYPE, INTER_SIZE, NB_COLS>(ne00, (float *) ((char *) dst->data) + src0_start, ne01,
                                                 (const char *) src0->data + src0_start * nb01,
                                                 (const char *) src1_wdata, ne11 - ne11 % 4, src0_end - src0_start);
        }
        for (int iter = ne11 - ne11 % 4; iter < ne11; iter++) {
            gemv<BLOC_TYPE, INTER_SIZE, NB_COLS>(ne00, (float *) ((char *) dst->data + (iter * nb1)) + src0_start, ne01,
                                                 (const char *) src0->data + src0_start * nb01,
                                                 (const char *) src1_wdata + (src1_col_stride * iter), 1,
                                                 src0_end - src0_start);
        }
    }

    void forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        const ggml_tensor * src0 = op->src[0];
        const ggml_tensor * src1 = op->src[1];
        const ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_BINARY_OP_LOCALS

        const int ith = params->ith;
        const int nth = params->nth;

        const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_size(src0->type));
        GGML_ASSERT(nb10 == ggml_type_size(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == sizeof(float));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        const int n_ids = ids->ne[0]; // n_expert_used
        const int n_as  = ne02;       // n_expert

        const size_t nbw1 = ggml_row_size(GGML_TYPE_Q8_0, ne10);
        const size_t nbw2 = nbw1*ne11;
        const size_t nbw3 = nbw2*ne12;

        struct mmid_row_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >= (GGML_PAD(nbw3, sizeof(int64_t)) + n_as * sizeof(int64_t) +
                                      n_as * ne12 * sizeof(mmid_row_mapping)));

        auto                      wdata             = (char *) params->wdata;
        auto                      wdata_src1_end    = (char *) wdata + GGML_PAD(nbw3, sizeof(int64_t));
        int64_t *                 matrix_row_counts = (int64_t *) (wdata_src1_end);                      // [n_as]
        struct mmid_row_mapping * matrix_rows = (struct mmid_row_mapping *) (matrix_row_counts + n_as);  // [n_as][ne12]

        // src1: float32 => block_q8_0
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((float *)((char *) src1->data + i12 * nb12 + i11 * nb11),
                           (void *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#define MMID_MATRIX_ROW(row_id, i1) matrix_rows[(row_id) * ne12 + (i1)]

        if (ith == 0) {
            // initialize matrix_row_counts
            memset(matrix_row_counts, 0, n_as * sizeof(int64_t));

            // group rows by src0 matrix
            for (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                for (int32_t id = 0; id < n_ids; ++id) {
                    const int32_t i02 =
                        *(const int32_t *) ((const char *) ids->data + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_MATRIX_ROW(i02, matrix_row_counts[i02]) = { id, iid1 };
                    matrix_row_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        for (int cur_a = 0; cur_a < n_as; ++cur_a) {
            const int64_t cne1 = matrix_row_counts[cur_a];

            if (cne1 == 0) {
                continue;
            }

            auto src0_cur = (const char *) src0->data + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            const int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_start = (ith * ne01) / nth;
            int64_t src0_cur_end   = ((ith + 1) * ne01) / nth;
            src0_cur_start =
                (src0_cur_start % NB_COLS) ? src0_cur_start + NB_COLS - (src0_cur_start % NB_COLS) : src0_cur_start;
            src0_cur_end = (src0_cur_end % NB_COLS) ? src0_cur_end + NB_COLS - (src0_cur_end % NB_COLS) : src0_cur_end;

            if (src0_cur_start >= src0_cur_end) return;

            for (int ir1 = 0; ir1 < nr1; ir1++) {
                struct mmid_row_mapping row_mapping = MMID_MATRIX_ROW(cur_a, ir1);
                const int id       = row_mapping.i1; // selected expert index

                const int64_t  i11 = id % ne11;
                const int64_t  i12 = row_mapping.i2; // row index in src1

                const int64_t  i1 = id;  // selected expert index
                const int64_t  i2 = i12; // row

                auto src1_col = (const char *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_SIZE, NB_COLS>(
                        ne00, (float *)((char *) dst->data + (i1 * nb1 + i2 * nb2)) + src0_cur_start,
                        ne01,                    src0_cur + src0_cur_start * nb01,
                        src1_col, 1, src0_cur_end - src0_cur_start);
            }
        }
#undef MMID_MATRIX_ROW
    }

    int repack(struct ggml_tensor * t, const void * data, size_t data_size) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (int) NB_COLS, (int) INTER_SIZE);
        return ggml::cpu::aarch64::repack<BLOC_TYPE, INTER_SIZE, NB_COLS>(t, data, data_size);
    }
};

// instance for Q4
static const tensor_traits<block_q4_0, 4, 4> q4_0_4x4_q8_0;
static const tensor_traits<block_q4_0, 8, 4> q4_0_4x8_q8_0;
static const tensor_traits<block_q4_0, 8, 8> q4_0_8x8_q8_0;

// instance for IQ4
static const tensor_traits<block_iq4_nl, 4, 4> iq4_nl_4x4_q8_0;

}  // namespace ggml::cpu::aarch64

static const ggml::cpu::tensor_traits * ggml_aarch64_get_optimal_repack_type(const struct ggml_tensor * cur) {
    if (cur->type == GGML_TYPE_Q4_0) {
        if (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            if (cur->ne[1] % 8 == 0) {
                return &ggml::cpu::aarch64::q4_0_8x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            if (cur->ne[1] % 4 == 0) {
                return &ggml::cpu::aarch64::q4_0_4x8_q8_0;
            }
        }
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &ggml::cpu::aarch64::q4_0_4x4_q8_0;
            }
        }
    } else if (cur->type == GGML_TYPE_IQ4_NL) {
        if (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            if (cur->ne[1] % 4 == 0) {
                return &ggml::cpu::aarch64::iq4_nl_4x4_q8_0;
            }
        }
    }

    return nullptr;
}

static void ggml_backend_cpu_aarch64_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    tensor->extra = (void *) const_cast<ggml::cpu::tensor_traits *>(ggml_aarch64_get_optimal_repack_type(tensor));

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_aarch64_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor,
                                                       const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto tensor_traits = (ggml::cpu::aarch64::tensor_traits_base *) tensor->extra;
    auto OK            = tensor_traits->repack(tensor, data, size);

    GGML_ASSERT(OK == 0);
    GGML_UNUSED(buffer);
}

static const char * ggml_backend_cpu_aarch64_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_AARCH64";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_aarch64_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);

    if (buffer == nullptr) {
        return nullptr;
    }

    buffer->buft              = buft;
    buffer->iface.init_tensor = ggml_backend_cpu_aarch64_buffer_init_tensor;
    buffer->iface.set_tensor  = ggml_backend_cpu_aarch64_buffer_set_tensor;
    buffer->iface.get_tensor  = nullptr;
    buffer->iface.cpy_tensor  = nullptr;
    return buffer;
}

static size_t ggml_backend_cpu_aarch64_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

namespace ggml::cpu::aarch64 {
class extra_buffer_type : ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const struct ggml_tensor * op) override {
        if (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->buffer &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->buffer->buft == ggml_backend_cpu_aarch64_buffer_type() &&
                ggml_aarch64_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } else if (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->buffer
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->buffer->buft == ggml_backend_cpu_aarch64_buffer_type()
                && ggml_aarch64_get_optimal_repack_type(op->src[0])
                ) {
            if (op->src[1]->buffer && !ggml_backend_buft_is_host(op->src[1]->buffer->buft)) {
                return false;
            }
            if (op->src[1]->type == GGML_TYPE_F32) {
                return true;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        return false;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const struct ggml_tensor * op) override {
        if (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            if (op->src[0]->buffer && op->src[0]->buffer->buft == ggml_backend_cpu_aarch64_buffer_type()) {
                return (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        return nullptr;
    }
};
}  // namespace ggml::cpu::aarch64

ggml_backend_buffer_type_t ggml_backend_cpu_aarch64_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_aarch64 = {
        /* .iface    = */ {
                           /* .get_name         = */ ggml_backend_cpu_aarch64_buffer_type_get_name,
                           /* .alloc_buffer     = */ ggml_backend_cpu_aarch64_buffer_type_alloc_buffer,
                           /* .get_alignment    = */ ggml_backend_cpu_aarch64_buffer_type_get_alignment,
                           /* .get_max_size     = */ nullptr,  // defaults to SIZE_MAX
                           /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
                           /* .is_host          = */ nullptr,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ new ggml::cpu::aarch64::extra_buffer_type(),
    };

    return &ggml_backend_cpu_buffer_type_aarch64;
}

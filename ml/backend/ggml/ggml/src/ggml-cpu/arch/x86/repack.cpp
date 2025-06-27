#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-backend-impl.h"

#include "ggml-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "traits.h"

#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib> // for qsort
#include <cstdio>  // for GGML_ASSERT

#define GGML_CPU_CLANG_WORKAROUND
#include "../../repack.h"

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Woverlength-strings"
#endif

#define UNUSED GGML_UNUSED

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
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    for (int i = 0; i < 8; i++) {
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(y[i]);
    }

    return _mm512_loadu_ps(tmp);
}
static inline __m512 __avx512_repeat_f32cx16_load(__m128i x) {
    float tmp[16];
    uint16_t tmphalf[8];
    _mm_storeu_si128((__m128i*)tmphalf, x);

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 8] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
        tmp[i + 12] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
    }

    return _mm512_loadu_ps(tmp);
}
#endif
static inline __m256 __avx_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_repeat_f32cx8_load(ggml_fp16_t *x) {
    float tmp[8];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(x[i]);
        tmp[i + 4] = GGML_CPU_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline __m256 __avx_rearranged_f32cx8_load(ggml_fp16_t *x, __m128i arrangeMask) {
    uint16_t tmphalf[8];
    float tmp[8];

    _mm_storeu_si128((__m128i*)tmphalf, _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *) x), arrangeMask));
    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_CPU_FP16_TO_FP32(tmphalf[i]);
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

static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

#if defined(__AVX2__) || defined(__AVX512F__)
#if defined(__AVX512F__)
// add int16_t pairwise and return as 512 bit int vector, then add the accumulator
static inline __m512i sum_i16_pairs_acc_int32x16(const __m512i acc, const __m512i x) {
    const __m512i ones = _mm512_set1_epi16(1);
    return _mm512_add_epi32(acc, _mm512_madd_epi16(ones, x));
}

static inline __m512i mul_sum_us8_pairs_acc_int32x16(const __m512i acc, const __m512i ax, const __m512i sy) {
#if defined(__AVX512VNNI__)
    return _mm512_dpbusd_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m512i dot = _mm512_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x16(acc, dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as 512 bit int vector，then add the accumulator
static inline __m512i mul_sum_i8_pairs_acc_int32x16(const __m512i acc, const __m512i x, const __m512i y) {
    const __m512i zero = _mm512_setzero_si512();
    // Get absolute values of x vectors
    const __m512i ax = _mm512_abs_epi8(x);
    // Sign the values of the y vectors
    __mmask64 blt0 = _mm512_movepi8_mask(x);
    const __m512i sy = _mm512_mask_sub_epi8(y, blt0, zero, y);
    return mul_sum_us8_pairs_acc_int32x16(acc, ax, sy);
}
#endif

// add int16_t pairwise and return as 256 bit int vector, then add the accumulator
static inline __m256i sum_i16_pairs_acc_int32x8(const __m256i acc, const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_add_epi32(acc, _mm256_madd_epi16(ones, x));
}

static inline __m256i mul_sum_us8_pairs_acc_int32x8(const __m256i acc, const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, ax, sy);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, ax, sy);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_acc_int32x8(acc, dot);
#endif
}

// Integer variant of the function defined in ggml-quants.c
// multiply int8_t, add results pairwise twice and return as 256 bit int vector, then add the accumulator
static inline __m256i mul_sum_i8_pairs_acc_int32x8(const __m256i acc, const __m256i x, const __m256i y) {
#if defined(__AVXVNNIINT8__)
    return _mm256_dpbssd_epi32(acc, x, y);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_acc_int32x8(acc, ax, sy);
#endif
}
#endif

void ggml_quantize_mat_q8_0_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0x4 * GGML_RESTRICT y = (block_q8_0x4 *) vy;

#if defined(__AVX2__) || defined(__AVX__)
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
            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);

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

            y[i].d[row_iter] = GGML_CPU_FP32_TO_FP16(d);
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

void ggml_quantize_mat_q8_K_4x8(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(QK_K == 256);
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    block_q8_Kx4 * GGML_RESTRICT y = (block_q8_Kx4 *) vy;

#if defined(__AVX2__)
    float iscale[4];
    __m256 srcv[4][32];
    __m256 iscale_vec[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            // Load elements into 4 AVX vectors
            __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 );
            __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 8 );
            __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 16 );
            __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + 24 );

            // Compute max(abs(e)) for the block
            const __m256 signBit = _mm256_set1_ps( -0.0f );
            __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
            __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
            __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
            __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

            __m256 maxAbs = _mm256_max_ps( abs0, abs1 );
            maxAbs = _mm256_max_ps( maxAbs, abs2 );
            maxAbs = _mm256_max_ps( maxAbs, abs3 );

            __m256 mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
            __m256 mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
            __m256 mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
            __m256 mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

            __m256 maskAbs = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));

            srcv[row_iter][0] = v0;
            srcv[row_iter][1] = v1;
            srcv[row_iter][2] = v2;
            srcv[row_iter][3] = v3;

            for (int sb = 1; sb < 8; sb++) {
                // Temporarily stores absolute quant values
                __m256 tempAbs = maxAbs;

                // Load elements into 4 AVX vectors
                __m256 v0 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32);
                __m256 v1 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 8 );
                __m256 v2 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 16 );
                __m256 v3 = _mm256_loadu_ps( x + row_iter * k + i * 256 + sb * 32 + 24 );

                // Compute max(abs(e)) for the block
                __m256 abs0 = _mm256_andnot_ps( signBit, v0 );
                __m256 abs1 = _mm256_andnot_ps( signBit, v1 );
                __m256 abs2 = _mm256_andnot_ps( signBit, v2 );
                __m256 abs3 = _mm256_andnot_ps( signBit, v3 );

                maxAbs = _mm256_max_ps( maxAbs, abs0 );
                maxAbs = _mm256_max_ps( maxAbs, abs1 );
                maxAbs = _mm256_max_ps( maxAbs, abs2 );
                maxAbs = _mm256_max_ps( maxAbs, abs3 );

                __m256 mask_prev = _mm256_cmp_ps( tempAbs, maxAbs, _CMP_EQ_OQ );
                maskAbs = _mm256_and_ps( maskAbs, mask_prev );

                mask0 = _mm256_cmp_ps( maxAbs, v0, _CMP_EQ_OQ );
                mask1 = _mm256_cmp_ps( maxAbs, v1, _CMP_EQ_OQ );
                mask2 = _mm256_cmp_ps( maxAbs, v2, _CMP_EQ_OQ );
                mask3 = _mm256_cmp_ps( maxAbs, v3, _CMP_EQ_OQ );

                __m256 mask_curr = _mm256_or_ps(_mm256_or_ps(mask0, mask1),_mm256_or_ps(mask2, mask3));
                maskAbs =  _mm256_or_ps(maskAbs, mask_curr);

                srcv[row_iter][sb * 4] = v0;
                srcv[row_iter][sb * 4 + 1] = v1;
                srcv[row_iter][sb * 4 + 2] = v2;
                srcv[row_iter][sb * 4 + 3] = v3;
            }

            __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
            max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
            max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
            const float maxScalar = _mm_cvtss_f32( max4 );

            __m256 maxScalarVec = _mm256_set1_ps(maxScalar);

            __m256 mask_next = _mm256_cmp_ps( maxScalarVec, maxAbs, _CMP_EQ_OQ );
            __m256 finalMask = _mm256_and_ps(maskAbs, mask_next);

            const int mask = _mm256_movemask_ps(finalMask);
            iscale[row_iter] = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;

            if(mask) {
                iscale[row_iter] = ( maxScalar != 0.0f ) ? -127.f / maxScalar: 0.0f;
            }

            y[i].d[row_iter] = maxScalar ? 1/iscale[row_iter] : 0;
            iscale_vec[row_iter] = _mm256_set1_ps(iscale[row_iter]);
        }

        __m256i quants_interleaved[32];
        for (int j = 0; j < 32; j++) {
            // Apply the multiplier
            __m256 v0 = _mm256_mul_ps(srcv[0][j], iscale_vec[0]);
            __m256 v1 = _mm256_mul_ps(srcv[1][j], iscale_vec[1]);
            __m256 v2 = _mm256_mul_ps(srcv[2][j], iscale_vec[2]);
            __m256 v3 = _mm256_mul_ps(srcv[3][j], iscale_vec[3]);

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

            // Convert int32 to int16
            i0 = _mm256_packs_epi32( i0, i1 );
            i2 = _mm256_packs_epi32( i2, i3 );
            // Convert int16 to int8
            i0 = _mm256_packs_epi16( i0, i2 );

            //  Permute and store the quantized weights in the required order after the pack instruction
            const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
            i0 = _mm256_permutevar8x32_epi32( i0, perm );

            _mm256_storeu_si256((__m256i *)(y[i].qs + 32 * j), i0);
            quants_interleaved[j] = i0;
        }

        // Masks to shuffle the quants of corresonding sub blocks for rearraning quants for vectorized bsums computation
        __m256i shuffle_mask_sb2 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 0, 1, 4, 5, 6, 7, 8, 9, 8, 9, 12, 13, 14, 15));
        shuffle_mask_sb2 = _mm256_permute2f128_si256(shuffle_mask_sb2, shuffle_mask_sb2, 0);
        __m256i shuffle_mask_sb3 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 0, 1, 6, 7, 8, 9, 10, 11, 8, 9, 14, 15));
        shuffle_mask_sb3 = _mm256_permute2f128_si256(shuffle_mask_sb3, shuffle_mask_sb3, 0);
        __m256i shuffle_mask_sb4 = _mm256_castsi128_si256(_mm_setr_epi8(0, 1, 2, 3, 4, 5, 0, 1, 8, 9, 10, 11, 12, 13, 8, 9));
        shuffle_mask_sb4 = _mm256_permute2f128_si256(shuffle_mask_sb4, shuffle_mask_sb4, 0);

        for (int k = 0; k < 4; k++) {
            // Quants from four different sub blocks are taken
            __m256i q0 = quants_interleaved[k * 8 + 0];
            __m256i q1 = quants_interleaved[k * 8 + 1];
            __m256i q2 = quants_interleaved[k * 8 + 2];
            __m256i q3 = quants_interleaved[k * 8 + 3];
            __m256i q4 = quants_interleaved[k * 8 + 4];
            __m256i q5 = quants_interleaved[k * 8 + 5];
            __m256i q6 = quants_interleaved[k * 8 + 6];
            __m256i q7 = quants_interleaved[k * 8 + 7];


            // The below code block has the first half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
            __m256i sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
            __m256i sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
            __m256i sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
            sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

            __m256i one = _mm256_set1_epi8(1);
            __m256i bsums_r1 = _mm256_maddubs_epi16(one, sb_h1_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q0 = _mm256_srli_epi64(q0, 16);
                q2 = _mm256_srli_epi64(q2, 16);
                q4 = _mm256_srli_epi64(q4, 16);
                q6 = _mm256_srli_epi64(q6, 16);

                sb2_h1_shuffled = _mm256_shuffle_epi8(q2, shuffle_mask_sb2);
                sb_h1_interleaved = _mm256_blend_epi16(q0, sb2_h1_shuffled, 34);
                sb3_h1_shuffled = _mm256_shuffle_epi8(q4, shuffle_mask_sb3);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb3_h1_shuffled, 68);
                sb4_h1_shuffled = _mm256_shuffle_epi8(q6, shuffle_mask_sb4);
                sb_h1_interleaved = _mm256_blend_epi16(sb_h1_interleaved, sb4_h1_shuffled, 136);

                bsums_r1 = _mm256_add_epi16(bsums_r1, _mm256_maddubs_epi16(one, sb_h1_interleaved));
            }

            // The below code block has the second half of different sub blocks shuffled and blended so as to process 2 values from each sub block at a time
            __m256i sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
            __m256i sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
            __m256i sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
            __m256i sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
            sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

            __m256i bsums_r2 = _mm256_maddubs_epi16(one, sb_h2_interleaved);

            for (int l = 0; l < 3; l++) {
                // Quants value shifted to process next two values from each sub block
                q1 = _mm256_srli_epi64(q1, 16);
                q3 = _mm256_srli_epi64(q3, 16);
                q5 = _mm256_srli_epi64(q5, 16);
                q7 = _mm256_srli_epi64(q7, 16);

                sb2_h2_shuffled = _mm256_shuffle_epi8(q3, shuffle_mask_sb2);
                sb_h2_interleaved = _mm256_blend_epi16(q1, sb2_h2_shuffled, 34);
                sb3_h2_shuffled = _mm256_shuffle_epi8(q5, shuffle_mask_sb3);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb3_h2_shuffled, 68);
                sb4_h2_shuffled = _mm256_shuffle_epi8(q7, shuffle_mask_sb4);
                sb_h2_interleaved = _mm256_blend_epi16(sb_h2_interleaved, sb4_h2_shuffled, 136);

                bsums_r2 = _mm256_add_epi16(bsums_r2, _mm256_maddubs_epi16(one, sb_h2_interleaved));
            }

            // Overall bsums in interleaved fashion computed by adding results of both halves
            __m256i bsums_r = _mm256_add_epi16(bsums_r1, bsums_r2);
            _mm256_storeu_si256((__m256i *)(y[i].bsums + 16 * k), bsums_r);
        }
    }

#else

    // scalar
    const int blck_size_interleave = 8;
    float srcv[4][QK_K];
    float iscale[4];

    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f; // absolute max
            float max = 0;

            for (int j = 0; j < QK_K; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                if(amax < fabsf(srcv[row_iter][j])) {
                    amax = fabsf(srcv[row_iter][j]);
                    max = srcv[row_iter][j];
                }
            }

            iscale[row_iter] = amax ? -127.f/max : 0;

            y[i].d[row_iter] = amax ? 1/iscale[row_iter] : 0;
        }

        for (int j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        for (int j = 0; j < QK_K * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id     = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            float x0 = srcv[src_id][src_offset] * iscale[src_id];
            y[i].qs[j] = nearest_int(x0);
            y[i].bsums[index] += y[i].qs[j];
        }
    }
#endif
}

void ggml_gemv_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined(__AVX2__)
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
                const __m256 row_scale_f32 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(a_ptr[b].d));

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

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_0 ,_mm256_shuffle_epi32(rhs_vec_4567_0, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_0, 177) ,rhs_vec_4567_0, 170), _mm256_shuffle_epi32(lhs_vec_0, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_1 ,_mm256_shuffle_epi32(rhs_vec_4567_1, 177), 170), _mm256_shuffle_epi32(lhs_vec_0, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_1, 177) ,rhs_vec_4567_1, 170), _mm256_shuffle_epi32(lhs_vec_0, 255));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_2 ,_mm256_shuffle_epi32(rhs_vec_4567_2, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 0));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_2, 177) ,rhs_vec_4567_2, 170), _mm256_shuffle_epi32(lhs_vec_1, 85));

                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(rhs_vec_0123_3 ,_mm256_shuffle_epi32(rhs_vec_4567_3, 177), 170), _mm256_shuffle_epi32(lhs_vec_1, 170));
                iacc = mul_sum_i8_pairs_acc_int32x8(iacc, _mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_3, 177) ,rhs_vec_4567_3, 170), _mm256_shuffle_epi32(lhs_vec_1, 255));

                // Accumulated values multipled with appropriate scales
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), acc_row);
        }
    }
    return;

#endif
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
                        sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d);
                    }
                }
            }
            for (int j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

void ggml_gemv_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

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

#if defined(__AVX2__)
    // Lookup table to convert signed nibbles to signed bytes
    __m256i signextendlut = _mm256_castsi128_si256(_mm_set_epi8(-1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0));
    signextendlut = _mm256_permute2f128_si256(signextendlut, signextendlut, 0);
    // Shuffle masks to rearrange delta and scale values to multiply with appropriate scales
    __m128i deltamask = _mm_set_epi8(15, 14, 7, 6, 13, 12, 5, 4, 11, 10, 3, 2, 9, 8, 1, 0);
    __m128i scalemask = _mm_set_epi8(7, 7, 3, 3, 6, 6, 2, 2, 5, 5, 1, 1, 4, 4, 0, 0);
    // Permute mask used for easier vector processing at later stages
    __m256i finalpermutemask = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);

    // Mask to extract nibbles from bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);

    int64_t b_nb = n / QK_K;

    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 *)vx;
    const block_q8_K * a_ptr_start = (const block_q8_K *)vy;

    // Process Q8_K blocks one by one
    for (int64_t y = 0; y < nr; y++) {

        // Pointers to LHS blocks of block_q8_K format
        const block_q8_K * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight interleaved block_q4_K structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < nc / 8; x++) {

            // Pointers to RHS blocks
            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_row = _mm256_setzero_ps();
            __m256 acc_min_rows = _mm256_setzero_ps();

            for (int64_t b = 0; b < nb; b++) {

                // Load and convert to FP32 scale from block_q8_K
                const __m256 row_scale_f32 = _mm256_set1_ps((a_ptr[b].d));

                // Load the scale values for the 8 blocks interleaved in block_q4_Kx8
                // col_scale_f32 rearranged so as to multiply with appropriate quants
                const __m256 col_scale_f32 = GGML_F32Cx8_REARRANGE_LOAD(b_ptr[b].d, deltamask);
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                __m256i iacc_b = _mm256_setzero_si256();
                __m256i iacc_min_b = _mm256_setzero_si256();

                const __m256i q8sums = _mm256_loadu_si256((const __m256i * )(a_ptr[b].bsums));
                __m256i q8s = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(q8sums), _mm256_extracti128_si256(q8sums, 1)));
                q8s = _mm256_permute2f128_si256(q8s, q8s, 0);

                // Processes two sub blocks from each Q4_K in each iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight - B0,B1 ....B6,B7
                    const __m256i rhs_raw_vec_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_vec_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_vec_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_vec_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_vec_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_vec_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_vec_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_vec_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // 4-bit -> 8-bit
                    // Values of the first sub block of eight block_q4_K structures for the sb loop
                    const __m256i rhs_vec_0123_00 = _mm256_and_si256(rhs_raw_vec_0123_0, m4b);
                    const __m256i rhs_vec_4567_00 = _mm256_and_si256(rhs_raw_vec_4567_0, m4b);
                    const __m256i rhs_vec_0123_01 = _mm256_and_si256(rhs_raw_vec_0123_1, m4b);
                    const __m256i rhs_vec_4567_01 = _mm256_and_si256(rhs_raw_vec_4567_1, m4b);
                    const __m256i rhs_vec_0123_02 = _mm256_and_si256(rhs_raw_vec_0123_2, m4b);
                    const __m256i rhs_vec_4567_02 = _mm256_and_si256(rhs_raw_vec_4567_2, m4b);
                    const __m256i rhs_vec_0123_03 = _mm256_and_si256(rhs_raw_vec_0123_3, m4b);
                    const __m256i rhs_vec_4567_03 = _mm256_and_si256(rhs_raw_vec_4567_3, m4b);

                    // Values of the second sub block of eight block_q4_K structures when sb = 1
                    const __m256i rhs_vec_0123_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_0, 4), m4b);
                    const __m256i rhs_vec_4567_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_0, 4), m4b);
                    const __m256i rhs_vec_0123_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_1, 4), m4b);
                    const __m256i rhs_vec_4567_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_1, 4), m4b);
                    const __m256i rhs_vec_0123_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_2, 4), m4b);
                    const __m256i rhs_vec_4567_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_2, 4), m4b);
                    const __m256i rhs_vec_0123_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_0123_3, 4), m4b);
                    const __m256i rhs_vec_4567_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_vec_4567_3, 4), m4b);

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q8_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    __m128i scales_rearrange_0 = _mm_shuffle_epi8(mins_and_scales_0, scalemask);
                    __m256i scales_0 = _mm256_cvtepu8_epi16(scales_rearrange_0);

                    // Scales of second sub block in the sb loop
                    __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    __m128i scales_rearrange_1 = _mm_shuffle_epi8(mins_and_scales_1, scalemask);
                    __m256i scales_1 = _mm256_cvtepu8_epi16(scales_rearrange_1);

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    // Load the two sub block values corresponding to sb in block_q8_K in batches of 16 bytes and replicate the same across 256 bit vector
                    __m256i lhs_vec_00 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + sb * 64)));
                    __m256i lhs_vec_01 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 16 + sb * 64)));
                    __m256i lhs_vec_10 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 32 + sb * 64)));
                    __m256i lhs_vec_11 = _mm256_castsi128_si256(_mm_loadu_si128((const __m128i *)(a_ptr[b].qs + 48 + sb * 64)));

                    lhs_vec_00 = _mm256_permute2f128_si256(lhs_vec_00, lhs_vec_00, 0);
                    lhs_vec_01 = _mm256_permute2f128_si256(lhs_vec_01, lhs_vec_01, 0);
                    lhs_vec_10 = _mm256_permute2f128_si256(lhs_vec_10, lhs_vec_10, 0);
                    lhs_vec_11 = _mm256_permute2f128_si256(lhs_vec_11, lhs_vec_11, 0);

                    // Dot product done within 32 bit lanes and accumulated in the same vector
                    // First done for first sub block and thenn for second sub block in each sb
                    // B0(0-3) B4(0-3) B1(0-3) B5(0-3) B2(0-3) B6(0-3) B3(0-3) B7(0-3) with A0(0-3)
                    // B0(4-7) B4(4-7) B1(4-7) B5(4-7) B2(4-7) B6(4-7) B3(4-7) B7(4-7) with A0(4-7)
                    // ...........................................................................
                    // B0(28-31) B4(28-31) B1(28-31) B5(28-31) B2(28-31) B6(28-31) B3(28-31) B7(28-31) with A0(28-31)


                    __m256i iacc_0 = _mm256_setzero_si256();
                    __m256i iacc_1 = _mm256_setzero_si256();

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_00 ,_mm256_shuffle_epi32(rhs_vec_4567_00, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_00, 177) ,rhs_vec_4567_00, 170), _mm256_shuffle_epi32(lhs_vec_00, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_01 ,_mm256_shuffle_epi32(rhs_vec_4567_01, 177), 170), _mm256_shuffle_epi32(lhs_vec_00, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_01, 177) ,rhs_vec_4567_01, 170), _mm256_shuffle_epi32(lhs_vec_00, 255)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_02 ,_mm256_shuffle_epi32(rhs_vec_4567_02, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 0)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_02, 177) ,rhs_vec_4567_02, 170), _mm256_shuffle_epi32(lhs_vec_01, 85)));

                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_03 ,_mm256_shuffle_epi32(rhs_vec_4567_03, 177), 170), _mm256_shuffle_epi32(lhs_vec_01, 170)));
                    iacc_0 = _mm256_add_epi16(iacc_0, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_03, 177) ,rhs_vec_4567_03, 170), _mm256_shuffle_epi32(lhs_vec_01, 255)));

                    iacc_0 = _mm256_madd_epi16(iacc_0, scales_0);

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_10 ,_mm256_shuffle_epi32(rhs_vec_4567_10, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_10, 177) ,rhs_vec_4567_10, 170), _mm256_shuffle_epi32(lhs_vec_10, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_11 ,_mm256_shuffle_epi32(rhs_vec_4567_11, 177), 170), _mm256_shuffle_epi32(lhs_vec_10, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_11, 177) ,rhs_vec_4567_11, 170), _mm256_shuffle_epi32(lhs_vec_10, 255)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_12 ,_mm256_shuffle_epi32(rhs_vec_4567_12, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 0)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_12, 177) ,rhs_vec_4567_12, 170), _mm256_shuffle_epi32(lhs_vec_11, 85)));

                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(rhs_vec_0123_13 ,_mm256_shuffle_epi32(rhs_vec_4567_13, 177), 170), _mm256_shuffle_epi32(lhs_vec_11, 170)));
                    iacc_1 = _mm256_add_epi16(iacc_1, _mm256_maddubs_epi16(_mm256_blend_epi32(_mm256_shuffle_epi32(rhs_vec_0123_13, 177) ,rhs_vec_4567_13, 170), _mm256_shuffle_epi32(lhs_vec_11, 255)));

                    iacc_1 = _mm256_madd_epi16(iacc_1, scales_1);

                    // Accumulate the iacc value for one sb
                    __m256i iacc_sb = _mm256_add_epi32(iacc_0, iacc_1);

                    // Broadcast the bsums of the two sub blocks  of the iteration of Q8_K across the vector
                    // Multiply-Add with corresponding mins of Q4_Kx8 with bsums
                    __m256i q8s_sb = _mm256_shuffle_epi32(q8s, 0);
                    __m256i iacc_min_sb = _mm256_madd_epi16(q8s_sb, mins_01);
                    q8s = _mm256_bsrli_epi128(q8s, 4);

                    // Accumulate for the complete block
                    iacc_b = _mm256_add_epi32(iacc_b, iacc_sb);
                    iacc_min_b = _mm256_add_epi32(iacc_min_b, iacc_min_sb);
                }

                // Multiply-Add with scale values for the complete super block
                acc_row = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_b), _mm256_mul_ps(col_scale_f32, row_scale_f32), acc_row);
                acc_min_rows = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_min_b), _mm256_mul_ps(col_dmin_f32, row_scale_f32), acc_min_rows);

            }

            // Accumulated output values permuted so as to be stored in appropriate order post accumulation
            acc_row = _mm256_permutevar8x32_ps(acc_row, finalpermutemask);
            _mm256_storeu_ps(s + (y * nr + x * 8), _mm256_sub_ps(acc_row, acc_min_rows));
        }
    }

#else

    float sumf[8];
    float sum_minf[8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    const block_q8_K * a_ptr = (const block_q8_K *) vy;
    for (int x = 0; x < nc / ncols_interleaved; x++) {
        const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);

        for (int j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            sum_minf[j] = 0.0;
        }
        for (int l = 0; l < nb; l++) {
            for (int sb = 0; sb < 8; sb++) {
                memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    for (int i = 0; i < blocklen; ++i) {
                        const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            for (int sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                for (int j = 0; j < ncols_interleaved; j++) {
                    sum_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        for (int j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - sum_minf[j];
        }
    }
#endif
}

void ggml_gemm_q4_0_8x8_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
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

#if defined(__AVX2__) || defined(__AVX512F__)
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
                        const __m512i zero = _mm512_setzero_epi32();
                        __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                        __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                        __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                        __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                        __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                        __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                        __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                        __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

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
                    const __m512i zero = _mm512_setzero_epi32();
                    __m512i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_01_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_01_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_01_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_014589CD_3_sp1), lhs_mat_23_2_sp1, rhs_mat_014589CD_2_sp1), lhs_mat_23_1_sp1, rhs_mat_014589CD_1_sp1), lhs_mat_23_0_sp1, rhs_mat_014589CD_0_sp1);
                    __m512i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp1, rhs_mat_2367ABEF_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367ABEF_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367ABEF_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367ABEF_0_sp1);
                    __m512i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_01_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_01_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_01_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_01_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367ABEF_0_sp2);
                    __m512i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_014589CD_3_sp2), lhs_mat_23_2_sp2, rhs_mat_014589CD_2_sp2), lhs_mat_23_1_sp2, rhs_mat_014589CD_1_sp2), lhs_mat_23_0_sp2, rhs_mat_014589CD_0_sp2);
                    __m512i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(mul_sum_i8_pairs_acc_int32x16(zero, lhs_mat_23_3_sp2, rhs_mat_2367ABEF_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367ABEF_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367ABEF_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367ABEF_0_sp2);

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
                        const __m256i zero = _mm256_setzero_si256();
                        __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                        __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                        __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                        __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                        __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                        __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                        __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                        __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

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
                    const __m256i zero = _mm256_setzero_si256();
                    __m256i iacc_mat_00_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_01_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_01_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_01_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_01_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_01_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_01_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_01_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_10_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_0145_3_sp1), lhs_mat_23_2_sp1, rhs_mat_0145_2_sp1), lhs_mat_23_1_sp1, rhs_mat_0145_1_sp1), lhs_mat_23_0_sp1, rhs_mat_0145_0_sp1);
                    __m256i iacc_mat_11_sp1 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp1, rhs_mat_2367_3_sp1), lhs_mat_23_2_sp1, rhs_mat_2367_2_sp1), lhs_mat_23_1_sp1, rhs_mat_2367_1_sp1), lhs_mat_23_0_sp1, rhs_mat_2367_0_sp1);
                    __m256i iacc_mat_00_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_01_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_01_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_01_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_01_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_01_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_01_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_01_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_01_0_sp2, rhs_mat_2367_0_sp2);
                    __m256i iacc_mat_10_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_0145_3_sp2), lhs_mat_23_2_sp2, rhs_mat_0145_2_sp2), lhs_mat_23_1_sp2, rhs_mat_0145_1_sp2), lhs_mat_23_0_sp2, rhs_mat_0145_0_sp2);
                    __m256i iacc_mat_11_sp2 = mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(mul_sum_i8_pairs_acc_int32x8(zero, lhs_mat_23_3_sp2, rhs_mat_2367_3_sp2), lhs_mat_23_2_sp2, rhs_mat_2367_2_sp2), lhs_mat_23_1_sp2, rhs_mat_2367_1_sp2), lhs_mat_23_0_sp2, rhs_mat_2367_0_sp2);

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
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_TO_FP32(a_ptr[l].d[m]);
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

void ggml_gemm_q4_K_8x8_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    const int qk = QK_K;
    const int nb = n / qk;
    const int ncols_interleaved = 8;
    const int blocklen = 8;
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

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

#if defined(__AVX2__) || defined(__AVX512F__)
    const block_q4_Kx8 * b_ptr_start = (const block_q4_Kx8 * ) vx;
    const block_q8_Kx4 * a_ptr_start = (const block_q8_Kx4 * ) vy;
    int64_t b_nb = n / QK_K;
    int64_t y = 0;

    // Mask to mask out nibbles from packed bytes
    const __m256i m4b = _mm256_set1_epi8(0x0F);
    // Permute mask used for easier vector processing at later stages
    __m256i requiredOrder = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);
    int64_t xstart = 0;
    int anr = nr - nr % 16;; // Used to align nr with boundary of 16
#ifdef __AVX512F__
    int anc = nc - nc % 16; // Used to align nc with boundary of 16
    // Mask to mask out nibbles from packed bytes expanded to 512 bit length
    const __m512i m4bexpanded = _mm512_set1_epi8(0x0F);
    //Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into 512 bit vector
                        __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                        __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                        __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                        __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                        __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                        __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                        __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                        __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                        __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                        __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                        __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                        __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                        __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                        __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                        __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                        __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                        __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                        __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                        __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                        __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                        __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                        __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                        __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                        __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                        __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                        __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                        __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                        __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                        __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                        __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                        __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                        __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                        __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                        // Shuffle pattern one - left side input
                        const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                        const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                        const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                        const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                        const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                        const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                        const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                        const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                        const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                        const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                        const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                        const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                        const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                        const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                        const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                        const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                        __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                        __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                        __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                        __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                        __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                        __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                        __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                        __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                        __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                        iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                        iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                        iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                        iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                        iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                        iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                        iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                        __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                        __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                        __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                        __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                        __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                        const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                        // Multiply with appropiate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4  + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                        __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                        __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                        __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                        acc_min_rows[rp * 4] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);
                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = 0; x < anc / 8; x += 2) {

            const block_q4_Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const block_q4_Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, (_MM_PERM_ENUM)221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, (_MM_PERM_ENUM)221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, (_MM_PERM_ENUM)221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, (_MM_PERM_ENUM)221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, (_MM_PERM_ENUM)221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, (_MM_PERM_ENUM)221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, (_MM_PERM_ENUM)221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, (_MM_PERM_ENUM)221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, (_MM_PERM_ENUM)221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, (_MM_PERM_ENUM)221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, (_MM_PERM_ENUM)221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, (_MM_PERM_ENUM)221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, (_MM_PERM_ENUM)221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, (_MM_PERM_ENUM)221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, (_MM_PERM_ENUM)221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, (_MM_PERM_ENUM)221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, (_MM_PERM_ENUM)238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, (_MM_PERM_ENUM)238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                    __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                    __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                    __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                    __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                    __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                    __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                    //Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into a 512 bit vector
                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                    __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                    __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                    __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                    __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                    __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                    __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                    __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                    __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                    __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                    const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                    const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                    const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                    const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, (_MM_PERM_ENUM)245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, (_MM_PERM_ENUM)245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, (_MM_PERM_ENUM)245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, (_MM_PERM_ENUM)245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                    const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, (_MM_PERM_ENUM)245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, (_MM_PERM_ENUM)245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                    const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, (_MM_PERM_ENUM)245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, (_MM_PERM_ENUM)245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, (_MM_PERM_ENUM)245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, (_MM_PERM_ENUM)245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, (_MM_PERM_ENUM)245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, (_MM_PERM_ENUM)245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                    const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, (_MM_PERM_ENUM)245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, (_MM_PERM_ENUM)245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                    const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, (_MM_PERM_ENUM)245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, (_MM_PERM_ENUM)245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, (_MM_PERM_ENUM)78), iacc_mat_01_0);
                    __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, (_MM_PERM_ENUM)78), iacc_mat_11_0);
                    __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, (_MM_PERM_ENUM)78), iacc_mat_01_1);
                    __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, (_MM_PERM_ENUM)78));
                    __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, (_MM_PERM_ENUM)78), iacc_mat_11_1);

                    __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropiate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)0), mins_01);
                    __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)85), mins_01);
                    __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)170), mins_01);
                    __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, (_MM_PERM_ENUM)255), mins_01);

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumlated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    if (anc != nc) {
        xstart = anc/8;
        y = 0;
    }
#endif //AVX512F

    // Take group of four block_q8_Kx4 structures at each pass of the loop and perform dot product operation
    for (; y < anr / 4; y += 4) {

        const block_q8_Kx4 * a_ptrs[4];

        a_ptrs[0] = a_ptr_start + (y * nb);
        for (int i = 0; i < 3; ++i) {
            a_ptrs[i + 1] = a_ptrs[i] + nb;
        }

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[16];
            for (int i = 0; i < 16; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_K for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)


                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    for (int rp = 0; rp < 4; rp++) {

                        // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                        // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                        __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 256 * sb)));
                        __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                        __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                        __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 32 + 256 * sb)));
                        __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                        __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                        __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 64 + 256 * sb)));
                        __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                        __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                        __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 96 + 256 * sb)));
                        __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                        __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                        __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 128 + 256 * sb)));
                        __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                        __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                        __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 160 + 256 * sb)));
                        __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                        __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                        __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 192 + 256 * sb)));
                        __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                        __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                        __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].qs + 224 + 256 * sb)));
                        __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                        __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                        // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                        __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptrs[rp][b].bsums + 16 * sb)));
                        __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                        lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                        // Shuffle pattern one - left side input
                        const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                        const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                        const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                        const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                        const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                        const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                        const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                        const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                        const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                        const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                        const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                        const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                        const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                        const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                        const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                        const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                        // Shuffle pattern two- left side input
                        const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                        const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                        const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                        const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                        const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                        const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                        const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                        const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                        const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                        const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                        const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                        const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                        const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                        const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                        const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                        const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                        // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                        __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                        __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                        __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                        __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                        __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                        __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                        __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                        __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                        __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                        __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                        __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                        __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                        __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                        __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                        __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                        __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                        __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                        // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                        iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                        iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                        iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                        iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                        iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                        iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                        iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                        iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                        // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                        __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                        __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                        __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                        __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                        __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                        __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                        __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                        __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                        __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                        __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                        __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                        __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                        // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                        const __m128 row_scale_f32_sse = _mm_load_ps(a_ptrs[rp][b].d);
                        const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);//GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                        // Multiply with appropiate scales and accumulate (for both d and dmin) below
                        acc_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[rp * 4]);
                        acc_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[rp * 4 + 1]);
                        acc_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[rp * 4 + 2]);
                        acc_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[rp * 4 + 3]);

                        __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                        __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                        __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                        __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                        acc_min_rows[rp * 4] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[rp * 4]);
                        acc_min_rows[rp * 4 + 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[rp * 4 + 1]);
                        acc_min_rows[rp * 4 + 2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[rp * 4 + 2]);
                        acc_min_rows[rp * 4 + 3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[rp * 4 + 3]);

                    }
                }
            }
            // Store the accumulated values
            for (int i = 0; i < 16; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
    for (; y < nr / 4; y++) {

        const block_q8_Kx4 * a_ptr = a_ptr_start + (y * nb);

        for (int64_t x = xstart; x < nc / 8; x++) {

            const block_q4_Kx8 * b_ptr = b_ptr_start + (x * b_nb);

            // Master FP accumulators
            __m256 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm256_setzero_ps();
            }

            __m256 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm256_setzero_ps();
            }

            for (int64_t b = 0; b < nb; b++) {

                // Scale values - Load the eight scale values of block_q4_Kx8
                const __m256 col_scale_f32 = GGML_F32Cx8_LOAD(b_ptr[b].d);

                // dmin values - Load the eight dmin values of block_q4_Kx8
                const __m256 col_dmin_f32 = GGML_F32Cx8_LOAD(b_ptr[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < QK_K / 64; sb++) {

                    // Load the eight block_q4_k for two sub blocks quantized values interleaved with each other in chunks of eight bytes - B0,B1 ....B6,B7
                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr[b].qs + 224 + sb * 256));

                    // Save the values in the following vectors in the formats B0B1B4B5, B2B3B6B7 for further processing and storing of values
                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    // 4-bit -> 8-bit
                    // First sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_00 = _mm256_and_si256(rhs_raw_mat_0145_0, m4b); //B00(0-7) B01(0-7) B04(0-7) B05(0-7)
                    const __m256i rhs_mat_2367_00 = _mm256_and_si256(rhs_raw_mat_2367_0, m4b); //B02(0-7) B03(0-7) B06(0-7) B07(0-7)

                    const __m256i rhs_mat_0145_01 = _mm256_and_si256(rhs_raw_mat_0145_1, m4b); //B00(8-15) B01(8-15) B04(8-15) B05(8-15)
                    const __m256i rhs_mat_2367_01 = _mm256_and_si256(rhs_raw_mat_2367_1, m4b); //B02(8-15) B03(8-15) B06(8-15) B07(8-15)

                    const __m256i rhs_mat_0145_02 = _mm256_and_si256(rhs_raw_mat_0145_2, m4b); //B00(16-23) B01(16-23) B04(16-23) B05(16-23)
                    const __m256i rhs_mat_2367_02 = _mm256_and_si256(rhs_raw_mat_2367_2, m4b); //B02(16-23) B03(16-23) B06(16-23) B07(16-23)

                    const __m256i rhs_mat_0145_03 = _mm256_and_si256(rhs_raw_mat_0145_3, m4b); //B00(24-31) B01(24-31) B04(24-31) B05(24-31)
                    const __m256i rhs_mat_2367_03 = _mm256_and_si256(rhs_raw_mat_2367_3, m4b); //B02(24-31) B03(24-31) B06(24-31) B07(24-31)

                    // Second sub block of the two sub blocks processed in the iteration
                    const __m256i rhs_mat_0145_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_0, 4), m4b); //B10(0-7) B11(0-7) B14(0-7) B15(0-7)
                    const __m256i rhs_mat_2367_10 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_0, 4), m4b); //B12(0-7) B13(0-7) B16(0-7) B17(0-7)

                    const __m256i rhs_mat_0145_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_1, 4), m4b); //B10(8-15) B11(8-15) B14(8-15) B15(8-15)
                    const __m256i rhs_mat_2367_11 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_1, 4), m4b); //B12(8-15) B13(8-15) B16(8-15) B17(8-15)

                    const __m256i rhs_mat_0145_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_2, 4), m4b); //B10(16-23) B11(16-23) B14(16-23) B15(16-23)
                    const __m256i rhs_mat_2367_12 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_2, 4), m4b); //B12(16-23) B13(16-23) B16(16-23) B17(16-23)

                    const __m256i rhs_mat_0145_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_0145_3, 4), m4b); //B10(24-31) B11(24-31) B14(24-31) B15(24-31)
                    const __m256i rhs_mat_2367_13 = _mm256_and_si256(_mm256_srli_epi16(rhs_raw_mat_2367_3, 4), m4b); //B12(24-31) B13(24-31) B16(24-31) B17(24-31)

                    // Shuffle pattern one - right side input
                    const __m256i rhs_mat_0145_00_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3)
                    const __m256i rhs_mat_2367_00_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3)

                    const __m256i rhs_mat_0145_01_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11)
                    const __m256i rhs_mat_2367_01_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11)

                    const __m256i rhs_mat_0145_02_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19)
                    const __m256i rhs_mat_2367_02_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19)

                    const __m256i rhs_mat_0145_03_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27)
                    const __m256i rhs_mat_2367_03_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27)

                    const __m256i rhs_mat_0145_10_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3)
                    const __m256i rhs_mat_2367_10_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3)

                    const __m256i rhs_mat_0145_11_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11)
                    const __m256i rhs_mat_2367_11_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11)

                    const __m256i rhs_mat_0145_12_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19)
                    const __m256i rhs_mat_2367_12_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19)

                    const __m256i rhs_mat_0145_13_sp1 = _mm256_shuffle_epi32(rhs_mat_0145_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27)
                    const __m256i rhs_mat_2367_13_sp1 = _mm256_shuffle_epi32(rhs_mat_2367_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27)

                    // Shuffle pattern two - right side input
                    const __m256i rhs_mat_0145_00_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7)
                    const __m256i rhs_mat_2367_00_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7)

                    const __m256i rhs_mat_0145_01_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15)
                    const __m256i rhs_mat_2367_01_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15)

                    const __m256i rhs_mat_0145_02_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23)
                    const __m256i rhs_mat_2367_02_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23)

                    const __m256i rhs_mat_0145_03_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31)
                    const __m256i rhs_mat_2367_03_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31)

                    const __m256i rhs_mat_0145_10_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7)
                    const __m256i rhs_mat_2367_10_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7)

                    const __m256i rhs_mat_0145_11_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15)
                    const __m256i rhs_mat_2367_11_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15)

                    const __m256i rhs_mat_0145_12_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23)
                    const __m256i rhs_mat_2367_12_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23)

                    const __m256i rhs_mat_0145_13_sp2 = _mm256_shuffle_epi32(rhs_mat_0145_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31)
                    const __m256i rhs_mat_2367_13_sp2 = _mm256_shuffle_epi32(rhs_mat_2367_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31)

                    uint32_t utmp_0[4], utmp_1[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_0, b_ptr[b].scales + 24 * sb, 12);
                    utmp_0[3] = ((utmp_0[2] >> 4) & kmask2) | (((utmp_0[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp_0[1] & kmask1;
                    utmp_0[1] = (utmp_0[2] & kmask2) | (((utmp_0[0] >> 6) & kmask3) << 4);
                    utmp_0[2] = uaux_0;
                    utmp_0[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures when sb = 1
                    memcpy(utmp_1, b_ptr[b].scales + 12 + sb * 24, 12);
                    utmp_1[3] = ((utmp_1[2] >> 4) & kmask2) | (((utmp_1[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_1 = utmp_1[1] & kmask1;
                    utmp_1[1] = (utmp_1[2] & kmask2) | (((utmp_1[0] >> 6) & kmask3) << 4);
                    utmp_1[2] = uaux_1;
                    utmp_1[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m128i mins_and_scales_0 = _mm_set_epi32(utmp_0[3], utmp_0[2], utmp_0[1], utmp_0[0]);
                    const __m256i scales_0 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m128i mins_and_scales_1 = _mm_set_epi32(utmp_1[3], utmp_1[2], utmp_1[1], utmp_1[0]);
                    const __m256i scales_1 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m256i mins_01 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(_mm_shuffle_epi32(mins_and_scales_0, 78), _mm_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m256i scale_0145_0 = _mm256_shuffle_epi32(scales_0, 68);
                    const __m256i scale_2367_0 = _mm256_shuffle_epi32(scales_0, 238);

                    const __m256i scale_0145_1 = _mm256_shuffle_epi32(scales_1, 68);
                    const __m256i scale_2367_1 = _mm256_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_01_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 0);
                    __m256i lhs_mat_23_00 = _mm256_permute2f128_si256(lhs_mat_0123_00, lhs_mat_0123_00, 17);
                    __m256i lhs_mat_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_01_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 0);
                    __m256i lhs_mat_23_01 = _mm256_permute2f128_si256(lhs_mat_0123_01, lhs_mat_0123_01, 17);
                    __m256i lhs_mat_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_01_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 0);
                    __m256i lhs_mat_23_02 = _mm256_permute2f128_si256(lhs_mat_0123_02, lhs_mat_0123_02, 17);
                    __m256i lhs_mat_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_01_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 0);
                    __m256i lhs_mat_23_03 = _mm256_permute2f128_si256(lhs_mat_0123_03, lhs_mat_0123_03, 17);
                    __m256i lhs_mat_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_01_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 0);
                    __m256i lhs_mat_23_10 = _mm256_permute2f128_si256(lhs_mat_0123_10, lhs_mat_0123_10, 17);
                    __m256i lhs_mat_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_01_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 0);
                    __m256i lhs_mat_23_11 = _mm256_permute2f128_si256(lhs_mat_0123_11, lhs_mat_0123_11, 17);
                    __m256i lhs_mat_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_01_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 0);
                    __m256i lhs_mat_23_12 = _mm256_permute2f128_si256(lhs_mat_0123_12, lhs_mat_0123_12, 17);
                    __m256i lhs_mat_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_01_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 0);
                    __m256i lhs_mat_23_13 = _mm256_permute2f128_si256(lhs_mat_0123_13, lhs_mat_0123_13, 17);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_0123_01, lhs_bsums_hsum_0123_01, 0);

                    // Shuffle pattern one - left side input
                    const __m256i lhs_mat_01_00_sp1 = _mm256_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m256i lhs_mat_23_00_sp1 = _mm256_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3) A02(0-3) A03(0-3)

                    const __m256i lhs_mat_01_01_sp1 = _mm256_shuffle_epi32(lhs_mat_01_01, 160);  //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m256i lhs_mat_23_01_sp1 = _mm256_shuffle_epi32(lhs_mat_23_01, 160);  //A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11) A02(8-11) A03(8-11)

                    const __m256i lhs_mat_01_02_sp1 = _mm256_shuffle_epi32(lhs_mat_01_02, 160);  //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m256i lhs_mat_23_02_sp1 = _mm256_shuffle_epi32(lhs_mat_23_02, 160);  //A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19) A02(16-19) A03(16-19)

                    const __m256i lhs_mat_01_03_sp1 = _mm256_shuffle_epi32(lhs_mat_01_03, 160);  //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m256i lhs_mat_23_03_sp1 = _mm256_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27) A02(24-27) A03(24-27)

                    const __m256i lhs_mat_01_10_sp1 = _mm256_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m256i lhs_mat_23_10_sp1 = _mm256_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3) A12(0-3) A13(0-3)

                    const __m256i lhs_mat_01_11_sp1 = _mm256_shuffle_epi32(lhs_mat_01_11, 160);  //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m256i lhs_mat_23_11_sp1 = _mm256_shuffle_epi32(lhs_mat_23_11, 160);  //A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11) A12(8-11) A13(8-11)

                    const __m256i lhs_mat_01_12_sp1 = _mm256_shuffle_epi32(lhs_mat_01_12, 160);  //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m256i lhs_mat_23_12_sp1 = _mm256_shuffle_epi32(lhs_mat_23_12, 160);  //A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19) A12(16-19) A13(16-19)

                    const __m256i lhs_mat_01_13_sp1 = _mm256_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m256i lhs_mat_23_13_sp1 = _mm256_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27) A12(24-27) A13(24-27)

                    // Shuffle pattern two- left side input
                    const __m256i lhs_mat_01_00_sp2 = _mm256_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m256i lhs_mat_23_00_sp2 = _mm256_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7) A02(4-7) A03(4-7)

                    const __m256i lhs_mat_01_01_sp2 = _mm256_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m256i lhs_mat_23_01_sp2 = _mm256_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15) A02(12-15) A03(12-15)

                    const __m256i lhs_mat_01_02_sp2 = _mm256_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m256i lhs_mat_23_02_sp2 = _mm256_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23) A02(20-23) A03(20-23)

                    const __m256i lhs_mat_01_03_sp2 = _mm256_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m256i lhs_mat_23_03_sp2 = _mm256_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31) A02(28-31) A03(28-31)

                    const __m256i lhs_mat_01_10_sp2 = _mm256_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m256i lhs_mat_23_10_sp2 = _mm256_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7) A12(4-7) A13(4-7)

                    const __m256i lhs_mat_01_11_sp2 = _mm256_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m256i lhs_mat_23_11_sp2 = _mm256_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15) A12(12-15) A13(12-15)

                    const __m256i lhs_mat_01_12_sp2 = _mm256_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m256i lhs_mat_23_12_sp2 = _mm256_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23) A12(20-23) A13(20-23)

                    const __m256i lhs_mat_01_13_sp2 = _mm256_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m256i lhs_mat_23_13_sp2 = _mm256_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31) A12(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m256i iacc_mat_00_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_01_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_01_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_01_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_01_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_01_00_sp1));
                    __m256i iacc_mat_10_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_0145_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_11_0_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp1, lhs_mat_23_03_sp1), _mm256_maddubs_epi16(rhs_mat_2367_02_sp1, lhs_mat_23_02_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp1, lhs_mat_23_01_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp1, lhs_mat_23_00_sp1));
                    __m256i iacc_mat_00_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_01_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_01_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_01_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_01_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_01_10_sp1));
                    __m256i iacc_mat_10_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_0145_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp1, lhs_mat_23_10_sp1));
                    __m256i iacc_mat_11_1_sp1 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp1, lhs_mat_23_13_sp1), _mm256_maddubs_epi16(rhs_mat_2367_12_sp1, lhs_mat_23_12_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp1, lhs_mat_23_11_sp1)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp1, lhs_mat_23_10_sp1));

                    __m256i iacc_mat_00_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_01_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_01_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_01_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_01_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_01_00_sp2));
                    __m256i iacc_mat_10_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_0145_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_11_0_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_03_sp2, lhs_mat_23_03_sp2), _mm256_maddubs_epi16(rhs_mat_2367_02_sp2, lhs_mat_23_02_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_01_sp2, lhs_mat_23_01_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_00_sp2, lhs_mat_23_00_sp2));
                    __m256i iacc_mat_00_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_01_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_01_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_01_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_01_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_01_10_sp2));
                    __m256i iacc_mat_10_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_0145_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_0145_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_0145_10_sp2, lhs_mat_23_10_sp2));
                    __m256i iacc_mat_11_1_sp2 = _mm256_add_epi16(_mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(rhs_mat_2367_13_sp2, lhs_mat_23_13_sp2), _mm256_maddubs_epi16(rhs_mat_2367_12_sp2, lhs_mat_23_12_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_11_sp2, lhs_mat_23_11_sp2)), _mm256_maddubs_epi16(rhs_mat_2367_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m256i iacc_mat_00_0 = _mm256_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m256i iacc_mat_01_0 = _mm256_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m256i iacc_mat_10_0 = _mm256_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m256i iacc_mat_11_0 = _mm256_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m256i iacc_mat_00_1 = _mm256_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m256i iacc_mat_01_1 = _mm256_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m256i iacc_mat_10_1 = _mm256_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m256i iacc_mat_11_1 = _mm256_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    iacc_mat_00_0 = _mm256_madd_epi16(iacc_mat_00_0, scale_0145_0);
                    iacc_mat_01_0 = _mm256_madd_epi16(iacc_mat_01_0, scale_2367_0);
                    iacc_mat_10_0 = _mm256_madd_epi16(iacc_mat_10_0, scale_0145_0);
                    iacc_mat_11_0 = _mm256_madd_epi16(iacc_mat_11_0, scale_2367_0);

                    iacc_mat_00_1 = _mm256_madd_epi16(iacc_mat_00_1, scale_0145_1);
                    iacc_mat_01_1 = _mm256_madd_epi16(iacc_mat_01_1, scale_2367_1);
                    iacc_mat_10_1 = _mm256_madd_epi16(iacc_mat_10_1, scale_0145_1);
                    iacc_mat_11_1 = _mm256_madd_epi16(iacc_mat_11_1, scale_2367_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m256i iacc_row_0_0 = _mm256_blend_epi32(iacc_mat_00_0, _mm256_shuffle_epi32(iacc_mat_01_0, 78), 204);
                    __m256i iacc_row_1_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0, 204);
                    __m256i iacc_row_2_0 = _mm256_blend_epi32(iacc_mat_10_0, _mm256_shuffle_epi32(iacc_mat_11_0, 78), 204);
                    __m256i iacc_row_3_0 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0, 204);
                    __m256i iacc_row_0_1 = _mm256_blend_epi32(iacc_mat_00_1, _mm256_shuffle_epi32(iacc_mat_01_1, 78), 204);
                    __m256i iacc_row_1_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1, 204);
                    __m256i iacc_row_2_1 = _mm256_blend_epi32(iacc_mat_10_1, _mm256_shuffle_epi32(iacc_mat_11_1, 78), 204);
                    __m256i iacc_row_3_1 = _mm256_blend_epi32(_mm256_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1, 204);

                    __m256i iacc_row_0 = _mm256_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m256i iacc_row_1 = _mm256_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m256i iacc_row_2 = _mm256_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m256i iacc_row_3 = _mm256_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32 = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse); //GGML_F32Cx8_REPEAT_LOAD(a_ptrs[rp][b].d, loadMask);

                    // Multiply with appropiate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_0), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_1), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_2), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_3), _mm256_mul_ps(col_scale_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m256i iacc_row_min_0 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m256i iacc_row_min_1 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m256i iacc_row_min_2 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m256i iacc_row_min_3 = _mm256_madd_epi16(_mm256_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_0), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_1), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_2), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(iacc_row_min_3), _mm256_mul_ps(col_dmin_f32, _mm256_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }

            // Store the accumulated values
            for (int i = 0; i < 4; i++) {
                _mm256_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm256_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }

#else

    float sumf[4][8];
    float sum_minf[4][8];
    uint32_t utmp[32];
    int sumi1;
    int sumi2;
    int sumi;

    for (int y = 0; y < nr / 4; y++) {
        const block_q8_Kx4 * a_ptr = (const block_q8_Kx4 *) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_Kx8 * b_ptr = (const block_q4_Kx8 *) vx + (x * nb);
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    sum_minf[m][j] = 0.0;
                }
            }
            for (int l = 0; l < nb; l++) {
                for (int sb = 0; sb < 8; sb++) {
                    memcpy(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                const int v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_TO_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                for (int sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    for(int m = 0; m < 4; m++) {
                        const int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        for(int j = 0; j < ncols_interleaved; j++) {
                            sum_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_TO_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++) {
                for (int j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - sum_minf[m][j];
                }
            }
        }
    }
#endif
}

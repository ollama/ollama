#include "vec.h"

#include <cassert>

// precomputed gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_f16[1 << 16];

// precomputed quick gelu table for f16 (128 KB)
ggml_fp16_t ggml_table_gelu_quick_f16[1 << 16];

void ggml_vec_dot_f32(int n, float * GGML_RESTRICT s, size_t bs, const float * GGML_RESTRICT x, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
   assert(nrc == 1);
   GGML_UNUSED(nrc);
   GGML_UNUSED(bx);
   GGML_UNUSED(by);
   GGML_UNUSED(bs);

#if defined(GGML_SIMD)
    float sumf = 0.0f;
    const int np = (n & ~(GGML_F32_STEP - 1));

    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];

    for (int i = 0; i < np; i += GGML_F32_STEP) {
        for (int j = 0; j < GGML_F32_ARR; j++) {
            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += x[i]*y[i];
    }
#else
    // scalar
    ggml_float sumf = 0.0;
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(x[i]*y[i]);
    }
#endif

    *s = sumf;
}

void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, size_t bs, ggml_bf16_t * GGML_RESTRICT x, size_t bx, ggml_bf16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    int i = 0;
    ggml_float sumf = 0;

#if defined(__AVX512BF16__)
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        c1 = _mm512_dpbf16_ps(c1, m512bh(_mm512_loadu_si512((x + i))),
                             m512bh(_mm512_loadu_si512((y + i))));
        c2 = _mm512_dpbf16_ps(c2, m512bh(_mm512_loadu_si512((x + i + 32))),
                             m512bh(_mm512_loadu_si512((y + i + 32))));
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#elif defined(__AVX512F__)
#define LOAD(p) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)(p))), 16))
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm512_add_ps(_mm512_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c2);
    }
    sumf += (ggml_float)_mm512_reduce_add_ps(c1);
    sumf += (ggml_float)_mm512_reduce_add_ps(c2);

#undef LOAD
#elif defined(__AVX2__) || defined(__AVX__)
#if defined(__AVX2__)
#define LOAD(p) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16))
#else
#define LOAD(p) _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(p))), 16)), (_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_bsrli_si128(_mm_loadu_si128((const __m128i *)(p)), 8)), 16)), 1))
#endif
    __m256 c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps();
    __m256 c3 = _mm256_setzero_ps();
    __m256 c4 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        c1 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i), LOAD(y + i)), c1);
        c2 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 8), LOAD(y + i + 8)), c2);
        c3 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 16), LOAD(y + i + 16)), c3);
        c4 = _mm256_add_ps(_mm256_mul_ps(LOAD(x + i + 24), LOAD(y + i + 24)), c4);
    }
    __m128 g;
    c1 = _mm256_add_ps(_mm256_add_ps(c1, c3),
                       _mm256_add_ps(c2, c4));
    g = _mm_add_ps(_mm256_extractf128_ps(c1, 1),
                   _mm256_castps256_ps128(c1));
    g = _mm_add_ps(g, _mm_movehl_ps(g, g));
    g = _mm_add_ss(g, _mm_movehdup_ps(g));
    sumf += (ggml_float)_mm_cvtss_f32(g);

#undef LOAD
#endif

    for (; i < n; ++i) {
        sumf += (ggml_float)(GGML_BF16_TO_FP32(x[i]) *
                             GGML_BF16_TO_FP32(y[i]));
    }
    *s = sumf;
}

void ggml_vec_dot_f16(int n, float * GGML_RESTRICT s, size_t bs, ggml_fp16_t * GGML_RESTRICT x, size_t bx, ggml_fp16_t * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);

    ggml_float sumf = 0.0;

#if defined(GGML_SIMD)
    const int np = (n & ~(GGML_F16_STEP - 1));

    GGML_F16_VEC sum[GGML_F16_ARR] = { GGML_F16_VEC_ZERO };

    GGML_F16_VEC ax[GGML_F16_ARR];
    GGML_F16_VEC ay[GGML_F16_ARR];

    for (int i = 0; i < np; i += GGML_F16_STEP) {
        for (int j = 0; j < GGML_F16_ARR; j++) {
            ax[j] = GGML_F16_VEC_LOAD(x + i + j*GGML_F16_EPR, j);
            ay[j] = GGML_F16_VEC_LOAD(y + i + j*GGML_F16_EPR, j);

            sum[j] = GGML_F16_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }

    // reduce sum0..sum3 to sum0
    GGML_F16_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#else
    for (int i = 0; i < n; ++i) {
        sumf += (ggml_float)(GGML_FP16_TO_FP32(x[i])*GGML_FP16_TO_FP32(y[i]));
    }
#endif

    *s = sumf;
}

void ggml_vec_silu_f32(const int n, float * y, const float * x) {
    int i = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        _mm512_storeu_ps(y + i, ggml_v_silu(_mm512_loadu_ps(x + i)));
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        _mm256_storeu_ps(y + i, ggml_v_silu(_mm256_loadu_ps(x + i)));
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        _mm_storeu_ps(y + i, ggml_v_silu(_mm_loadu_ps(x + i)));
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        vst1q_f32(y + i, ggml_v_silu(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        y[i] = ggml_silu_f32(x[i]);
    }
}

ggml_float ggml_vec_soft_max_f32(const int n, float * y, const float * x, float max) {
    int i = 0;
    ggml_float sum = 0;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    for (; i + 15 < n; i += 16) {
        __m512 val = ggml_v_expf(_mm512_sub_ps(_mm512_loadu_ps(x + i),
                                               _mm512_set1_ps(max)));
        _mm512_storeu_ps(y + i, val);
        sum += (ggml_float)_mm512_reduce_add_ps(val);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    for (; i + 7 < n; i += 8) {
        __m256 val = ggml_v_expf(_mm256_sub_ps(_mm256_loadu_ps(x + i),
                                               _mm256_set1_ps(max)));
        _mm256_storeu_ps(y + i, val);
        __m128 val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1),
                                 _mm256_castps256_ps128(val));
        val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += (ggml_float)_mm_cvtss_f32(val2);
    }
#elif defined(__SSE2__)
    for (; i + 3 < n; i += 4) {
        __m128 val = ggml_v_expf(_mm_sub_ps(_mm_loadu_ps(x + i),
                                            _mm_set1_ps(max)));
        _mm_storeu_ps(y + i, val);
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
        val = _mm_add_ps(val, _mm_movehl_ps(val, val));
        val = _mm_add_ss(val, _mm_movehdup_ps(val));
#else
        __m128 tmp = _mm_shuffle_ps(val, val, _MM_SHUFFLE(2, 3, 0, 1));
        val = _mm_add_ps(val, tmp);
        tmp = _mm_movehl_ps(tmp, val);
        val = _mm_add_ss(val, tmp);
#endif
        sum += (ggml_float)_mm_cvtss_f32(val);
    }
#elif defined(__ARM_NEON) && defined(__aarch64__)
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i),
                                                vdupq_n_f32(max)));
        vst1q_f32(y + i, val);
        sum += (ggml_float)vaddvq_f32(val);
    }
#endif
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (ggml_float)val;
        y[i] = val;
    }
    return sum;
}

ggml_float ggml_vec_log_soft_max_f32(const int n, float * y, const float * x, float max) {
    // log(soft_max) = log(soft_max_i / soft_max_sum) = log(soft_max_i) - log(soft_max_sum) = (logit_i - max) - log(soft_max_i)

    int i = 0;
    ggml_float sum = 0;
    for (; i < n; ++i) {
        float val = x[i] - max;
        y[i] = val;
        sum += (ggml_float)expf(val);
    }
    return sum = (ggml_float)logf(sum);
}

#define MXFP4 32
typedef struct {
    uint8_t d;              // scale E8M0 float 
    uint8_t qs[MXFP4 / 2];  // (32) 4 bit elements E2M1 float
} block_mxfp4;
static_assert(sizeof(block_mxfp4) == sizeof(uint8_t) + MXFP4/2, "wrong mxfp4 block size/padding");
#define MXFP4_VALS {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0}

void ggml_vec_dot_mxfp4(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const float * GGML_RESTRICT y, size_t by, int nrc) {
    assert(nrc == 1);
    GGML_UNUSED(nrc);
    GGML_UNUSED(bx);
    GGML_UNUSED(by);
    GGML_UNUSED(bs);
    ggml_float mxfp4_table[] = MXFP4_VALS;

#if defined(GGML_SIMD)
    float sumf = 0.0f;
    const int np = (n & ~(GGML_F32_STEP - 1));
    const block_mxfp4 * GGML_RESTRICT xx = (const block_mxfp4 *) vx;
    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

    GGML_F32_VEC scalev;
    GGML_F32_VEC ax[GGML_F32_ARR];
    GGML_F32_VEC ay[GGML_F32_ARR];
    for (int i = 0; i < np; i += GGML_F32_STEP) { // ARM: +16  AVX512: +64
        for (int j = 0; j < GGML_F32_ARR; j++) { // ARM: 0 .. 4 AVX512: 0 .. 4
            // convert GGML_F32_ARR X elements 
            const int ib = (i + j*GGML_F32_EPR) / MXFP4;
            const block_mxfp4 * GGML_RESTRICT x = &xx[ib];
            union {
                uint32_t as_bits;
                float as_value;
            } scale;
            scale.as_bits = (((uint32_t)x->d) << 23);
            scalev = GGML_F32_VEC_SET1(scale.as_value);
            float xf[GGML_F32_EPR]= {0.f};
            assert(((i+j*GGML_F32_EPR) % MXFP4)+GGML_F32_ARR < MXFP4 && "block overrun");
            for (int qi = 0; qi < GGML_F32_EPR/2 ; ++qi) {
                xf[qi*2] = mxfp4_table[(x->qs[((i+j*GGML_F32_EPR)%MXFP4)/2+qi] & 0xf)];
                xf[qi*2+1] = mxfp4_table[(x->qs[((i+j*GGML_F32_EPR)%MXFP4)/2+qi] & 0xf0) >> 4];
            }

            ax[j] = GGML_F32_VEC_MUL(GGML_F32_VEC_LOAD(xf), scalev);
            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
        }
    }
    GGML_F32_VEC_REDUCE(sumf, sum);

    // leftovers
    for (int i = np; i < n; i+=2) {
        const int ib = i / MXFP4;
        const block_mxfp4 * GGML_RESTRICT x = &xx[ib];
        union {
            uint32_t as_bits;
            float as_value;
        } scale;
        scale.as_bits = (((uint32_t)x->d) << 23);
        sumf += y[i] * scale.as_value * mxfp4_table[(x->qs[(i%MXFP4)/2] & 0xf)];
        sumf += y[i+1] * scale.as_value * mxfp4_table[(x->qs[(i%MXFP4)/2] & 0xf0) >> 4];
    }


#else // defined(GGML_SIMD)
    const int nb = n / MXFP4;
    assert(n % MXFP4 == 0);

    int yi = 0;

    const block_mxfp4 * GGML_RESTRICT xx = (const block_mxfp4 *) vx;

    ggml_float sumf = 0.0;
    for (int ib = 0; ib < nb; ++ib) {
        const block_mxfp4 * GGML_RESTRICT x = &xx[ib + 0];
        union {
            uint32_t as_bits;
            float as_value;
        } scale;
        scale.as_bits = (((uint32_t)x->d) << 23);
        for (int i = 0; i < MXFP4/2; ++i) {
            sumf += mxfp4_table[(x->qs[i] & 0xf)] * (ggml_float)(scale.as_value) * (ggml_float)(y[ib*MXFP4 + i*2]);
            sumf += mxfp4_table[(x->qs[i] & 0xf0) >> 4] * (ggml_float)(scale.as_value) * (ggml_float)(y[ib*MXFP4 + i*2+1]);
        }
    }
#endif

    *s = sumf;
}

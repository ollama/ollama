#pragma once

#include "ggml-cpu-impl.h"

//
// simd mappings
//

// we define a common set of C macros which map to specific intrinsics based on the current architecture
// we then implement the fundamental computation operations below using only these macros
// adding support for new architectures requires to define the corresponding SIMD macros
//
// GGML_F32_STEP / GGML_F16_STEP
//   number of elements to process in a single step
//
// GGML_F32_EPR / GGML_F16_EPR
//   number of elements to fit in a single register
//

#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)

#define GGML_SIMD

// F32 NEON

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              float32x4_t
#define GGML_F32x4_ZERO         vdupq_n_f32(0.0f)
#define GGML_F32x4_SET1(x)      vdupq_n_f32(x)
#define GGML_F32x4_LOAD         vld1q_f32
#define GGML_F32x4_STORE        vst1q_f32
#define GGML_F32x4_FMA(a, b, c) vfmaq_f32(a, b, c)
#define GGML_F32x4_ADD          vaddq_f32
#define GGML_F32x4_MUL          vmulq_f32
#define GGML_F32x4_REDUCE_ONE(x) vaddvq_f32(x)
#define GGML_F32x4_REDUCE(res, x)                       \
{                                                       \
    int offset = GGML_F32_ARR >> 1;                     \
    for (int i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    offset >>= 1;                                       \
    for (int i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    offset >>= 1;                                       \
    for (int i = 0; i < offset; ++i) {                  \
        (x)[i] = vaddq_f32((x)[i], (x)[offset+i]);      \
    }                                                   \
    (res) = (ggml_float) GGML_F32x4_REDUCE_ONE((x)[0]); \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    #define GGML_F16_STEP 32
    #define GGML_F16_EPR  8

    #define GGML_F16x8              float16x8_t
    #define GGML_F16x8_ZERO         vdupq_n_f16(0.0f)
    #define GGML_F16x8_SET1(x)      vdupq_n_f16(x)
    #define GGML_F16x8_LOAD(x)      vld1q_f16((const __fp16 *)(x))
    #define GGML_F16x8_STORE        vst1q_f16
    #define GGML_F16x8_FMA(a, b, c) vfmaq_f16(a, b, c)
    #define GGML_F16x8_ADD          vaddq_f16
    #define GGML_F16x8_MUL          vmulq_f16
    #define GGML_F16x8_REDUCE(res, x)                               \
    do {                                                            \
        int offset = GGML_F16_ARR >> 1;                             \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        offset >>= 1;                                               \
        for (int i = 0; i < offset; ++i) {                          \
            (x)[i] = vaddq_f16((x)[i], (x)[offset+i]);              \
        }                                                           \
        const float32x4_t t0 = vcvt_f32_f16(vget_low_f16 ((x)[0])); \
        const float32x4_t t1 = vcvt_f32_f16(vget_high_f16((x)[0])); \
        (res) = (ggml_float) vaddvq_f32(vaddq_f32(t0, t1));         \
    } while (0)

    #define GGML_F16_VEC                GGML_F16x8
    #define GGML_F16_VEC_ZERO           GGML_F16x8_ZERO
    #define GGML_F16_VEC_SET1           GGML_F16x8_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F16x8_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F16x8_STORE((__fp16 *)(p), (r)[i])
    #define GGML_F16_VEC_FMA            GGML_F16x8_FMA
    #define GGML_F16_VEC_ADD            GGML_F16x8_ADD
    #define GGML_F16_VEC_MUL            GGML_F16x8_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F16x8_REDUCE
#else
    // if FP16 vector arithmetic is not supported, we use FP32 instead
    // and take advantage of the vcvt_ functions to convert to/from FP16

    #define GGML_F16_STEP 16
    #define GGML_F16_EPR  4

    #define GGML_F32Cx4              float32x4_t
    #define GGML_F32Cx4_ZERO         vdupq_n_f32(0.0f)
    #define GGML_F32Cx4_SET1(x)      vdupq_n_f32(x)
    #define GGML_F32Cx4_LOAD(x)      vcvt_f32_f16(vld1_f16((const __fp16 *)(x)))
    #define GGML_F32Cx4_STORE(x, y)  vst1_f16(x, vcvt_f16_f32(y))
    #define GGML_F32Cx4_FMA(a, b, c) vfmaq_f32(a, b, c)
    #define GGML_F32Cx4_ADD          vaddq_f32
    #define GGML_F32Cx4_MUL          vmulq_f32
    #define GGML_F32Cx4_REDUCE       GGML_F32x4_REDUCE

    #define GGML_F16_VEC                GGML_F32Cx4
    #define GGML_F16_VEC_ZERO           GGML_F32Cx4_ZERO
    #define GGML_F16_VEC_SET1           GGML_F32Cx4_SET1
    #define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx4_LOAD(p)
    #define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx4_STORE((__fp16 *)(p), r[i])
    #define GGML_F16_VEC_FMA            GGML_F32Cx4_FMA
    #define GGML_F16_VEC_ADD            GGML_F32Cx4_ADD
    #define GGML_F16_VEC_MUL            GGML_F32Cx4_MUL
    #define GGML_F16_VEC_REDUCE         GGML_F32Cx4_REDUCE
#endif

#elif defined(__AVX512F__)

#define GGML_SIMD

// F32 AVX512

#define GGML_F32_STEP 64
#define GGML_F32_EPR  16

#define GGML_F32x16         __m512
#define GGML_F32x16_ZERO    _mm512_setzero_ps()
#define GGML_F32x16_SET1(x) _mm512_set1_ps(x)
#define GGML_F32x16_LOAD    _mm512_loadu_ps
#define GGML_F32x16_STORE   _mm512_storeu_ps
// _mm512_fmadd_ps is defined in AVX512F so no guard is required
#define GGML_F32x16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32x16_ADD     _mm512_add_ps
#define GGML_F32x16_MUL     _mm512_mul_ps
#define GGML_F32x16_REDUCE(res, x)                                    \
do {                                                                  \
    int offset = GGML_F32_ARR >> 1;                                   \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    offset >>= 1;                                                     \
    for (int i = 0; i < offset; ++i) {                                \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                      \
    }                                                                 \
    res = (ggml_float) _mm512_reduce_add_ps(x[0]);                    \
} while (0)

// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x16
#define GGML_F32_VEC_ZERO   GGML_F32x16_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x16_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x16_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x16_STORE
#define GGML_F32_VEC_FMA    GGML_F32x16_FMA
#define GGML_F32_VEC_ADD    GGML_F32x16_ADD
#define GGML_F32_VEC_MUL    GGML_F32x16_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x16_REDUCE

// F16 AVX512

// F16 AVX

#define GGML_F16_STEP 64
#define GGML_F16_EPR  16

// AVX512 has FP16 extension (AVX512_FP16) but I don't have it on my machine so I use FP32 instead

#define GGML_F32Cx16             __m512
#define GGML_F32Cx16_ZERO        _mm512_setzero_ps()
#define GGML_F32Cx16_SET1(x)     _mm512_set1_ps(x)

// unlike  _mm256_cvt intrinsics that require F16C, _mm512_cvt is defined in AVX512F
// so F16C guard isn't required
#define GGML_F32Cx16_LOAD(x)     _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(x)))
#define GGML_F32Cx16_STORE(x, y) _mm256_storeu_si256((__m256i *)(x), _mm512_cvtps_ph(y, 0))

#define GGML_F32Cx16_FMA(a, b, c) _mm512_fmadd_ps(b, c, a)
#define GGML_F32Cx16_ADD         _mm512_add_ps
#define GGML_F32Cx16_MUL         _mm512_mul_ps
#define GGML_F32Cx16_REDUCE(res, x)                               \
do {                                                              \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm512_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    res = (ggml_float) _mm512_reduce_add_ps(x[0]);                \
} while (0)

#define GGML_F16_VEC                GGML_F32Cx16
#define GGML_F16_VEC_ZERO           GGML_F32Cx16_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx16_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx16_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx16_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx16_FMA
#define GGML_F16_VEC_ADD            GGML_F32Cx16_ADD
#define GGML_F16_VEC_MUL            GGML_F32Cx16_MUL

#define GGML_F16_VEC_REDUCE         GGML_F32Cx16_REDUCE
#elif defined(__AVX__)

#define GGML_SIMD

// F32 AVX

#define GGML_F32_STEP 32
#define GGML_F32_EPR  8

#define GGML_F32x8         __m256
#define GGML_F32x8_ZERO    _mm256_setzero_ps()
#define GGML_F32x8_SET1(x) _mm256_set1_ps(x)
#define GGML_F32x8_LOAD    _mm256_loadu_ps
#define GGML_F32x8_STORE   _mm256_storeu_ps
#if defined(__FMA__)
    #define GGML_F32x8_FMA(a, b, c) _mm256_fmadd_ps(b, c, a)
#else
    #define GGML_F32x8_FMA(a, b, c) _mm256_add_ps(_mm256_mul_ps(b, c), a)
#endif
#define GGML_F32x8_ADD     _mm256_add_ps
#define GGML_F32x8_MUL     _mm256_mul_ps
#define GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm256_add_ps(x[i], x[offset+i]);                  \
    }                                                             \
    const __m128 t0 = _mm_add_ps(_mm256_castps256_ps128(x[0]),    \
                                 _mm256_extractf128_ps(x[0], 1)); \
    const __m128 t1 = _mm_hadd_ps(t0, t0);                        \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t1, t1));        \
} while (0)
// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x8
#define GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x8_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x8_STORE
#define GGML_F32_VEC_FMA    GGML_F32x8_FMA
#define GGML_F32_VEC_ADD    GGML_F32x8_ADD
#define GGML_F32_VEC_MUL    GGML_F32x8_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 AVX

#define GGML_F16_STEP 32
#define GGML_F16_EPR  8

// F16 arithmetic is not supported by AVX, so we use F32 instead

#define GGML_F32Cx8             __m256
#define GGML_F32Cx8_ZERO        _mm256_setzero_ps()
#define GGML_F32Cx8_SET1(x)     _mm256_set1_ps(x)

#if defined(__F16C__)
// the  _mm256_cvt intrinsics require F16C
#define GGML_F32Cx8_LOAD(x)     _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(x)))
#define GGML_F32Cx8_STORE(x, y) _mm_storeu_si128((__m128i *)(x), _mm256_cvtps_ph(y, 0))
#else
static inline __m256 __avx_f32cx8_load(const ggml_fp16_t * x) {
    float tmp[8];

    for (int i = 0; i < 8; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    return _mm256_loadu_ps(tmp);
}
static inline void __avx_f32cx8_store(ggml_fp16_t *x, __m256 y) {
    float arr[8];

    _mm256_storeu_ps(arr, y);

    for (int i = 0; i < 8; i++)
        x[i] = GGML_FP32_TO_FP16(arr[i]);
}
#define GGML_F32Cx8_LOAD(x)     __avx_f32cx8_load(x)
#define GGML_F32Cx8_STORE(x, y) __avx_f32cx8_store(x, y)
#endif

#define GGML_F32Cx8_FMA         GGML_F32x8_FMA
#define GGML_F32Cx8_ADD         _mm256_add_ps
#define GGML_F32Cx8_MUL         _mm256_mul_ps
#define GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#define GGML_F16_VEC                GGML_F32Cx8
#define GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#define GGML_F16_VEC_ADD            GGML_F32Cx8_ADD
#define GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#define GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#elif defined(__POWER9_VECTOR__)

#define GGML_SIMD

// F32 POWER9

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4              vector float
#define GGML_F32x4_ZERO         0.0f
#define GGML_F32x4_SET1         vec_splats
#define GGML_F32x4_LOAD(p)      vec_xl(0, p)
#define GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define GGML_F32x4_ADD          vec_add
#define GGML_F32x4_MUL          vec_mul
#define GGML_F32x4_REDUCE(res, x)              \
{                                              \
    int offset = GGML_F32_ARR >> 1;            \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    offset >>= 1;                              \
    for (int i = 0; i < offset; ++i) {         \
        x[i] = vec_add(x[i], x[offset+i]);     \
    }                                          \
    res = vec_extract(x[0], 0) +               \
          vec_extract(x[0], 1) +               \
          vec_extract(x[0], 2) +               \
          vec_extract(x[0], 3);                \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 POWER9
#define GGML_F16_STEP       GGML_F32_STEP
#define GGML_F16_EPR        GGML_F32_EPR
#define GGML_F16_VEC        GGML_F32x4
#define GGML_F16_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F16_VEC_SET1   GGML_F32x4_SET1
#define GGML_F16_VEC_FMA    GGML_F32x4_FMA
#define GGML_F16_VEC_ADD    GGML_F32x4_ADD
#define GGML_F16_VEC_MUL    GGML_F32x4_MUL
#define GGML_F16_VEC_REDUCE GGML_F32x4_REDUCE
// Use vec_xl, not vec_ld, in case the load address is not aligned.
#define GGML_F16_VEC_LOAD(p, i) (i & 0x1) ?                   \
  vec_extract_fp32_from_shorth(vec_xl(0, p - GGML_F16_EPR)) : \
  vec_extract_fp32_from_shortl(vec_xl(0, p))
static inline unsigned char ggml_endian_byte(int i) {
       uint16_t tmp_val = 1;
       return ((unsigned char *)&tmp_val)[i];
}
#define GGML_ENDIAN_BYTE(i) ggml_endian_byte(i)
#define GGML_F16_VEC_STORE(p, r, i)                             \
  if (i & 0x1)                                                  \
    vec_xst(vec_pack_to_short_fp32(r[i - GGML_ENDIAN_BYTE(1)],  \
                                   r[i - GGML_ENDIAN_BYTE(0)]), \
            0, p - GGML_F16_EPR)

#elif defined(__wasm_simd128__)

#define GGML_SIMD

// F32 WASM

#define GGML_F32_STEP 16
#define GGML_F32_EPR  4

#define GGML_F32x4              v128_t
#define GGML_F32x4_ZERO         wasm_f32x4_splat(0.0f)
#define GGML_F32x4_SET1(x)      wasm_f32x4_splat(x)
#define GGML_F32x4_LOAD         wasm_v128_load
#define GGML_F32x4_STORE        wasm_v128_store
#define GGML_F32x4_FMA(a, b, c) wasm_f32x4_add(wasm_f32x4_mul(b, c), a)
#define GGML_F32x4_ADD          wasm_f32x4_add
#define GGML_F32x4_MUL          wasm_f32x4_mul
#define GGML_F32x4_REDUCE(res, x)                  \
{                                                  \
    int offset = GGML_F32_ARR >> 1;                \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    offset >>= 1;                                  \
    for (int i = 0; i < offset; ++i) {             \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);  \
    }                                              \
    res = wasm_f32x4_extract_lane(x[0], 0) +       \
          wasm_f32x4_extract_lane(x[0], 1) +       \
          wasm_f32x4_extract_lane(x[0], 2) +       \
          wasm_f32x4_extract_lane(x[0], 3);        \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 WASM

#define GGML_F16_STEP 16
#define GGML_F16_EPR  4

inline static v128_t __wasm_f16x4_load(const ggml_fp16_t * p) {
    float tmp[4];

    tmp[0] = GGML_FP16_TO_FP32(p[0]);
    tmp[1] = GGML_FP16_TO_FP32(p[1]);
    tmp[2] = GGML_FP16_TO_FP32(p[2]);
    tmp[3] = GGML_FP16_TO_FP32(p[3]);

    return wasm_v128_load(tmp);
}

inline static void __wasm_f16x4_store(ggml_fp16_t * p, v128_t x) {
    float tmp[4];

    wasm_v128_store(tmp, x);

    p[0] = GGML_FP32_TO_FP16(tmp[0]);
    p[1] = GGML_FP32_TO_FP16(tmp[1]);
    p[2] = GGML_FP32_TO_FP16(tmp[2]);
    p[3] = GGML_FP32_TO_FP16(tmp[3]);
}

#define GGML_F16x4             v128_t
#define GGML_F16x4_ZERO        wasm_f32x4_splat(0.0f)
#define GGML_F16x4_SET1(x)     wasm_f32x4_splat(x)
#define GGML_F16x4_LOAD(x)     __wasm_f16x4_load(x)
#define GGML_F16x4_STORE(x, y) __wasm_f16x4_store(x, y)
#define GGML_F16x4_FMA         GGML_F32x4_FMA
#define GGML_F16x4_ADD         wasm_f32x4_add
#define GGML_F16x4_MUL         wasm_f32x4_mul
#define GGML_F16x4_REDUCE(res, x)                           \
{                                                           \
    int offset = GGML_F16_ARR >> 1;                         \
    for (int i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    offset >>= 1;                                           \
    for (int i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    offset >>= 1;                                           \
    for (int i = 0; i < offset; ++i) {                      \
        x[i] = wasm_f32x4_add(x[i], x[offset+i]);           \
    }                                                       \
    res = (ggml_float) (wasm_f32x4_extract_lane(x[0], 0) +  \
          wasm_f32x4_extract_lane(x[0], 1) +                \
          wasm_f32x4_extract_lane(x[0], 2) +                \
          wasm_f32x4_extract_lane(x[0], 3));                \
}

#define GGML_F16_VEC                GGML_F16x4
#define GGML_F16_VEC_ZERO           GGML_F16x4_ZERO
#define GGML_F16_VEC_SET1           GGML_F16x4_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F16x4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F16x4_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F16x4_FMA
#define GGML_F16_VEC_ADD            GGML_F16x4_ADD
#define GGML_F16_VEC_MUL            GGML_F16x4_MUL
#define GGML_F16_VEC_REDUCE         GGML_F16x4_REDUCE

#elif defined(__SSE3__)

#define GGML_SIMD

// F32 SSE

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4         __m128
#define GGML_F32x4_ZERO    _mm_setzero_ps()
#define GGML_F32x4_SET1(x) _mm_set1_ps(x)
#define GGML_F32x4_LOAD    _mm_loadu_ps
#define GGML_F32x4_STORE   _mm_storeu_ps
#if defined(__FMA__)
    // TODO: Does this work?
    #define GGML_F32x4_FMA(a, b, c) _mm_fmadd_ps(b, c, a)
#else
    #define GGML_F32x4_FMA(a, b, c) _mm_add_ps(_mm_mul_ps(b, c), a)
#endif
#define GGML_F32x4_ADD     _mm_add_ps
#define GGML_F32x4_MUL     _mm_mul_ps
#define GGML_F32x4_REDUCE(res, x)                                 \
{                                                                 \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = _mm_add_ps(x[i], x[offset+i]);                     \
    }                                                             \
    const __m128 t0 = _mm_hadd_ps(x[0], x[0]);                    \
    res = (ggml_float) _mm_cvtss_f32(_mm_hadd_ps(t0, t0));        \
}
// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 SSE

#define GGML_F16_STEP 32
#define GGML_F16_EPR  4

static inline __m128 __sse_f16x4_load(const ggml_fp16_t * x) {
    float tmp[4];

    tmp[0] = GGML_FP16_TO_FP32(x[0]);
    tmp[1] = GGML_FP16_TO_FP32(x[1]);
    tmp[2] = GGML_FP16_TO_FP32(x[2]);
    tmp[3] = GGML_FP16_TO_FP32(x[3]);

    return _mm_loadu_ps(tmp);
}

static inline void __sse_f16x4_store(ggml_fp16_t * x, __m128 y) {
    float arr[4];

    _mm_storeu_ps(arr, y);

    x[0] = GGML_FP32_TO_FP16(arr[0]);
    x[1] = GGML_FP32_TO_FP16(arr[1]);
    x[2] = GGML_FP32_TO_FP16(arr[2]);
    x[3] = GGML_FP32_TO_FP16(arr[3]);
}

#define GGML_F32Cx4             __m128
#define GGML_F32Cx4_ZERO        _mm_setzero_ps()
#define GGML_F32Cx4_SET1(x)     _mm_set1_ps(x)
#define GGML_F32Cx4_LOAD(x)     __sse_f16x4_load(x)
#define GGML_F32Cx4_STORE(x, y) __sse_f16x4_store(x, y)
#define GGML_F32Cx4_FMA         GGML_F32x4_FMA
#define GGML_F32Cx4_ADD         _mm_add_ps
#define GGML_F32Cx4_MUL         _mm_mul_ps
#define GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#define GGML_F16_VEC                 GGML_F32Cx4
#define GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#define GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#define GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#define GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#define GGML_F16_VEC_ADD             GGML_F32Cx4_ADD
#define GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#define GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#elif defined(__loongarch_asx)

#define GGML_SIMD

// F32 LASX
#define GGML_F32_STEP 32
#define GGML_F32_EPR  8

#define GGML_F32x8         __m256
#define GGML_F32x8_ZERO    (__m256)__lasx_xvldi(0)
#define GGML_F32x8_SET1(x) (__m256)__lasx_xvreplfr2vr_s((x))
#define GGML_F32x8_LOAD(x) (__m256)__lasx_xvld((x), 0)
#define GGML_F32x8_STORE(x,y)   __lasx_xvst((y), (x), 0)
#define GGML_F32x8_FMA(a, b, c) __lasx_xvfmadd_s(b, c, a)
#define GGML_F32x8_ADD     __lasx_xvfadd_s
#define GGML_F32x8_MUL     __lasx_xvfmul_s
#define GGML_F32x8_REDUCE(res, x)                                 \
do {                                                              \
    int offset = GGML_F32_ARR >> 1;                               \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    offset >>= 1;                                                 \
    for (int i = 0; i < offset; ++i) {                            \
        x[i] = __lasx_xvfadd_s(x[i], x[offset+i]);                  \
    }                                                             \
    float *tmp_p = (float *)&x[0]; \
    res = tmp_p[0] + tmp_p[1] + tmp_p[2] + tmp_p[3] + tmp_p[4] + tmp_p[5] + tmp_p[6] + tmp_p[7];  \
} while (0)
// TODO: is this optimal ?

#define GGML_F32_VEC        GGML_F32x8
#define GGML_F32_VEC_ZERO   GGML_F32x8_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x8_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x8_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x8_STORE
#define GGML_F32_VEC_FMA    GGML_F32x8_FMA
#define GGML_F32_VEC_ADD    GGML_F32x8_ADD
#define GGML_F32_VEC_MUL    GGML_F32x8_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x8_REDUCE

// F16 LASX

#define GGML_F16_STEP 32
#define GGML_F16_EPR  8

// F16 arithmetic is not supported by LASX, so we use F32 instead

#define GGML_F32Cx8          __m256
#define GGML_F32Cx8_ZERO    (__m256)__lasx_xvldi(0)
#define GGML_F32Cx8_SET1(x) (__m256)__lasx_xvreplgr2vr_w((x))

static inline __m256 __lasx_f32cx8_load(const ggml_fp16_t * x) {
    __m256i a;
    memcpy(&a, x, sizeof(ggml_fp16_t) * 8);
    a = __lasx_xvpermi_d(a, 0 | (1 << 4));
    return __lasx_xvfcvtl_s_h(a);
}

static inline void __lasx_f32cx8_store(ggml_fp16_t * x, __m256 y) {
    __m256i a = __lasx_xvfcvt_h_s(y, y);
    a = __lasx_xvpermi_d(a, 0 | (2 << 2));
    memcpy(x, &a, sizeof(ggml_fp16_t) * 8);
}
#define GGML_F32Cx8_LOAD(x)     __lasx_f32cx8_load(x)
#define GGML_F32Cx8_STORE(x, y) __lasx_f32cx8_store(x, y)

#define GGML_F32Cx8_FMA         GGML_F32x8_FMA
#define GGML_F32Cx8_ADD         __lasx_xvfadd_s
#define GGML_F32Cx8_MUL         __lasx_xvfmul_s
#define GGML_F32Cx8_REDUCE      GGML_F32x8_REDUCE

#define GGML_F16_VEC                GGML_F32Cx8
#define GGML_F16_VEC_ZERO           GGML_F32Cx8_ZERO
#define GGML_F16_VEC_SET1           GGML_F32Cx8_SET1
#define GGML_F16_VEC_LOAD(p, i)     GGML_F32Cx8_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i) GGML_F32Cx8_STORE(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32Cx8_FMA
#define GGML_F16_VEC_ADD            GGML_F32Cx8_ADD
#define GGML_F16_VEC_MUL            GGML_F32Cx8_MUL
#define GGML_F16_VEC_REDUCE         GGML_F32Cx8_REDUCE

#elif defined(__loongarch_sx)

#define GGML_SIMD

// F32 LSX

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4         __m128
#define GGML_F32x4_ZERO    __lsx_vldi(0)
#define GGML_F32x4_SET1(x) __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define GGML_F32x4_LOAD(x) __lsx_vld((x), 0)
#define GGML_F32x4_STORE((x),(y))   __lsx_vst((y), (x), 0)
#define GGML_F32x4_FMA(a, b, c) __lsx_vfmadd_s(b, c, a)
#define GGML_F32x4_ADD     __lsx_vfadd_s
#define GGML_F32x4_MUL     __lsx_vfmul_s
#define GGML_F32x4_REDUCE(res, x)                                                     \
{                                                                                     \
    int offset = GGML_F32_ARR >> 1;                                                   \
    for (int i = 0; i < offset; ++i) {                                                \
        x[i] = __lsx_vfadd_s(x[i], x[offset + i]);                                    \
    }                                                                                 \
    offset >>= 1;                                                                     \
    for (int i = 0; i < offset; ++i) {                                                \
        x[i] = __lsx_vfadd_s(x[i], x[offset + i]);                                    \
    }                                                                                 \
    offset >>= 1;                                                                     \
    for (int i = 0; i < offset; ++i) {                                                \
        x[i] = __lsx_vfadd_s(x[i], x[offset + i]);                                    \
    }                                                                                 \
    __m128i tmp     = __lsx_vsrli_d((__m128i) x[0], 32);                              \
    tmp             = (__m128i) __lsx_vfadd_s((__m128) tmp, x[0]);                    \
    tmp             = __lsx_vpickev_w(__lsx_vldi(0), tmp);                            \
    const __m128 t0 = __lsx_vshuf4i_w(tmp, 0x88);                                     \
    tmp             = __lsx_vsrli_d((__m128i) t0, 32);                                \
    tmp             = (__m128i) __lsx_vfadd_s((__m128) tmp, t0);                      \
    tmp             = __lsx_vpickev_w(__lsx_vldi(0), tmp);                            \
    res             = (ggml_float) __lsx_vpickve2gr_w(__lsx_vshuf4i_w(tmp, 0x88), 0); \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 LSX

#define GGML_F16_STEP 32
#define GGML_F16_EPR  4

static inline __m128 __lsx_f16x4_load(const ggml_fp16_t * x) {
    float tmp[4];

    tmp[0] = GGML_FP16_TO_FP32(x[0]);
    tmp[1] = GGML_FP16_TO_FP32(x[1]);
    tmp[2] = GGML_FP16_TO_FP32(x[2]);
    tmp[3] = GGML_FP16_TO_FP32(x[3]);

    return __lsx_vld(tmp, 0);
}

static inline void __lsx_f16x4_store(ggml_fp16_t * x, __m128 y) {
    float arr[4];

    __lsx_vst(y, arr, 0);

    x[0] = GGML_FP32_TO_FP16(arr[0]);
    x[1] = GGML_FP32_TO_FP16(arr[1]);
    x[2] = GGML_FP32_TO_FP16(arr[2]);
    x[3] = GGML_FP32_TO_FP16(arr[3]);
}

#define GGML_F32Cx4             __m128
#define GGML_F32Cx4_ZERO        __lsx_vldi(0)
#define GGML_F32Cx4_SET1(x)     __lsx_vinsgr2vr_w(__lsx_vldi(0),(x), 0)
#define GGML_F32Cx4_LOAD(x)     __lsx_f16x4_load(x)
#define GGML_F32Cx4_STORE(x, y) __lsx_f16x4_store(x, y)
#define GGML_F32Cx4_FMA         GGML_F32x4_FMA
#define GGML_F32Cx4_ADD         __lsx_vfadd_s
#define GGML_F32Cx4_MUL         __lsx_vfmul_s
#define GGML_F32Cx4_REDUCE      GGML_F32x4_REDUCE

#define GGML_F16_VEC                 GGML_F32Cx4
#define GGML_F16_VEC_ZERO            GGML_F32Cx4_ZERO
#define GGML_F16_VEC_SET1            GGML_F32Cx4_SET1
#define GGML_F16_VEC_LOAD(p, i)      GGML_F32Cx4_LOAD(p)
#define GGML_F16_VEC_STORE(p, r, i)  GGML_F32Cx4_STORE(p, r[i])
#define GGML_F16_VEC_FMA             GGML_F32Cx4_FMA
#define GGML_F16_VEC_ADD             GGML_F32Cx4_ADD
#define GGML_F16_VEC_MUL             GGML_F32Cx4_MUL
#define GGML_F16_VEC_REDUCE          GGML_F32Cx4_REDUCE

#elif defined(__VXE__) || defined(__VXE2__)

#define GGML_SIMD

// F32 s390x

#define GGML_F32_STEP 32
#define GGML_F32_EPR  4

#define GGML_F32x4              __vector float
#define GGML_F32x4_ZERO         vec_splats(0.0f)
#define GGML_F32x4_SET1         vec_splats
#define GGML_F32x4_LOAD(p)      vec_xl(0, p)
#define GGML_F32x4_STORE(p, r)  vec_xst(r, 0, p)
#define GGML_F32x4_FMA(a, b, c) vec_madd(b, c, a)
#define GGML_F32x4_ADD          vec_add
#define GGML_F32x4_MUL          vec_mul
#define GGML_F32x4_REDUCE(res, x)                   \
{                                                   \
    int offset = GGML_F32_ARR >> 1;                 \
    for (int i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    offset >>= 1;                                   \
    for (int i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    offset >>= 1;                                   \
    for (int i = 0; i < offset; ++i) {              \
        x[i] = vec_add(x[i], x[offset + i]);        \
    }                                               \
    res = vec_extract(x[0], 0) +                    \
          vec_extract(x[0], 1) +                    \
          vec_extract(x[0], 2) +                    \
          vec_extract(x[0], 3);                     \
}

#define GGML_F32_VEC        GGML_F32x4
#define GGML_F32_VEC_ZERO   GGML_F32x4_ZERO
#define GGML_F32_VEC_SET1   GGML_F32x4_SET1
#define GGML_F32_VEC_LOAD   GGML_F32x4_LOAD
#define GGML_F32_VEC_STORE  GGML_F32x4_STORE
#define GGML_F32_VEC_FMA    GGML_F32x4_FMA
#define GGML_F32_VEC_ADD    GGML_F32x4_ADD
#define GGML_F32_VEC_MUL    GGML_F32x4_MUL
#define GGML_F32_VEC_REDUCE GGML_F32x4_REDUCE

// F16 s390x
#define GGML_F16_STEP GGML_F32_STEP
#define GGML_F16_EPR  GGML_F32_EPR

static inline __vector float __lzs_f16cx4_load(const ggml_fp16_t * x) {
    float tmp[4];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(x[i]);
    }

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    return vec_xl(0, (const float *)(tmp));
}

static inline void __lzs_f16cx4_store(ggml_fp16_t * x, __vector float y) {
    float arr[4];

    // note: keep type-cast here to prevent compiler bugs
    // see: https://github.com/ggml-org/llama.cpp/issues/12846
    vec_xst(y, 0, (float *)(arr));

    for (int i = 0; i < 4; i++) {
        x[i] = GGML_FP32_TO_FP16(arr[i]);
    }
}

#define GGML_F16_VEC                GGML_F32x4
#define GGML_F16_VEC_ZERO           GGML_F32x4_ZERO
#define GGML_F16_VEC_SET1           GGML_F32x4_SET1
#define GGML_F16_VEC_LOAD(p, i)     __lzs_f16cx4_load(p)
#define GGML_F16_VEC_STORE(p, r, i) __lzs_f16cx4_store(p, r[i])
#define GGML_F16_VEC_FMA            GGML_F32x4_FMA
#define GGML_F16_VEC_ADD            GGML_F32x4_ADD
#define GGML_F16_VEC_MUL            GGML_F32x4_MUL
#define GGML_F16_VEC_REDUCE         GGML_F32x4_REDUCE

#endif

// GGML_F32_ARR / GGML_F16_ARR
//   number of registers to use per step
#ifdef GGML_SIMD
#define GGML_F32_ARR (GGML_F32_STEP/GGML_F32_EPR)
#define GGML_F16_ARR (GGML_F16_STEP/GGML_F16_EPR)
#endif

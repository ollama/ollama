#pragma once

// GGML CPU internal header

#include "ggml.h"
#include "ggml-impl.h"

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};


#if defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#else

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
#if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
#ifndef __FMA__
#define __FMA__
#endif
#ifndef __F16C__
#define __F16C__
#endif
#endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
#if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
#ifndef __SSE3__
#define __SSE3__
#endif
#ifndef __SSSE3__
#define __SSSE3__
#endif
#endif

#if defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif  // __VXE__
#ifndef __VXE2__
#define __VXE2__
#endif  // __VXE2__
#endif  // __s390x__ && __VEC__

#if defined(__ARM_FEATURE_SVE) && defined(__linux__)
#include <sys/prctl.h>
#endif

#if defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ifdef _MSC_VER
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#else
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#if !defined(__aarch64__)

// 32-bit ARM compatibility

// vaddlvq_s16
// vpaddq_s16
// vpaddq_s32
// vaddvq_s32
// vaddvq_f32
// vmaxvq_f32
// vcvtnq_s32_f32
// vzip1_u8
// vzip2_u8

inline static int32_t vaddlvq_s16(int16x8_t v) {
    int32x4_t v0 = vreinterpretq_s32_s64(vpaddlq_s32(vpaddlq_s16(v)));
    return vgetq_lane_s32(v0, 0) + vgetq_lane_s32(v0, 2);
}

inline static int16x8_t vpaddq_s16(int16x8_t a, int16x8_t b) {
    int16x4_t a0 = vpadd_s16(vget_low_s16(a), vget_high_s16(a));
    int16x4_t b0 = vpadd_s16(vget_low_s16(b), vget_high_s16(b));
    return vcombine_s16(a0, b0);
}

inline static int32x4_t vpaddq_s32(int32x4_t a, int32x4_t b) {
    int32x2_t a0 = vpadd_s32(vget_low_s32(a), vget_high_s32(a));
    int32x2_t b0 = vpadd_s32(vget_low_s32(b), vget_high_s32(b));
    return vcombine_s32(a0, b0);
}

inline static int32_t vaddvq_s32(int32x4_t v) {
    return vgetq_lane_s32(v, 0) + vgetq_lane_s32(v, 1) + vgetq_lane_s32(v, 2) + vgetq_lane_s32(v, 3);
}

inline static float vaddvq_f32(float32x4_t v) {
    return vgetq_lane_f32(v, 0) + vgetq_lane_f32(v, 1) + vgetq_lane_f32(v, 2) + vgetq_lane_f32(v, 3);
}

inline static float vmaxvq_f32(float32x4_t v) {
    return
        MAX(MAX(vgetq_lane_f32(v, 0), vgetq_lane_f32(v, 1)),
            MAX(vgetq_lane_f32(v, 2), vgetq_lane_f32(v, 3)));
}

inline static int32x4_t vcvtnq_s32_f32(float32x4_t v) {
    int32x4_t res;

    res[0] = roundf(vgetq_lane_f32(v, 0));
    res[1] = roundf(vgetq_lane_f32(v, 1));
    res[2] = roundf(vgetq_lane_f32(v, 2));
    res[3] = roundf(vgetq_lane_f32(v, 3));

    return res;
}

inline static uint8x8_t vzip1_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[0]; res[1] = b[0];
    res[2] = a[1]; res[3] = b[1];
    res[4] = a[2]; res[5] = b[2];
    res[6] = a[3]; res[7] = b[3];

    return res;
}

inline static uint8x8_t vzip2_u8(uint8x8_t a, uint8x8_t b) {
    uint8x8_t res;

    res[0] = a[4]; res[1] = b[4];
    res[2] = a[5]; res[3] = b[5];
    res[4] = a[6]; res[5] = b[6];
    res[6] = a[7]; res[7] = b[7];

    return res;
}

// vld1q_s16_x2
// vld1q_u8_x2
// vld1q_u8_x4
// vld1q_s8_x2
// vld1q_s8_x4
// TODO: double-check these work correctly

typedef struct ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vld1q_s16_x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vld1q_s16(ptr + 0);
    res.val[1] = vld1q_s16(ptr + 8);

    return res;
}

typedef struct ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vld1q_u8_x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);

    return res;
}

typedef struct ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vld1q_u8_x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vld1q_u8(ptr + 0);
    res.val[1] = vld1q_u8(ptr + 16);
    res.val[2] = vld1q_u8(ptr + 32);
    res.val[3] = vld1q_u8(ptr + 48);

    return res;
}

typedef struct ggml_int8x16x2_t {
    int8x16_t val[2];
} ggml_int8x16x2_t;

inline static ggml_int8x16x2_t ggml_vld1q_s8_x2(const int8_t * ptr) {
    ggml_int8x16x2_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);

    return res;
}

typedef struct ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vld1q_s8_x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vld1q_s8(ptr + 0);
    res.val[1] = vld1q_s8(ptr + 16);
    res.val[2] = vld1q_s8(ptr + 32);
    res.val[3] = vld1q_s8(ptr + 48);

    return res;
}

// NOTE: not tested
inline static int8x16_t ggml_vqtbl1q_s8(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

// NOTE: not tested
inline static uint8x16_t ggml_vqtbl1q_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

#else

#define ggml_int16x8x2_t  int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8

#endif // !defined(__aarch64__)

#if !defined(__ARM_FEATURE_DOTPROD)

inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

#endif // defined(__ARM_NEON)

#ifdef __wasm_simd128__
#include <wasm_simd128.h>
#endif

#ifdef __POWER9_VECTOR__
#include <altivec.h>
#endif

#if defined(_MSC_VER) || defined(__MINGW32__)
#include <intrin.h>
#elif defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__) || defined(__SSE3__) || defined(__SSE__)
#include <immintrin.h>
#endif

#ifdef __riscv_v_intrinsic
#include <riscv_vector.h>
#endif

#if defined(__loongarch64)
#if defined(__loongarch_asx)
#include <lasxintrin.h>
#endif
#if defined(__loongarch_sx)
#include <lsxintrin.h>
#endif
#endif

#if defined(__VXE__) || defined(__VXE2__)
#include <vecintrin.h>

#define vec_neg(a)    (-(a))                // Vector Negate
#define vec_add(a, b) ((a) + (b))           // Vector Add
#define vec_sub(a, b) ((a) - (b))           // Vector Subtract
#define vec_mul(a, b) ((a) * (b))           // Vector Multiply
#define vec_div(a, b) ((a) / (b))           // Vector Divide
#define vec_sl(a, b)  ((a) << (b))          // Vector Shift Left
#define vec_sra(a, b) ((a) >> (b))          // Vector Shift Right
#define vec_sr(a, b)  ((a) >> (b))          // Vector Shift Right Algebraic
#define vec_slo(a, b) vec_slb(a, (b) << 64) // Vector Shift Left by Octet
#define vec_sro(a, b) vec_srb(a, (b) << 64) // Vector Shift Right by Octet

#ifndef vec_and
#define vec_and(a, b) ((a) & (b)) // Vector AND
#endif

#ifndef vec_or
#define vec_or(a, b)  ((a) | (b)) // Vector OR
#endif

#ifndef vec_xor
#define vec_xor(a, b) ((a) ^ (b)) // Vector XOR
#endif

typedef signed   char char8x16_t  __attribute__((vector_size(16)));
typedef unsigned char uchar8x16_t __attribute__((vector_size(16)));

typedef int8_t  int8x16_t __attribute__((vector_size(16)));
typedef int16_t int16x8_t __attribute__((vector_size(16)));
typedef int32_t int32x4_t __attribute__((vector_size(16)));

typedef uint8_t  uint8x16_t __attribute__((vector_size(16)));
typedef uint16_t uint16x8_t __attribute__((vector_size(16)));
typedef uint32_t uint32x4_t __attribute__((vector_size(16)));

typedef float  float32x4_t  __attribute__((vector_size(16)));
typedef double double64x2_t __attribute__((vector_size(16)));

typedef signed   long long long64x2_t  __attribute__((vector_size(16)));
typedef unsigned long long ulong64x2_t __attribute__((vector_size(16)));

typedef struct ggml_uint8x16x2_t {
    uint8x16_t val[2];
} ggml_uint8x16x2_t;

inline static ggml_uint8x16x2_t ggml_vec_xl_u8x2(const uint8_t * ptr) {
    ggml_uint8x16x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;
}

typedef struct ggml_uint8x16x4_t {
    uint8x16_t val[4];
} ggml_uint8x16x4_t;

inline static ggml_uint8x16x4_t ggml_vec_xl_u8x4(const uint8_t * ptr) {
    ggml_uint8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;
}

typedef struct ggml_int8x16x4_t {
    int8x16_t val[4];
} ggml_int8x16x4_t;

inline static ggml_int8x16x4_t ggml_vec_xl_s8x4(const int8_t * ptr) {
    ggml_int8x16x4_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);
    res.val[2] = vec_xl(32, ptr);
    res.val[3] = vec_xl(48, ptr);

    return res;
}

typedef struct ggml_int16x8x2_t {
    int16x8_t val[2];
} ggml_int16x8x2_t;

inline static ggml_int16x8x2_t ggml_vec_xl_s16x2(const int16_t * ptr) {
    ggml_int16x8x2_t res;

    res.val[0] = vec_xl( 0, ptr);
    res.val[1] = vec_xl(16, ptr);

    return res;
}

/*
    ! WARNING: Very slow. Use vec_perm if possible. Refer to iq4_xs
    !          or iq4_nl for example implementation.
*/
inline static int8x16_t ggml_vec_tbl(int8x16_t a, uint8x16_t b) {
    int8x16_t res;

    res[ 0] = a[b[ 0]];
    res[ 1] = a[b[ 1]];
    res[ 2] = a[b[ 2]];
    res[ 3] = a[b[ 3]];
    res[ 4] = a[b[ 4]];
    res[ 5] = a[b[ 5]];
    res[ 6] = a[b[ 6]];
    res[ 7] = a[b[ 7]];
    res[ 8] = a[b[ 8]];
    res[ 9] = a[b[ 9]];
    res[10] = a[b[10]];
    res[11] = a[b[11]];
    res[12] = a[b[12]];
    res[13] = a[b[13]];
    res[14] = a[b[14]];
    res[15] = a[b[15]];

    return res;
}

inline static int16x8_t vec_padd_s16(int16x8_t a, int16x8_t b) {
    const uchar8x16_t v_maske = {  0,  1,  4,  5,  8,  9, 12, 13,
                                  16, 17, 20, 21, 24, 25, 28, 29 };

    const int16x8_t v_abo = vec_pack((int32x4_t)a, (int32x4_t)b);
    const int16x8_t v_abe = vec_perm(a, b, v_maske);
    return v_abo + v_abe;
}

/**
 * @see https://github.com/ggml-org/llama.cpp/pull/14037
 */
inline static float vec_hsum_f32x4(float32x4_t v) {
    float32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];
}

inline static int32_t vec_hsum_i32x4(int32x4_t v) {
    int32x4_t v_temp = v + vec_reve(v);
    return v_temp[0] + v_temp[1];
}

inline static int32x4_t ggml_vec_dot(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p = vec_mule(a, b) + vec_mulo(a, b);
    return acc + (vec_unpackh(p) + vec_unpackl(p));
}

#endif

#if defined(__loongarch_asx)
/* float type data load instructions */
static __m128 __lsx_vreplfr2vr_s(const float val) {
    v4f32 res = {val, val, val, val};
    return (__m128)res;
}

static __m256 __lasx_xvreplfr2vr_s(const float val) {
    v8f32 res = {val, val, val, val, val, val, val, val};
    return (__m256)res;
}
#endif

// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);

#ifdef __cplusplus
}
#endif

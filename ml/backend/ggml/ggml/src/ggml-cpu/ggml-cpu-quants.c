#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-cpu-quants.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-cpu.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>
#include <stdlib.h> // for qsort
#include <stdio.h>  // for GGML_ASSERT

#define GROUP_MAX_EPS 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS 1e-8f
#define GROUP_MAX_EPS_IQ2_S 1e-8f
#define GROUP_MAX_EPS_IQ1_M 1e-7f
#define GROUP_MAX_EPS_IQ1_S 1e-12f

#if defined(_MSC_VER)
// disable "possible loss of data" to avoid warnings for hundreds of casts
// we should just be careful :)
#pragma warning(disable: 4244 4267)
#endif

#define UNUSED GGML_UNUSED

// some compilers don't provide _mm256_set_m128i, e.g. gcc 7
#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = _mm_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m128i sy = _mm_sign_epi8(y, x);
    // Perform multiplication and create 16-bit values
    const __m128i dot = _mm_maddubs_epi16(ax, sy);
    const __m128i ones = _mm_set1_epi16(1);
    return _mm_madd_epi16(ones, dot);
}

#if __AVX__ || __AVX2__ || __AVX512F__
// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    const __m128i hi64 = _mm_unpackhi_epi64(a, a);
    const __m128i sum64 = _mm_add_epi32(hi64, a);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

#if defined(__AVX2__) || defined(__AVX512F__)
// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8( 0xF );
    return _mm256_and_si256(lowMask, bytes);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Perform multiplication and create 16-bit values
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
#if __AVXVNNIINT8__
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbssd_epi32(zero, x, y);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    // Get absolute values of x vectors
    const __m256i ax = _mm256_sign_epi8(x, x);
    // Sign the values of the y vectors
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
#endif
}

static inline __m128i packNibbles( __m256i bytes )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
#if __AVX512F__
    const __m256i bytes_srli_4 = _mm256_srli_epi16(bytes, 4);   // 0000_0000_abcd_0000
    bytes = _mm256_or_si256(bytes, bytes_srli_4);               // 0000_abcd_abcd_efgh
    return _mm256_cvtepi16_epi8(bytes);                         // abcd_efgh
#else
    const __m256i lowByte = _mm256_set1_epi16( 0xFF );
    __m256i high = _mm256_andnot_si256( lowByte, bytes );
    __m256i low = _mm256_and_si256( lowByte, bytes );
    high = _mm256_srli_epi16( high, 4 );
    bytes = _mm256_or_si256( low, high );

    // Compress uint16_t lanes into bytes
    __m128i r0 = _mm256_castsi256_si128( bytes );
    __m128i r1 = _mm256_extracti128_si256( bytes, 1 );
    return _mm_packus_epi16( r0, r1 );
#endif
}
#elif defined(__AVX__)
static inline __m128i packNibbles( __m128i bytes1, __m128i bytes2 )
{
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m128i lowByte = _mm_set1_epi16( 0xFF );
    __m128i high = _mm_andnot_si128( lowByte, bytes1 );
    __m128i low = _mm_and_si128( lowByte, bytes1 );
    high = _mm_srli_epi16( high, 4 );
    bytes1 = _mm_or_si128( low, high );
    high = _mm_andnot_si128( lowByte, bytes2 );
    low = _mm_and_si128( lowByte, bytes2 );
    high = _mm_srli_epi16( high, 4 );
    bytes2 = _mm_or_si128( low, high );

    return _mm_packus_epi16( bytes1, bytes2);
}

static inline __m128i mul_add_epi8_sse(const __m128i x, const __m128i y) {
    const __m128i ax = _mm_sign_epi8(x, x);
    const __m128i sy = _mm_sign_epi8(y, x);
    return _mm_maddubs_epi16(ax, sy);
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m128i shuf_maskl = _mm_set_epi64x(0x0101010101010101, 0x0000000000000000);
    const __m128i shuf_maskh = _mm_set_epi64x(0x0303030303030303, 0x0202020202020202);
    __m128i bytesl = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskl);
    __m128i bytesh = _mm_shuffle_epi8(_mm_set1_epi32(x32), shuf_maskh);
    const __m128i bit_mask = _mm_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytesl = _mm_or_si128(bytesl, bit_mask);
    bytesh = _mm_or_si128(bytesh, bit_mask);
    bytesl = _mm_cmpeq_epi8(bytesl, _mm_set1_epi64x(-1));
    bytesh = _mm_cmpeq_epi8(bytesh, _mm_set1_epi64x(-1));
    return MM256_SET_M128I(bytesh, bytesl);
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi)
{
    // Load 16 bytes from memory
    __m128i tmpl = _mm_loadu_si128((const __m128i *)rsi);
    __m128i tmph = _mm_srli_epi16(tmpl, 4);
    const __m128i lowMask = _mm_set1_epi8(0xF);
    tmpl = _mm_and_si128(lowMask, tmpl);
    tmph = _mm_and_si128(lowMask, tmph);
    return MM256_SET_M128I(tmph, tmpl);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m128i xh, const __m128i xl) {
    const __m128i ones = _mm_set1_epi16(1);
    const __m128i summed_pairsl = _mm_madd_epi16(ones, xl);
    const __m128i summed_pairsh = _mm_madd_epi16(ones, xh);
    const __m256i summed_pairs = MM256_SET_M128I(summed_pairsh, summed_pairsl);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m128i axl = _mm256_castsi256_si128(ax);
    const __m128i axh = _mm256_extractf128_si256(ax, 1);
    const __m128i syl = _mm256_castsi256_si128(sy);
    const __m128i syh = _mm256_extractf128_si256(sy, 1);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m128i xl = _mm256_castsi256_si128(x);
    const __m128i xh = _mm256_extractf128_si256(x, 1);
    const __m128i yl = _mm256_castsi256_si128(y);
    const __m128i yh = _mm256_extractf128_si256(y, 1);
    // Get absolute values of x vectors
    const __m128i axl = _mm_sign_epi8(xl, xl);
    const __m128i axh = _mm_sign_epi8(xh, xh);
    // Sign the values of the y vectors
    const __m128i syl = _mm_sign_epi8(yl, xl);
    const __m128i syh = _mm_sign_epi8(yh, xh);
    // Perform multiplication and create 16-bit values
    const __m128i dotl = _mm_maddubs_epi16(axl, syl);
    const __m128i doth = _mm_maddubs_epi16(axh, syh);
    return sum_i16_pairs_float(doth, dotl);
}

// larger version of mul_sum_i8_pairs_float where x and y are each represented by four 128-bit vectors
static inline __m256 mul_sum_i8_quad_float(const __m128i x_1_0, const __m128i x_1_1, const __m128i x_2_0, const __m128i x_2_1,
                                           const __m128i y_1_0, const __m128i y_1_1, const __m128i y_2_0, const __m128i y_2_1) {
    const __m128i mone = _mm_set1_epi16(1);

    const __m128i p16_1_0 = mul_add_epi8_sse(x_1_0, y_1_0);
    const __m128i p16_1_1 = mul_add_epi8_sse(x_1_1, y_1_1);
    const __m128i p16_2_0 = mul_add_epi8_sse(x_2_0, y_2_0);
    const __m128i p16_2_1 = mul_add_epi8_sse(x_2_1, y_2_1);
    const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, mone);
    const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, mone);
    const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, mone);
    const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, mone);
    const __m128i p_1 = _mm_add_epi32(p_1_0, p_1_1);
    const __m128i p_2 = _mm_add_epi32(p_2_0, p_2_1);
    return _mm256_cvtepi32_ps(MM256_SET_M128I(p_2, p_1));
}

// quad fp16 delta calculation
static inline __m256 quad_fp16_delta_float(const float x0, const float y0, const float x1, const float y1) {
    // GGML_FP16_TO_FP32 is faster than Intel F16C
    return _mm256_set_m128(_mm_set1_ps(GGML_FP16_TO_FP32(x1) * GGML_FP16_TO_FP32(y1)),
                           _mm_set1_ps(GGML_FP16_TO_FP32(x0) * GGML_FP16_TO_FP32(y0)));
}
#endif
#elif defined(__SSSE3__)
// horizontally add 4x4 floats
static inline float hsum_float_4x4(const __m128 a, const __m128 b, const __m128 c, const __m128 d) {
    __m128 res_0 =_mm_hadd_ps(a, b);
    __m128 res_1 =_mm_hadd_ps(c, d);
    __m128 res =_mm_hadd_ps(res_0, res_1);
    res =_mm_hadd_ps(res, res);
    res =_mm_hadd_ps(res, res);

    return _mm_cvtss_f32(res);
}
#endif // __AVX__ || __AVX2__ || __AVX512F__
#endif // defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)

#if defined(__ARM_NEON) || defined(__wasm_simd128__) || defined(__POWER9_VECTOR__)
#define B1(c,s,n)  0x ## n ## c ,  0x ## n ## s
#define B2(c,s,n) B1(c,s,n ## c), B1(c,s,n ## s)
#define B3(c,s,n) B2(c,s,n ## c), B2(c,s,n ## s)
#define B4(c,s,n) B3(c,s,n ## c), B3(c,s,n ## s)
#define B5(c,s,n) B4(c,s,n ## c), B4(c,s,n ## s)
#define B6(c,s,n) B5(c,s,n ## c), B5(c,s,n ## s)
#define B7(c,s,n) B6(c,s,n ## c), B6(c,s,n ## s)
#define B8(c,s  ) B7(c,s,     c), B7(c,s,     s)

// precomputed tables for expanding 8bits to 8 bytes:
static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) }; // ( b) << 4
static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) }; // (!b) << 4
#endif

#if defined(__loongarch_asx)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"
// Convert __m128i to __m256i
static inline __m256i ____m256i(__m128i in) {
    __m256i out = __lasx_xvldi(0);
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX"\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "+f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert two __m128i to __m256i
static inline __m256i lasx_set_q(__m128i inhi, __m128i inlo) {
    __m256i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".ifnc %[out], %[hi]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
        "    xvori.b $xr\\i, $xr\\j, 0       \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out), [hi] "+f" (inhi)
        : [lo] "f" (inlo)
    );
    return out;
}
// Convert __m256i low part to __m128i
static inline __m128i lasx_extracti128_lo(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".ifnc %[out], %[in]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    vori.b $vr\\i, $vr\\j, 0        \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert __m256i high part to __m128i
static inline __m128i lasx_extracti128_hi(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}

static __m256i lasx_set_w(int e7, int e6, int e5, int e4, int e3, int e2, int e1, int e0) {
    v8i32 __ret = {e0, e1, e2, e3, e4, e5, e6, e7};
    return (__m256i)__ret;
}

static __m128i lsx_set_w(int32_t a, int32_t b, int32_t c, int32_t d) {
    v4i32 __ret = {d, c, b, a};
    return (__m128i)__ret;
}

static __m256i lasx_set_d(int64_t a, int64_t b, int64_t c, int64_t d) {
    v4i64 __ret = {d, c, b, a};
    return (__m256i)__ret;
}

static __m256i lasx_insertf128( __m128i x, __m128i y) {
    return lasx_set_q(x, y);
}

static __m128i lsx_shuffle_b(__m128i a, __m128i b) {
    __m128i mask_f, zero, tmp0, tmp2, mask;
    int f = 0x8f;
    mask_f = __lsx_vreplgr2vr_b(f);
    zero = __lsx_vldi(0);
    tmp0 = __lsx_vand_v(b, mask_f); // get mask with low 4 bit and sign bits
    tmp0 = __lsx_vori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    mask = __lsx_vsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lsx_vand_v(tmp0, mask); // maskout the in2 < ones
    return __lsx_vshuf_b(a, zero, tmp2);
}

static __m256i lasx_shuffle_b(__m256i a, __m256i b) {
    __m256i mask_f, zero, tmp0, tmp2, mask;
    int f = 0x8f;
    mask_f = __lasx_xvreplgr2vr_b(f);
    zero = __lasx_xvldi(0);
    tmp0 = __lasx_xvand_v(b, mask_f); // get mask with low 4 bit and sign bits
    tmp0 = __lasx_xvori_b(tmp0, 0x10); // make each mask or  with 0x10 prepare for positive
    mask = __lasx_xvsle_b(zero, tmp0); // if mask >= 0, set mask
    tmp2 = __lasx_xvand_v(tmp0, mask); // maskout the in2 < ones
    return __lasx_xvshuf_b(a, zero, tmp2);
}

static __m256i lasx_extu8_16(__m128i a) {
    __m128i zero = __lsx_vldi(0);
    __m128i vlo = __lsx_vilvl_b(zero, a);
    __m128i vhi = __lsx_vilvh_b(zero, a);
    return lasx_set_q(vhi, vlo);
}

static __m256i lasx_ext8_16(__m128i a) {
     __m128i sign = __lsx_vslti_b(a, 0);
     __m128i vlo = __lsx_vilvl_b(sign, a);
     __m128i vhi = __lsx_vilvh_b(sign, a);
     return lasx_set_q(vhi, vlo);
}

static __m256i lasx_ext16_32(__m128i a) {
    __m256i tmp1;
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 0), 0);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 1), 1);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 2), 2);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 3), 3);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 4), 4);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 5), 5);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 6), 6);
    tmp1 = __lasx_xvinsgr2vr_w(tmp1, __lsx_vpickve2gr_h(a, 7), 7);
    return tmp1;
}

static __m128i lasx_extracti128( __m256i a, int pos) {
    __m128i ret;
    if( pos == 0)
    {
       ret = lasx_extracti128_lo(a);
    } else {
       ret = lasx_extracti128_hi(a);
    }
    return ret;
}

static __m128 lasx_extractf128( __m256 a, int pos) {
    __m128 ret;
    if( pos == 0)
    {
       ret = (__m128)lasx_extracti128_lo((__m256i)a);
    } else {
       ret = (__m128)lasx_extracti128_hi((__m256i)a);
    }
    return ret;
}

static __m128i lsx_hadd_h(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_h(b, a);
    __m128i tmp2 = __lsx_vpickod_h(b, a);
    return __lsx_vadd_h(tmp1, tmp2);
}

static __m128i lsx_hadd_w(__m128i a, __m128i b) {
    __m128i tmp1 = __lsx_vpickev_w(b, a);
    __m128i tmp2 = __lsx_vpickod_w(b, a);
    return __lsx_vadd_w(tmp1, tmp2);
}

static __m128 lsx_hadd_s(__m128 a, __m128 b) {
    __m128 tmp1 = (__m128)__lsx_vpickev_w((__m128i)b, (__m128i)a);
    __m128 tmp2 = (__m128)__lsx_vpickod_w((__m128i)b, (__m128i)a);

    return __lsx_vfadd_s(tmp1, tmp2);
}

static __m256i lasx_maddubs_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_h_b(a, b);
    tmp2 = __lasx_xvmulwod_h_b(a, b);
    return __lasx_xvsadd_h(tmp1, tmp2);
}

static __m256i lasx_madd_h(__m256i a, __m256i b) {
    __m256i tmp1, tmp2;
    tmp1 = __lasx_xvmulwev_w_h(a, b);
    tmp2 = __lasx_xvmulwod_w_h(a, b);
    return __lasx_xvadd_w(tmp1, tmp2);
}

static __m256i lasx_packs_w(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_w(a, 15);
    tmp1 = __lasx_xvsat_w(b, 15);
    return __lasx_xvpickev_h(tmp1, tmp);
}

static __m256i lasx_packs_h(__m256i a, __m256i b) {
    __m256i tmp, tmp1;
    tmp = __lasx_xvsat_h(a, 7);
    tmp1 = __lasx_xvsat_h(b, 7);
    return __lasx_xvpickev_b(tmp1, tmp);
}

static __m128i lsx_packs_w(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_w(a, 15);
    tmp1 = __lsx_vsat_w(b, 15);
    return __lsx_vpickev_h(tmp1, tmp);
}

static __m128i lsx_packs_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_h(a, 7);
    tmp1 = __lsx_vsat_h(b, 7);
    return __lsx_vpickev_b(tmp1, tmp);
}

static __m128i lsx_packus_h(__m128i a, __m128i b) {
    __m128i tmp, tmp1;
    tmp = __lsx_vsat_hu(a, 7);
    tmp1 = __lsx_vsat_hu(b, 7);
    return __lsx_vpickev_b(tmp1, tmp);
}


static __m128i lsx_maddubs_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_h_b(a, b);
    tmp2 = __lsx_vmulwod_h_b(a, b);
    return __lsx_vsadd_h(tmp1, tmp2);
}

static __m128i lsx_madd_h(__m128i a, __m128i b) {
    __m128i tmp1, tmp2;
    tmp1 = __lsx_vmulwev_w_h(a, b);
    tmp2 = __lsx_vmulwod_w_h(a, b);
    return __lsx_vadd_w(tmp1, tmp2);
}

// multiply int8_t, add results pairwise twice
static inline __m128i mul_sum_i8_pairs(const __m128i x, const __m128i y) {
    // Get absolute values of x vectors
    const __m128i ax = __lsx_vsigncov_b(x, x);
    // Sign the values of the y vectors
    const __m128i sy = __lsx_vsigncov_b(x, y);
    // Perform multiplication and create 16-bit values
    const __m128i dot = lsx_maddubs_h(ax, sy);
    const __m128i ones = __lsx_vreplgr2vr_h(1);
    return lsx_madd_h(ones, dot);
}

// horizontally add 8 floats
static inline float hsum_float_8(const __m256 x) {
    __m128 res = lasx_extractf128(x, 1);
    ft_union tmp;
    res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
    res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
    tmp.i = __lsx_vpickve2gr_w(res, 0);
    return tmp.f;
}

// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}

// horizontally add 4 int32_t
static inline int hsum_i32_4(const __m128i a) {
    __m128i ev = __lsx_vpickev_w(a, a);
    __m128i od = __lsx_vpickod_w(a, a);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}

// spread 32 bits to 32 bytes { 0x00, 0xFF }
static inline __m256i bytes_from_bits_32(const uint8_t * x) {

    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = lasx_set_d(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);

    __m256i bytes = lasx_shuffle_b(__lasx_xvreplgr2vr_w(x32), shuf_mask);
    const __m256i bit_mask = __lasx_xvreplgr2vr_d(0x7fbfdfeff7fbfdfe);
    bytes = __lasx_xvor_v(bytes, bit_mask);
    return __lasx_xvseq_b(bytes, __lasx_xvreplgr2vr_d(-1));
}

// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i lo = __lsx_vld((const __m128i *)rsi, 0);
    __m128i hi = __lsx_vsrli_h(lo, 4);
    return __lasx_xvandi_b(lasx_insertf128(hi, lo), 0xf);
}

// add int16_t pairwise and return as float vector
static inline __m256 sum_i16_pairs_float(const __m256i x) {
    __m256i v = __lasx_xvpackod_h(x, x);
    __m256i summed_pairs = __lasx_xvaddwev_w_h(x, v);
    return __lasx_xvffint_s_w(summed_pairs);
}

static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    // Perform multiplication and create 16-bit values
    const __m256i dot = lasx_maddubs_h(ax, sy);
    return sum_i16_pairs_float(dot);
}

// multiply int8_t, add results pairwise twice and return as float vector
static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {

    // Get absolute values of x vectors
    const __m256i ax = __lasx_xvsigncov_b(x, x);
    // Sign the values of the y vectors
    const __m256i sy = __lasx_xvsigncov_b(x, y);

    return mul_sum_us8_pairs_float(ax, sy);
}

static inline __m128i packNibbles( __m256i bytes ) {
    // Move bits within 16-bit lanes from 0000_abcd_0000_efgh into 0000_0000_abcd_efgh
    const __m256i lowByte = __lasx_xvreplgr2vr_h(0xFF);
     __m256i high = __lasx_xvandn_v(lowByte, bytes);
    __m256i low = __lasx_xvand_v(lowByte, bytes);
    high = __lasx_xvsrli_h(high, 4);
    bytes = __lasx_xvor_v(low, high);
    // Compress uint16_t lanes into bytes
    __m128i *r0 = (__m128i *)&bytes;
    __m256i tmp_h128 = __lasx_xvpermi_q(bytes, bytes, 0x11);
    __m128i *r1 = (__m128i *)&tmp_h128;

    __m128i zero = __lsx_vldi(0);
    __m128i tmp, tmp2, tmp3;

    tmp = __lsx_vmax_h(zero, *r0);
    tmp2 = __lsx_vsat_hu(tmp, 7);

    tmp = __lsx_vmax_h(zero, *r1);
    tmp3 = __lsx_vsat_hu(tmp, 7);
    return  __lsx_vpickev_b(tmp3, tmp2);
}
#endif  //__loongarch_asx

void quantize_row_q4_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_0_ref(x, y, k);
}

void quantize_row_q4_1(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q4_1_ref(x, y, k);
}

void quantize_row_q5_0(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q5_0_ref(x, y, k);
}

void quantize_row_q5_1(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q5_1_ref(x, y, k);
}

void quantize_row_q8_0(const float * restrict x, void * restrict vy, int64_t k) {
    assert(QK8_0 == 32);
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);
        }
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);
        }
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

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

        // Quantize these floats
        const float d = maxScalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( maxScalar != 0.0f ) ? 127.f / maxScalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

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
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
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

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_0);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_0, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);
    }

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        ft_union fi;
        __m256 v0 = (__m256)__lasx_xvld( x , 0);
        __m256 v1 = (__m256)__lasx_xvld( x , 32);
        __m256 v2 = (__m256)__lasx_xvld( x , 64);
        __m256 v3 = (__m256)__lasx_xvld( x , 96);
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs , 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vinsgr2vr_w(tmp, __lsx_vpickve2gr_w( max4, 1 ), 0 ));
        fi.i = __lsx_vpickve2gr_w( (__m128i)max4, 0 );
        const float max_scalar = fi.f;

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = (__m256)__lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128( i0, 0 );
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0);
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);

    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_0_ref(x, y, k);
#endif
}

void quantize_row_q8_1(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK8_1 == 0);
    const int nb = k / QK8_1;

    block_q8_1 * restrict y = vy;

#if defined(__ARM_NEON)
    for (int i = 0; i < nb; i++) {
        float32x4_t srcv [8];
        float32x4_t asrcv[8];
        float32x4_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vld1q_f32(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vabsq_f32(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vmaxq_f32(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vmaxq_f32(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vmaxq_f32(amaxv[8*j], amaxv[8*j+4]);

        const float amax = vmaxvq_f32(amaxv[0]);

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        int32x4_t accv = vdupq_n_s32(0);

        for (int j = 0; j < 8; j++) {
            const float32x4_t v  = vmulq_n_f32(srcv[j], id);
            const int32x4_t   vi = vcvtnq_s32_f32(v);

            y[i].qs[4*j + 0] = vgetq_lane_s32(vi, 0);
            y[i].qs[4*j + 1] = vgetq_lane_s32(vi, 1);
            y[i].qs[4*j + 2] = vgetq_lane_s32(vi, 2);
            y[i].qs[4*j + 3] = vgetq_lane_s32(vi, 3);

            accv = vaddq_s32(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(d * vaddvq_s32(accv));
    }
#elif defined(__wasm_simd128__)
    for (int i = 0; i < nb; i++) {
        v128_t srcv [8];
        v128_t asrcv[8];
        v128_t amaxv[8];

        for (int j = 0; j < 8; j++) srcv[j]  = wasm_v128_load(x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = wasm_f32x4_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = wasm_f32x4_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = wasm_f32x4_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = wasm_f32x4_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(wasm_f32x4_extract_lane(amaxv[0], 0),
                                   wasm_f32x4_extract_lane(amaxv[0], 1)),
                               MAX(wasm_f32x4_extract_lane(amaxv[0], 2),
                                   wasm_f32x4_extract_lane(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        v128_t accv = wasm_i32x4_splat(0);

        for (int j = 0; j < 8; j++) {
            const v128_t v  = wasm_f32x4_mul(srcv[j], wasm_f32x4_splat(id));
            const v128_t vi = wasm_i32x4_trunc_sat_f32x4(v);

            y[i].qs[4*j + 0] = wasm_i32x4_extract_lane(vi, 0);
            y[i].qs[4*j + 1] = wasm_i32x4_extract_lane(vi, 1);
            y[i].qs[4*j + 2] = wasm_i32x4_extract_lane(vi, 2);
            y[i].qs[4*j + 3] = wasm_i32x4_extract_lane(vi, 3);

            accv = wasm_i32x4_add(accv, vi);
        }

        y[i].s = GGML_FP32_TO_FP16(
                d * (wasm_i32x4_extract_lane(accv, 0) +
                     wasm_i32x4_extract_lane(accv, 1) +
                     wasm_i32x4_extract_lane(accv, 2) +
                     wasm_i32x4_extract_lane(accv, 3)));
    }
#elif defined(__AVX2__) || defined(__AVX__)
    for (int i = 0; i < nb; i++) {
        // Load elements into 4 AVX vectors
        __m256 v0 = _mm256_loadu_ps( x );
        __m256 v1 = _mm256_loadu_ps( x + 8 );
        __m256 v2 = _mm256_loadu_ps( x + 16 );
        __m256 v3 = _mm256_loadu_ps( x + 24 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 signBit = _mm256_set1_ps( -0.0f );
        __m256 maxAbs = _mm256_andnot_ps( signBit, v0 );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v1 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v2 ) );
        maxAbs = _mm256_max_ps( maxAbs, _mm256_andnot_ps( signBit, v3 ) );

        __m128 max4 = _mm_max_ps( _mm256_extractf128_ps( maxAbs, 1 ), _mm256_castps256_ps128( maxAbs ) );
        max4 = _mm_max_ps( max4, _mm_movehl_ps( max4, max4 ) );
        max4 = _mm_max_ss( max4, _mm_movehdup_ps( max4 ) );
        const float max_scalar = _mm_cvtss_f32( max4 );

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = _mm256_set1_ps( id );

        // Apply the multiplier
        v0 = _mm256_mul_ps( v0, mul );
        v1 = _mm256_mul_ps( v1, mul );
        v2 = _mm256_mul_ps( v2, mul );
        v3 = _mm256_mul_ps( v3, mul );

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
        // Compute the sum of the quants and set y[i].s
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_8(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3))));

        // Convert int32 to int16
        i0 = _mm256_packs_epi32( i0, i1 );	// 0, 1, 2, 3,  8, 9, 10, 11,  4, 5, 6, 7, 12, 13, 14, 15
        i2 = _mm256_packs_epi32( i2, i3 );	// 16, 17, 18, 19,  24, 25, 26, 27,  20, 21, 22, 23, 28, 29, 30, 31
                                            // Convert int16 to int8
        i0 = _mm256_packs_epi16( i0, i2 );	// 0, 1, 2, 3,  8, 9, 10, 11,  16, 17, 18, 19,  24, 25, 26, 27,  4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31

        // We got our precious signed bytes, but the order is now wrong
        // These AVX2 pack instructions process 16-byte pieces independently
        // The following instruction is fixing the order
        const __m256i perm = _mm256_setr_epi32( 0, 4, 1, 5, 2, 6, 3, 7 );
        i0 = _mm256_permutevar8x32_epi32( i0, perm );

        _mm256_storeu_si256((__m256i *)y[i].qs, i0);
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

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = _mm_add_epi32(_mm_add_epi32(ni0, ni1), _mm_add_epi32(ni2, ni3));
        const __m128i s1 = _mm_add_epi32(_mm_add_epi32(ni4, ni5), _mm_add_epi32(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(_mm_add_epi32(s0, s1)));

        // Convert int32 to int16
        ni0 = _mm_packs_epi32( ni0, ni1 );
        ni2 = _mm_packs_epi32( ni2, ni3 );
        ni4 = _mm_packs_epi32( ni4, ni5 );
        ni6 = _mm_packs_epi32( ni6, ni7 );
        // Convert int16 to int8
        ni0 = _mm_packs_epi16( ni0, ni2 );
        ni4 = _mm_packs_epi16( ni4, ni6 );

        _mm_storeu_si128((__m128i *)(y[i].qs +  0), ni0);
        _mm_storeu_si128((__m128i *)(y[i].qs + 16), ni4);
#endif
    }
#elif defined(__riscv_v_intrinsic)

    size_t vl = __riscv_vsetvl_e32m4(QK8_1);

    for (int i = 0; i < nb; i++) {
        // load elements
        vfloat32m4_t v_x   = __riscv_vle32_v_f32m4(x+i*QK8_1, vl);

        vfloat32m4_t vfabs = __riscv_vfabs_v_f32m4(v_x, vl);
        vfloat32m1_t tmp   = __riscv_vfmv_v_f_f32m1(0.0, vl);
        vfloat32m1_t vmax  = __riscv_vfredmax_vs_f32m4_f32m1(vfabs, tmp, vl);
        float amax = __riscv_vfmv_f_s_f32m1_f32(vmax);

        const float d  = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        vfloat32m4_t x0 = __riscv_vfmul_vf_f32m4(v_x, id, vl);

        // convert to integer
        vint16m2_t   vi = __riscv_vfncvt_x_f_w_i16m2(x0, vl);
        vint8m1_t    vs = __riscv_vncvt_x_x_w_i8m1(vi, vl);

        // store result
        __riscv_vse8_v_i8m1(y[i].qs , vs, vl);

        // compute sum for y[i].s
        vint16m1_t tmp2 = __riscv_vmv_v_x_i16m1(0, vl);
        vint16m1_t vwrs = __riscv_vwredsum_vs_i8m1_i16m1(vs, tmp2, vl);

        // set y[i].s
        int sum = __riscv_vmv_x_s_i16m1_i16(vwrs);
        y[i].s = GGML_FP32_TO_FP16(sum*d);
    }

#elif defined(__POWER9_VECTOR__)
    for (int i = 0; i < nb; i++) {
        vector float srcv [8];
        vector float asrcv[8];
        vector float amaxv[8];
        vector signed int vi[8];

        for (int j = 0; j < 8; j++) srcv[j]  = vec_xl(0, x + i*32 + 4*j);
        for (int j = 0; j < 8; j++) asrcv[j] = vec_abs(srcv[j]);

        for (int j = 0; j < 4; j++) amaxv[2*j] = vec_max(asrcv[2*j], asrcv[2*j+1]);
        for (int j = 0; j < 2; j++) amaxv[4*j] = vec_max(amaxv[4*j], amaxv[4*j+2]);
        for (int j = 0; j < 1; j++) amaxv[8*j] = vec_max(amaxv[8*j], amaxv[8*j+4]);

        const float amax = MAX(MAX(vec_extract(amaxv[0], 0),
                                   vec_extract(amaxv[0], 1)),
                               MAX(vec_extract(amaxv[0], 2),
                                   vec_extract(amaxv[0], 3)));

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;
        const vector float vid = vec_splats(id);

        y[i].d = GGML_FP32_TO_FP16(d);

        vector int accv = vec_splats(0);

        for (int j = 0; j < 8; j++) {
            const vector float v  = vec_round(vec_mul(srcv[j], vid));
            vi[j] = vec_cts(v, 0);

            accv = vec_add(accv, vi[j]);
        }
        vec_xst(vec_pack(vec_pack(vi[0], vi[1]), vec_pack(vi[2], vi[3])),  0, &y[i].qs[0]);
        vec_xst(vec_pack(vec_pack(vi[4], vi[5]), vec_pack(vi[6], vi[7])), 16, &y[i].qs[0]);

        accv = vec_add(accv, vec_sld(accv, accv, 4));
        accv = vec_add(accv, vec_sld(accv, accv, 8));
        y[i].s = GGML_FP32_TO_FP16(d * vec_extract(accv, 0));
    }

#elif defined(__loongarch_asx)
    for (int i = 0; i < nb; i++) {
        ft_union ft;
        __m256 v0 = (__m256)__lasx_xvld( x , 0 );
        __m256 v1 = (__m256)__lasx_xvld( x , 32 );
        __m256 v2 = (__m256)__lasx_xvld( x , 64 );
        __m256 v3 = (__m256)__lasx_xvld( x , 96 );
        x += 32;

        // Compute max(abs(e)) for the block
        const __m256 sign_bit = __lasx_xvreplfr2vr_s( -0.0f );
        __m256 max_abs = (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v0 );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v1 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v2 ) );
        max_abs = __lasx_xvfmax_s( max_abs, (__m256)__lasx_xvandn_v( (__m256i)sign_bit, (__m256i)v3 ) );

        __m128 max4 = __lsx_vfmax_s( lasx_extractf128( max_abs, 1 ), lasx_extractf128( max_abs, 0) );
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vpickod_d((__m128i) max4, (__m128i)max4 ) );
        __m128 tmp = max4;
        max4 = __lsx_vfmax_s( max4, (__m128)__lsx_vextrins_w((__m128i)tmp, (__m128i)max4, 0x10 ));
        ft.i = __lsx_vpickve2gr_w( (__m128i)max4, 0 );
        const float max_scalar = ft.f;

        // Quantize these floats
        const float d = max_scalar / 127.f;
        y[i].d = GGML_FP32_TO_FP16(d);
        const float id = ( max_scalar != 0.0f ) ? 127.f / max_scalar : 0.0f;
        const __m256 mul = __lasx_xvreplfr2vr_s( id );

        // Apply the multiplier
        v0 = __lasx_xvfmul_s( v0, mul );
        v1 = __lasx_xvfmul_s( v1, mul );
        v2 = __lasx_xvfmul_s( v2, mul );
        v3 = __lasx_xvfmul_s( v3, mul );

        // Round to nearest integer
        __m256i i0 = __lasx_xvftintrne_w_s( v0 );
        __m256i i1 = __lasx_xvftintrne_w_s( v1 );
        __m256i i2 = __lasx_xvftintrne_w_s( v2 );
        __m256i i3 = __lasx_xvftintrne_w_s( v3 );

        __m128i ni0 = lasx_extracti128(i0, 0);
        __m128i ni1 = lasx_extracti128( i0, 1);
        __m128i ni2 = lasx_extracti128( i1, 0);
        __m128i ni3 = lasx_extracti128( i1, 1);
        __m128i ni4 = lasx_extracti128( i2, 0 );
        __m128i ni5 = lasx_extracti128( i2, 1);
        __m128i ni6 = lasx_extracti128( i3, 0);
        __m128i ni7 = lasx_extracti128( i3, 1);

        // Compute the sum of the quants and set y[i].s
        const __m128i s0 = __lsx_vadd_w(__lsx_vadd_w(ni0, ni1), __lsx_vadd_w(ni2, ni3));
        const __m128i s1 = __lsx_vadd_w(__lsx_vadd_w(ni4, ni5), __lsx_vadd_w(ni6, ni7));
        y[i].s = GGML_FP32_TO_FP16(d * hsum_i32_4(__lsx_vadd_w(s0, s1)));

        // Convert int32 to int16
        ni0 = lsx_packs_w( ni0, ni1 );
        ni2 = lsx_packs_w( ni2, ni3 );
        ni4 = lsx_packs_w( ni4, ni5 );
        ni6 = lsx_packs_w( ni6, ni7 );
        // Convert int16 to int8
        ni0 = lsx_packs_h( ni0, ni2 );
        ni4 = lsx_packs_h( ni4, ni6 );

        __lsx_vst(ni0, (__m128i *)(y[i].qs +  0), 0);
        __lsx_vst(ni4, (__m128i *)(y[i].qs + 16), 0);
    }
#else
    GGML_UNUSED(nb);
    // scalar
    quantize_row_q8_1_ref(x, y, k);
#endif
}

//
// 2-6 bit quantization in super-blocks
//

//
// ===================== Helper functions
//
static inline int nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static float make_qx_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, int rmse_type,
        const float * restrict qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 0; i < n; ++i) {
#else
    for (int i = 0; i < n; ++i) {
#endif
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

static float make_q3_quants(int n, int nmax, const float * restrict x, int8_t * restrict L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < GROUP_MAX_EPS) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}

static float make_qkx1_quants(int n, int nmax, const float * restrict x, uint8_t * restrict L, float * restrict the_min,
        int ntry, float alpha) {
    float min = x[0];
    float max = x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
    }
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = 0;
        return 0.f;
    }
    if (min > 0) min = 0;
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    for (int itry = 0; itry < ntry; ++itry) {
        float sumlx = 0; int suml2 = 0;
        bool did_change = false;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            if (l != L[i]) {
                L[i] = l;
                did_change = true;
            }
            sumlx += (x[i] - min)*l;
            suml2 += l*l;
        }
        scale = sumlx/suml2;
        float sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += x[i] - scale*L[i];
        }
        min = alpha*min + (1 - alpha)*sum/n;
        if (min > 0) min = 0;
        iscale = 1/scale;
        if (!did_change) break;
    }
    *the_min = -min;
    return scale;
}

static float make_qkx2_quants(int n, int nmax, const float * restrict x, const float * restrict weights,
        uint8_t * restrict L, float * restrict the_min, uint8_t * restrict Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
#ifdef HAVE_BUGGY_APPLE_LINKER
    // use 'volatile' to prevent unroll and work around a bug in Apple ld64 1015.7
    for (volatile int i = 1; i < n; ++i) {
#else
    for (int i = 1; i < n; ++i) {
#endif
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

//========================- 2-bit (de)-quantization

void quantize_row_q2_K(const float * restrict x, void * restrict vy, int64_t k) {
    quantize_row_q2_K_ref(x, vy, k);
}

//========================= 3-bit (de)-quantization

void quantize_row_q3_K(const float * restrict x, void * restrict vy, int64_t k) {
    quantize_row_q3_K_ref(x, vy, k);
}

// ====================== 4-bit (de)-quantization

void quantize_row_q4_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q4_K * restrict y = vy;
    quantize_row_q4_K_ref(x, y, k);
}

// ====================== 5-bit (de)-quantization

void quantize_row_q5_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q5_K * restrict y = vy;
    quantize_row_q5_K_ref(x, y, k);
}

// ====================== 6-bit (de)-quantization

void quantize_row_q6_K(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_q6_K * restrict y = vy;
    quantize_row_q6_K_ref(x, y, k);
}

// ====================== Ternary (de)-quantization (BitNet b1.58 and TriLMs)

void quantize_row_tq1_0(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq1_0 * restrict y = vy;
    quantize_row_tq1_0_ref(x, y, k);
}

void quantize_row_tq2_0(const float * restrict x, void * restrict vy, int64_t k) {
    assert(k % QK_K == 0);
    block_tq2_0 * restrict y = vy;
    quantize_row_tq2_0_ref(x, y, k);
}

static const int8_t kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

//===================================== Q8_K ==============================================

void quantize_row_q8_K(const float * restrict x, void * restrict y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}

//===================================== Dot products =================================

//
// Helper functions
//
#if __AVX__ || __AVX2__ || __AVX512F__

// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*)k_shuffle + i);
}
#elif defined(__loongarch_asx)
// shuffles to pick the required scales in dot products
static inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return __lasx_xvld((const __m256i*)k_shuffle + i, 0);
}
static inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return __lasx_xvld((const __m256i*)k_shuffle + i, 0);
}
static inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return __lsx_vld((const __m128i*)k_shuffle + i, 0);
}
#endif

void ggml_vec_dot_q4_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_0 * restrict vx0 = vx;
        const block_q4_0 * restrict vx1 = (const block_q4_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * restrict vy0 = vy;
        const block_q8_0 * restrict vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_0 * restrict b_x0 = &vx0[i];
            const block_q4_0 * restrict b_x1 = &vx1[i];
            const block_q8_0 * restrict b_y0 = &vy0[i];
            const block_q8_0 * restrict b_y1 = &vy1[i];

            const uint8x16_t m4b = vdupq_n_u8(0x0F);
            const int8x16_t  s8b = vdupq_n_s8(0x8);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // sub 8
            const int8x16_t x0_l = vsubq_s8(v0_0l, s8b);
            const int8x16_t x0_h = vsubq_s8(v0_0h, s8b);
            const int8x16_t x1_l = vsubq_s8(v0_1l, s8b);
            const int8x16_t x1_h = vsubq_s8(v0_1h, s8b);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt()*8;

    // VLA Implementation using switch case
    switch (vector_length) {
        case 128:
            {
                // predicate for activating higher lanes for 4 float32 elements
                const svbool_t ph4 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * restrict x0 = &x[ib + 0];
                    const block_q4_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx0r, 0x0F));
                    const svint8_t qx0h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx0r, 0x04));
                    const svint8_t qx1l = svreinterpret_s8_u8(svand_n_u8_m(svptrue_b8(), qx1r, 0x0F));
                    const svint8_t qx1h = svreinterpret_s8_u8(svlsr_n_u8_m(svptrue_b8(), qx1r, 0x04));

                    // sub 8
                    const svint8_t qx0ls = svsub_n_s8_x(svptrue_b8(), qx0h, 8);
                    const svint8_t qx0hs = svsub_n_s8_x(svptrue_b8(), qx0l, 8);
                    const svint8_t qx1ls = svsub_n_s8_x(svptrue_b8(), qx1h, 8);
                    const svint8_t qx1hs = svsub_n_s8_x(svptrue_b8(), qx1l, 8);

                    // load y
                    const svint8_t qy0h = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy0l = svld1_s8(svptrue_b8(), y0->qs + 16);
                    const svint8_t qy1h = svld1_s8(svptrue_b8(), y1->qs);
                    const svint8_t qy1l = svld1_s8(svptrue_b8(), y1->qs + 16);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph4, sumv0, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx0ls, qy0l),
                                    svdot_s32(svdup_n_s32(0), qx0hs, qy0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph4, sumv1, svcvt_f32_s32_x(ph4, svadd_x(ph4,
                                    svdot_s32(svdup_n_s32(0), qx1ls, qy1l),
                                    svdot_s32(svdup_n_s32(0), qx1hs, qy1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 256:
            {
                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for  16 int8 elements
                const svbool_t pl16 = svnot_b_z(svptrue_b8(), ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * restrict x0 = &x[ib + 0];
                    const block_q4_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(svptrue_b8(), x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(svptrue_b8(), x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(svptrue_b8(), qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(svptrue_b8(), qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating higher lanes for 32 int8 elements
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);

                // predicate for activating higher lanes for 16 int8 elements
                const svbool_t ph16 = svptrue_pat_b8(SV_VL16);
                // predicate for activating lower lanes for 16 int8 elements from first 32 int8 activated lanes
                const svbool_t pl16 = svnot_b_z(ph32, ph16);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q4_0 * restrict x0 = &x[ib + 0];
                    const block_q4_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    // load x
                    const svuint8_t qx0r = svld1rq_u8(ph32, x0->qs);
                    const svuint8_t qx1r = svld1rq_u8(ph32, x1->qs);

                    // 4-bit -> 8-bit
                    const svint8_t qx0 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx0r, 0x0F), 0x04));
                    const svint8_t qx1 = svreinterpret_s8_u8(svlsr_n_u8_m(pl16, svand_n_u8_m(ph16, qx1r, 0x0F), 0x04));

                    // sub 8
                    const svint8_t qx0s = svsub_n_s8_x(ph32, qx0, 8);
                    const svint8_t qx1s = svsub_n_s8_x(ph32, qx1, 8);

                    // load y
                    const svint8_t qy0 = svld1_s8(ph32, y0->qs);
                    const svint8_t qy1 = svld1_s8(ph32, y1->qs);

                    // dot product
                    sumv0 = svmla_n_f32_x(ph32, sumv0, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx0s, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(ph32, sumv1, svcvt_f32_s32_x(ph32,
                                svdot_s32(svdup_n_s32(0), qx1s, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(ph32, svadd_f32_x(ph32, sumv0, sumv1));
            } break;
        default:
            assert(false && "Unsupported vector length");
            break;
    }

#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_0 * restrict x0 = &x[ib + 0];
        const block_q4_0 * restrict x1 = &x[ib + 1];
        const block_q8_0 * restrict y0 = &y[ib + 0];
        const block_q8_0 * restrict y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);
        const int8x16_t  s8b = vdupq_n_s8(0x8);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // sub 8
        const int8x16_t v0_0ls = vsubq_s8(v0_0l, s8b);
        const int8x16_t v0_0hs = vsubq_s8(v0_0h, s8b);
        const int8x16_t v0_1ls = vsubq_s8(v0_1l, s8b);
        const int8x16_t v0_1hs = vsubq_s8(v0_1h, s8b);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0ls, v1_0l), v0_0hs, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1ls, v1_1l), v0_1hs, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = _mm256_set1_epi8( 8 );
        qx = _mm256_sub_epi8( qx, off );

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_1), _mm_set1_epi8(8));
        const __m128i q4b_1_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_1, 4)), _mm_set1_epi8(8));
        const __m128i q4b_2_0 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), q4bits_2), _mm_set1_epi8(8));
        const __m128i q4b_2_1 = _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(q4bits_2, 4)), _mm_set1_epi8(8));

        const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
        const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
        const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
        const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
        const __m128i p_1 = _mm_add_epi16(p16_1_0, p16_1_1);
        const __m128i p_2 = _mm_add_epi16(p16_2_0, p16_2_1);
        const __m256 p =  sum_i16_pairs_float(p_2, p_1);

        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__SSSE3__)
    // set constants
    const __m128i lowMask = _mm_set1_epi8(0xF);
    const __m128i off = _mm_set1_epi8(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = _mm_setzero_ps();
    __m128 acc_1 = _mm_setzero_ps();
    __m128 acc_2 = _mm_setzero_ps();
    __m128 acc_3 = _mm_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        _mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = _mm_set1_ps( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = _mm_loadu_si128((const __m128i *)x[ib].qs);

        __m128i bx_0 = _mm_and_si128(lowMask, tmp_0_1);
        __m128i by_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        bx_0 = _mm_sub_epi8(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_0_1, 4));
        __m128i by_1 = _mm_loadu_si128((const __m128i *)(y[ib].qs + 16));
        bx_1 = _mm_sub_epi8(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        _mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        _mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = _mm_set1_ps( GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);

        __m128i bx_2 = _mm_and_si128(lowMask, tmp_2_3);
        __m128i by_2 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        bx_2 = _mm_sub_epi8(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = _mm_and_si128(lowMask, _mm_srli_epi64(tmp_2_3, 4));
        __m128i by_3 = _mm_loadu_si128((const __m128i *)(y[ib + 1].qs + 16));
        bx_3 = _mm_sub_epi8(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = _mm_cvtepi32_ps(i32_0);
        __m128 p1 = _mm_cvtepi32_ps(i32_1);
        __m128 p2 = _mm_cvtepi32_ps(i32_2);
        __m128 p3 = _mm_cvtepi32_ps(i32_3);

        // Apply the scale
        __m128 p0_d = _mm_mul_ps( d_0_1, p0 );
        __m128 p1_d = _mm_mul_ps( d_0_1, p1 );
        __m128 p2_d = _mm_mul_ps( d_2_3, p2 );
        __m128 p3_d = _mm_mul_ps( d_2_3, p3 );

        // Acummulate
        acc_0 = _mm_add_ps(p0_d, acc_0);
        acc_1 = _mm_add_ps(p1_d, acc_1);
        acc_2 = _mm_add_ps(p2_d, acc_2);
        acc_3 = _mm_add_ps(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (; ib < nb; ++ib) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        // subtract offset
        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 8, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 8, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector signed char v8 = vec_splats((signed char)0x8);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_sub(q4x0, v8);
        q4x1 = vec_sub(q4x1, v8);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi0 = vec_sum4s(qv1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);

        // Now we have a vector with bytes in [ 0 .. 15 ] interval. Offset them into [ -8 .. +7 ] interval.
        const __m256i off = __lasx_xvreplgr2vr_b( 8 );
        qx = __lasx_xvsub_b( qx, off );

        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__loongarch_sx)
    // set constants
    const __m128i low_mask = __lsx_vreplgr2vr_b(0xF);
    const __m128i off = __lsx_vreplgr2vr_b(8);

    // Initialize accumulator with zeros
    __m128 acc_0 = __lsx_vldi(0);
    __m128 acc_1 = __lsx_vldi(0);
    __m128 acc_2 = __lsx_vldi(0);
    __m128 acc_3 = __lsx_vldi(0);

    for (; ib + 1 < nb; ib += 2) {

        // Compute combined scale for the block 0 and 1
        const __m128 d_0_1 = __lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d) );

        const __m128i tmp_0_1 = __lsx_vld((const __m128i *)x[ib].qs, 0);

        __m128i bx_0 = __lsx_vand_v(low_mask, tmp_0_1);
        __m128i by_0 = __lsx_vld((const __m128i *)y[ib].qs, 0);
        bx_0 = __lsx_vsub_b(bx_0, off);
        const __m128i i32_0 = mul_sum_i8_pairs(bx_0, by_0);

        __m128i bx_1 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_0_1, 4));
        __m128i by_1 = __lsx_vld((const __m128i *)(y[ib].qs + 16), 0);
        bx_1 = __lsx_vsub_b(bx_1, off);
        const __m128i i32_1 = mul_sum_i8_pairs(bx_1, by_1);

        //_mm_prefetch(&x[ib] + 2 * sizeof(block_q4_0), _MM_HINT_T0);
        //_mm_prefetch(&y[ib] + 2 * sizeof(block_q8_0), _MM_HINT_T0);

        // Compute combined scale for the block 2 and 3
        const __m128 d_2_3 = __lsx_vreplgr2vr_w( GGML_FP16_TO_FP32(x[ib + 1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) );

        const __m128i tmp_2_3 = __lsx_vld((const __m128i *)x[ib + 1].qs, 0);

        __m128i bx_2 = __lsx_vand_v(low_mask, tmp_2_3);
        __m128i by_2 = __lsx_vld((const __m128i *)y[ib + 1].qs, 0);
        bx_2 = __lsx_vsub_b(bx_2, off);
        const __m128i i32_2 = mul_sum_i8_pairs(bx_2, by_2);

        __m128i bx_3 = __lsx_vand_v(low_mask, __lsx_vsrli_d(tmp_2_3, 4));
        __m128i by_3 = __lsx_vld((const __m128i *)(y[ib + 1].qs + 16), 0);
        bx_3 = __lsx_vsub_b(bx_3, off);
        const __m128i i32_3 = mul_sum_i8_pairs(bx_3, by_3);

        // Convert int32_t to float
        __m128 p0 = __lsx_vffint_s_w(i32_0);
        __m128 p1 = __lsx_vffint_s_w(i32_1);
        __m128 p2 = __lsx_vffint_s_w(i32_2);
        __m128 p3 = __lsx_vffint_s_w(i32_3);

        // Apply the scale
        __m128 p0_d = __lsx_vfmul_s( d_0_1, p0 );
        __m128 p1_d = __lsx_vfmul_s( d_0_1, p1 );
        __m128 p2_d = __lsx_vfmul_s( d_2_3, p2 );
        __m128 p3_d = __lsx_vfmul_s( d_2_3, p3 );

        // Acummulate
        acc_0 = __lsx_vfadd_s(p0_d, acc_0);
        acc_1 = __lsx_vfadd_s(p1_d, acc_1);
        acc_2 = __lsx_vfadd_s(p2_d, acc_2);
        acc_3 = __lsx_vfadd_s(p3_d, acc_3);
    }

    sumf = hsum_float_4x4(acc_0, acc_1, acc_2, acc_3);
#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F) - 8;
            const int v1 = (x[ib].qs[j] >>   4) - 8;

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += sumi*GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d);
    }

    *s = sumf;
}

void ggml_vec_dot_q4_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q4_1 * restrict vx0 = vx;
        const block_q4_1 * restrict vx1 = (const block_q4_1 *) ((const uint8_t*)vx + bx);
        const block_q8_1 * restrict vy0 = vy;
        const block_q8_1 * restrict vy1 = (const block_q8_1 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);
        float32x4_t summs0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q4_1 * restrict b_x0 = &vx0[i];
            const block_q4_1 * restrict b_x1 = &vx1[i];
            const block_q8_1 * restrict b_y0 = &vy0[i];
            const block_q8_1 * restrict b_y1 = &vy1[i];

            float32_t summs_t[4] = {
                GGML_FP16_TO_FP32(b_x0->m) * GGML_FP16_TO_FP32(b_y0->s),
                GGML_FP16_TO_FP32(b_x1->m) * GGML_FP16_TO_FP32(b_y0->s),
                GGML_FP16_TO_FP32(b_x0->m) * GGML_FP16_TO_FP32(b_y1->s),
                GGML_FP16_TO_FP32(b_x1->m) * GGML_FP16_TO_FP32(b_y1->s)
            };
            summs0 = vaddq_f32(summs0, vld1q_f32(summs_t));

            const uint8x16_t m4b = vdupq_n_u8(0x0F);

            const uint8x16_t v0_0 = vld1q_u8(b_x0->qs);
            const uint8x16_t v0_1 = vld1q_u8(b_x1->qs);

            // 4-bit -> 8-bit
            const int8x16_t x0_l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
            const int8x16_t x0_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
            const int8x16_t x1_l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
            const int8x16_t x1_h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            // mmla into int32x4_t
            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        sumv2 = vaddq_f32(sumv2, summs0);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

    // TODO: add WASM SIMD
#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs = 0;

    for (; ib + 1 < nb; ib += 2) {
        const block_q4_1 * restrict x0 = &x[ib + 0];
        const block_q4_1 * restrict x1 = &x[ib + 1];
        const block_q8_1 * restrict y0 = &y[ib + 0];
        const block_q8_1 * restrict y1 = &y[ib + 1];

        summs += GGML_FP16_TO_FP32(x0->m) * GGML_FP16_TO_FP32(y0->s) + GGML_FP16_TO_FP32(x1->m) * GGML_FP16_TO_FP32(y1->s);

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        // dot product into int32x4_t
        const int32x4_t p_0 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_0l, v1_0l), v0_0h, v1_0h);
        const int32x4_t p_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), v0_1l, v1_1l), v0_1h, v1_1h);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs;
#elif defined(__AVX2__) || defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0;

    // Main loop
    for (; ib < nb; ++ib) {
        const float d0 = GGML_FP16_TO_FP32(x[ib].d);
        const float d1 = GGML_FP16_TO_FP32(y[ib].d);

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        const __m256 d0v = _mm256_set1_ps( d0 );
        const __m256 d1v = _mm256_set1_ps( d1 );

        // Compute combined scales
        const __m256 d0d1 = _mm256_mul_ps( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        const __m256i qy = _mm256_loadu_si256( (const __m256i *)y[ib].qs );

        const __m256 xy = mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
#if defined(__AVX2__)
        acc = _mm256_fmadd_ps( d0d1, xy, acc );
#else
        acc = _mm256_add_ps( _mm256_mul_ps( d0d1, xy ), acc );
#endif
    }

    sumf = hsum_float_8(acc) + summs;
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    for (; ib < nb; ++ib) {
        // load elements
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs+16, vl);

        // mask and store lower part of x, and then upper part
        vuint8mf2_t x_a = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_l = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vint8mf2_t v0 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t v1 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[ib].m));
        vector float vys = {GGML_FP16_TO_FP32(y[ib].s), 0.0f, 0.0f, 0.0f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector unsigned char q4x0 = (vector unsigned char)vec_and(qxs, lowMask);
        vector unsigned char q4x1 = (vector unsigned char)vec_sr(qxs, v4);

        vector signed int vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q4x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q4x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    float summs = 0;

    // Main loop
    for (; ib < nb; ++ib) {
        const float d0 = GGML_FP16_TO_FP32(x[ib].d);
        const float d1 = GGML_FP16_TO_FP32(y[ib].d);

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        const __m256 d0v = __lasx_xvreplfr2vr_s( d0 );
        const __m256 d1v = __lasx_xvreplfr2vr_s( d1 );

        // Compute combined scales
        const __m256 d0d1 = __lasx_xvfmul_s( d0v, d1v );

        // Load 16 bytes, and unpack 4 bit fields into bytes, making 32 bytes
        const __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        const __m256i qy = __lasx_xvld( (const __m256i *)y[ib].qs, 0);

        const __m256 xy = mul_sum_us8_pairs_float(qx, qy);

        // Accumulate d0*d1*x*y
        acc = __lasx_xvfmadd_s( d0d1, xy, acc );
    }

    sumf = hsum_float_8(acc) + summs;
#endif
    for (; ib < nb; ++ib) {
        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const int v0 = (x[ib].qs[j] & 0x0F);
            const int v1 = (x[ib].qs[j] >>   4);

            sumi0 += (v0 * y[ib].qs[j]);
            sumi1 += (v1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q5_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_0 * restrict x0 = &x[ib];
        const block_q5_0 * restrict x1 = &x[ib + 1];
        const block_q8_0 * restrict y0 = &y[ib];
        const block_q8_0 * restrict y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        // extract the 5th bit via lookup table ((!b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_1[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_1[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_1[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_1[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_1[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_1[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_1[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_1[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const int8x16_t v0_0lf = vsubq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vsubq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vsubq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vsubq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    uint32_t qh;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    for (; ib < nb; ++ib) {
        const block_q5_0 * restrict x0 = &x[ib];
        const block_q8_0 * restrict y0 = &y[ib];

        const v128_t m4b  = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_1[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_1[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_1[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_1[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit and sub 16 (equivalent to sub 0x10 when bit is zero)
        const v128_t v0lf = wasm_i8x16_sub(v0l, qhl);
        const v128_t v0hf = wasm_i8x16_sub(v0h, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_add(sumv, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(
                        wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d))));
    }

    sumf = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
           wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char)0xF0));
        qx = _mm256_or_si256(qx, bxhi);

        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8((char)0xF0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));

        __m256i bx_0 = bytes_from_nibbles_32(x[ib].qs);
        const __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_andnot_si128(bxhil, mask);
        bxhih = _mm_andnot_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(bx_0, by_0);

        /* Multiply q with scale and accumulate */
        acc = _mm256_add_ps(_mm256_mul_ps(d, q), acc);
    }

    sumf = hsum_float_8(acc);
#elif defined(__riscv_v_intrinsic)
    uint32_t qh;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    // These temporary registers are for masking and shift operations
    vuint32m2_t vt_1 = __riscv_vid_v_u32m2(vl);
    vuint32m2_t vt_2 = __riscv_vsll_vv_u32m2(__riscv_vmv_v_x_u32m2(1, vl), vt_1, vl);

    vuint32m2_t vt_3 = __riscv_vsll_vx_u32m2(vt_2, 16, vl);
    vuint32m2_t vt_4 = __riscv_vadd_vx_u32m2(vt_1, 12, vl);

    for (; ib < nb; ++ib) {
        memcpy(&qh, x[ib].qh, sizeof(uint32_t));

        // ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
        vuint32m2_t xha_0 = __riscv_vand_vx_u32m2(vt_2, qh, vl);
        vuint32m2_t xhr_0 = __riscv_vsrl_vv_u32m2(xha_0, vt_1, vl);
        vuint32m2_t xhl_0 = __riscv_vsll_vx_u32m2(xhr_0, 4, vl);

        // ((qh & (1u << (j + 16))) >> (j + 12));
        vuint32m2_t xha_1 = __riscv_vand_vx_u32m2(vt_3, qh, vl);
        vuint32m2_t xhl_1 = __riscv_vsrl_vv_u32m2(xha_1, vt_4, vl);

        // narrowing
        vuint16m1_t xhc_0 = __riscv_vncvt_x_x_w_u16m1(xhl_0, vl);
        vuint8mf2_t xh_0 = __riscv_vncvt_x_x_w_u8mf2(xhc_0, vl);

        vuint16m1_t xhc_1 = __riscv_vncvt_x_x_w_u16m1(xhl_1, vl);
        vuint8mf2_t xh_1 = __riscv_vncvt_x_x_w_u8mf2(xhc_1, vl);

        // load
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs+16, vl);

        vuint8mf2_t x_at = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_lt = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vuint8mf2_t x_a = __riscv_vor_vv_u8mf2(x_at, xh_0, vl);
        vuint8mf2_t x_l = __riscv_vor_vv_u8mf2(x_lt, xh_1, vl);

        vint8mf2_t x_ai = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t x_li = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint8mf2_t v0 = __riscv_vsub_vx_i8mf2(x_ai, 16, vl);
        vint8mf2_t v1 = __riscv_vsub_vx_i8mf2(x_li, 16, vl);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d)) * sumi;
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector unsigned char v4 = vec_splats((unsigned char)4);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed long long aux64x2_0 = {(uint64_t)(table_b2b_1[x[ib].qh[0]]), (uint64_t)(table_b2b_1[x[ib].qh[1]])};
        vector signed long long aux64x2_1 = {(uint64_t)(table_b2b_1[x[ib].qh[2]]), (uint64_t)(table_b2b_1[x[ib].qh[3]])};

        vector signed char qh0 = (vector signed char)aux64x2_0;
        vector signed char qh1 = (vector signed char)aux64x2_1;

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);

        vector signed char q5x0 = vec_sub(vec_and (qxs, lowMask), qh0);
        vector signed char q5x1 = vec_sub(vec_sr(qxs, v4), qh1);

        vector signed char q8y0 = vec_xl(  0, y[ib].qs);
        vector signed char q8y1 = vec_xl( 16, y[ib].qs);

        vector signed short qv0 = vec_add(vec_mule(q5x0, q8y0), vec_mulo(q5x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q5x1, q8y1), vec_mulo(q5x1, q8y1));

        qv0 = vec_add(qv0, qv1);

        vector signed int vsumi0 = vec_add(vec_unpackh(qv0), vec_unpackl(qv0));

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        /* Compute combined scale for the block */
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d)); //FIXME

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = __lasx_xvandn_v(bxhi, __lasx_xvreplgr2vr_b((char)0xF0));
        qx = __lasx_xvor_v(qx, bxhi);

        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        /* Multiply q with scale and accumulate */
        acc = __lasx_xvfmadd_s(d, q, acc);
    }

    sumf = hsum_float_8(acc);
#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh & (1u << (j + 0 ))) >> (j + 0 )) << 4;
            const uint8_t xh_1 = ((qh & (1u << (j + 16))) >> (j + 12));

            const int32_t x0 = (int8_t)(((x[ib].qs[j] & 0x0F) | xh_0) - 16);
            const int32_t x1 = (int8_t)(((x[ib].qs[j] >>   4) | xh_1) - 16);

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d)) * sumi;
    }

    *s = sumf;
}

void ggml_vec_dot_q5_1_q8_1(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_1;
    const int nb = n / qk;

    int ib = 0;
    float sumf = 0;

    assert(n % qk == 0);
    assert(qk == QK5_1);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_1 * restrict x = vx;
    const block_q8_1 * restrict y = vy;

#if defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    float summs0 = 0.0f;
    float summs1 = 0.0f;

    uint32_t qh0;
    uint32_t qh1;

    uint64_t tmp0[4];
    uint64_t tmp1[4];

    for (; ib + 1 < nb; ib += 2) {
        const block_q5_1 * restrict x0 = &x[ib];
        const block_q5_1 * restrict x1 = &x[ib + 1];
        const block_q8_1 * restrict y0 = &y[ib];
        const block_q8_1 * restrict y1 = &y[ib + 1];

        const uint8x16_t m4b = vdupq_n_u8(0x0F);

        summs0 += GGML_FP16_TO_FP32(x0->m) * GGML_FP16_TO_FP32(y0->s);
        summs1 += GGML_FP16_TO_FP32(x1->m) * GGML_FP16_TO_FP32(y1->s);

        // extract the 5th bit via lookup table ((b) << 4)
        memcpy(&qh0, x0->qh, sizeof(qh0));
        memcpy(&qh1, x1->qh, sizeof(qh1));

        tmp0[0] = table_b2b_0[(qh0 >>  0) & 0xFF];
        tmp0[1] = table_b2b_0[(qh0 >>  8) & 0xFF];
        tmp0[2] = table_b2b_0[(qh0 >> 16) & 0xFF];
        tmp0[3] = table_b2b_0[(qh0 >> 24)       ];

        tmp1[0] = table_b2b_0[(qh1 >>  0) & 0xFF];
        tmp1[1] = table_b2b_0[(qh1 >>  8) & 0xFF];
        tmp1[2] = table_b2b_0[(qh1 >> 16) & 0xFF];
        tmp1[3] = table_b2b_0[(qh1 >> 24)       ];

        const int8x16_t qhl0 = vld1q_s8((const int8_t *)(tmp0 + 0));
        const int8x16_t qhh0 = vld1q_s8((const int8_t *)(tmp0 + 2));
        const int8x16_t qhl1 = vld1q_s8((const int8_t *)(tmp1 + 0));
        const int8x16_t qhh1 = vld1q_s8((const int8_t *)(tmp1 + 2));

        const uint8x16_t v0_0 = vld1q_u8(x0->qs);
        const uint8x16_t v0_1 = vld1q_u8(x1->qs);

        // 4-bit -> 8-bit
        const int8x16_t v0_0l = vreinterpretq_s8_u8(vandq_u8  (v0_0, m4b));
        const int8x16_t v0_0h = vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4));
        const int8x16_t v0_1l = vreinterpretq_s8_u8(vandq_u8  (v0_1, m4b));
        const int8x16_t v0_1h = vreinterpretq_s8_u8(vshrq_n_u8(v0_1, 4));

        // add high bit
        const int8x16_t v0_0lf = vorrq_s8(v0_0l, qhl0);
        const int8x16_t v0_0hf = vorrq_s8(v0_0h, qhh0);
        const int8x16_t v0_1lf = vorrq_s8(v0_1l, qhl1);
        const int8x16_t v0_1hf = vorrq_s8(v0_1h, qhh1);

        // load y
        const int8x16_t v1_0l = vld1q_s8(y0->qs);
        const int8x16_t v1_0h = vld1q_s8(y0->qs + 16);
        const int8x16_t v1_1l = vld1q_s8(y1->qs);
        const int8x16_t v1_1h = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0lf, v1_0l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_0hf, v1_0h))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1lf, v1_1l),
                        ggml_vdotq_s32(vdupq_n_s32(0), v0_1hf, v1_1h))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + summs0 + summs1;
#elif defined(__wasm_simd128__)
    v128_t sumv = wasm_f32x4_splat(0.0f);

    float summs = 0.0f;

    uint32_t qh;
    uint64_t tmp[4];

    // TODO: check if unrolling this is better
    for (; ib < nb; ++ib) {
        const block_q5_1 * restrict x0 = &x[ib];
        const block_q8_1 * restrict y0 = &y[ib];

        summs += GGML_FP16_TO_FP32(x0->m) * GGML_FP16_TO_FP32(y0->s);

        const v128_t m4b = wasm_i8x16_splat(0x0F);

        // extract the 5th bit
        memcpy(&qh, x0->qh, sizeof(qh));

        tmp[0] = table_b2b_0[(qh >>  0) & 0xFF];
        tmp[1] = table_b2b_0[(qh >>  8) & 0xFF];
        tmp[2] = table_b2b_0[(qh >> 16) & 0xFF];
        tmp[3] = table_b2b_0[(qh >> 24)       ];

        const v128_t qhl = wasm_v128_load(tmp + 0);
        const v128_t qhh = wasm_v128_load(tmp + 2);

        const v128_t v0 = wasm_v128_load(x0->qs);

        // 4-bit -> 8-bit
        const v128_t v0l = wasm_v128_and (v0, m4b);
        const v128_t v0h = wasm_u8x16_shr(v0, 4);

        // add high bit
        const v128_t v0lf = wasm_v128_or(v0l, qhl);
        const v128_t v0hf = wasm_v128_or(v0h, qhh);

        // load y
        const v128_t v1l = wasm_v128_load(y0->qs);
        const v128_t v1h = wasm_v128_load(y0->qs + 16);

        // int8x16 -> int16x8
        const v128_t v0lfl = wasm_i16x8_extend_low_i8x16 (v0lf);
        const v128_t v0lfh = wasm_i16x8_extend_high_i8x16(v0lf);
        const v128_t v0hfl = wasm_i16x8_extend_low_i8x16 (v0hf);
        const v128_t v0hfh = wasm_i16x8_extend_high_i8x16(v0hf);

        const v128_t v1ll = wasm_i16x8_extend_low_i8x16 (v1l);
        const v128_t v1lh = wasm_i16x8_extend_high_i8x16(v1l);
        const v128_t v1hl = wasm_i16x8_extend_low_i8x16 (v1h);
        const v128_t v1hh = wasm_i16x8_extend_high_i8x16(v1h);

        // dot product
        sumv = wasm_f32x4_add(sumv,
                wasm_f32x4_mul(wasm_f32x4_convert_i32x4(wasm_i32x4_add(
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0lfl, v1ll),
                                           wasm_i32x4_dot_i16x8(v0lfh, v1lh)),
                            wasm_i32x4_add(wasm_i32x4_dot_i16x8(v0hfl, v1hl),
                                           wasm_i32x4_dot_i16x8(v0hfh, v1hh)))),
                    wasm_f32x4_splat(GGML_FP16_TO_FP32(x0->d) * GGML_FP16_TO_FP32(y0->d))));
    }

    sumf = wasm_f32x4_extract_lane(sumv, 0) + wasm_f32x4_extract_lane(sumv, 1) +
           wasm_f32x4_extract_lane(sumv, 2) + wasm_f32x4_extract_lane(sumv, 3) + summs;
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d));

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx = _mm256_or_si256(qx, bxhi);

        const __m256 dy = _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib].d));
        const __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_us8_pairs_float(qx, qy);

        acc = _mm256_fmadd_ps(q, _mm256_mul_ps(dx, dy), acc);
    }

    sumf = hsum_float_8(acc) + summs;
#elif defined(__AVX__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();
    __m128i mask = _mm_set1_epi8(0x10);

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d));

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        __m256i bx_0 = bytes_from_nibbles_32(x[ib].qs);
        const __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        __m128i bxhil = _mm256_castsi256_si128(bxhi);
        __m128i bxhih = _mm256_extractf128_si256(bxhi, 1);
        bxhil = _mm_and_si128(bxhil, mask);
        bxhih = _mm_and_si128(bxhih, mask);
        __m128i bxl = _mm256_castsi256_si128(bx_0);
        __m128i bxh = _mm256_extractf128_si256(bx_0, 1);
        bxl = _mm_or_si128(bxl, bxhil);
        bxh = _mm_or_si128(bxh, bxhih);
        bx_0 = MM256_SET_M128I(bxh, bxl);

        const __m256 dy = _mm256_set1_ps(GGML_FP16_TO_FP32(y[ib].d));
        const __m256i by_0 = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_us8_pairs_float(bx_0, by_0);

        acc = _mm256_add_ps(_mm256_mul_ps(q, _mm256_mul_ps(dx, dy)), acc);
    }

    sumf = hsum_float_8(acc) + summs;
#elif defined(__riscv_v_intrinsic)
    uint32_t qh;

    size_t vl = __riscv_vsetvl_e8m1(qk/2);

    // temporary registers for shift operations
    vuint32m2_t vt_1 = __riscv_vid_v_u32m2(vl);
    vuint32m2_t vt_2 = __riscv_vadd_vx_u32m2(vt_1, 12, vl);

    for (; ib < nb; ++ib) {
        memcpy(&qh, x[ib].qh, sizeof(uint32_t));

        // load qh
        vuint32m2_t vqh = __riscv_vmv_v_x_u32m2(qh, vl);

        // ((qh >> (j +  0)) << 4) & 0x10;
        vuint32m2_t xhr_0 = __riscv_vsrl_vv_u32m2(vqh, vt_1, vl);
        vuint32m2_t xhl_0 = __riscv_vsll_vx_u32m2(xhr_0, 4, vl);
        vuint32m2_t xha_0 = __riscv_vand_vx_u32m2(xhl_0, 0x10, vl);

        // ((qh >> (j + 12))     ) & 0x10;
        vuint32m2_t xhr_1 = __riscv_vsrl_vv_u32m2(vqh, vt_2, vl);
        vuint32m2_t xha_1 = __riscv_vand_vx_u32m2(xhr_1, 0x10, vl);

        // narrowing
        vuint16m1_t xhc_0 = __riscv_vncvt_x_x_w_u16m1(xha_0, vl);
        vuint8mf2_t xh_0 = __riscv_vncvt_x_x_w_u8mf2(xhc_0, vl);

        vuint16m1_t xhc_1 = __riscv_vncvt_x_x_w_u16m1(xha_1, vl);
        vuint8mf2_t xh_1 = __riscv_vncvt_x_x_w_u8mf2(xhc_1, vl);

        // load
        vuint8mf2_t tx = __riscv_vle8_v_u8mf2(x[ib].qs, vl);

        vint8mf2_t y0 = __riscv_vle8_v_i8mf2(y[ib].qs, vl);
        vint8mf2_t y1 = __riscv_vle8_v_i8mf2(y[ib].qs+16, vl);

        vuint8mf2_t x_at = __riscv_vand_vx_u8mf2(tx, 0x0F, vl);
        vuint8mf2_t x_lt = __riscv_vsrl_vx_u8mf2(tx, 0x04, vl);

        vuint8mf2_t x_a = __riscv_vor_vv_u8mf2(x_at, xh_0, vl);
        vuint8mf2_t x_l = __riscv_vor_vv_u8mf2(x_lt, xh_1, vl);

        vint8mf2_t v0 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_a);
        vint8mf2_t v1 = __riscv_vreinterpret_v_u8mf2_i8mf2(x_l);

        vint16m1_t vec_mul1 = __riscv_vwmul_vv_i16m1(v0, y0, vl);
        vint16m1_t vec_mul2 = __riscv_vwmul_vv_i16m1(v1, y1, vl);

        vint32m1_t vec_zero = __riscv_vmv_v_x_i32m1(0, vl);

        vint32m1_t vs1 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul1, vec_zero, vl);
        vint32m1_t vs2 = __riscv_vwredsum_vs_i16m1_i32m1(vec_mul2, vs1, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(vs2);

        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[ib].m));
        vector float vys = {GGML_FP16_TO_FP32(y[ib].s), 0.f, 0.f, 0.f};
        vsumf0 = vec_madd(vxmin, vys, vsumf0);

        vector unsigned long long aux64x2_0 = {(uint64_t)(table_b2b_0[x[ib].qh[0]]), (uint64_t)(table_b2b_0[x[ib].qh[1]])};
        vector unsigned long long aux64x2_1 = {(uint64_t)(table_b2b_0[x[ib].qh[2]]), (uint64_t)(table_b2b_0[x[ib].qh[3]])};

        vector signed char qh0 = (vector signed char)aux64x2_0;
        vector signed char qh1 = (vector signed char)aux64x2_1;

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);

        vector unsigned char q5x0 = (vector unsigned char)vec_or(vec_and(qxs, lowMask), qh0);
        vector unsigned char q5x1 = (vector unsigned char)vec_or(vec_sr(qxs, v4), qh1);

        vector signed char q8y0 = vec_xl(  0, y[ib].qs);
        vector signed char q8y1 = vec_xl( 16, y[ib].qs);

        vector signed int vsumi0 = v0;

        vsumi0 = vec_msum(q8y0, q5x0, vsumi0);
        vsumi0 = vec_msum(q8y1, q5x1, vsumi0);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    float summs = 0.0f;

    // Main loop
    for (; ib < nb; ++ib) {
        const __m256 dx = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d));

        summs += GGML_FP16_TO_FP32(x[ib].m) * GGML_FP16_TO_FP32(y[ib].s);

        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = __lasx_xvand_v(bxhi, __lasx_xvreplgr2vr_b(0x10));
        qx = __lasx_xvor_v(qx, bxhi);

        const __m256 dy = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib].d));
        const __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_us8_pairs_float(qx, qy);

        acc = __lasx_xvfmadd_s(q, __lasx_xvfmul_s(dx, dy), acc);
    }

    sumf = hsum_float_8(acc) + summs;
#endif
    for (; ib < nb; ++ib) {
        uint32_t qh;
        memcpy(&qh, x[ib].qh, sizeof(qh));

        int sumi0 = 0;
        int sumi1 = 0;

        for (int j = 0; j < qk/2; ++j) {
            const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;

            const int32_t x0 = (x[ib].qs[j] & 0xF) | xh_0;
            const int32_t x1 = (x[ib].qs[j] >>  4) | xh_1;

            sumi0 += (x0 * y[ib].qs[j]);
            sumi1 += (x1 * y[ib].qs[j + qk/2]);
        }

        int sumi = sumi0 + sumi1;
        sumf += (GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d))*sumi + GGML_FP16_TO_FP32(x[ib].m)*GGML_FP16_TO_FP32(y[ib].s);
    }

    *s = sumf;
}

void ggml_vec_dot_q8_0_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    const int qk = QK8_0;
    const int nb = n / qk;

    assert(n % qk == 0);
#if defined(__ARM_FEATURE_MATMUL_INT8)
    assert((nrc == 2) || (nrc == 1));
#else
    assert(nrc == 1);
#endif
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q8_0 * restrict x = vx;
    const block_q8_0 * restrict y = vy;

#if defined(__ARM_FEATURE_MATMUL_INT8)
    if (nrc == 2) {
        const block_q8_0 * restrict vx0 = vx;
        const block_q8_0 * restrict vx1 = (const block_q8_0 *) ((const uint8_t*)vx + bx);
        const block_q8_0 * restrict vy0 = vy;
        const block_q8_0 * restrict vy1 = (const block_q8_0 *) ((const uint8_t*)vy + by);

        float32x4_t sumv0 = vdupq_n_f32(0.0f);

        for (int i = 0; i < nb; i++) {
            const block_q8_0 * restrict b_x0 = &vx0[i];
            const block_q8_0 * restrict b_y0 = &vy0[i];

            const block_q8_0 * restrict b_x1 = &vx1[i];
            const block_q8_0 * restrict b_y1 = &vy1[i];

            const int8x16_t x0_l = vld1q_s8(b_x0->qs);
            const int8x16_t x0_h = vld1q_s8(b_x0->qs + 16);
            const int8x16_t x1_l = vld1q_s8(b_x1->qs);
            const int8x16_t x1_h = vld1q_s8(b_x1->qs + 16);

            // load y
            const int8x16_t y0_l = vld1q_s8(b_y0->qs);
            const int8x16_t y0_h = vld1q_s8(b_y0->qs + 16);
            const int8x16_t y1_l = vld1q_s8(b_y1->qs);
            const int8x16_t y1_h = vld1q_s8(b_y1->qs + 16);

            float32_t _scale[4] = {
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x0->d)*GGML_FP16_TO_FP32(b_y1->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y0->d),
                GGML_FP16_TO_FP32(b_x1->d)*GGML_FP16_TO_FP32(b_y1->d)
            };
            float32x4_t scale = vld1q_f32(_scale);

            int8x16_t l0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));
            int8x16_t l1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_l), vreinterpretq_s64_s8(x1_l)));

            int8x16_t l2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));
            int8x16_t l3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(x0_h), vreinterpretq_s64_s8(x1_h)));

            int8x16_t r0 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));
            int8x16_t r1 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_l), vreinterpretq_s64_s8(y1_l)));

            int8x16_t r2 = vreinterpretq_s8_s64(vzip1q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));
            int8x16_t r3 = vreinterpretq_s8_s64(vzip2q_s64(vreinterpretq_s64_s8(y0_h), vreinterpretq_s64_s8(y1_h)));

            sumv0 = vmlaq_f32(sumv0,(vcvtq_f32_s32(vmmlaq_s32((vmmlaq_s32((vmmlaq_s32((vmmlaq_s32(vdupq_n_s32(0), l0, r0)),
                                                l1, r1)), l2, r2)), l3, r3))), scale);
        }

        float32x4_t sumv1 = vextq_f32 (sumv0, sumv0, 2);
        float32x4_t sumv2 = vzip1q_f32(sumv0, sumv1);

        vst1_f32(s,      vget_low_f32 (sumv2));
        vst1_f32(s + bs, vget_high_f32(sumv2));

        return;
    }
#endif

    int ib = 0;
    float sumf = 0;

#if defined(__ARM_FEATURE_SVE)
    svfloat32_t sumv0 = svdup_n_f32(0.0f);
    svfloat32_t sumv1 = svdup_n_f32(0.0f);

    const int vector_length = ggml_cpu_get_sve_cnt()*8;

    //VLA Implemenation for SVE
    switch (vector_length) {
        case 128:
            {
                // predicate for activating lanes for 16 Int8 elements
                const svbool_t ph16 = svptrue_pat_b8 (SV_VL16);
                const svbool_t pl16 = svptrue_pat_b32(SV_VL4);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * restrict x0 = &x[ib + 0];
                    const block_q8_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0_0 = svld1_s8(ph16, x0->qs);
                    const svint8_t qx0_1 = svld1_s8(ph16, x0->qs+16);
                    const svint8_t qx1_0 = svld1_s8(ph16, x1->qs);
                    const svint8_t qx1_1 = svld1_s8(ph16, x1->qs+16);

                    // load y
                    const svint8_t qy0_0 = svld1_s8(ph16, y0->qs);
                    const svint8_t qy0_1 = svld1_s8(ph16, y0->qs+16);
                    const svint8_t qy1_0 = svld1_s8(ph16, y1->qs);
                    const svint8_t qy1_1 = svld1_s8(ph16, y1->qs+16);

                    sumv0 = svmla_n_f32_x(pl16, sumv0, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx0_0, qy0_0),
                                    svdot_s32(svdup_n_s32(0), qx0_1, qy0_1))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(pl16, sumv1, svcvt_f32_s32_x(pl16, svadd_x(pl16,
                                    svdot_s32(svdup_n_s32(0), qx1_0, qy1_0),
                                    svdot_s32(svdup_n_s32(0), qx1_1, qy1_1))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(pl16, svadd_f32_x(pl16, sumv0, sumv1));
            } break;
        case 256:
            {
                //printf("sve256");
                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * restrict x0 = &x[ib + 0];
                    const block_q8_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    // load x
                    const svint8_t qx0 = svld1_s8(svptrue_b8(), x0->qs);
                    const svint8_t qx1 = svld1_s8(svptrue_b8(), x1->qs);

                    // load y
                    const svint8_t qy0 = svld1_s8(svptrue_b8(), y0->qs);
                    const svint8_t qy1 = svld1_s8(svptrue_b8(), y1->qs);

                    sumv0 = svmla_n_f32_x(svptrue_b32(), sumv0, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx0, qy0)), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));
                    sumv1 = svmla_n_f32_x(svptrue_b32(), sumv1, svcvt_f32_s32_x(svptrue_b32(),
                                svdot_s32(svdup_n_s32(0), qx1, qy1)), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
                }

                sumf = svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), sumv0, sumv1));
            } break;
        case 512:
            {
                // predicate for activating high 256 bit
                const svbool_t ph32 = svptrue_pat_b8(SV_VL32);
                // predicate for activating low 256 bit
                const svbool_t pl32 = svnot_b_z(svptrue_b8(), ph32);

                // predicate for activating high lanes for 8 float32 elements
                const svbool_t ph8 = svptrue_pat_b32(SV_VL8);
                // predicate for activating low lanes for 8 float32 elements
                const svbool_t pl8 = svnot_b_z(svptrue_b32(), ph8);

                svfloat32_t sumv00 = svdup_n_f32(0.0f);

                for (; ib + 1 < nb; ib += 2) {
                    const block_q8_0 * restrict x0 = &x[ib + 0];
                    const block_q8_0 * restrict x1 = &x[ib + 1];
                    const block_q8_0 * restrict y0 = &y[ib + 0];
                    const block_q8_0 * restrict y1 = &y[ib + 1];

                    //load 32 int8_t in first half of vector and put another 32 int8_t in second vector lower bits
                    // and add them to make one 64 element vector
                    // load x
                    const svint8_t qx_32 = svld1_s8(ph32, x0->qs);
                          svint8_t qx_64 = svld1_s8(pl32, x0->qs + 2);

                    qx_64 = svadd_s8_x(svptrue_b8(), qx_32, qx_64);

                    // load y
                    const svint8_t qy_32 = svld1_s8(ph32, y0->qs);
                          svint8_t qy_64 = svld1_s8(pl32, y0->qs + 2);

                    qy_64 = svadd_s8_x(svptrue_b8(), qy_32, qy_64);

                    // scale creation
                    const float32_t deq1 = GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d);
                    const float32_t deq2 = GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d);

                    // duplicate deq1 in first half of vector and deq2 in second half of vector
                    const svfloat32_t temp = svdup_f32_m(svdup_f32_z(ph8, deq1), pl8, deq2);

                    const svfloat32_t sumvt = svcvt_f32_s32_x(svptrue_b32(), svdot_s32(svdup_n_s32(0), qx_64, qy_64));

                    sumv00 = svmla_f32_m(svptrue_b32(), sumv00, sumvt, temp);
                }

                sumf = svaddv_f32(svptrue_b32(), sumv00);
                break;
            }
        default:
            assert(false && "Unsupported vector length");
            break;
    }
#elif defined(__ARM_NEON)
    float32x4_t sumv0 = vdupq_n_f32(0.0f);
    float32x4_t sumv1 = vdupq_n_f32(0.0f);

    for (; ib + 1 < nb; ib += 2) {
        const block_q8_0 * restrict x0 = &x[ib + 0];
        const block_q8_0 * restrict x1 = &x[ib + 1];
        const block_q8_0 * restrict y0 = &y[ib + 0];
        const block_q8_0 * restrict y1 = &y[ib + 1];

        const int8x16_t x0_0 = vld1q_s8(x0->qs);
        const int8x16_t x0_1 = vld1q_s8(x0->qs + 16);
        const int8x16_t x1_0 = vld1q_s8(x1->qs);
        const int8x16_t x1_1 = vld1q_s8(x1->qs + 16);

        // load y
        const int8x16_t y0_0 = vld1q_s8(y0->qs);
        const int8x16_t y0_1 = vld1q_s8(y0->qs + 16);
        const int8x16_t y1_0 = vld1q_s8(y1->qs);
        const int8x16_t y1_1 = vld1q_s8(y1->qs + 16);

        sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_0, y0_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x0_1, y0_1))), GGML_FP16_TO_FP32(x0->d)*GGML_FP16_TO_FP32(y0->d));

        sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_0, y1_0),
                        ggml_vdotq_s32(vdupq_n_s32(0), x1_1, y1_1))), GGML_FP16_TO_FP32(x1->d)*GGML_FP16_TO_FP32(y1->d));
    }

    sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
#elif defined(__AVX2__)
    // Initialize accumulator with zeros
    __m256 acc = _mm256_setzero_ps();

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = _mm256_set1_ps(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = _mm256_loadu_si256((const __m256i *)x[ib].qs);
        __m256i qy = _mm256_loadu_si256((const __m256i *)y[ib].qs);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = _mm256_fmadd_ps( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#elif defined(__AVX__)
    __m256 accum = _mm256_setzero_ps();

    for (; ib + 1 < nb; ib += 2) {
        const __m128i qx_1_0 = _mm_loadu_si128((const __m128i *)x[ib].qs);
        const __m128i qx_1_1 = _mm_loadu_si128((const __m128i *)x[ib].qs + 1);
        const __m128i qx_2_0 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i qx_2_1 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs + 1);
        const __m128i qy_1_0 = _mm_loadu_si128((const __m128i *)y[ib].qs);
        const __m128i qy_1_1 = _mm_loadu_si128((const __m128i *)y[ib].qs + 1);
        const __m128i qy_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i qy_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m256 p = mul_sum_i8_quad_float(qx_1_0, qx_1_1, qx_2_0, qx_2_1, qy_1_0, qy_1_1, qy_2_0, qy_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);
#elif defined(__riscv_v_intrinsic)
    size_t vl = __riscv_vsetvl_e8m1(qk);

    for (; ib < nb; ++ib) {
        // load elements
        vint8m1_t bx_0 = __riscv_vle8_v_i8m1(x[ib].qs, vl);
        vint8m1_t by_0 = __riscv_vle8_v_i8m1(y[ib].qs, vl);

        vint16m2_t vw_mul = __riscv_vwmul_vv_i16m2(bx_0, by_0, vl);

        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vl);
        vint32m1_t v_sum = __riscv_vwredsum_vs_i16m2_i32m1(vw_mul, v_zero, vl);

        int sumi = __riscv_vmv_x_s_i32m1_i32(v_sum);

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }
#elif defined(__POWER9_VECTOR__)
    const vector signed int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);

#pragma GCC unroll 8
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char q8x0 = vec_xl( 0, x[ib].qs);
        vector signed char q8x1 = vec_xl(16, x[ib].qs);
        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_mule(q8x0, q8y0);
        vector signed short qv1 = vec_mulo(q8x0, q8y0);
        vector signed short qv2 = vec_mule(q8x1, q8y1);
        vector signed short qv3 = vec_mulo(q8x1, q8y1);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);
        vsumi0 = vec_sum4s(qv2, vsumi0);
        vsumi1 = vec_sum4s(qv3, vsumi1);

        vsumi0 = vec_add(vsumi0, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)
    // Initialize accumulator with zeros
    __m256 acc = (__m256)__lasx_xvldi(0);

    // Main loop
    for (; ib < nb; ++ib) {
        // Compute combined scale for the block
        const __m256 d = __lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ib].d) * GGML_FP16_TO_FP32(y[ib].d));
        __m256i qx = __lasx_xvld((const __m256i *)x[ib].qs, 0);
        __m256i qy = __lasx_xvld((const __m256i *)y[ib].qs, 0);

        const __m256 q = mul_sum_i8_pairs_float(qx, qy);

        // Multiply q with scale and accumulate
        acc = __lasx_xvfmadd_s( d, q, acc );
    }

    sumf = hsum_float_8(acc);
#endif
    for (; ib < nb; ++ib) {
        int sumi = 0;

        for (int j = 0; j < qk; j++) {
            sumi += x[ib].qs[j]*y[ib].qs[j];
        }

        sumf += sumi*(GGML_FP16_TO_FP32(x[ib].d)*GGML_FP16_TO_FP32(y[ib].d));
    }

    *s = sumf;
}

void ggml_vec_dot_tq1_0_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq1_0 * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)
    float sumf = 0.0f;

    uint8_t k_shift[16] = {1, 1, 1, 1, 3, 3, 3, 3, 9, 9, 9, 9, 27, 27, 27, 27};

    const uint8x16_t shift = vld1q_u8(k_shift);

    for (int i = 0; i < nb; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t sumi0 = vdupq_n_s32(0);
        int32x4_t sumi1 = vdupq_n_s32(0);
#else
        int16x8_t sumi0 = vdupq_n_s16(0);
        int16x8_t sumi1 = vdupq_n_s16(0);
#endif

        // first 32 bytes of 5 elements
        {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + 0);
            uint8x16_t qx1 = vld1q_u8(x[i].qs + 16);
            uint8x16_t qx2 = vmulq_u8(qx0, vdupq_n_u8(3));
            uint8x16_t qx3 = vmulq_u8(qx1, vdupq_n_u8(3));
            uint8x16_t qx4 = vmulq_u8(qx0, vdupq_n_u8(9));
            uint8x16_t qx5 = vmulq_u8(qx1, vdupq_n_u8(9));
            uint8x16_t qx6 = vmulq_u8(qx0, vdupq_n_u8(27));
            uint8x16_t qx7 = vmulq_u8(qx1, vdupq_n_u8(27));
            uint8x16_t qx8 = vmulq_u8(qx0, vdupq_n_u8(81));
            uint8x16_t qx9 = vmulq_u8(qx1, vdupq_n_u8(81));

            // multiply by 3 and keep the 2 bits above 8 bits
            int8x16_t sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));
            int8x16_t sqx6 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx6, vshrq_n_u8(qx6, 1)), 6));
            int8x16_t sqx7 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx7, vshrq_n_u8(qx7, 1)), 6));
            int8x16_t sqx8 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx8, vshrq_n_u8(qx8, 1)), 6));
            int8x16_t sqx9 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx9, vshrq_n_u8(qx9, 1)), 6));

            const int8x16_t qy0 = vld1q_s8(y[i].qs +   0);
            const int8x16_t qy1 = vld1q_s8(y[i].qs +  16);
            const int8x16_t qy2 = vld1q_s8(y[i].qs +  32);
            const int8x16_t qy3 = vld1q_s8(y[i].qs +  48);
            const int8x16_t qy4 = vld1q_s8(y[i].qs +  64);
            const int8x16_t qy5 = vld1q_s8(y[i].qs +  80);
            const int8x16_t qy6 = vld1q_s8(y[i].qs +  96);
            const int8x16_t qy7 = vld1q_s8(y[i].qs + 112);
            const int8x16_t qy8 = vld1q_s8(y[i].qs + 128);
            const int8x16_t qy9 = vld1q_s8(y[i].qs + 144);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32(sumi1, sqx7, qy7);
            sumi0 = vdotq_s32(sumi0, sqx8, qy8);
            sumi1 = vdotq_s32(sumi1, sqx9, qy9);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx6), vget_low_s8(qy6));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx6), vget_high_s8(qy6));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx7), vget_low_s8(qy7));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx7), vget_high_s8(qy7));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx8), vget_low_s8(qy8));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx8), vget_high_s8(qy8));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx9), vget_low_s8(qy9));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx9), vget_high_s8(qy9));
#endif
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + 32);
            uint8x16_t qx1 = vmulq_u8(qx0, vdupq_n_u8(3));
            uint8x16_t qx2 = vmulq_u8(qx0, vdupq_n_u8(9));
            uint8x16_t qx3 = vmulq_u8(qx0, vdupq_n_u8(27));
            uint8x16_t qx4 = vmulq_u8(qx0, vdupq_n_u8(81));
            uint32_t qh;
            memcpy(&qh, x[i].qh, sizeof(qh)); // potentially unaligned
            uint8x16_t qx5 = vreinterpretq_u8_u32(vdupq_n_u32(qh));
            qx5 = vmulq_u8(qx5, shift);

            // multiply by 3 and keep the 2 bits above 8 bits
            int8x16_t sqx0 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx0, vshrq_n_u8(qx0, 1)), 6));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx1, vshrq_n_u8(qx1, 1)), 6));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx2, vshrq_n_u8(qx2, 1)), 6));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx3, vshrq_n_u8(qx3, 1)), 6));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx4, vshrq_n_u8(qx4, 1)), 6));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vshrq_n_u8(vhaddq_u8(qx5, vshrq_n_u8(qx5, 1)), 6));

            const int8x16_t qy0 = vld1q_s8(y[i].qs + 160);
            const int8x16_t qy1 = vld1q_s8(y[i].qs + 176);
            const int8x16_t qy2 = vld1q_s8(y[i].qs + 192);
            const int8x16_t qy3 = vld1q_s8(y[i].qs + 208);
            const int8x16_t qy4 = vld1q_s8(y[i].qs + 224);
            const int8x16_t qy5 = vld1q_s8(y[i].qs + 240);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
#endif
        }

        const int16x8_t ysum0 = vld1q_s16(y[i].bsums);
        const int16x8_t ysum1 = vld1q_s16(y[i].bsums + 8);

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

#if defined(__ARM_FEATURE_DOTPROD)
        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        sumf += d * (float) vaddvq_s32(sumi0);
#else
        sumi0 = vaddq_s16(sumi0, sumi1);
        sumi0 = vsubq_s16(sumi0, vaddq_s16(ysum0, ysum1));

        sumf += d * (float) vaddlvq_s16(sumi0);
#endif
    }

    *s = sumf;

#elif defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // 16-bit sums
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();

        // first 32 bytes of 5 elements
        {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs));
            // 8-bit multiplies with shifts, masks and adds
            __m256i qx1 = _mm256_add_epi8(qx0, _mm256_add_epi8(qx0, qx0)); // 1 * 3
            __m256i qx2 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx0, 3), _mm256_set1_epi8(-8)), qx0); // 1 * 9
            __m256i qx3 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx1, 3), _mm256_set1_epi8(-8)), qx1); // 3 * 9
            __m256i qx4 = _mm256_add_epi8(_mm256_and_si256(_mm256_slli_epi16(qx2, 3), _mm256_set1_epi8(-8)), qx2); // 9 * 9

            // TODO: can _mm256_mulhi_epu16 be faster even if 16-bits?

            // Cancel the +1 from avg so that it behaves like a halving add
            qx0 = _mm256_subs_epu8(qx0, _mm256_set1_epi8(1));
            qx1 = _mm256_subs_epu8(qx1, _mm256_set1_epi8(1));
            qx2 = _mm256_subs_epu8(qx2, _mm256_set1_epi8(1));
            qx3 = _mm256_subs_epu8(qx3, _mm256_set1_epi8(1));
            qx4 = _mm256_subs_epu8(qx4, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx0 = _mm256_avg_epu8(qx0, _mm256_avg_epu8(qx0, _mm256_setzero_si256()));
            qx1 = _mm256_avg_epu8(qx1, _mm256_avg_epu8(qx1, _mm256_setzero_si256()));
            qx2 = _mm256_avg_epu8(qx2, _mm256_avg_epu8(qx2, _mm256_setzero_si256()));
            qx3 = _mm256_avg_epu8(qx3, _mm256_avg_epu8(qx3, _mm256_setzero_si256()));
            qx4 = _mm256_avg_epu8(qx4, _mm256_avg_epu8(qx4, _mm256_setzero_si256()));
            qx0 = _mm256_and_si256(_mm256_srli_epi16(qx0, 6), _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(_mm256_srli_epi16(qx1, 6), _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(_mm256_srli_epi16(qx2, 6), _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(_mm256_srli_epi16(qx3, 6), _mm256_set1_epi8(3));
            qx4 = _mm256_and_si256(_mm256_srli_epi16(qx4, 6), _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs +   0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs +  96));
            const __m256i qy4 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 128));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);
            qx4 = _mm256_maddubs_epi16(qx4, qy4);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
            sumi2 = _mm256_add_epi16(sumi2, qx4);
        }

        // last 16 bytes of 5-element, along with the 4 bytes of 4 elements
        {
            __m128i qx0 = _mm_loadu_si128((const __m128i *) (x[i].qs + 32));
            uint32_t qh;
            memcpy(&qh, x[i].qh, sizeof(qh)); // potentially unaligned
            __m256i qx5_l = _mm256_cvtepu8_epi16(_mm_set1_epi32(qh));
            __m128i qx1 = _mm_add_epi8(qx0, _mm_add_epi8(qx0, qx0)); // 1 * 3
            __m128i qx2 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx0, 3), _mm_set1_epi8(-8)), qx0); // 1 * 9
            __m128i qx3 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx1, 3), _mm_set1_epi8(-8)), qx1); // 3 * 9
            __m128i qx4 = _mm_add_epi8(_mm_and_si128(_mm_slli_epi16(qx2, 3), _mm_set1_epi8(-8)), qx2); // 9 * 9
            __m256i qx01 = MM256_SET_M128I(qx1, qx0);
            __m256i qx23 = MM256_SET_M128I(qx3, qx2);

            // avx2 does not have 8-bit multiplies, so 16-bit it is.
            qx5_l = _mm256_mullo_epi16(qx5_l, _mm256_set_epi16(27, 27, 27, 27, 9, 9, 9, 9, 3, 3, 3, 3, 1, 1, 1, 1));
            qx5_l = _mm256_and_si256(qx5_l, _mm256_set1_epi16(0xFF));
            __m128i qx5 = _mm_packus_epi16(_mm256_castsi256_si128(qx5_l), _mm256_extracti128_si256(qx5_l, 1));

            __m256i qx45 = MM256_SET_M128I(qx5, qx4);

            // Cancel the +1 from avg so that it behaves like a halving add
            qx01 = _mm256_subs_epu8(qx01, _mm256_set1_epi8(1));
            qx23 = _mm256_subs_epu8(qx23, _mm256_set1_epi8(1));
            qx45 = _mm256_subs_epu8(qx45, _mm256_set1_epi8(1));
            // Multiply by 3 and get the top 2 bits
            qx01 = _mm256_avg_epu8(qx01, _mm256_avg_epu8(qx01, _mm256_setzero_si256()));
            qx23 = _mm256_avg_epu8(qx23, _mm256_avg_epu8(qx23, _mm256_setzero_si256()));
            qx45 = _mm256_avg_epu8(qx45, _mm256_avg_epu8(qx45, _mm256_setzero_si256()));
            qx01 = _mm256_and_si256(_mm256_srli_epi16(qx01, 6), _mm256_set1_epi8(3));
            qx23 = _mm256_and_si256(_mm256_srli_epi16(qx23, 6), _mm256_set1_epi8(3));
            qx45 = _mm256_and_si256(_mm256_srli_epi16(qx45, 6), _mm256_set1_epi8(3));

            const __m256i qy01 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 160));
            const __m256i qy23 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 192));
            const __m256i qy45 = _mm256_loadu_si256((const __m256i *) (y[i].qs + 224));

            qx01 = _mm256_maddubs_epi16(qx01, qy01);
            qx23 = _mm256_maddubs_epi16(qx23, qy23);
            qx45 = _mm256_maddubs_epi16(qx45, qy45);

            sumi0 = _mm256_add_epi16(sumi0, qx01);
            sumi1 = _mm256_add_epi16(sumi1, qx23);
            sumi2 = _mm256_add_epi16(sumi2, qx45);
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(x[i].d));

        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(sumi1, sumi2));
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#else
    const uint8_t pow3[6] = {1, 3, 9, 27, 81, 243};

    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int sum = 0;

        for (size_t j = 0; j < sizeof(x->qs) - sizeof(x->qs) % 32; j += 32) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 32; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*32 + m];
                }
            }
        }
        for (size_t j = sizeof(x->qs) - sizeof(x->qs) % 32; j < sizeof(x->qs); j += 16) {
            for (size_t l = 0; l < 5; ++l) {
                for (size_t m = 0; m < 16; ++m) {
                    uint8_t q = x[i].qs[j + m] * pow3[l];
                    uint16_t xi = ((uint16_t) q * 3) >> 8;
                    sum += (xi - 1) * y[i].qs[j*5 + l*16 + m];
                }
            }
        }

        for (size_t l = 0; l < 4; ++l) {
            for (size_t j = 0; j < sizeof(x->qh); ++j) {
                uint8_t q = x[i].qh[j] * pow3[l];
                uint16_t xi = ((uint16_t) q * 3) >> 8;
                sum += (xi - 1) * y[i].qs[sizeof(x->qs)*5 + l*sizeof(x->qh) + j];
            }
        }

        sumf += (float) sum * (GGML_FP16_TO_FP32(x[i].d) * y[i].d);
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_tq2_0_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_tq2_0 * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)
    float sumf = 0.0f;

    const uint8x16_t m3 = vdupq_n_u8(3);

    for (int i = 0; i < nb; ++i) {
#if defined(__ARM_FEATURE_DOTPROD)
        int32x4_t sumi0 = vdupq_n_s32(0);
        int32x4_t sumi1 = vdupq_n_s32(0);
#else
        int16x8_t sumi0 = vdupq_n_s16(0);
        int16x8_t sumi1 = vdupq_n_s16(0);
#endif

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            uint8x16_t qx0 = vld1q_u8(x[i].qs + j);
            uint8x16_t qx1 = vld1q_u8(x[i].qs + j + 16);
            uint8x16_t qx2 = vshrq_n_u8(qx0, 2);
            uint8x16_t qx3 = vshrq_n_u8(qx1, 2);
            uint8x16_t qx4 = vshrq_n_u8(qx0, 4);
            uint8x16_t qx5 = vshrq_n_u8(qx1, 4);
            uint8x16_t qx6 = vshrq_n_u8(qx0, 6);
            uint8x16_t qx7 = vshrq_n_u8(qx1, 6);

            int8x16_t sqx0 = vreinterpretq_s8_u8(vandq_u8(qx0, m3));
            int8x16_t sqx1 = vreinterpretq_s8_u8(vandq_u8(qx1, m3));
            int8x16_t sqx2 = vreinterpretq_s8_u8(vandq_u8(qx2, m3));
            int8x16_t sqx3 = vreinterpretq_s8_u8(vandq_u8(qx3, m3));
            int8x16_t sqx4 = vreinterpretq_s8_u8(vandq_u8(qx4, m3));
            int8x16_t sqx5 = vreinterpretq_s8_u8(vandq_u8(qx5, m3));
            int8x16_t sqx6 = vreinterpretq_s8_u8(vandq_u8(qx6, m3));
            int8x16_t sqx7 = vreinterpretq_s8_u8(vandq_u8(qx7, m3));

            const int8x16_t qy0 = vld1q_s8(y[i].qs + j*4 +   0);
            const int8x16_t qy1 = vld1q_s8(y[i].qs + j*4 +  16);
            const int8x16_t qy2 = vld1q_s8(y[i].qs + j*4 +  32);
            const int8x16_t qy3 = vld1q_s8(y[i].qs + j*4 +  48);
            const int8x16_t qy4 = vld1q_s8(y[i].qs + j*4 +  64);
            const int8x16_t qy5 = vld1q_s8(y[i].qs + j*4 +  80);
            const int8x16_t qy6 = vld1q_s8(y[i].qs + j*4 +  96);
            const int8x16_t qy7 = vld1q_s8(y[i].qs + j*4 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            sumi0 = vdotq_s32(sumi0, sqx0, qy0);
            sumi1 = vdotq_s32(sumi1, sqx1, qy1);
            sumi0 = vdotq_s32(sumi0, sqx2, qy2);
            sumi1 = vdotq_s32(sumi1, sqx3, qy3);
            sumi0 = vdotq_s32(sumi0, sqx4, qy4);
            sumi1 = vdotq_s32(sumi1, sqx5, qy5);
            sumi0 = vdotq_s32(sumi0, sqx6, qy6);
            sumi1 = vdotq_s32(sumi1, sqx7, qy7);
#else
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx0), vget_low_s8(qy0));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx0), vget_high_s8(qy0));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx1), vget_low_s8(qy1));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx1), vget_high_s8(qy1));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx2), vget_low_s8(qy2));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx2), vget_high_s8(qy2));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx3), vget_low_s8(qy3));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx3), vget_high_s8(qy3));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx4), vget_low_s8(qy4));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx4), vget_high_s8(qy4));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx5), vget_low_s8(qy5));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx5), vget_high_s8(qy5));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx6), vget_low_s8(qy6));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx6), vget_high_s8(qy6));
            sumi0 = vmlal_s8(sumi0, vget_low_s8(sqx7), vget_low_s8(qy7));
            sumi1 = vmlal_s8(sumi1, vget_high_s8(sqx7), vget_high_s8(qy7));
#endif
        }

        const int16x8_t ysum0 = vld1q_s16(y[i].bsums);
        const int16x8_t ysum1 = vld1q_s16(y[i].bsums + 8);

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

#if defined(__ARM_FEATURE_DOTPROD)
        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));

        sumf += d * (float) vaddvq_s32(sumi0);
#else
        sumi0 = vaddq_s16(sumi0, sumi1);
        sumi0 = vsubq_s16(sumi0, vaddq_s16(ysum0, ysum1));

        sumf += d * (float) vaddlvq_s16(sumi0);
#endif
    }

    *s = sumf;

#elif defined(__AVX2__)
    __m256 sumf = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        // 16-bit sums, because 256*127 still fits
        __m256i sumi0 = _mm256_setzero_si256();
        __m256i sumi1 = _mm256_setzero_si256();

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            __m256i qx0 = _mm256_loadu_si256((const __m256i *) (x[i].qs + j));
            __m256i qx1 = _mm256_srli_epi16(qx0, 2);
            __m256i qx2 = _mm256_srli_epi16(qx0, 4);
            __m256i qx3 = _mm256_srli_epi16(qx0, 6);

            // 0, 1, 2 (should not be 3)
            qx0 = _mm256_and_si256(qx0, _mm256_set1_epi8(3));
            qx1 = _mm256_and_si256(qx1, _mm256_set1_epi8(3));
            qx2 = _mm256_and_si256(qx2, _mm256_set1_epi8(3));
            qx3 = _mm256_and_si256(qx3, _mm256_set1_epi8(3));

            const __m256i qy0 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 +  0));
            const __m256i qy1 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 32));
            const __m256i qy2 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 64));
            const __m256i qy3 = _mm256_loadu_si256((const __m256i *) (y[i].qs + j*4 + 96));

            qx0 = _mm256_maddubs_epi16(qx0, qy0);
            qx1 = _mm256_maddubs_epi16(qx1, qy1);
            qx2 = _mm256_maddubs_epi16(qx2, qy2);
            qx3 = _mm256_maddubs_epi16(qx3, qy3);

            sumi0 = _mm256_add_epi16(sumi0, _mm256_add_epi16(qx0, qx1));
            sumi1 = _mm256_add_epi16(sumi1, _mm256_add_epi16(qx2, qx3));
        }

        const __m256i ysum = _mm256_loadu_si256((const __m256i *) y[i].bsums);
        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(x[i].d));

        sumi0 = _mm256_add_epi16(sumi0, sumi1);
        sumi0 = _mm256_sub_epi16(sumi0, ysum);
        sumi0 = _mm256_madd_epi16(sumi0, _mm256_set1_epi16(1));

        sumf = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(sumi0), d), sumf);
    }

    *s = hsum_float_8(sumf);

#else
    float sumf = 0.0f;

    for (int i = 0; i < nb; ++i) {
        int32_t sumi = 0;

        for (size_t j = 0; j < sizeof(x->qs); j += 32) {
            for (size_t l = 0; l < 4; ++l) {
                for (size_t k = 0; k < 32; ++k) {
                    sumi += y[i].qs[j*4 + l*32 + k] * (((x[i].qs[j + k] >> (l*2)) & 3) - 1);
                }
            }
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        sumf += (float) sumi * d;
    }

    *s = sumf;
#endif
}

void ggml_vec_dot_q2_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q2_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);

    const int32x4_t vzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q2bytes;
    uint8_t aux[16];

    float sum = 0;

    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        const uint8_t * restrict sc = x[i].scales;

        const uint8x16_t mins_and_scales = vld1q_u8(sc);
        const uint8x16_t scales = vandq_u8(mins_and_scales, m4);
        vst1q_u8(aux, scales);

        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
        const ggml_int16x8x2_t mins16 = {{vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins))), vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)))}};
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[0]), vget_low_s16 (q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins16.val[0]), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16 (mins16.val[1]), vget_low_s16 (q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins16.val[1]), vget_high_s16(q8sums.val[1])));
        sum += dmin * vaddvq_s32(vaddq_s32(s0, s1));

        int isum = 0;
        int is = 0;

// We use this macro instead of a function call because for some reason
// the code runs 2-3% slower, even if the function is declared inline
#define MULTIPLY_ACCUM_WITH_SCALE(index)\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[0], q8bytes.val[0])) * aux[is+(index)];\
        isum += vaddvq_s32(ggml_vdotq_s32(vzero, q2bytes.val[1], q8bytes.val[1])) * aux[is+1+(index)];

#define SHIFT_MULTIPLY_ACCUM_WITH_SCALE(shift, index)\
        q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;\
        q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[0], (shift)), m3));\
        q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q2bits.val[1], (shift)), m3));\
        MULTIPLY_ACCUM_WITH_SCALE((index));

        for (int j = 0; j < QK_K/128; ++j) {
            const ggml_uint8x16x2_t q2bits = ggml_vld1q_u8_x2(q2); q2 += 32;

            ggml_int8x16x2_t q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q2bytes.val[0] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[0], m3));
            q2bytes.val[1] = vreinterpretq_s8_u8(vandq_u8(q2bits.val[1], m3));

            MULTIPLY_ACCUM_WITH_SCALE(0);

            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(2, 2);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(4, 4);
            SHIFT_MULTIPLY_ACCUM_WITH_SCALE(6, 6);

            is += 8;
        }

        sum += d * isum;
    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales8 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m256i mins = _mm256_cvtepi8_epi16(mins8);
        const __m256i prod = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y[i].bsums));

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

        const __m256i all_scales = _mm256_cvtepi8_epi16(scales8);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = _mm256_loadu_si256((const __m256i*)q2); q2 += 32;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i q2_0 = _mm256_and_si256(q2bits, m3);
            const __m256i q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
            const __m256i q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
            const __m256i q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

            __m256i p0 = _mm256_maddubs_epi16(q2_0, q8_0);
            __m256i p1 = _mm256_maddubs_epi16(q2_1, q8_1);
            __m256i p2 = _mm256_maddubs_epi16(q2_2, q8_2);
            __m256i p3 = _mm256_maddubs_epi16(q2_3, q8_3);

            p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = _mm256_add_epi32(p0, p1);
            p2 = _mm256_add_epi32(p2, p3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(0x3);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // load mins and scales from block_q2_K.scales[QK_K/16]
        const __m128i mins_and_scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales16 = _mm_and_si128(mins_and_scales, m4);
        const __m128i mins16 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        const __m128i mins_0 = _mm_cvtepi8_epi16(mins16);
        const __m128i mins_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(mins16, mins16));

        // summs = y[i].bsums * (x[i].scales >> 4) in 16bits*8*2 to 32bits*4*2
        const __m128i summs_0 = _mm_madd_epi16(mins_0, _mm_loadu_si128((const __m128i*)&y[i].bsums[0]));
        const __m128i summs_1 = _mm_madd_epi16(mins_1, _mm_loadu_si128((const __m128i*)&y[i].bsums[8]));

        // sumf += -dmin * summs in 32bits*8
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(MM256_SET_M128I(summs_1, summs_0))), acc);

        const __m128i scales_0 = _mm_cvtepi8_epi16(scales16);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales16, scales16));
        const __m128i scales[2] = { scales_0, scales_1 };

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {

            // load Q8 quants int8*16*8 from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // load 2bits*16*8 from block_q2_K.qs[QK_K/4]
            __m128i q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_0 = _mm_and_si128(q2bits, m3);
            const __m128i q2_2 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_4 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_6 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);
            q2bits = _mm_loadu_si128((const __m128i*)q2); q2 += 16;
            const __m128i q2_1 = _mm_and_si128(q2bits, m3);
            const __m128i q2_3 = _mm_and_si128(_mm_srli_epi16(q2bits, 2), m3);
            const __m128i q2_5 = _mm_and_si128(_mm_srli_epi16(q2bits, 4), m3);
            const __m128i q2_7 = _mm_and_si128(_mm_srli_epi16(q2bits, 6), m3);

            // isuml = q8[l] * ((q2[l] >> shift) & 3) in 8bits*16*8 to 16bits*8*8
            __m128i p0 = _mm_maddubs_epi16(q2_0, q8_0);
            __m128i p1 = _mm_maddubs_epi16(q2_1, q8_1);
            __m128i p2 = _mm_maddubs_epi16(q2_2, q8_2);
            __m128i p3 = _mm_maddubs_epi16(q2_3, q8_3);
            __m128i p4 = _mm_maddubs_epi16(q2_4, q8_4);
            __m128i p5 = _mm_maddubs_epi16(q2_5, q8_5);
            __m128i p6 = _mm_maddubs_epi16(q2_6, q8_6);
            __m128i p7 = _mm_maddubs_epi16(q2_7, q8_7);

            // isum += (x[i].scales[is++] & 0xF) * isuml in 16bits*8*8 to 32bits*4*8
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p7);

            p0 = _mm_add_epi32(p0, p1);
            p2 = _mm_add_epi32(p2, p3);
            p4 = _mm_add_epi32(p4, p5);
            p6 = _mm_add_epi32(p6, p7);

            // isum in 32bits*4*2
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p0, p2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p4, p6));
        }

        // sumf += dall * isum - dmin * summs in 32bits
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&dall), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    float sumf = 0;
    uint8_t temp_01[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        size_t vl = 16;

        vuint8m1_t scales = __riscv_vle8_v_u8m1(sc, vl);
        vuint8m1_t aux = __riscv_vand_vx_u8m1(scales, 0x0F, vl);

        vint16m1_t q8sums = __riscv_vle16_v_i16m1(y[i].bsums, vl);

        vuint8mf2_t scales_2 = __riscv_vle8_v_u8mf2(sc, vl);
        vuint8mf2_t mins8 = __riscv_vsrl_vx_u8mf2(scales_2, 0x4, vl);
        vint16m1_t mins = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mins8, vl));
        vint32m2_t prod = __riscv_vwmul_vv_i32m2(q8sums, mins, vl);
        vint32m1_t vsums = __riscv_vredsum_vs_i32m2_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);

        sumf  += dmin * __riscv_vmv_x_s_i32m1_i32(vsums);

        vl = 32;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t v_b = __riscv_vle8_v_u8m1(temp_01, vl);

        uint8_t is=0;
        int isum=0;

        for (int j = 0; j < QK_K/128; ++j) {
            // load Q2
            vuint8m1_t q2_x = __riscv_vle8_v_u8m1(q2, vl);

            vuint8m1_t q2_0 = __riscv_vand_vx_u8m1(q2_x, 0x03, vl);
            vuint8m1_t q2_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x2, vl), 0x03 , vl);
            vuint8m1_t q2_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x4, vl), 0x03 , vl);
            vuint8m1_t q2_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q2_x, 0x6, vl), 0x03 , vl);

            // duplicate scale elements for product
            vuint8m1_t sc0 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 0+is, vl), vl);
            vuint8m1_t sc1 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 2+is, vl), vl);
            vuint8m1_t sc2 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 4+is, vl), vl);
            vuint8m1_t sc3 = __riscv_vrgather_vv_u8m1(aux, __riscv_vadd_vx_u8m1(v_b, 6+is, vl), vl);

            vint16m2_t p0 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_0, sc0, vl));
            vint16m2_t p1 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_1, sc1, vl));
            vint16m2_t p2 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_2, sc2, vl));
            vint16m2_t p3 = __riscv_vreinterpret_v_u16m2_i16m2(__riscv_vwmulu_vv_u16m2(q2_3, sc3, vl));

            // load Q8
            vint8m1_t q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
            vint8m1_t q8_2 = __riscv_vle8_v_i8m1(q8+64, vl);
            vint8m1_t q8_3 = __riscv_vle8_v_i8m1(q8+96, vl);

            vint32m4_t s0 = __riscv_vwmul_vv_i32m4(p0, __riscv_vwcvt_x_x_v_i16m2(q8_0, vl), vl);
            vint32m4_t s1 = __riscv_vwmul_vv_i32m4(p1, __riscv_vwcvt_x_x_v_i16m2(q8_1, vl), vl);
            vint32m4_t s2 = __riscv_vwmul_vv_i32m4(p2, __riscv_vwcvt_x_x_v_i16m2(q8_2, vl), vl);
            vint32m4_t s3 = __riscv_vwmul_vv_i32m4(p3, __riscv_vwcvt_x_x_v_i16m2(q8_3, vl), vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s0, s1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m4_i32m1(__riscv_vadd_vv_i32m4(s2, s3, vl), isum0, vl);

            isum += __riscv_vmv_x_s_i32m1_i32(isum1);

            q2+=32;  q8+=128;  is=8;

        }

        sumf += dall * isum;

    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0x3);
    const vector signed char lowScaleMask = vec_splats((signed char)0xF);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v6 = vec_splats((unsigned char)0x6);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[i].dmin));
        vector float vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        vector signed char q2xmins = (vector signed char)vec_xl( 0, x[i].scales);
        vector signed char vscales = vec_and(q2xmins, lowScaleMask);

        q2xmins = vec_sr(q2xmins, v4);
        vector signed short q2xmins0 = vec_unpackh(q2xmins);
        vector signed short q2xmins1 = vec_unpackl(q2xmins);

        vector signed int prod0 = vec_mule(q2xmins0, q8ysums0);
        vector signed int prod1 = vec_mulo(q2xmins0, q8ysums0);
        vector signed int prod2 = vec_mule(q2xmins1, q8ysums1);
        vector signed int prod3 = vec_mulo(q2xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;
        vector signed int vsumi4 = v0;
        vector signed int vsumi5 = v0;
        vector signed int vsumi6 = v0;
        vector signed int vsumi7 = v0;

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q2);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q2);
            q2 += 32;

            vector unsigned char q2x00 = (vector unsigned char)vec_and(qxs0, lowMask);
            vector unsigned char q2x01 = (vector unsigned char)vec_and(vec_sr(qxs0, v2), lowMask);
            vector unsigned char q2x02 = (vector unsigned char)vec_and(vec_sr(qxs0, v4), lowMask);
            vector unsigned char q2x03 = (vector unsigned char)vec_and(vec_sr(qxs0, v6), lowMask);
            vector unsigned char q2x10 = (vector unsigned char)vec_and(qxs1, lowMask);
            vector unsigned char q2x11 = (vector unsigned char)vec_and(vec_sr(qxs1, v2), lowMask);
            vector unsigned char q2x12 = (vector unsigned char)vec_and(vec_sr(qxs1, v4), lowMask);
            vector unsigned char q2x13 = (vector unsigned char)vec_and(vec_sr(qxs1, v6), lowMask);

            vector signed char q8y00 = vec_xl(  0, q8);
            vector signed char q8y10 = vec_xl( 16, q8);
            vector signed char q8y01 = vec_xl( 32, q8);
            vector signed char q8y11 = vec_xl( 48, q8);
            vector signed char q8y02 = vec_xl( 64, q8);
            vector signed char q8y12 = vec_xl( 80, q8);
            vector signed char q8y03 = vec_xl( 96, q8);
            vector signed char q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed int qv0 = vec_msum(q8y00, q2x00, v0);
            vector signed int qv1 = vec_msum(q8y01, q2x01, v0);
            vector signed int qv2 = vec_msum(q8y02, q2x02, v0);
            vector signed int qv3 = vec_msum(q8y03, q2x03, v0);
            vector signed int qv4 = vec_msum(q8y10, q2x10, v0);
            vector signed int qv5 = vec_msum(q8y11, q2x11, v0);
            vector signed int qv6 = vec_msum(q8y12, q2x12, v0);
            vector signed int qv7 = vec_msum(q8y13, q2x13, v0);

            vector signed short vscales_07 = vec_unpackh(vscales);
            vector signed int vscales_03 = vec_unpackh(vscales_07);
            vector signed int vscales_47 = vec_unpackl(vscales_07);
            vector signed int vs0 = vec_splat(vscales_03, 0);
            vector signed int vs1 = vec_splat(vscales_03, 1);
            vector signed int vs2 = vec_splat(vscales_03, 2);
            vector signed int vs3 = vec_splat(vscales_03, 3);
            vector signed int vs4 = vec_splat(vscales_47, 0);
            vector signed int vs5 = vec_splat(vscales_47, 1);
            vector signed int vs6 = vec_splat(vscales_47, 2);
            vector signed int vs7 = vec_splat(vscales_47, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv0, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv1, vs2), vsumi1);
            vsumi2 = vec_add(vec_mul(qv2, vs4), vsumi2);
            vsumi3 = vec_add(vec_mul(qv3, vs6), vsumi3);
            vsumi4 = vec_add(vec_mul(qv4, vs1), vsumi4);
            vsumi5 = vec_add(vec_mul(qv5, vs3), vsumi5);
            vsumi6 = vec_add(vec_mul(qv6, vs5), vsumi6);
            vsumi7 = vec_add(vec_mul(qv7, vs7), vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

    const __m256i m3 = __lasx_xvreplgr2vr_b(3);
    const __m128i m4 = __lsx_vreplgr2vr_b(0xF);

    __m256 acc = (__m256)__lasx_xvldi(0);

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q2 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i mins_and_scales = __lsx_vld((const __m128i*)x[i].scales, 0);
        const __m128i scales8 = __lsx_vand_v(mins_and_scales, m4);
        const __m128i mins8 = __lsx_vand_v(__lsx_vsrli_h(mins_and_scales, 4), m4);
        const __m256i mins = lasx_ext8_16(mins8);
        const __m256i prod = lasx_madd_h(mins, __lasx_xvld((const __m256i*)y[i].bsums, 0));

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(dmin), __lasx_xvffint_s_w(prod), acc);

        const __m256i all_scales = lasx_ext8_16(scales8);
        const __m128i l_scales = lasx_extracti128(all_scales, 0);
        const __m128i h_scales = lasx_extracti128(all_scales, 1);
        const __m256i scales[2] = {lasx_insertf128(l_scales, l_scales), lasx_insertf128(h_scales, h_scales)};

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/128; ++j) {

            const __m256i q2bits = __lasx_xvld((const __m256i*)q2, 0); q2 += 32;

            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            const __m256i q2_0 = __lasx_xvand_v(q2bits, m3);
            const __m256i q2_1 = __lasx_xvand_v(__lasx_xvsrli_h(q2bits, 2), m3);
            const __m256i q2_2 = __lasx_xvand_v(__lasx_xvsrli_h(q2bits, 4), m3);
            const __m256i q2_3 = __lasx_xvand_v(__lasx_xvsrli_h(q2bits, 6), m3);

            __m256i p0 = lasx_maddubs_h(q2_0, q8_0);
            __m256i p1 = lasx_maddubs_h(q2_1, q8_1);
            __m256i p2 = lasx_maddubs_h(q2_2, q8_2);
            __m256i p3 = lasx_maddubs_h(q2_3, q8_3);

            p0 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(0)), p0);
            p1 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(1)), p1);
            p2 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(2)), p2);
            p3 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(3)), p3);

            p0 = __lasx_xvadd_w(p0, p1);
            p2 = __lasx_xvadd_w(p2, p3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p0, p2));
        }

        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);

    }

    *s = hsum_float_8(acc);

#else

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const uint8_t * q2 = x[i].qs;
        const  int8_t * q8 = y[i].qs;
        const uint8_t * sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; ++j) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        int isum = 0;
        int is = 0;
        int d;
        for (int k = 0; k < QK_K/128; ++k) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                d = sc[is++] & 0xF;
                int isuml = 0;
                for (int l =  0; l < 16; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                d = sc[is++] & 0xF;
                isuml = 0;
                for (int l = 16; l < 32; ++l) isuml += q8[l] * ((q2[l] >> shift) & 3);
                isum += d * isuml;
                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * isum - dmin * summs;
    }
    *s = sumf;
#endif
}

void ggml_vec_dot_q3_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    const block_q3_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON

    uint32_t aux[3];
    uint32_t utmp[4];

    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const int32x4_t  vzero = vdupq_n_s32(0);

    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0, 1);
    const uint8x16_t m2 = vshlq_n_u8(m0, 2);
    const uint8x16_t m3 = vshlq_n_u8(m0, 3);
    const int8_t m32 = 32;

    ggml_int8x16x4_t q3bytes;

    float sum = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const int8_t  * restrict q8 = y[i].qs;

        ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);

        ggml_uint8x16x4_t q3h;

        int32_t isum = 0;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= m32;

        for (int j = 0; j < QK_K/128; ++j) {

            const ggml_uint8x16x2_t q3bits = ggml_vld1q_u8_x2(q3); q3 += 32;
            const ggml_int8x16x4_t q8bytes_1 = ggml_vld1q_s8_x4(q8); q8 += 64;
            const ggml_int8x16x4_t q8bytes_2 = ggml_vld1q_s8_x4(q8); q8 += 64;

            q3h.val[0] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
            q3h.val[1] = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
            q3h.val[2] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
            q3h.val[3] = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_1.val[0])) * scale[0];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_1.val[1])) * scale[1];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_1.val[2])) * scale[2];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_1.val[3])) * scale[3];

            scale += 4;

            q3h.val[0] = vbicq_u8(m2, qhbits.val[0]);
            q3h.val[1] = vbicq_u8(m2, qhbits.val[1]);
            q3h.val[2] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
            q3h.val[3] = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);

            q3bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(q3h.val[0]));
            q3bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(q3h.val[1]));
            q3bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(q3h.val[2]));
            q3bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(q3h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[0], q8bytes_2.val[0])) * scale[0];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[1], q8bytes_2.val[1])) * scale[1];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[2], q8bytes_2.val[2])) * scale[2];
            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q3bytes.val[3], q8bytes_2.val[3])) * scale[3];

            scale += 4;

            if (j == 0) {
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
            }

        }
        sum += d * isum;

    }

    *s = sum;

#elif defined __AVX2__

    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
        const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
        const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
        const __m256i scales[2] = {MM256_SET_M128I(l_scales, l_scales), MM256_SET_M128I(h_scales, h_scales)};

        // high bit
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);

        // integer accumulator
        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;
        int is  = 0;

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3); q3 += 32;

            // prepare low and high bits
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            // multiply with scales
            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = _mm256_add_epi32(p16_0, p16_1);
            p16_2 = _mm256_add_epi32(p16_2, p16_3);
            sumi  = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));

        }

        // multiply with block scale and accumulate
        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i mone = _mm_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    const uint32_t *aux;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        // Set up scales
        aux = (const uint32_t *)x[i].scales;
        __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = _mm_sub_epi8(scales128, m32);
        const __m128i scales_0 = _mm_cvtepi8_epi16(scales128);
        const __m128i scales_1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(scales128, scales128));
        const __m128i scales[2] = { scales_0, scales_1 };

        // high bit *128*2 from block_q3_K.hmask[QK_K/8]
        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].hmask[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].hmask[16]);

        // integer accumulator
        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits *64*2 from block_q3_K.qs[QK_K/4]
            const __m128i q3bits_0 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;
            const __m128i q3bits_1 = _mm_loadu_si128((const __m128i*)q3); q3 += 16;

            // prepare low and high bits
            const int bit = j << 2;

            const __m128i q3l_0 = _mm_and_si128(q3bits_0, m3);
            const __m128i q3l_1 = _mm_and_si128(q3bits_1, m3);
            const __m128i q3h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit)), bit), 2);
            const __m128i q3h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit)), bit), 2);

            const __m128i q3l_2 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 2), m3);
            const __m128i q3l_3 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 2), m3);
            const __m128i q3h_2 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+1)), bit+1), 2);
            const __m128i q3h_3 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+1)), bit+1), 2);

            const __m128i q3l_4 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 4), m3);
            const __m128i q3l_5 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 4), m3);
            const __m128i q3h_4 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+2)), bit+2), 2);
            const __m128i q3h_5 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+2)), bit+2), 2);

            const __m128i q3l_6 = _mm_and_si128(_mm_srli_epi16(q3bits_0, 6), m3);
            const __m128i q3l_7 = _mm_and_si128(_mm_srli_epi16(q3bits_1, 6), m3);
            const __m128i q3h_6 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_0, _mm_slli_epi16(mone, bit+3)), bit+3), 2);
            const __m128i q3h_7 = _mm_slli_epi16(_mm_srli_epi16(_mm_andnot_si128(hbits_1, _mm_slli_epi16(mone, bit+3)), bit+3), 2);

            // load Q8 quants from block_q8_K.qs[QK_K]
            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use _mm256_maddubs_epi16,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m128i q8s_0 = _mm_maddubs_epi16(q3h_0, q8_0);
            __m128i q8s_1 = _mm_maddubs_epi16(q3h_1, q8_1);
            __m128i q8s_2 = _mm_maddubs_epi16(q3h_2, q8_2);
            __m128i q8s_3 = _mm_maddubs_epi16(q3h_3, q8_3);
            __m128i q8s_4 = _mm_maddubs_epi16(q3h_4, q8_4);
            __m128i q8s_5 = _mm_maddubs_epi16(q3h_5, q8_5);
            __m128i q8s_6 = _mm_maddubs_epi16(q3h_6, q8_6);
            __m128i q8s_7 = _mm_maddubs_epi16(q3h_7, q8_7);

            __m128i p16_0 = _mm_maddubs_epi16(q3l_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q3l_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q3l_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q3l_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q3l_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q3l_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q3l_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q3l_7, q8_7);

            p16_0 = _mm_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm_sub_epi16(p16_3, q8s_3);
            p16_4 = _mm_sub_epi16(p16_4, q8s_4);
            p16_5 = _mm_sub_epi16(p16_5, q8s_5);
            p16_6 = _mm_sub_epi16(p16_6, q8s_6);
            p16_7 = _mm_sub_epi16(p16_7, q8s_7);

            // multiply with scales
            __m128i shuffle = _mm_set1_epi16(0x0100);
            p16_0 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_0);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_1 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_1);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_2 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_2);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_3 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_3);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_4 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_4);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_5 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_5);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_6 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_6);
            shuffle = _mm_add_epi16(shuffle, m2);
            p16_7 = _mm_madd_epi16(_mm_shuffle_epi8(scales[j], shuffle), p16_7);

            // accumulate
            p16_0 = _mm_add_epi32(p16_0, p16_1);
            p16_2 = _mm_add_epi32(p16_2, p16_3);
            p16_4 = _mm_add_epi32(p16_4, p16_5);
            p16_6 = _mm_add_epi32(p16_6, p16_7);
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_4, p16_6));

        }

        // multiply with block scale and accumulate
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    uint32_t aux[3];
    uint32_t utmp[4];

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict qh = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;

        memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);

        int8_t * scale = (int8_t *)utmp;
        for (int j = 0; j < 16; ++j) scale[j] -= 32;


        size_t vl = 32;
        uint8_t m =  1;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t vqh = __riscv_vle8_v_u8m1(qh, vl);

        int sum_t = 0;

        for (int j = 0; j < QK_K; j += 128) {

            vl = 32;

            // load Q3
            vuint8m1_t q3_x = __riscv_vle8_v_u8m1(q3, vl);

            vint8m1_t q3_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q3_x, 0x03, vl));
            vint8m1_t q3_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x2, vl), 0x03 , vl));
            vint8m1_t q3_2 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x4, vl), 0x03 , vl));
            vint8m1_t q3_3 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(q3_x, 0x6, vl), 0x03 , vl));

            // compute mask for subtraction
            vuint8m1_t qh_m0 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_0 = __riscv_vmseq_vx_u8m1_b8(qh_m0, 0, vl);
            vint8m1_t q3_m0 = __riscv_vsub_vx_i8m1_mu(vmask_0, q3_0, q3_0, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_1 = __riscv_vmseq_vx_u8m1_b8(qh_m1, 0, vl);
            vint8m1_t q3_m1 = __riscv_vsub_vx_i8m1_mu(vmask_1, q3_1, q3_1, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_2 = __riscv_vmseq_vx_u8m1_b8(qh_m2, 0, vl);
            vint8m1_t q3_m2 = __riscv_vsub_vx_i8m1_mu(vmask_2, q3_2, q3_2, 0x4, vl);
            m <<= 1;

            vuint8m1_t qh_m3 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_3 = __riscv_vmseq_vx_u8m1_b8(qh_m3, 0, vl);
            vint8m1_t q3_m3 = __riscv_vsub_vx_i8m1_mu(vmask_3, q3_3, q3_3, 0x4, vl);
            m <<= 1;

            // load Q8 and take product with Q3
            vint16m2_t a0 = __riscv_vwmul_vv_i16m2(q3_m0, __riscv_vle8_v_i8m1(q8, vl), vl);
            vint16m2_t a1 = __riscv_vwmul_vv_i16m2(q3_m1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
            vint16m2_t a2 = __riscv_vwmul_vv_i16m2(q3_m2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
            vint16m2_t a3 = __riscv_vwmul_vv_i16m2(q3_m3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

            vl = 16;

            // retrieve lane to multiply with scale
            vint32m2_t aux0_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 0), (scale[0]), vl);
            vint32m2_t aux0_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a0, 1), (scale[1]), vl);
            vint32m2_t aux1_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 0), (scale[2]), vl);
            vint32m2_t aux1_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a1, 1), (scale[3]), vl);
            vint32m2_t aux2_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 0), (scale[4]), vl);
            vint32m2_t aux2_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a2, 1), (scale[5]), vl);
            vint32m2_t aux3_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 0), (scale[6]), vl);
            vint32m2_t aux3_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(a3, 1), (scale[7]), vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux0_0, aux0_1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux1_0, aux1_1, vl), isum0, vl);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux2_0, aux2_1, vl), isum1, vl);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(aux3_0, aux3_1, vl), isum2, vl);

            sum_t +=  __riscv_vmv_x_s_i32m1_i32(isum3);

            q3 += 32;    q8 += 128;   scale += 8;

        }

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

        sumf += d*sum_t;

    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0x3);
    const vector signed char lowMask1 = vec_splats((int8_t)0xf);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector signed char v1 = vec_splats((signed char)0x1);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v3 = vec_splats((unsigned char)0x3);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector unsigned char v6 = vec_splats((unsigned char)0x6);
    const vector signed char off = vec_splats((signed char)0x20);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);

        vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
        vector signed char u1 = vec_and(u0, lowMask1);
        vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
        vector signed char u3 = (vector signed char)vec_mergeh((vector signed int)u2, (vector signed int)vec_sr(u2, v2));
        vector signed char u30 = vec_sl(vec_and(u3, lowMask), v4);
        vector signed char u31 = vec_and(u3, lowMask2);

        u1 = vec_or(u1, u30);
        u2 = vec_or(vec_sr(u0, v4), u31);

        vector signed char vscales = (vector signed char)vec_mergeh((vector signed long long)u1, (vector signed long long)u2);
        vector signed char qxhs0 = (vector signed char)vec_xl( 0, x[i].hmask);
        vector signed char qxhs1 = (vector signed char)vec_xl(16, x[i].hmask);

        vscales = vec_sub(vscales, off);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;
        vector signed int vsumi4 = v0;
        vector signed int vsumi5 = v0;
        vector signed int vsumi6 = v0;
        vector signed int vsumi7 = v0;

        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q3);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q3);
            q3 += 32;

            //the low 2 bits
            vector signed char qxs00 = vec_and(qxs0, lowMask);
            vector signed char qxs01 = vec_and(vec_sr(qxs0, v2), lowMask);
            vector signed char qxs02 = vec_and(vec_sr(qxs0, v4), lowMask);
            vector signed char qxs03 = vec_and(vec_sr(qxs0, v6), lowMask);
            vector signed char qxs10 = vec_and(qxs1, lowMask);
            vector signed char qxs11 = vec_and(vec_sr(qxs1, v2), lowMask);
            vector signed char qxs12 = vec_and(vec_sr(qxs1, v4), lowMask);
            vector signed char qxs13 = vec_and(vec_sr(qxs1, v6), lowMask);

            //the 3rd bit
            vector signed char qxh00 = vec_sl(vec_andc(v1, qxhs0), v2);
            vector signed char qxh01 = vec_sl(vec_andc(v1, vec_sr(qxhs0, (vector unsigned char)v1)), v2);
            vector signed char qxh02 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v2)), v2);
            vector signed char qxh03 = vec_sl(vec_andc(v1, vec_sr(qxhs0, v3)), v2);
            vector signed char qxh10 = vec_sl(vec_andc(v1, qxhs1), v2);
            vector signed char qxh11 = vec_sl(vec_andc(v1, vec_sr(qxhs1, (vector unsigned char)v1)), v2);
            vector signed char qxh12 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v2)), v2);
            vector signed char qxh13 = vec_sl(vec_andc(v1, vec_sr(qxhs1, v3)), v2);
            qxhs0 = vec_sr(qxhs0, v4);
            qxhs1 = vec_sr(qxhs1, v4);

            vector signed char q3x00 = vec_sub(qxs00, qxh00);
            vector signed char q3x01 = vec_sub(qxs01, qxh01);
            vector signed char q3x02 = vec_sub(qxs02, qxh02);
            vector signed char q3x03 = vec_sub(qxs03, qxh03);
            vector signed char q3x10 = vec_sub(qxs10, qxh10);
            vector signed char q3x11 = vec_sub(qxs11, qxh11);
            vector signed char q3x12 = vec_sub(qxs12, qxh12);
            vector signed char q3x13 = vec_sub(qxs13, qxh13);

            vector signed char q8y00 = vec_xl(  0, q8);
            vector signed char q8y10 = vec_xl( 16, q8);
            vector signed char q8y01 = vec_xl( 32, q8);
            vector signed char q8y11 = vec_xl( 48, q8);
            vector signed char q8y02 = vec_xl( 64, q8);
            vector signed char q8y12 = vec_xl( 80, q8);
            vector signed char q8y03 = vec_xl( 96, q8);
            vector signed char q8y13 = vec_xl(112, q8);
            q8 += 128;

            vector signed short vscales_h = vec_unpackh(vscales);
            vector signed short vs0 = vec_splat(vscales_h, 0);
            vector signed short vs1 = vec_splat(vscales_h, 1);
            vector signed short vs2 = vec_splat(vscales_h, 2);
            vector signed short vs3 = vec_splat(vscales_h, 3);
            vector signed short vs4 = vec_splat(vscales_h, 4);
            vector signed short vs5 = vec_splat(vscales_h, 5);
            vector signed short vs6 = vec_splat(vscales_h, 6);
            vector signed short vs7 = vec_splat(vscales_h, 7);
            vscales = vec_sld(vscales, vscales, 8);

            vector signed short qv00 = vec_add(vec_mule(q3x00, q8y00), vec_mulo(q3x00, q8y00));
            vector signed short qv01 = vec_add(vec_mule(q3x01, q8y01), vec_mulo(q3x01, q8y01));
            vector signed short qv02 = vec_add(vec_mule(q3x02, q8y02), vec_mulo(q3x02, q8y02));
            vector signed short qv03 = vec_add(vec_mule(q3x03, q8y03), vec_mulo(q3x03, q8y03));
            vector signed short qv10 = vec_add(vec_mule(q3x10, q8y10), vec_mulo(q3x10, q8y10));
            vector signed short qv11 = vec_add(vec_mule(q3x11, q8y11), vec_mulo(q3x11, q8y11));
            vector signed short qv12 = vec_add(vec_mule(q3x12, q8y12), vec_mulo(q3x12, q8y12));
            vector signed short qv13 = vec_add(vec_mule(q3x13, q8y13), vec_mulo(q3x13, q8y13));

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs2, vsumi1);
            vsumi2 = vec_msum(qv02, vs4, vsumi2);
            vsumi3 = vec_msum(qv03, vs6, vsumi3);
            vsumi4 = vec_msum(qv10, vs1, vsumi4);
            vsumi5 = vec_msum(qv11, vs3, vsumi5);
            vsumi6 = vec_msum(qv12, vs5, vsumi6);
            vsumi7 = vec_msum(qv13, vs7, vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

    const __m256i m3 = __lasx_xvreplgr2vr_b(3);
    const __m256i mone = __lasx_xvreplgr2vr_b(1);
    const __m128i m32 = __lsx_vreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    uint32_t aux[3];

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * restrict q3 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;
        // Set up scales
        memcpy(aux, x[i].scales, 12);
        __m128i scales128 = lsx_set_w(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
        scales128 = __lsx_vsub_b(scales128, m32);
        const __m256i all_scales = lasx_ext8_16(scales128);
        const __m128i l_scales = lasx_extracti128(all_scales, 0);
        const __m128i h_scales = lasx_extracti128(all_scales, 1);
        const __m256i scales[2] = {lasx_insertf128(l_scales, l_scales), lasx_insertf128(h_scales, h_scales)};

        // high bit
        const __m256i hbits = __lasx_xvld((const __m256i*)x[i].hmask, 0);

        // integer accumulator
        __m256i sumi = __lasx_xvldi(0);

        int bit = 0;
        int is  = 0;
        __m256i xvbit;


        for (int j = 0; j < QK_K/128; ++j) {
            // load low 2 bits
            const __m256i q3bits = __lasx_xvld((const __m256i*)q3, 0); q3 += 32;

            xvbit = __lasx_xvreplgr2vr_h(bit);
            // prepare low and high bits
            const __m256i q3l_0 = __lasx_xvand_v(q3bits, m3);
            const __m256i q3h_0 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvandn_v(hbits, __lasx_xvsll_h(mone, xvbit)), xvbit), 2);
            ++bit;

            xvbit = __lasx_xvreplgr2vr_h(bit);
            const __m256i q3l_1 = __lasx_xvand_v(__lasx_xvsrli_h(q3bits, 2), m3);
            const __m256i q3h_1 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvandn_v(hbits, __lasx_xvsll_h(mone, xvbit)), xvbit), 2);
            ++bit;

            xvbit = __lasx_xvreplgr2vr_h(bit);
            const __m256i q3l_2 = __lasx_xvand_v(__lasx_xvsrli_h(q3bits, 4), m3);
            const __m256i q3h_2 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvandn_v(hbits, __lasx_xvsll_h(mone, xvbit)), xvbit), 2);
            ++bit;

            xvbit = __lasx_xvreplgr2vr_h(bit);
            const __m256i q3l_3 = __lasx_xvand_v(__lasx_xvsrli_h(q3bits, 6), m3);
            const __m256i q3h_3 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvandn_v(hbits, __lasx_xvsll_h(mone, xvbit)), xvbit), 2);
            ++bit;

            // load Q8 quants
            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we can use lasx_maddubs_h,
            // and then subtract. The high bit part has the 2 already subtracted (and so, it is zero if the high bit was not set,
            // and 2 if the high bit was set)
            __m256i q8s_0 = lasx_maddubs_h(q3h_0, q8_0);
            __m256i q8s_1 = lasx_maddubs_h(q3h_1, q8_1);
            __m256i q8s_2 = lasx_maddubs_h(q3h_2, q8_2);
            __m256i q8s_3 = lasx_maddubs_h(q3h_3, q8_3);

            __m256i p16_0 = lasx_maddubs_h(q3l_0, q8_0);
            __m256i p16_1 = lasx_maddubs_h(q3l_1, q8_1);
            __m256i p16_2 = lasx_maddubs_h(q3l_2, q8_2);
            __m256i p16_3 = lasx_maddubs_h(q3l_3, q8_3);

            p16_0 = __lasx_xvsub_h(p16_0, q8s_0);
            p16_1 = __lasx_xvsub_h(p16_1, q8s_1);
            p16_2 = __lasx_xvsub_h(p16_2, q8s_2);
            p16_3 = __lasx_xvsub_h(p16_3, q8s_3);

            // multiply with scales
            p16_0 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(is + 0)), p16_0);
            p16_1 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(is + 1)), p16_1);
            p16_2 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(is + 2)), p16_2);
            p16_3 = lasx_madd_h(lasx_shuffle_b(scales[j], get_scale_shuffle_q3k(is + 3)), p16_3);

            // accumulate
            p16_0 = __lasx_xvadd_w(p16_0, p16_1);
            p16_2 = __lasx_xvadd_w(p16_2, p16_3);
            sumi  = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_2));
        }
        // multiply with block scale and accumulate
        acc = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);//FIXME
    }

    *s = hsum_float_8(acc);

#else
    // scalar version
    // This function is written like this so the compiler can manage to vectorize most of it
    // Using -Ofast, GCC and clang manage to produce code that is within a factor of 2 or so from the
    // manually vectorized version above. Every other version I tried would run at least 4 times slower.
    // The ideal situation would be if we could just write the code once, and the compiler would
    // automatically produce the best possible set of machine instructions, instead of us having to manually
    // write vectorized versions for AVX, ARM_NEON, etc.

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    uint32_t auxs[4];
    const int8_t * scales = (const int8_t*)auxs;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict hm = x[i].hmask;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) a[l] = q3[l] & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 2) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 4) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (q3[l] >> 6) & 3;
            for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m ? 0 : 4);
            a += 32; m <<= 1;
            q3 += 32;
        }
        a = aux8;

        memcpy(auxs, x[i].scales, 12);
        uint32_t tmp = auxs[2];
        auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        for (int j = 0; j < QK_K/16; ++j) {
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;

#endif

}

void ggml_vec_dot_q4_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q4_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x2_t q4bytes;
    ggml_int8x16x2_t q8bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);

        uint32x2_t mins8 = { 0 };
        mins8 = vset_lane_u32(utmp[1] & kmask1, mins8, 0);
        mins8 = vset_lane_u32(((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4), mins8, 1);

        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[0] &= kmask1;

        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins8)));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        sumf -= dmin * vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        int32_t sumi1 = 0;
        int32_t sumi2 = 0;

        for (int j = 0; j < QK_K/64; ++j) {
            const ggml_uint8x16x2_t q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[0], m4b));
            q4bytes.val[1] = vreinterpretq_s8_u8(vandq_u8  (q4bits.val[1], m4b));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);
            sumi1 += vaddvq_s32(p1) * scales[2*j+0];

            q8bytes = ggml_vld1q_s8_x2(q8); q8 += 32;
            q4bytes.val[0] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[0], 4));
            q4bytes.val[1] = vreinterpretq_s8_u8(vshrq_n_u8(q4bits.val[1], 4));

            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(mzero, q4bytes.val[0], q8bytes.val[0]), q4bytes.val[1], q8bytes.val[1]);

            sumi2 += vaddvq_s32(p2) * scales[2*j+1];
        }

        sumf += d * (sumi1 + sumi2);

    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        __m256i sumi = _mm256_setzero_si256();

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_maddubs_epi16(q4l, q8l);
            p16l = _mm256_madd_epi16(scale_l, p16l);

            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16h = _mm256_maddubs_epi16(q4h, q8h);
            p16h = _mm256_madd_epi16(scale_h, p16h);
            const __m256i sumj = _mm256_add_epi32(p16l, p16h);

            sumi = _mm256_add_epi32(sumi, sumj);
        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i m2 = _mm_set1_epi8(0x2);

    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        acc_m = _mm_add_ps(_mm_mul_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod)), acc_m);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_l = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_h = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            __m128i q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_0 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_0 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);
            q4bits = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4l_1 = _mm_and_si128(q4bits, m4);
            const __m128i q4h_1 = _mm_and_si128(_mm_srli_epi16(q4bits, 4), m4);

            const __m128i q8l_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16l = _mm_maddubs_epi16(q4l_0, q8l_0);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_0 = _mm_add_epi32(sumi_0, p16l);
            const __m128i q8l_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16l = _mm_maddubs_epi16(q4l_1, q8l_1);
            p16l = _mm_madd_epi16(scale_l, p16l);
            sumi_1 = _mm_add_epi32(sumi_1, p16l);

            const __m128i q8h_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16h = _mm_maddubs_epi16(q4h_0, q8h_0);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_0 = _mm_add_epi32(sumi_0, p16h);
            const __m128i q8h_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            p16h = _mm_maddubs_epi16(q4h_1, q8h_1);
            p16h = _mm_madd_epi16(scale_h, p16h);
            sumi_1 = _mm_add_epi32(sumi_1, p16h);

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

    *s = hsum_float_8(acc) + _mm_cvtss_f32(acc_m);

#elif defined __riscv_v_intrinsic

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        size_t vl = 8;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
        vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
        vint16mf2_t q8sums   = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf4_t mins8  = __riscv_vle8_v_u8mf4(mins, vl);
        vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
        vint32m1_t  prod   = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        vl = 32;

        int32_t sum_1 = 0;
        int32_t sum_2 = 0;

        vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, 1);

        for (int j = 0; j < QK_K/64; ++j) {
            // load Q4
            vuint8m1_t q4_x = __riscv_vle8_v_u8m1(q4, vl);

            // load Q8 and multiply it with lower Q4 nibble
            vint8m1_t  q8_0 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t  q4_0 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q4_x, 0x0F, vl));
            vint16m2_t qv_0 = __riscv_vwmul_vv_i16m2(q4_0, q8_0, vl);
            vint16m1_t vs_0 = __riscv_vredsum_vs_i16m2_i16m1(qv_0, vzero, vl);

            sum_1 += __riscv_vmv_x_s_i16m1_i16(vs_0) * scales[2*j+0];

            // load Q8 and multiply it with upper Q4 nibble
            vint8m1_t  q8_1 = __riscv_vle8_v_i8m1(q8+32, vl);
            vint8m1_t  q4_1 = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q4_x, 0x04, vl));
            vint16m2_t qv_1 = __riscv_vwmul_vv_i16m2(q4_1, q8_1, vl);
            vint16m1_t vs_1 = __riscv_vredsum_vs_i16m2_i16m1(qv_1, vzero, vl);

            sum_2 += __riscv_vmv_x_s_i16m1_i16(vs_1) * scales[2*j+1];

            q4 += 32;    q8 += 64;

        }

        sumf += d*(sum_1 + sum_2);

    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed char lowMask1 = vec_splats((int8_t)0x3f);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((uint8_t)2);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[i].dmin));
        vector float vdmin = vec_mul(vxmin, vyd);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
        vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
        vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
        vector signed char u3 = vec_sr(u2, v4);

        vector signed char u30 = u1;
        vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);

        u1 = vec_and(u0, lowMask1);
        u2 = vec_or(u30, u31);

        vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);

        vector signed short vscales = vec_unpackh(utmps);
        vector signed short q4xmins = vec_unpackl(utmps);
        vector signed short q4xmins0 = vec_mergeh(q4xmins, q4xmins);
        vector signed short q4xmins1 = vec_mergel(q4xmins, q4xmins);

        vector signed int prod0 = vec_mule(q4xmins0, q8ysums0);
        vector signed int prod1 = vec_mule(q4xmins1, q8ysums1);
        vector signed int prod2 = vec_mulo(q4xmins0, q8ysums0);
        vector signed int prod3 = vec_mulo(q4xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/64; j+=2) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q4);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q4);
            vector signed char qxs2 = (vector signed char)vec_xl(32, q4);
            vector signed char qxs3 = (vector signed char)vec_xl(48, q4);
            q4 += 64;

            vector unsigned char q4x00 = (vector unsigned char)vec_and(qxs0, lowMask);
            vector unsigned char q4x01 = (vector unsigned char)vec_sr(qxs0, v4);
            vector unsigned char q4x10 = (vector unsigned char)vec_and(qxs1, lowMask);
            vector unsigned char q4x11 = (vector unsigned char)vec_sr(qxs1, v4);
            vector unsigned char q4x20 = (vector unsigned char)vec_and(qxs2, lowMask);
            vector unsigned char q4x21 = (vector unsigned char)vec_sr(qxs2, v4);
            vector unsigned char q4x30 = (vector unsigned char)vec_and(qxs3, lowMask);
            vector unsigned char q4x31 = (vector unsigned char)vec_sr(qxs3, v4);

            vector signed char q8y00 = vec_xl(  0, q8);
            vector signed char q8y10 = vec_xl( 16, q8);
            vector signed char q8y01 = vec_xl( 32, q8);
            vector signed char q8y11 = vec_xl( 48, q8);
            vector signed char q8y20 = vec_xl( 64, q8);
            vector signed char q8y30 = vec_xl( 80, q8);
            vector signed char q8y21 = vec_xl( 96, q8);
            vector signed char q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector signed int qv00 = vec_msum(q8y00, q4x00, v0);
            vector signed int qv01 = vec_msum(q8y01, q4x01, v0);
            vector signed int qv10 = vec_msum(q8y10, q4x10, v0);
            vector signed int qv11 = vec_msum(q8y11, q4x11, v0);
            vector signed int qv20 = vec_msum(q8y20, q4x20, v0);
            vector signed int qv21 = vec_msum(q8y21, q4x21, v0);
            vector signed int qv30 = vec_msum(q8y30, q4x30, v0);
            vector signed int qv31 = vec_msum(q8y31, q4x31, v0);

            vector signed int vscales_h = vec_unpackh(vscales);
            vector signed int vs0 = vec_splat(vscales_h, 0);
            vector signed int vs1 = vec_splat(vscales_h, 1);
            vector signed int vs2 = vec_splat(vscales_h, 2);
            vector signed int vs3 = vec_splat(vscales_h, 3);
            vscales = vec_sld(vscales, vscales, 8);

            vsumi0 = vec_add(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv01, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv20, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv21, vs3), vsumi3);

            vsumi0 = vec_add(vec_mul(qv10, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv11, vs1), vsumi1);
            vsumi2 = vec_add(vec_mul(qv30, vs2), vsumi2);
            vsumi3 = vec_add(vec_mul(qv31, vs3), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx
    GGML_UNUSED(kmask1);
    GGML_UNUSED(kmask2);
    GGML_UNUSED(kmask3);

    const __m256i m4 = __lasx_xvreplgr2vr_b(0xF);

    __m256 acc = (__m256)__lasx_xvldi(0);
    __m128 acc_m = (__m128)__lsx_vldi(0);

   for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t * restrict q4 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const __m256i mins_and_scales = lasx_extu8_16(lsx_set_w(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = __lasx_xvld((const __m256i*)y[i].bsums, 0);
        const __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        const __m128i prod = lsx_madd_h(lasx_extracti128(mins_and_scales, 1), q8s);
        acc_m = __lsx_vfmadd_s(__lsx_vreplfr2vr_s(dmin), __lsx_vffint_s_w(prod), acc_m);

        const __m128i sc128  = lasx_extracti128(mins_and_scales, 0);
        const __m256i scales = lasx_insertf128(sc128, sc128);

        __m256i sumi = __lasx_xvldi(0);

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_l = lasx_shuffle_b(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_h = lasx_shuffle_b(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q4bits = __lasx_xvld((const __m256i*)q4, 0); q4 += 32;
            const __m256i q4l = __lasx_xvand_v(q4bits, m4);
            const __m256i q4h = __lasx_xvand_v(__lasx_xvsrli_h(q4bits, 4), m4);

            const __m256i q8l = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16l = lasx_maddubs_h(q4l, q8l);
            p16l = lasx_madd_h(scale_l, p16l);

            const __m256i q8h = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            __m256i p16h = lasx_maddubs_h(q4h, q8h);
            p16h = lasx_madd_h(scale_h, p16h);
            const __m256i sumj = __lasx_xvadd_w(p16l, p16h);

            sumi = __lasx_xvadd_w(sumi, sumj);
        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    acc_m = __lsx_vfadd_s(acc_m, (__m128)__lsx_vpermi_w((__m128i)acc_m, (__m128i)acc_m, 0xee));
    __m128i tmp1 = __lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w((__m128i)acc_m, 1), 0);
    acc_m = __lsx_vfadd_s(acc_m, (__m128)tmp1);


    ft_union fi;
    fi.i = __lsx_vpickve2gr_w(acc_m, 0);
    *s = hsum_float_8(acc) + fi.f ;
#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            a += 32;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            a += 32; q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q5_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy,  size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q5_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    uint32_t utmp[4];

#ifdef __ARM_NEON
    const uint8x16_t m4b = vdupq_n_u8(0xf);
    const uint8x16_t mone = vdupq_n_u8(1);
    const uint8x16_t mtwo = vdupq_n_u8(2);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x4_t q5bytes;

    float sumf = 0;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const int16x8_t q8sums = vpaddq_s16(vld1q_s16(y[i].bsums), vld1q_s16(y[i].bsums + 8));

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8x8_t mins8 = vld1_u8((const uint8_t*)utmp + 8);
        const int16x8_t mins = vreinterpretq_s16_u16(vmovl_u8(mins8));
        const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16 (q8sums), vget_low_s16 (mins)),
                                         vmull_s16(vget_high_s16(q8sums), vget_high_s16(mins)));
        int32_t sumi_mins = vaddvq_s32(prod);

        const uint8_t * scales = (const uint8_t *)utmp;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh);

        ggml_uint8x16x4_t q5h;

        int32_t sumi = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const ggml_uint8x16x2_t q5bits = ggml_vld1q_u8_x2(q5); q5 += 32;
            const ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            q5h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q5h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            q5h.val[2] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[0]), 3);
            q5h.val[3] = vshlq_n_u8(vandq_u8(mtwo, qhbits.val[1]), 3);
            qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 2);
            qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 2);

            q5bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[0], m4b), q5h.val[0]));
            q5bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q5bits.val[1], m4b), q5h.val[1]));
            q5bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[0], 4), q5h.val[2]));
            q5bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q5bits.val[1], 4), q5h.val[3]));

            sumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(mzero, q5bytes.val[0], q8bytes.val[0]), q5bytes.val[1], q8bytes.val[1])) * *scales++;
            sumi += vaddvq_s32(ggml_vdotq_s32(ggml_vdotq_s32(mzero, q5bytes.val[2], q8bytes.val[2]), q5bytes.val[3], q8bytes.val[3])) * *scales++;
        }

        sumf += d * sumi - dmin * sumi_mins;
    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m256i mone  = _mm256_set1_epi8(1);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i sc128  = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = MM256_SET_M128I(sc128, sc128);

        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].qh);
        __m256i hmask = mone;

        __m256i sumi = _mm256_setzero_si256();

        int bit = 0;

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = _mm256_loadu_si256((const __m256i*)q5); q5 += 32;

            const __m256i q5l_0 = _mm256_and_si256(q5bits, m4);
            const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
            const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), bit++), 4);
            const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);
            hmask = _mm256_slli_epi16(hmask, 1);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

            p16_0 = _mm256_madd_epi16(scale_0, p16_0);
            p16_1 = _mm256_madd_epi16(scale_1, p16_1);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));

        }

        __m256 vd = _mm256_set1_ps(d);
        acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#elif defined __AVX__

    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m128i mzero = _mm_setzero_si128();
    const __m128i mone  = _mm_set1_epi8(1);
    const __m128i m2 = _mm_set1_epi8(2);

    __m256 acc = _mm256_setzero_ps();

    float summs = 0.f;

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m128i utmps = _mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]);
        const __m128i scales = _mm_cvtepu8_epi16(utmps);
        const __m128i mins = _mm_cvtepu8_epi16(_mm_unpackhi_epi64(utmps, utmps));

        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)&y[i].bsums[0]);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)&y[i].bsums[8]);
        const __m128i q8s = _mm_hadd_epi16(q8sums_0, q8sums_1);
        const __m128i prod = _mm_madd_epi16(mins, q8s);
        const __m128i hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
        summs += dmin * _mm_extract_epi32(hsum, 0);

        const __m128i hbits_0 = _mm_loadu_si128((const __m128i*)&x[i].qh[0]);
        const __m128i hbits_1 = _mm_loadu_si128((const __m128i*)&x[i].qh[16]);
        __m128i hmask = mone;

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int bit = 0;

        __m128i shuffle = _mm_set1_epi16(0x0100);
        for (int j = 0; j < QK_K/64; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);
            const __m128i scale_1 = _mm_shuffle_epi8(scales, shuffle);
            shuffle = _mm_add_epi16(shuffle, m2);

            const __m128i q5bits_0 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;
            const __m128i q5bits_1 = _mm_loadu_si128((const __m128i*)q5); q5 += 16;

            __m128i q5l_0 = _mm_and_si128(q5bits_0, m4);
            __m128i q5l_1 = _mm_and_si128(q5bits_1, m4);
            __m128i q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            __m128i q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            __m128i q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            __m128i q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_0 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_0 = _mm_madd_epi16(scale_0, p16_0);
            p16_1 = _mm_madd_epi16(scale_0, p16_1);

            q5l_0 = _mm_and_si128(_mm_srli_epi16(q5bits_0, 4), m4);
            q5l_1 = _mm_and_si128(_mm_srli_epi16(q5bits_1, 4), m4);
            q5h_0 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_0, hmask), bit), 4);
            q5h_1 = _mm_slli_epi16(_mm_srli_epi16(_mm_and_si128(hbits_1, hmask), bit++), 4);
            q5_0  = _mm_add_epi8(q5l_0, q5h_0);
            q5_1  = _mm_add_epi8(q5l_1, q5h_1);
            hmask = _mm_slli_epi16(hmask, 1);

            q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            __m128i p16_2 = _mm_maddubs_epi16(q5_0, q8_0);
            __m128i p16_3 = _mm_maddubs_epi16(q5_1, q8_1);
            p16_2 = _mm_madd_epi16(scale_1, p16_2);
            p16_3 = _mm_madd_epi16(scale_1, p16_3);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));

        }

        __m256 vd = _mm256_set1_ps(d);
        __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(vd, _mm256_cvtepi32_ps(sumi)), acc);

    }

    *s = hsum_float_8(acc) + summs;

#elif defined __riscv_v_intrinsic

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    float sumf = 0;
    float sums = 0.0;

    size_t vl;

    for (int i = 0; i < nb; ++i) {

        vl = 8;

        const uint8_t * restrict q5 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;

        vint16mf2_t q8sums_0 = __riscv_vlse16_v_i16mf2(y[i].bsums, 4, vl);
        vint16mf2_t q8sums_1 = __riscv_vlse16_v_i16mf2(y[i].bsums+1, 4, vl);
        vint16mf2_t q8sums = __riscv_vadd_vv_i16mf2(q8sums_0, q8sums_1, vl);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        vuint8mf4_t mins8 = __riscv_vle8_v_u8mf4(mins, vl);
        vint16mf2_t v_mins = __riscv_vreinterpret_v_u16mf2_i16mf2(__riscv_vzext_vf2_u16mf2(mins8, vl));
        vint32m1_t prod = __riscv_vwmul_vv_i32m1(q8sums, v_mins, vl);

        vint32m1_t sumi = __riscv_vredsum_vs_i32m1_i32m1(prod, __riscv_vmv_v_x_i32m1(0, 1), vl);
        sumf -= dmin * __riscv_vmv_x_s_i32m1_i32(sumi);

        vl = 32;
        int32_t aux32 = 0;
        int is = 0;

        uint8_t m = 1;
        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);
        vuint8m1_t vqh = __riscv_vle8_v_u8m1(hm, vl);

        for (int j = 0; j < QK_K/64; ++j) {
            // load Q5 and Q8
            vuint8m1_t q5_x = __riscv_vle8_v_u8m1(q5, vl);
            vint8m1_t  q8_y1 = __riscv_vle8_v_i8m1(q8, vl);
            vint8m1_t  q8_y2 = __riscv_vle8_v_i8m1(q8+32, vl);

            // compute mask for addition
            vint8m1_t q5_a = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(q5_x, 0x0F, vl));
            vuint8m1_t qh_m1 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_1 = __riscv_vmsne_vx_u8m1_b8(qh_m1, 0, vl);
            vint8m1_t q5_m1 = __riscv_vadd_vx_i8m1_mu(vmask_1, q5_a, q5_a, 16, vl);
            m <<= 1;

            vint8m1_t q5_l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(q5_x, 0x04, vl));
            vuint8m1_t qh_m2 = __riscv_vand_vx_u8m1(vqh, m, vl);
            vbool8_t vmask_2 = __riscv_vmsne_vx_u8m1_b8(qh_m2, 0, vl);
            vint8m1_t q5_m2 = __riscv_vadd_vx_i8m1_mu(vmask_2, q5_l, q5_l, 16, vl);
            m <<= 1;

            vint16m2_t v0 = __riscv_vwmul_vv_i16m2(q5_m1, q8_y1, vl);
            vint16m2_t v1 = __riscv_vwmul_vv_i16m2(q5_m2, q8_y2, vl);

            vint32m4_t vs1 = __riscv_vwmul_vx_i32m4(v0, scales[is++], vl);
            vint32m4_t vs2 = __riscv_vwmul_vx_i32m4(v1, scales[is++], vl);

            vint32m1_t vacc1 = __riscv_vredsum_vs_i32m4_i32m1(vs1, vzero, vl);
            vint32m1_t vacc2 = __riscv_vredsum_vs_i32m4_i32m1(vs2, vzero, vl);

            aux32 += __riscv_vmv_x_s_i32m1_i32(vacc1) + __riscv_vmv_x_s_i32m1_i32(vacc2);
            q5 += 32;    q8 += 64;

        }

        vfloat32m1_t vaux = __riscv_vfmul_vf_f32m1(__riscv_vfmv_v_f_f32m1(aux32, 1), d, 1);
        sums += __riscv_vfmv_f_s_f32m1_f32(vaux);

    }

    *s = sumf+sums;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed char lowMask1 = vec_splats((int8_t)0x3f);
    const vector signed char lowMask2 = vec_splats((int8_t)0x30);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v1 = vec_splats((unsigned char)0x1);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v3 = vec_splats((unsigned char)0x3);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector float vxmin = vec_splats(GGML_FP16_TO_FP32(x[i].dmin));
        vector float vdmin = vec_mul(vxmin, vyd);

        UNUSED(kmask1);
        UNUSED(kmask2);
        UNUSED(kmask3);
        UNUSED(utmp);

        vector signed char u0 = (vector signed char)vec_xl_len(x[i].scales, 8);
        vector signed char u1 = vec_and(vec_sr(u0, v2), lowMask2);
        vector signed char u2 = (vector signed char)vec_xl_len(x[i].scales + 8, 4);
        vector signed char u3 = vec_sr(u2, v4);

        vector signed char u30 = u1;
        vector signed char u31 = (vector signed char)vec_mergeh((vector signed int)vec_and(u2, lowMask), (vector signed int)u3);

        u1 = vec_and(u0, lowMask1);
        u2 = vec_or(u30, u31);

        vector signed char utmps = (vector signed char)vec_mergeh((vector signed int)u1, (vector signed int)u2);

        vector signed short q8ysums0 = vec_xl( 0, y[i].bsums);
        vector signed short q8ysums1 = vec_xl(16, y[i].bsums);

        vector signed short vscales = vec_unpackh(utmps);

        vector signed short q5xmins = vec_unpackl(utmps);
        vector signed short q5xmins0 = vec_mergeh(q5xmins, q5xmins);
        vector signed short q5xmins1 = vec_mergel(q5xmins, q5xmins);

        vector signed int prod0 = vec_mule(q5xmins0, q8ysums0);
        vector signed int prod1 = vec_mule(q5xmins1, q8ysums1);
        vector signed int prod2 = vec_mulo(q5xmins0, q8ysums0);
        vector signed int prod3 = vec_mulo(q5xmins1, q8ysums1);

        vsumf0 = vec_nmsub(vec_ctf(prod0, 0), vdmin, vsumf0);
        vsumf1 = vec_nmsub(vec_ctf(prod1, 0), vdmin, vsumf1);
        vsumf2 = vec_nmsub(vec_ctf(prod2, 0), vdmin, vsumf2);
        vsumf3 = vec_nmsub(vec_ctf(prod3, 0), vdmin, vsumf3);

        vector signed char qxhs0 = (vector signed char)vec_xl( 0, x[i].qh);
        vector signed char qxhs1 = (vector signed char)vec_xl(16, x[i].qh);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q5, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q5);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q5);
            q5 += 32;

            vector signed char qxs00 = vec_and(qxs0, lowMask);
            vector signed char qxs01 = vec_sr(qxs0, v4);
            vector signed char qxs10 = vec_and(qxs1, lowMask);
            vector signed char qxs11 = vec_sr(qxs1, v4);

            vector signed char q5h00 = vec_sl(vec_and((vector signed char)v1, qxhs0), v4);
            vector signed char q5h01 = vec_sl(vec_and((vector signed char)v2, qxhs0), v3);
            vector signed char q5h10 = vec_sl(vec_and((vector signed char)v1, qxhs1), v4);
            vector signed char q5h11 = vec_sl(vec_and((vector signed char)v2, qxhs1), v3);
            qxhs0 = vec_sr(qxhs0, v2);
            qxhs1 = vec_sr(qxhs1, v2);

            vector unsigned char q5x00 = (vector unsigned char)vec_or(q5h00, qxs00);
            vector unsigned char q5x01 = (vector unsigned char)vec_or(q5h01, qxs01);
            vector unsigned char q5x10 = (vector unsigned char)vec_or(q5h10, qxs10);
            vector unsigned char q5x11 = (vector unsigned char)vec_or(q5h11, qxs11);

            vector signed char q8y00 = vec_xl( 0, q8);
            vector signed char q8y10 = vec_xl(16, q8);
            vector signed char q8y01 = vec_xl(32, q8);
            vector signed char q8y11 = vec_xl(48, q8);
            q8 += 64;

            vector signed int qv00 = vec_msum(q8y00, q5x00, v0);
            vector signed int qv01 = vec_msum(q8y01, q5x01, v0);
            vector signed int qv10 = vec_msum(q8y10, q5x10, v0);
            vector signed int qv11 = vec_msum(q8y11, q5x11, v0);

            vector signed int vscales_h = vec_unpackh(vscales);
            vector signed int vs0 = vec_splat(vscales_h, 0);
            vector signed int vs1 = vec_splat(vscales_h, 1);
            vscales = vec_sld(vscales, vscales, 12);

            vsumi0 = vec_add(vec_mul(qv00, vs0), vsumi0);
            vsumi1 = vec_add(vec_mul(qv10, vs0), vsumi1);
            vsumi2 = vec_add(vec_mul(qv01, vs1), vsumi2);
            vsumi3 = vec_add(vec_mul(qv11, vs1), vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx
    GGML_UNUSED(kmask1);
    GGML_UNUSED(kmask2);
    GGML_UNUSED(kmask3);

    const __m256i m4 = __lasx_xvreplgr2vr_b(0xF);
    const __m128i mzero = __lsx_vldi(0);
    const __m256i mone  = __lasx_xvreplgr2vr_b(1);

    __m256 acc = (__m256)__lasx_xvldi(0);

    float summs = 0.f;

   for (int i = 0; i < nb; ++i) {

        const uint8_t * restrict q5 = x[i].qs;
        const int8_t  * restrict q8 = y[i].qs;

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        const float dmin = -y[i].d * GGML_FP16_TO_FP32(x[i].dmin);

        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales = lasx_extu8_16(lsx_set_w(utmp[3], utmp[2], utmp[1], utmp[0]));

        const __m256i q8sums = __lasx_xvld((const __m256i*)y[i].bsums, 0);
        const __m128i q8s = lsx_hadd_h(lasx_extracti128(q8sums, 0), lasx_extracti128(q8sums, 1));
        const __m128i prod = lsx_madd_h(lasx_extracti128(mins_and_scales, 1), q8s);
        const __m128i hsum = lsx_hadd_w(lsx_hadd_w(prod, mzero), mzero);
        summs += dmin * __lsx_vpickve2gr_w(hsum, 0);    //TODO check

        const __m128i sc128  = lasx_extracti128(mins_and_scales, 0);
        const __m256i scales = lasx_insertf128(sc128, sc128);

        const __m256i hbits = __lasx_xvld((const __m256i*)x[i].qh, 0);
        __m256i hmask = mone;

        __m256i sumi = __lasx_xvldi(0);

        int bit = 0;
        __m256i xvbit;

        for (int j = 0; j < QK_K/64; ++j) {

            const __m256i scale_0 = lasx_shuffle_b(scales, get_scale_shuffle_k4(2*j+0));
            const __m256i scale_1 = lasx_shuffle_b(scales, get_scale_shuffle_k4(2*j+1));

            const __m256i q5bits = __lasx_xvld((const __m256i*)q5, 0); q5 += 32;

            xvbit = __lasx_xvreplgr2vr_h(bit++);
            const __m256i q5l_0 = __lasx_xvand_v(q5bits, m4);
            const __m256i q5h_0 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvand_v(hbits, hmask), xvbit), 4);
            const __m256i q5_0  = __lasx_xvadd_b(q5l_0, q5h_0);
            hmask = __lasx_xvslli_h(hmask, 1);

            xvbit = __lasx_xvreplgr2vr_h(bit++);
            const __m256i q5l_1 = __lasx_xvand_v(__lasx_xvsrli_h(q5bits, 4), m4);
            const __m256i q5h_1 = __lasx_xvslli_h(__lasx_xvsrl_h(__lasx_xvand_v(hbits, hmask), xvbit), 4);
            const __m256i q5_1  = __lasx_xvadd_b(q5l_1, q5h_1);
            hmask = __lasx_xvslli_h(hmask, 1);

            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            __m256i p16_0 = lasx_maddubs_h(q5_0, q8_0);
            __m256i p16_1 = lasx_maddubs_h(q5_1, q8_1);

            p16_0 = lasx_madd_h(scale_0, p16_0);
            p16_1 = lasx_madd_h(scale_1, p16_1);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_1));

        }

        __m256 vd = __lasx_xvreplfr2vr_s(d);
        acc = __lasx_xvfmadd_s(vd, __lasx_xvffint_s_w(sumi), acc);

    }

    *s = hsum_float_8(acc) + summs;

#else

    const uint8_t * scales = (const uint8_t*)&utmp[0];
    const uint8_t * mins   = (const uint8_t*)&utmp[2];

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].qs;
        const uint8_t * restrict hm = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        uint8_t m = 1;
        for (int j = 0; j < QK_K/64; ++j) {
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l] & 0xF);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4[l]  >> 4);
            for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m ? 16 : 0);
            a += 32; m <<= 1;
            q4 += 32;
        }
        memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        int sumi = 0;
        for (int j = 0; j < QK_K/16; ++j) sumi += y[i].bsums[j] * mins[j/2];
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            int32_t scale = scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
        const float dmin = GGML_FP16_TO_FP32(x[i].dmin) * y[i].d;
        sumf -= dmin * sumi;
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

void ggml_vec_dot_q6_K_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_q6_K * restrict x = vx;
    const block_q8_K * restrict y = vy;

    const int nb = n / QK_K;

#ifdef __ARM_NEON
    float sum = 0;

    const uint8x16_t m4b = vdupq_n_u8(0xF);
    const int32x4_t  vzero = vdupq_n_s32(0);
    //const int8x16_t  m32s = vdupq_n_s8(32);

    const uint8x16_t mone = vdupq_n_u8(3);

    ggml_int8x16x4_t q6bytes;
    ggml_uint8x16x4_t q6h;

    for (int i = 0; i < nb; ++i) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        const ggml_int16x8x2_t q8sums = ggml_vld1q_s16_x2(y[i].bsums);
        const int8x16_t scales = vld1q_s8(scale);
        const ggml_int16x8x2_t q6scales = {{vmovl_s8(vget_low_s8(scales)), vmovl_s8(vget_high_s8(scales))}};

        const int32x4_t prod = vaddq_s32(vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[0]), vget_low_s16 (q6scales.val[0])),
                                                   vmull_s16(vget_high_s16(q8sums.val[0]), vget_high_s16(q6scales.val[0]))),
                                         vaddq_s32(vmull_s16(vget_low_s16 (q8sums.val[1]), vget_low_s16 (q6scales.val[1])),
                                                   vmull_s16(vget_high_s16(q8sums.val[1]), vget_high_s16(q6scales.val[1]))));
        int32_t isum_mins = vaddvq_s32(prod);

        int32_t isum = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            ggml_uint8x16x2_t qhbits = ggml_vld1q_u8_x2(qh); qh += 32;
            ggml_uint8x16x4_t q6bits = ggml_vld1q_u8_x4(q6); q6 += 64;
            ggml_int8x16x4_t q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, qhbits.val[0]), 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, qhbits.val[1]), 4);
            uint8x16_t shifted = vshrq_n_u8(qhbits.val[0], 2);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 2);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[0], m4b), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[1], m4b), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[2], m4b), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q6bits.val[3], m4b), q6h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];

            scale += 4;

            q8bytes = ggml_vld1q_s8_x4(q8); q8 += 64;

            shifted = vshrq_n_u8(qhbits.val[0], 4);
            q6h.val[0] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 4);
            q6h.val[1] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[0], 6);
            q6h.val[2] = vshlq_n_u8(vandq_u8(mone, shifted), 4);
            shifted = vshrq_n_u8(qhbits.val[1], 6);
            q6h.val[3] = vshlq_n_u8(vandq_u8(mone, shifted), 4);

            //q6bytes.val[0] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0])), m32s);
            //q6bytes.val[1] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1])), m32s);
            //q6bytes.val[2] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2])), m32s);
            //q6bytes.val[3] = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3])), m32s);
            q6bytes.val[0] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[0], 4), q6h.val[0]));
            q6bytes.val[1] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[1], 4), q6h.val[1]));
            q6bytes.val[2] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[2], 4), q6h.val[2]));
            q6bytes.val[3] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(q6bits.val[3], 4), q6h.val[3]));

            isum += vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[0], q8bytes.val[0])) * scale[0] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[1], q8bytes.val[1])) * scale[1] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[2], q8bytes.val[2])) * scale[2] +
                    vaddvq_s32(ggml_vdotq_s32(vzero, q6bytes.val[3], q8bytes.val[3])) * scale[3];
            scale += 4;
        }
        //sum += isum * d_all * y[i].d;
        sum += d_all * y[i].d * (isum - 32 * isum_mins);

    }
    *s = sum;

#elif defined __AVX2__

    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32s = _mm256_set1_epi8(32);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);

        __m256i sumi = _mm256_setzero_si256();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            const __m256i q4bits1 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bits2 = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4bitsH = _mm256_loadu_si256((const __m256i*)qh); qh += 32;

            const __m256i q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bitsH, m2), 4);
            const __m256i q4h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            const __m256i q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            const __m256i q4_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
            __m256i q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
            __m256i q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
            __m256i q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

            __m256i p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
            __m256i p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
            __m256i p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
            __m256i p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

            p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
            p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
            p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
            p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

            p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));

        }

        acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __AVX__

    const __m128i m3 = _mm_set1_epi8(3);
    const __m128i m15 = _mm_set1_epi8(15);

    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        // handle the q6_k -32 offset separately using bsums
        const __m128i q8sums_0 = _mm_loadu_si128((const __m128i*)y[i].bsums);
        const __m128i q8sums_1 = _mm_loadu_si128((const __m128i*)y[i].bsums + 1);
        const __m128i scales = _mm_loadu_si128((const __m128i*)x[i].scales);
        const __m128i scales_16_0 = _mm_cvtepi8_epi16(scales);
        const __m128i scales_16_1 = _mm_cvtepi8_epi16(_mm_bsrli_si128(scales, 8));
        const __m128i q8sclsub_0 = _mm_slli_epi32(_mm_madd_epi16(q8sums_0, scales_16_0), 5);
        const __m128i q8sclsub_1 = _mm_slli_epi32(_mm_madd_epi16(q8sums_1, scales_16_1), 5);

        __m128i sumi_0 = _mm_setzero_si128();
        __m128i sumi_1 = _mm_setzero_si128();

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i q4bitsH_0 = _mm_loadu_si128((const __m128i*)qh); qh += 16;
            const __m128i q4bitsH_1 = _mm_loadu_si128((const __m128i*)qh); qh += 16;

            const __m128i q4h_0 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, m3), 4);
            const __m128i q4h_1 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, m3), 4);
            const __m128i q4h_2 = _mm_slli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(12)), 2);
            const __m128i q4h_3 = _mm_slli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(12)), 2);
            const __m128i q4h_4 = _mm_and_si128(q4bitsH_0, _mm_set1_epi8(48));
            const __m128i q4h_5 = _mm_and_si128(q4bitsH_1, _mm_set1_epi8(48));
            const __m128i q4h_6 = _mm_srli_epi16(_mm_and_si128(q4bitsH_0, _mm_set1_epi8(-64)), 2);
            const __m128i q4h_7 = _mm_srli_epi16(_mm_and_si128(q4bitsH_1, _mm_set1_epi8(-64)), 2);

            const __m128i q4bits1_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits1_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_0 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;
            const __m128i q4bits2_1 = _mm_loadu_si128((const __m128i*)q4); q4 += 16;

            const __m128i q4_0 = _mm_or_si128(_mm_and_si128(q4bits1_0, m15), q4h_0);
            const __m128i q4_1 = _mm_or_si128(_mm_and_si128(q4bits1_1, m15), q4h_1);
            const __m128i q4_2 = _mm_or_si128(_mm_and_si128(q4bits2_0, m15), q4h_2);
            const __m128i q4_3 = _mm_or_si128(_mm_and_si128(q4bits2_1, m15), q4h_3);
            const __m128i q4_4 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_0, 4), m15), q4h_4);
            const __m128i q4_5 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits1_1, 4), m15), q4h_5);
            const __m128i q4_6 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_0, 4), m15), q4h_6);
            const __m128i q4_7 = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(q4bits2_1, 4), m15), q4h_7);

            const __m128i q8_0 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_1 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_2 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_3 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_4 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_5 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_6 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;
            const __m128i q8_7 = _mm_loadu_si128((const __m128i*)q8); q8 += 16;

            __m128i p16_0 = _mm_maddubs_epi16(q4_0, q8_0);
            __m128i p16_1 = _mm_maddubs_epi16(q4_1, q8_1);
            __m128i p16_2 = _mm_maddubs_epi16(q4_2, q8_2);
            __m128i p16_3 = _mm_maddubs_epi16(q4_3, q8_3);
            __m128i p16_4 = _mm_maddubs_epi16(q4_4, q8_4);
            __m128i p16_5 = _mm_maddubs_epi16(q4_5, q8_5);
            __m128i p16_6 = _mm_maddubs_epi16(q4_6, q8_6);
            __m128i p16_7 = _mm_maddubs_epi16(q4_7, q8_7);

            const __m128i scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));
            is += 4;

            p16_0 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_0), p16_0);
            p16_1 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_0, 8)), p16_1);
            p16_2 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_1), p16_2);
            p16_3 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_1, 8)), p16_3);
            p16_4 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_2), p16_4);
            p16_5 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_2, 8)), p16_5);
            p16_6 = _mm_madd_epi16(_mm_cvtepi8_epi16(scale_3), p16_6);
            p16_7 = _mm_madd_epi16(_mm_cvtepi8_epi16(_mm_bsrli_si128(scale_3, 8)), p16_7);

            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_0, p16_2));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_1, p16_3));
            sumi_0 = _mm_add_epi32(sumi_0, _mm_add_epi32(p16_4, p16_6));
            sumi_1 = _mm_add_epi32(sumi_1, _mm_add_epi32(p16_5, p16_7));

        }

        sumi_0 = _mm_sub_epi32(sumi_0, q8sclsub_0);
        sumi_1 = _mm_sub_epi32(sumi_1, q8sclsub_1);
        const __m256i sumi = MM256_SET_M128I(sumi_1, sumi_0);
        acc = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi)), acc);
    }

    *s = hsum_float_8(acc);

#elif defined __riscv_v_intrinsic

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;

        const int8_t * restrict scale = x[i].scales;

        size_t vl;

        vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, 1);

        int sum_t = 0;
        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            vl = 32;

            // load qh
            vuint8m1_t qh_x = __riscv_vle8_v_u8m1(qh, vl);

            // load Q6
            vuint8m1_t q6_0 = __riscv_vle8_v_u8m1(q6, vl);
            vuint8m1_t q6_1 = __riscv_vle8_v_u8m1(q6+32, vl);

            vuint8m1_t q6a_0 = __riscv_vand_vx_u8m1(q6_0, 0x0F, vl);
            vuint8m1_t q6a_1 = __riscv_vand_vx_u8m1(q6_1, 0x0F, vl);
            vuint8m1_t q6s_0 = __riscv_vsrl_vx_u8m1(q6_0, 0x04, vl);
            vuint8m1_t q6s_1 = __riscv_vsrl_vx_u8m1(q6_1, 0x04, vl);

            vuint8m1_t qh_0 = __riscv_vand_vx_u8m1(qh_x, 0x03, vl);
            vuint8m1_t qh_1 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x2, vl), 0x03 , vl);
            vuint8m1_t qh_2 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x4, vl), 0x03 , vl);
            vuint8m1_t qh_3 = __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(qh_x, 0x6, vl), 0x03 , vl);

            vuint8m1_t qhi_0 = __riscv_vor_vv_u8m1(q6a_0, __riscv_vsll_vx_u8m1(qh_0, 0x04, vl), vl);
            vuint8m1_t qhi_1 = __riscv_vor_vv_u8m1(q6a_1, __riscv_vsll_vx_u8m1(qh_1, 0x04, vl), vl);
            vuint8m1_t qhi_2 = __riscv_vor_vv_u8m1(q6s_0, __riscv_vsll_vx_u8m1(qh_2, 0x04, vl), vl);
            vuint8m1_t qhi_3 = __riscv_vor_vv_u8m1(q6s_1, __riscv_vsll_vx_u8m1(qh_3, 0x04, vl), vl);

            vint8m1_t a_0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_0), 32, vl);
            vint8m1_t a_1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_1), 32, vl);
            vint8m1_t a_2 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_2), 32, vl);
            vint8m1_t a_3 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(qhi_3), 32, vl);

            // load Q8 and take product
            vint16m2_t va_q_0 = __riscv_vwmul_vv_i16m2(a_0, __riscv_vle8_v_i8m1(q8, vl), vl);
            vint16m2_t va_q_1 = __riscv_vwmul_vv_i16m2(a_1, __riscv_vle8_v_i8m1(q8+32, vl), vl);
            vint16m2_t va_q_2 = __riscv_vwmul_vv_i16m2(a_2, __riscv_vle8_v_i8m1(q8+64, vl), vl);
            vint16m2_t va_q_3 = __riscv_vwmul_vv_i16m2(a_3, __riscv_vle8_v_i8m1(q8+96, vl), vl);

            vl = 16;

            vint32m2_t vaux_0 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 0), scale[is+0], vl);
            vint32m2_t vaux_1 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_0, 1), scale[is+1], vl);
            vint32m2_t vaux_2 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 0), scale[is+2], vl);
            vint32m2_t vaux_3 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_1, 1), scale[is+3], vl);
            vint32m2_t vaux_4 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 0), scale[is+4], vl);
            vint32m2_t vaux_5 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_2, 1), scale[is+5], vl);
            vint32m2_t vaux_6 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 0), scale[is+6], vl);
            vint32m2_t vaux_7 = __riscv_vwmul_vx_i32m2(__riscv_vget_v_i16m2_i16m1(va_q_3, 1), scale[is+7], vl);

            vint32m1_t isum0 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_0, vaux_1, vl), vzero, vl);
            vint32m1_t isum1 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_2, vaux_3, vl), isum0, vl);
            vint32m1_t isum2 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_4, vaux_5, vl), isum1, vl);
            vint32m1_t isum3 = __riscv_vredsum_vs_i32m2_i32m1(__riscv_vadd_vv_i32m2(vaux_6, vaux_7, vl), isum2, vl);

            sum_t += __riscv_vmv_x_s_i32m1_i32(isum3);

            q6 += 64;   qh += 32;   q8 += 128;   is=8;

        }

        sumf += d * sum_t;

    }

    *s = sumf;

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v2 = vec_splats((unsigned char)0x2);
    const vector unsigned char v3 = vec_splats((unsigned char)0x3);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);
    const vector unsigned char v6 = vec_splats((unsigned char)0x6);
    const vector signed char off = vec_splats((signed char)0x20);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;
        vector signed int vsumi4 = v0;
        vector signed int vsumi5 = v0;
        vector signed int vsumi6 = v0;
        vector signed int vsumi7 = v0;

        const uint8_t * restrict q6 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict qs = x[i].scales;
        const int8_t  * restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/128; ++j) {
            __builtin_prefetch(q6, 0, 0);
            __builtin_prefetch(qh, 0, 0);
            __builtin_prefetch(q8, 0, 0);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q6);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q6);
            vector signed char qxs2 = (vector signed char)vec_xl(32, q6);
            vector signed char qxs3 = (vector signed char)vec_xl(48, q6);
            q6 += 64;

            vector signed char qxs00 = vec_and(qxs0, lowMask);
            vector signed char qxs01 = vec_sr(qxs0, v4);
            vector signed char qxs10 = vec_and(qxs1, lowMask);
            vector signed char qxs11 = vec_sr(qxs1, v4);
            vector signed char qxs20 = vec_and(qxs2, lowMask);
            vector signed char qxs21 = vec_sr(qxs2, v4);
            vector signed char qxs30 = vec_and(qxs3, lowMask);
            vector signed char qxs31 = vec_sr(qxs3, v4);

            vector signed char qxhs0 = (vector signed char)vec_xl( 0, qh);
            vector signed char qxhs1 = (vector signed char)vec_xl(16, qh);
            qh += 32;

            vector signed char qxh00 = vec_sl(vec_and((vector signed char)v3, qxhs0), v4);
            vector signed char qxh01 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v4)), v4);
            vector signed char qxh10 = vec_sl(vec_and((vector signed char)v3, qxhs1), v4);
            vector signed char qxh11 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v4)), v4);
            vector signed char qxh20 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v2)), v4);
            vector signed char qxh21 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs0, v6)), v4);
            vector signed char qxh30 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v2)), v4);
            vector signed char qxh31 = vec_sl(vec_and((vector signed char)v3, vec_sr(qxhs1, v6)), v4);

            vector signed char q6x00 = vec_sub(vec_or(qxh00, qxs00), off);
            vector signed char q6x01 = vec_sub(vec_or(qxh01, qxs01), off);
            vector signed char q6x10 = vec_sub(vec_or(qxh10, qxs10), off);
            vector signed char q6x11 = vec_sub(vec_or(qxh11, qxs11), off);
            vector signed char q6x20 = vec_sub(vec_or(qxh20, qxs20), off);
            vector signed char q6x21 = vec_sub(vec_or(qxh21, qxs21), off);
            vector signed char q6x30 = vec_sub(vec_or(qxh30, qxs30), off);
            vector signed char q6x31 = vec_sub(vec_or(qxh31, qxs31), off);

            vector signed char q8y00 = vec_xl(  0, q8);
            vector signed char q8y10 = vec_xl( 16, q8);
            vector signed char q8y20 = vec_xl( 32, q8);
            vector signed char q8y30 = vec_xl( 48, q8);
            vector signed char q8y01 = vec_xl( 64, q8);
            vector signed char q8y11 = vec_xl( 80, q8);
            vector signed char q8y21 = vec_xl( 96, q8);
            vector signed char q8y31 = vec_xl(112, q8);
            q8 += 128;

            vector signed short qv00 = vec_add(vec_mule(q6x00, q8y00), vec_mulo(q6x00, q8y00));
            vector signed short qv10 = vec_add(vec_mule(q6x10, q8y10), vec_mulo(q6x10, q8y10));
            vector signed short qv20 = vec_add(vec_mule(q6x20, q8y20), vec_mulo(q6x20, q8y20));
            vector signed short qv30 = vec_add(vec_mule(q6x30, q8y30), vec_mulo(q6x30, q8y30));
            vector signed short qv01 = vec_add(vec_mule(q6x01, q8y01), vec_mulo(q6x01, q8y01));
            vector signed short qv11 = vec_add(vec_mule(q6x11, q8y11), vec_mulo(q6x11, q8y11));
            vector signed short qv21 = vec_add(vec_mule(q6x21, q8y21), vec_mulo(q6x21, q8y21));
            vector signed short qv31 = vec_add(vec_mule(q6x31, q8y31), vec_mulo(q6x31, q8y31));

            vector signed short vscales = vec_unpackh(vec_xl_len(qs, 8));
            qs += 8;

            vector signed short vs0 = vec_splat(vscales, 0);
            vector signed short vs1 = vec_splat(vscales, 1);
            vector signed short vs2 = vec_splat(vscales, 2);
            vector signed short vs3 = vec_splat(vscales, 3);
            vector signed short vs4 = vec_splat(vscales, 4);
            vector signed short vs5 = vec_splat(vscales, 5);
            vector signed short vs6 = vec_splat(vscales, 6);
            vector signed short vs7 = vec_splat(vscales, 7);

            vsumi0 = vec_msum(qv00, vs0, vsumi0);
            vsumi1 = vec_msum(qv01, vs4, vsumi1);
            vsumi2 = vec_msum(qv10, vs1, vsumi2);
            vsumi3 = vec_msum(qv11, vs5, vsumi3);
            vsumi4 = vec_msum(qv20, vs2, vsumi4);
            vsumi5 = vec_msum(qv21, vs6, vsumi5);
            vsumi6 = vec_msum(qv30, vs3, vsumi6);
            vsumi7 = vec_msum(qv31, vs7, vsumi7);
        }

        vsumi0 = vec_add(vsumi0, vsumi4);
        vsumi1 = vec_add(vsumi1, vsumi5);
        vsumi2 = vec_add(vsumi2, vsumi6);
        vsumi3 = vec_add(vsumi3, vsumi7);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined __loongarch_asx

    const __m256i m4 = __lasx_xvreplgr2vr_b(0xF);
    const __m256i m2 = __lasx_xvreplgr2vr_b(3);
    const __m256i m32s = __lasx_xvreplgr2vr_b(32);

    __m256 acc = (__m256)__lasx_xvldi(0);

    for (int i = 0; i < nb; ++i) {

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const int8_t  * restrict q8 = y[i].qs;

        const __m128i scales = __lsx_vld((const __m128i*)x[i].scales, 0);

        __m256i sumi = __lasx_xvldi(0);

        int is = 0;

        for (int j = 0; j < QK_K/128; ++j) {

            const __m128i scale_0 = lsx_shuffle_b(scales, get_scale_shuffle(is + 0));
            const __m128i scale_1 = lsx_shuffle_b(scales, get_scale_shuffle(is + 1));
            const __m128i scale_2 = lsx_shuffle_b(scales, get_scale_shuffle(is + 2));
            const __m128i scale_3 = lsx_shuffle_b(scales, get_scale_shuffle(is + 3));
            is += 4;

            const __m256i q4bits1 = __lasx_xvld((const __m256i*)q4, 0); q4 += 32;
            const __m256i q4bits2 = __lasx_xvld((const __m256i*)q4, 0); q4 += 32;
            const __m256i q4bitsH = __lasx_xvld((const __m256i*)qh, 0); qh += 32;

            const __m256i q4h_0 = __lasx_xvslli_h(__lasx_xvand_v(q4bitsH, m2), 4);
            const __m256i q4h_1 = __lasx_xvslli_h(__lasx_xvand_v(__lasx_xvsrli_h(q4bitsH, 2), m2), 4);
            const __m256i q4h_2 = __lasx_xvslli_h(__lasx_xvand_v(__lasx_xvsrli_h(q4bitsH, 4), m2), 4);
            const __m256i q4h_3 = __lasx_xvslli_h(__lasx_xvand_v(__lasx_xvsrli_h(q4bitsH, 6), m2), 4);

            const __m256i q4_0 = __lasx_xvor_v(__lasx_xvand_v(q4bits1, m4), q4h_0);
            const __m256i q4_1 = __lasx_xvor_v(__lasx_xvand_v(q4bits2, m4), q4h_1);
            const __m256i q4_2 = __lasx_xvor_v(__lasx_xvand_v(__lasx_xvsrli_h(q4bits1, 4), m4), q4h_2);
            const __m256i q4_3 = __lasx_xvor_v(__lasx_xvand_v(__lasx_xvsrli_h(q4bits2, 4), m4), q4h_3);

            const __m256i q8_0 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            __m256i q8s_0 = lasx_maddubs_h(m32s, q8_0);
            __m256i q8s_1 = lasx_maddubs_h(m32s, q8_1);
            __m256i q8s_2 = lasx_maddubs_h(m32s, q8_2);
            __m256i q8s_3 = lasx_maddubs_h(m32s, q8_3);

            __m256i p16_0 = lasx_maddubs_h(q4_0, q8_0);
            __m256i p16_1 = lasx_maddubs_h(q4_1, q8_1);
            __m256i p16_2 = lasx_maddubs_h(q4_2, q8_2);
            __m256i p16_3 = lasx_maddubs_h(q4_3, q8_3);

            p16_0 = __lasx_xvsub_h(p16_0, q8s_0);
            p16_1 = __lasx_xvsub_h(p16_1, q8s_1);
            p16_2 = __lasx_xvsub_h(p16_2, q8s_2);
            p16_3 = __lasx_xvsub_h(p16_3, q8s_3);

            p16_0 = lasx_madd_h(lasx_ext8_16(scale_0), p16_0);
            p16_1 = lasx_madd_h(lasx_ext8_16(scale_1), p16_1);
            p16_2 = lasx_madd_h(lasx_ext8_16(scale_2), p16_2);
            p16_3 = lasx_madd_h(lasx_ext8_16(scale_3), p16_3);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_0, p16_1));
            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p16_2, p16_3));
        }

        acc = __lasx_xvfmadd_s((__m256)__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), acc);
    }

    *s = hsum_float_8(acc);

#else

    int8_t  aux8[QK_K];
    int16_t aux16[8];
    float   sums [8];
    int32_t aux32[8];
    memset(sums, 0, 8*sizeof(float));

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const uint8_t * restrict q4 = x[i].ql;
        const uint8_t * restrict qh = x[i].qh;
        const  int8_t * restrict q8 = y[i].qs;
        memset(aux32, 0, 8*sizeof(int32_t));
        int8_t * restrict a = aux8;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                a[l +  0] = (int8_t)((q4[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                a[l + 32] = (int8_t)((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                a[l + 64] = (int8_t)((q4[l +  0] >>  4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                a[l + 96] = (int8_t)((q4[l + 32] >>  4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            }
            a  += 128;
            q4 += 64;
            qh += 32;
        }
        a = aux8;
        int is = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            int scale = x[i].scales[is++];
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
            for (int l = 0; l < 8; ++l) aux16[l] = q8[l] * a[l];
            for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
            q8 += 8; a += 8;
        }
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        for (int l = 0; l < 8; ++l) sums[l] += d * aux32[l];
    }
    for (int l = 0; l < 8; ++l) sumf += sums[l];
    *s = sumf;
#endif
}

#if defined (__AVX__) || defined (__AVX2__) || defined (__ARM_NEON) || defined (__POWER9_VECTOR__) || defined(__loongarch_asx)
static const int8_t keven_signs_q2xs[1024] = {
     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,
     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,
     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,
     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,
     1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1, -1,
     1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1,  1,
     1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1,  1,
     1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1, -1,
     1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1,  1, -1,  1,  1, -1, -1,  1,  1,  1, -1,  1, -1,
     1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1,  1,
     1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,
     1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1, -1,
     1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1,  1,
     1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,
     1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1,  1,  1,  1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1, -1,
     1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1,  1,
     1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1,  1, -1, -1,
     1,  1, -1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1,  1,
     1,  1,  1, -1,  1,  1, -1,  1, -1,  1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1,  1,
     1,  1, -1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1, -1,
     1,  1,  1,  1, -1,  1, -1,  1, -1,  1,  1,  1, -1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1,  1,
     1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1, -1,
     1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1,  1, -1,  1,  1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1, -1,
     1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1,  1,
     1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1, -1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1,  1,
     1,  1, -1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1, -1, -1, -1,  1,  1, -1, -1, -1,
     1,  1,  1, -1,  1, -1, -1, -1, -1,  1,  1, -1,  1, -1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
     1,  1, -1, -1,  1, -1, -1,  1, -1,  1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1,  1,
     1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1, -1, -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1, -1, -1,  1,  1, -1, -1, -1, -1,
     1,  1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1, -1, -1, -1, -1,  1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1,  1,
     1,  1,  1, -1, -1, -1, -1,  1, -1,  1,  1, -1, -1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1,  1,
     1,  1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,  1,  1, -1, -1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1,
};
#endif

void ggml_vec_dot_iq2_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 0])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 1])));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 2])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 3])));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[ 8])), vld1_s8((const void *)(iq2xxs_grid + aux8[ 9])));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xxs_grid + aux8[10])), vld1_s8((const void *)(iq2xxs_grid + aux8[11])));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >>  7) & 127))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[3] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[3] >> 21) & 127))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]), q2u.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]), q2u.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[1] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[3] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.25f * sumf;

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m256i q2_1 = _mm256_set_epi64x(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xxs_grid[aux8[3]], iq2xxs_grid[aux8[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]]);
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    const vector int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint16_t * restrict q2 = x[i].qs;
        const int8_t  *  restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            uint32_t aux32[4];
            const uint8_t * aux8 = (const uint8_t *)aux32;

            memcpy(aux32, q2, 4*sizeof(uint32_t));
            q2 += 8;

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq2xxs_grid + aux8[ 0]), *(const int64_t *)(iq2xxs_grid + aux8[ 1])};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq2xxs_grid + aux8[ 2]), *(const int64_t *)(iq2xxs_grid + aux8[ 3])};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq2xxs_grid + aux8[ 8]), *(const int64_t *)(iq2xxs_grid + aux8[ 9])};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq2xxs_grid + aux8[10]), *(const int64_t *)(iq2xxs_grid + aux8[11])};

            vector signed long long vsigns0 = {*(const int64_t *)(signs64 + ((aux32[1] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >>  7) & 127))};
            vector signed long long vsigns1 = {*(const int64_t *)(signs64 + ((aux32[1] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[1] >> 21) & 127))};
            vector signed long long vsigns2 = {*(const int64_t *)(signs64 + ((aux32[3] >>  0) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >>  7) & 127))};
            vector signed long long vsigns3 = {*(const int64_t *)(signs64 + ((aux32[3] >> 14) & 127)), *(const int64_t *)(signs64 + ((aux32[3] >> 21) & 127))};

            vector signed char q2x0 = (vector signed char)vec_mul((vector signed char)vsigns0, (vector signed char)aux64x2_0);
            vector signed char q2x1 = (vector signed char)vec_mul((vector signed char)vsigns1, (vector signed char)aux64x2_1);
            vector signed char q2x2 = (vector signed char)vec_mul((vector signed char)vsigns2, (vector signed char)aux64x2_2);
            vector signed char q2x3 = (vector signed char)vec_mul((vector signed char)vsigns3, (vector signed char)aux64x2_3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            const uint16_t ls0 = aux32[1] >> 28;
            const uint16_t ls1 = aux32[3] >> 28;

            vector signed short vscales01 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales23 = vec_splats((int16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[4];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            memcpy(aux32, q2, 4*sizeof(uint32_t)); q2 += 8;

            const __m256i q2_1 = lasx_set_d(iq2xxs_grid[aux8[ 3]], iq2xxs_grid[aux8[ 2]], iq2xxs_grid[aux8[1]], iq2xxs_grid[aux8[0]]);
            const __m256i q2_2 = lasx_set_d(iq2xxs_grid[aux8[11]], iq2xxs_grid[aux8[10]], iq2xxs_grid[aux8[9]], iq2xxs_grid[aux8[8]]);
            const __m256i s2_1 = lasx_set_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i s2_2 = lasx_set_d(signs64[(aux32[3] >> 21) & 127], signs64[(aux32[3] >> 14) & 127],
                                                   signs64[(aux32[3] >>  7) & 127], signs64[(aux32[3] >>  0) & 127]);
            const __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            const __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = aux32[1] >> 28;
            const uint16_t ls2 = aux32[3] >> 28;
            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#else

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, q2, 2*sizeof(uint32_t));
            q2 += 4;
            const uint32_t ls = 2*(aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}

void ggml_vec_dot_iq2_xs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_xs * restrict x = vx;
    const block_q8_K   * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    ggml_int8x16x4_t q2u;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    int32x4x4_t scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;
        const uint8x8_t scales8 = vld1_u8(x[i].scales);
        const uint8x8_t scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        const uint8x8_t scales_h = vshr_n_u8(scales8, 4);
        uint8x16_t scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        const uint16x8_t scales1 = vmovl_u8(vget_low_u8(scales));
        const uint16x8_t scales2 = vmovl_u8(vget_high_u8(scales));
        scales32.val[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        scales32.val[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        scales32.val[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        scales32.val[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2u.val[0] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[0] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[1] & 511))));
            q2u.val[1] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[2] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[3] & 511))));
            q2u.val[2] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[4] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[5] & 511))));
            q2u.val[3] = vcombine_s8(vld1_s8((const void *)(iq2xs_grid + (q2[6] & 511))), vld1_s8((const void *)(iq2xs_grid + (q2[7] & 511))));
            q2s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[0] >> 9))), vld1_s8((const void *)(signs64 + (q2[1] >> 9))));
            q2s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[2] >> 9))), vld1_s8((const void *)(signs64 + (q2[3] >> 9))));
            q2s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[4] >> 9))), vld1_s8((const void *)(signs64 + (q2[5] >> 9))));
            q2s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + (q2[6] >> 9))), vld1_s8((const void *)(signs64 + (q2[7] >> 9))));
            q2u.val[0] = vmulq_s8(q2u.val[0], q2s.val[0]);
            q2u.val[1] = vmulq_s8(q2u.val[1], q2s.val[1]);
            q2u.val[2] = vmulq_s8(q2u.val[2], q2s.val[2]);
            q2u.val[3] = vmulq_s8(q2u.val[3], q2s.val[3]);
            const int32x4_t p1 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vdupq_n_s32(0), q2u.val[3], q8b.val[3]);
            const int32x4_t p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));
            sumi = vmlaq_s32(sumi, p, scales32.val[ib64]);
            q2 += 8;
        }
        sumf += d*vaddvq_s32(sumi);
    }
    *s = 0.125f * sumf;

#elif defined(__AVX2__)

    const __m256i mone = _mm256_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*)bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*)block_sign_shuffle_mask_2);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*)k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m256i q2_data = _mm256_loadu_si256((const __m256i*)q2);  q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);

            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);

            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;

            const __m256i q2_1 = _mm256_set_epi64x(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

            __m256i signs;
            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

            signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
            signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const __m256i dot3  = _mm256_maddubs_epi16(q2_3, q8s_3);
            const __m256i dot4  = _mm256_maddubs_epi16(q2_4, q8s_4);

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const __m128i mone = _mm_set1_epi8(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i bit_selector_mask_0 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes);
    const __m128i bit_selector_mask_1 = _mm_loadu_si128((const __m128i*)bit_selector_mask_bytes + 1);
    const __m128i block_sign_shuffle_1_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1);
    const __m128i block_sign_shuffle_1_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_1 + 1);
    const __m128i block_sign_shuffle_2_0 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2);
    const __m128i block_sign_shuffle_2_1 = _mm_loadu_si128((const __m128i*)block_sign_shuffle_mask_2 + 1);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m128i bit_helper_0 = _mm_loadu_si128((const __m128i*)k_bit_helper);
    const __m128i bit_helper_1 = _mm_loadu_si128((const __m128i*)k_bit_helper + 1);
    const __m128i m511 = _mm_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x(aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m128i q2_data_0 = _mm_loadu_si128((const __m128i*)q2);
            const __m128i q2_data_1 = _mm_loadu_si128((const __m128i*)q2 + 1);  q2 += 16;
            aux_gindex = MM256_SET_M128I(_mm_and_si128(q2_data_1, m511), _mm_and_si128(q2_data_0, m511));

            const __m128i partial_sign_bits_0 = _mm_srli_epi16(q2_data_0, 9);
            const __m128i partial_sign_bits_1 = _mm_srli_epi16(q2_data_1, 9);
            const __m128i partial_sign_bits_upper_0 = _mm_srli_epi16(q2_data_0, 13);
            const __m128i partial_sign_bits_upper_1 = _mm_srli_epi16(q2_data_1, 13);
            const __m128i partial_sign_bits_for_counting_0 = _mm_xor_si128(partial_sign_bits_0, partial_sign_bits_upper_0);
            const __m128i partial_sign_bits_for_counting_1 = _mm_xor_si128(partial_sign_bits_1, partial_sign_bits_upper_1);

            const __m128i odd_bits_0 = _mm_shuffle_epi8(bit_helper_0, partial_sign_bits_for_counting_0);
            const __m128i odd_bits_1 = _mm_shuffle_epi8(bit_helper_1, partial_sign_bits_for_counting_1);
            const __m128i full_sign_bits_0 = _mm_or_si128(partial_sign_bits_0, odd_bits_0);
            const __m128i full_sign_bits_1 = _mm_or_si128(partial_sign_bits_1, odd_bits_1);

            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_3_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_4_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i q2_1_0 = _mm_set_epi64x(iq2xs_grid[gindex[1]], iq2xs_grid[gindex[0]]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2xs_grid[gindex[3]], iq2xs_grid[gindex[2]]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2xs_grid[gindex[5]], iq2xs_grid[gindex[4]]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2xs_grid[gindex[7]], iq2xs_grid[gindex[6]]);
            const __m128i q2_3_0 = _mm_set_epi64x(iq2xs_grid[gindex[9]], iq2xs_grid[gindex[8]]);
            const __m128i q2_3_1 = _mm_set_epi64x(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]]);
            const __m128i q2_4_0 = _mm_set_epi64x(iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);
            const __m128i q2_4_1 = _mm_set_epi64x(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]]);

            // AVX2 full_signs_1 is full_sign_bits_0 here
            // AVX2 full_signs_2 is full_sign_bits_1 here
            __m128i signs_0, signs_1;
            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_0, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_1_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_3_0 = _mm_sign_epi8(q8_3_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_3_1 = _mm_sign_epi8(q8_3_1, _mm_or_si128(signs_1, mone));

            signs_0 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_0);
            signs_1 = _mm_shuffle_epi8(full_sign_bits_1, block_sign_shuffle_2_1);
            signs_0 = _mm_cmpeq_epi8(_mm_and_si128(signs_0, bit_selector_mask_0), bit_selector_mask_0);
            signs_1 = _mm_cmpeq_epi8(_mm_and_si128(signs_1, bit_selector_mask_1), bit_selector_mask_1);
            const __m128i q8s_4_0 = _mm_sign_epi8(q8_4_0, _mm_or_si128(signs_0, mone));
            const __m128i q8s_4_1 = _mm_sign_epi8(q8_4_1, _mm_or_si128(signs_1, mone));

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const __m128i dot3_0  = _mm_maddubs_epi16(q2_3_0, q8s_3_0);
            const __m128i dot3_1  = _mm_maddubs_epi16(q2_3_1, q8s_3_1);
            const __m128i dot4_0  = _mm_maddubs_epi16(q2_4_0, q8s_4_0);
            const __m128i dot4_1  = _mm_maddubs_epi16(q2_4_1, q8s_4_1);

            __m128i sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+0));
            const __m128i sc1_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc1_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+1));
            const __m128i sc2_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc2_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+2));
            const __m128i sc3_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc3_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));
            sc_tmp = _mm_shuffle_epi8(scales, get_scale_shuffle(ib32+3));
            const __m128i sc4_0 = _mm_cvtepi8_epi16(sc_tmp);
            const __m128i sc4_1 = _mm_cvtepi8_epi16(_mm_srli_si128(sc_tmp, 8));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot1_0, sc1_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot1_1, sc1_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot2_0, sc2_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot2_1, sc2_1));
            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_madd_epi16(dot3_0, sc3_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_madd_epi16(dot3_1, sc3_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_madd_epi16(dot4_0, sc4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_madd_epi16(dot4_1, sc4_1));
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__loongarch_asx)

    const __m256i mone = __lasx_xvreplgr2vr_b(1);
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i bit_selector_mask = __lasx_xvld((const __m256i*)bit_selector_mask_bytes, 0);
    const __m256i block_sign_shuffle_1 = __lasx_xvld((const __m256i*)block_sign_shuffle_mask_1, 0);
    const __m256i block_sign_shuffle_2 = __lasx_xvld((const __m256i*)block_sign_shuffle_mask_2, 0);

    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_helper = __lasx_xvld((const __m256i*)k_bit_helper, 0);
    const __m256i m511 = __lasx_xvreplgr2vr_h(511);
    const __m128i m4 = __lsx_vreplgr2vr_b(0xf);
    const __m128i m1 = __lsx_vreplgr2vr_b(1);

    uint64_t aux64;

    // somewhat hacky, but gives a significant boost in performance
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *)&aux_gindex;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const int8_t   * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = __lsx_vreplgr2vr_d(aux64);
        stmp = __lsx_vilvl_b( __lsx_vand_v(__lsx_vsrli_h(stmp, 4), m4), __lsx_vand_v(stmp, m4));
        const __m128i scales = __lsx_vadd_b(__lsx_vslli_h(stmp, 1), m1);

        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 4) {

            const __m256i q2_data = __lasx_xvld((const __m256i*)q2, 0);  q2 += 16;
            aux_gindex = __lasx_xvand_v(q2_data, m511);

            const __m256i partial_sign_bits = __lasx_xvsrli_h(q2_data, 9);
            const __m256i partial_sign_bits_upper = __lasx_xvsrli_h(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = __lasx_xvxor_v(partial_sign_bits, partial_sign_bits_upper);

            const __m256i odd_bits = lasx_shuffle_b(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = __lasx_xvor_v(partial_sign_bits, odd_bits);

            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_3 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_4 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;

            const __m256i q2_1 = lasx_set_d(iq2xs_grid[gindex[ 3]], iq2xs_grid[gindex[ 2]],
                                                   iq2xs_grid[gindex[ 1]], iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = lasx_set_d(iq2xs_grid[gindex[ 7]], iq2xs_grid[gindex[ 6]],
                                                   iq2xs_grid[gindex[ 5]], iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = lasx_set_d(iq2xs_grid[gindex[11]], iq2xs_grid[gindex[10]],
                                                   iq2xs_grid[gindex[ 9]], iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = lasx_set_d(iq2xs_grid[gindex[15]], iq2xs_grid[gindex[14]],
                                                   iq2xs_grid[gindex[13]], iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = lasx_extracti128(full_sign_bits, 0);
            const __m128i full_signs_h = lasx_extracti128(full_sign_bits, 1);
            const __m256i full_signs_1 = lasx_insertf128(full_signs_l, full_signs_l);
            const __m256i full_signs_2 = lasx_insertf128(full_signs_h, full_signs_h);

            __m256i signs;
            signs = lasx_shuffle_b(full_signs_1, block_sign_shuffle_1);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_1 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_1);

            signs = lasx_shuffle_b(full_signs_1, block_sign_shuffle_2);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_2 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_2);

            signs = lasx_shuffle_b(full_signs_2, block_sign_shuffle_1);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_3 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_3);

            signs = lasx_shuffle_b(full_signs_2, block_sign_shuffle_2);
            signs = __lasx_xvseq_b(__lasx_xvand_v(signs, bit_selector_mask), bit_selector_mask);
            const __m256i q8s_4 = __lasx_xvsigncov_b(__lasx_xvor_v(signs, mone), q8_4);

            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const __m256i dot3  = lasx_maddubs_h(q2_3, q8s_3);
            const __m256i dot4  = lasx_maddubs_h(q2_4, q8s_4);

            const __m256i sc1 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+0)));
            const __m256i sc2 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+1)));
            const __m256i sc3 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+2)));
            const __m256i sc4 = lasx_ext8_16(lsx_shuffle_b(scales, get_scale_shuffle(ib32+3)));

            sumi1 = __lasx_xvadd_w(sumi1, lasx_madd_h(dot1, sc1));
            sumi2 = __lasx_xvadd_w(sumi2, lasx_madd_h(dot2, sc2));
            sumi1 = __lasx_xvadd_w(sumi1, lasx_madd_h(dot3, sc3));
            sumi2 = __lasx_xvadd_w(sumi2, lasx_madd_h(dot4, sc4));
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);
#elif defined(__POWER9_VECTOR__)
    const vector int v0 = vec_splats((int32_t)0);
    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint16_t * restrict q2 = x[i].qs;
        const uint8_t  * restrict sc = x[i].scales;
        const int8_t  *  restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/64; ++j) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq2xs_grid + (q2[0] & 511)), *(const int64_t *)(iq2xs_grid + (q2[1] & 511))};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq2xs_grid + (q2[2] & 511)), *(const int64_t *)(iq2xs_grid + (q2[3] & 511))};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq2xs_grid + (q2[4] & 511)), *(const int64_t *)(iq2xs_grid + (q2[5] & 511))};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq2xs_grid + (q2[6] & 511)), *(const int64_t *)(iq2xs_grid + (q2[7] & 511))};

            vector signed long long vsigns0 = {*(const int64_t *)(signs64 + ((q2[0] >> 9))), *(const int64_t *)(signs64 + ((q2[1] >> 9)))};
            vector signed long long vsigns1 = {*(const int64_t *)(signs64 + ((q2[2] >> 9))), *(const int64_t *)(signs64 + ((q2[3] >> 9)))};
            vector signed long long vsigns2 = {*(const int64_t *)(signs64 + ((q2[4] >> 9))), *(const int64_t *)(signs64 + ((q2[5] >> 9)))};
            vector signed long long vsigns3 = {*(const int64_t *)(signs64 + ((q2[6] >> 9))), *(const int64_t *)(signs64 + ((q2[7] >> 9)))};
            q2 += 8;

            vector signed char q2x0 = (vector signed char)vec_mul((vector signed char)vsigns0, (vector signed char)aux64x2_0);
            vector signed char q2x1 = (vector signed char)vec_mul((vector signed char)vsigns1, (vector signed char)aux64x2_1);
            vector signed char q2x2 = (vector signed char)vec_mul((vector signed char)vsigns2, (vector signed char)aux64x2_2);
            vector signed char q2x3 = (vector signed char)vec_mul((vector signed char)vsigns3, (vector signed char)aux64x2_3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            const uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            const uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            const uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            const uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector signed short vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector signed short vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector signed short vscales3 = vec_splats((int16_t)(2*ls3+1));

            vsumi0 = vec_msum(qv0, vscales0, vsumi0);
            vsumi1 = vec_msum(qv1, vscales1, vsumi1);
            vsumi2 = vec_msum(qv2, vscales2, vsumi2);
            vsumi3 = vec_msum(qv3, vscales3, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);
#else

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * restrict q2 = x[i].qs;
        const uint8_t  * restrict sc = x[i].scales;
        const int8_t   * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const uint16_t ls1 = 2*(sc[ib32] & 0xf) + 1;
            const uint16_t ls2 = 2*(sc[ib32] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xs_grid + (q2[l] & 511));
                const uint8_t  signs = ksigns_iq2xs[q2[l] >> 9];
                for (int j = 0; j < 8; ++j) {
                    sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += sumi * ls2;
            q2 += 4;
        }
        sumf += d * bsum;
    }
    *s = 0.125f * sumf;
#endif
}

void ggml_vec_dot_iq2_s_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq2_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);
    const uint8x16_t m1 = vdupq_n_u8(1);
    const int32x4_t vzero = vdupq_n_s32(0);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q2s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;

        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * restrict q8 = y[i].qs;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            q2s.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[0] | ((qh[ib32+0] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[1] | ((qh[ib32+0] << 6) & 0x300)))));
            q2s.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[2] | ((qh[ib32+0] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[3] | ((qh[ib32+0] << 2) & 0x300)))));
            q2s.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[4] | ((qh[ib32+1] << 8) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[5] | ((qh[ib32+1] << 6) & 0x300)))));
            q2s.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq2s_grid + (qs[6] | ((qh[ib32+1] << 4) & 0x300)))),
                                     vld1_s8((const int8_t *)(iq2s_grid + (qs[7] | ((qh[ib32+1] << 2) & 0x300)))));
            qs += 8;

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            q2s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[0]);
            q2s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[1]);

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vceqq_u8(vs.val[0], mask2);
            vs.val[1] = vceqq_u8(vs.val[1], mask2);

            signs += 4;

            q2s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[0], m1)), q2s.val[2]);
            q2s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vorrq_u8(vs.val[1], m1)), q2s.val[3]);

            const int32x4_t p1 = ggml_vdotq_s32(vzero, q2s.val[0], q8b.val[0]);
            const int32x4_t p2 = ggml_vdotq_s32(vzero, q2s.val[1], q8b.val[1]);
            const int32x4_t p3 = ggml_vdotq_s32(vzero, q2s.val[2], q8b.val[2]);
            const int32x4_t p4 = ggml_vdotq_s32(vzero, q2s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * (1 + 2*(x[i].scales[ib32+0] & 0xf));
            sumi2 += vaddvq_s32(p2) * (1 + 2*(x[i].scales[ib32+0] >>  4));
            sumi1 += vaddvq_s32(p3) * (1 + 2*(x[i].scales[ib32+1] & 0xf));
            sumi2 += vaddvq_s32(p4) * (1 + 2*(x[i].scales[ib32+1] >>  4));
        }
        sumf += d*(sumi1 + sumi2);
    }

    *s = 0.125f * sumf;

#elif defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m256i scales16 = _mm256_cvtepi8_epi16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = _mm256_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = _mm256_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_shuffle_epi8(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    uint64_t aux64;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * restrict q8 = y[i].qs;

        memcpy(&aux64, x[i].scales, 8);
        const __m128i scales8 = _mm_add_epi8(_mm_slli_epi16(_mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), m4), 1), m1);
        const __m128i scales16_0 = _mm_cvtepi8_epi16(scales8);
        const __m128i scales16_1 = _mm_cvtepi8_epi16(_mm_srli_si128(scales8, 8));

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi64x(iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                  iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m128i q2_1_1 = _mm_set_epi64x(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                  iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)]);
            const __m128i q2_2_0 = _mm_set_epi64x(iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                  iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            const __m128i q2_2_1 = _mm_set_epi64x(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                  iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)]);
            qs += 8;

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | ((uint32_t) signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | ((uint32_t) signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);

            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 0)));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+0), 1)));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_shuffle_epi8(scales16_0, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 0)));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_shuffle_epi8(scales16_1, _mm256_extractf128_si256(get_scale_shuffle_k4(ib32+1), 1)));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.125f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector unsigned char mask0 = vec_xl( 0, k_mask1);
    const vector unsigned char mask1 = vec_xl(16, k_mask1);
    const vector signed char mask2 = (vector signed char)vec_xl( 0, k_mask2);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t *  restrict q2 = x[i].qs;
        const uint8_t *  restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const uint8_t *  restrict sc = x[i].scales;
        const int8_t  *  restrict q8 = y[i].qs;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q2, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq2s_grid + (q2[0] | ((qh[0] << 8) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[1] | ((qh[0] << 6) & 0x300)))};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq2s_grid + (q2[2] | ((qh[0] << 4) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[3] | ((qh[0] << 2) & 0x300)))};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq2s_grid + (q2[4] | ((qh[1] << 8) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[5] | ((qh[1] << 6) & 0x300)))};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq2s_grid + (q2[6] | ((qh[1] << 4) & 0x300))), *(const int64_t *)(iq2s_grid + (q2[7] | ((qh[1] << 2) & 0x300)))};
            q2 += 8;
            qh += 2;

            vector signed char vsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&signs[0]);
            vector signed char vsigns23 = (vector signed char)vec_splats(*(const uint32_t *)&signs[2]);
            signs += 4;

            vector signed char vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed char vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed char vsigns2 = vec_perm(vsigns23, vsigns23, mask0);
            vector signed char vsigns3 = vec_perm(vsigns23, vsigns23, mask1);

            vsigns0 = (vector signed char)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed char)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed char)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed char)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed char q2x0 = vec_sub(vec_xor(vsigns0, (vector signed char)aux64x2_0), vsigns0);
            vector signed char q2x1 = vec_sub(vec_xor(vsigns1, (vector signed char)aux64x2_1), vsigns1);
            vector signed char q2x2 = vec_sub(vec_xor(vsigns2, (vector signed char)aux64x2_2), vsigns2);
            vector signed char q2x3 = vec_sub(vec_xor(vsigns3, (vector signed char)aux64x2_3), vsigns3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q2x0, q8y0), vec_mulo(q2x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q2x1, q8y1), vec_mulo(q2x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q2x2, q8y2), vec_mulo(q2x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q2x3, q8y3), vec_mulo(q2x3, q8y3));

            const uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            const uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            const uint16_t ls2 = (uint16_t)(sc[1] & 0xf);
            const uint16_t ls3 = (uint16_t)(sc[1] >>  4);
            sc += 2;

            vector signed short vscales0 = vec_splats((int16_t)(2*ls0+1));
            vector signed short vscales1 = vec_splats((int16_t)(2*ls1+1));
            vector signed short vscales2 = vec_splats((int16_t)(2*ls2+1));
            vector signed short vscales3 = vec_splats((int16_t)(2*ls3+1));

            vsumi0 = vec_msum(qv0, vscales0, vsumi0);
            vsumi1 = vec_msum(qv1, vscales1, vsumi1);
            vsumi2 = vec_msum(qv2, vscales2, vsumi2);
            vsumi3 = vec_msum(qv3, vscales3, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.125f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };


    const __m128i m4 = __lsx_vreplgr2vr_b(0xf);
    const __m128i m1 = __lsx_vreplgr2vr_b(1);

    const __m256i mask1 = __lasx_xvld((const __m256i*)k_mask1, 0);
    const __m256i mask2 = __lasx_xvld((const __m256i*)k_mask2, 0);
    uint64_t aux64;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].qs + QK_K/8);
        const int8_t  * restrict q8 = y[i].qs;

        __m128i tmp1;
        memcpy(&aux64, x[i].scales, 8);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64, 0);
        tmp1 = __lsx_vinsgr2vr_d(tmp1, aux64 >> 4, 1);
        const __m128i scales8 = __lsx_vadd_b(__lsx_vslli_h(__lsx_vand_v(tmp1, m4), 1), m1);
        const __m256i scales16 = lasx_ext8_16(scales8); // 0 2 4 6 8 10 12 14 1 3 5 7 9 11 13 15

        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q2_1 = lasx_set_d(iq2s_grid[qs[3] | ((qh[ib32+0] << 2) & 0x300)],
                                                   iq2s_grid[qs[2] | ((qh[ib32+0] << 4) & 0x300)],
                                                   iq2s_grid[qs[1] | ((qh[ib32+0] << 6) & 0x300)],
                                                   iq2s_grid[qs[0] | ((qh[ib32+0] << 8) & 0x300)]);
            const __m256i q2_2 = lasx_set_d(iq2s_grid[qs[7] | ((qh[ib32+1] << 2) & 0x300)],
                                                   iq2s_grid[qs[6] | ((qh[ib32+1] << 4) & 0x300)],
                                                   iq2s_grid[qs[5] | ((qh[ib32+1] << 6) & 0x300)],
                                                   iq2s_grid[qs[4] | ((qh[ib32+1] << 8) & 0x300)]);
            qs += 8;

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | ((uint32_t) signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | ((uint32_t) signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1); // blocks 2*ib32+0, 2*ib32+1
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2); // blocks 2*ib32+2, 2*ib32+3

            const __m256i p1 = lasx_madd_h(dot1, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+0)));
            const __m256i p2 = lasx_madd_h(dot2, lasx_shuffle_b(scales16, get_scale_shuffle_k4(ib32+1)));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.125f * hsum_float_8(accumf);

#else

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const int8_t  * q8 = y[i].qs;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = qs + QK_K/8;

        int bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            int ls1 = 1 + 2*(x[i].scales[ib32] & 0xf);
            int ls2 = 1 + 2*(x[i].scales[ib32] >>  4);
            int sumi1 = 0, sumi2 = 0;
            for (int l = 0; l < 2; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi1 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            for (int l = 2; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2s_grid + (qs[l] | (qh[ib32] << (8-2*l) & 0x300)));
                for (int j = 0; j < 8; ++j) {
                    sumi2 += q8[j] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1 : 1);
                }
                q8 += 8;
            }
            bsum += ls1 * sumi1 + ls2 * sumi2;
            qs += 4;
            signs += 4;
        }

        sumf += d * bsum;
    }

    *s = 0.125f * sumf;

#endif

}

void ggml_vec_dot_iq3_xxs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_xxs * restrict x = vx;
    const block_q8_K    * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t   * restrict q8 = y[i].qs;
        float sumf1 = 0, sumf2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;
            memcpy(aux32, gas, 2*sizeof(uint32_t)); gas += 2*sizeof(uint32_t);
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]);
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]);
            q3 += 16;
            q3s.val[0] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >>  7) & 127))));
            q3s.val[1] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[0] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[0] >> 21) & 127))));
            q3s.val[2] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >>  0) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >>  7) & 127))));
            q3s.val[3] = vcombine_s8(vld1_s8((const void *)(signs64 + ((aux32[1] >> 14) & 127))), vld1_s8((const void *)(signs64 + ((aux32[1] >> 21) & 127))));
            q3s.val[0] = vmulq_s8(q3s.val[0], vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(q3s.val[1], vreinterpretq_s8_u32(aux32x4_1));
            q3s.val[2] = vmulq_s8(q3s.val[2], vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(q3s.val[3], vreinterpretq_s8_u32(aux32x4_3));
            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);
            sumf1 += vaddvq_s32(p1) * (0.5f + (aux32[0] >> 28));
            sumf2 += vaddvq_s32(p2) * (0.5f + (aux32[1] >> 28));
        }
        sumf += d*(sumf1 + sumf2);
    }
    *s = 0.5f * sumf;

#elif defined(__AVX2__)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t  * restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q2_1 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = _mm256_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                  iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m256i s2_1 = _mm256_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = _mm256_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = _mm256_sign_epi8(q8_1, s2_1);
            const __m256i q8s_2 = _mm256_sign_epi8(q8_2, s2_2);
            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#elif defined(__AVX__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t  * restrict q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q2_1_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            const __m128i q2_2_0 = _mm_set_epi32(iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;
            const __m128i s2_1_0 = _mm_set_epi64x(signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m128i s2_1_1 = _mm_set_epi64x(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127]);
            const __m128i s2_2_0 = _mm_set_epi64x(signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m128i s2_2_1 = _mm_set_epi64x(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127]);
            const __m128i q8s_1_0 = _mm_sign_epi8(q8_1_0, s2_1_0);
            const __m128i q8s_1_1 = _mm_sign_epi8(q8_1_1, s2_1_1);
            const __m128i q8s_2_0 = _mm_sign_epi8(q8_2_0, s2_2_0);
            const __m128i q8s_2_1 = _mm_sign_epi8(q8_2_1, s2_2_1);
            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = 0.25f * hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        const uint8_t * restrict q3 = x[i].qs;
        const uint32_t * restrict signs = (const uint32_t *)(x[i].qs + QK_K/4);
        const int8_t  * restrict q8 = y[i].qs;

#pragma GCC unroll 1
        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned int aux32x4_0 = {iq3xxs_grid[q3[ 0]], iq3xxs_grid[q3[ 1]], iq3xxs_grid[q3[ 2]], iq3xxs_grid[q3[ 3]]};
            vector unsigned int aux32x4_1 = {iq3xxs_grid[q3[ 4]], iq3xxs_grid[q3[ 5]], iq3xxs_grid[q3[ 6]], iq3xxs_grid[q3[ 7]]};
            vector unsigned int aux32x4_2 = {iq3xxs_grid[q3[ 8]], iq3xxs_grid[q3[ 9]], iq3xxs_grid[q3[10]], iq3xxs_grid[q3[11]]};
            vector unsigned int aux32x4_3 = {iq3xxs_grid[q3[12]], iq3xxs_grid[q3[13]], iq3xxs_grid[q3[14]], iq3xxs_grid[q3[15]]};
            q3 += 16;

            vector unsigned long long aux64x2_0 = {(uint64_t)(signs64[(signs[0] >>  0) & 127]), (uint64_t)(signs64[(signs[0] >>  7) & 127])};
            vector unsigned long long aux64x2_1 = {(uint64_t)(signs64[(signs[0] >> 14) & 127]), (uint64_t)(signs64[(signs[0] >> 21) & 127])};
            vector unsigned long long aux64x2_2 = {(uint64_t)(signs64[(signs[1] >>  0) & 127]), (uint64_t)(signs64[(signs[1] >>  7) & 127])};
            vector unsigned long long aux64x2_3 = {(uint64_t)(signs64[(signs[1] >> 14) & 127]), (uint64_t)(signs64[(signs[1] >> 21) & 127])};

            vector signed char q3x0 = vec_mul((vector signed char)aux64x2_0, (vector signed char)aux32x4_0);
            vector signed char q3x1 = vec_mul((vector signed char)aux64x2_1, (vector signed char)aux32x4_1);
            vector signed char q3x2 = vec_mul((vector signed char)aux64x2_2, (vector signed char)aux32x4_2);
            vector signed char q3x3 = vec_mul((vector signed char)aux64x2_3, (vector signed char)aux32x4_3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            const uint16_t ls0 = (uint16_t)(signs[0] >> 28);
            const uint16_t ls1 = (uint16_t)(signs[1] >> 28);
            signs += 2;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = 0.25f * vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const uint64_t * signs64 = (const uint64_t *)keven_signs_q2xs;

    uint32_t aux32[2];

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t  * restrict q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q2_1 = lasx_set_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            const __m256i q2_2 = lasx_set_w(iq3xxs_grid[q3[7]], iq3xxs_grid[q3[6]], iq3xxs_grid[q3[5]], iq3xxs_grid[q3[4]],
                                                iq3xxs_grid[q3[3]], iq3xxs_grid[q3[2]], iq3xxs_grid[q3[1]], iq3xxs_grid[q3[0]]);
            q3 += 8;
            memcpy(aux32, gas, 8); gas += 8;

            const __m256i s2_1 = lasx_set_d(signs64[(aux32[0] >> 21) & 127], signs64[(aux32[0] >> 14) & 127],
                                                   signs64[(aux32[0] >>  7) & 127], signs64[(aux32[0] >>  0) & 127]);
            const __m256i s2_2 = lasx_set_d(signs64[(aux32[1] >> 21) & 127], signs64[(aux32[1] >> 14) & 127],
                                                   signs64[(aux32[1] >>  7) & 127], signs64[(aux32[1] >>  0) & 127]);
            const __m256i q8s_1 = __lasx_xvsigncov_b(s2_1, q8_1);
            const __m256i q8s_2 = __lasx_xvsigncov_b(s2_2, q8_2);
            const __m256i dot1  = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = aux32[0] >> 28;
            const uint16_t ls2 = aux32[1] >> 28;

            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = 0.25f * hsum_float_8(accumf);

#else

    uint32_t aux32;

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict q3 = x[i].qs;
        const uint8_t * restrict gas = x[i].qs + QK_K/4;
        const int8_t  * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(&aux32, gas, sizeof(uint32_t)); gas += sizeof(uint32_t);
            const uint32_t ls = 2*(aux32 >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*l+0]);
                const uint8_t * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*l+1]);
                const uint8_t  signs = ksigns_iq2xs[(aux32 >> 7*l) & 127];
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            q3 += 8;
            bsum += sumi * ls;
        }
        sumf += d * bsum;
    }
    *s = 0.25f * sumf;
#endif
}

void ggml_vec_dot_iq3_s_q8_K (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq3_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

#if defined(__ARM_NEON)

    typedef union {
        uint16x8_t vec_index;
        uint16_t   index[8];
    } vec_index_t;

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};

    const ggml_uint8x16x2_t mask1 = ggml_vld1q_u8_x2(k_mask1);
    const uint8x16_t        mask2 = vld1q_u8(k_mask2);

    const int16x8_t  hshift = vld1q_s16(k_shift);
    const uint16x8_t m256   = vdupq_n_u16(256);
    const uint8x16_t m1     = vdupq_n_u8(1);

    uint8x16x2_t vs;
    ggml_int8x16x4_t q3s;
    ggml_int8x16x4_t q8b;
    vec_index_t idx;

    uint32_t scales32[2];
    const uint8_t * scales8 = (const uint8_t *)scales32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)x[i].signs;
        const int8_t   * restrict q8 = y[i].qs;

        memcpy(scales32, x[i].scales, 4);
        scales32[1] = (((scales32[0] >> 4) & 0x0f0f0f0f) << 1) | 0x01010101;
        scales32[0] = ((scales32[0] & 0x0f0f0f0f) << 1) | 0x01010101;

        int sumi1 = 0, sumi2 = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const uint8x16_t idx_l = vld1q_u8(qs); qs += 16;
            idx.vec_index = vorrq_u16(vmovl_u8(vget_low_u8 (idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+0]), hshift), m256));
            const uint32x4_t aux32x4_0 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_1 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);
            idx.vec_index = vorrq_u16(vmovl_u8(vget_high_u8(idx_l)), vandq_u16(vshlq_u16(vdupq_n_u16(qh[ib32+1]), hshift), m256));
            const uint32x4_t aux32x4_2 = ggml_vld1q_u32(iq3s_grid[idx.index[0]], iq3s_grid[idx.index[1]],
                                                        iq3s_grid[idx.index[2]], iq3s_grid[idx.index[3]]);
            const uint32x4_t aux32x4_3 = ggml_vld1q_u32(iq3s_grid[idx.index[4]], iq3s_grid[idx.index[5]],
                                                        iq3s_grid[idx.index[6]], iq3s_grid[idx.index[7]]);


            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[0] | ((uint32_t) signs[1] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            q3s.val[0] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_0));
            q3s.val[1] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_1));

            vs.val[0] = vreinterpretq_u8_u32(vdupq_n_u32(signs[2] | ((uint32_t) signs[3] << 16)));
            vs.val[1] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[1]), mask2);
            vs.val[0] = vandq_u8(ggml_vqtbl1q_u8(vs.val[0], mask1.val[0]), mask2);
            vs.val[0] = vorrq_u8(vceqq_u8(vs.val[0], mask2), m1);
            vs.val[1] = vorrq_u8(vceqq_u8(vs.val[1], mask2), m1);

            signs += 4;

            q3s.val[2] = vmulq_s8(vreinterpretq_s8_u8(vs.val[0]), vreinterpretq_s8_u32(aux32x4_2));
            q3s.val[3] = vmulq_s8(vreinterpretq_s8_u8(vs.val[1]), vreinterpretq_s8_u32(aux32x4_3));

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[0], q8b.val[0]), q3s.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q3s.val[2], q8b.val[2]), q3s.val[3], q8b.val[3]);

            sumi1 += vaddvq_s32(p1) * scales8[ib32/2+0];
            sumi2 += vaddvq_s32(p2) * scales8[ib32/2+4];
        }
        sumf += d*(sumi1 + sumi2);
    }
    *s = sumf;

#elif defined(__AVX2__)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = _mm256_loadu_si256((const __m256i*)k_mask1);
    const __m256i mask2 = _mm256_loadu_si256((const __m256i*)k_mask2);

    const __m256i idx_shift = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = _mm256_set1_epi32(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)x[i].signs;
        const int8_t  * restrict q8 = y[i].qs;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i idx_l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)qs)); qs += 16;
            idx.vec[0] = _mm256_set1_epi32(qh[ib32+0]);
            idx.vec[1] = _mm256_set1_epi32(qh[ib32+1]);
            idx.vec[0] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = _mm256_and_si256(_mm256_sllv_epi32(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = _mm256_or_si256(idx.vec[0], _mm256_cvtepi16_epi32(_mm256_castsi256_si128(idx_l)));
            idx.vec[1] = _mm256_or_si256(idx.vec[1], _mm256_cvtepi16_epi32(_mm256_extractf128_si256(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = _mm256_set_epi32(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = _mm256_set_epi32(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = _mm256_set1_epi32(signs[0] | (signs[1] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_1 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_1 = _mm256_sub_epi8(_mm256_xor_si256(s2_1, q8_1), s2_1);

            aux256 = _mm256_set1_epi32(signs[2] | (signs[3] << 16));
            aux256 = _mm256_and_si256(_mm256_shuffle_epi8(aux256,mask1), mask2);
            const __m256i s2_2 = _mm256_cmpeq_epi8(aux256, mask2);
            const __m256i q8s_2 = _mm256_sub_epi8(_mm256_xor_si256(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1  = _mm256_maddubs_epi16(q2_1, q8s_1);
            const __m256i dot2  = _mm256_maddubs_epi16(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(2*ls1+1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(2*ls2+1));
            sumi1 = _mm256_add_epi32(sumi1, p1);
            sumi2 = _mm256_add_epi32(sumi2, p2);
        }

        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);

    }

    *s = hsum_float_8(accumf);

#elif defined(__AVX__)
   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m128i mask1_0 = _mm_loadu_si128((const __m128i*)k_mask1);
    const __m128i mask1_1 = _mm_loadu_si128((const __m128i*)k_mask1 + 1);
    const __m128i mask2_0 = _mm_loadu_si128((const __m128i*)k_mask2);
    const __m128i mask2_1 = _mm_loadu_si128((const __m128i*)k_mask2 + 1);

    const __m128i idx_mul_0 = _mm_set_epi32(32, 64, 128, 256);
    const __m128i idx_mul_1 = _mm_set_epi32(2, 4, 8, 16);
    const __m128i idx_mask  = _mm_set1_epi32(256);

    typedef union {
        __m128i  vec[4];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)x[i].signs;
        const int8_t  * restrict q8 = y[i].qs;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m128i q8_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i qs_tmp = _mm_loadu_si128((const __m128i *)qs);
            const __m128i idx_l_0 = _mm_cvtepu8_epi16(qs_tmp);
            const __m128i idx_l_1 = _mm_cvtepu8_epi16(_mm_srli_si128(qs_tmp, 8)); qs += 16;
            idx.vec[0] = _mm_set1_epi32(qh[ib32+0]);
            idx.vec[1] = idx.vec[0];
            idx.vec[2] = _mm_set1_epi32(qh[ib32+1]);
            idx.vec[3] = idx.vec[2];

            idx.vec[0] = _mm_and_si128(_mm_mullo_epi32(idx.vec[0], idx_mul_0), idx_mask);
            idx.vec[1] = _mm_and_si128(_mm_mullo_epi32(idx.vec[1], idx_mul_1), idx_mask);
            idx.vec[2] = _mm_and_si128(_mm_mullo_epi32(idx.vec[2], idx_mul_0), idx_mask);
            idx.vec[3] = _mm_and_si128(_mm_mullo_epi32(idx.vec[3], idx_mul_1), idx_mask);

            idx.vec[0] = _mm_or_si128(idx.vec[0], _mm_cvtepi16_epi32(idx_l_0));
            idx.vec[1] = _mm_or_si128(idx.vec[1], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_0, 8)));
            idx.vec[2] = _mm_or_si128(idx.vec[2], _mm_cvtepi16_epi32(idx_l_1));
            idx.vec[3] = _mm_or_si128(idx.vec[3], _mm_cvtepi16_epi32(_mm_srli_si128(idx_l_1, 8)));

            const __m128i q2_1_0 = _mm_set_epi32(iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]);
            const __m128i q2_1_1 = _mm_set_epi32(iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]]);
            const __m128i q2_2_0 = _mm_set_epi32(iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[9]], iq3s_grid[idx.index[8]]);
            const __m128i q2_2_1 = _mm_set_epi32(iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]]);

            __m128i aux128_0 = _mm_set1_epi32(signs[0] | (signs[1] << 16));
            __m128i aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_1_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_1_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_1_0 = _mm_sub_epi8(_mm_xor_si128(s2_1_0, q8_1_0), s2_1_0);
            const __m128i q8s_1_1 = _mm_sub_epi8(_mm_xor_si128(s2_1_1, q8_1_1), s2_1_1);

            aux128_0 = _mm_set1_epi32(signs[2] | (signs[3] << 16));
            aux128_1 = aux128_0;
            aux128_0 = _mm_and_si128(_mm_shuffle_epi8(aux128_0,mask1_0), mask2_0);
            aux128_1 = _mm_and_si128(_mm_shuffle_epi8(aux128_1,mask1_1), mask2_1);
            const __m128i s2_2_0 = _mm_cmpeq_epi8(aux128_0, mask2_0);
            const __m128i s2_2_1 = _mm_cmpeq_epi8(aux128_1, mask2_1);
            const __m128i q8s_2_0 = _mm_sub_epi8(_mm_xor_si128(s2_2_0, q8_2_0), s2_2_0);
            const __m128i q8s_2_1 = _mm_sub_epi8(_mm_xor_si128(s2_2_1, q8_2_1), s2_2_1);

            signs += 4;

            const __m128i dot1_0  = _mm_maddubs_epi16(q2_1_0, q8s_1_0);
            const __m128i dot1_1  = _mm_maddubs_epi16(q2_1_1, q8s_1_1);
            const __m128i dot2_0  = _mm_maddubs_epi16(q2_2_0, q8s_2_0);
            const __m128i dot2_1  = _mm_maddubs_epi16(q2_2_1, q8s_2_1);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(2*ls1+1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(2*ls1+1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(2*ls2+1));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(2*ls2+1));
            sumi1_0 = _mm_add_epi32(sumi1_0, p1_0);
            sumi1_1 = _mm_add_epi32(sumi1_1, p1_1);
            sumi2_0 = _mm_add_epi32(sumi2_0, p2_0);
            sumi2_1 = _mm_add_epi32(sumi2_1, p2_1);
        }

        accumf = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_add_epi32(sumi1_1, sumi2_1), _mm_add_epi32(sumi1_0, sumi2_0)))), accumf);

    }

    *s = hsum_float_8(accumf);

#elif defined(__POWER9_VECTOR__)
    static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                        0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
    };

    static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,};

    const vector int v0 = vec_splats((int32_t)0);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector unsigned char mask0 = vec_xl( 0, k_mask1);
    const vector unsigned char mask1 = vec_xl(16, k_mask1);
    const vector signed char mask2 = (vector signed char)vec_xl( 0, k_mask2);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        const uint8_t *  restrict q3 = x[i].qs;
        const uint8_t *  restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)(x[i].signs);
        const uint8_t *  restrict sc = x[i].scales;
        const int8_t  *  restrict q8 = y[i].qs;

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q3, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector unsigned int aux32x4_0 = {iq3s_grid[q3[ 0] | ((qh[0] << 8) & 256)], iq3s_grid[q3[ 1] | ((qh[0] << 7) & 256)],
                                             iq3s_grid[q3[ 2] | ((qh[0] << 6) & 256)], iq3s_grid[q3[ 3] | ((qh[0] << 5) & 256)]};
            vector unsigned int aux32x4_1 = {iq3s_grid[q3[ 4] | ((qh[0] << 4) & 256)], iq3s_grid[q3[ 5] | ((qh[0] << 3) & 256)],
                                             iq3s_grid[q3[ 6] | ((qh[0] << 2) & 256)], iq3s_grid[q3[ 7] | ((qh[0] << 1) & 256)]};
            vector unsigned int aux32x4_2 = {iq3s_grid[q3[ 8] | ((qh[1] << 8) & 256)], iq3s_grid[q3[ 9] | ((qh[1] << 7) & 256)],
                                             iq3s_grid[q3[10] | ((qh[1] << 6) & 256)], iq3s_grid[q3[11] | ((qh[1] << 5) & 256)]};
            vector unsigned int aux32x4_3 = {iq3s_grid[q3[12] | ((qh[1] << 4) & 256)], iq3s_grid[q3[13] | ((qh[1] << 3) & 256)],
                                             iq3s_grid[q3[14] | ((qh[1] << 2) & 256)], iq3s_grid[q3[15] | ((qh[1] << 1) & 256)]};
            q3 += 16;
            qh += 2;

            vector signed char vsigns01 = (vector signed char)vec_splats(*(const uint32_t *)&signs[0]);
            vector signed char vsigns02 = (vector signed char)vec_splats(*(const uint32_t *)&signs[2]);
            signs += 4;

            vector signed char vsigns0 = vec_perm(vsigns01, vsigns01, mask0);
            vector signed char vsigns1 = vec_perm(vsigns01, vsigns01, mask1);
            vector signed char vsigns2 = vec_perm(vsigns02, vsigns02, mask0);
            vector signed char vsigns3 = vec_perm(vsigns02, vsigns02, mask1);

            vsigns0 = (vector signed char)vec_cmpeq(vec_and(vsigns0, mask2), mask2);
            vsigns1 = (vector signed char)vec_cmpeq(vec_and(vsigns1, mask2), mask2);
            vsigns2 = (vector signed char)vec_cmpeq(vec_and(vsigns2, mask2), mask2);
            vsigns3 = (vector signed char)vec_cmpeq(vec_and(vsigns3, mask2), mask2);

            vector signed char q3x0 = vec_sub(vec_xor(vsigns0, (vector signed char)aux32x4_0), vsigns0);
            vector signed char q3x1 = vec_sub(vec_xor(vsigns1, (vector signed char)aux32x4_1), vsigns1);
            vector signed char q3x2 = vec_sub(vec_xor(vsigns2, (vector signed char)aux32x4_2), vsigns2);
            vector signed char q3x3 = vec_sub(vec_xor(vsigns3, (vector signed char)aux32x4_3), vsigns3);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q3x0, q8y0), vec_mulo(q3x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q3x1, q8y1), vec_mulo(q3x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q3x2, q8y2), vec_mulo(q3x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q3x3, q8y3), vec_mulo(q3x3, q8y3));

            const uint16_t ls0 = (uint16_t)(sc[0] & 0xf);
            const uint16_t ls1 = (uint16_t)(sc[0] >>  4);
            sc ++;

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

   static const uint8_t k_mask1[32] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
                                       0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03, 0x03
   };

    static const uint8_t k_mask2[32] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };

    const __m256i mask1 = __lasx_xvld((const __m256i*)k_mask1, 0);
    const __m256i mask2 = __lasx_xvld((const __m256i*)k_mask2, 0);

    __m256i idx_shift = lasx_set_w(1, 2, 3, 4, 5, 6, 7, 8);
    const __m256i idx_mask  = __lasx_xvreplgr2vr_w(256);

    typedef union {
        __m256i  vec[2];
        uint32_t index[16];
    } index_t;

    index_t idx;

    __m256 accumf = (__m256)__lasx_xvldi(0);
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint16_t * restrict signs = (const uint16_t *)x[i].signs;
        const int8_t  * restrict q8 = y[i].qs;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const __m256i q8_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i idx_l = lasx_extu8_16(__lsx_vld(qs, 0)); qs += 16;
            idx.vec[0] = __lasx_xvreplgr2vr_w(qh[ib32+0]);
            idx.vec[1] = __lasx_xvreplgr2vr_w(qh[ib32+1]);
            idx.vec[0] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[0], idx_shift), idx_mask);
            idx.vec[1] = __lasx_xvand_v(__lasx_xvsll_w(idx.vec[1], idx_shift), idx_mask);
            idx.vec[0] = __lasx_xvor_v(idx.vec[0], lasx_ext16_32(lasx_extracti128(idx_l, 0)));
            idx.vec[1] = __lasx_xvor_v(idx.vec[1], lasx_ext16_32(lasx_extracti128(idx_l, 1)));

            // At leat on my CPU (Ryzen 7950X), using _mm256_i32gather_epi32 is slower than _mm256_set_epi32. Strange.
            //const __m256i q2_1 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[0], 4);
            //const __m256i q2_2 = _mm256_i32gather_epi32((const int *)iq3s_grid, idx.vec[1], 4);
            const __m256i q2_1 = lasx_set_w(
                    iq3s_grid[idx.index[7]], iq3s_grid[idx.index[6]], iq3s_grid[idx.index[5]], iq3s_grid[idx.index[4]],
                    iq3s_grid[idx.index[3]], iq3s_grid[idx.index[2]], iq3s_grid[idx.index[1]], iq3s_grid[idx.index[0]]
            );
            const __m256i q2_2 = lasx_set_w(
                    iq3s_grid[idx.index[15]], iq3s_grid[idx.index[14]], iq3s_grid[idx.index[13]], iq3s_grid[idx.index[12]],
                    iq3s_grid[idx.index[11]], iq3s_grid[idx.index[10]], iq3s_grid[idx.index[ 9]], iq3s_grid[idx.index[ 8]]
            );

            __m256i aux256 = __lasx_xvreplgr2vr_w(signs[0] | (signs[1] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_1 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_1 = __lasx_xvsub_b(__lasx_xvxor_v(s2_1, q8_1), s2_1);

            aux256 = __lasx_xvreplgr2vr_w(signs[2] | (signs[3] << 16));
            aux256 = __lasx_xvand_v(lasx_shuffle_b(aux256,mask1), mask2);
            const __m256i s2_2 = __lasx_xvseq_b(aux256, mask2);
            const __m256i q8s_2 = __lasx_xvsub_b(__lasx_xvxor_v(s2_2, q8_2), s2_2);

            signs += 4;

            const __m256i dot1 = lasx_maddubs_h(q2_1, q8s_1);
            const __m256i dot2  = lasx_maddubs_h(q2_2, q8s_2);
            const uint16_t ls1 = x[i].scales[ib32/2] & 0xf;
            const uint16_t ls2 = x[i].scales[ib32/2] >>  4;
            const __m256i p1 = lasx_madd_h(dot1, __lasx_xvreplgr2vr_h(2*ls1+1));
            const __m256i p2 = lasx_madd_h(dot2, __lasx_xvreplgr2vr_h(2*ls2+1));
            sumi1 = __lasx_xvadd_w(sumi1, p1);
            sumi2 = __lasx_xvadd_w(sumi2, p2);
        }

        accumf = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accumf);
    }

    *s = hsum_float_8(accumf);

#else

    float sumf = 0.f;
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * restrict qs = x[i].qs;
        const uint8_t * restrict qh = x[i].qh;
        const uint8_t * restrict signs = x[i].signs;
        const int8_t  * restrict q8 = y[i].qs;
        int32_t bsum = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const uint32_t ls1 = 2*(x[i].scales[ib32/2] & 0xf) + 1;
            const uint32_t ls2 = 2*(x[i].scales[ib32/2] >>  4) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls1;
            sumi = 0;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[ib32+1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[ib32+1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    sumi += grid1[j] * q8[j+0] * (signs[l] & kmask_iq2xs[j+0] ? -1 : 1);
                    sumi += grid2[j] * q8[j+4] * (signs[l] & kmask_iq2xs[j+4] ? -1 : 1);
                }
                q8 += 8;
            }
            qs += 8;
            signs += 4;
            bsum += sumi * ls2;
        }
        sumf += d * bsum;
    }
    *s = sumf;
#endif
}

#if defined(__AVX2__)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}
#elif defined(__loongarch_asx)
static inline __m256i mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = __lasx_xvsigncov_b(x, x);
    const __m256i sy = __lasx_xvsigncov_b(x, y);
    __m256i tmp1, tmp2, tmp3;
    tmp1 = __lasx_xvmulwev_h_bu_b(ax, sy);
    tmp2 = __lasx_xvmulwod_h_bu_b(ax, sy);
    tmp3 = __lasx_xvadd_h(tmp1, tmp2);
    return __lasx_xvsat_h(tmp3, 15);
}
#endif

void ggml_vec_dot_iq1_s_q8_K  (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_s * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

#if defined __ARM_NEON

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi1 = 0, sumi2 = 0, sumi3 = 0;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[ib+0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[ib+0] << 5) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[ib+0] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[ib+0] >> 1) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[ib+1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[ib+1] << 5) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[ib+1] << 2) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[ib+1] >> 1) & 0x700)))));
            qs += 8;

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[0], q8b.val[0]), q1b.val[1], q8b.val[1]);
            const int32x4_t p2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q1b.val[2], q8b.val[2]), q1b.val[3], q8b.val[3]);

            const int ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            sumi1 += vaddvq_s32(p1) * ls1;
            sumi2 += vaddvq_s32(p2) * ls2;
            sumi3 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * ls1 * (qh[ib+0] & 0x8000 ? -1 : 1)
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * ls2 * (qh[ib+1] & 0x8000 ? -1 : 1);

        }

        sumf += y[i].d * GGML_FP16_TO_FP32(x[i].d) * (sumi1 + sumi2 + IQ1S_DELTA * sumi3);
    }

    *s = sumf;

#elif defined __AVX2__

    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = _mm256_setzero_si256();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)],
                                                    iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m256i q1b_2 = _mm256_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)],
                                                    iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            qs += 8;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m256i p1 = _mm256_madd_epi16(dot1, _mm256_set1_epi16(ls1));
            const __m256i p2 = _mm256_madd_epi16(dot2, _mm256_set1_epi16(ls2));

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#elif defined __AVX__
    __m256 accum = _mm256_setzero_ps();
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)]);
            qs += 8;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, _mm_set1_epi16(ls1));
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, _mm_set1_epi16(ls1));
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, _mm_set1_epi16(ls2));
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, _mm_set1_epi16(ls2));

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum);
        accum1 += d * sumi1;

    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#elif defined(__POWER9_VECTOR__)
    const vector unsigned char v0 = vec_splats((unsigned char)0x0);
    const vector unsigned short vsign = vec_splats((unsigned short)0x8000);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    for (int i = 0; i < nb; ++i) {
        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[i].d));
        vector float vyd = vec_splats(y[i].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = vec_splats((int32_t)0);
        vector signed int vsumi1 = vec_splats((int32_t)0);
        vector signed int vsumi2 = vec_splats((int32_t)0);
        vector signed int vsumi3 = vec_splats((int32_t)0);
        vector signed int vsumi8 = vec_splats((int32_t)0);

        const uint8_t  * restrict q1 = x[i].qs;
        const uint16_t * restrict qh = x[i].qh;
        const int8_t   * restrict q8 = y[i].qs;
        const int16_t  * restrict qs = y[i].bsums;

        for (int j = 0; j < QK_K/32; j += 2) {
            __builtin_prefetch(q1, 0, 1);
            __builtin_prefetch(qh, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed long long aux64x2_0 = {*(const int64_t *)(iq1s_grid + (q1[0] | ((qh[0] << 8) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[1] | ((qh[0] << 5) & 0x700)))};
            vector signed long long aux64x2_1 = {*(const int64_t *)(iq1s_grid + (q1[2] | ((qh[0] << 2) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[3] | ((qh[0] >> 1) & 0x700)))};
            vector signed long long aux64x2_2 = {*(const int64_t *)(iq1s_grid + (q1[4] | ((qh[1] << 8) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[5] | ((qh[1] << 5) & 0x700)))};
            vector signed long long aux64x2_3 = {*(const int64_t *)(iq1s_grid + (q1[6] | ((qh[1] << 2) & 0x700))), *(const int64_t *)(iq1s_grid + (q1[7] | ((qh[1] >> 1) & 0x700)))};
            q1 += 8;

            vector signed char q1x0 = (vector signed char)aux64x2_0;
            vector signed char q1x1 = (vector signed char)aux64x2_1;
            vector signed char q1x2 = (vector signed char)aux64x2_2;
            vector signed char q1x3 = (vector signed char)aux64x2_3;

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q1x0, q8y0), vec_mulo(q1x0, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q1x1, q8y1), vec_mulo(q1x1, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q1x2, q8y2), vec_mulo(q1x2, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q1x3, q8y3), vec_mulo(q1x3, q8y3));

            const uint16_t ls0 = (uint16_t)((qh[0] >> 12) & 7);
            const uint16_t ls1 = (uint16_t)((qh[1] >> 12) & 7);

            vector signed short vscales01 = (vector signed short)vec_splats((uint16_t)(2*ls0+1));
            vector signed short vscales23 = (vector signed short)vec_splats((uint16_t)(2*ls1+1));
            vector signed short vscales = vec_sld(vscales23, vscales01, 8);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);

            vector signed short q8ysums = vec_xl_len(qs, 8);
            qs += 4;
            q8ysums = vec_mergeh(q8ysums, (vector signed short)v0);

            vector signed short qxh = (vector signed short)vec_sld(vec_splats(qh[1]), vec_splats(qh[0]), 8);
            qh += 2;
            vector __bool short vsel = vec_cmpge(qxh, (vector signed short)v0);

            vector signed short q8ysum = vec_sel((vector signed short)vec_xor((vector unsigned short)q8ysums, vsign), q8ysums, vsel);

            vsumi8 = vec_add(vec_mule(q8ysum, vscales), vsumi8);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);

        vsumf0 = vec_madd(vec_ctf(vsumi8, 0), vec_mul(vd, vec_splats(IQ1S_DELTA)), vsumf0);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    __m256 accum = (__m256)__lasx_xvldi(0);
    float accum1 = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        __m256i sumi = __lasx_xvldi(0);
        int sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            __m256i q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[0] | ((qh[ib+0] << 8) & 0x700)], 0);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[1] | ((qh[ib+0] << 5) & 0x700)], 1);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[2] | ((qh[ib+0] << 2) & 0x700)], 2);
            q1b_1 = __lasx_xvinsgr2vr_d(q1b_1, iq1s_grid[qs[3] | ((qh[ib+0] >> 1) & 0x700)], 3);

            __m256i q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[4] | ((qh[ib+1] << 8) & 0x700)], 0);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[5] | ((qh[ib+1] << 5) & 0x700)], 1);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[6] | ((qh[ib+1] << 2) & 0x700)], 2);
            q1b_2 = __lasx_xvinsgr2vr_d(q1b_2, iq1s_grid[qs[7] | ((qh[ib+1] >> 1) & 0x700)], 3);

            qs += 8;
            const __m256i q8b_1 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;
            const __m256i q8b_2 = __lasx_xvld((const __m256i*)q8, 0); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);
            const int16_t ls1 = 2*((qh[ib+0] >> 12) & 7) + 1;
            const int16_t ls2 = 2*((qh[ib+1] >> 12) & 7) + 1;

            __m256i tmp1, tmp5, tmp6;
            tmp1 = __lasx_xvreplgr2vr_h(ls1);
            tmp5 = __lasx_xvmulwev_w_h(dot1, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot1, tmp1);
            const __m256i p1 = __lasx_xvadd_w(tmp5, tmp6);

            tmp1 = __lasx_xvreplgr2vr_h(ls2);
            tmp5 = __lasx_xvmulwev_w_h(dot2, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(dot2, tmp1);
            const __m256i p2 = __lasx_xvadd_w(tmp5, tmp6);

            sumi = __lasx_xvadd_w(sumi, __lasx_xvadd_w(p1, p2));
            sumi1 += (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]) * (qh[ib+0] & 0x8000 ? -1 : 1) * ls1
                   + (y[i].bsums[2*ib+2] + y[i].bsums[2*ib+3]) * (qh[ib+1] & 0x8000 ? -1 : 1) * ls2;
        }

        const float d = y[i].d * GGML_FP16_TO_FP32(x[i].d);
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(d), __lasx_xvffint_s_w(sumi), accum);
        accum1 += d * sumi1;
    }

    *s = hsum_float_8(accum) + IQ1S_DELTA * accum1;

#else

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint16_t * qh = x[i].qh;

        int sumi = 0, sumi1 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = 2*((qh[ib] >> 12) & 7) + 1;
            const int delta = qh[ib] & 0x8000 ? -1 : 1;
            int lsum = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((qh[ib] >> 3*l) & 7) << 8)));
                for (int j = 0; j < 8; ++j) {
                    lsum += q8[j] * grid[j];
                }
                q8 += 8;
            }
            sumi  += ls * lsum;
            sumi1 += ls * delta * (y[i].bsums[2*ib+0] + y[i].bsums[2*ib+1]);
            qs += 4;
        }

        sumf += GGML_FP16_TO_FP32(x[i].d) * y[i].d * (sumi + IQ1S_DELTA * sumi1);
    }

    *s = sumf;

#endif
}

void ggml_vec_dot_iq1_m_q8_K  (int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);

    const block_iq1_m * restrict x = vx;
    const block_q8_K  * restrict y = vy;

    const int nb = n / QK_K;

    iq1m_scale_t scale;

#if defined __ARM_NEON
    const int32x4_t mask  = vdupq_n_s32(0x7);
    const int32x4_t mone  = vdupq_n_s32(1);
    const int32x4_t mzero = vdupq_n_s32(0);

    ggml_int8x16x4_t deltas;
    deltas.val[0] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(+1));
    deltas.val[1] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(+1));
    deltas.val[2] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(-1));
    deltas.val[3] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1));

    ggml_int8x16x4_t q1b;
    ggml_int8x16x4_t q8b;

    uint32_t aux32;
    const uint8_t * aux8 = (const uint8_t *)&aux32;

    float sumf = 0;
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int32x4_t sumi1 = mzero;
        int32x4_t sumi2 = mzero;

        for (int ib = 0; ib < QK_K/32; ib += 2) {

            q1b.val[0] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[0] | ((qh[0] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[1] | ((qh[0] << 4) & 0x700)))));
            q1b.val[1] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[2] | ((qh[1] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[3] | ((qh[1] << 4) & 0x700)))));
            q1b.val[2] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[4] | ((qh[2] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[5] | ((qh[2] << 4) & 0x700)))));
            q1b.val[3] = vcombine_s8(vld1_s8((const int8_t *)(iq1s_grid + (qs[6] | ((qh[3] << 8) & 0x700)))),
                                     vld1_s8((const int8_t *)(iq1s_grid + (qs[7] | ((qh[3] << 4) & 0x700)))));

            q8b = ggml_vld1q_s8_x4(q8); q8 += 64;

            const int32x4_t p1 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[0], q8b.val[0]), ggml_vdotq_s32(mzero, q1b.val[1], q8b.val[1]));
            const int32x4_t p2 = vpaddq_s32(ggml_vdotq_s32(mzero, q1b.val[2], q8b.val[2]), ggml_vdotq_s32(mzero, q1b.val[3], q8b.val[3]));
            const int32x4_t p12 = vpaddq_s32(p1, p2);

            const uint32_t * qh32 = (const uint32_t *)qh; // we are 4-byte aligned, so we can do that
            aux32 = ((qh32[0] >> 3) & 0x01010101) | ((qh32[0] >> 6) & 0x02020202);

            const int32x4_t p3 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[0]], q8b.val[0]), ggml_vdotq_s32(mzero, deltas.val[aux8[1]], q8b.val[1]));
            const int32x4_t p4 = vpaddq_s32(ggml_vdotq_s32(mzero, deltas.val[aux8[2]], q8b.val[2]), ggml_vdotq_s32(mzero, deltas.val[aux8[3]], q8b.val[3]));
            const int32x4_t p34 = vpaddq_s32(p3, p4);

            int32x4_t scales_4 = ggml_vld1q_u32(sc[ib/2] >> 0, sc[ib/2] >> 3, sc[ib/2] >> 6, sc[ib/2] >> 9);

            scales_4 = vaddq_s32(vshlq_n_s32(vandq_s32(scales_4, mask), 1), mone);

            sumi1 = vmlaq_s32(sumi1, scales_4, p12);
            sumi2 = vmlaq_s32(sumi2, scales_4, p34);

            qs += 8; qh += 4;

        }

        sumf += y[i].d * GGML_FP16_TO_FP32(scale.f16) * (vaddvq_s32(sumi1) + IQ1M_DELTA * vaddvq_s32(sumi2));
    }

    *s = sumf;

#elif defined __AVX2__

    const __m256i mask = _mm256_set1_epi16(0x7);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m256i q1b_1 = _mm256_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)],
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]
            );
            const __m256i q1b_2 = _mm256_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)],
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]
            );
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            const __m256i dot1 = mul_add_epi8(q1b_1, q8b_1);
            const __m256i dot2 = mul_add_epi8(q1b_2, q8b_2);

            const __m256i delta1 = _mm256_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m256i delta2 = _mm256_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            const __m256i dot3 = mul_add_epi8(delta1, q8b_1);
            const __m256i dot4 = mul_add_epi8(delta2, q8b_2);

            __m256i scale1 = MM256_SET_M128I(_mm_set1_epi16(sc[ib/2] >> 3), _mm_set1_epi16(sc[ib/2] >> 0));
            __m256i scale2 = MM256_SET_M128I(_mm_set1_epi16(sc[ib/2] >> 9), _mm_set1_epi16(sc[ib/2] >> 6));

            scale1 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scale1, mask), 1), mone);
            scale2 = _mm256_add_epi16(_mm256_slli_epi16(_mm256_and_si256(scale2, mask), 1), mone);
            const __m256i p1 = _mm256_madd_epi16(dot1, scale1);
            const __m256i p2 = _mm256_madd_epi16(dot2, scale2);
            const __m256i p3 = _mm256_madd_epi16(dot3, scale1);
            const __m256i p4 = _mm256_madd_epi16(dot4, scale2);

            sumi1 = _mm256_add_epi32(sumi1, _mm256_add_epi32(p1, p2));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_add_epi32(p3, p4));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi1), accum1);
        accum2 = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi2), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#elif defined __AVX__
    const __m128i mask = _mm_set1_epi16(0x7);
    const __m128i mone = _mm_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q1b_1_0 = _mm_set_epi64x(
                    iq1s_grid[qs[1] | (((uint16_t)qh[0] << 4) & 0x700)], iq1s_grid[qs[0] | (((uint16_t)qh[0] << 8) & 0x700)]);
            const __m128i q1b_1_1 = _mm_set_epi64x(
                    iq1s_grid[qs[3] | (((uint16_t)qh[1] << 4) & 0x700)], iq1s_grid[qs[2] | (((uint16_t)qh[1] << 8) & 0x700)]);
            const __m128i q1b_2_0 = _mm_set_epi64x(
                    iq1s_grid[qs[5] | (((uint16_t)qh[2] << 4) & 0x700)], iq1s_grid[qs[4] | (((uint16_t)qh[2] << 8) & 0x700)]);
            const __m128i q1b_2_1 = _mm_set_epi64x(
                    iq1s_grid[qs[7] | (((uint16_t)qh[3] << 4) & 0x700)], iq1s_grid[qs[6] | (((uint16_t)qh[3] << 8) & 0x700)]);
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;

            const __m128i dot1_0 = mul_add_epi8_sse(q1b_1_0, q8b_1_0);
            const __m128i dot1_1 = mul_add_epi8_sse(q1b_1_1, q8b_1_1);
            const __m128i dot2_0 = mul_add_epi8_sse(q1b_2_0, q8b_2_0);
            const __m128i dot2_1 = mul_add_epi8_sse(q1b_2_1, q8b_2_1);

            const __m128i delta1_0 = _mm_set_epi64x(qh[0] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[0] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta1_1 = _mm_set_epi64x(qh[1] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[1] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_0 = _mm_set_epi64x(qh[2] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[2] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);
            const __m128i delta2_1 = _mm_set_epi64x(qh[3] & 0x80 ? 0xffffffffffffffff : 0x0101010101010101,
                                                     qh[3] & 0x08 ? 0xffffffffffffffff : 0x0101010101010101);

            const __m128i dot3_0 = mul_add_epi8_sse(delta1_0, q8b_1_0);
            const __m128i dot3_1 = mul_add_epi8_sse(delta1_1, q8b_1_1);
            const __m128i dot4_0 = mul_add_epi8_sse(delta2_0, q8b_2_0);
            const __m128i dot4_1 = mul_add_epi8_sse(delta2_1, q8b_2_1);

            __m128i scale1_0 = _mm_set1_epi16(sc[ib/2] >> 0);
            __m128i scale1_1 = _mm_set1_epi16(sc[ib/2] >> 3);
            __m128i scale2_0 = _mm_set1_epi16(sc[ib/2] >> 6);
            __m128i scale2_1 = _mm_set1_epi16(sc[ib/2] >> 9);

            scale1_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_0, mask), 1), mone);
            scale1_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale1_1, mask), 1), mone);
            scale2_0 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_0, mask), 1), mone);
            scale2_1 = _mm_add_epi16(_mm_slli_epi16(_mm_and_si128(scale2_1, mask), 1), mone);
            const __m128i p1_0 = _mm_madd_epi16(dot1_0, scale1_0);
            const __m128i p1_1 = _mm_madd_epi16(dot1_1, scale1_1);
            const __m128i p2_0 = _mm_madd_epi16(dot2_0, scale2_0);
            const __m128i p2_1 = _mm_madd_epi16(dot2_1, scale2_1);
            const __m128i p3_0 = _mm_madd_epi16(dot3_0, scale1_0);
            const __m128i p3_1 = _mm_madd_epi16(dot3_1, scale1_1);
            const __m128i p4_0 = _mm_madd_epi16(dot4_0, scale2_0);
            const __m128i p4_1 = _mm_madd_epi16(dot4_1, scale2_1);

            sumi1_0 = _mm_add_epi32(sumi1_0, _mm_add_epi32(p1_0, p2_0));
            sumi1_1 = _mm_add_epi32(sumi1_1, _mm_add_epi32(p1_1, p2_1));
            sumi2_0 = _mm_add_epi32(sumi2_0, _mm_add_epi32(p3_0, p4_0));
            sumi2_1 = _mm_add_epi32(sumi2_1, _mm_add_epi32(p3_1, p4_1));

            qs += 8; qh += 4;
        }

        const __m256 d = _mm256_set1_ps(y[i].d * GGML_FP16_TO_FP32(scale.f16));

        accum1 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi1_1, sumi1_0))), accum1);
        accum2 = _mm256_add_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(MM256_SET_M128I(sumi2_1, sumi2_0))), accum2);
    }

    *s = hsum_float_8(accum1) + IQ1M_DELTA * hsum_float_8(accum2);

#else

    int sum1[2], sum2[2], delta[4];

    float sumf = 0;
    for (int i = 0; i < nb; i++) {

        const int8_t   * q8 = y[i].qs;
        const uint8_t  * qs = x[i].qs;
        const uint8_t  * qh = x[i].qh;
        const uint16_t * sc = (const uint16_t *)x[i].scales;

        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            delta[0] = qh[0] & 0x08 ? -1 : 1;
            delta[1] = qh[0] & 0x80 ? -1 : 1;
            delta[2] = qh[1] & 0x08 ? -1 : 1;
            delta[3] = qh[1] & 0x80 ? -1 : 1;
            sum1[0] = sum1[1] = sum2[0] = sum2[1] = 0;
            for (int l = 0; l < 4; ++l) {
                const int8_t * grid = (const int8_t *)(iq1s_grid + (qs[l] | (((uint16_t)qh[l/2] << (8 - 4*(l%2))) & 0x700)));
                int lsum1 = 0, lsum2 = 0;
                for (int j = 0; j < 8; ++j) {
                    lsum1 += q8[j] * grid[j];
                    lsum2 += q8[j];
                }
                q8 += 8;
                sum1[l/2] += lsum1;
                sum2[l/2] += lsum2*delta[l];
            }

            const int ls1 = 2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1;
            const int ls2 = 2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1;

            sumi1 += sum1[0] * ls1 + sum1[1] * ls2;
            sumi2 += sum2[0] * ls1 + sum2[1] * ls2;
            qs += 4;
            qh += 2;
        }

        sumf += GGML_FP16_TO_FP32(scale.f16) * y[i].d * (sumi1 + IQ1M_DELTA * sumi2);
    }

    *s = sumf;

#endif
}

void ggml_vec_dot_iq4_nl_q8_0(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK4_NL == 0);
    static_assert(QK4_NL == QK8_0, "QK4_NL and QK8_0 must be the same");

    const block_iq4_nl * restrict x = vx;
    const block_q8_0   * restrict y = vy;

    const int nb = n / QK4_NL;

    int ib = 0;
    float sumf = 0;

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    uint8x16x2_t q4bits;
    int8x16x4_t q4b;
    int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    for (; ib + 1 < nb; ib += 2) {

        q4bits.val[0] = vld1q_u8(x[ib + 0].qs);
        q4bits.val[1] = vld1q_u8(x[ib + 1].qs);
        q8b.val[0]    = vld1q_s8(y[ib + 0].qs);
        q8b.val[1]    = vld1q_s8(y[ib + 0].qs + 16);
        q8b.val[2]    = vld1q_s8(y[ib + 1].qs);
        q8b.val[3]    = vld1q_s8(y[ib + 1].qs + 16);

        q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
        q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
        q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
        q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

        prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
        prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

        sumf +=
            GGML_FP16_TO_FP32(x[ib+0].d) * GGML_FP16_TO_FP32(y[ib + 0].d) * vaddvq_s32(prod_1) +
            GGML_FP16_TO_FP32(x[ib+1].d) * GGML_FP16_TO_FP32(y[ib + 1].d) * vaddvq_s32(prod_2);
    }

#elif defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);

    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        accum1 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[ib + 0].d)*GGML_FP16_TO_FP32(x[ib + 0].d)),
                _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(y[ib + 1].d)*GGML_FP16_TO_FP32(x[ib + 1].d)),
                _mm256_cvtepi32_ps(p_2), accum2);
    }

    sumf = hsum_float_8(_mm256_add_ps(accum1, accum2));

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)x[ib + 1].qs);
        const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs);
        const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)y[ib + 0].qs + 1);
        const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs);
        const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)y[ib + 1].qs + 1);

        const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
        const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
        const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
        const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));

        const __m256 p = mul_sum_i8_quad_float(q4b_1_0, q4b_1_1, q4b_2_0, q4b_2_1, q8b_1_0, q8b_1_1, q8b_2_0, q8b_2_1);
        const __m256 deltas = quad_fp16_delta_float(x[ib].d, y[ib].d, x[ib + 1].d, y[ib + 1].d);
        accum = _mm256_add_ps(_mm256_mul_ps(deltas, p), accum);
    }

    sumf = hsum_float_8(accum);

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector signed int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);

    const vector signed char values = vec_xl( 0, kvalues_iq4nl);

#pragma GCC unroll 4
    for (; ib < nb; ++ib) {
        __builtin_prefetch(x[ib].qs, 0, 1);
        __builtin_prefetch(y[ib].qs, 0, 1);


        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ib].d));
        vector float vyd = vec_splats(GGML_FP16_TO_FP32(y[ib].d));
        vector float vd = vec_mul(vxd, vyd);

        vector signed char qxs = (vector signed char)vec_xl( 0, x[ib].qs);
        vector signed char q4x0 = vec_and(qxs, lowMask);
        vector signed char q4x1 = vec_sr(qxs, v4);

        q4x0 = vec_perm(values, values, (vector unsigned char)q4x0);
        q4x1 = vec_perm(values, values, (vector unsigned char)q4x1);

        vector signed char q8y0 = vec_xl( 0, y[ib].qs);
        vector signed char q8y1 = vec_xl(16, y[ib].qs);

        vector signed short qv0 = vec_add(vec_mule(q4x0, q8y0), vec_mulo(q4x0, q8y0));
        vector signed short qv1 = vec_add(vec_mule(q4x1, q8y1), vec_mulo(q4x1, q8y1));

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;

        vsumi0 = vec_sum4s(qv0, vsumi0);
        vsumi1 = vec_sum4s(qv1, vsumi1);

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
    }

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    sumf = vec_extract(vsumf0, 0);

#elif defined (__loongarch_asx)

    const __m128i values128 = __lsx_vld((const __m128i*)kvalues_iq4nl, 0);
    const __m128i m4b  = __lsx_vreplgr2vr_b(0x0f);
    const __m256i mone = __lasx_xvreplgr2vr_h(1);

    __m256 accum1 = (__m256)__lasx_xvldi(0);
    __m256 accum2 = (__m256)__lasx_xvldi(0);
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = __lsx_vld((const __m128i*)x[ib + 0].qs, 0);
        const __m128i q4bits_2 = __lsx_vld((const __m128i*)x[ib + 1].qs, 0);
        const __m256i q8b_1 = __lasx_xvld((const __m256i *)y[ib + 0].qs, 0);
        const __m256i q8b_2 = __lasx_xvld((const __m256i *)y[ib + 1].qs, 0);
        const __m256i q4b_1 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_1, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_1, m4b)));
        const __m256i q4b_2 = lasx_insertf128(lsx_shuffle_b(values128, __lsx_vand_v(__lsx_vsrli_h(q4bits_2, 4), m4b)),
                                              lsx_shuffle_b(values128, __lsx_vand_v(q4bits_2, m4b)));
        const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = lasx_madd_h(p16_1, mone);
        const __m256i p_2 = lasx_madd_h(p16_2, mone);
        accum1 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib + 0].d)*GGML_FP16_TO_FP32(x[ib + 0].d)),
                __lasx_xvffint_s_w(p_1), accum1);
        accum2 = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(y[ib + 1].d)*GGML_FP16_TO_FP32(x[ib + 1].d)),
                __lasx_xvffint_s_w(p_2), accum2);
    }

    sumf = hsum_float_8(__lasx_xvfadd_s(accum1, accum2));

#endif
    for (; ib < nb; ++ib) {
        const float d = GGML_FP16_TO_FP32(y[ib].d)*GGML_FP16_TO_FP32(x[ib].d);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK4_NL/2; ++j) {
            sumi1 += y[ib].qs[j+       0] * kvalues_iq4nl[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j+QK4_NL/2] * kvalues_iq4nl[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}

void ggml_vec_dot_iq4_xs_q8_K(int n, float * restrict s, size_t bs, const void * restrict vx, size_t bx, const void * restrict vy, size_t by, int nrc) {
    assert(nrc == 1);
    UNUSED(nrc);
    UNUSED(bx);
    UNUSED(by);
    UNUSED(bs);
    assert(n % QK_K == 0);

    const block_iq4_xs * restrict x = vx;
    const block_q8_K   * restrict y = vy;

    const int nb = n / QK_K;

#if defined __ARM_NEON
    const int8x16_t values = vld1q_s8(kvalues_iq4nl);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    ggml_uint8x16x2_t q4bits;
    ggml_int8x16x4_t q4b;
    ggml_int8x16x4_t q8b;
    int32x4_t prod_1, prod_2;

    float sumf = 0;

    for (int ibl = 0; ibl < nb; ++ibl) {

        const int8_t  * q8 = y[ibl].qs;
        const uint8_t * q4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;

        int sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < QK_K/64; ++ib) {

            q4bits = ggml_vld1q_u8_x2(q4); q4 += 32;
            q8b    = ggml_vld1q_s8_x4(q8); q8 += 64;

            q4b.val[0] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[0], m4b));
            q4b.val[1] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
            q4b.val[2] = ggml_vqtbl1q_s8(values, vandq_u8  (q4bits.val[1], m4b));
            q4b.val[3] = ggml_vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));

            prod_1 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[0], q8b.val[0]), q4b.val[1], q8b.val[1]);
            prod_2 = ggml_vdotq_s32(ggml_vdotq_s32(vdupq_n_s32(0), q4b.val[2], q8b.val[2]), q4b.val[3], q8b.val[3]);

            int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            sumi1 += vaddvq_s32(prod_1) * ls1;
            sumi2 += vaddvq_s32(prod_2) * ls2;

        }

        sumf += GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d * (sumi1 + sumi2);
    }

    *s = sumf;

#elif defined __AVX2__

    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)qs);  qs += 16;
            const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)q8); q8 += 32;
            const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
            const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                                  _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m256i p_1 = _mm256_madd_epi16(p16_1, _mm256_set1_epi16(ls1));
            const __m256i p_2 = _mm256_madd_epi16(p16_2, _mm256_set1_epi16(ls2));
            sumi1 = _mm256_add_epi32(p_1, sumi1);
            sumi2 = _mm256_add_epi32(p_2, sumi2);
        }
        accum = _mm256_fmadd_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#elif defined __AVX__
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_iq4nl);
    const __m128i m4b  = _mm_set1_epi8(0x0f);

    __m256 accum = _mm256_setzero_ps();
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m128i sumi1_0 = _mm_setzero_si128();
        __m128i sumi1_1 = _mm_setzero_si128();
        __m128i sumi2_0 = _mm_setzero_si128();
        __m128i sumi2_1 = _mm_setzero_si128();
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q4bits_2 = _mm_loadu_si128((const __m128i *)qs); qs += 16;
            const __m128i q8b_1_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_1_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_0 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q8b_2_1 = _mm_loadu_si128((const __m128i *)q8); q8 += 16;
            const __m128i q4b_1_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b));
            const __m128i q4b_1_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b));
            const __m128i q4b_2_0 = _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b));
            const __m128i q4b_2_1 = _mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b));
            const __m128i p16_1_0 = mul_add_epi8_sse(q4b_1_0, q8b_1_0);
            const __m128i p16_1_1 = mul_add_epi8_sse(q4b_1_1, q8b_1_1);
            const __m128i p16_2_0 = mul_add_epi8_sse(q4b_2_0, q8b_2_0);
            const __m128i p16_2_1 = mul_add_epi8_sse(q4b_2_1, q8b_2_1);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            const __m128i p_1_0 = _mm_madd_epi16(p16_1_0, _mm_set1_epi16(ls1));
            const __m128i p_1_1 = _mm_madd_epi16(p16_1_1, _mm_set1_epi16(ls1));
            const __m128i p_2_0 = _mm_madd_epi16(p16_2_0, _mm_set1_epi16(ls2));
            const __m128i p_2_1 = _mm_madd_epi16(p16_2_1, _mm_set1_epi16(ls2));
            sumi1_0 = _mm_add_epi32(p_1_0, sumi1_0);
            sumi1_1 = _mm_add_epi32(p_1_1, sumi1_1);
            sumi2_0 = _mm_add_epi32(p_2_0, sumi2_0);
            sumi2_1 = _mm_add_epi32(p_2_1, sumi2_1);
        }
        __m128i sumi12_0 = _mm_add_epi32(sumi1_0, sumi2_0);
        __m128i sumi12_1 = _mm_add_epi32(sumi1_1, sumi2_1);
        accum = _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                _mm256_cvtepi32_ps(MM256_SET_M128I(sumi12_1, sumi12_0))), accum);
    }

    *s = hsum_float_8(accum);

#elif defined(__POWER9_VECTOR__)
    const vector signed char lowMask = vec_splats((signed char)0xF);
    const vector int v0 = vec_splats((int32_t)0);
    const vector unsigned char v4 = vec_splats((unsigned char)0x4);

    vector float vsumf0 = vec_splats(0.0f);
    vector float vsumf1 = vec_splats(0.0f);
    vector float vsumf2 = vec_splats(0.0f);
    vector float vsumf3 = vec_splats(0.0f);

    const vector signed char values = vec_xl( 0, kvalues_iq4nl);

    for (int ibl = 0; ibl < nb; ++ibl) {

        vector float vxd = vec_splats(GGML_FP16_TO_FP32(x[ibl].d));
        vector float vyd = vec_splats(y[ibl].d);
        vector float vd = vec_mul(vxd, vyd);

        vector signed int vsumi0 = v0;
        vector signed int vsumi1 = v0;
        vector signed int vsumi2 = v0;
        vector signed int vsumi3 = v0;

        uint16_t h = x[ibl].scales_h;

        const uint8_t * restrict q4 = x[ibl].qs;
        const uint8_t * restrict sc = x[ibl].scales_l;
        const int8_t  * restrict q8 = y[ibl].qs;

        for (int ib = 0; ib < QK_K/64; ib ++ ) {
            __builtin_prefetch(q4, 0, 1);
            __builtin_prefetch(q8, 0, 1);

            vector signed char qxs0 = (vector signed char)vec_xl( 0, q4);
            vector signed char qxs1 = (vector signed char)vec_xl(16, q4);
            q4 += 32;

            vector signed char q4x00 = (vector signed char)vec_and(qxs0, lowMask);
            vector signed char q4x01 = (vector signed char)vec_sr(qxs0, v4);
            vector signed char q4x10 = (vector signed char)vec_and(qxs1, lowMask);
            vector signed char q4x11 = (vector signed char)vec_sr(qxs1, v4);

            q4x00 = vec_perm(values, values, (vector unsigned char)q4x00);
            q4x01 = vec_perm(values, values, (vector unsigned char)q4x01);
            q4x10 = vec_perm(values, values, (vector unsigned char)q4x10);
            q4x11 = vec_perm(values, values, (vector unsigned char)q4x11);

            vector signed char q8y0 = vec_xl( 0, q8);
            vector signed char q8y1 = vec_xl(16, q8);
            vector signed char q8y2 = vec_xl(32, q8);
            vector signed char q8y3 = vec_xl(48, q8);
            q8 += 64;

            vector signed short qv0 = vec_add(vec_mule(q4x00, q8y0), vec_mulo(q4x00, q8y0));
            vector signed short qv1 = vec_add(vec_mule(q4x01, q8y1), vec_mulo(q4x01, q8y1));
            vector signed short qv2 = vec_add(vec_mule(q4x10, q8y2), vec_mulo(q4x10, q8y2));
            vector signed short qv3 = vec_add(vec_mule(q4x11, q8y3), vec_mulo(q4x11, q8y3));

            const uint16_t ls0 = (uint16_t)(((sc[0] & 0xf) | ((h << 4) & 0x30)) - 32);
            const uint16_t ls1 = (uint16_t)(((sc[0] >>  4) | ((h << 2) & 0x30)) - 32);
            h >>= 4;
            sc ++;

            vector signed short vscales01 = vec_splats((int16_t)ls0);
            vector signed short vscales23 = vec_splats((int16_t)ls1);

            vsumi0 = vec_msum(qv0, vscales01, vsumi0);
            vsumi1 = vec_msum(qv1, vscales01, vsumi1);
            vsumi2 = vec_msum(qv2, vscales23, vsumi2);
            vsumi3 = vec_msum(qv3, vscales23, vsumi3);
        }

        vsumf0 = vec_madd(vec_ctf(vsumi0, 0), vd, vsumf0);
        vsumf1 = vec_madd(vec_ctf(vsumi1, 0), vd, vsumf1);
        vsumf2 = vec_madd(vec_ctf(vsumi2, 0), vd, vsumf2);
        vsumf3 = vec_madd(vec_ctf(vsumi3, 0), vd, vsumf3);
    }

    vsumf0 = vec_add(vsumf0, vsumf2);
    vsumf1 = vec_add(vsumf1, vsumf3);

    vsumf0 = vec_add(vsumf0, vsumf1);

    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 4));
    vsumf0 = vec_add(vsumf0, vec_sld(vsumf0, vsumf0, 8));

    *s = vec_extract(vsumf0, 0);

#elif defined(__loongarch_asx)

    const __m128i values128 = __lsx_vld((const __m128i*)kvalues_iq4nl, 0);
    const __m128i m4b  = __lsx_vreplgr2vr_b(0x0f);

    __m256 accum = (__m256)__lasx_xvldi(0);
    __m256i tmp1;
    __m128i tmp0, tmp2, tmp3, tmp4, mask_8f, mask;

    mask_8f = __lsx_vreplgr2vr_b(0x8f);
    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        __m256i sumi1 = __lasx_xvldi(0);
        __m256i sumi2 = __lasx_xvldi(0);
        __m128i zero = __lsx_vldi(0);
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const __m128i q4bits_1 = __lsx_vld((const __m128i*)qs, 0);  qs += 16;
            const __m128i q4bits_2 = __lsx_vld((const __m128i*)qs, 0);  qs += 16;
            const __m256i q8b_1 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            const __m256i q8b_2 = __lasx_xvld((const __m256i *)q8, 0); q8 += 32;
            tmp2 = __lsx_vand_v(__lsx_vand_v(__lsx_vsrli_h(q4bits_1, 4), m4b), mask_8f);
            tmp0 = __lsx_vori_b(tmp2, 0x10);
            mask = __lsx_vsle_b(zero, tmp2);
            tmp3 = __lsx_vand_v(tmp0, mask);
            tmp3 = __lsx_vshuf_b(values128, zero, tmp3);

            tmp2 = __lsx_vand_v(__lsx_vand_v(q4bits_1, m4b), mask_8f);
            tmp0 = __lsx_vori_b(tmp2, 0x10);
            mask = __lsx_vsle_b(zero, tmp2);
            tmp4 = __lsx_vand_v(tmp0, mask);
            tmp4 = __lsx_vshuf_b(values128, zero, tmp4);

            const __m256i q4b_1 = lasx_insertf128(tmp3, tmp4);

            tmp2 = __lsx_vand_v(__lsx_vand_v(__lsx_vsrli_h(q4bits_2, 4), m4b), mask_8f);
            tmp0 = __lsx_vori_b(tmp2, 0x10);
            mask = __lsx_vsle_b(zero, tmp2);
            tmp3 = __lsx_vand_v(tmp0, mask);
            tmp3 = __lsx_vshuf_b(values128, zero, tmp3);

            tmp2 = __lsx_vand_v(__lsx_vand_v(q4bits_2, m4b), mask_8f);
            tmp0 = __lsx_vori_b(tmp2, 0x10);
            mask = __lsx_vsle_b(zero, tmp2);
            tmp4 = __lsx_vand_v(tmp0, mask);
            tmp4 = __lsx_vshuf_b(values128, zero, tmp4);

            const __m256i q4b_2 = lasx_insertf128(tmp3, tmp4);

            const __m256i p16_1 = mul_add_epi8(q4b_1, q8b_1);
            const __m256i p16_2 = mul_add_epi8(q4b_2, q8b_2);
            const int16_t ls1 = ((x[ibl].scales_l[ib/2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((x[ibl].scales_l[ib/2] >>  4) | ((sh << 2) & 0x30)) - 32;
            sh >>= 4;
            __m256i tmp5, tmp6;
            tmp1 = __lasx_xvreplgr2vr_h(ls1);
            tmp5 = __lasx_xvmulwev_w_h(p16_1, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(p16_1, tmp1);
            const __m256i p_1 = __lasx_xvadd_w(tmp5, tmp6);
            tmp1 = __lasx_xvreplgr2vr_h(ls2);
            tmp5 = __lasx_xvmulwev_w_h(p16_2, tmp1);
            tmp6 = __lasx_xvmulwod_w_h(p16_2, tmp1);
            const __m256i p_2 = __lasx_xvadd_w(tmp5, tmp6);
            sumi1 = __lasx_xvadd_w(p_1, sumi1);
            sumi2 = __lasx_xvadd_w(p_2, sumi2);
        }
        accum = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(GGML_FP16_TO_FP32(x[ibl].d)*y[ibl].d),
                __lasx_xvffint_s_w(__lasx_xvadd_w(sumi1, sumi2)), accum);
    }

    *s = hsum_float_8(accum);

#else
    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t h = x[ibl].scales_h;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ib += 2) {
            const uint8_t ls1 = (x[ibl].scales_l[ib/2] & 0xf) | ((h << 4) & 0x30);
            const uint8_t ls2 = (x[ibl].scales_l[ib/2] >>  4) | ((h << 2) & 0x30);
            h >>= 4;
            const float d1 = d4d8*(ls1 - 32);
            const float d2 = d4d8*(ls2 - 32);
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d1 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
            sumi1 = sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * kvalues_iq4nl[qs[j] & 0xf];
                sumi2 += q8[j+16] * kvalues_iq4nl[qs[j] >>  4];
            }
            sumf += d2 * (sumi1 + sumi2);
            qs += 16;
            q8 += 32;
        }
    }
    *s = sumf;
#endif
}

// ============================ 4-bit non-linear quants

void quantize_row_iq4_nl(const float * restrict x, void * restrict y, int64_t k) {
    assert(k % QK4_NL == 0);
    quantize_row_iq4_nl_ref(x, y, k);
}

void quantize_row_iq4_xs(const float * restrict x, void * restrict y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_xs(x, y, 1, k, NULL);
}

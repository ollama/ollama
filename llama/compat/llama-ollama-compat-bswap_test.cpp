// llama-ollama-compat-bswap_test.cpp
//
// Standalone unit tests for the big-endian tensor byte-swap logic introduced
// in 003-tensor-data-big-endian-byteswap.patch.
//
// The tests are architecture-agnostic: they verify the byte-swap behavior
// directly regardless of the host's endianness.  Running on little-endian
// hardware (x86_64, ARM64) is sufficient to validate correctness.
//
// Build (without any llama.cpp dependency):
//
//   c++ -std=c++17 -DTEST_BSWAP_STANDALONE \
//       llama-ollama-compat-bswap_test.cpp -o bswap_test && ./bswap_test
//
// The -DTEST_BSWAP_STANDALONE guard lets the same file be compiled as part of
// a future CMake test target that links against the real llama.cpp types.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Minimal type stubs — only needed when building standalone.
// When linked against real llama.cpp these come from ggml.h.
// ---------------------------------------------------------------------------
#if defined(TEST_BSWAP_STANDALONE)

typedef enum ggml_type {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_I8   = 24,
    GGML_TYPE_I32  = 26,
    GGML_TYPE_I64  = 27,
} ggml_type;

// Block sizes match llama.cpp (QK_K = 256, QK4_0 = 32 etc.).
static size_t ggml_type_size(ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 2 + 16;          // 18
        case GGML_TYPE_Q4_1: return 2 + 2 + 16;      // 20
        case GGML_TYPE_Q5_0: return 2 + 4 + 16;      // 22
        case GGML_TYPE_Q5_1: return 2 + 2 + 4 + 16;  // 24
        case GGML_TYPE_Q8_0: return 2 + 32;           // 34
        case GGML_TYPE_Q2_K: return 84;
        case GGML_TYPE_Q3_K: return 110;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        default:             return 0;
    }
}
#endif // TEST_BSWAP_STANDALONE

// ---------------------------------------------------------------------------
// The byte-swap implementation under test.
// This is the same logic as bswap_buf() in 003-tensor-data-big-endian-byteswap.patch.
// It lives here so the tests compile standalone without patching llama.cpp.
// ---------------------------------------------------------------------------
static void bswap2(uint8_t * b) {
    const uint8_t t = b[0]; b[0] = b[1]; b[1] = t;
}

static void bswap4(uint8_t * b) {
    uint8_t t;
    t = b[0]; b[0] = b[3]; b[3] = t;
    t = b[1]; b[1] = b[2]; b[2] = t;
}

static void bswap_buf(ggml_type type, uint8_t * data, size_t nbytes) {
    if (type == GGML_TYPE_F16 || type == GGML_TYPE_BF16) {
        for (size_t off = 0; off + 1 < nbytes; off += 2) bswap2(data + off);
        return;
    }
    if (type == GGML_TYPE_F32) {
        for (size_t off = 0; off + 3 < nbytes; off += 4) bswap4(data + off);
        return;
    }
    if (type == GGML_TYPE_I8 || type == GGML_TYPE_I32 || type == GGML_TYPE_I64) return;

    const size_t blk = ggml_type_size(type);
    if (blk == 0 || nbytes % blk != 0) return;

    if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q8_0) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) bswap2(data + off);
        return;
    }
    if (type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_1) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) {
            bswap2(data + off);
            bswap2(data + off + 2);
        }
        return;
    }
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) {
            bswap2(data + off);
            bswap2(data + off + 2);
        }
        return;
    }
    if (type == GGML_TYPE_Q2_K) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) {
            bswap2(data + off + 80);
            bswap2(data + off + 82);
        }
        return;
    }
    if (type == GGML_TYPE_Q3_K) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) bswap2(data + off + 108);
        return;
    }
    if (type == GGML_TYPE_Q6_K) {
        for (size_t off = 0; off + blk <= nbytes; off += blk) bswap2(data + off + 208);
        return;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Fill every byte in [begin, end) with a sentinel that must not change.
static void fill_sentinel(uint8_t * buf, size_t begin, size_t end, uint8_t val = 0xAB) {
    for (size_t i = begin; i < end; ++i) buf[i] = val;
}

static int g_failures = 0;

#define EXPECT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL [%s:%d] %s\n", __func__, __LINE__, msg); \
            ++g_failures; \
        } \
    } while (0)

// ---------------------------------------------------------------------------
// Endianness self-consistency test (B)
// ---------------------------------------------------------------------------

static void test_endian_detection_consistent() {
    // Verify that the compile-time __BYTE_ORDER__ macro agrees with a runtime
    // probe.  Both paths must return the same answer.
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
    const bool compile_time_be = (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__);
#elif defined(__BIG_ENDIAN__)
    const bool compile_time_be = true;
#else
    const bool compile_time_be = false;
#endif

    // Runtime probe: store a known 4-byte value and inspect the first byte.
    const uint32_t probe = 0x01020304u;
    uint8_t buf[4];
    std::memcpy(buf, &probe, 4);
    const bool runtime_be = (buf[0] == 0x01);

    EXPECT(compile_time_be == runtime_be,
           "compile-time and runtime endian detection disagree");

    if (runtime_be) {
        std::printf("  (note: running on a big-endian host)\n");
    }
}

// ---------------------------------------------------------------------------
// Scalar type tests (A)
// ---------------------------------------------------------------------------

static void test_bswap_f16_single_element() {
    // FP16 1.0f in little-endian: bytes [0x00, 0x3C]
    // After swap:                        [0x3C, 0x00]
    uint8_t buf[2] = {0x00, 0x3C};
    bswap_buf(GGML_TYPE_F16, buf, 2);
    EXPECT(buf[0] == 0x3C && buf[1] == 0x00, "F16 swap bytes 0-1");
}

static void test_bswap_f16_two_elements() {
    // Two FP16 values: 1.0f [00 3C] and -1.0f [00 BC]
    uint8_t buf[4] = {0x00, 0x3C, 0x00, 0xBC};
    bswap_buf(GGML_TYPE_F16, buf, 4);
    EXPECT(buf[0] == 0x3C && buf[1] == 0x00, "F16 element 0 swapped");
    EXPECT(buf[2] == 0xBC && buf[3] == 0x00, "F16 element 1 swapped");
}

static void test_bswap_bf16_single_element() {
    // BF16 1.0f: [0x00, 0x3F] in LE -> [0x3F, 0x00] after swap
    uint8_t buf[2] = {0x00, 0x3F};
    bswap_buf(GGML_TYPE_BF16, buf, 2);
    EXPECT(buf[0] == 0x3F && buf[1] == 0x00, "BF16 swap");
}

static void test_bswap_f32_single_element() {
    // F32 1.0f in LE: [0x00, 0x00, 0x80, 0x3F]
    // After swap:     [0x3F, 0x80, 0x00, 0x00]
    uint8_t buf[4] = {0x00, 0x00, 0x80, 0x3F};
    bswap_buf(GGML_TYPE_F32, buf, 4);
    EXPECT(buf[0] == 0x3F, "F32 byte[0]");
    EXPECT(buf[1] == 0x80, "F32 byte[1]");
    EXPECT(buf[2] == 0x00, "F32 byte[2]");
    EXPECT(buf[3] == 0x00, "F32 byte[3]");
}

// ---------------------------------------------------------------------------
// Legacy quantization type tests (A) — one block per type
// ---------------------------------------------------------------------------

// Macro: allocate one block, set scale byte(s) to known LE values,
// fill the rest as sentinel, call bswap_buf, verify scale was swapped and
// sentinel bytes are untouched.
#define TEST_QUANT_ONE_SCALE(testname, qtype, scale_offset) \
static void testname() { \
    const size_t blk = ggml_type_size(qtype); \
    uint8_t buf[256] = {}; \
    fill_sentinel(buf, 0, blk); \
    /* Write LE FP16 1.0f [0x00, 0x3C] at scale_offset */ \
    buf[scale_offset]     = 0x00; \
    buf[scale_offset + 1] = 0x3C; \
    bswap_buf(qtype, buf, blk); \
    EXPECT(buf[scale_offset]     == 0x3C, #testname " scale byte[0]"); \
    EXPECT(buf[scale_offset + 1] == 0x00, #testname " scale byte[1]"); \
    /* Every non-scale byte must be untouched sentinel (0xAB) */ \
    for (size_t i = 0; i < blk; ++i) { \
        if (i == (size_t)(scale_offset) || i == (size_t)(scale_offset) + 1) continue; \
        EXPECT(buf[i] == 0xAB, #testname " sentinel byte disturbed"); \
    } \
}

TEST_QUANT_ONE_SCALE(test_bswap_q4_0_scale, GGML_TYPE_Q4_0, 0)
TEST_QUANT_ONE_SCALE(test_bswap_q5_0_scale, GGML_TYPE_Q5_0, 0)
TEST_QUANT_ONE_SCALE(test_bswap_q8_0_scale, GGML_TYPE_Q8_0, 0)
TEST_QUANT_ONE_SCALE(test_bswap_q3_k_scale, GGML_TYPE_Q3_K, 108)
TEST_QUANT_ONE_SCALE(test_bswap_q6_k_scale, GGML_TYPE_Q6_K, 208)

// Two-scale variants: Q4_1, Q5_1, Q4_K, Q5_K
#define TEST_QUANT_TWO_SCALES(testname, qtype, d_offset, m_offset) \
static void testname() { \
    const size_t blk = ggml_type_size(qtype); \
    uint8_t buf[256] = {}; \
    fill_sentinel(buf, 0, blk); \
    buf[d_offset]     = 0x00; buf[d_offset + 1] = 0x3C; /* d: LE FP16 1.0 */ \
    buf[m_offset]     = 0x00; buf[m_offset + 1] = 0xBC; /* m: LE FP16 -1.0 */ \
    bswap_buf(qtype, buf, blk); \
    EXPECT(buf[d_offset]     == 0x3C, #testname " d byte[0]"); \
    EXPECT(buf[d_offset + 1] == 0x00, #testname " d byte[1]"); \
    EXPECT(buf[m_offset]     == 0xBC, #testname " m byte[0]"); \
    EXPECT(buf[m_offset + 1] == 0x00, #testname " m byte[1]"); \
    for (size_t i = 0; i < blk; ++i) { \
        if (i==(size_t)(d_offset)||i==(size_t)(d_offset)+1) continue; \
        if (i==(size_t)(m_offset)||i==(size_t)(m_offset)+1) continue; \
        EXPECT(buf[i] == 0xAB, #testname " sentinel byte disturbed"); \
    } \
}

TEST_QUANT_TWO_SCALES(test_bswap_q4_1_scales, GGML_TYPE_Q4_1, 0, 2)
TEST_QUANT_TWO_SCALES(test_bswap_q5_1_scales, GGML_TYPE_Q5_1, 0, 2)
TEST_QUANT_TWO_SCALES(test_bswap_q4_k_scales, GGML_TYPE_Q4_K, 0, 2)
TEST_QUANT_TWO_SCALES(test_bswap_q5_k_scales, GGML_TYPE_Q5_K, 0, 2)

// Q2_K: d at 80, dmin at 82
static void test_bswap_q2_k_scales() {
    const size_t blk = ggml_type_size(GGML_TYPE_Q2_K);
    uint8_t buf[256] = {};
    fill_sentinel(buf, 0, blk);
    buf[80] = 0x00; buf[81] = 0x3C; // d
    buf[82] = 0x00; buf[83] = 0xBC; // dmin
    bswap_buf(GGML_TYPE_Q2_K, buf, blk);
    EXPECT(buf[80] == 0x3C && buf[81] == 0x00, "Q2_K d byte swapped");
    EXPECT(buf[82] == 0xBC && buf[83] == 0x00, "Q2_K dmin byte swapped");
    for (size_t i = 0; i < blk; ++i) {
        if (i==80||i==81||i==82||i==83) continue;
        EXPECT(buf[i] == 0xAB, "Q2_K sentinel byte disturbed");
    }
}

// ---------------------------------------------------------------------------
// Multi-block test: verify the per-block stride is correct (two blocks)
// ---------------------------------------------------------------------------
static void test_bswap_f16_multi_block_stride() {
    // Two F16 values in a row.  Verifies loop advances by 2 bytes, not 1.
    uint8_t buf[4] = {0xAA, 0xBB, 0xCC, 0xDD};
    bswap_buf(GGML_TYPE_F16, buf, 4);
    EXPECT(buf[0] == 0xBB && buf[1] == 0xAA, "F16 block 0 stride");
    EXPECT(buf[2] == 0xDD && buf[3] == 0xCC, "F16 block 1 stride");
}

static void test_bswap_q4_0_two_blocks_second_scale_correct() {
    const size_t blk = ggml_type_size(GGML_TYPE_Q4_0);
    uint8_t buf[256] = {};
    fill_sentinel(buf, 0, 2 * blk);
    // Block 0: scale at offset 0
    buf[0] = 0x11; buf[1] = 0x22;
    // Block 1: scale at offset blk
    buf[blk]   = 0x33; buf[blk+1] = 0x44;
    bswap_buf(GGML_TYPE_Q4_0, buf, 2 * blk);
    EXPECT(buf[0] == 0x22 && buf[1] == 0x11, "Q4_0 block 0 scale swapped");
    EXPECT(buf[blk] == 0x44 && buf[blk+1] == 0x33, "Q4_0 block 1 scale swapped");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("bswap unit tests\n");

    test_endian_detection_consistent();
    test_bswap_f16_single_element();
    test_bswap_f16_two_elements();
    test_bswap_bf16_single_element();
    test_bswap_f32_single_element();
    test_bswap_q4_0_scale();
    test_bswap_q5_0_scale();
    test_bswap_q8_0_scale();
    test_bswap_q4_1_scales();
    test_bswap_q5_1_scales();
    test_bswap_q4_k_scales();
    test_bswap_q5_k_scales();
    test_bswap_q2_k_scales();
    test_bswap_q3_k_scale();
    test_bswap_q6_k_scale();
    test_bswap_f16_multi_block_stride();
    test_bswap_q4_0_two_blocks_second_scale_correct();

    if (g_failures == 0) {
        std::printf("All tests passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d test(s) FAILED.\n", g_failures);
    return 1;
}

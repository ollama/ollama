// ============================================================================
// ggml_turboquant.h — TurboQuant KV Cache Compression for OLLaMA/llama.cpp
// Target: ROCm 7.x / gfx1201 (RX 9070 XT)
// Based on: TurboQuant (ICLR 2026, Google DeepMind) 
// Implementation adapted from: AmesianX/TurboQuant, AtomicBot-ai/atomic-llama-cpp-turboquant
//
// TurboQuant achieves 4-5x KV cache compression with near-lossless quality by:
// 1. Walsh-Hadamard Transform (WHT) rotation → Gaussian-like distribution
// 2. Lloyd-Max scalar quantization (2/3/4-bit) with analytically derived codebooks
// 3. QJL (Quantized Johnson-Lindenstrauss) 1-bit residual correction for Keys
// 4. Dynamic attention sharpening to compensate for quantization noise
//
// Compile with: hipcc -O3 -ffast-math -fPIC -shared -o libggml_turboquant.so ggml_turboquant.cpp
// ============================================================================

#ifndef GGML_TURBOQUANT_H
#define GGML_TURBOQUANT_H

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TurboQuant Type Definitions
// ============================================================================

typedef enum {
    TURBOQUANT_TYPE_TBQ2_0  = 0,  // 2-bit WHT + Lloyd-Max, head_dim=256
    TURBOQUANT_TYPE_TBQ3_0  = 1,  // 3-bit WHT + Lloyd-Max, head_dim=256
    TURBOQUANT_TYPE_TBQ3_1  = 2,  // 3-bit WHT + Lloyd-Max, head_dim=128
    TURBOQUANT_TYPE_TBQ3_2  = 3,  // 3-bit WHT + Lloyd-Max, head_dim=64
    TURBOQUANT_TYPE_TBQ3_4  = 4,  // 3-bit WHT + Lloyd-Max, head_dim=576 (MLA)
    TURBOQUANT_TYPE_TBQ4_0  = 5,  // 4-bit WHT + Lloyd-Max, head_dim=256
    TURBOQUANT_TYPE_TBQ4_1  = 6,  // 4-bit WHT + Lloyd-Max, head_dim=128
    TURBOQUANT_TYPE_TBQ4_2  = 7,  // 4-bit WHT + Lloyd-Max, head_dim=64
    TURBOQUANT_TYPE_TBQP3_0 = 8,  // 3-bit: 2-bit MSE + 1-bit QJL, head_dim=256
    TURBOQUANT_TYPE_TBQP3_1 = 9,  // 3-bit: 2-bit MSE + 1-bit QJL, head_dim=128
    TURBOQUANT_TYPE_TBQP3_3 = 10, // 3-bit: 2-bit MSE + 1-bit QJL, head_dim=64 (double WHT)
    TURBOQUANT_TYPE_TBQP4_0 = 11, // 4-bit: 3-bit MSE + 1-bit QJL, head_dim=256
    TURBOQUANT_TYPE_TBQP4_1 = 12, // 4-bit: 3-bit MSE + 1-bit QJL, head_dim=128
    TURBOQUANT_TYPE_AMX3_1  = 13, // AMX hybrid: WHT Part A + polar Part B, head_dim=128
    TURBOQUANT_TYPE_COUNT
} turboquant_type_t;

// Block size for all TurboQuant variants (elements per block)
#define TURBOQUANT_BLOCK_SIZE 256

// Maximum WHT size supported
#define TURBOQUANT_MAX_WHT_SIZE 512

// ============================================================================
// TurboQuant Block Structures
// ============================================================================

// TBQ3 block (3-bit Lloyd-Max, 256 elements)
// 256 elements * 3 bits = 768 bits = 96 bytes + 2 bytes scale = 98 bytes
// Effective: ~2.9x compression vs f16 (256*2=512 bytes)
typedef struct {
    half d;              // scale (2 bytes)
    uint8_t qs[96];      // 3-bit quantized values packed (96 bytes)
} tbq3_block_t;          // Total: 98 bytes

// TBQ4 block (4-bit Lloyd-Max, 256 elements)
// 256 elements * 4 bits = 1024 bits = 128 bytes + 2 bytes scale = 130 bytes
// Effective: ~3.9x compression vs f16
typedef struct {
    half d;              // scale (2 bytes)
    uint8_t qs[128];     // 4-bit quantized values packed (128 bytes)
} tbq4_block_t;          // Total: 130 bytes

// TBQP3 block (2-bit MSE + 1-bit QJL, 256 elements)
// 256 elements * 2 bits = 512 bits = 64 bytes (MSE)
// + 256 elements * 1 bit = 256 bits = 32 bytes (QJL signs)
// + 2 bytes scale + 2 bytes QJL scale = 100 bytes
// Effective: ~5.1x compression vs f16
typedef struct {
    half d;              // MSE scale (2 bytes)
    half d_qjl;          // QJL scale (2 bytes)
    uint8_t qs[64];      // 2-bit MSE values (64 bytes)
    uint8_t signs[32];   // 1-bit QJL signs (32 bytes)
} tbqp3_block_t;         // Total: 100 bytes

// TBQP4 block (3-bit MSE + 1-bit QJL, 256 elements)
// 256 * 3 = 768 bits = 96 bytes + 256 bits signs = 32 bytes + 4 bytes scales = 132 bytes
typedef struct {
    half d;              // MSE scale (2 bytes)
    half d_qjl;          // QJL scale (2 bytes)
    uint8_t qs[96];      // 3-bit MSE values (96 bytes)
    uint8_t signs[32];   // 1-bit QJL signs (32 bytes)
} tbqp4_block_t;         // Total: 132 bytes

// TBQ2 block (2-bit Lloyd-Max, 256 elements)
// 256 * 2 = 512 bits = 64 bytes + 2 bytes scale = 66 bytes
// Effective: ~7.8x compression vs f16
typedef struct {
    half d;              // scale (2 bytes)
    uint8_t qs[64];      // 2-bit quantized values (64 bytes)
} tbq2_block_t;          // Total: 66 bytes

// AMX3_1 hybrid block (head_dim=128 only)
// Part A: WHT for attention (50 bytes)
// Part B: polar for TriAttention scoring (58 bytes)
// Total: 108 bytes
typedef struct {
    // Part A: WHT attention path
    half d_wht;          // WHT scale (2 bytes)
    uint8_t qs_wht[48];  // 3-bit WHT values (48 bytes)
    // Part B: polar scoring path
    half d_r;            // Rayleigh scale (2 bytes)
    uint8_t qr[24];      // 3-bit r indices (24 bytes)
    uint8_t qphi[32];    // 3-bit phi indices (32 bytes)
} amx3_1_block_t;        // Total: 108 bytes

// ============================================================================
// Lloyd-Max Codebooks (Analytically Derived)
// ============================================================================
// For Gaussian-distributed WHT coefficients, Lloyd-Max optimal quantizers
// can be derived analytically. These are precomputed for efficiency.

// 2-bit Lloyd-Max levels (8 levels, symmetric)
static const float TURBOQUANT_LM2_LEVELS[4] = {
    0.0000f,   // center (unused, implicit)
    0.4528f,   // level 1
    1.0476f,   // level 2
    1.8744f    // level 3 (max)
};

// 3-bit Lloyd-Max levels (8 levels)
static const float TURBOQUANT_LM3_LEVELS[8] = {
    0.0000f,   // 0
    0.2451f,   // 1
    0.5014f,   // 2
    0.7674f,   // 3
    1.0499f,   // 4
    1.3573f,   // 5
    1.7054f,   // 6
    2.1510f    // 7 (max)
};

// 4-bit Lloyd-Max levels (16 levels)
static const float TURBOQUANT_LM4_LEVELS[16] = {
    0.0000f, 0.1226f, 0.2490f, 0.3793f, 0.5139f, 0.6531f,
    0.7975f, 0.9477f, 1.1045f, 1.2688f, 1.4418f, 1.6250f,
    1.8205f, 2.0312f, 2.2616f, 2.5200f
};

// ============================================================================
// WHT (Walsh-Hadamard Transform) Constants
// ============================================================================

// WHT butterfly patterns are generated recursively:
// H_1 = [1]
// H_2n = [H_n, H_n; H_n, -H_n]
//
// For GPU implementation, we use in-place butterfly with stride doubling.
// No multiplication needed — only additions and subtractions.

// Maximum stages for supported WHT sizes
#define WHT_MAX_STAGES 9  // log2(512)

// ============================================================================
// API Functions
// ============================================================================

// Initialize TurboQuant backend
// Must be called before any other TurboQuant function
void turboquant_init(int gpu_device);

// Get block size in bytes for a given TurboQuant type
size_t turboquant_block_size(turboquant_type_t type);

// Get compression ratio vs f16 for a given type
float turboquant_compression_ratio(turboquant_type_t type);

// Quantize K or V cache from f16 to TurboQuant format
// src: [num_blocks, block_size] half precision on device
// dst: quantized blocks on device
// type: TurboQuant variant
void turboquant_quantize(const half* src, void* dst, int num_blocks,
                         turboquant_type_t type, hipStream_t stream);

// Dequantize K cache from TurboQuant to f16 for attention
// src: quantized blocks
// dst: [num_blocks, block_size] half precision
// type: TurboQuant variant
void turboquant_dequantize_k(const void* src, half* dst, int num_blocks,
                             turboquant_type_t type, hipStream_t stream);

// Dequantize V cache from TurboQuant to f16 for attention
// V dequant is simpler (no QJL correction)
void turboquant_dequantize_v(const void* src, half* dst, int num_blocks,
                             turboquant_type_t type, hipStream_t stream);

// Fused WHT + quantize for K cache (more efficient)
// Applies WHT rotation then quantizes in one kernel
void turboquant_encode_k(const half* src, void* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream);

// Fused dequantize + IWHT for K cache
// Used in attention: reads quantized blocks, applies IWHT, outputs f16
void turboquant_decode_k(const void* src, half* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream);

// Fused dequantize + IWHT for V cache
void turboquant_decode_v(const void* src, half* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream);

// QJL correction for attention scores
// Applies 1-bit QJL correction to dot products (TBQP types only)
// scores: [num_tokens] attention scores to correct
// qjl_signs: QJL sign bits from K cache
// d_qjl: QJL scale
void turboquant_qjl_correct(float* scores, const uint8_t* qjl_signs,
                            float d_qjl, int num_tokens, hipStream_t stream);

// Attention sharpening
// Applies dynamic sharpening factor to compensate for quantization noise
// scores: [num_tokens] attention scores
// num_tokens: current sequence length
// sqnr: signal-to-quantization-noise ratio for this type
void turboquant_attention_sharpen(float* scores, int num_tokens,
                                  float sqnr, hipStream_t stream);

// WHT kernels (exposed for custom integration)
void turboquant_wht_forward(const half* src, half* dst, int n,
                            hipStream_t stream);
void turboquant_wht_inverse(const half* src, half* dst, int n,
                            hipStream_t stream);

// Auto-select best TurboQuant type for given head_dim and quality target
// quality: 0=fastest (TBQ2), 1=balanced (TBQ3), 2=best (TBQ4), 3=best+QJL (TBQP)
turboquant_type_t turboquant_auto_select(int head_dim, int quality_target);

// Get human-readable name for type
const char* turboquant_type_name(turboquant_type_t type);

// Cleanup
void turboquant_deinit();

#ifdef __cplusplus
}
#endif

#endif // GGML_TURBOQUANT_H

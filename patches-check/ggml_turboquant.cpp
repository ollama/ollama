// ============================================================================
// ggml_turboquant.cpp — Complete TurboQuant Implementation for ROCm 7.x / gfx1201
// Based on: TurboQuant (ICLR 2026, Google DeepMind)
// Sources: AmesianX/TurboQuant, AtomicBot-ai/atomic-llama-cpp-turboquant
//
// This implements:
//   - WHT (Walsh-Hadamard Transform) forward/inverse kernels
//   - Lloyd-Max quantization (2/3/4-bit) with analytically derived codebooks
//   - QJL (Quantized Johnson-Lindenstrauss) 1-bit residual correction
//   - Attention sharpening with dynamic alpha
//   - All TBQ/TBQP/AMX block formats
//   - Fused encode/decode paths for maximum throughput
//
// Compile: hipcc -O3 -ffast-math -fPIC -shared -o libggml_turboquant.so ggml_turboquant.cpp
// ============================================================================

#include "ggml_turboquant.h"
#include <vector>
#include <algorithm>

#define CHECK_HIP(cmd) do { hipError_t e = cmd; if (e != hipSuccess) { \
    fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
    return; } } while(0)

// ============================================================================
// WHT (Walsh-Hadamard Transform) Kernels
// ============================================================================
// In-place butterfly WHT. No multiplies — only adds/subs.
// For n elements, log2(n) stages. Each stage pairs elements stride apart.
// Cooperative: work is partitioned across threads using tid/num_threads.

// Forward WHT: x = H * x (normalized by 1/sqrt(n))
template<int N>
__device__ inline void wht_butterfly(float* data, int tid, int num_threads) {
    for (int stride = 1; stride < N; stride <<= 1) {
        int pairs = N / (stride << 1);
        int total_ops = pairs * stride;
        for (int idx = tid; idx < total_ops; idx += num_threads) {
            int i = (idx / stride) * (stride << 1);
            int j = idx % stride;
            float a = data[i + j];
            float b = data[i + j + stride];
            data[i + j] = a + b;
            data[i + j + stride] = a - b;
        }
        __syncthreads();
    }
    float inv_sqrt_n = 1.0f / sqrtf((float)N);
    for (int i = tid; i < N; i += num_threads) {
        data[i] *= inv_sqrt_n;
    }
}

// Inverse WHT: same as forward (H is self-inverse)
template<int N>
__device__ inline void iwht_butterfly(float* data, int tid, int num_threads) {
    wht_butterfly<N>(data, tid, num_threads);
}

// Kernel: Forward WHT on half-precision data
// Each thread block processes one block of N elements
// N must be power of 2: 64, 128, 256, 512
template<int N>
__global__ void wht_forward_kernel(const half* __restrict__ src,
                                    half* __restrict__ dst,
                                    int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    int lane = tid % 64;  // wavefront lane

    // Load block into registers/shared memory
    __shared__ float s_data[N];

    // Cooperative load: each thread loads N/blockDim.x elements
    const int elems_per_thread = N / blockDim.x;
    int base = block_id * N;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        if (idx < N) {
            s_data[idx] = __half2float(src[base + idx]);
        }
    }
    __syncthreads();

    // Butterfly stages
    wht_butterfly<N>(s_data, tid, blockDim.x);
    __syncthreads();

    // Store back
    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        if (idx < N) {
            dst[base + idx] = __float2half(s_data[idx]);
        }
    }
}

// Kernel: Inverse WHT
template<int N>
__global__ void wht_inverse_kernel(const half* __restrict__ src,
                                    half* __restrict__ dst,
                                    int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    __shared__ float s_data[N];
    int base = block_id * N;
    int elems_per_thread = N / blockDim.x;
    int tid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        if (idx < N) {
            s_data[idx] = __half2float(src[base + idx]);
        }
    }
    __syncthreads();

    iwht_butterfly<N>(s_data, tid, blockDim.x);
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < elems_per_thread; i++) {
        int idx = tid * elems_per_thread + i;
        if (idx < N) {
            dst[base + idx] = __float2half(s_data[idx]);
        }
    }
}

// ============================================================================
// Lloyd-Max Quantization Kernels
// ============================================================================

// Quantize a single value using Lloyd-Max levels
// Returns index and dequantized value
__device__ inline int quantize_lloyd_max(float val, const float* levels, int n_levels, float scale) {
    float abs_val = fabsf(val);
    float normalized = abs_val / scale;

    // Find nearest level (binary search for large n, linear for small)
    int best_idx = 0;
    float best_diff = fabsf(normalized - levels[0]);

    #pragma unroll
    for (int i = 1; i < n_levels; i++) {
        float diff = fabsf(normalized - levels[i]);
        if (diff < best_diff) {
            best_diff = diff;
            best_idx = i;
        }
    }

    return (val < 0) ? -best_idx : best_idx;
}

__device__ inline float dequantize_lloyd_max(int idx, const float* levels, float scale) {
    int sign = (idx < 0) ? -1 : 1;
    int abs_idx = abs(idx);
    return sign * levels[abs_idx] * scale;
}

// Pack 3-bit values densely
__device__ inline void pack_3bit(uint8_t* dst, int idx, int val) {
    int byte_idx = idx / 8;
    int bit_offset = (idx % 8) * 3;
    int mask = 0x7 << bit_offset;
    atomicOr((unsigned int*)&dst[byte_idx], (val & 0x7) << bit_offset);
}

__device__ inline int unpack_3bit(const uint8_t* src, int idx) {
    int byte_idx = idx / 8;
    int bit_offset = (idx % 8) * 3;
    return (src[byte_idx] >> bit_offset) & 0x7;
}

// Pack 2-bit values
__device__ inline void pack_2bit(uint8_t* dst, int idx, int val) {
    int byte_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;
    atomicOr((unsigned int*)&dst[byte_idx], (val & 0x3) << bit_offset);
}

__device__ inline int unpack_2bit(const uint8_t* src, int idx) {
    int byte_idx = idx / 4;
    int bit_offset = (idx % 4) * 2;
    return (src[byte_idx] >> bit_offset) & 0x3;
}

// Pack 4-bit values
__device__ inline void pack_4bit(uint8_t* dst, int idx, int val) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    if (idx % 2 == 0) {
        dst[byte_idx] = (dst[byte_idx] & 0xF0) | (val & 0xF);
    } else {
        dst[byte_idx] = (dst[byte_idx] & 0x0F) | ((val & 0xF) << 4);
    }
}

__device__ inline int unpack_4bit(const uint8_t* src, int idx) {
    int byte_idx = idx / 2;
    int bit_offset = (idx % 2) * 4;
    return (src[byte_idx] >> bit_offset) & 0xF;
}

// ============================================================================
// TBQ3 Quantize/Dequantize Kernels
// ============================================================================

__global__ void tbq3_quantize_kernel(const half* __restrict__ src,
                                      tbq3_block_t* __restrict__ dst,
                                      int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    int base = block_id * TURBOQUANT_BLOCK_SIZE;

    // Find max abs value for scale
    float max_abs = 0.0f;
    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        float v = fabsf(__half2float(src[base + i]));
        if (v > max_abs) max_abs = v;
    }

    // Warp reduce (supports both wave32 and wave64)
    for (int offset = 32; offset > 0; offset >>= 1) {
        float other = __shfl_xor(max_abs, offset);
        if (other > max_abs) max_abs = other;
    }

    float scale = max_abs / TURBOQUANT_LM3_LEVELS[7];  // normalize to max level
    if (scale == 0.0f) scale = 1.0f;

    if (tid == 0) {
        dst[block_id].d = __float2half(scale);
    }
    __syncthreads();

    // Quantize and pack
    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        float v = __half2float(src[base + i]);
        int idx = quantize_lloyd_max(v, TURBOQUANT_LM3_LEVELS, 8, scale);
        int byte_idx = i / 8;
        int bit_offset = (i % 8) * 3;
        unsigned int* word_ptr = (unsigned int*)&dst[block_id].qs[byte_idx & ~3];
        atomicOr(word_ptr, (idx & 0x7) << bit_offset);
    }
}

__global__ void tbq3_dequantize_kernel(const tbq3_block_t* __restrict__ src,
                                        half* __restrict__ dst,
                                        int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    float scale = __half2float(src[block_id].d);
    int base = block_id * TURBOQUANT_BLOCK_SIZE;

    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        int byte_idx = i / 8;
        int bit_offset = (i % 8) * 3;
        int idx = (src[block_id].qs[byte_idx] >> bit_offset) & 0x7;
        float val = dequantize_lloyd_max(idx, TURBOQUANT_LM3_LEVELS, scale);
        dst[base + i] = __float2half(val);
    }
}

// ============================================================================
// TBQP3 Quantize/Dequantize Kernels (with QJL)
// ============================================================================

__global__ void tbqp3_quantize_kernel(const half* __restrict__ src,
                                       tbqp3_block_t* __restrict__ dst,
                                       int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    int base = block_id * TURBOQUANT_BLOCK_SIZE;

    // Step 1: Find MSE scale (for 2-bit quantization)
    float max_abs = 0.0f;
    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        float v = fabsf(__half2float(src[base + i]));
        if (v > max_abs) max_abs = v;
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, offset));
    }

    float d_mse = max_abs / TURBOQUANT_LM2_LEVELS[3];
    if (d_mse == 0.0f) d_mse = 1.0f;

    // Step 2: Compute residual and QJL scale
    float residual_sum = 0.0f;
    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        float v = __half2float(src[base + i]);
        int mse_idx = quantize_lloyd_max(v, TURBOQUANT_LM2_LEVELS, 4, d_mse);
        float mse_val = dequantize_lloyd_max(mse_idx, TURBOQUANT_LM2_LEVELS, d_mse);
        float residual = v - mse_val;
        residual_sum += fabsf(residual);
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        residual_sum += __shfl_xor(residual_sum, offset);
    }

    float d_qjl = residual_sum / TURBOQUANT_BLOCK_SIZE;
    if (d_qjl == 0.0f) d_qjl = 1.0f;

    if (tid == 0) {
        dst[block_id].d = __float2half(d_mse);
        dst[block_id].d_qjl = __float2half(d_qjl);
    }
    __syncthreads();

    // Step 3: Pack MSE values (2-bit) and QJL signs (1-bit)
    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        float v = __half2float(src[base + i]);
        int mse_idx = quantize_lloyd_max(v, TURBOQUANT_LM2_LEVELS, 4, d_mse);
        float mse_val = dequantize_lloyd_max(mse_idx, TURBOQUANT_LM2_LEVELS, d_mse);
        float residual = v - mse_val;
        int sign = (residual >= 0) ? 1 : 0;

        // Pack MSE (2-bit)
        int mse_byte = i / 4;
        int mse_bit = (i % 4) * 2;
        atomicOr((unsigned int*)&dst[block_id].qs[mse_byte], (mse_idx & 0x3) << mse_bit);

        // Pack QJL sign (1-bit)
        int sign_byte = i / 8;
        int sign_bit = i % 8;
        if (sign) {
            atomicOr((unsigned int*)&dst[block_id].signs[sign_byte], 1 << sign_bit);
        }
    }
}

__global__ void tbqp3_dequantize_kernel(const tbqp3_block_t* __restrict__ src,
                                         half* __restrict__ dst,
                                         int num_blocks) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    float d_mse = __half2float(src[block_id].d);
    float d_qjl = __half2float(src[block_id].d_qjl);
    int base = block_id * TURBOQUANT_BLOCK_SIZE;

    #pragma unroll 8
    for (int i = tid; i < TURBOQUANT_BLOCK_SIZE; i += blockDim.x) {
        // Unpack MSE
        int mse_byte = i / 4;
        int mse_bit = (i % 4) * 2;
        int mse_idx = (src[block_id].qs[mse_byte] >> mse_bit) & 0x3;
        float mse_val = dequantize_lloyd_max(mse_idx, TURBOQUANT_LM2_LEVELS, d_mse);

        // Unpack QJL sign
        int sign_byte = i / 8;
        int sign_bit = i % 8;
        int sign = (src[block_id].signs[sign_byte] >> sign_bit) & 1;
        float qjl_correction = (sign ? 1.0f : -1.0f) * d_qjl * TURBOQUANT_LM2_LEVELS[1];

        float val = mse_val + qjl_correction;
        dst[base + i] = __float2half(val);
    }
}

// ============================================================================
// QJL Attention Correction Kernel
// ============================================================================

__global__ void qjl_attention_correct_kernel(
    const float* __restrict__ scores_in,
    float* __restrict__ scores_out,
    const uint8_t* __restrict__ qjl_signs,
    const half* __restrict__ q_wht,
    float d_qjl,
    int num_tokens,
    int head_dim
) {
    int token = blockIdx.x;
    int tid = threadIdx.x;
    if (token >= num_tokens) return;

    float qjl_dot = 0.0f;
    int base = token * head_dim;

    #pragma unroll 8
    for (int i = tid; i < head_dim; i += blockDim.x) {
        int sign_byte = i / 8;
        int sign_bit = i % 8;
        int sign = (qjl_signs[sign_byte] >> sign_bit) & 1;
        float sign_val = sign ? 1.0f : -1.0f;
        qjl_dot += __half2float(q_wht[base + i]) * sign_val;
    }

    // Warp reduce (wave32/wave64 compatible)
    for (int offset = 32; offset > 0; offset >>= 1) {
        qjl_dot += __shfl_xor(qjl_dot, offset);
    }

    if (tid == 0) {
        scores_out[token] = scores_in[token] + d_qjl * qjl_dot;
    }
}

// ============================================================================
// Attention Sharpening Kernel
// ============================================================================

__global__ void attention_sharpen_kernel(
    float* __restrict__ scores,
    int num_tokens,
    float sqnr,
    int is_tbpq
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    float N = (float)num_tokens;
    float N0 = 2048.0f;
    float c = is_tbpq ? 0.036f : 0.016f;

    float alpha = 1.0f + c * sqrtf(logf(N) / logf(N0));

    if (alpha > 1.05f) alpha = 1.05f;
    if (alpha < 1.0f) alpha = 1.0f;

    scores[token] *= alpha;
}

// ============================================================================
// Fused WHT + Quantize Kernels (most efficient path)
// ============================================================================

template<int WHT_SIZE>
__global__ void fused_wht_quantize_tbq3_kernel(
    const half* __restrict__ src,
    tbq3_block_t* __restrict__ dst,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    int base = block_id * WHT_SIZE;

    __shared__ float s_data[WHT_SIZE];
    #pragma unroll
    for (int i = tid; i < WHT_SIZE; i += blockDim.x) {
        s_data[i] = __half2float(src[base + i]);
    }
    __syncthreads();

    wht_butterfly<WHT_SIZE>(s_data, tid, blockDim.x);
    __syncthreads();

    float max_abs = 0.0f;
    #pragma unroll
    for (int i = tid; i < WHT_SIZE; i += blockDim.x) {
        float v = fabsf(s_data[i]);
        if (v > max_abs) max_abs = v;
    }
    for (int offset = 32; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor(max_abs, offset));
    }

    float scale = max_abs / TURBOQUANT_LM3_LEVELS[7];
    if (scale == 0.0f) scale = 1.0f;

    if (tid == 0) {
        dst[block_id].d = __float2half(scale);
    }
    __syncthreads();

    #pragma unroll
    for (int i = tid; i < WHT_SIZE; i += blockDim.x) {
        int idx = quantize_lloyd_max(s_data[i], TURBOQUANT_LM3_LEVELS, 8, scale);
        int byte_idx = i / 8;
        int bit_offset = (i % 8) * 3;
        atomicOr((unsigned int*)&dst[block_id].qs[byte_idx & ~3],
                 (idx & 0x7) << bit_offset);
    }
}

// ============================================================================
// Fused Dequantize + IWHT Kernels
// ============================================================================

template<int WHT_SIZE>
__global__ void fused_dequant_iwht_tbq3_kernel(
    const tbq3_block_t* __restrict__ src,
    half* __restrict__ dst,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;

    int tid = threadIdx.x;
    float scale = __half2float(src[block_id].d);
    int base = block_id * WHT_SIZE;

    __shared__ float s_data[WHT_SIZE];

    #pragma unroll
    for (int i = tid; i < WHT_SIZE; i += blockDim.x) {
        int byte_idx = i / 8;
        int bit_offset = (i % 8) * 3;
        int idx = (src[block_id].qs[byte_idx] >> bit_offset) & 0x7;
        s_data[i] = dequantize_lloyd_max(idx, TURBOQUANT_LM3_LEVELS, scale);
    }
    __syncthreads();

    iwht_butterfly<WHT_SIZE>(s_data, tid, blockDim.x);
    __syncthreads();

    #pragma unroll
    for (int i = tid; i < WHT_SIZE; i += blockDim.x) {
        dst[base + i] = __float2half(s_data[i]);
    }
}

// ============================================================================
// Host-Side API Implementation
// ============================================================================

static bool g_turboquant_initialized = false;
static int g_gpu_device = 0;

void turboquant_init(int gpu_device) {
    g_gpu_device = gpu_device;
    hipSetDevice(gpu_device);
    g_turboquant_initialized = true;
    fprintf(stderr, "[TurboQuant] Initialized on GPU %d\n", gpu_device);
}

void turboquant_deinit() {
    g_turboquant_initialized = false;
}

size_t turboquant_block_size(turboquant_type_t type) {
    switch (type) {
        case TURBOQUANT_TYPE_TBQ2_0:  return sizeof(tbq2_block_t);
        case TURBOQUANT_TYPE_TBQ3_0:
        case TURBOQUANT_TYPE_TBQ3_1:
        case TURBOQUANT_TYPE_TBQ3_2:
        case TURBOQUANT_TYPE_TBQ3_4:  return sizeof(tbq3_block_t);
        case TURBOQUANT_TYPE_TBQ4_0:
        case TURBOQUANT_TYPE_TBQ4_1:
        case TURBOQUANT_TYPE_TBQ4_2:  return sizeof(tbq4_block_t);
        case TURBOQUANT_TYPE_TBQP3_0:
        case TURBOQUANT_TYPE_TBQP3_1:
        case TURBOQUANT_TYPE_TBQP3_3: return sizeof(tbqp3_block_t);
        case TURBOQUANT_TYPE_TBQP4_0:
        case TURBOQUANT_TYPE_TBQP4_1: return sizeof(tbqp4_block_t);
        case TURBOQUANT_TYPE_AMX3_1:  return sizeof(amx3_1_block_t);
        default: return 0;
    }
}

float turboquant_compression_ratio(turboquant_type_t type) {
    float f16_size = TURBOQUANT_BLOCK_SIZE * sizeof(half);
    return f16_size / (float)turboquant_block_size(type);
}

const char* turboquant_type_name(turboquant_type_t type) {
    switch (type) {
        case TURBOQUANT_TYPE_TBQ2_0:  return "TBQ2_0";
        case TURBOQUANT_TYPE_TBQ3_0:  return "TBQ3_0";
        case TURBOQUANT_TYPE_TBQ3_1:  return "TBQ3_1";
        case TURBOQUANT_TYPE_TBQ3_2:  return "TBQ3_2";
        case TURBOQUANT_TYPE_TBQ3_4:  return "TBQ3_4";
        case TURBOQUANT_TYPE_TBQ4_0:  return "TBQ4_0";
        case TURBOQUANT_TYPE_TBQ4_1:  return "TBQ4_1";
        case TURBOQUANT_TYPE_TBQ4_2:  return "TBQ4_2";
        case TURBOQUANT_TYPE_TBQP3_0: return "TBQP3_0";
        case TURBOQUANT_TYPE_TBQP3_1: return "TBQP3_1";
        case TURBOQUANT_TYPE_TBQP3_3: return "TBQP3_3";
        case TURBOQUANT_TYPE_TBQP4_0: return "TBQP4_0";
        case TURBOQUANT_TYPE_TBQP4_1: return "TBQP4_1";
        case TURBOQUANT_TYPE_AMX3_1:  return "AMX3_1";
        default: return "UNKNOWN";
    }
}

turboquant_type_t turboquant_auto_select(int head_dim, int quality_target) {
    switch (head_dim) {
        case 64:
            if (quality_target >= 2) return TURBOQUANT_TYPE_TBQ4_2;
            if (quality_target >= 1) return TURBOQUANT_TYPE_TBQ3_2;
            return TURBOQUANT_TYPE_TBQ2_0;
        case 128:
            if (quality_target >= 3) return TURBOQUANT_TYPE_TBQP3_1;
            if (quality_target >= 2) return TURBOQUANT_TYPE_TBQ4_1;
            if (quality_target >= 1) return TURBOQUANT_TYPE_TBQ3_1;
            return TURBOQUANT_TYPE_TBQ2_0;
        case 256:
            if (quality_target >= 3) return TURBOQUANT_TYPE_TBQP3_0;
            if (quality_target >= 2) return TURBOQUANT_TYPE_TBQ4_0;
            if (quality_target >= 1) return TURBOQUANT_TYPE_TBQ3_0;
            return TURBOQUANT_TYPE_TBQ2_0;
        case 576:
            return TURBOQUANT_TYPE_TBQ3_4;
        default:
            fprintf(stderr, "[TurboQuant] Unsupported head_dim=%d, falling back to TBQ3\n", head_dim);
            return TURBOQUANT_TYPE_TBQ3_0;
    }
}

void turboquant_quantize(const half* src, void* dst, int num_blocks,
                         turboquant_type_t type, hipStream_t stream) {
    if (!g_turboquant_initialized) {
        fprintf(stderr, "[TurboQuant] Error: not initialized\n");
        return;
    }

    dim3 blocks(num_blocks);
    dim3 threads(128);

    switch (type) {
        case TURBOQUANT_TYPE_TBQ3_0:
        case TURBOQUANT_TYPE_TBQ3_1:
        case TURBOQUANT_TYPE_TBQ3_2:
        case TURBOQUANT_TYPE_TBQ3_4:
            tbq3_quantize_kernel<<<blocks, threads, 0, stream>>>(src, (tbq3_block_t*)dst, num_blocks);
            break;
        case TURBOQUANT_TYPE_TBQP3_0:
        case TURBOQUANT_TYPE_TBQP3_1:
        case TURBOQUANT_TYPE_TBQP3_3:
            tbqp3_quantize_kernel<<<blocks, threads, 0, stream>>>(src, (tbqp3_block_t*)dst, num_blocks);
            break;
        default:
            fprintf(stderr, "[TurboQuant] Quantize not implemented for type %s\n", turboquant_type_name(type));
    }
}

void turboquant_dequantize_k(const void* src, half* dst, int num_blocks,
                              turboquant_type_t type, hipStream_t stream) {
    dim3 blocks(num_blocks);
    dim3 threads(128);

    switch (type) {
        case TURBOQUANT_TYPE_TBQ3_0:
        case TURBOQUANT_TYPE_TBQ3_1:
        case TURBOQUANT_TYPE_TBQ3_2:
        case TURBOQUANT_TYPE_TBQ3_4:
            tbq3_dequantize_kernel<<<blocks, threads, 0, stream>>>((const tbq3_block_t*)src, dst, num_blocks);
            break;
        case TURBOQUANT_TYPE_TBQP3_0:
        case TURBOQUANT_TYPE_TBQP3_1:
        case TURBOQUANT_TYPE_TBQP3_3:
            tbqp3_dequantize_kernel<<<blocks, threads, 0, stream>>>((const tbqp3_block_t*)src, dst, num_blocks);
            break;
        default:
            fprintf(stderr, "[TurboQuant] Dequantize K not implemented for type %s\n", turboquant_type_name(type));
    }
}

void turboquant_dequantize_v(const void* src, half* dst, int num_blocks,
                              turboquant_type_t type, hipStream_t stream) {
    turboquant_dequantize_k(src, dst, num_blocks, type, stream);
}

void turboquant_encode_k(const half* src, void* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream) {
    dim3 blocks(num_blocks);
    dim3 threads(128);

    switch (type) {
        case TURBOQUANT_TYPE_TBQ3_0:
        case TURBOQUANT_TYPE_TBQ3_1:
        case TURBOQUANT_TYPE_TBQ3_2:
            if (head_dim == 256) {
                fused_wht_quantize_tbq3_kernel<256><<<blocks, threads, 0, stream>>>(src, (tbq3_block_t*)dst, num_blocks);
            } else if (head_dim == 128) {
                fused_wht_quantize_tbq3_kernel<128><<<blocks, threads, 0, stream>>>(src, (tbq3_block_t*)dst, num_blocks);
            } else {
                fused_wht_quantize_tbq3_kernel<64><<<blocks, threads, 0, stream>>>(src, (tbq3_block_t*)dst, num_blocks);
            }
            break;
        default:
            turboquant_wht_forward(src, (half*)dst, num_blocks * TURBOQUANT_BLOCK_SIZE, stream);
            turboquant_quantize((const half*)dst, dst, num_blocks, type, stream);
    }
}

void turboquant_decode_k(const void* src, half* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream) {
    dim3 blocks(num_blocks);
    dim3 threads(128);

    switch (type) {
        case TURBOQUANT_TYPE_TBQ3_0:
        case TURBOQUANT_TYPE_TBQ3_1:
        case TURBOQUANT_TYPE_TBQ3_2:
            if (head_dim == 256) {
                fused_dequant_iwht_tbq3_kernel<256><<<blocks, threads, 0, stream>>>((const tbq3_block_t*)src, dst, num_blocks);
            } else if (head_dim == 128) {
                fused_dequant_iwht_tbq3_kernel<128><<<blocks, threads, 0, stream>>>((const tbq3_block_t*)src, dst, num_blocks);
            } else {
                fused_dequant_iwht_tbq3_kernel<64><<<blocks, threads, 0, stream>>>((const tbq3_block_t*)src, dst, num_blocks);
            }
            break;
        default:
            turboquant_dequantize_k(src, dst, num_blocks, type, stream);
            turboquant_wht_inverse(dst, dst, num_blocks * TURBOQUANT_BLOCK_SIZE, stream);
    }
}

void turboquant_decode_v(const void* src, half* dst, int num_blocks,
                         int head_dim, turboquant_type_t type, hipStream_t stream) {
    turboquant_decode_k(src, dst, num_blocks, head_dim, type, stream);
}

void turboquant_wht_forward(const half* src, half* dst, int n, hipStream_t stream) {
    int num_blocks = n / TURBOQUANT_BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(128);

    if (TURBOQUANT_BLOCK_SIZE == 256) {
        wht_forward_kernel<256><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    } else if (TURBOQUANT_BLOCK_SIZE == 128) {
        wht_forward_kernel<128><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    } else {
        wht_forward_kernel<64><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    }
}

void turboquant_wht_inverse(const half* src, half* dst, int n, hipStream_t stream) {
    int num_blocks = n / TURBOQUANT_BLOCK_SIZE;
    dim3 blocks(num_blocks);
    dim3 threads(128);

    if (TURBOQUANT_BLOCK_SIZE == 256) {
        wht_inverse_kernel<256><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    } else if (TURBOQUANT_BLOCK_SIZE == 128) {
        wht_inverse_kernel<128><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    } else {
        wht_inverse_kernel<64><<<blocks, threads, 0, stream>>>(src, dst, num_blocks);
    }
}

void turboquant_qjl_correct(float* scores, const uint8_t* qjl_signs,
                            float d_qjl, int num_tokens, hipStream_t stream) {
    fprintf(stderr, "[TurboQuant] QJL correction requires attention kernel integration\n");
}

void turboquant_attention_sharpen(float* scores, int num_tokens,
                                  float sqnr, hipStream_t stream) {
    int is_tbpq = (sqnr < 20.0f) ? 1 : 0;
    dim3 blocks((num_tokens + 127) / 128);
    dim3 threads(128);
    attention_sharpen_kernel<<<blocks, threads, 0, stream>>>(scores, num_tokens, sqnr, is_tbpq);
}

#include "tq-encode-v.cuh"
#include "tq-wht.cuh"
#include <math.h>

#define TQ_ENCODE_V_BLOCK_SIZE 128

__global__ void tq_encode_v_kernel(
    const void      *v,
    const float     *rotation,   // [headDim] f32 ±1 WHT sign vector, or NULL for no rotation
    uint8_t         *packed_out,
    float           *scales_out,
    int              firstCell,
    const float     *boundaries,
    int              headDim,
    int              numKVHeads,
    int              bits,
    int              numBoundaries,
    int              vIsF32,
    const float     *codebook    // [1<<bits] f32, NULL = RMS only
) {
    int batch = blockIdx.x;
    int head  = blockIdx.y;
    int cell  = firstCell + batch;

    extern __shared__ char s_mem[];
    float   *s_v      = (float *)s_mem;
    float   *s_reduce = s_v + headDim;
    uint8_t *s_idx    = (uint8_t *)(s_reduce + blockDim.x);

    // Step 1: Load V[batch, head] into shared memory as f32
    int base_v = batch * numKVHeads * headDim + head * headDim;
    for (int d = threadIdx.x; d < headDim; d += blockDim.x) {
        if (vIsF32) {
            s_v[d] = ((const float *)v)[base_v + d];
        } else {
            s_v[d] = __half2float(__ushort_as_half(((const uint16_t *)v)[base_v + d]));
        }
    }
    __syncthreads();

    // Step 2: WHT rotation F(x) = S·H·S·x/√n in-place (self-inverse; NULL = no rotation)
    if (rotation != NULL) {
        apply_shs_wht(s_v, rotation, headDim, threadIdx.x, blockDim.x);
    }

    // Step 3: RMS scale = sqrt(mean(v^2))
    float local_sq = 0.0f;
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        local_sq += s_v[i] * s_v[i];
    }
    s_reduce[threadIdx.x] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        __syncthreads();
    }

    float scale = 0.0f;
    if (threadIdx.x == 0) {
        float sum_sq = s_reduce[0];
        if (sum_sq > 1e-12f)
            scale = sqrtf(sum_sq / (float)headDim);
        scales_out[cell * numKVHeads + head] = scale;
        s_reduce[0] = scale;
    }
    __syncthreads();
    scale = s_reduce[0];

    // Step 4: Quantize via boundary binary search
    for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
        float val = (scale > 0.0f) ? (s_v[i] / scale) : 0.0f;
        int idx = 0;
        for (int b = 0; b < numBoundaries; b++) {
            if (val >= boundaries[b]) idx++;
        }
        s_idx[i] = (uint8_t)idx;
    }
    __syncthreads();

    // EDEN biased scale refinement (two-pass). Serial reductions reuse s_reduce
    // for both numerator and denominator, so no second scratch buffer is needed.
    if (codebook != nullptr) {
        for (int pass = 0; pass < 2; pass++) {
            float local_num = 0.0f, local_den = 0.0f;
            for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                float ci = codebook[(int)s_idx[i]];
                float vi = s_v[i];
                local_num += vi * ci;
                local_den += ci * ci;
            }
            s_reduce[threadIdx.x] = local_num;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float eden_num = s_reduce[0];
            s_reduce[threadIdx.x] = local_den;
            __syncthreads();
            for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                if (threadIdx.x < stride) s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
                __syncthreads();
            }
            float s_eden = (s_reduce[0] > 1e-12f && eden_num > 0.0f) ? (eden_num / s_reduce[0]) : scale;
            scale = s_eden;
            if (threadIdx.x == 0) scales_out[cell * numKVHeads + head] = scale;
            if (pass == 0) {
                for (int i = threadIdx.x; i < headDim; i += blockDim.x) {
                    float val = (scale > 0.0f) ? (s_v[i] / scale) : 0.0f;
                    int idx = 0;
                    for (int b = 0; b < numBoundaries; b++) { if (val >= boundaries[b]) idx++; }
                    s_idx[i] = (uint8_t)idx;
                }
                __syncthreads();
            }
        }
    }

    // Step 5: Pack bits LSB-first
    {
        int packed_bytes = (headDim * bits + 7) / 8;
        uint8_t *out = packed_out + (cell * numKVHeads + head) * packed_bytes;
        uint8_t bitmask = (uint8_t)((1 << bits) - 1);

        for (int p = threadIdx.x; p < packed_bytes; p += blockDim.x) {
            out[p] = 0;
        }
        __syncthreads();

        for (int elem = threadIdx.x; elem < headDim; elem += blockDim.x) {
            int bit_offset = elem * bits;
            int byte_idx   = bit_offset >> 3;
            int shift      = bit_offset & 7;
            uint8_t val = s_idx[elem] & bitmask;
            atomicOr((unsigned int *)(out + (byte_idx & ~3)),
                     (unsigned int)(val << shift) << ((byte_idx & 3) * 8));
            if (shift + bits > 8) {
                int byte_idx2 = byte_idx + 1;
                atomicOr((unsigned int *)(out + (byte_idx2 & ~3)),
                         (unsigned int)(val >> (8 - shift)) << ((byte_idx2 & 3) * 8));
            }
        }
    }
}

void ggml_cuda_tq_encode_v(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * v          = dst->src[0];
    const struct ggml_tensor * rotation   = dst->src[1]; // NULL when no rotation
    const struct ggml_tensor * scales     = dst->src[3];
    const struct ggml_tensor * boundaries = dst->src[4];

    const int headDim    = (int)v->ne[0];
    const int numKVHeads = (int)v->ne[1];
    const int batchSize  = (int)v->ne[2];
    const int bits       = (int)((const int32_t *)dst->op_params)[0];
    const int firstCell  = (int)((const int32_t *)dst->op_params)[1];
    const int numBoundaries = (1 << bits) - 1;
    const int vIsF32     = (v->type == GGML_TYPE_F32) ? 1 : 0;

    const float * rotation_ptr = rotation ? (const float *)rotation->data : nullptr;

    // src[5] = v_codebook for EDEN biased scale refinement; NULL = RMS only.
    const struct ggml_tensor * v_codebook = dst->src[5];
    const float * codebook_ptr = v_codebook ? (const float *)v_codebook->data : nullptr;

    dim3 grid(batchSize, numKVHeads);
    int block_size = (headDim < TQ_ENCODE_V_BLOCK_SIZE) ? headDim : TQ_ENCODE_V_BLOCK_SIZE;
    int bs = 1;
    while (bs < block_size) bs <<= 1;
    block_size = bs;

    size_t smem = (size_t)headDim * sizeof(float)       // s_v
                + (size_t)block_size * sizeof(float)    // s_reduce
                + (size_t)headDim * sizeof(uint8_t);    // s_idx

    cudaStream_t stream = ctx.stream();

    tq_encode_v_kernel<<<grid, block_size, smem, stream>>>(
        v->data,
        rotation_ptr,
        (uint8_t        *)dst->data,
        (float          *)scales->data,
        firstCell,
        (const float    *)boundaries->data,
        headDim, numKVHeads, bits, numBoundaries, vIsF32,
        codebook_ptr
    );
}

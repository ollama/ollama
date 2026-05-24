#include "pagedattn.cuh"
#include "common.cuh"

#include <cfloat>

static constexpr int PAGED_ATTN_THREADS = 128;

template <int HEAD_DIM, int BLOCK_SIZE>
static __global__ void paged_attention_kernel_f32(
        const float * __restrict__ Q,
        const float * __restrict__ K_cache,
        const float * __restrict__ V_cache,
        const float * __restrict__ mask,
        const int32_t * __restrict__ block_tables,
        const int32_t * __restrict__ seq_lengths,
        float * __restrict__ dst,
        const int32_t nbq2,  const int32_t nbq3,
        const int32_t nbk1,  const int32_t nbk2,  const int32_t nbk3,
        const int32_t nbv1,  const int32_t nbv2,  const int32_t nbv3,
        const int32_t max_blocks_per_seq,
        const int32_t gqa_ratio,
        const float   scale) {

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int kv_head_idx = head_idx / gqa_ratio;

    const int seq_len = seq_lengths[seq_idx];
    if (seq_len <= 0) {
        return;
    }

    // Load query vector into shared memory
    __shared__ float q_vec[HEAD_DIM];
    const float * q_src = (const float *)((const char *)Q + head_idx * nbq2 + seq_idx * nbq3);

    for (int i = threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
        q_vec[i] = q_src[i];
    }
    __syncthreads();

    // Online softmax state and output accumulator (per thread)
    constexpr int N_ELEMS = (HEAD_DIM + PAGED_ATTN_THREADS - 1) / PAGED_ATTN_THREADS;
    int elem_start = threadIdx.x * N_ELEMS;
    int elem_end   = min(elem_start + N_ELEMS, HEAD_DIM);

    float out_acc[N_ELEMS];
    float max_logit = -FLT_MAX;
    float sum_exp   = 0.0f;

    #pragma unroll
    for (int i = 0; i < N_ELEMS; i++) {
        out_acc[i] = 0.0f;
    }

    // Process KV blocks
    const int num_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for block-level scores
    __shared__ float block_scores[BLOCK_SIZE];
    __shared__ float block_max;
    __shared__ float block_sum_exp;

    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        const int32_t physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        const int block_start = block_idx * BLOCK_SIZE;
        const int block_end   = min(block_start + BLOCK_SIZE, seq_len);
        const int block_len   = block_end - block_start;

        // Compute attention scores: Q @ K
        for (int pos = threadIdx.x; pos < block_len; pos += blockDim.x) {
            float score = 0.0f;
            const char * k_ptr = (const char *)K_cache +
                pos * nbk2 + kv_head_idx * nbk1 + physical_block * nbk3;
            const float * k_data = (const float *)k_ptr;

            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_vec[d] * k_data[d];
            }

            block_scores[pos] = score * scale;

            if (mask) {
                const float * mask_data = (const float *)((const char *)mask + seq_idx * nbq2);
                if (mask_data) {
                    block_scores[pos] += mask_data[block_start + pos];
                }
            }
        }
        __syncthreads();

        // Find block max (reduction by thread 0)
        if (threadIdx.x == 0) {
            float local_max = -FLT_MAX;
            #pragma unroll
            for (int i = 0; i < block_len; i++) {
                if (block_scores[i] > local_max) {
                    local_max = block_scores[i];
                }
            }
            block_max = local_max;
        }
        __syncthreads();

        float m_new = block_max;
        if (block_idx == 0) {
            max_logit = m_new;
        } else {
            m_new = fmaxf(max_logit, block_max);
        }

        // Compute sum of exp(scores - m_new)
        if (threadIdx.x == 0) {
            float local_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < block_len; i++) {
                local_sum += expf(block_scores[i] - m_new);
            }
            block_sum_exp = local_sum;
        }
        __syncthreads();

        float correction = expf(max_logit - m_new);
        float new_sum_exp = sum_exp * correction + block_sum_exp;

        // Rescale output accumulator
        #pragma unroll
        for (int i = 0; i < N_ELEMS; i++) {
            out_acc[i] *= correction;
        }

        // Accumulate weighted values
        for (int pos = threadIdx.x; pos < block_len; pos += blockDim.x) {
            float weight = expf(block_scores[pos] - m_new);

            const char * v_ptr = (const char *)V_cache +
                pos * nbv2 + kv_head_idx * nbv1 + physical_block * nbv3;
            const float * v_data = (const float *)v_ptr;

            for (int i = 0; i < N_ELEMS; i++) {
                int d = elem_start + i;
                if (d < HEAD_DIM) {
                    out_acc[i] += weight * v_data[d];
                }
            }
        }
        __syncthreads();

        max_logit = m_new;
        sum_exp = new_sum_exp;
    }

    // Normalize output
    float inv_sum = 1.0f / sum_exp;

    float * dst_row = (float *)((char *)dst + head_idx * nbq2 + seq_idx * nbq3);
    for (int i = 0; i < N_ELEMS; i++) {
        int d = elem_start + i;
        if (d < HEAD_DIM) {
            dst_row[d] = out_acc[i] * inv_sum;
        }
    }
}

// Host-side dispatch function
static void ggml_cuda_paged_attention_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * q            = dst->src[0];
    ggml_tensor * k_cache      = dst->src[1];
    ggml_tensor * v_cache      = dst->src[2];
    ggml_tensor * mask         = dst->src[3];
    ggml_tensor * block_tables = dst->src[4];
    ggml_tensor * seq_lengths  = dst->src[5];

    const float scale      = *(const float *)dst->op_params;
    const int   block_size = *(const int *)(dst->op_params + sizeof(float));

    const int head_dim       = q->ne[0];
    const int num_q_heads    = q->ne[2];
    const int num_kv_heads   = k_cache->ne[1];
    const int batch_size     = q->ne[3];
    const int max_blocks_seq = block_tables->ne[0];

    const int gqa_ratio = num_q_heads / num_kv_heads;

    // Strides in bytes
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbk1 = k_cache->nb[1];
    const int nbk2 = k_cache->nb[2];
    const int nbk3 = k_cache->nb[3];

    const int nbv1 = v_cache->nb[1];
    const int nbv2 = v_cache->nb[2];
    const int nbv3 = v_cache->nb[3];

    // Launch one block per (head, batch)
    dim3 grid(num_q_heads, batch_size, 1);
    dim3 block(PAGED_ATTN_THREADS, 1, 1);

    const float * q_data    = (const float *)q->data;
    const float * k_data    = (const float *)k_cache->data;
    const float * v_data    = (const float *)v_cache->data;
    const float * mask_data = mask ? (const float *)mask->data : nullptr;
    const int32_t * bt_data = (const int32_t *)block_tables->data;
    const int32_t * sl_data = (const int32_t *)seq_lengths->data;
    float * dst_data        = (float *)dst->data;

    switch (head_dim) {
#define CASE_HEAD_DIM(D)                                                           \
        case D:                                                                    \
            switch (block_size) {                                                  \
                case 8:                                                            \
                    paged_attention_kernel_f32<D, 8><<<grid, block, 0, ctx.stream()>>>( \
                        q_data, k_data, v_data, mask_data, bt_data, sl_data,       \
                        dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, \
                        max_blocks_seq, gqa_ratio, scale);                         \
                    break;                                                         \
                case 16:                                                           \
                    paged_attention_kernel_f32<D, 16><<<grid, block, 0, ctx.stream()>>>( \
                        q_data, k_data, v_data, mask_data, bt_data, sl_data,       \
                        dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, \
                        max_blocks_seq, gqa_ratio, scale);                         \
                    break;                                                         \
                case 32:                                                           \
                    paged_attention_kernel_f32<D, 32><<<grid, block, 0, ctx.stream()>>>( \
                        q_data, k_data, v_data, mask_data, bt_data, sl_data,       \
                        dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, \
                        max_blocks_seq, gqa_ratio, scale);                         \
                    break;                                                         \
                default:                                                           \
                    paged_attention_kernel_f32<D, 16><<<grid, block, 0, ctx.stream()>>>( \
                        q_data, k_data, v_data, mask_data, bt_data, sl_data,       \
                        dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3, \
                        max_blocks_seq, gqa_ratio, scale);                         \
                    break;                                                         \
            }                                                                      \
            break;

        CASE_HEAD_DIM( 64)
        CASE_HEAD_DIM( 80)
        CASE_HEAD_DIM( 96)
        CASE_HEAD_DIM(112)
        CASE_HEAD_DIM(128)
        CASE_HEAD_DIM(256)
        CASE_HEAD_DIM(512)
        CASE_HEAD_DIM(576)

#undef CASE_HEAD_DIM

        default:
            // Fallback: round up to nearest supported
            if (head_dim <= 80) {
                paged_attention_kernel_f32<80, 16><<<grid, block, 0, ctx.stream()>>>(
                    q_data, k_data, v_data, mask_data, bt_data, sl_data,
                    dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3,
                    max_blocks_seq, gqa_ratio, scale);
            } else if (head_dim <= 128) {
                paged_attention_kernel_f32<128, 16><<<grid, block, 0, ctx.stream()>>>(
                    q_data, k_data, v_data, mask_data, bt_data, sl_data,
                    dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3,
                    max_blocks_seq, gqa_ratio, scale);
            } else if (head_dim <= 256) {
                paged_attention_kernel_f32<256, 16><<<grid, block, 0, ctx.stream()>>>(
                    q_data, k_data, v_data, mask_data, bt_data, sl_data,
                    dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3,
                    max_blocks_seq, gqa_ratio, scale);
            } else {
                paged_attention_kernel_f32<512, 16><<<grid, block, 0, ctx.stream()>>>(
                    q_data, k_data, v_data, mask_data, bt_data, sl_data,
                    dst_data, nbq2, nbq3, nbk1, nbk2, nbk3, nbv1, nbv2, nbv3,
                    max_blocks_seq, gqa_ratio, scale);
            }
            break;
    }

    GGML_UNUSED(ctx); // ctx used via ctx.stream()
}

bool ggml_cuda_paged_attention_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];

    if (q->type != GGML_TYPE_F32) return false;
    if (k->type != GGML_TYPE_F32) return false;
    if (v->type != GGML_TYPE_F32) return false;

    return true;
}

void ggml_cuda_op_paged_attention(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_set_device(ctx.device);
    ggml_cuda_paged_attention_f32(ctx, dst);
}

#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"

#include <initializer_list>

/*
    This kernel does the following:
    1. softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool with_norm>
__launch_bounds__(4 * WARP_SIZE, 1) __global__ void topk_moe_cuda(const float * logits,
                                                                  float *       weights,
                                                                  int32_t *     ids,
                                                                  const int     n_rows,
                                                                  const int     n_expert_used) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float logits_r[experts_per_thread];

#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert        = i + threadIdx.x;
        logits_r[i / WARP_SIZE] = n_experts % WARP_SIZE == 0 || expert < n_experts ? logits[expert] : -INFINITY;
    }

    float max_val = logits_r[0];

#pragma unroll
    for (int i = 1; i < experts_per_thread; i++) {
        const float val = logits_r[i];
        max_val         = max(val, max_val);
    }

    max_val = warp_reduce_max(max_val);

    float wt[experts_per_thread];
    float tmp = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const float val = logits_r[i];
        wt[i]           = expf(val - max_val);
        tmp += wt[i];
    }

    tmp = warp_reduce_sum(tmp);

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = wt[i] * inv_sum;
    }

    //at this point, each thread holds a portion of softmax,
    //we do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    extern __shared__ float data_topk_shared[];
    float *                 wt_shared_ptr = data_topk_shared + threadIdx.y * n_expert_used;

    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

#pragma unroll
        for (int i = 1; i < experts_per_thread; i++) {
            const int expert = threadIdx.x + i * WARP_SIZE;
            if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                max_val    = wt[i];
                max_expert = expert;
            }
        }

#pragma unroll
        for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
            const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
            const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
            if (val > max_val || (val == max_val && expert < max_expert)) {
                max_val    = val;
                max_expert = expert;
            }
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            wt[max_expert / WARP_SIZE] = -INFINITY;

            wt_shared_ptr[k] = max_val;
            ids[k]           = max_expert;
            if constexpr (with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if constexpr (with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = threadIdx.x; i < n_expert_used; i += WARP_SIZE) {
            wt_shared_ptr[i] = wt_shared_ptr[i] * inv_sum;
        }
    }

    for (int i = threadIdx.x; i < n_expert_used; i += WARP_SIZE) {
        weights[i] = wt_shared_ptr[i];
    }
}

template <bool with_norm>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used) {
    const int    rows_per_block = 4;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();

    const int nbytes_shared = n_expert_used * rows_per_block * sizeof(float);

    switch (n_expert) {
        case 1:
            topk_moe_cuda<1, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 2:
            topk_moe_cuda<2, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 4:
            topk_moe_cuda<4, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 8:
            topk_moe_cuda<8, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 16:
            topk_moe_cuda<16, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 32:
            topk_moe_cuda<32, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 64:
            topk_moe_cuda<64, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 128:
            topk_moe_cuda<128, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 256:
            topk_moe_cuda<256, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        case 512:
            topk_moe_cuda<512, with_norm>
                <<<grid_dims, block_dims, nbytes_shared, stream>>>(logits, weights, ids, n_rows, n_expert_used);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
            break;
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context & ctx,
                           const ggml_tensor *         logits,
                           ggml_tensor *               weights,
                           ggml_tensor *               ids,
                           const bool                  with_norm) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->src[0]->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    if (with_norm) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, n_rows, n_experts, n_expert_used);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * softmax, const ggml_tensor * weights) {
    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(weights)) {
        return false;
    }

    if (scale != 1.0f || max_bias != 0.0f) {
        return false;
    }

    // don't fuse when masks or sinks are present
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    const int n_expert = softmax->ne[0];
    // n_expert must be a power of 2
    if ((n_expert & (n_expert - 1)) != 0 || n_expert > 512) {
        return false;
    }

    return true;
}

std::initializer_list<enum ggml_op> ggml_cuda_topk_moe_ops(bool norm) {
    static std::initializer_list<enum ggml_op> norm_ops = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE,  GGML_OP_ARGSORT,
                                                            GGML_OP_VIEW,     GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                                            GGML_OP_SUM_ROWS, GGML_OP_DIV,      GGML_OP_RESHAPE };

    static std::initializer_list<enum ggml_op> no_norm_ops = { GGML_OP_SOFT_MAX, GGML_OP_RESHAPE, GGML_OP_ARGSORT,
                                                               GGML_OP_VIEW, GGML_OP_GET_ROWS };

    if (norm) {
        return norm_ops;
    }
    return no_norm_ops;
}

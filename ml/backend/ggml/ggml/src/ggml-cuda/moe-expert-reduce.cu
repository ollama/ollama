#include "moe-expert-reduce.cuh"

// This kernel is a fusion of the expert weight reduce, common in MoE models

template <int n_expert_used_template>
__global__ void moe_expert_reduce_cuda(const float * __restrict__ experts,
                                       const float * __restrict__ weights,
                                       float * __restrict__ dst,
                                       const int n_expert_used,
                                       const int n_cols) {
    const int row = blockIdx.x;
    const int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= n_cols) {
        return;
    }

    experts += row * n_cols * n_expert_used;
    weights += row * n_expert_used;
    dst += row * n_cols;

    float acc = 0.f;
    if constexpr (n_expert_used_template == 0) {
        for (int expert = 0; expert < n_expert_used; ++expert) {
            ggml_cuda_mad(acc, experts[col], weights[expert]);
            experts += n_cols;
        }
        dst[col] = acc;
    } else {
#pragma unroll
        for (int i = 0; i < n_expert_used_template; ++i) {
            ggml_cuda_mad(acc, experts[col], weights[i]);
            experts += n_cols;
        }
        dst[col] = acc;
    }
}

static void launch_moe_expert_reduce(ggml_backend_cuda_context & ctx,
                                     const float *               experts,
                                     const float *               weights,
                                     float *                     dst,
                                     const int                   n_expert_used,
                                     const int                   n_cols,
                                     const int                   n_rows) {
    const int block_size = 32;

    const int n_blocks_x = n_rows;
    const int n_blocks_y = (n_cols + block_size - 1) / block_size;

    dim3 block_dims(block_size);
    dim3 grid_dims(n_blocks_x, n_blocks_y);

    cudaStream_t stream = ctx.stream();
    switch (n_expert_used) {
        case 1:
            moe_expert_reduce_cuda<1>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 2:
            moe_expert_reduce_cuda<2>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 4:
            moe_expert_reduce_cuda<4>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 6:
            moe_expert_reduce_cuda<6>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 8:
            moe_expert_reduce_cuda<8>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 16:
            moe_expert_reduce_cuda<16>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 32:
            moe_expert_reduce_cuda<32>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 64:
            moe_expert_reduce_cuda<64>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        case 128:
            moe_expert_reduce_cuda<128>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
        default:
            moe_expert_reduce_cuda<0>
                <<<grid_dims, block_dims, 0, stream>>>(experts, weights, dst, n_expert_used, n_cols);
            break;
    }
}

bool ggml_cuda_should_use_moe_expert_reduce(const ggml_cgraph * cgraph, int start_index, int end_index) {
    const ggml_tensor * mul = cgraph->nodes[start_index];

    if (mul->op != GGML_OP_MUL || !ggml_is_contiguous(mul->src[0]) || !ggml_is_contiguous(mul->src[1])) {
        return false;
    }

    int    current_node   = start_index + 1;
    size_t current_offset = 0;

    std::vector<const ggml_tensor *> view_nodes;
    //check if all are views of the expert in increasing order
    while (current_node < end_index && cgraph->nodes[current_node]->op == GGML_OP_VIEW) {
        const ggml_tensor * node = cgraph->nodes[current_node];
        if (node->view_src != mul) {
            return false;
        }
        if (node->view_offs < current_offset) {
            return false;
        }
        current_offset = node->view_offs;
        current_node++;
        view_nodes.push_back(node);
    }

    //check if all the adds are in increasing order
    const ggml_tensor * prev_add_src = view_nodes.empty() ? nullptr : view_nodes[0];
    int                 num_adds     = 0;
    int                 num_views    = view_nodes.size();
    while (current_node < end_index && cgraph->nodes[current_node]->op == GGML_OP_ADD) {
        const ggml_tensor * add_node = cgraph->nodes[current_node];

        bool is_first_op_ok  = num_views > num_adds ? add_node->src[0] == prev_add_src : false;
        bool is_second_op_ok = num_views > num_adds ? add_node->src[1] == view_nodes[num_adds + 1] : false;

        if (!is_first_op_ok || !is_second_op_ok) {
            return false;
        }
        prev_add_src = add_node;

        num_adds++;
        current_node++;
    }

    if (num_views != num_adds + 1) {
        return false;
    }

    return true;
}

void ggml_cuda_op_moe_expert_reduce(ggml_backend_cuda_context & ctx,
                                    const ggml_tensor *         experts,
                                    const ggml_tensor *         weights,
                                    ggml_tensor *               dst) {
    const int n_rows        = experts->ne[2];
    const int n_expert_used = experts->ne[1];
    const int n_cols        = experts->ne[0];

    GGML_ASSERT(experts->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(experts));
    GGML_ASSERT(ggml_is_contiguous(weights));
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const float * experts_d = (const float *) experts->data;
    const float * weights_d = (const float *) weights->data;
    float *       dst_d     = (float *) dst->data;

    launch_moe_expert_reduce(ctx, experts_d, weights_d, dst_d, n_expert_used, n_cols, n_rows);
}

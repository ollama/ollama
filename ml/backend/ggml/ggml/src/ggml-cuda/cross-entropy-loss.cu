#include "common.cuh"
#include "cross-entropy-loss.cuh"
#include "sum.cuh"

#include <cmath>
#include <cstdint>

static __global__ void cross_entropy_loss_f32(const float * logits, const float * labels, float * dst, const int nclasses, const int k) {
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int i0 = blockDim.x*blockIdx.x + warp_id*WARP_SIZE;

    const int ne_tmp = WARP_SIZE*nclasses;

    extern __shared__ float tmp_all[];
    float * tmp_logits = tmp_all + (2*warp_id + 0)*ne_tmp;
    float * tmp_labels = tmp_all + (2*warp_id + 1)*ne_tmp;

    // Each warp first loads ne_tmp logits/labels into shared memory:
    for (int i = lane_id; i < ne_tmp; i += WARP_SIZE) {
        const int ig = i0*nclasses + i; // ig == i global

        tmp_logits[i] = ig < k*nclasses ? logits[ig] : 0.0f;
        tmp_labels[i] = ig < k*nclasses ? labels[ig] : 0.0f;
    }

    // Each thread in the warp then calculates the cross entropy loss for a single row.
    // TODO: pad in order to avoid shared memory bank conflicts.

    // Find maximum for softmax:
    float max = -INFINITY;
    for (int i = 0; i < nclasses; ++i) {
        max = fmaxf(max, tmp_logits[lane_id*nclasses + i]);
    }

    // Calculate log(softmax(logits)) which is just logits - max:
    float sum = 0.0f;
    for (int i = 0; i < nclasses; ++i) {
        float val = tmp_logits[lane_id*nclasses + i] - max;
        sum += expf(val);
        tmp_logits[lane_id*nclasses + i] = val;
    }
    sum = logf(sum);

    // log(exp(logits - max) / sum) = (logits - max) - log(sum)
    float loss = 0.0f;
    for (int i = 0; i < nclasses; ++i) {
        loss += (tmp_logits[lane_id*nclasses + i] - sum) * tmp_labels[lane_id*nclasses + i];
    }
    loss = -warp_reduce_sum(loss) / (float)k;

    __syncthreads();

    if (lane_id == 0) {
        tmp_all[warp_id] = loss;
    }

    __syncthreads();

    if (warp_id != 0) {
        return;
    }

    loss = lane_id < CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE/WARP_SIZE ? tmp_all[lane_id] : 0.0f;
    loss = warp_reduce_sum(loss);

    if (lane_id != 0) {
        return;
    }

    dst[blockIdx.x] = loss;
}

static __global__ void cross_entropy_loss_back_f32(const float * logits, const float * labels, const float * loss, float * dst, const int nclasses) {
    extern __shared__ float tmp[];

    float maxval = -INFINITY;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = logits[blockIdx.x*nclasses + i];
        maxval = fmaxf(maxval, val);
        tmp[i] = val;
    }
    maxval = warp_reduce_max(maxval);

    float sum = 0.0f;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        const float val = expf(tmp[i] - maxval);
        sum += val;
        tmp[i] = val;
    }
    sum = warp_reduce_sum(sum);
    const float sm_scale = 1.0f/sum;

    const float d_by_nrows = *loss/gridDim.x;
    for (int i = threadIdx.x; i < nclasses; i += WARP_SIZE) {
        dst[blockIdx.x*nclasses + i] = (tmp[i]*sm_scale - labels[blockIdx.x*nclasses + i])*d_by_nrows;
    }
}

void ggml_cuda_cross_entropy_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE, 1, 1);
    const dim3 blocks_num((nrows + CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE - 1) / CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE, 1, 1);
    const int shmem = 2*CUDA_CROSS_ENTROPY_LOSS_BLOCK_SIZE*ne00*sizeof(float);

    ggml_cuda_pool_alloc<float> dst_tmp(pool, blocks_num.x);

    cross_entropy_loss_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, dst_tmp.ptr, ne00, nrows);

    // Combine results from individual blocks:
    sum_f32_cuda(pool, dst_tmp.ptr, dst_d, blocks_num.x, stream);
}

void ggml_cuda_cross_entropy_loss_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * opt0 = dst->src[2];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(opt0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(opt0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * opt0_d = (const float *) opt0->data;
    float       * dst_d  = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    const dim3 blocks_dim(WARP_SIZE, 1, 1);
    const dim3 blocks_num(nrows, 1, 1);
    const int shmem = ne00*sizeof(float);

    cross_entropy_loss_back_f32<<<blocks_num, blocks_dim, shmem, stream>>>(src0_d, src1_d, opt0_d, dst_d, ne00);
}

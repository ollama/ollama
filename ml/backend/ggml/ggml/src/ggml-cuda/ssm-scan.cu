#include "ssm-scan.cuh"

template <size_t splitD, size_t N>
__global__ void __launch_bounds__(splitD, 2)
    ssm_scan_f32(const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
                 const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
                 const int32_t * __restrict__ src6, float * __restrict__ dst,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L) {

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int bidx = blockIdx.x;  // split along B (sequences)
    const int bidy = blockIdx.y;  // split along D (d_inner)
    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int wtid = tid % 32;

    extern __shared__ float smem[];
    const int               stride_sA  = N + 1;
    const int               stride_ss0 = N + 1;
    float *                 smem_A     = smem;
    float *                 smem_s0    = smem_A + splitD * stride_sA;

    const float * s0_block = (const float *) ((const char *) src0 + src6[bidx] * src0_nb3 + bidy * splitD * src0_nb2);
    const float * x_block  = (const float *) ((const char *) src1 + (bidx * src1_nb3) + bidy * splitD * sizeof(float));
    const float * dt_block = (const float *) ((const char *) src2 + (bidx * src2_nb2) + bidy * splitD * sizeof(float));
    const float * A_block  = (const float *) ((const char *) src3 + bidy * splitD * src3_nb1);
    const float * B_block  = (const float *) ((const char *) src4 + (bidx * src4_nb3));
    const float * C_block  = (const float *) ((const char *) src5 + (bidx * src5_nb3));
    float *       y_block  = (float *) ((char *) dst + (bidx * d_inner * L * sizeof(float)) + bidy * splitD * sizeof(float));
    float *       s_block  = (float *) ((char *) dst + s_off + bidx * src0_nb3 + bidy * splitD * src0_nb2);

    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_A  = src3_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_s  = stride_s0;
    const int stride_y  = d_inner;

    // can N not be 16? for example 32?
    if (N == 16) {
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = A_block[(wid * warp_size + i) * stride_A + wtid];
            // todo: bank conflict
            // I am always confused with how to use the swizzling method to solve
            // bank conflit. Hoping somebody can tell me.
            smem_A[(wid * warp_size + i) * stride_sA + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
        }
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = s0_block[(wid * warp_size + i) * stride_s0 + wtid];
            smem_s0[(wid * warp_size + i) * stride_ss0 + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
        }
    }

    __syncthreads();

    for (int64_t i = 0; i < L; i++) {
        float dt_soft_plus = dt_block[i * stride_dt + tid];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(exp(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + tid] * dt_soft_plus;
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < N; j++) {
            float state = (smem_s0[tid * stride_ss0 + j] * expf(dt_soft_plus * smem_A[tid * stride_sA + j])) +
                          (B_block[i * stride_B + j] * x_dt);
            sumf += state * C_block[i * stride_C + j];
            if (i == L - 1) {
                s_block[tid * stride_s + j] = state;
            } else {
                smem_s0[tid * stride_ss0 + j] = state;
            }
        }
        __syncthreads();
        y_block[i * stride_y + tid] = sumf;
    }
}

// assumes as many threads as d_state
template <int splitH, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan_f32_group(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        const int32_t * __restrict__ src6, float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {

    const int head_idx = (blockIdx.x * splitH) / d_head;
    const int head_off = ((blockIdx.x * splitH) % d_head) * sizeof(float);
    const int seq_idx = blockIdx.y;

    const int group_off = (head_idx & (n_group - 1)) * d_state * sizeof(float);

    const float * s0_block = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_block  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + blockIdx.x * splitH * sizeof(float));
    const float * dt_block = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_block  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_block  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_block  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_block  = dst + (seq_idx * n_tok * n_head * d_head) + blockIdx.x * splitH;
    float *       s_block  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[splitH];
    // for the parallel accumulation
    __shared__ float stateC[splitH * d_state];

#pragma unroll
    for (int j = 0; j < splitH; j++) {
        state[j] = s0_block[j * d_state + threadIdx.x];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // TODO: only calculate dA and dt_soft_plus once per head instead of every splitH head elements
        // TODO: only calculate B and C once per head group
        // NOTE: dt_soft_plus, dA and x_dt have the same value across threads here.
        float dt_soft_plus = dt_block[i * stride_dt];
        if (dt_soft_plus <= 20.0f) {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        const float dA = expf(dt_soft_plus * A_block[0]);
        const float B = B_block[i * stride_B + threadIdx.x];
        const float C = C_block[i * stride_C + threadIdx.x];

        // across d_head
#pragma unroll
        for (int j = 0; j < splitH; j++) {
            const float x_dt = x_block[i * stride_x + j] * dt_soft_plus;

            state[j] = (state[j] * dA) + (B * x_dt);

            stateC[j * d_state + threadIdx.x] = state[j] * C;
        }

        __syncthreads();

        // parallel accumulation for stateC
        // TODO: simplify
        {
            static_assert((d_state & -d_state) == d_state, "the state size has to be a power of 2");
            static_assert((splitH & -splitH) == splitH, "splitH has to be a power of 2");

            // reduce until w matches the warp size
            // TODO: does this work even when the physical warp size is 64?
#pragma unroll
            for (int w = d_state; w > WARP_SIZE; w >>= 1) {
                // (assuming there are d_state threads)
#pragma unroll
                for (int j = 0; j < ((w >> 1) * splitH + d_state - 1) / d_state; j++) {
                    // TODO: check for bank conflicts
                    const int k = (threadIdx.x % (w >> 1)) + (d_state * (threadIdx.x / (w >> 1))) + j * d_state * (d_state / (w >> 1));
                    stateC[k] += stateC[k + (w >> 1)];

                }
                __syncthreads();
            }

            static_assert(splitH >= d_state / WARP_SIZE);

#pragma unroll
            for (int j = 0; j < splitH / (d_state / WARP_SIZE); j++) {
                float y = stateC[(threadIdx.x % WARP_SIZE) + d_state * (threadIdx.x / WARP_SIZE) + j * d_state * (d_state / WARP_SIZE)];
                y = warp_reduce_sum(y);

                // store the above accumulations
                if (threadIdx.x % WARP_SIZE == 0) {
                    const int k = threadIdx.x / WARP_SIZE + j * (d_state / WARP_SIZE);
                    y_block[i * stride_y + k] = y;
                }
            }
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < splitH; j++) {
        s_block[j * d_state + threadIdx.x] = state[j];
    }
}

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int32_t * src6, float * dst,
                              const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3, const int src2_nb1,
                              const int src2_nb2, const int src3_nb1, const int src4_nb2, const int src4_nb3, const int src5_nb2,
                              const int src5_nb3, const int64_t s_off, const int64_t d_state, const int64_t head_dim,
                              const int64_t n_head, const int64_t n_group, const int64_t n_tok, const int64_t n_seq,
                              cudaStream_t stream) {
    // NOTE: if you change conditions here, be sure to update the corresponding supports_op condition!
    if (src3_nb1 == sizeof(float)) {
        // Mamba-2
        if (d_state == 128) {
            const int threads = 128;
            GGML_ASSERT(d_state % threads == 0);
            // NOTE: can be any power of two between 4 and 64
            const int splitH = 16;
            GGML_ASSERT(head_dim % splitH == 0);
            const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
            ssm_scan_f32_group<16, 128><<<blocks, threads, 0, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else if (d_state == 256) { // Falcon-H1
            const int threads = 256;
            // NOTE: can be any power of two between 8 and 64
            const int splitH = 16;
            GGML_ASSERT(head_dim % splitH == 0);
            const dim3 blocks((n_head * head_dim + (splitH - 1)) / splitH, n_seq, 1);
            ssm_scan_f32_group<16, 256><<<blocks, threads, 0, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    } else {
        const int threads = 128;
        // Mamba-1
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);
        const dim3 blocks(n_seq, (n_head + threads - 1) / threads, 1);
        const int  smem_size = (threads * (d_state + 1) * 2) * sizeof(float);
        if (d_state == 16) {
            ssm_scan_f32<128, 16><<<blocks, threads, smem_size, stream>>>(
                src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
        } else {
            GGML_ABORT("doesn't support d_state!=16.");
        }
    }
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C
    const struct ggml_tensor * src6 = dst->src[6];  // ids

    const int64_t nc  = src0->ne[0];  // d_state
    const int64_t nr  = src0->ne[1];  // head_dim or 1
    const int64_t nh  = src1->ne[1];  // n_head
    const int64_t ng  = src4->ne[1];  // n_group
    const int64_t n_t = src1->ne[2];  // number of tokens per sequence
    const int64_t n_s = src1->ne[3];  // number of sequences in the batch

    const int64_t s_off = ggml_nelements(src1) * sizeof(float);

    GGML_ASSERT(ggml_nelements(src1) + nc*nr*nh*n_s == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    GGML_ASSERT(src6->nb[0] == sizeof(int32_t));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    const int32_t * src6_d = (const int32_t *) src6->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src6->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src6_d, dst_d,
                      src0->nb[2], src0->nb[3], src1->nb[2], src1->nb[3], src2->nb[1], src2->nb[2],
                      src3->nb[1], src4->nb[2], src4->nb[3], src5->nb[2], src5->nb[3],
                      s_off, nc, nr, nh, ng, n_t, n_s, stream);
}

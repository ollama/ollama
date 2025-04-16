#include "ssm-scan.cuh"

template <size_t splitD, size_t N>
__global__ void __launch_bounds__(splitD, 2)
    ssm_scan_f32(const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
                 const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
                 const int src0_nb1, const int src0_nb2, const int src1_nb0, const int src1_nb1, const int src1_nb2,
                 const int src1_nb3, const int src2_nb0, const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
                 float * __restrict__ dst, const int64_t L) {
    GGML_UNUSED(src1_nb0);
    GGML_UNUSED(src2_nb0);
    const int bidx = blockIdx.x;  // split along B
    const int bidy = blockIdx.y;  // split along D
    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int wtid = tid % 32;

    extern __shared__ float smem[];
    const int               stride_sA  = N + 1;
    const int               stride_ss0 = N + 1;
    float *                 smem_A     = smem;
    float *                 smem_s0    = smem_A + splitD * stride_sA;

    const float * s0_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * splitD * src0_nb1);
    const float * x_block  = (const float *) ((const char *) src1 + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    const float * dt_block = (const float *) ((const char *) src2 + (bidx * src2_nb2) + bidy * splitD * sizeof(float));
    const float * A_block  = (const float *) ((const char *) src3 + bidy * splitD * src3_nb1);
    const float * B_block  = (const float *) ((const char *) src4 + (bidx * src4_nb2));
    const float * C_block  = (const float *) ((const char *) src5 + (bidx * src5_nb2));
    float *       y_block  = (float *) ((char *) dst + (bidx * src1_nb2) + bidy * splitD * sizeof(float));
    float *       s_block  = (float *) ((char *) dst + src1_nb3 + bidx * src0_nb2 + bidy * splitD * src0_nb1);

    const int stride_s0 = src0_nb1 / sizeof(float);
    const int stride_x  = src1_nb1 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_A  = src3_nb1 / sizeof(float);
    const int stride_B  = src4_nb1 / sizeof(float);
    const int stride_C  = src5_nb1 / sizeof(float);
    const int stride_s  = stride_s0;
    const int stride_y  = stride_x;

    // can N not be 16? for example 32?
    if (N == 16) {
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = A_block[(wid * warpSize + i) * stride_A + wtid];
            // todo: bank conflict
            // I am always confused with how to use the swizzling method to solve
            // bank conflit. Hoping somebody can tell me.
            smem_A[(wid * warpSize + i) * stride_sA + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
        }
#pragma unroll
        for (size_t i = 0; i < splitD / 4; i += 2) {
            float value = s0_block[(wid * warpSize + i) * stride_s0 + wtid];
            smem_s0[(wid * warpSize + i) * stride_ss0 + wtid + ((wtid / 16) > 0 ? 1 : 0)] = value;
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

static void ssm_scan_f32_cuda(const float * src0, const float * src1, const float * src2, const float * src3,
                              const float * src4, const float * src5, const int src0_nb1, const int src0_nb2,
                              const int src1_nb0, const int src1_nb1, const int src1_nb2, const int src1_nb3,
                              const int src2_nb0, const int src2_nb1, const int src2_nb2, const int src3_nb1,
                              const int src4_nb1, const int src4_nb2, const int src5_nb1, const int src5_nb2,
                              float * dst, const int64_t N, const int64_t D, const int64_t L, const int64_t B,
                              cudaStream_t stream) {
    const int threads = 128;
    // todo: consider D cannot be divided,does this situation exist?
    GGML_ASSERT(D % threads == 0);
    const dim3 blocks(B, (D + threads - 1) / threads, 1);
    const int  smem_size = (threads * (N + 1) * 2) * sizeof(float);
    if (N == 16) {
        ssm_scan_f32<128, 16><<<blocks, threads, smem_size, stream>>>(
            src0, src1, src2, src3, src4, src5, src0_nb1, src0_nb2, src1_nb0, src1_nb1, src1_nb2, src1_nb3, src2_nb0,
            src2_nb1, src2_nb2, src3_nb1, src4_nb1, src4_nb2, src5_nb1, src5_nb2, dst, L);
    } else {
        GGML_ABORT("doesn't support N!=16.");
    }
}

void ggml_cuda_op_ssm_scan(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // s
    const struct ggml_tensor * src1 = dst->src[1];  // x
    const struct ggml_tensor * src2 = dst->src[2];  // dt
    const struct ggml_tensor * src3 = dst->src[3];  // A
    const struct ggml_tensor * src4 = dst->src[4];  // B
    const struct ggml_tensor * src5 = dst->src[5];  // C

    //   const int64_t d_state = src0->ne[0];
    //   const int64_t d_inner = src0->ne[1];
    //   const int64_t l = src1->ne[1];
    //   const int64_t b = src0->ne[2];

    const int64_t nc  = src0->ne[0];  // d_state
    const int64_t nr  = src0->ne[1];  // d_inner
    const int64_t n_t = src1->ne[1];  // number of tokens per sequence
    const int64_t n_s = src0->ne[2];  // number of sequences in the batch

    GGML_ASSERT(ggml_nelements(src1) + ggml_nelements(src0) == ggml_nelements(dst));
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src2->nb[0] == sizeof(float));
    GGML_ASSERT(src3->nb[0] == sizeof(float));
    GGML_ASSERT(src4->nb[0] == sizeof(float));
    GGML_ASSERT(src5->nb[0] == sizeof(float));
    // required for the dot product between s and C
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    // required for per-sequence offsets for states
    GGML_ASSERT(src0->nb[2] == src0->ne[0] * src0->ne[1] * sizeof(float));
    // required to get correct offset for state destination (i.e. src1->nb[3])
    GGML_ASSERT(src1->nb[3] == src1->ne[0] * src1->ne[1] * src1->ne[2] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * src2_d = (const float *) src2->data;
    const float * src3_d = (const float *) src3->data;
    const float * src4_d = (const float *) src4->data;
    const float * src5_d = (const float *) src5->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ssm_scan_f32_cuda(src0_d, src1_d, src2_d, src3_d, src4_d, src5_d, src0->nb[1], src0->nb[2], src1->nb[0],
                      src1->nb[1], src1->nb[2], src1->nb[3], src2->nb[0], src2->nb[1], src2->nb[2], src3->nb[1],
                      src4->nb[1], src4->nb[2], src5->nb[1], src5->nb[2], dst_d, nc, nr, n_t, n_s, stream);
}

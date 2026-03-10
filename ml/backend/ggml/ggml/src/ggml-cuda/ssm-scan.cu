#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070
#define USE_CUB
#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 11070

#ifdef USE_CUB
#include <cub/cub.cuh>
using namespace cub;
#endif // USE_CUB

#include "ssm-scan.cuh"

// We would like to keep pragma unroll for cases where L_template is not 0,
// so we suppress the clang transformation warning.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpass-failed"
#endif // __clang__
template <size_t splitD, size_t N, size_t L_template>
__global__ void __launch_bounds__(splitD, 1)
    ssm_scan_f32(const float *__restrict__ src0, const float *__restrict__ src1, const float *__restrict__ src2,
                 const float *__restrict__ src3, const float *__restrict__ src4, const float *__restrict__ src5,
                 const int32_t * __restrict__ src6, float * __restrict__ dst,
                 const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
                 const int src2_nb1, const int src2_nb2, const int src3_nb1,
                 const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
                 const int64_t s_off, const int64_t d_inner, const int64_t L_param)
{
    const size_t L = L_template == 0 ? L_param : L_template;
    const float *s0_block = (const float *)((const char *)src0 + src6[blockIdx.x] * src0_nb3 + blockIdx.y * splitD * src0_nb2);
    const float *x_block = (const float *)((const char *)src1 + (blockIdx.x * src1_nb3) + blockIdx.y * splitD * sizeof(float));
    const float *dt_block = (const float *)((const char *)src2 + (blockIdx.x * src2_nb2) + blockIdx.y * splitD * sizeof(float));
    const float *A_block = (const float *)((const char *)src3 + blockIdx.y * splitD * src3_nb1);
    const float *B_block = (const float *)((const char *)src4 + (blockIdx.x * src4_nb3));
    const float *C_block = (const float *)((const char *)src5 + (blockIdx.x * src5_nb3));
    float *y_block = (float *)((char *)dst + (blockIdx.x * d_inner * L * sizeof(float)) + blockIdx.y * splitD * sizeof(float));
    float *s_block = (float *)((char *)dst + s_off + blockIdx.x * src0_nb3 + blockIdx.y * splitD * src0_nb2);

    const int stride_x = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B = src4_nb2 / sizeof(float);
    const int stride_C = src5_nb2 / sizeof(float);
    const int stride_y = d_inner;

    float regA[N];
    float regs0[N];

    __shared__ float smemB[N];
    __shared__ float smemC[N];

#ifdef USE_CUB
    using BlockLoad = cub::BlockLoad<float, splitD, N, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStore = cub::BlockStore<float, splitD, N, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    union CubTempStorage {
        typename BlockLoad::TempStorage load_temp;
        typename BlockStore::TempStorage store_temp;
    };
    __shared__ CubTempStorage cub_temp_storage;

    BlockLoad(cub_temp_storage.load_temp).Load(A_block, regA);
    BlockLoad(cub_temp_storage.load_temp).Load(s0_block, regs0);
#else
    const int stride_s0 = src0_nb2 / sizeof(float);
    const int stride_A = src3_nb1 / sizeof(float);
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        regA[n] = A_block[threadIdx.x * stride_A + n];
        regs0[n] = s0_block[threadIdx.x * stride_s0 + n];
    }
#endif

#pragma unroll
    for (size_t i = 0; i < L; i++)
    {
        if (threadIdx.x < N)
        {
            smemB[threadIdx.x] = B_block[i * stride_B + threadIdx.x];
            smemC[threadIdx.x] = C_block[i * stride_C + threadIdx.x];
        }
        __syncthreads();

        float dt_soft_plus = dt_block[i * stride_dt + threadIdx.x];
        if (dt_soft_plus <= 20.0f)
        {
            dt_soft_plus = log1pf(expf(dt_soft_plus));
        }
        float x_dt = x_block[i * stride_x + threadIdx.x] * dt_soft_plus;

        float sumf = 0.0f;
#pragma unroll
        for (size_t n = 0; n < N; n++)
        {
            float state = regs0[n] * expf(dt_soft_plus * regA[n]) + smemB[n] * x_dt;
            sumf += state * smemC[n];
            regs0[n] = state;
        }
        y_block[i * stride_y + threadIdx.x] = sumf;
    }

#ifdef USE_CUB
    BlockStore(cub_temp_storage.store_temp).Store(s_block, regs0);
#else
    const int stride_s = stride_s0;
#pragma unroll
    for (size_t n = 0; n < N; ++n)
    {
        s_block[threadIdx.x * stride_s + n] = regs0[n];
    }
#endif
}
#ifdef __clang__
#pragma clang diagnostic pop
#endif // __clang__

// assumes as many threads as d_state
template <int c_factor, int d_state>
__global__ void __launch_bounds__(d_state, 1)
    ssm_scan_f32_group(
        const float * __restrict__ src0, const float * __restrict__ src1, const float * __restrict__ src2,
        const float * __restrict__ src3, const float * __restrict__ src4, const float * __restrict__ src5,
        const int32_t * __restrict__ src6, float * __restrict__ dst,
        const int src0_nb2, const int src0_nb3, const int src1_nb2, const int src1_nb3,
        const int src2_nb1, const int src2_nb2, const int src3_nb1,
        const int src4_nb2, const int src4_nb3, const int src5_nb2, const int src5_nb3,
        const int64_t s_off, const int64_t n_head, const int64_t d_head, const int64_t n_group, const int64_t n_tok) {

    const int warp     = threadIdx.x / WARP_SIZE;
    const int lane     = threadIdx.x % WARP_SIZE;
    const int warp_idx = blockIdx.x  * c_factor + warp;

    const int head_idx =  warp_idx / d_head;
    const int head_off = (warp_idx % d_head) * sizeof(float);
    const int seq_idx  = blockIdx.y;

    const int group_off = (head_idx / (n_head / n_group)) * d_state * sizeof(float);

    // TODO: refactor strides to be in elements/floats instead of bytes to be cleaner and consistent with the rest of the codebase
    const float * s0_warp = (const float *) ((const char *) src0 + src6[seq_idx] * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);
    const float * x_warp  = (const float *) ((const char *) src1 + (seq_idx * src1_nb3) + (warp_idx * sizeof(float)));
    const float * dt_warp = (const float *) ((const char *) src2 + (seq_idx * src2_nb2) + head_idx * sizeof(float));
    const float * A_warp  = (const float *) ((const char *) src3 + head_idx * src3_nb1);
    const float * B_warp  = (const float *) ((const char *) src4 + (seq_idx * src4_nb3) + (group_off));
    const float * C_warp  = (const float *) ((const char *) src5 + (seq_idx * src5_nb3) + (group_off));
    float *       y_warp  = dst + (seq_idx * n_tok * n_head * d_head) + warp_idx;
    float *       s_warp  = (float *) ((char *) dst + s_off + seq_idx * src0_nb3 + head_idx * src0_nb2 + head_off * d_state);

    // strides across n_seq_tokens
    const int stride_x  = src1_nb2 / sizeof(float);
    const int stride_dt = src2_nb1 / sizeof(float);
    const int stride_B  = src4_nb2 / sizeof(float);
    const int stride_C  = src5_nb2 / sizeof(float);
    const int stride_y  = n_head * d_head;

    float state[c_factor];
    float state_sum = 0.0f;

#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        state[j] = s0_warp[WARP_SIZE * j + lane];
    }

    for (int64_t i = 0; i < n_tok; i++) {
        // NOTE: dt_soft_plus, dA and x_dt have the same value for a warp here.
        // Recalculation is intentional; sharing via shuffles/smem proved slower due to sync overhead.
        const float dt_soft_plus = (dt_warp[i * stride_dt] <= 20.0f ? log1pf(expf(dt_warp[i * stride_dt])) : dt_warp[i * stride_dt]);

        state_sum = 0.0f;
        const float dA   = expf(dt_soft_plus * A_warp[0]);
        const float x_dt = x_warp[i * stride_x] * dt_soft_plus;
#pragma unroll
        for (int j = 0; j < c_factor; j++) {
            const float B_val = B_warp[i * stride_B + WARP_SIZE * j + lane];
            const float C_val = C_warp[i * stride_C + WARP_SIZE * j + lane];
            state[j] = (state[j] * dA) + (B_val * x_dt);
            state_sum += state[j] * C_val;
        }

        // parallel accumulation for output
        state_sum = warp_reduce_sum(state_sum);

        if (lane == 0) {
            y_warp[i * stride_y] = state_sum;
        }
    }

    // write back the state
#pragma unroll
    for (int j = 0; j < c_factor; j++) {
        s_warp[WARP_SIZE * j + lane] = state[j];
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
            constexpr int threads   = 128;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            ssm_scan_f32_group<128/WARP_SIZE, 128><<<blocks, threads, 0, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else if (d_state == 256) { // Falcon-H1
            constexpr int threads   = 256;
            constexpr int num_warps = threads/WARP_SIZE;

            const dim3 blocks((n_head * head_dim + (num_warps - 1)) / num_warps, n_seq, 1);
            ssm_scan_f32_group<256/WARP_SIZE, 256><<<blocks, threads, 0, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                    src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2, src3_nb1,
                    src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, head_dim, n_group, n_tok);
        } else {
            GGML_ABORT("doesn't support d_state!=(128 or 256).");
        }
    } else {
        // Mamba-1
        constexpr int threads = 128;
        GGML_ASSERT(n_head % threads == 0);
        GGML_ASSERT(head_dim == 1);
        GGML_ASSERT(n_group == 1);
        const dim3 blocks(n_seq, (n_head + threads - 1) / threads, 1);
        const int  smem_size = (threads * (d_state + 1) * 2) * sizeof(float);
        if (d_state == 16) {
            switch (n_tok)
            {
            case 1:
                ssm_scan_f32<threads, 16, 1><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 2:
                ssm_scan_f32<threads, 16, 2><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 3:
                ssm_scan_f32<threads, 16, 3><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 4:
                ssm_scan_f32<threads, 16, 4><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 5:
                ssm_scan_f32<threads, 16, 5><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 6:
                ssm_scan_f32<threads, 16, 6><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 7:
                ssm_scan_f32<threads, 16, 7><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            case 8:
                ssm_scan_f32<threads, 16, 8><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            default:
                ssm_scan_f32<threads, 16, 0><<<blocks, threads, smem_size, stream>>>(
                    src0, src1, src2, src3, src4, src5, src6, dst,
                src0_nb2, src0_nb3, src1_nb2, src1_nb3, src2_nb1, src2_nb2,
                src3_nb1, src4_nb2, src4_nb3, src5_nb2, src5_nb3, s_off, n_head, n_tok);
                break;
            }
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

#include "common.cuh"
#include "gla.cuh"

template<int HEAD_SIZE>
static __global__ void gated_linear_attn_f32(const int B, const int T, const int C, const int H, const float scale,
     const float * k, const float * v, const float * r, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = HEAD_SIZE;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4 & k = (float4 &)(_k[j]);
            const float4 & r = (float4 &)(_r[j]);
            const float4 & td = (float4 &)(_td[j]);
            float4 & s = (float4 &)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;

            y += r.x * s.x;
            y += r.y * s.y;
            y += r.z * s.z;
            y += r.w * s.w;
        }
        dst[t] = y * scale;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

void ggml_cuda_op_gated_linear_attn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * td_d = (const float *)dst->src[3]->data;
    const float * s_d  = (const float *)dst->src[4]->data;

    const int64_t B = dst->src[4]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float scale;
    memcpy(&scale, (float*)dst->op_params, sizeof(float));

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[4]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == 64 || C / H == 128);


    if (C / H == 64) {
        gated_linear_attn_f32<64><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    } else {
        gated_linear_attn_f32<128><<<B * H, C / H, 0, stream>>>(B, T, C, H, scale, k_d, v_d, r_d, td_d, s_d, dst_d);
    }
}

#include "rope.cuh"

struct rope_corr_dims {
    float v[2];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static __device__ void rope_yarn(
    float theta_extrap, float freq_scale, rope_corr_dims corr_dims, int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

template<typename T, bool has_ff>
static __global__ void rope_norm(
    const T * x, T * dst, int ne0, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ne0 + i0;
    const int i2 = row/p_delta_rows;

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

template<typename T, bool has_ff>
static __global__ void rope_neox(
    const T * x, T * dst, int ne0, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, const float * freq_factors) {
    const int i0 = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (i0 >= ne0) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= n_dims) {
        const int i = row*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ne0 + i0/2;
    const int i2 = row/p_delta_rows;

    const float theta_base = pos[i2]*powf(theta_scale, i0/2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + n_dims/2];

    dst[i + 0]        = x0*cos_theta - x1*sin_theta;
    dst[i + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template<typename T>
static void rope_norm_cuda(
    const T * x, T * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_norm<T, false><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
                theta_scale, freq_factors
                );
    } else {
        rope_norm<T, true><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
                theta_scale, freq_factors
                );
    }
}

template<typename T>
static void rope_neox_cuda(
    const T * x, T * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int n_blocks_x = (ne0 + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nr, n_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    if (freq_factors == nullptr) {
        rope_neox<T, false><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
                theta_scale, freq_factors
                );
    } else {
        rope_neox<T, true><<<block_nums, block_dims, 0, stream>>>(
                x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
                theta_scale, freq_factors
                );
    }
}

static void rope_norm_cuda_f16(
    const half * x, half * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {

    rope_norm_cuda<half>(x, dst, ne0, n_dims, nr, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
}

static void rope_norm_cuda_f32(
    const float * x, float * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {

    rope_norm_cuda<float>(x, dst, ne0, n_dims, nr, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
}

static void rope_neox_cuda_f16(
    const half * x, half * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream) {

    rope_neox_cuda<half>(x, dst, ne0, n_dims, nr, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
}

static void rope_neox_cuda_f32(
    const float * x, float * dst, int ne0, int n_dims, int nr, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, cudaStream_t stream
) {

    rope_neox_cuda<float>(x, dst, ne0, n_dims, nr, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, stream);
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;

    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t nr = ggml_nrows(src0);

    //const int n_past     = ((int32_t *) dst->op_params)[0];
    const int n_dims     = ((int32_t *) dst->op_params)[1];
    const int mode       = ((int32_t *) dst->op_params)[2];
    //const int n_ctx      = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig = ((int32_t *) dst->op_params)[4];

    // RoPE alteration for extended context
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;

    const int32_t * pos = (const int32_t *) src1_d;

    const float * freq_factors = nullptr;
    if (src2 != nullptr) {
        freq_factors = (const float *) src2->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_neox) {
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, stream
            );
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_norm_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_norm_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, stream
            );
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

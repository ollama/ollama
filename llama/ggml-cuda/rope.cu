#include "rope.cuh"

struct rope_corr_dims {
    float v[4];
};

static __device__ float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static __device__ void rope_yarn(
    float theta_extrap, float freq_scale, rope_corr_dims corr_dims, int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta
) {
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

// rope == RoPE == rotary positional embedding
template<typename T, bool has_pos>
static __global__ void rope(
    const T * x, T * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims
) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base = p*powf(freq_base, -float(col)/ncols);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, col, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + 1];

    dst[i + 0] = x0*cos_theta - x1*sin_theta;
    dst[i + 1] = x0*sin_theta + x1*cos_theta;
}

template<typename T, bool has_pos>
static __global__ void rope_neox(
    const T * x, T * dst, int ncols, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, float inv_ndims
) {
    const int col = 2*(blockDim.y*blockIdx.y + threadIdx.y);

    if (col >= ncols) {
        return;
    }

    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int ib = col / n_dims;
    const int ic = col % n_dims;

    if (ib > 0) {
        const int i = row*ncols + ib*n_dims + ic;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ncols + ib*n_dims + ic/2;
    const int i2 = row/p_delta_rows;

    float cur_rot = inv_ndims * ic - ib;

    const int p = has_pos ? pos[i2] : 0;
    const float theta_base = p*freq_scale*powf(theta_scale, col/2.0f);

    float cos_theta, sin_theta;
    rope_yarn(theta_base, freq_scale, corr_dims, cur_rot, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + n_dims/2];

    dst[i + 0]        = x0*cos_theta - x1*sin_theta;
    dst[i + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

static __global__ void rope_glm_f32(
    const float * x, float * dst, int ncols, const int32_t * pos, float freq_scale, int p_delta_rows, float freq_base,
    int n_ctx
) {
    const int col = blockDim.x*blockIdx.x + threadIdx.x;
    const int half_n_dims = ncols/4;

    if (col >= half_n_dims) {
        return;
    }

    const int row = blockDim.y*blockIdx.y + threadIdx.y;
    const int i = row*ncols + col;
    const int i2 = row/p_delta_rows;

    const float col_theta_scale = powf(freq_base, -2.0f*col/ncols);
     // FIXME: this is likely wrong
    const int p = pos != nullptr ? pos[i2] : 0;

    const float theta = min(p, n_ctx - 2)*freq_scale*col_theta_scale;
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + half_n_dims];

    dst[i + 0]           = x0*cos_theta - x1*sin_theta;
    dst[i + half_n_dims] = x0*sin_theta + x1*cos_theta;

    const float block_theta = ((float)max(p - n_ctx - 2, 0))*col_theta_scale;
    const float sin_block_theta = sinf(block_theta);
    const float cos_block_theta = cosf(block_theta);

    const float x2 = x[i + half_n_dims * 2];
    const float x3 = x[i + half_n_dims * 3];

    dst[i + half_n_dims * 2] = x2*cos_block_theta - x3*sin_block_theta;
    dst[i + half_n_dims * 3] = x2*sin_block_theta + x3*cos_block_theta;
}


template<typename T>
static void rope_cuda(
    const T * x, T * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);
    if (pos == nullptr) {
        rope<T, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims
        );
    } else {
        rope<T, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims
        );
    }
}

template<typename T>
static void rope_neox_cuda(
    const T * x, T * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 2 == 0);
    const dim3 block_dims(1, CUDA_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ncols + 2*CUDA_ROPE_BLOCK_SIZE - 1) / (2*CUDA_ROPE_BLOCK_SIZE);
    const dim3 block_nums(nrows, num_blocks_x, 1);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);
    const float inv_ndims = -1.0f / n_dims;

    if (pos == nullptr) {
        rope_neox<T, false><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
            theta_scale, inv_ndims
        );
    } else {
        rope_neox<T, true><<<block_nums, block_dims, 0, stream>>>(
            x, dst, ncols, n_dims, pos, freq_scale, p_delta_rows, ext_factor, attn_factor, corr_dims,
            theta_scale, inv_ndims
        );
    }
}

static void rope_glm_f32_cuda(
    const float * x, float * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, int n_ctx, cudaStream_t stream
) {
    GGML_ASSERT(ncols % 4 == 0);
    const dim3 block_dims(CUDA_ROPE_BLOCK_SIZE/4, 1, 1);
    const int num_blocks_x = (ncols + CUDA_ROPE_BLOCK_SIZE - 1) / CUDA_ROPE_BLOCK_SIZE;
    const dim3 block_nums(num_blocks_x, nrows, 1);
    rope_glm_f32<<<block_nums, block_dims, 0, stream>>>(x, dst, ncols, pos, freq_scale, p_delta_rows, freq_base, n_ctx);
}

static void rope_cuda_f16(
    const half * x, half * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_cuda<half>(x, dst, ncols, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_cuda_f32(
    const float * x, float * dst, int ncols, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_cuda<float>(x, dst, ncols, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_neox_cuda_f16(
    const half * x, half * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream) {

    rope_neox_cuda<half>(x, dst, ncols, n_dims, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

static void rope_neox_cuda_f32(
    const float * x, float * dst, int ncols, int n_dims, int nrows, const int32_t * pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, cudaStream_t stream
) {

    rope_neox_cuda<float>(x, dst, ncols, n_dims, nrows, pos, freq_scale, p_delta_rows, freq_base, ext_factor, attn_factor, corr_dims, stream);
}

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const float * src0_d = (const float *)src0->data;
    const float * src1_d = (const float *)src1->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne2 = dst->ne[2];
    const int64_t nrows = ggml_nrows(src0);

    //const int n_past      = ((int32_t *) dst->op_params)[0];
    const int n_dims      = ((int32_t *) dst->op_params)[1];
    const int mode        = ((int32_t *) dst->op_params)[2];
    const int n_ctx       = ((int32_t *) dst->op_params)[3];
    const int n_orig_ctx  = ((int32_t *) dst->op_params)[4];

    // RoPE alteration for extended context
    float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    memcpy(&freq_base,   (int32_t *) dst->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (int32_t *) dst->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (int32_t *) dst->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (int32_t *) dst->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (int32_t *) dst->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (int32_t *) dst->op_params + 10, sizeof(float));

    const int32_t * pos = nullptr;
    if ((mode & 1) == 0) {
        GGML_ASSERT(src1->type == GGML_TYPE_I32);
        GGML_ASSERT(src1->ne[0] == ne2);
        pos = (const int32_t *) src1_d;
    }

    const bool is_neox = mode & 2;
    const bool is_glm  = mode & 4;

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_orig_ctx, freq_base, beta_fast, beta_slow, corr_dims.v);

    // compute
    if (is_glm) {
        GGML_ASSERT(false);
        rope_glm_f32_cuda(src0_d, dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, n_ctx, stream);
    } else if (is_neox) {
        if (src0->type == GGML_TYPE_F32) {
            rope_neox_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, n_dims, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_neox_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, n_dims, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else {
            GGML_ASSERT(false);
        }
    } else {
        if (src0->type == GGML_TYPE_F32) {
            rope_cuda_f32(
                (const float *)src0_d, (float *)dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else if (src0->type == GGML_TYPE_F16) {
            rope_cuda_f16(
                (const half *)src0_d, (half *)dst_d, ne00, nrows, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, stream
            );
        } else {
            GGML_ASSERT(false);
        }
    }
}

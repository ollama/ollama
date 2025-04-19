#include "rope.hpp"

struct rope_corr_dims {
    float v[2];
};

static float rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / sycl::max(0.001f, high - low);
    return 1.0f - sycl::min(1.0f, sycl::max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
static void rope_yarn(
    float theta_extrap, float freq_scale, rope_corr_dims corr_dims, int64_t i0, float ext_factor, float mscale,
    float * cos_theta, float * sin_theta) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.v[0], corr_dims.v[1], i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * sycl::log(1.0f / freq_scale);
    }
    *cos_theta = sycl::cos(theta) * mscale;
    *sin_theta = sycl::sin(theta) * mscale;
}

template<typename T, bool has_ff>
static void rope_norm(
    const T * x, T * dst, int ne0, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, const float * freq_factors,
    const sycl::nd_item<3> &item_ct1) {
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                         item_ct1.get_local_id(1));

    if (i0 >= ne0) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (i0 >= n_dims) {
        const int i = row*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i = row*ne0 + i0;
    const int i2 = row/p_delta_rows;

    const float theta_base = pos[i2] * sycl::pow(theta_scale, i0 / 2.0f);

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
static void rope_neox(
    const T * x, T * dst, int ne0, int n_dims, const int32_t * pos, float freq_scale, int p_delta_rows,
    float ext_factor, float attn_factor, rope_corr_dims corr_dims, float theta_scale, const float * freq_factors,
    const sycl::nd_item<3> &item_ct1) {
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                         item_ct1.get_local_id(1));

    if (i0 >= ne0) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    if (i0 >= n_dims) {
        const int i = row*ne0 + i0;

        dst[i + 0] = x[i + 0];
        dst[i + 1] = x[i + 1];

        return;
    }

    const int i  = row*ne0 + i0/2;
    const int i2 = row/p_delta_rows;

    const float theta_base = pos[i2] * sycl::pow(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0/2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i + 0];
    const float x1 = x[i + n_dims/2];

    dst[i + 0]        = x0*cos_theta - x1*sin_theta;
    dst[i + n_dims/2] = x0*sin_theta + x1*cos_theta;
}

template <typename T>
static void rope_norm_sycl(
    const T *x, T *dst, int ne0, int n_dims, int nr, const int32_t *pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ne0 + 2*SYCL_ROPE_BLOCK_SIZE - 1) / (2*SYCL_ROPE_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, num_blocks_x, nr);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

    if (freq_factors == nullptr) {
        /*
        DPCT1049:40: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_norm<T, false>(x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows,
                               ext_factor, attn_factor, corr_dims, theta_scale, freq_factors,
                               item_ct1);
            });
    } else {
        /*
        DPCT1049:41: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_norm<T, true>(x, dst, ne0, n_dims, pos, freq_scale, p_delta_rows,
                              ext_factor, attn_factor, corr_dims, theta_scale, freq_factors,
                              item_ct1);
            });
    }
}

template <typename T>
static void rope_neox_sycl(
    const T *x, T *dst, int ne0, int n_dims, int nr, const int32_t *pos, float freq_scale, int p_delta_rows,
    float freq_base, float ext_factor, float attn_factor, rope_corr_dims corr_dims, const float * freq_factors, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int num_blocks_x = (ne0 + 2*SYCL_ROPE_BLOCK_SIZE - 1) / (2*SYCL_ROPE_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, num_blocks_x, nr);

    const float theta_scale = powf(freq_base, -2.0f/n_dims);

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    if (freq_factors == nullptr) {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_neox<T, false>(x, dst, ne0, n_dims, pos, freq_scale,
                                    p_delta_rows, ext_factor, attn_factor,
                                    corr_dims, theta_scale, freq_factors,
                                    item_ct1);
            });
    } else {
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                rope_neox<T, true>(x, dst, ne0, n_dims, pos, freq_scale,
                                    p_delta_rows, ext_factor, attn_factor,
                                    corr_dims, theta_scale, freq_factors,
                                    item_ct1);
            });
    }
}

void ggml_sycl_op_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    const int64_t ne00 = dst->src[0]->ne[0];
    const int64_t ne01 = dst->src[0]->ne[1];
    const int64_t nr = ggml_nrows(dst->src[0]);

    //const int n_past      = ((int32_t *) dst->op_params)[0];
    const int n_dims      = ((int32_t *) dst->op_params)[1];
    const int mode        = ((int32_t *) dst->op_params)[2];
    //const int n_ctx       = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig  = ((int32_t *) dst->op_params)[4];

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

    const int32_t * pos = (const int32_t *) dst->src[1]->data;

    const float * freq_factors = nullptr;
    if (dst->src[2] != nullptr) {
        freq_factors = (const float *) dst->src[2]->data;
    }

    rope_corr_dims corr_dims;
    ggml_rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow, corr_dims.v);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    // compute
    if (is_neox) {
        if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_neox_sycl(
                (const float *)dst->src[0]->data, (float *)dst->data, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, main_stream
            );
        } else if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_neox_sycl(
                (const sycl::half *)dst->src[0]->data, (sycl::half *)dst->data, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, main_stream
            );
        } else {
            GGML_ABORT("fatal error");
        }
    } else {
        if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_norm_sycl(
                (const float *)dst->src[0]->data, (float *)dst->data, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, main_stream
            );
        } else if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_norm_sycl(
                (const sycl::half *)dst->src[0]->data, (sycl::half *)dst->data, ne00, n_dims, nr, pos, freq_scale, ne01, freq_base, ext_factor,
                attn_factor, corr_dims, freq_factors, main_stream
            );
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

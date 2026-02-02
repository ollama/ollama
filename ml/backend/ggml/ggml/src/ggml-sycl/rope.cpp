#include "rope.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml.h"

struct rope_corr_dims {
    float v[2];
};

struct mrope_sections {
    int v[4];
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

template <typename T, bool has_ff>
static void rope_norm(const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
                      const int32_t * pos, float freq_scale, float ext_factor, float attn_factor,
                      const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors,
                      const sycl::nd_item<3> & item_ct1) {
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1));

    if (i0 >= ne0) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    const int row0     = row % ne1;
    const int channel0 = row / ne1;

    const int i  = row * ne0 + i0;
    const int i2 = channel0 * s2 + row0 * s1 + i0;

    if (i0 >= n_dims) {
        *reinterpret_cast<sycl::vec<T, 2> *>(dst + i) = *reinterpret_cast<const sycl::vec<T, 2> *>(x + i2);
        return;
    }

    const float theta_base = pos[channel0] * sycl::pow(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i2 + 0];
    const float x1 = x[i2 + 1];

    dst[i + 0] = x0 * cos_theta - x1 * sin_theta;
    dst[i + 1] = x0 * sin_theta + x1 * cos_theta;
}

template <typename T, bool has_ff>
static void rope_neox(const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2, const int n_dims,
                      const int32_t * pos, const float freq_scale, const float ext_factor, const float attn_factor,
                      const rope_corr_dims corr_dims, const float theta_scale, const float * freq_factors,
                      const sycl::nd_item<3> & item_ct1) {
    const int i0 = 2 * (item_ct1.get_local_range(1) * item_ct1.get_group(1) + item_ct1.get_local_id(1));

    if (i0 >= ne0) {
        return;
    }

    const int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    const int row0     = row % ne1;
    const int channel0 = row / ne1;

    const int i  = row * ne0 + i0 / 2;
    const int i2 = channel0 * s2 + row0 * s1 + i0 / 2;

    if (i0 >= n_dims) {
        *reinterpret_cast<sycl::vec<T, 2> *>(dst + i + i0 / 2) = *reinterpret_cast<const sycl::vec<T, 2> *>(x + i2 + i0 / 2);
        return;
    }

    const float theta_base = pos[channel0] * sycl::pow(theta_scale, i0 / 2.0f);

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;

    float cos_theta;
    float sin_theta;

    rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);

    const float x0 = x[i2 + 0];
    const float x1 = x[i2 + n_dims / 2];

    dst[i + 0]          = x0 * cos_theta - x1 * sin_theta;
    dst[i + n_dims / 2] = x0 * sin_theta + x1 * cos_theta;
}

template <typename T, bool has_ff>
static void rope_multi(const T * x, T * dst, const int ne0, const int ne1, const int ne2, const size_t s1,
                        const size_t s2, const int n_dims, const int32_t * pos, const float freq_scale,
                        const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                        const float theta_scale, const float * freq_factors, const mrope_sections sections,
                        const bool is_imrope, const sycl::nd_item<3> & item_ct1) {
    // get index pos
    const int i0 = 2 * (item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1));
    if (i0 >= ne0) {
        return;
    }
    const int    row_dst   = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);

    const int    row_x     = row_dst % ne1;
    const int    channel_x = row_dst / ne1;
    const int    idst      = (row_dst * ne0) + (i0 / 2);
    const size_t ix        = ((size_t) channel_x * s2) + ((size_t) row_x * s1) + (i0 / 2);

    if (i0 >= n_dims) {
        *reinterpret_cast<sycl::vec<T, 2> *>(dst + idst + i0 / 2) = *reinterpret_cast<const sycl::vec<T, 2> *>(x + i0 / 2 + ix);
        return;
    }

    const int sect_dims = sections.v[0] + sections.v[1] + sections.v[2] + sections.v[3];
    const int sec_w = sections.v[1] + sections.v[0];
    const int sector = (i0 / 2) % sect_dims;


    float theta_base = 0.0;
    if (is_imrope) {
        if (sector % 3 == 1 && sector < 3 * sections.v[1]) {
            theta_base = pos[channel_x + ne2 * 1]*sycl::pow(theta_scale, i0/2.0f);
        } else if (sector % 3 == 2 && sector < 3 * sections.v[2]) {
            theta_base = pos[channel_x + ne2 * 2]*sycl::pow(theta_scale, i0/2.0f);
        } else if (sector % 3 == 0 && sector < 3 * sections.v[0]) {
            theta_base = pos[channel_x]*sycl::pow(theta_scale, i0/2.0f);
        } else {
            theta_base = pos[channel_x + ne2 * 3]*sycl::pow(theta_scale, i0/2.0f);
        }
    } else {
        if (sector < sections.v[0]) {
            theta_base = pos[channel_x]*sycl::pow(theta_scale, i0/2.0f);
        }
        else if (sector >= sections.v[0] && sector < sec_w) {
            theta_base = pos[channel_x + ne2 * 1]*sycl::pow(theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w && sector < sec_w + sections.v[2]) {
            theta_base = pos[channel_x + ne2 * 2]*sycl::pow(theta_scale, i0/2.0f);
        }
        else if (sector >= sec_w + sections.v[2]) {
            theta_base = pos[channel_x + ne2 * 3]*sycl::pow(theta_scale, i0/2.0f);
        }
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;
    float       cos_theta;
    float       sin_theta;
    rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);
    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims/2];

    // store results in dst
    dst[idst + 0]      = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims/2] = x0 * sin_theta + x1 * cos_theta;
}



template <typename T, bool has_ff>
static void rope_vision(const T * x, T * dst, const int ne0, const int ne1, const int ne2, const size_t s1,
                        const size_t s2, const int n_dims, const int32_t * pos, const float freq_scale,
                        const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                        const float theta_scale, const float * freq_factors, const mrope_sections sections,
                        const sycl::nd_item<3> & item_ct1) {
    // get index pos
    const int i0 = 2 * (item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1));
    if (i0 >= ne0) {
        return;
    }
    const int    row_dst   = (item_ct1.get_group(2) * item_ct1.get_local_range(2)) + item_ct1.get_local_id(2);
    const int    row_x     = row_dst % ne1;
    const int    channel_x = row_dst / ne1;
    const int    idst      = (row_dst * ne0) + (i0 / 2);
    const size_t ix        = ((size_t) channel_x * s2) + ((size_t) row_x * s1) + (i0 / 2);

    const int sect_dims = sections.v[0] + sections.v[1];
    const int sector    = (i0 / 2) % sect_dims;

    float theta_base = 0.0f;
    if (sector < sections.v[0]) {
        const int p = sector;
        theta_base  = pos[channel_x] * sycl::pow(theta_scale, (float) p);
    } else {
        // Simplified from CUDA backend code: if (sector >= sections.v[0] && sector < sec_w) which is just sector >= sections.v[0]
        const int p = sector - sections.v[0];
        theta_base  = pos[channel_x + ne2] * sycl::pow(theta_scale, (float) p);
    }

    const float freq_factor = has_ff ? freq_factors[i0 / 2] : 1.0f;
    float       cos_theta;
    float       sin_theta;
    rope_yarn(theta_base / freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor, &cos_theta, &sin_theta);
    const float x0 = x[ix + 0];
    const float x1 = x[ix + n_dims];

    // store results in dst
    dst[idst + 0]      = x0 * cos_theta - x1 * sin_theta;
    dst[idst + n_dims] = x0 * sin_theta + x1 * cos_theta;
}

template <typename T>
static void rope_norm_sycl(const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2,
                           const int n_dims, int nr, const int32_t * pos, const float freq_scale, const float freq_base,
                           const float ext_factor, const float attn_factor, const rope_corr_dims corr_dims,
                           const float * freq_factors, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int            num_blocks_x = ceil_div(ne0, (2 * SYCL_ROPE_BLOCK_SIZE));
    const sycl::range<3> block_nums(1, num_blocks_x, nr);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    if (freq_factors == nullptr) {
        /*
        DPCT1049:40: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            rope_norm<T, false>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                                theta_scale, freq_factors, item_ct1);
        });
    } else {
        /*
        DPCT1049:41: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            rope_norm<T, true>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                               theta_scale, freq_factors, item_ct1);
        });
    }
}

template <typename T>
static void rope_neox_sycl(const T * x, T * dst, const int ne0, const int ne1, const int s1, const int s2,
                           const int n_dims, const int nr, const int32_t * pos, const float freq_scale,
                           const float freq_base, const float ext_factor, const float attn_factor,
                           const rope_corr_dims corr_dims, const float * freq_factors, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3> block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int            num_blocks_x = ceil_div(ne0, (2 * SYCL_ROPE_BLOCK_SIZE));
    const sycl::range<3> block_nums(1, num_blocks_x, nr);

    const float theta_scale = powf(freq_base, -2.0f / n_dims);

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    if (freq_factors == nullptr) {
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            rope_neox<T, false>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                                theta_scale, freq_factors, item_ct1);
        });
    } else {
        stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
            rope_neox<T, true>(x, dst, ne0, ne1, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor, corr_dims,
                               theta_scale, freq_factors, item_ct1);
        });
    }
}

template <typename T>
static void rope_multi_sycl(const T * x, T * dst, const int ne0, const int ne1, const int ne2, const size_t s1,
                             const size_t s2, const int n_dims, const int nr, const int32_t * pos,
                             const float freq_scale, const float freq_base, const float ext_factor,
                             const float attn_factor, const rope_corr_dims corr_dims, const float * freq_factors,
                             const mrope_sections sections, const bool is_imrope, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3>    block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int               n_blocks_y = ceil_div(ne0, (2 * SYCL_ROPE_BLOCK_SIZE));
    const sycl::range<3>    grid_dims(1, n_blocks_y, nr);
    const sycl::nd_range<3> nd_range(grid_dims * block_dims, block_dims);

    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    // Add FP16 capability check if T could be sycl::half
    if constexpr (std::is_same_v<T, sycl::half>) {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    }
    // launch kernel
    if (freq_factors == nullptr) {
        stream->parallel_for(nd_range, [=](sycl::nd_item<3> item_ct1) {
            rope_multi<T, false>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                  corr_dims, theta_scale, freq_factors, sections, is_imrope, item_ct1);
        });
    } else {
        stream->parallel_for(nd_range, [=](sycl::nd_item<3> item_ct1) {
            rope_multi<T, true>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                 corr_dims, theta_scale, freq_factors, sections, is_imrope, item_ct1);
        });
    }
}




// rope vision
template <typename T>
static void rope_vision_sycl(const T * x, T * dst, const int ne0, const int ne1, const int ne2, const size_t s1,
                             const size_t s2, const int n_dims, const int nr, const int32_t * pos,
                             const float freq_scale, const float freq_base, const float ext_factor,
                             const float attn_factor, const rope_corr_dims corr_dims, const float * freq_factors,
                             const mrope_sections sections, queue_ptr stream) {
    GGML_ASSERT(ne0 % 2 == 0);
    const sycl::range<3>    block_dims(1, SYCL_ROPE_BLOCK_SIZE, 1);
    const int               n_blocks_y = ceil_div(ne0, (2 * SYCL_ROPE_BLOCK_SIZE));
    const sycl::range<3>    grid_dims(1, n_blocks_y, nr);
    const sycl::nd_range<3> nd_range(grid_dims * block_dims, block_dims);

    const float theta_scale = std::pow(freq_base, -2.0f / n_dims);
    // Add FP16 capability check if T could be sycl::half
    if constexpr (std::is_same_v<T, sycl::half>) {
        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    }
    // launch kernel
    if (freq_factors == nullptr) {
        stream->parallel_for(nd_range, [=](sycl::nd_item<3> item_ct1) {
            rope_vision<T, false>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                  corr_dims, theta_scale, freq_factors, sections, item_ct1);
        });
    } else {
        stream->parallel_for(nd_range, [=](sycl::nd_item<3> item_ct1) {
            rope_vision<T, true>(x, dst, ne0, ne1, ne2, s1, s2, n_dims, pos, freq_scale, ext_factor, attn_factor,
                                 corr_dims, theta_scale, freq_factors, sections, item_ct1);
        });
    }
}

inline void ggml_sycl_op_rope(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);
    const int64_t ne00 = dst->src[0]->ne[0]; // head dims
    const int64_t ne01 = dst->src[0]->ne[1]; // num heads
    const int64_t ne02 = dst->src[0]->ne[2]; // num heads
    const int64_t nr = ggml_nrows(dst->src[0]);

    const size_t s01 = dst->src[0]->nb[1] / ggml_type_size(dst->src[0]->type);
    const size_t s02 = dst->src[0]->nb[2] / ggml_type_size(dst->src[0]->type);


    //const int n_past      = ((int32_t *) dst->op_params)[0];
    const int n_dims      = ((int32_t *) dst->op_params)[1];
    const int mode        = ((int32_t *) dst->op_params)[2];
    //const int n_ctx       = ((int32_t *) dst->op_params)[3];
    const int n_ctx_orig  = ((int32_t *) dst->op_params)[4];
    mrope_sections sections;

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
    memcpy(&sections.v,  (int32_t *) dst->op_params + 11, sizeof(int)*4);

    const bool is_neox = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_mrope) {
        GGML_ASSERT(sections.v[0] > 0 || sections.v[1] > 0 || sections.v[2] > 0);
    }

    if (is_vision) {
        GGML_ASSERT(n_dims == ne00/2);
    }

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
        GGML_SYCL_DEBUG("%s: neox path\n", __func__);
        if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_neox_sycl((const float *) dst->src[0]->data, (float *) dst->data, ne00, ne01, s01, s02, n_dims, nr,
                           pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
        } else if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_neox_sycl((const sycl::half *) dst->src[0]->data, (sycl::half *) dst->data, ne00, ne01, s01, s02,
                           n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                           main_stream);
        } else {
            GGML_ABORT("fatal error");
        }
    } else if (is_mrope && !is_vision) {
        GGML_SYCL_DEBUG("%s: mrope path\n", __func__);
        if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_multi_sycl((const sycl::half *)dst->src[0]->data, (sycl::half *)dst->data, ne00, ne01, ne02, s01,
                s02, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                freq_factors, sections, is_imrope, main_stream);
        } else if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_multi_sycl((const float *) dst->src[0]->data, (float *) dst->data, ne00, ne01, ne02, s01, s02, n_dims,
                             nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections,
                             is_imrope, main_stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else if (is_vision) {
        GGML_SYCL_DEBUG("%s: vision path\n", __func__);
        if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_vision_sycl((const sycl::half *) dst->src[0]->data, (sycl::half *) dst->data, ne00, ne01, ne02, s01,
                             s02, n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims,
                             freq_factors, sections, main_stream);
        } else if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_vision_sycl((const float *) dst->src[0]->data, (float *) dst->data, ne00, ne01, ne02, s01, s02, n_dims,
                             nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, sections,
                             main_stream);
        } else {
            GGML_ABORT("Fatal error: Tensor type unsupported!");
        }
    } else {
        GGML_SYCL_DEBUG("%s: norm path\n", __func__);
        if (dst->src[0]->type == GGML_TYPE_F32) {
            rope_norm_sycl((const float *) dst->src[0]->data, (float *) dst->data, ne00, ne01, s01, s02, n_dims, nr,
                           pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors, main_stream);
        } else if (dst->src[0]->type == GGML_TYPE_F16) {
            rope_norm_sycl((const sycl::half *) dst->src[0]->data, (sycl::half *) dst->data, ne00, ne01, s01, s02,
                           n_dims, nr, pos, freq_scale, freq_base, ext_factor, attn_factor, corr_dims, freq_factors,
                           main_stream);
        } else {
            GGML_ABORT("fatal error");
        }
    }
}

void ggml_sycl_rope(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ggml_sycl_op_rope(ctx, dst);
}


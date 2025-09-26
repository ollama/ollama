#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ggml_metal_op * ggml_metal_op_t;

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        struct ggml_cgraph * gf,
        int  idx_start,
        int  idx_end,
        bool use_fusion,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph,
        int  debug_fusion);

void ggml_metal_op_free(ggml_metal_op_t ctx);

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx);

//
// available ops:
//

// tokens per expert
size_t ggml_metal_op_mul_mat_id_extra_tpe(const struct ggml_tensor * op);

// id map [n_tokens, n_expert]
size_t ggml_metal_op_mul_mat_id_extra_ids(const struct ggml_tensor * op);

// return true if we should use the FA vector kernel for this op
bool ggml_metal_op_flash_attn_ext_use_vec(const struct ggml_tensor * op);

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const struct ggml_tensor * op);

int ggml_metal_op_concat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_repeat            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_acc               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_scale             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_clamp             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_unary             (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_glu               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_sum_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_get_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_set_rows          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_soft_max          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_conv          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_ssm_scan          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rwkv              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_cpy               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pool_2d           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_mul_mat_id        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_add_id            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_flash_attn_ext    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_bin               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rms_norm          (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_l2_norm           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_group_norm        (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_norm              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_rope              (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_im2col            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_conv_transpose_1d (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_upscale           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad               (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_pad_reflect_1d    (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_arange            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argmax            (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_argsort           (ggml_metal_op_t ctx, int idx);
int ggml_metal_op_leaky_relu        (ggml_metal_op_t ctx, int idx);

#ifdef __cplusplus
}
#endif

#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(void);
void ggml_metal_cv_free(ggml_metal_cv_t cv);

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx);
void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nsg(ggml_metal_pipeline_t pipeline, int nsg);
int  ggml_metal_pipeline_get_nsg(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nr0(ggml_metal_pipeline_t pipeline, int nr0);
int  ggml_metal_pipeline_get_nr0(ggml_metal_pipeline_t pipeline);

void ggml_metal_pipeline_set_nr1(ggml_metal_pipeline_t pipeline, int nr1);
int  ggml_metal_pipeline_get_nr1(ggml_metal_pipeline_t pipeline);

void   ggml_metal_pipeline_set_smem(ggml_metal_pipeline_t pipeline, size_t smem);
size_t ggml_metal_pipeline_get_smem(ggml_metal_pipeline_t pipeline);

int ggml_metal_pipeline_max_theads_per_threadgroup(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, ggml_metal_pipeline_t pipeline);

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev);
void ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
ggml_metal_pipeline_t ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, enum ggml_type tsrc);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, enum ggml_type tidx, enum ggml_type tdst);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, enum ggml_type tsrc);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, enum ggml_type tsrc0, enum ggml_type tsrc1, int nsg, int nxpsg, int r1ptg);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, int ne02, int ne20);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, enum ggml_op op, int32_t n_fuse, bool row);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, const struct ggml_tensor * op);
ggml_metal_pipeline_t ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, const struct ggml_tensor * op);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        int32_t nwg);

ggml_metal_pipeline_t ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

//
// device
//

struct ggml_metal_device_props {
    char name[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool has_unified_memory;
    bool has_bfloat;
    bool use_residency_sets;
    bool use_shared_buffers;

    bool supports_gpu_family_apple7;
};

ggml_metal_device_t ggml_metal_device_init(void);
void ggml_metal_device_free(ggml_metal_device_t dev);

// return a singleton that is automatically destroyed when the program exits
ggml_metal_device_t ggml_metal_device_get(void);

void * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
void * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op);

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t);

#ifdef __cplusplus
}
#endif

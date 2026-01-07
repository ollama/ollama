#if !defined(GGML_ROPE_PARAMS)
#define GGML_ROPE_PARAMS

#include "rte.glsl"

struct rope_params {
    uint rope_mode;
    uint ncols;
    uint n_dims;
    float freq_scale;
    uint p_delta_rows;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    uint has_ff;
    uint ne02;
    uint nb01;
    uint nb02;
    int sections[4];
    uint is_imrope;
    uint is_back;
    uint set_rows_stride;
};

#endif // !defined(GGML_ROPE_PARAMS)

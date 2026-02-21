#if !defined(GGML_ROPE_PARAMS)
#define GGML_ROPE_PARAMS

#include "rte.glsl"

struct rope_params {
    uint rope_mode;
    uint nrows;
    uint n_dims;
    float freq_scale;
    float freq_base;
    float ext_factor;
    float attn_factor;
    float corr_dims[2];
    float theta_scale;
    uint has_ff;
    int sections[4];
    uint is_imrope;
    uint is_back;
    uint set_rows_stride;

    uint ne00;
    uint ne01;
    uint ne02;
    uint nb01;
    uint nb02;
    uint nb03;
    uint nb11;
    uint nb12;
    uint nb13;
};

#endif // !defined(GGML_ROPE_PARAMS)

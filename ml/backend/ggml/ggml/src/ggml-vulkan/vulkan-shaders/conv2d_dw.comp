#version 450

#include "types.glsl"

layout (push_constant) uniform parameter
{
    uint ne;
    uint batches;
    uint channels;
    uint dst_w;
    uint dst_h;
    uint src_w;
    uint src_h;
    uint knl_w;
    uint knl_h;
    int stride_x;
    int stride_y;
    int pad_x;
    int pad_y;
    int dilation_x;
    int dilation_y;
} p;

layout (binding = 0) readonly buffer A {A_TYPE knl_data[];};
layout (binding = 1) readonly buffer B {B_TYPE src_data[];};
layout (binding = 2) writeonly buffer D {D_TYPE dst_data[];};

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

FLOAT_TYPE conv_2d_dw_whcn(uint idx) {
    uint i0 = idx / p.dst_w;
    uint dst_x = idx - i0 * p.dst_w;
    uint i1 = i0 / p.dst_h;
    uint dst_y = i0 - i1 * p.dst_h;
    uint n = i1 / p.channels;
    uint c = i1 - n * p.channels;

    uint src_i = n * p.channels * p.src_h * p.src_w + c * p.src_h * p.src_w;
    uint knl_i = c * p.knl_h * p.knl_w;

    FLOAT_TYPE sum = 0.0;
    for (uint knl_y = 0; knl_y < p.knl_h; ++knl_y) {
        uint src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
        if (src_y >= p.src_h) { // src_y < 0 will wrap to a large unsigned int
            continue;
        }
        for (uint knl_x = 0; knl_x < p.knl_w; ++knl_x) {
            uint src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
            if (src_x >= p.src_w) { // src_x < 0 will wrap to a large unsigned int
                continue;
            }
            FLOAT_TYPE v = FLOAT_TYPE(src_data[src_i + src_y * p.src_w + src_x]);
            FLOAT_TYPE k = FLOAT_TYPE(knl_data[knl_i + knl_y * p.knl_w + knl_x]);
            sum = fma(v, k, sum);
        }
    }
    return sum;
}

FLOAT_TYPE conv_2d_dw_cwhn(uint idx) {
    uint i0 = idx / p.channels;
    uint c = idx - i0 * p.channels;
    uint i1 = i0 / p.dst_w;
    uint dst_x = i0 - i1 * p.dst_w;
    uint n = i1 / p.dst_h;
    uint dst_y = i1 - n * p.dst_h;

    uint src_i = n * p.channels * p.src_h * p.src_w;
    uint src_row = p.src_w * p.channels;
    uint knl_row = p.knl_w * p.channels;

    FLOAT_TYPE sum = 0.0;
    for (uint knl_y = 0; knl_y < p.knl_h; ++knl_y) {
        uint src_y = dst_y * p.stride_y + knl_y * p.dilation_y - p.pad_y;
        if (src_y >= p.src_h) { // src_y < 0 will wrap to a large unsigned int
            continue;
        }
        for (uint knl_x = 0; knl_x < p.knl_w; ++knl_x) {
            uint src_x = dst_x * p.stride_x + knl_x * p.dilation_x - p.pad_x;
            if (src_x >= p.src_w) { // src_x < 0 will wrap to a large unsigned int
                continue;
            }
            FLOAT_TYPE v = FLOAT_TYPE(src_data[src_i + src_y * src_row + src_x * p.channels + c]);
            FLOAT_TYPE k = FLOAT_TYPE(knl_data[        knl_y * knl_row + knl_x * p.channels + c]);
            sum = fma(v, k, sum);
        }
    }
    return sum;
}

void main() {
    uint idx = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
    if (idx >= p.ne) {
        return;
    }

    FLOAT_TYPE result =
#ifdef WHCN
        conv_2d_dw_whcn(idx);
#else
        conv_2d_dw_cwhn(idx);
#endif
    dst_data[idx] = D_TYPE(result);
}


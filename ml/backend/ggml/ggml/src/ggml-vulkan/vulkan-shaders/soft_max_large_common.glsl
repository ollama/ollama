#extension GL_EXT_control_flow_attributes : enable

layout (push_constant) uniform parameter
{
    uint KX;
    uint KY;
    uint ne00;
    uint ne01;
    uint ne02;
    uint ne12;
    uint ne13;
    uint nb11;
    uint nb12;
    uint nb13;
    float scale;
    float max_bias;
    float m0;
    float m1;
    uint n_head_log2;
    uint nrows_x;
    uint has_sinks;
} p;

#include "types.glsl"

layout(constant_id = 0) const uint BLOCK_SIZE = 128;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint num_iters = 4;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) readonly buffer Y {B_TYPE data_b[];};
layout (binding = 2) readonly buffer Z {float data_c[];};
layout (binding = 3) buffer D {D_TYPE data_d[];};
layout (binding = 4) buffer M {float data_m[];};
layout (binding = 5) buffer S {float data_s[];};

shared FLOAT_TYPE vals[BLOCK_SIZE];

float get_slope(uint rowx) {
    float slope = 1.0f;

    // ALiBi
    if (p.max_bias > 0.0f) {
        const uint h = (rowx / p.ne01) % p.ne02; // head index

        const float base = h < p.n_head_log2 ? p.m0 : p.m1;
        const uint   exp = h < p.n_head_log2 ? h + 1 : 2*(h - p.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    return slope;
}

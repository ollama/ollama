#version 450

#include "generic_head.glsl"
#include "types.glsl"

#extension GL_EXT_control_flow_attributes : enable

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

void main() {
    // based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
    // ref: https://www.johndcook.com/blog/python_erf/
    const float p_erf  = 0.3275911f;
    const float a1_erf = 0.254829592f;
    const float a2_erf = -0.284496736f;
    const float a3_erf = 1.421413741f;
    const float a4_erf = -1.453152027f;
    const float a5_erf = 1.061405429f;

    const float SQRT_2_INV = 0.70710678118654752440084436210484f;
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.KX) {
        return;
    }

    const float a = float(data_a[i]);
    const float a_div_sqr2 = a * SQRT_2_INV;
    const float sign_x = sign(a_div_sqr2);
    const float x = abs(a_div_sqr2);
    const float t = 1.0f / (1.0f + p_erf * x);
    const float y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    const float erf_approx = sign_x * y;

    data_d[i] = D_TYPE(0.5f * a * (1.0f + erf_approx));
}

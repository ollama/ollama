#extension GL_EXT_shader_16bit_storage : require

#include "rte.glsl"

layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {A_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (push_constant) uniform parameter
{
    uint N;
    uint ne00;
    uint ne20;
    uint mode;
    float alpha;
    float limit;
} p;

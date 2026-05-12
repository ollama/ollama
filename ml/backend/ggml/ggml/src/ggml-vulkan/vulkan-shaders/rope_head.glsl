#include "types.glsl"

#extension GL_EXT_shader_16bit_storage : require

#include "rte.glsl"
#include "rope_params.glsl"

layout(local_size_x = 1, local_size_y = 256, local_size_z = 1) in;

layout (binding = 0) readonly buffer X {A_TYPE rope_data_a[];};
layout (binding = 1) readonly buffer Y {int rope_data_pos[];};
layout (binding = 2) readonly buffer Z {float rope_data_ff[];};
layout (binding = 3) writeonly buffer D {ROPE_D_TYPE rope_data_d[];};
layout (binding = 4) readonly buffer I {uvec2 rope_data_i[];}; // indices for set_rows


layout (push_constant) uniform parameter {
    rope_params pc;
};


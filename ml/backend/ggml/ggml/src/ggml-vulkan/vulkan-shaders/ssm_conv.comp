#version 450

#extension GL_EXT_control_flow_attributes : require

#include "types.glsl"

layout(constant_id = 0) const uint BLOCK_SIZE = 32;
layout(constant_id = 1) const uint TOKENS_PER_WG = 16;

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;

layout(binding = 0) readonly buffer Src0 { float src0[]; };
layout(binding = 1) readonly buffer Src1 { float src1[]; };
layout(binding = 2) buffer Dst { float dst[]; };

layout(push_constant) uniform PushConstants {
    uint nb01; uint nb02;
    uint nb11;
    uint dst_nb0; uint dst_nb1; uint dst_nb2;
    uint nc; uint ncs; uint nr; uint n_t; uint n_s;
};

void main() {
    const uint i1 = gl_GlobalInvocationID.x;
    const uint i2 = gl_WorkGroupID.y * TOKENS_PER_WG + gl_LocalInvocationID.y;
    const uint i3 = gl_WorkGroupID.z;

    if (i1 >= nr || i2 >= n_t || i3 >= n_s) {
        return;
    }

    const uint src0_base = i3 * (nb02 / 4) + i2 + i1 * (nb01 / 4);
    const uint src1_base = i1 * (nb11 / 4);

    float sum = 0.0;

    if (nc == 4) {
        sum = dot(
            vec4(src0[src0_base], src0[src0_base + 1], src0[src0_base + 2], src0[src0_base + 3]),
            vec4(src1[src1_base], src1[src1_base + 1], src1[src1_base + 2], src1[src1_base + 3])
        );
    } else {
        [[unroll]] for (uint i0 = 0; i0 < nc; i0++) {
            sum += src0[src0_base + i0] * src1[src1_base + i0];
        }
    }

    const uint dst_idx = i3 * (dst_nb2 / 4) + i2 * (dst_nb1 / 4) + i1;
    dst[dst_idx] = sum;
}

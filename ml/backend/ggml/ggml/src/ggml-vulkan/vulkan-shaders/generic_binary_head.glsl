#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

#include "rte.glsl"
#include "utils.glsl"

layout (push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint ne20; uint ne21; uint ne22; uint ne23; uint nb20; uint nb21; uint nb22; uint nb23;
    uint misalign_offsets;
    float param1; float param2; int param3;
} p;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

// true if src0/src1 are the same shape and the indices can be reused without additional modulus
layout(constant_id = 0) const bool norepeat = false;

uint get_idx() {
    return gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
}

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_boffset() { return (p.misalign_offsets >> 8) & 0xFF; }
uint get_doffset() { return p.misalign_offsets & 0xFF; }


void get_indices(uint idx, out uint i00, out uint i01, out uint i02, out uint i03) {
    get_indices(idx, i00, i01, i02, i03, p.ne00, p.ne01, p.ne02, p.ne03);
}

uint src0_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
}

uint src1_idx(uint i00, uint i01, uint i02, uint i03) {
    if (norepeat) {
        return i03*p.nb13 + i02*p.nb12 + i01*p.nb11 + i00*p.nb10;
    } else {
        return fastmod(i03, p.ne13)*p.nb13 + fastmod(i02, p.ne12)*p.nb12 + fastmod(i01, p.ne11)*p.nb11 + fastmod(i00, p.ne10)*p.nb10;
    }
}

uint dst_idx(uint i00, uint i01, uint i02, uint i03) {
    return i03*p.nb23 + i02*p.nb22 + i01*p.nb21 + i00*p.nb20;
}

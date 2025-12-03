#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_control_flow_attributes : require

layout (push_constant) uniform parameter
{
    uint ne;
    uint ne00; uint ne01; uint ne02; uint ne03; uint nb00; uint nb01; uint nb02; uint nb03;
    uint ne10; uint ne11; uint ne12; uint ne13; uint nb10; uint nb11; uint nb12; uint nb13;
    uint misalign_offsets;
    float param1; float param2;

    uint ne0_012mp; uint ne0_012L;
    uint ne0_01mp;  uint ne0_01L;
    uint ne0_0mp;   uint ne0_0L;
    uint ne1_012mp; uint ne1_012L;
    uint ne1_01mp;  uint ne1_01L;
    uint ne1_0mp;   uint ne1_0L;
} p;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) writeonly buffer D {D_TYPE data_d[];};

uint get_idx() {
    return gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;
}

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }

// see init_fastdiv_values in ggml-vulkan.cpp
uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;
    // msbs = mulhi(n, mp)
    umulExtended(n, mp, msbs, lsbs);
    return (msbs + n) >> L;
}

uint src0_idx(uint idx) {
    const uint i03 = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02*p.ne01*p.ne00;
    const uint i01 = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00 = idx - i03_offset - i02_offset - i01*p.ne00;
    return i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + i00*p.nb00;
}

uint dst_idx(uint idx) {
    const uint i13 = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12*p.ne11*p.ne10;
    const uint i12 = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12*p.ne11*p.ne10;
    const uint i11 = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10 = idx - i13_offset - i12_offset - i11*p.ne10;
    return i13*p.nb13 + i12*p.nb12 + i11*p.nb11 + i10*p.nb10;
}

uint src0_idx_quant(uint idx, uint qk) {
    const uint i03 = fastdiv(idx, p.ne0_012mp, p.ne0_012L);
    const uint i03_offset = i03 * p.ne02*p.ne01*p.ne00;
    const uint i02 = fastdiv(idx - i03_offset, p.ne0_01mp, p.ne0_01L);
    const uint i02_offset = i02*p.ne01*p.ne00;
    const uint i01 = fastdiv(idx - i03_offset - i02_offset, p.ne0_0mp, p.ne0_0L);
    const uint i00 = idx - i03_offset - i02_offset - i01*p.ne00;
    return i03*p.nb03 + i02*p.nb02 + i01*p.nb01 + (i00/qk)*p.nb00;
}

uint dst_idx_quant(uint idx, uint qk) {
    const uint i13 = fastdiv(idx, p.ne1_012mp, p.ne1_012L);
    const uint i13_offset = i13 * p.ne12*p.ne11*p.ne10;
    const uint i12 = fastdiv(idx - i13_offset, p.ne1_01mp, p.ne1_01L);
    const uint i12_offset = i12*p.ne11*p.ne10;
    const uint i11 = fastdiv(idx - i13_offset - i12_offset, p.ne1_0mp, p.ne1_0L);
    const uint i10 = idx - i13_offset - i12_offset - i11*p.ne10;
    return i13*p.nb13 + i12*p.nb12 + i11*p.nb11 + (i10/qk)*p.nb10;
}

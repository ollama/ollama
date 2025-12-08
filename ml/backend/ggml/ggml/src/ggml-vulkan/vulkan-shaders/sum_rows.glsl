
// vk_op_sum_rows_push_constants
layout (push_constant) uniform parameter
{
    uint n_cols;
    uint ne01, ne02;
    uint nb01, nb02, nb03;
    uint nb11, nb12, nb13;
    float weight;
    uint misalign_offsets;
    uint ne0_12mp, ne0_12L;
    uint ne0_1mp, ne0_1L;
} p;

uint get_aoffset() { return p.misalign_offsets >> 16; }
uint get_doffset() { return p.misalign_offsets & 0xFFFF; }

// see init_fastdiv_values in ggml-vulkan.cpp
uint fastdiv(uint n, uint mp, uint L) {
    uint msbs, lsbs;
    // msbs = mulhi(n, mp)
    umulExtended(n, mp, msbs, lsbs);
    return (msbs + n) >> L;
}


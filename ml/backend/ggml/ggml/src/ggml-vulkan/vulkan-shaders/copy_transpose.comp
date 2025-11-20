#version 450

#include "types.glsl"
#include "generic_unary_head.glsl"

// workgroup does 32x32 tile, but uses 32x8 threads
#define TILE_DIM 32
layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in;

shared uint sh[TILE_DIM][TILE_DIM + 1];

void iter(uvec3 wg_id) {
    const uint tile_col = wg_id.x;
    const uint tile_row = wg_id.y;

    const uint tid_col = gl_LocalInvocationID.x;
    const uint tid_row = gl_LocalInvocationID.y;

    const uint i2 = wg_id.z % p.ne12;
    const uint i3 = wg_id.z / p.ne12;
    const uint i02 = i2;
    const uint i03 = i3;

    // The workgroup does TILE_DIM x TILE_DIM, but swaps the LSBs of the
    // src coords to make memory accesses contiguous, dst has tid.x in i0,
    // src has tid.x in i01

    [[unroll]] for (uint y = 0; y < 4; ++y) {
        const uint i00 = tile_col * TILE_DIM + tid_row + 8 * y;
        const uint i01 = tile_row * TILE_DIM + tid_col;
        if (i00 < p.ne00 && i01 < p.ne01 && i02 < p.ne02 && i03 < p.ne03) {
            const uint src_idx = i00 * p.nb00 + i01 * p.nb01 + i02 * p.nb02 + i03 * p.nb03;
            sh[tid_row + 8 * y][tid_col] = uint(data_a[get_aoffset() + src_idx]);
        }
    }

    barrier();

    [[unroll]] for (uint y = 0; y < 4; ++y) {
        const uint i0 = tile_col * TILE_DIM + tid_col;
        const uint i1 = tile_row * TILE_DIM + tid_row + 8 * y;
        if (i0 < p.ne10 && i1 < p.ne11 && i2 < p.ne12 && i3 < p.ne13) {
            const uint dst_idx = i0 * p.nb10 + i1 * p.nb11 + i2 * p.nb12 + i3 * p.nb13;
            // load transposed
            data_d[get_doffset() + dst_idx] = D_TYPE(sh[tid_col][tid_row + 8 * y]);
        }
    }
}

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

void main() {
    uint z = gl_WorkGroupID.z;
    uint y = gl_WorkGroupID.y;
    bool need_barrier = false;
    for (uint z = gl_WorkGroupID.z; z < p.ne12 * p.ne13; z += gl_NumWorkGroups.z) {
        for (uint y = gl_WorkGroupID.y; y < CEIL_DIV(p.ne11, TILE_DIM); y += gl_NumWorkGroups.y) {
            for (uint x = gl_WorkGroupID.x; x < CEIL_DIV(p.ne10, TILE_DIM); x += gl_NumWorkGroups.x) {
                if (need_barrier) {
                    barrier();
                }
                need_barrier = true;
                iter(uvec3(x, y, z));
            }
        }
    }
}

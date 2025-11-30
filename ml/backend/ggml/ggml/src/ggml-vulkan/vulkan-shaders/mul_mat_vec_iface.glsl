#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8

#ifndef MMQ
layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPE_VEC4)
layout (binding = 0) readonly buffer AV4 {A_TYPE_VEC4 data_a_v4[];};
#endif
#else
layout (binding = 0) readonly buffer A {A_TYPE_PACKED16 data_a[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPE_VEC2
layout (binding = 1) readonly buffer BV2 {B_TYPE_VEC2 data_b_v2[];};
#endif
#ifdef B_TYPE_VEC4
layout (binding = 1) readonly buffer BV4 {B_TYPE_VEC4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#ifdef MUL_MAT_ID
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif


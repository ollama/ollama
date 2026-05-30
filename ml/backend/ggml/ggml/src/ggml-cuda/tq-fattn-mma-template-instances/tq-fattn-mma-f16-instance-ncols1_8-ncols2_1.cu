// Phase D — Stage 1 placeholder instantiation.
// This file exists to validate that tq-fattn-mma-f16.cuh compiles and
// instantiates cleanly alongside stock. The kernel is not routed yet.

#include "../tq-fattn-mma-f16.cuh"

DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 1);
DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 1);
DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 1);

// Phase D — Stage 7 ncols2=2 instantiation.
// Covers GQA-2 models (gemma2:9b) at prefill batch sizes where stock
// Turing dispatch selects ncols1=8, ncols2=2 (Q->ne[1] >= 8,
// gqa_ratio divisible by 2 but not 4).

#include "../tq-fattn-mma-f16.cuh"

DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 2);
DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 2);
DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 2);

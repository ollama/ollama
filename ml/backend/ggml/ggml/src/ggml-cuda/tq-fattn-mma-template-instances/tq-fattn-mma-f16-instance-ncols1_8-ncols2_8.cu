// Phase D — Stage 6 ncols2=8 instantiation.
// Covers GQA-8 models (llama3.1:70b, qwen2.5:72b) at prefill batch sizes
// where stock Turing dispatch selects ncols1=8, ncols2=8 (Q->ne[1] >= 8,
// gqa_ratio divisible by 8).

#include "../tq-fattn-mma-f16.cuh"

DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 8);
DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 8);
DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 8);

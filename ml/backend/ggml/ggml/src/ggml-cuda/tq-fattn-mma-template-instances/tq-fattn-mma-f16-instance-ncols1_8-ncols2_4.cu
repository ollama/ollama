// Phase D — Stage 5 ncols2=4 instantiation.
// Covers GQA-4 models (llama3.2:3b, gemma3:1b, qwen3:8b) at prefill
// batch sizes where stock Turing dispatch selects ncols1=8, ncols2=4
// (Q->ne[1] in [5..8] on Turing, matching our nTokensQ>=8 gate).

#include "../tq-fattn-mma-f16.cuh"

DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 4);
DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 4);
DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 4);

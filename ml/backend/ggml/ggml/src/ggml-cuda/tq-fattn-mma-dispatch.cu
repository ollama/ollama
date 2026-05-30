// Phase D — TQ-aware copy-and-patch flash attention dispatcher.
//
// Routes K-only TQ presets (tq*k) with prefill batch (ncols1*ncols2 >= 8)
// on Turing+ to the patched stock mma-f16 case template, which decodes
// TQ-compressed K bytes into the same fragment shape stock FA expects.
// All other configurations fall through to the existing inline-decode
// path (returns false from this dispatcher).
//
// Stage 3 first cut covers a narrow slice: ncols1=8, ncols2=1
// (non-GQA-friendly prefill at Q->ne[1] >= 8, e.g. qwen2.5:7b GQA=7).
// Stage 4 adds V decode capability to the kernel; K+V routing not wired here yet.
// Stage 5 expands to ncols2=4 for GQA-4 models (llama3.2:3b, gemma3:1b, qwen3:8b).
// Stage 6 expands to ncols2=8 for GQA-8 models (llama3.1:70b, qwen2.5:72b).
// Stage 7 expands to ncols2=2 for GQA-2 models (gemma2:9b).

#include "tq-fattn.cuh"
#include "tq-fattn-mma-f16.cuh"
#include <cstdlib>

// Forward-declare the case-template instances used by this dispatcher.
// ncols1=8, ncols2=1 — Stage 3 (odd GQA, e.g. qwen2.5:7b GQA=7).
extern DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 1);
extern DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 1);
extern DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 1);
// ncols1=8, ncols2=4 — Stage 5 (GQA-4, e.g. llama3.2:3b / gemma3:1b / qwen3:8b).
extern DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 4);
extern DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 4);
extern DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 4);
// ncols1=8, ncols2=8 — Stage 6 (GQA-8, e.g. llama3.1:70b / qwen2.5:72b).
extern DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 8);
extern DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 8);
extern DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 8);
// ncols1=8, ncols2=2 — Stage 7 (GQA-2, e.g. gemma2:9b).
extern DECL_TQ_FATTN_MMA_F16_CASE( 64,  64, 8, 2);
extern DECL_TQ_FATTN_MMA_F16_CASE(128, 128, 8, 2);
extern DECL_TQ_FATTN_MMA_F16_CASE(256, 256, 8, 2);

bool ggml_cuda_tq_flash_attn_ext_mma_d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    // Phase D requires Turing+ (mma.sync.aligned + ldmatrix). Lower archs
    // (Pascal, Volta) stay on the inline-decode path.
    if (cc < GGML_CUDA_CC_TURING) {
        return false;
    }
    // HIP path uses the same .cuh but the kernel body is gated NO_DEVICE_CODE
    // via TURING_MMA_AVAILABLE; routing to it would deadlock.
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
    return false;
#endif

    const ggml_tensor * Q          = dst->src[0];
    const ggml_tensor * V          = dst->src[2];
    const ggml_tensor * v_scales_t = dst->src[6];
    const bool v_packed            = (v_scales_t != nullptr);
    const int D        = (int) Q->ne[0];
    const int nTokensQ = (int) Q->ne[1];

    // Stage 3 scope: only D ∈ {64, 128, 256} (Stage 1's instance set).
    if (D != 64 && D != 128 && D != 256) {
        return false;
    }

    // Require prefill batch: ncols1=8 needs Q->ne[1] >= 8. Decode (nTokensQ<8)
    // falls through to Phase B regardless of GQA ratio.
    if (nTokensQ < 8) {
        return false;
    }
    // V-only mode: K is raw f16, not TQ-packed. Phase D only handles TQ-packed K.
    // Without this guard, Phase D intercepts V-only prefill (v_packed=true,
    // nTokensQ>=8) and routes it to the K-only MMA kernel, which tries to
    // TQ-decode raw f16 K bytes => illegal memory access on sm_120.
    {
        int32_t k_bits;
        memcpy(&k_bits, (const int32_t *)dst->op_params + 2, sizeof(int32_t));
        if (k_bits == 0) {
            return false;
        }
    }
    // nKVHeads derivation:
    // K-only presets: V is f16, ggml_permute(0,2,1,3) applied → ne[2]=nKVHeads, ne[1]=nCells.
    // K+V presets: V is TQ-compressed (ne[2] != nKVHeads); use v_scales_t->ne[0]
    //   which is always shaped [nKVHeads, nCells] (see turboquant_compressed.go:79).
    GGML_ASSERT(V != nullptr && "Phase D dispatcher: V is required for nKVHeads");
    const int nKVHeads  = v_packed ? (int) v_scales_t->ne[0] : (int) V->ne[2];
    const int gqa_ratio = (int) (Q->ne[2] / nKVHeads);
    if (gqa_ratio == 0) {
        return false;
    }
    // Route to the highest ncols2 with an instantiated template.
    // gqa_ratio % 8 == 0 → ncols2=8 (Stage 6: llama3.1:70b, qwen2.5:72b).
    // gqa_ratio % 4 == 0 → ncols2=4 (Stage 5: llama3.2:3b, gemma3:1b, qwen3:8b).
    // gqa_ratio % 2 == 0 → ncols2=2 (Stage 7: gemma2:9b).
    // odd → ncols2=1 (Stage 3: qwen2.5:7b GQA=7, etc.).
    if ((gqa_ratio % 8) == 0) {
        if (D == 64) {
            tq_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 8>(ctx, dst);
        } else if (D == 128) {
            tq_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 8>(ctx, dst);
        } else {
            tq_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 8>(ctx, dst);
        }
        return true;
    }
    if ((gqa_ratio % 4) == 0) {
        if (D == 64) {
            tq_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 4>(ctx, dst);
        } else if (D == 128) {
            tq_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 4>(ctx, dst);
        } else {
            tq_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 4>(ctx, dst);
        }
        return true;
    }
    if ((gqa_ratio % 2) == 0) {
        if (D == 64) {
            tq_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 2>(ctx, dst);
        } else if (D == 128) {
            tq_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 2>(ctx, dst);
        } else {
            tq_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 2>(ctx, dst);
        }
        return true;
    }
    // ncols2=1: odd GQA ratio.
    if (D == 64) {
        tq_cuda_flash_attn_ext_mma_f16_case<64, 64, 8, 1>(ctx, dst);
    } else if (D == 128) {
        tq_cuda_flash_attn_ext_mma_f16_case<128, 128, 8, 1>(ctx, dst);
    } else {
        tq_cuda_flash_attn_ext_mma_f16_case<256, 256, 8, 1>(ctx, dst);
    }
    return true;
}

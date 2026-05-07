#include "tq-fattn.cuh"

// Pure router — no tq-fattn-vec.cuh included here, so no template instantiations.
// Each per-dim TU (tq-fattn-d{64,128,256,512}.cu) holds 16–18 instantiations
// × 10 architectures, staying under the gas single-object size limit (~2 GiB).
void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant fused flash attention requires compute capability 6.0+");

    const ggml_tensor * Q             = dst->src[0];
    const ggml_tensor * v_scales_t    = dst->src[6];
    const int D             = (int)Q->ne[0];
    const bool v_packed     = (v_scales_t != nullptr);
    int32_t outlier_count;
    memcpy(&outlier_count, (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    const bool has_outliers = (outlier_count > 0);

    GGML_ASSERT((D == 64 || D == 128 || D == 256 || D == 512) &&
                "TurboQuant fused kernel: unsupported head_dim (need 64, 128, 256, or 512)");

    // D=512: all combos (including K-only outlier) handled in tq-fattn-d512.cu.
    if (D == 512) {
        ggml_cuda_tq_flash_attn_ext_d512(ctx, dst);
        return;
    }

    // K-only outlier (V_PACKED=false, HAS_OUTLIERS=true) for D=64/128/256:
    // split across two TUs to stay under gas size limit.
    if (!v_packed && has_outliers) {
        if (D == 256) {
            ggml_cuda_tq_flash_attn_ext_konly_outlier_wide(ctx, dst);
        } else {
            ggml_cuda_tq_flash_attn_ext_konly_outlier(ctx, dst);
        }
        return;
    }

    // Remaining cases (K-only no-outlier, K+V fused ±outlier) per dim:
    if (D == 64) {
        ggml_cuda_tq_flash_attn_ext_d64(ctx, dst);
    } else if (D == 256) {
        ggml_cuda_tq_flash_attn_ext_d256(ctx, dst);
    } else {
        ggml_cuda_tq_flash_attn_ext_d128(ctx, dst);
    }
}

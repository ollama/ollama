#include "tq-fattn-ctx.cuh"

// D=64 flash-attention (K-only and K+V fused, no outliers or V_PACKED+outliers).
// 18 template instantiations × 10 archs — split from tq-fattn.cu to stay under
// the gas single-object size limit.
void ggml_cuda_tq_flash_attn_ext_d64(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    TQFattnCtx c = tq_fattn_extract(dst, 64);
    const int ncols = (c.nTokensQ == 1) ? 1 : (c.nTokensQ < 8) ? 2 : 8;

    #define DISPATCH_NCOLS(NCOLS) \
        if (c.logit_softcap == 0.0f) { TQ_DISPATCH_DIM(c, 64, NCOLS, false); } \
        else                         { TQ_DISPATCH_DIM(c, 64, NCOLS, true);  }

    if      (ncols == 1) { DISPATCH_NCOLS(1); }
    else if (ncols == 2) { DISPATCH_NCOLS(2); }
    else                 { DISPATCH_NCOLS(8); }
    #undef DISPATCH_NCOLS

    CUDA_CHECK(cudaGetLastError());
}

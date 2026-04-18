#include "tq-fattn.cuh"
#include "tq-fattn-vec.cuh"

// Launch the TQ fused flash-attention kernel for a given (D, ncols, use_logit_softcap, V_PACKED).
template<int D, int ncols, bool use_logit_softcap, bool V_PACKED>
static void tq_fattn_vec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                float scale, float logit_softcap,
                                int bits, int firstCell, int nCells, int nKVHeads, int packedBytes,
                                int v_bits, int v_packedBytes,
                                const float * v_scales_ptr, const float * v_codebook_ptr)
{
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K_p  = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * scales   = dst->src[4];
    const ggml_tensor * codebook = dst->src[5];

    GGML_ASSERT(Q->ne[0] == D);
    GGML_ASSERT(Q->type  == GGML_TYPE_F32);
    if constexpr (!V_PACKED) {
        GGML_ASSERT(V->type  == GGML_TYPE_F16);
    } else {
        GGML_ASSERT(V->type  == GGML_TYPE_I8);
    }

    const int nTokensQ = (int)Q->ne[1];
    const int nHeadsQ  = (int)Q->ne[2];
    const int nSeq     = (int)Q->ne[3];

    GGML_ASSERT(!mask || mask->type == GGML_TYPE_F16);

    const uint3 ne01 = init_fastdiv_values((uint64_t)nTokensQ);

    const int ntiles_x = (nTokensQ + ncols - 1) / ncols;
    dim3 blocks(ntiles_x, 1, nHeadsQ * nSeq);
    dim3 threads(WARP_SIZE, 4);

    // V strides: only used by !V_PACKED path; pass V strides for the f16 case.
    // For the V_PACKED case these are passed but ignored by the kernel.
    tq_flash_attn_ext_vec<D, ncols, use_logit_softcap, V_PACKED><<<blocks, threads, 0, ctx.stream()>>>(
        (const char    *)Q->data,
        (const uint8_t *)K_p->data,
        (const char    *)V->data,
        mask ? (const char *)mask->data : nullptr,
        (float *)dst->data,
        (const float *)scales->data,
        (const float *)codebook->data,
        scale, logit_softcap, bits, firstCell, nCells, nKVHeads, packedBytes,
        (int32_t)Q->ne[0],
        ne01,
        (int32_t)Q->ne[2],
        (int32_t)Q->ne[3],
        (int32_t)Q->nb[1],
        (int32_t)Q->nb[2],
        (int64_t)Q->nb[3],
        (int32_t)V->nb[1],
        (int32_t)V->nb[2],
        (int64_t)V->nb[3],
        mask ? (int32_t)mask->ne[0] : 0,   // nCells (mask row width)
        mask ? (int32_t)mask->nb[1] : 0,   // bytes between token rows
        v_scales_ptr, v_codebook_ptr, v_bits, v_packedBytes
    );
}

void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant fused flash attention requires compute capability 6.0+");
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * v_scales_t   = dst->src[6];  // NULL for K-only fused
    const ggml_tensor * v_codebook_t = dst->src[7];

    float   scale;
    float   logit_softcap;
    int32_t bits;
    int32_t firstCell;
    int32_t v_bits;
    memcpy(&scale,         (const float   *)dst->op_params + 0, sizeof(float));
    memcpy(&logit_softcap, (const float   *)dst->op_params + 1, sizeof(float));
    memcpy(&bits,          (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&firstCell,     (const int32_t *)dst->op_params + 3, sizeof(int32_t));
    memcpy(&v_bits,        (const int32_t *)dst->op_params + 4, sizeof(int32_t));

    const int D         = (int)Q->ne[0];
    const int nTokensQ  = (int)Q->ne[1];
    // nCells: when V_PACKED, V is the raw buffer so ne[1] is capacity not nCells.
    // Compute nCells from the mask if available, else from V geometry.
    const bool v_packed = (v_scales_t != nullptr);

    int nCells;
    if (v_packed) {
        // V is packed i8 buffer: nCells is not directly in V->ne[].
        // The mask always carries nCells; require mask when V is packed.
        GGML_ASSERT(dst->src[3] != nullptr && "TQ K+V fused requires mask to determine nCells");
        nCells = (int)dst->src[3]->ne[0];
    } else {
        nCells = (int)V->ne[1]; // after SDPA permute: [D, nCells, nKVHeads]
    }

    // packedBytes for K:
    const int packedBytes = (D * bits + 7) / 8;
    // packedBytes for V (computed before nKVHeads so we can derive nKVHeads from packed V dims):
    const int v_packedBytes = v_packed ? ((D * v_bits + 7) / 8) : 0;
    // nKVHeads: for K-only fused, V is a permuted f16 tensor with ne[2]=nKVHeads.
    // For K+V fused, V is the raw packed i8 buffer with ne[0]=v_packedBytes*nKVHeads.
    const int nKVHeads  = v_packed ? (int)V->ne[0] / v_packedBytes : (int)V->ne[2];

    const float * v_scales_ptr   = v_scales_t   ? (const float *)v_scales_t->data   : nullptr;
    const float * v_codebook_ptr = v_codebook_t ? (const float *)v_codebook_t->data : nullptr;

    GGML_ASSERT(D == 128); // Phase 1: head_dim=128 only

    if (logit_softcap != 0.0f) { scale /= logit_softcap; }

    const int ncols = (nTokensQ == 1) ? 1 : 2;

    #define DISPATCH(NCOLS, SOFTCAP) \
        if (v_packed) { \
            tq_fattn_vec_launch<128, NCOLS, SOFTCAP, true>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                v_bits, v_packedBytes, v_scales_ptr, v_codebook_ptr); \
        } else { \
            tq_fattn_vec_launch<128, NCOLS, SOFTCAP, false>(ctx, dst, scale, logit_softcap, \
                bits, firstCell, nCells, nKVHeads, packedBytes, \
                0, 0, nullptr, nullptr); \
        }

    if (ncols == 1) {
        if (logit_softcap == 0.0f) { DISPATCH(1, false); }
        else                       { DISPATCH(1, true);  }
    } else {
        if (logit_softcap == 0.0f) { DISPATCH(2, false); }
        else                       { DISPATCH(2, true);  }
    }
    #undef DISPATCH

    CUDA_CHECK(cudaGetLastError());
}

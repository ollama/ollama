#include "tq-fattn.cuh"
#include "tq-fattn-vonly.cuh"  // V-only launcher declarations (extern "C", no templates)

// Pure router — no tq-fattn-vec.cuh included here, so no template instantiations.
// Each per-dim TU (tq-fattn-d{64,128,256,512}.cu) holds 16–18 instantiations
// × 10 architectures, staying under the gas single-object size limit (~2 GiB).
void ggml_cuda_tq_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_info().devices[ctx.device].cc >= 600 &&
                "TurboQuant fused flash attention requires compute capability 6.0+");

    // Phase D wedge: Turing+, K-only, prefill, narrow ncols1×ncols2 slice.
    // Returns false if Phase D declines; fall through to the inline-decode
    // dispatch below.
    if (ggml_cuda_tq_flash_attn_ext_mma_d(ctx, dst)) {
        return;
    }

    const ggml_tensor * Q             = dst->src[0];
    const ggml_tensor * K             = dst->src[1];
    const ggml_tensor * V             = dst->src[2];
    const ggml_tensor * v_scales_t    = dst->src[6];
    const ggml_tensor * v_codebook_t  = dst->src[7];
    const ggml_tensor * mask_t        = dst->src[3];
    const int D             = (int)Q->ne[0];
    const bool v_packed     = (v_scales_t != nullptr);
    int32_t k_bits, outlier_count;
    memcpy(&k_bits,        (const int32_t *)dst->op_params + 2, sizeof(int32_t));
    memcpy(&outlier_count, (const int32_t *)dst->op_params + 8, sizeof(int32_t));
    const bool has_outliers = (outlier_count > 0);

    // V-only fused: bits=0 sentinel (K is raw f16) + v_packed (V is TQ-packed).
    // Route to the per-D V-only kernel (tq-fattn-vonly.cu); do not fall
    // through to the K+V inline-decode per-dim dispatch below.
    if (v_packed && k_bits == 0) {
        GGML_ASSERT((D == 64 || D == 128 || D == 256 || D == 512) &&
                    "TQ V-only fused kernel: unsupported head_dim (need 64, 128, 256, or 512)");

        // Extract op_params.
        float scale, logit_softcap;
        int32_t firstCell, v_bits;
        memcpy(&scale,         (const float   *)dst->op_params + 0, sizeof(float));
        memcpy(&logit_softcap, (const float   *)dst->op_params + 1, sizeof(float));
        memcpy(&firstCell,     (const int32_t *)dst->op_params + 3, sizeof(int32_t));
        memcpy(&v_bits,        (const int32_t *)dst->op_params + 4, sizeof(int32_t));

        // Scale is pre-divided by logit_softcap so the kernel computes
        // logit_softcap * tanh(scale * Q·K) = logit_softcap * tanh(Q·K / softcap).
        if (logit_softcap != 0.0f) scale /= logit_softcap;

        // Byte strides for Q and K (ggml strides are always in bytes).
        // Q is permuted (0,2,1,3) by Go from [D, nHeadsQ, ncols, nSeq]
        //   → [D, ncols, nHeadsQ, nSeq]: nb[1]=ncols stride, nb[2]=nHeadsQ stride
        // K is raw f16 from the inner Causal cache, NOT permuted: [D, nKVHeads, nCells, nSeq]
        //   nb[1]=nKVHeads stride, nb[2]=nCells stride
        const int32_t nb_q01 = (int32_t)Q->nb[1]; // ncols byte stride
        const int32_t nb_q02 = (int32_t)Q->nb[2]; // nHeadsQ byte stride
        const int64_t nb_q03 = (int64_t)Q->nb[3]; // nSeq byte stride
        const int32_t nb_k11 = (int32_t)K->nb[2]; // nCells byte stride
        const int32_t nb_k12 = (int32_t)K->nb[1]; // nKVHeads byte stride
        const int64_t nb_k13 = (int64_t)K->nb[3]; // nSeq byte stride

        const int nHeadsQ  = (int)Q->ne[2]; // after permute(0,2,1,3), ne[2] is nHeadsQ
        const int ncols    = (int)Q->ne[1]; // after permute(0,2,1,3), ne[1] is ncols
        const int nKVHeads = (int)K->ne[1];
        const int nCells   = (int)K->ne[2];
        const int nSeq     = (int)Q->ne[3];
        const int v_packedBytes = (D * v_bits + 7) / 8;

        const ggml_tensor * locs_t = dst->src[11]; // NULL = contiguous; [nCells] i32 = physical slots
        // Indexed mode: K comes from DequantKAt (dense, firstCell=0); V uses locs[] for physical slot.
        const int32_t * locs_ptr = locs_t ? (const int32_t *)locs_t->data : nullptr;
        if (locs_ptr) firstCell = 0; // DequantKAt output is dense starting at slot 0

        const float   * Q_ptr = (const float   *)Q->data;
        const __half  * K_ptr = (const __half  *)K->data;
        const uint8_t * V_ptr = (const uint8_t *)V->data;
        float         * dst_ptr = (float *)dst->data;
        const float   * v_scales_ptr   = (const float *)v_scales_t->data;
        const float   * v_codebook_ptr = (const float *)v_codebook_t->data;
        const __half  * mask_ptr  = mask_t ? (const __half *)mask_t->data : nullptr;
        const int32_t   mask_ne0  = mask_t ? (int32_t)mask_t->ne[0] : 0;
        const int32_t   mask_nb1  = mask_t ? (int32_t)mask_t->nb[1] : 0;

        // Flash-decode splitting: parallelize across the KV cell dimension.
        // Each split handles ≥512 cells (4 nthreads passes) to amortise the
        // reduce-kernel overhead; short sequences stay at nSplits=1.
        // ROCm (GDDR6 memory-bound): splitting adds overhead with no gain —
        // benchmarks show 0% throughput improvement at ctx=8192 on gfx1102.
#ifdef GGML_USE_HIP
        const int nSplits = 1;
#else
        const int nSplits = max(1, min(16, nCells / 512));
#endif
        const int nTiles  = nSeq * nHeadsQ * ncols;
        ggml_cuda_pool_alloc<float> partial_alloc(ctx.pool());
        float * partial_buf = nullptr;
        if (nSplits > 1) {
            partial_buf = partial_alloc.alloc((size_t)nSplits * nTiles * (D + 2));
        }

        cudaStream_t stream = ctx.stream();
        if (D == 64) {
            ggml_cuda_tq_fattn_vec_vonly_d64(stream,
                Q_ptr, nb_q01, nb_q02, nb_q03,
                K_ptr, nb_k11, nb_k12, nb_k13,
                V_ptr, v_packedBytes,
                mask_ptr, mask_ne0, mask_nb1,
                dst_ptr, partial_buf, v_scales_ptr, v_codebook_ptr, locs_ptr,
                v_bits, scale, logit_softcap,
                firstCell, nCells, nSplits, nHeadsQ, nKVHeads, nSeq, ncols);
        } else if (D == 128) {
            ggml_cuda_tq_fattn_vec_vonly_d128(stream,
                Q_ptr, nb_q01, nb_q02, nb_q03,
                K_ptr, nb_k11, nb_k12, nb_k13,
                V_ptr, v_packedBytes,
                mask_ptr, mask_ne0, mask_nb1,
                dst_ptr, partial_buf, v_scales_ptr, v_codebook_ptr, locs_ptr,
                v_bits, scale, logit_softcap,
                firstCell, nCells, nSplits, nHeadsQ, nKVHeads, nSeq, ncols);
        } else if (D == 256) {
            ggml_cuda_tq_fattn_vec_vonly_d256(stream,
                Q_ptr, nb_q01, nb_q02, nb_q03,
                K_ptr, nb_k11, nb_k12, nb_k13,
                V_ptr, v_packedBytes,
                mask_ptr, mask_ne0, mask_nb1,
                dst_ptr, partial_buf, v_scales_ptr, v_codebook_ptr, locs_ptr,
                v_bits, scale, logit_softcap,
                firstCell, nCells, nSplits, nHeadsQ, nKVHeads, nSeq, ncols);
        } else {
            ggml_cuda_tq_fattn_vec_vonly_d512(stream,
                Q_ptr, nb_q01, nb_q02, nb_q03,
                K_ptr, nb_k11, nb_k12, nb_k13,
                V_ptr, v_packedBytes,
                mask_ptr, mask_ne0, mask_nb1,
                dst_ptr, partial_buf, v_scales_ptr, v_codebook_ptr, locs_ptr,
                v_bits, scale, logit_softcap,
                firstCell, nCells, nSplits, nHeadsQ, nKVHeads, nSeq, ncols);
        }
        return;
    }

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

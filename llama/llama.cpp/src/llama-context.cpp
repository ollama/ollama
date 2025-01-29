#include "llama-context.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>

void llama_set_k_shift(struct llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_K_shift->buffer));

    int32_t * data = (int32_t *) lctx.inp_K_shift->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].delta;
    }
}

void llama_set_s_copy(struct llama_context & lctx) {
    const int64_t kv_size = lctx.kv_self.size;

    assert(ggml_backend_buffer_is_host(lctx.inp_s_copy->buffer));

    int32_t * data = (int32_t *) lctx.inp_s_copy->data;

    for (int i = 0; i < kv_size; ++i) {
        data[i] = lctx.kv_self.cells[i].src;
    }
}

// llama input

static int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional) {
    // TODO move to hparams if a T5 variant appears that uses a different value
    const int64_t max_distance = 128;

    if (bidirectional) {
        n_buckets >>= 1;
    }

    const int64_t max_exact = n_buckets >> 1;

    int32_t relative_position = x - y;
    int32_t relative_bucket = 0;
    if (bidirectional) {
        relative_bucket += (relative_position > 0) * n_buckets;
        relative_position = abs(relative_position);
    } else {
        relative_position = -std::min<int32_t>(relative_position, 0);
    }
    int32_t relative_position_if_large = floorf(max_exact + logf(1.0 * relative_position / max_exact) * (n_buckets - max_exact) / log(1.0 * max_distance / max_exact));
    relative_position_if_large = std::min<int32_t>(relative_position_if_large, n_buckets - 1);
    relative_bucket += (relative_position < max_exact ? relative_position : relative_position_if_large);
    return relative_bucket;
}

void llama_set_inputs(llama_context & lctx, const llama_ubatch & ubatch) {
    //
    // set input data
    //

    const auto & hparams = lctx.model.hparams;
    const auto & cparams = lctx.cparams;
    const auto & kv_self = lctx.kv_self;

    if (ubatch.token) {
        const int64_t n_tokens = ubatch.n_tokens;

        ggml_backend_tensor_set(lctx.inp_tokens, ubatch.token, 0, n_tokens*ggml_element_size(lctx.inp_tokens));
    }

    if (ubatch.embd) {
        if (lctx.inp_cross_attn_state && lctx.inp_cross_attn_state->buffer) {
            ggml_backend_tensor_set(lctx.inp_cross_attn_state, ubatch.embd, 0, ggml_nbytes(lctx.inp_cross_attn_state));
            // zero out inp_embd since it's not used
            float * inp_embd_data = (float *)lctx.inp_embd->data;
            for (int i = 0; i < ggml_nelements(lctx.inp_embd); ++i) {
                inp_embd_data[i] = 0.0f;
            }
        } else {
            const int64_t n_embd   = hparams.n_embd;
            const int64_t n_tokens = ubatch.n_tokens;

            ggml_backend_tensor_set(lctx.inp_embd, ubatch.embd, 0, n_tokens*n_embd*ggml_element_size(lctx.inp_embd));
        }
    }

    if (ubatch.pos && lctx.inp_pos) {
        const int64_t n_tokens = ubatch.n_tokens;
        auto n_pos = lctx.n_pos_per_token;
        ggml_backend_tensor_set(lctx.inp_pos, ubatch.pos, 0, n_tokens*n_pos*ggml_element_size(lctx.inp_pos));
    }

    if (hparams.causal_attn || cparams.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        //GGML_ASSERT(lctx.inp_out_ids && "every model that can must skip unused outputs");

        if (!lctx.inp_out_ids) {
            LLAMA_LOG_WARN("%s: 'lctx.inp_out_ids' is not created\n", __func__);
        } else {
            const int64_t n_tokens = ubatch.n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_out_ids->buffer));
            int32_t * data = (int32_t *) lctx.inp_out_ids->data;

            if (lctx.n_outputs == n_tokens) {
                for (int i = 0; i < n_tokens; ++i) {
                    data[i] = i;
                }
            } else if (ubatch.output) {
                int32_t n_outputs = 0;
                for (int i = 0; i < n_tokens; ++i) {
                    if (ubatch.output[i]) {
                        data[n_outputs++] = i;
                    }
                }
                // the graph needs to have been passed the correct number of outputs
                GGML_ASSERT(lctx.n_outputs == n_outputs);
            } else if (lctx.n_outputs == 1) {
                // only keep last output
                data[0] = n_tokens - 1;
            } else {
                GGML_ASSERT(lctx.n_outputs == 0);
            }
        }
    }

    GGML_ASSERT(
        // (!a || b) is a logical implication (a -> b)
        // !hparams.causal_attn -> !cparams.causal_attn
        (hparams.causal_attn || !cparams.causal_attn) &&
        "causal attention is not supported by this model"
    );

    if (lctx.inp_KQ_mask || lctx.inp_KQ_mask_swa) {
        // NOTE: hparams.causal_attn indicates the model is capable of generation and uses the kv cache.
        if (cparams.causal_attn && !lctx.is_encoding) {
            const int64_t n_kv         = kv_self.n;
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;


            float * data     = nullptr;
            float * data_swa = nullptr;

            if (lctx.inp_KQ_mask) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));
                data = (float *) lctx.inp_KQ_mask->data;
            }

            if (lctx.inp_KQ_mask_swa) {
                GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_swa->buffer));
                data_swa = (float *) lctx.inp_KQ_mask_swa->data;
            }

            // For causal attention, use only the previous KV cells
            // of the correct sequence for each token of the ubatch.
            // It's assumed that if a token in the batch has multiple sequences, they are equivalent.
            for (int h = 0; h < 1; ++h) {
                for (int s = 0; s < n_seqs; ++s) {
                    const llama_seq_id seq_id = ubatch.seq_id[s][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const llama_pos pos = ubatch.pos[s*n_seq_tokens + j];

                        for (int i = 0; i < n_kv; ++i) {
                            float f;
                            if (!kv_self.cells[i].has_seq_id(seq_id) || kv_self.cells[i].pos > pos) {
                                f = -INFINITY;
                            } else {
                                if (hparams.use_alibi) {
                                    f = -std::abs(kv_self.cells[i].pos - pos);
                                } else {
                                    f = 0.0f;
                                }
                            }

                            if (data) {
                                data[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }

                            // may need to cut off old tokens for sliding window
                            if (data_swa) {
                                if (pos - kv_self.cells[i].pos >= (int32_t)hparams.n_swa) {
                                    f = -INFINITY;
                                }
                                data_swa[h*(n_kv*n_tokens) + s*(n_kv*n_seq_tokens) + j*n_kv + i] = f;
                            }
                        }
                    }
                }

                if (data) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }

                if (data_swa) {
                    for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                        for (int j = 0; j < n_kv; ++j) {
                            data_swa[h*(n_kv*n_tokens) + i*n_kv + j] = -INFINITY;
                        }
                    }
                }
            }
        } else {
            const int64_t n_tokens     = ubatch.n_tokens;
            const int64_t n_seq_tokens = ubatch.n_seq_tokens;
            const int64_t n_seqs       = ubatch.n_seqs;
            // when using kv cache, the mask needs to match the kv cache size
            const int64_t n_stride = hparams.causal_attn && !lctx.is_encoding ? kv_self.n : n_tokens;

            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask->buffer));

            float * data = (float *) lctx.inp_KQ_mask->data;

            for (int h = 0; h < 1; ++h) {
                for (int s1 = 0; s1 < n_seqs; ++s1) {
                    const llama_seq_id seq_id = ubatch.seq_id[s1][0];

                    for (int j = 0; j < n_seq_tokens; ++j) {
                        const int32_t tj = s1*n_seq_tokens + j;

                        for (int s0 = 0; s0 < n_seqs; ++s0) {
                            for (int i = 0; i < n_seq_tokens; ++i) {
                                const int32_t ti = s0*n_seq_tokens + i;
                                float f = -INFINITY;

                                for (int s = 0; s < ubatch.n_seq_id[s0]; ++s) {
                                    if (ubatch.seq_id[s0][s] == seq_id) {
                                        if (hparams.use_alibi) {
                                            f = -std::abs(ubatch.pos[ti] - ubatch.pos[tj]);
                                        } else {
                                            f = 0.0f;
                                        }
                                        break;
                                    }
                                }

                                data[h*(n_tokens*n_tokens) + tj*n_stride + ti] = f;
                            }
                        }

                        for (int i = n_tokens; i < n_stride; ++i) {
                            data[h*(n_tokens*n_tokens) + tj*n_stride + i] = -INFINITY;
                        }
                    }
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_MEAN) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(lctx.inp_mean);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_mean->buffer));

        float * data = (float *) lctx.inp_mean->data;
        memset(lctx.inp_mean->data, 0, n_tokens * n_tokens * ggml_element_size(lctx.inp_mean));

        std::vector<uint64_t> sum(n_tokens, 0);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == MEAN");

            sum[seq_id] += ubatch.n_seq_tokens;
        }

        std::vector<float> div(n_tokens, 0.0f);
        for (int i = 0; i < n_tokens; ++i) {
            const uint64_t s = sum[i];
            if (s > 0) {
                div[i] = 1.0f/float(s);
            }
        }

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            for (int i = 0; i < n_seq_tokens; ++i) {
                data[seq_id*n_tokens + s*n_seq_tokens + i] = div[seq_id];
            }
        }
    }

    if (cparams.embeddings && (
                cparams.pooling_type == LLAMA_POOLING_TYPE_CLS ||
                cparams.pooling_type == LLAMA_POOLING_TYPE_RANK)) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(lctx.inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_cls->buffer));

        uint32_t * data = (uint32_t *) lctx.inp_cls->data;
        memset(lctx.inp_cls->data, 0, n_tokens * ggml_element_size(lctx.inp_cls));

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == CLS or RANK");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos == 0) {
                    data[seq_id] = s*n_seq_tokens + i;
                }
            }
        }
    }

    if (cparams.embeddings && cparams.pooling_type == LLAMA_POOLING_TYPE_LAST) {
        const int64_t n_tokens     = ubatch.n_tokens;
        const int64_t n_seq_tokens = ubatch.n_seq_tokens;
        const int64_t n_seqs       = ubatch.n_seqs;

        GGML_ASSERT(lctx.inp_cls);
        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_cls->buffer));

        uint32_t * data = (uint32_t *) lctx.inp_cls->data;
        memset(lctx.inp_cls->data, 0, n_tokens * ggml_element_size(lctx.inp_cls));

        std::vector<int> last_pos(n_tokens, -1);
        std::vector<int> last_row(n_tokens, -1);

        for (int s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s][0];

            // TODO: adapt limits to n_seqs when ubatch.equal_seqs is true
            GGML_ASSERT(seq_id < n_tokens && "seq_id cannot be larger than n_tokens with pooling_type == LAST");

            for (int i = 0; i < n_seq_tokens; ++i) {
                const llama_pos pos = ubatch.pos[s*n_seq_tokens + i];

                if (pos >= last_pos[seq_id]) {
                    last_pos[seq_id] = pos;
                    last_row[seq_id] = s*n_seq_tokens + i;
                }
            }
        }

        for (int i = 0; i < n_tokens; ++i) {
            if (last_row[i] >= 0) {
                data[i] = last_row[i];
            }
        }
    }

    if (kv_self.recurrent) {
        const int64_t n_kv = kv_self.n;

        if (lctx.inp_s_mask) {
            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_mask->buffer));
            float * data = (float *) lctx.inp_s_mask->data;

            // clear unused states
            for (int i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = lctx.kv_self.cells[cell_id];

                data[i] = (float) (kv_cell.src >= 0);

                // only clear once
                if (kv_cell.src < 0) {
                    kv_cell.src = cell_id;
                }
            }
        }

        if (lctx.inp_s_copy) {
            GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_s_copy->buffer));
            int32_t * data = (int32_t *) lctx.inp_s_copy->data;

            // assuming copy destinations ALWAYS happen ONLY on the cells between head and head+n
            for (uint32_t i = 0; i < n_kv; ++i) {
                const uint32_t  cell_id = i + kv_self.head;
                llama_kv_cell & kv_cell = lctx.kv_self.cells[cell_id];

                // prevent out-of-bound sources
                if (kv_cell.src < 0 || (uint32_t) kv_cell.src >= kv_self.size) {
                    kv_cell.src = cell_id;
                }

                data[i] = kv_cell.src;

                // ensure copy only happens once
                if (kv_cell.src != (int32_t) cell_id) {
                    kv_cell.src = cell_id;
                }
            }
        }
    }

    if (lctx.inp_pos_bucket) {
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_pos_bucket->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        int32_t * data = (int32_t *) lctx.inp_pos_bucket->data;

        if (!lctx.is_encoding) {
            const int64_t n_kv = kv_self.n;
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_kv; ++i) {
                        data[h*(n_kv*n_tokens) + j*n_kv + i] = llama_relative_position_bucket(lctx.kv_self.cells[i].pos, ubatch.pos[j], hparams.n_rel_attn_bkts, lctx.is_encoding);
                    }
                }
            }
        } else {
            for (int h = 0; h < 1; ++h) {
                for (int j = 0; j < n_tokens; ++j) {
                    for (int i = 0; i < n_tokens; ++i) {
                        data[h*(n_tokens*n_tokens) + j*n_tokens + i] = llama_relative_position_bucket(ubatch.pos[i], ubatch.pos[j], hparams.n_rel_attn_bkts, lctx.is_encoding);
                    }
                }
            }
        }
    }

    if (!lctx.is_encoding && lctx.inp_embd_enc) {
        assert(lctx.inp_embd_enc->type == GGML_TYPE_F32);
        assert((size_t) ggml_nelements(lctx.inp_embd_enc) == lctx.embd_enc.size());

        ggml_backend_tensor_set(lctx.inp_embd_enc, lctx.embd_enc.data(), 0, ggml_nbytes(lctx.inp_embd_enc));
    }

    if (!lctx.is_encoding && lctx.inp_KQ_mask_cross) {
        const int64_t n_output_enc = lctx.embd_enc.size() / hparams.n_embd;
        const int64_t n_tokens = ubatch.n_tokens;

        GGML_ASSERT(ggml_backend_buffer_is_host(lctx.inp_KQ_mask_cross->buffer));
        GGML_ASSERT(!ubatch.equal_seqs); // TODO: use ubatch.n_seqs instead of failing

        float * data = (float *) lctx.inp_KQ_mask_cross->data;

        for (int h = 0; h < 1; ++h) {
            for (int j = 0; j < n_tokens; ++j) {
                for (int i = 0; i < n_output_enc; ++i) {
                    float f = -INFINITY;
                    for (int s = 0; s < ubatch.n_seq_id[j]; ++s) {
                        const llama_seq_id seq_id = ubatch.seq_id[j][s];
                        if (lctx.seq_ids_enc[i].find(seq_id) != lctx.seq_ids_enc[i].end()) {
                            f = 0.0f;
                        }
                    }
                    data[h*(n_output_enc*n_tokens) + j*n_output_enc + i] = f;
                }
            }

            for (int i = n_tokens; i < GGML_PAD(n_tokens, GGML_KQ_MASK_PAD); ++i) {
                for (int j = 0; j < n_output_enc; ++j) {
                    data[h*(n_output_enc*n_tokens) + i*n_output_enc + j] = -INFINITY;
                }
            }
        }
    }
}

// llama output

size_t llama_output_reserve(struct llama_context & lctx, size_t n_outputs) {
    const auto & cparams = lctx.cparams;
    const auto & hparams = lctx.model.hparams;

    const size_t n_outputs_max = std::max(n_outputs, (size_t) cparams.n_seq_max);

    const auto n_batch = cparams.n_batch;
    const auto n_vocab = hparams.n_vocab;
    const auto n_embd  = hparams.n_embd;

    // TODO: use a per-batch flag for logits presence instead
    const bool has_logits =  cparams.causal_attn;
    const bool has_embd   =  cparams.embeddings && (cparams.pooling_type == LLAMA_POOLING_TYPE_NONE);

    const size_t logits_size = has_logits ? n_vocab*n_outputs_max : 0;
    const size_t embd_size   = has_embd   ?  n_embd*n_outputs_max : 0;

    if (lctx.output_ids.empty()) {
        // init, never resized afterwards
        lctx.output_ids.resize(n_batch);
    }

    const size_t prev_size = lctx.buf_output ? ggml_backend_buffer_get_size(lctx.buf_output.get()) : 0;
    const size_t new_size  = (logits_size + embd_size) * sizeof(float);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!lctx.buf_output || prev_size < new_size) {
        if (lctx.buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_INFO("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            lctx.buf_output = nullptr;
            lctx.logits = nullptr;
            lctx.embd = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = lctx.model.dev_output.dev;
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        lctx.buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (lctx.buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(lctx.buf_output.get());

    lctx.logits = has_logits ? output_base               : nullptr;
    lctx.embd   = has_embd   ? output_base + logits_size : nullptr;

    lctx.output_size = n_outputs_max;
    lctx.logits_size = logits_size;
    lctx.embd_size   = embd_size;

    // set all ids as invalid (negative)
    std::fill(lctx.output_ids.begin(), lctx.output_ids.end(), -1);

    ggml_backend_buffer_clear(lctx.buf_output.get(), 0);

    lctx.n_outputs = 0;

    return n_outputs_max;
}

void llama_output_reorder(struct llama_context & ctx) {
    std::vector<size_t> & out_ids = ctx.sbatch.out_ids;
    if (!out_ids.empty()) {
        const uint32_t n_vocab = ctx.model.hparams.n_vocab;
        const uint32_t n_embd  = ctx.model.hparams.n_embd;

        const int32_t n_outputs = ctx.n_outputs;
        GGML_ASSERT((size_t) n_outputs == out_ids.size());

        // TODO: is there something more efficient which also minimizes swaps?
        // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
        for (int32_t i = 0; i < n_outputs - 1; ++i) {
            int32_t j_min = i;
            for (int32_t j = i + 1; j < n_outputs; ++j) {
                if (out_ids[j] < out_ids[j_min]) {
                    j_min = j;
                }
            }
            if (j_min == i) { continue; }
            std::swap(out_ids[i], out_ids[j_min]);
            if (ctx.logits_size > 0) {
                for (uint32_t k = 0; k < n_vocab; k++) {
                    std::swap(ctx.logits[i*n_vocab + k], ctx.logits[j_min*n_vocab + k]);
                }
            }
            if (ctx.embd_size > 0) {
                for (uint32_t k = 0; k < n_embd; k++) {
                    std::swap(ctx.embd[i*n_embd + k], ctx.embd[j_min*n_embd + k]);
                }
            }
        }
        std::fill(ctx.output_ids.begin(), ctx.output_ids.end(), -1);
        for (int32_t i = 0; i < n_outputs; ++i) {
            ctx.output_ids[out_ids[i]] = i;
        }
        out_ids.clear();
    }
}

//
// interface implementation
//

void llama_free(struct llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const struct llama_context * ctx) {
    return ctx->cparams.n_ctx;
}

uint32_t llama_n_batch(const struct llama_context * ctx) {
    return ctx->cparams.n_batch;
}

uint32_t llama_n_ubatch(const struct llama_context * ctx) {
    return ctx->cparams.n_ubatch;
}

uint32_t llama_n_seq_max(const struct llama_context * ctx) {
    return ctx->kv_self.size;
}

const struct llama_model * llama_get_model(const struct llama_context * ctx) {
    return &ctx->model;
}

enum llama_pooling_type llama_pooling_type(const struct llama_context * ctx) {
    return ctx->cparams.pooling_type;
}

void llama_attach_threadpool(
             struct llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->threadpool       = threadpool;
    ctx->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_detach_threadpool(struct llama_context * ctx) {
    ctx->threadpool       = nullptr;
    ctx->threadpool_batch = nullptr;
}

void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->cparams.n_threads       = n_threads;
    ctx->cparams.n_threads_batch = n_threads_batch;
}

int32_t llama_n_threads(struct llama_context * ctx) {
    return ctx->cparams.n_threads;
}

int32_t llama_n_threads_batch(struct llama_context * ctx) {
    return ctx->cparams.n_threads_batch;
}

void llama_set_abort_callback(struct llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->abort_callback      = abort_callback;
    ctx->abort_callback_data = abort_callback_data;

    for (auto & backend : ctx->backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
        if (set_abort_callback_fn) {
            set_abort_callback_fn(backend.get(), ctx->abort_callback, ctx->abort_callback_data);
        }
    }
}

void llama_set_embeddings(struct llama_context * ctx, bool embeddings) {
    ctx->cparams.embeddings = embeddings;
}

void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn) {
    ctx->cparams.causal_attn = causal_attn;
}

void llama_set_cross_attention(struct llama_context * ctx, bool cross_attention) {
    ctx->cparams.cross_attn = cross_attention;
}

void llama_synchronize(struct llama_context * ctx) {
    ggml_backend_sched_synchronize(ctx->sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (ctx->n_queued_tokens == 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_eval++;
    } else if (ctx->n_queued_tokens > 1) {
        if (!ctx->cparams.no_perf) {
            ctx->t_p_eval_us += ggml_time_us() - ctx->t_compute_start_us;
        }
        ctx->n_p_eval += ctx->n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (ctx->n_queued_tokens > 0 && !ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }

    ctx->n_queued_tokens = 0;
    ctx->t_compute_start_us = 0;
}

float * llama_get_logits(struct llama_context * ctx) {
    llama_synchronize(ctx);

    // reorder logits for backward compatibility
    // TODO: maybe deprecate this
    llama_output_reorder(*ctx);

    return ctx->logits;
}

float * llama_get_logits_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;

    llama_synchronize(ctx);

    try {
        if (ctx->logits == nullptr) {
            throw std::runtime_error("no logits");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->logits + j*ctx->model.hparams.n_vocab;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_get_embeddings(struct llama_context * ctx) {
    llama_synchronize(ctx);

    // reorder embeddings for backward compatibility
    // TODO: maybe deprecate this
    llama_output_reorder(*ctx);

    return ctx->embd;
}

float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i) {
    int32_t j = -1;

    llama_synchronize(ctx);

    try {
        if (ctx->embd == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        if (i < 0) {
            j = ctx->n_outputs + i;
            if (j < 0) {
                throw std::runtime_error(format("negative index out of range [0, %d)", ctx->n_outputs));
            }
        } else if ((size_t) i >= ctx->output_ids.size()) {
            throw std::runtime_error(format("out of range [0, %zu)", ctx->output_ids.size()));
        } else {
            j = ctx->output_ids[i];
        }

        if (j < 0) {
            throw std::runtime_error(format("batch.logits[%d] != true", i));
        }
        if (j >= ctx->n_outputs) {
            // This should not happen
            throw std::runtime_error(format("corrupt output buffer (j=%d, n_outputs=%d)", j, ctx->n_outputs));
        }

        return ctx->embd + j*ctx->model.hparams.n_embd;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    auto it = ctx->embd_seq.find(seq_id);
    if (it == ctx->embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

// llama state API

// deprecated
size_t llama_get_state_size(struct llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(struct llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(struct llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// TODO: replace all non-fatal assertions with returned errors or exceptions
struct llama_data_write {
    virtual void write(const void * src, size_t size) = 0;
    virtual void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) = 0;
    virtual size_t get_size_written() = 0;
    virtual ~llama_data_write() = default;

    void write_string(const std::string & str) {
        uint32_t str_size = str.size();

        write(&str_size,  sizeof(str_size));
        write(str.data(), str_size);
    }

    void write_model_info(const struct llama_context * ctx) {
        const std::string arch_str = llm_arch_name(ctx->model.arch);
        write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    //void write_rng(const std::mt19937 & rng) {
    //    std::ostringstream rng_ss;
    //    rng_ss << rng;

    //    const std::string & rng_str = rng_ss.str();

    //    write_string(rng_str);
    //}

    void write_output_ids(struct llama_context * ctx) {
        llama_output_reorder(*ctx);

        const uint32_t n_outputs = ctx->n_outputs;

        std::vector<int32_t> output_pos;

        const size_t    n_batch = ctx->cparams.n_batch;
        const auto & output_ids = ctx->output_ids;

        GGML_ASSERT(n_outputs <= ctx->output_size);

        output_pos.resize(n_outputs);

        // build a more compact representation of the output ids
        for (size_t i = 0; i < n_batch; ++i) {
            // map an output id to a position in the batch
            int32_t pos = output_ids[i];
            if (pos >= 0) {
                GGML_ASSERT((uint32_t) pos < n_outputs);
                output_pos[pos] = i;
            }
        }

        write(&n_outputs, sizeof(n_outputs));

        if (n_outputs) {
            write(output_pos.data(), n_outputs * sizeof(int32_t));
        }
    }

    void write_logits(const struct llama_context * ctx) {
        const uint64_t logits_size = std::min((uint64_t) ctx->logits_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_vocab);

        write(&logits_size, sizeof(logits_size));

        if (logits_size) {
            write(ctx->logits, logits_size * sizeof(float));
        }
    }

    void write_embeddings(const struct llama_context * ctx) {
        const uint64_t embeddings_size = std::min((uint64_t) ctx->embd_size, (uint64_t) ctx->n_outputs * ctx->model.hparams.n_embd);

        write(&embeddings_size, sizeof(embeddings_size));

        if (embeddings_size) {
            write(ctx->embd, embeddings_size * sizeof(float));
        }
    }

    void write_kv_cache_meta(const llama_kv_cache & kv_self, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) {
        for (const auto & range : cell_ranges) {
            for (uint32_t i = range.first; i < range.second; ++i) {
                const auto & cell = kv_self.cells[i];
                const llama_pos pos      = cell.pos;
                const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

                write(&pos,      sizeof(pos));
                write(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id) {
                    for (auto seq_id : cell.seq_id) {
                        write(&seq_id, sizeof(seq_id));
                    }
                }
            }
        }
    }

    void write_kv_cache_data(const struct llama_context * ctx, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        const struct llama_hparams & hparams = ctx->model.hparams;

        const uint32_t v_trans = kv_self.v_trans ? 1 : 0;
        const uint32_t n_layer = hparams.n_layer;

        write(&v_trans, sizeof(v_trans));
        write(&n_layer, sizeof(n_layer));

        std::vector<uint8_t> tmp_buf;

        // Iterate and write all the keys first, each row is a cell
        // Get whole range at a time
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

            // Write key type
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            write(&k_type_i, sizeof(k_type_i));

            // Write row size of key
            const uint64_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
            write(&k_size_row, sizeof(k_size_row));

            // Read each range of cells of k_size length each into tmp_buf and write out
            for (const auto & range : cell_ranges) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * k_size_row;
                write_tensor_data(kv_self.k_l[il], range.first * k_size_row, buf_size);
            }
        }

        if (!kv_self.v_trans) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write row size of value
                const uint64_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                write(&v_size_row, sizeof(v_size_row));

                // Read each range of cells of v_size length each into tmp_buf and write out
                for (const auto & range : cell_ranges) {
                    const size_t range_size = range.second - range.first;
                    const size_t buf_size = range_size * v_size_row;
                    write_tensor_data(kv_self.v_l[il], range.first * v_size_row, buf_size);
                }
            }
        } else {
            // When v is transposed, we also need the element size and get the element ranges from each row
            const uint32_t kv_size = kv_self.size;
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Write value type
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                write(&v_type_i, sizeof(v_type_i));

                // Write element size
                const uint32_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                write(&v_size_el, sizeof(v_size_el));

                // Write GQA embedding size
                write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

                // For each row, we get the element values of each cell
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    // Read each range of cells of v_size_el length each into tmp_buf and write out
                    for (const auto & range : cell_ranges) {
                        const size_t range_size = range.second - range.first;
                        const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                        const size_t buf_size = range_size * v_size_el;
                        write_tensor_data(kv_self.v_l[il], src_offset, buf_size);
                    }
                }
            }
        }
    }

    void write_kv_cache(const struct llama_context * ctx, llama_seq_id seq_id = -1) {
        const struct llama_kv_cache & kv_self = ctx->kv_self;
        std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
        uint32_t cell_count = 0;

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = kv_self.size;
        for (uint32_t i = 0; i < kv_self.size; ++i) {
            const auto & cell = kv_self.cells[i];
            if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
                ++cell_count;
                if (cell_range_begin == kv_self.size) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != kv_self.size) {
                    cell_ranges.emplace_back(cell_range_begin, i);
                    cell_range_begin = kv_self.size;
                }
            }
        }
        if (cell_range_begin != kv_self.size) {
            cell_ranges.emplace_back(cell_range_begin, kv_self.size);
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cell_ranges) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        write(&cell_count, sizeof(cell_count));

        write_kv_cache_meta(kv_self, cell_ranges, seq_id);
        write_kv_cache_data(ctx, cell_ranges);
    }
};

struct llama_data_read {
    virtual const uint8_t * read(size_t size) = 0;
    virtual void read_to(void * dst, size_t size) = 0;
    virtual size_t get_size_read() = 0;
    virtual ~llama_data_read() = default;

    void read_string(std::string & str) {
        uint32_t str_size;
        read_to(&str_size, sizeof(str_size));

        str.assign((const char *) read(str_size), str_size);
    }

    // validate model information
    void read_model_info(const struct llama_context * ctx) {
        const std::string cur_arch_str = llm_arch_name(ctx->model.arch);

        std::string arch_str;
        read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    //void read_rng(std::mt19937 & rng) {
    //    std::string rng_str;
    //    read_string(rng_str);

    //    std::istringstream rng_ss(rng_str);
    //    rng_ss >> rng;

    //    if (rng_ss.fail()) {
    //        throw std::runtime_error("failed to load RNG state");
    //    }
    //}

    void read_output_ids(struct llama_context * ctx) {
        std::vector<int32_t> output_pos;

        uint32_t n_outputs;
        read_to(&n_outputs, sizeof(n_outputs));

        if (n_outputs > llama_output_reserve(*ctx, n_outputs)) {
            throw std::runtime_error("could not reserve outputs");
        }

        if (n_outputs) {
            output_pos.resize(n_outputs);
            read_to(output_pos.data(), n_outputs * sizeof(int32_t));

            for (int32_t i = 0; i < (int32_t) output_pos.size(); ++i) {
                int32_t id = output_pos[i];
                if ((uint32_t) id >= ctx->cparams.n_batch) {
                    throw std::runtime_error(format("invalid output id, %d does not fit in batch size of %u", id, ctx->cparams.n_batch));
                }
                ctx->output_ids[id] = i;
            }

            ctx->n_outputs = n_outputs;
        }
    }

    void read_logits(struct llama_context * ctx) {
        uint64_t logits_size;
        read_to(&logits_size, sizeof(logits_size));

        if (ctx->logits_size < logits_size) {
            throw std::runtime_error("logits buffer too small");
        }

        if (logits_size) {
            read_to(ctx->logits, logits_size * sizeof(float));
        }
    }

    void read_embeddings(struct llama_context * ctx) {
        uint64_t embeddings_size;
        read_to(&embeddings_size, sizeof(embeddings_size));

        if (ctx->embd_size < embeddings_size) {
            throw std::runtime_error("embeddings buffer too small");
        }

        if (embeddings_size) {
            read_to(ctx->embd, embeddings_size * sizeof(float));
        }
    }

    bool read_kv_cache_meta(struct llama_context * ctx, uint32_t cell_count, llama_seq_id dest_seq_id = -1) {
        struct llama_kv_cache & kv_self = ctx->kv_self;

        if (dest_seq_id != -1) {
            // single sequence

            llama_kv_cache_seq_rm(kv_self, dest_seq_id, -1, -1);

            llama_ubatch batch = ctx->sbatch.reserve_ubatch(cell_count, /* has_embd */ false);
            batch.n_tokens = cell_count;
            batch.n_seq_tokens = cell_count;
            batch.n_seqs = 1;

            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_pos pos;
                uint32_t n_seq_id;

                read_to(&pos, sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                if (n_seq_id != 0) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                    return false;
                }

                batch.pos[i] = pos;
            }
            batch.n_seq_id[0] = 1;
            batch.seq_id[0] = &dest_seq_id;
            if (!llama_kv_cache_find_slot(kv_self, batch)) {
                LLAMA_LOG_ERROR("%s: failed to find available cells in kv cache\n", __func__);
                return false;
            }

            // DEBUG CHECK: kv_self.head should be our first cell, kv_self.head + cell_count - 1 should be our last cell (verify seq_id and pos values)
            // Assume that this is one contiguous block of cells
            GGML_ASSERT(kv_self.head + cell_count <= kv_self.size);
            GGML_ASSERT(kv_self.cells[kv_self.head].pos == batch.pos[0]);
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].pos == batch.pos[cell_count - 1]);
            GGML_ASSERT(kv_self.cells[kv_self.head].has_seq_id(dest_seq_id));
            GGML_ASSERT(kv_self.cells[kv_self.head + cell_count - 1].has_seq_id(dest_seq_id));
        } else {
            // whole KV cache restore

            if (cell_count > kv_self.size) {
                LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
                return false;
            }

            llama_kv_cache_clear(kv_self);

            for (uint32_t i = 0; i < cell_count; ++i) {
                llama_kv_cell & cell = kv_self.cells[i];

                llama_pos pos;
                uint32_t  n_seq_id;

                read_to(&pos,      sizeof(pos));
                read_to(&n_seq_id, sizeof(n_seq_id));

                cell.pos = pos;

                for (uint32_t j = 0; j < n_seq_id; ++j) {
                    llama_seq_id seq_id;
                    read_to(&seq_id, sizeof(seq_id));

                    if (seq_id < 0 || (uint32_t) seq_id >= llama_n_seq_max(ctx)) {
                        LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, llama_n_seq_max(ctx));
                        return false;
                    }

                    cell.seq_id.insert(seq_id);

                    if (kv_self.recurrent) {
                        int32_t & tail = kv_self.cells[seq_id].tail;
                        if (tail != -1) {
                            LLAMA_LOG_ERROR("%s: duplicate tail for seq_id %d in cell %d and %d\n", __func__, seq_id, i, tail);
                            return false;
                        }
                        tail = i;
                    }
                }
            }

            kv_self.head = 0;
            kv_self.used = cell_count;
        }

        if (kv_self.recurrent) {
            for (uint32_t i = 0; i < cell_count; ++i) {
                uint32_t cell_id = kv_self.head + i;
                // make sure the recurrent states will keep their restored state
                kv_self.cells[cell_id].src = cell_id;
            }
        }

        return true;
    }

    bool read_kv_cache_data(struct llama_context * ctx, uint32_t cell_count) {
        const struct llama_hparams & hparams = ctx->model.hparams;
        struct llama_kv_cache & kv_self = ctx->kv_self;
        uint32_t v_trans;
        uint32_t n_layer;
        read_to(&v_trans, sizeof(v_trans));
        read_to(&n_layer, sizeof(n_layer));

        if (n_layer != hparams.n_layer) {
            LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer);
            return false;
        }
        if (cell_count > kv_self.size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, kv_self.size);
            return false;
        }
        if (kv_self.v_trans != (bool) v_trans) {
            LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
            return false;
        }

        // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
        for (uint32_t il = 0; il < n_layer; ++il) {
            const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il) + hparams.n_embd_k_s();

            // Read type of key
            int32_t k_type_i_ref;
            read_to(&k_type_i_ref, sizeof(k_type_i_ref));
            const int32_t k_type_i = (int32_t)kv_self.k_l[il]->type;
            if (k_type_i != k_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
                return false;
            }

            // Read row size of key
            uint64_t k_size_row_ref;
            read_to(&k_size_row_ref, sizeof(k_size_row_ref));
            const size_t k_size_row = ggml_row_size(kv_self.k_l[il]->type, n_embd_k_gqa);
            if (k_size_row != k_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                // Read and set the keys for the whole cell range
                ggml_backend_tensor_set(kv_self.k_l[il], read(cell_count * k_size_row), kv_self.head * k_size_row, cell_count * k_size_row);
            }
        }

        if (!kv_self.v_trans) {
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read row size of value
                uint64_t v_size_row_ref;
                read_to(&v_size_row_ref, sizeof(v_size_row_ref));
                const size_t v_size_row = ggml_row_size(kv_self.v_l[il]->type, n_embd_v_gqa);
                if (v_size_row != v_size_row_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                    return false;
                }

                if (cell_count) {
                    // Read and set the values for the whole cell range
                    ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_row), kv_self.head * v_size_row, cell_count * v_size_row);
                }
            }
        } else {
            // For each layer, read the values for each cell (transposed)
            for (uint32_t il = 0; il < n_layer; ++il) {
                const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il) + hparams.n_embd_v_s();

                // Read type of value
                int32_t v_type_i_ref;
                read_to(&v_type_i_ref, sizeof(v_type_i_ref));
                const int32_t v_type_i = (int32_t)kv_self.v_l[il]->type;
                if (v_type_i != v_type_i_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                    return false;
                }

                // Read element size of value
                uint32_t v_size_el_ref;
                read_to(&v_size_el_ref, sizeof(v_size_el_ref));
                const size_t v_size_el = ggml_type_size(kv_self.v_l[il]->type);
                if (v_size_el != v_size_el_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                    return false;
                }

                // Read GQA embedding size
                uint32_t n_embd_v_gqa_ref;
                read_to(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
                if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                    LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                    return false;
                }

                if (cell_count) {
                    // For each row in the transposed matrix, read the values for the whole cell range
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (kv_self.head + j * kv_self.size) * v_size_el;
                        ggml_backend_tensor_set(kv_self.v_l[il], read(cell_count * v_size_el), dst_offset, cell_count * v_size_el);
                    }
                }
            }
        }
        return true;
    }

    void read_kv_cache(struct llama_context * ctx, llama_seq_id seq_id = -1) {
        uint32_t cell_count;
        read_to(&cell_count, sizeof(cell_count));

        bool res = read_kv_cache_meta(ctx, cell_count, seq_id) && read_kv_cache_data(ctx, cell_count);

        if (!res) {
            if (seq_id == -1) {
                llama_kv_cache_clear(ctx);
            } else {
                llama_kv_cache_seq_rm(ctx, seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
};

struct llama_data_write_dummy : llama_data_write {
    size_t size_written = 0;

    llama_data_write_dummy() {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        size_written += size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_write_buffer : llama_data_write {
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    llama_data_write_buffer(uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ggml_backend_tensor_get(tensor, ptr, offset, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_buffer : llama_data_read {
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    llama_data_read_buffer(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    const uint8_t * read(size_t size) override {
        const uint8_t * base_ptr = ptr;
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        ptr += size;
        size_read += size;
        buf_size -= size;
        return base_ptr;
    }

    void read_to(void * dst, size_t size) override {
        memcpy(dst, read(size), size);
    }

    size_t get_size_read() override {
        return size_read;
    }
};

struct llama_data_write_file : llama_data_write {
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;

    llama_data_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor_data(const struct ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t get_size_written() override {
        return size_written;
    }
};

struct llama_data_read_file : llama_data_read {
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;

    llama_data_read_file(llama_file * f) : file(f) {}

    void read_to(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    const uint8_t * read(size_t size) override {
        temp_buffer.resize(size);
        read_to(temp_buffer.data(), size);
        return temp_buffer.data();
    }

    size_t get_size_read() override {
        return size_read;
    }
};

/** copy state data into either a buffer or file depending on the passed in context
 *
 * file context:
 * llama_file file("/path", "wb");
 * llama_data_write_file data_ctx(&file);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
 * buffer context:
 * std::vector<uint8_t> buf(max_size, 0);
 * llama_data_write_buffer data_ctx(buf.data(), max_size);
 * llama_state_get_data_internal(ctx, data_ctx);
 *
*/
static size_t llama_state_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.write_model_info(ctx);

    // copy outputs
    data_ctx.write_output_ids(ctx);
    data_ctx.write_logits(ctx);
    data_ctx.write_embeddings(ctx);

    data_ctx.write_kv_cache(ctx);

    return data_ctx.get_size_written();
}

size_t llama_state_get_data(struct llama_context * ctx, uint8_t * dst, size_t size) {
    llama_data_write_buffer data_ctx(dst, size);
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(struct llama_context * ctx) {
    llama_data_write_dummy data_ctx;
    try {
        return llama_state_get_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx) {
    llama_synchronize(ctx);

    data_ctx.read_model_info(ctx);

    // set outputs
    data_ctx.read_output_ids(ctx);
    data_ctx.read_logits(ctx);
    data_ctx.read_embeddings(ctx);

    data_ctx.read_kv_cache(ctx);

    return data_ctx.get_size_read();
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(struct llama_context * ctx, const uint8_t * src, size_t size) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_set_data_internal(ctx, data_ctx);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static bool llama_state_load_file_internal(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(path_session, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_data_read_file data_ctx(&file);
        const size_t n_read = llama_state_set_data_internal(ctx, data_ctx);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }
    return true;
}

bool llama_state_load_file(struct llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_load_file_internal(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

static bool llama_state_save_file_internal(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    llama_file file(path_session, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file);
    llama_state_get_data_internal(ctx, data_ctx);

    return true;
}

bool llama_state_save_file(struct llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_save_file_internal(ctx, path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

static size_t llama_state_seq_get_data_internal(struct llama_context * ctx, llama_data_write & data_ctx, llama_seq_id seq_id) {
    llama_synchronize(ctx);

    data_ctx.write_kv_cache(ctx, seq_id);

    return data_ctx.get_size_written();
}

size_t llama_state_seq_get_size(struct llama_context * ctx, llama_seq_id seq_id) {
    llama_data_write_dummy data_ctx;
    return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
}

size_t llama_state_seq_get_data(struct llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    llama_data_write_buffer data_ctx(dst, size);
    try {
        return llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_set_data_internal(struct llama_context * ctx, llama_data_read & data_ctx, llama_seq_id dest_seq_id) {
    llama_synchronize(ctx);

    data_ctx.read_kv_cache(ctx, dest_seq_id);

    return data_ctx.get_size_read();
}

size_t llama_state_seq_set_data(struct llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id dest_seq_id) {
    llama_data_read_buffer data_ctx(src, size);
    try {
        return llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state: %s\n", __func__, err.what());
        return 0;
    }
}

static size_t llama_state_seq_save_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_data_write_file data_ctx(&file);
    llama_state_seq_get_data_internal(ctx, data_ctx, seq_id);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + data_ctx.get_size_written());
    return res;
}

static size_t llama_state_seq_load_file_internal(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_data_read_file data_ctx(&file);
        const size_t nread = llama_state_seq_set_data_internal(ctx, data_ctx, dest_seq_id);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_state_seq_save_file(struct llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    try {
        return llama_state_seq_save_file_internal(ctx, filepath, seq_id, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(struct llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    try {
        return llama_state_seq_load_file_internal(ctx, filepath, dest_seq_id, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

const std::vector<std::pair<std::string, struct ggml_tensor *>> & llama_internal_get_tensor_map(
    struct llama_context * ctx
) {
    return ctx->model.tensors_by_name;
}

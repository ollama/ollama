#include "llama-hparams.h"

#include "ggml.h"

#include <algorithm>
#include <cassert>

void llama_hparams::set_swa_pattern(uint32_t n_pattern, bool dense_first) {
    if (dense_first) {
        for (uint32_t il = 0; il < n_layer; ++il) {
            swa_layers[il] = n_pattern == 0 || (il % n_pattern != 0);
        }
    } else {
        for (uint32_t il = 0; il < n_layer; ++il) {
            swa_layers[il] = n_pattern == 0 || (il % n_pattern < (n_pattern - 1));
        }
    }
}

bool llama_hparams::is_swa_any() const {
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (swa_layers[il]) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_head(uint32_t il) const {
    if (il < n_layer) {
        return n_head_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_head_kv(uint32_t il) const {
    if (il < n_layer) {
        return n_head_kv_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_ff(uint32_t il) const {
    if (il < n_layer) {
        return n_ff_arr[il];
    }

    GGML_ABORT("fatal error");
}

uint32_t llama_hparams::n_gqa(uint32_t il) const {
    const uint32_t n_head    = this->n_head(il);
    const uint32_t n_head_kv = this->n_head_kv(il);

    if (n_head_kv == 0) {
        return 0;
    }

    return n_head/n_head_kv;
}

uint32_t llama_hparams::n_embd_inp() const {
    uint32_t n_embd_inp = n_embd;

    if (n_deepstack_layers > 0) {
        n_embd_inp += n_embd * n_deepstack_layers;
    }

    return n_embd_inp;
}

uint32_t llama_hparams::n_embd_out() const {
    return n_embd_out_impl > 0 ? n_embd_out_impl : n_embd;
}

uint32_t llama_hparams::n_embd_k_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_k * n_head_kv;
}

uint32_t llama_hparams::n_embd_v_gqa(uint32_t il) const {
    const uint32_t n_head_kv = this->n_head_kv(il);

    return n_embd_head_v * n_head_kv;
}

bool llama_hparams::is_n_embd_k_gqa_variable() const {
    const uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_k_gqa(il)) {
            return true;
        }
    }

    return false;
}

bool llama_hparams::is_n_embd_v_gqa_variable() const {
    const uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (val != n_embd_v_gqa(il)) {
            return true;
        }
    }

    return false;
}

uint32_t llama_hparams::n_embd_k_gqa_max() const {
    uint32_t val = n_embd_k_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_k_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_v_gqa_max() const {
    uint32_t val = n_embd_v_gqa();
    for (uint32_t il = 0; il < n_layer; ++il) {
        val = std::max(val, n_embd_v_gqa(il));
    }

    return val;
}

uint32_t llama_hparams::n_embd_r() const {
    if (wkv_head_size != 0) {
        // for RWKV models
        return token_shift_count * n_embd;
    }

    if (n_shortconv_l_cache != 0) {
        // for LFM2 models
        return n_embd * (n_shortconv_l_cache - 1);
    }

    if (n_embd_head_kda != 0) {
        // for Kimi KDA layers
        // Conv state for Q, K, V: 3 * (d_conv - 1) * n_head * head_dim
        const uint32_t d_inner = n_head() * n_embd_head_kda;  // 32 * 128 = 4096
        return 3 * (ssm_d_conv > 0 ? ssm_d_conv - 1 : 3) * d_inner;
    }

    // TODO: maybe support other convolution strides than 1
    // NOTE: since the first column of the conv_state is shifted out each time, it's not actually needed
    // Corresponds to Mamba's conv_states size
    return (ssm_d_conv > 0 ? ssm_d_conv - 1 : 0) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state);
}

uint32_t llama_hparams::n_embd_s() const {
    if (wkv_head_size != 0) {
        // corresponds to RWKV's wkv_states size
        return n_embd * wkv_head_size;
    }

    if (n_embd_head_kda != 0) {
        // for Kimi KDA layers
        // Full recurrent state: head_dim * head_dim * n_head
        // h tensor shape for delta attention: [head_dim, head_dim, n_head]
        return n_embd_head_kda * n_embd_head_kda * n_head();  // 128 * 128 * 32 = 524288
    }

    // corresponds to Mamba's ssm_states size
    return ssm_d_state * ssm_d_inner;
}

bool llama_hparams::is_recurrent(uint32_t il) const {
    if (il < n_layer) {
        return recurrent_layer_arr[il];
    }

    GGML_ABORT("%s: il (%u) out of bounds (n_layer: %u)\n", __func__, il, n_layer);
}

uint32_t llama_hparams::n_pos_per_embd() const {
    return rope_type == LLAMA_ROPE_TYPE_MROPE || rope_type == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
}

bool llama_hparams::n_bskcn(uint32_t n, uint32_t il) const {
    if (il < n_layer) {
        return n_bskcn_arr[n][il] > 0;
    }

    GGML_ABORT("fatal error");
}

bool llama_hparams::is_swa(uint32_t il) const {
    if (il < n_layer) {
        return swa_layers[il];
    }

    GGML_ABORT("fatal error");
}

bool llama_hparams::is_mla() const {
    assert((n_embd_head_k_mla_impl == 0 && n_embd_head_v_mla_impl == 0) ||
           (n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0));

    return n_embd_head_k_mla_impl != 0 && n_embd_head_v_mla_impl != 0;
}

uint32_t llama_hparams::n_embd_head_k_mla() const {
    return is_mla() ? n_embd_head_k_mla_impl : n_embd_head_k;
}

uint32_t llama_hparams::n_embd_head_v_mla() const {
    return is_mla() ? n_embd_head_v_mla_impl : n_embd_head_v;
}

bool llama_hparams::has_kv(uint32_t il) const {
    if (n_layer_kv_from_start >= 0) {
        if (il < (uint32_t) n_layer_kv_from_start) {
            return true;
        }

        return false;
    }

    // by default, all layers have kv
    return true;
}

uint32_t llama_hparams::n_layer_kv() const {
    uint32_t res = 0;

    for (uint32_t il = 0; il < n_layer; ++il) {
        if (has_kv(il)) {
            res++;
        }
    }

    return res;
}

bool llama_hparams::use_mrope() const {
    return rope_sections[0] > 0 && rope_sections[1] > 0;
}

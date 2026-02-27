#include "llama-adapter.h"

#include "llama-impl.h"
#include "llama-mmap.h"
#include "llama-model.h"

#include <map>
#include <cassert>
#include <sstream>
#include <stdexcept>

// vec

ggml_tensor * llama_adapter_cvec::tensor_for(int il) const {
    if (il < 0 || il < layer_start || il > layer_end || (size_t) il >= tensors.size()) {
        return nullptr;
    }

    return tensors[il];
}

ggml_tensor * llama_adapter_cvec::apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const {
    ggml_tensor * layer_dir = tensor_for(il);
    if (layer_dir != nullptr) {
        cur = ggml_add(ctx, cur, layer_dir);
    }

    return cur;
}

bool llama_adapter_cvec::init(const llama_model & model) {
    const auto & hparams = model.hparams;

    GGML_ASSERT(tensors.empty());
    GGML_ASSERT(ctxs.empty());
    GGML_ASSERT(bufs.empty());

    // create a context for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ hparams.n_layer*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map[buft] = ctx;
            ctxs.emplace_back(ctx);

            return ctx;
        }

        return it->second;
    };

    // make tensors
    tensors.reserve(hparams.n_layer);
    tensors.push_back(nullptr); // there's never a tensor for layer 0
    for (size_t il = 1; il < hparams.n_layer; il++) {
        ggml_backend_buffer_type_t buft = model.select_buft(il);
        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            LLAMA_LOG_ERROR("%s: failed to allocate context for control vector\n", __func__);
            return false;
        }
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hparams.n_embd);
        tensors.push_back(tensor);
    }

    // allocate tensors / buffers and zero
    bufs.reserve(ctx_map.size());
    for (auto it : ctx_map) {
        ggml_backend_buffer_type_t buft = it.first;
        ggml_context * ctx = it.second;
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
        if (!buf) {
            LLAMA_LOG_ERROR("%s: failed to allocate buffer for control vector\n", __func__);
            return false;
        }
        ggml_backend_buffer_clear(buf, 0);
        bufs.emplace_back(buf);
    }

    return true;
}

bool llama_adapter_cvec::apply(
        const llama_model & model,
        const float * data,
        size_t len,
        int32_t n_embd,
        int32_t il_start,
        int32_t il_end) {
    const auto & hparams = model.hparams;

    if (data == nullptr) {
        // disable the current control vector (but leave allocated for later)
        layer_start = -1;
        layer_end   = -1;
        return true;
    }

    if (n_embd != (int) hparams.n_embd) {
        LLAMA_LOG_ERROR("%s: control vector n_embd does not match model\n", __func__);
        return false;
    }

    if (tensors.empty()) {
        if (!init(model)) {
            return false;
        }
    }

    layer_start = il_start;
    layer_end   = il_end;

    for (size_t il = 1; il < hparams.n_layer; il++) {
        assert(tensors[il] != nullptr);

        const size_t off = n_embd * (il - 1); // buffer doesn't have data for layer 0, since it's never present
        if (off + n_embd <= len) {
            ggml_backend_tensor_set(tensors[il], data + off, 0, n_embd * ggml_element_size(tensors[il]));
        }
    }

    return true;
}

// lora

llama_adapter_lora_weight * llama_adapter_lora::get_weight(ggml_tensor * w) {
    const std::string name(w->name);

    const auto pos = ab_map.find(name);
    if (pos != ab_map.end()) {
        return &pos->second;
    }

    return nullptr;
}

static void llama_adapter_lora_init_impl(const char * path_lora, llama_adapter_lora & adapter) {
    LLAMA_LOG_INFO("%s: loading lora adapter from '%s' ...\n", __func__, path_lora);

    llama_model & model = adapter.model;

    ggml_context * ctx_init;
    gguf_init_params meta_gguf_params = {
        /* .no_alloc = */ true,
        /* .ctx      = */ &ctx_init,
    };

    gguf_context_ptr ctx_gguf { gguf_init_from_file(path_lora, meta_gguf_params) };
    if (!ctx_gguf) {
        throw std::runtime_error("failed to load lora adapter file from " + std::string(path_lora));
    }

    ggml_context_ptr ctx { ctx_init };

    // check metadata
    {
        const gguf_context * gguf_ctx = ctx_gguf.get();

        LLAMA_LOG_INFO("%s: Dumping metadata keys/values.\n", __func__);

        // get metadata as string
        for (int i = 0; i < gguf_get_n_kv(gguf_ctx); i++) {
            gguf_type type = gguf_get_kv_type(gguf_ctx, i);
            const std::string type_name =
                type == GGUF_TYPE_ARRAY
                ? format("%s[%s,%zu]", gguf_type_name(type), gguf_type_name(gguf_get_arr_type(gguf_ctx, i)), gguf_get_arr_n(gguf_ctx, i))
                : gguf_type_name(type);
            const char * name = gguf_get_key(gguf_ctx, i);
            const std::string value = gguf_kv_to_str(gguf_ctx, i);

            if (type != GGUF_TYPE_ARRAY) {
                adapter.gguf_kv.emplace(name, value);
            }

            const size_t MAX_VALUE_LEN = 40;
            std::string print_value = value.size() > MAX_VALUE_LEN ? format("%s...", value.substr(0, MAX_VALUE_LEN - 3).c_str()) : value;
            replace_all(print_value, "\n", "\\n");

            LLAMA_LOG_INFO("%s: - kv %3d: %42s %-16s = %s\n", __func__, i, name, type_name.c_str(), print_value.c_str());
        }

        auto get_kv_str = [&](const std::string & key) -> std::string {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? "" : std::string(gguf_get_val_str(gguf_ctx, id));
        };
        auto get_kv_f32 = [&](const std::string & key) -> float {
            int id = gguf_find_key(gguf_ctx, key.c_str());
            return id < 0 ? 0.0f : gguf_get_val_f32(gguf_ctx, id);
        };
        LLM_KV llm_kv = LLM_KV(LLM_ARCH_UNKNOWN);

        auto general_type = get_kv_str(llm_kv(LLM_KV_GENERAL_TYPE));
        if (general_type != "adapter") {
            throw std::runtime_error("expect general.type to be 'adapter', but got: " + general_type);
        }

        auto general_arch_str = get_kv_str(llm_kv(LLM_KV_GENERAL_ARCHITECTURE));
        auto general_arch = llm_arch_from_string(general_arch_str);
        if (general_arch != model.arch) {
            throw std::runtime_error("model arch and LoRA arch mismatch");
        }

        auto adapter_type = get_kv_str(llm_kv(LLM_KV_ADAPTER_TYPE));
        if (adapter_type != "lora") {
            throw std::runtime_error("expect adapter.type to be 'lora', but got: " + adapter_type);
        }

        adapter.alpha = get_kv_f32(llm_kv(LLM_KV_ADAPTER_LORA_ALPHA));

        // parse alora invocation sequence vector
        const auto & key = llm_kv(LLM_KV_ADAPTER_ALORA_INVOCATION_TOKENS);
        const int kid = gguf_find_key(ctx_gguf.get(), key.c_str());
        if (kid >= 0) {
            if (gguf_get_kv_type(ctx_gguf.get(), kid) != GGUF_TYPE_ARRAY) {
                throw std::runtime_error("invalid gguf type for " + key);
            }
            const auto arr_type = gguf_get_arr_type(ctx_gguf.get(), kid);
            if (arr_type != GGUF_TYPE_UINT32) {
                throw std::runtime_error("invalid gguf element type for " + key);
            }
            const size_t seq_len = gguf_get_arr_n(ctx_gguf.get(), kid);
            const void * data = gguf_get_arr_data(ctx_gguf.get(), kid);
            adapter.alora_invocation_tokens.resize(seq_len);
            std::copy(
                (const llama_token *)data,
                (const llama_token *)data + seq_len,
                adapter.alora_invocation_tokens.begin());
        }
    }

    int n_tensors = gguf_get_n_tensors(ctx_gguf.get());

    // contexts for each buffer type
    std::map<ggml_backend_buffer_type_t, ggml_context *> ctx_map;
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // add a new context
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * buft_ctx = ggml_init(params);
            if (!buft_ctx) {
                return nullptr;
            }
            ctx_map[buft] = buft_ctx;
            adapter.ctxs.emplace_back(buft_ctx);
            return buft_ctx;
        };
        return it->second;
    };

    // bundle lora_a and lora_b into pairs
    std::map<std::string, llama_adapter_lora_weight> ab_map;
    auto str_endswith = [](const std::string & str, const std::string & suffix) {
        return str.size() >= suffix.size() && str.compare(str.size()-suffix.size(), suffix.size(), suffix) == 0;
    };

    for (ggml_tensor * cur = ggml_get_first_tensor(ctx.get()); cur; cur = ggml_get_next_tensor(ctx.get(), cur)) {
        std::string name(cur->name);
        if (str_endswith(name, ".lora_a")) {
            replace_all(name, ".lora_a", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(cur, nullptr);
            } else {
                ab_map[name].a = cur;
            }
        } else if (str_endswith(name, ".lora_b")) {
            replace_all(name, ".lora_b", "");
            if (ab_map.find(name) == ab_map.end()) {
                ab_map[name] = llama_adapter_lora_weight(nullptr, cur);
            } else {
                ab_map[name].b = cur;
            }
        } else if (str_endswith(name, "_norm.weight")) {
            // TODO: add support for norm vector
            // for now, we don't really care because most adapters still work fine without it
            continue;
        } else {
            throw std::runtime_error("LoRA tensor '" + name + "' has unexpected suffix");
        }
    }

    // get extra buffer types of the CPU
    // TODO: a more general solution for non-CPU extra buft should be imlpemented in the future
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12593#pullrequestreview-2718659948
    std::vector<ggml_backend_buffer_type_t> buft_extra;
    {
        auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (!cpu_dev) {
            throw std::runtime_error(format("%s: no CPU backend found", __func__));
        }
        auto * cpu_reg = ggml_backend_dev_backend_reg(cpu_dev);

        auto ggml_backend_dev_get_extra_bufts_fn = (ggml_backend_dev_get_extra_bufts_t)
            ggml_backend_reg_get_proc_address(cpu_reg, "ggml_backend_dev_get_extra_bufts");

        if (ggml_backend_dev_get_extra_bufts_fn) {
            ggml_backend_buffer_type_t * extra_bufts = ggml_backend_dev_get_extra_bufts_fn(cpu_dev);
            while (extra_bufts && *extra_bufts) {
                buft_extra.emplace_back(*extra_bufts);
                ++extra_bufts;
            }
        }
    }

    // add tensors
    for (auto & it : ab_map) {
        const std::string & name = it.first;
        llama_adapter_lora_weight & w = it.second;
        bool is_token_embd = str_endswith(name, "token_embd.weight");

        if (!w.a || !w.b) {
            throw std::runtime_error("LoRA tensor pair for '" + name + "' is missing one component");
        }

        // device buft and device ctx
        const auto * model_tensor = model.get_tensor(name.c_str());
        if (!model_tensor) {
            throw std::runtime_error("LoRA tensor '" + name + "' does not exist in base model (hint: maybe wrong base model?)");
        }

        auto * buft = ggml_backend_buffer_get_type(model_tensor->buffer);

        // do not load loras to extra buffer types (i.e. bufts for repacking) -> use the CPU in that case
        for (auto & ex : buft_extra) {
            if (ex == buft) {
                LLAMA_LOG_WARN("%s: lora for '%s' cannot use buft '%s', fallback to CPU\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

                auto * cpu_dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
                if (!cpu_dev) {
                    throw std::runtime_error(format("%s: no CPU backend found", __func__));
                }
                buft = ggml_backend_dev_buffer_type(cpu_dev);

                break;
            }
        }

        LLAMA_LOG_DEBUG("%s: lora for '%s' -> '%s'\n", __func__, model_tensor->name, ggml_backend_buft_name(buft));

        ggml_context * dev_ctx = ctx_for_buft(buft);
        // validate tensor shape
        if (is_token_embd) {
            // expect B to be non-transposed, A and B are flipped; see llm_build_inp_embd()
            if (model_tensor->ne[0] != w.b->ne[1] || model_tensor->ne[1] != w.a->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
        } else {
            if (model_tensor->ne[0] != w.a->ne[0] || model_tensor->ne[1] != w.b->ne[1]) {
                throw std::runtime_error("tensor '" + name + "' has incorrect shape (hint: maybe wrong base model?)");
            }
            if (w.a->ne[1] != w.b->ne[0]) {
                throw std::runtime_error("lora_a tensor is not transposed (hint: adapter from \"finetune\" example is no longer supported)");
            }
        }

        // save tensor to adapter
        ggml_tensor * tensor_a = ggml_dup_tensor(dev_ctx, w.a);
        ggml_tensor * tensor_b = ggml_dup_tensor(dev_ctx, w.b);
        ggml_set_name(tensor_a, w.a->name);
        ggml_set_name(tensor_b, w.b->name);
        adapter.ab_map[name] = llama_adapter_lora_weight(tensor_a, tensor_b);
    }

    // allocate tensors / buffers and zero
    {
        adapter.ctxs.reserve(ctx_map.size());
        adapter.bufs.reserve(ctx_map.size());
        for (auto & it : ctx_map) {
            ggml_backend_buffer_type_t buft = it.first;
            ggml_context * ctx_dev = it.second;
            ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev, buft) };
            if (!buf) {
                throw std::runtime_error("failed to allocate buffer for lora adapter\n");
            }
            LLAMA_LOG_INFO("%s: %10s LoRA buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf.get()), ggml_backend_buffer_get_size(buf.get())/1024.0/1024.0);
            adapter.bufs.emplace_back(std::move(buf));
        }
    }

    // set tensor data
    {
        llama_file gguf_file(path_lora, "rb");
        std::vector<uint8_t> read_buf;
        auto set_tensor = [&](ggml_tensor * orig, ggml_tensor * dev) {
            size_t offs = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), gguf_find_tensor(ctx_gguf.get(), orig->name));
            size_t size = ggml_nbytes(orig);
            read_buf.resize(size);
            gguf_file.seek(offs, SEEK_SET);
            gguf_file.read_raw(read_buf.data(), size);
            ggml_backend_tensor_set(dev, read_buf.data(), 0, size);
        };
        for (auto & it : adapter.ab_map) {
            auto orig = ab_map[it.first];
            auto dev  = it.second;
            set_tensor(orig.a, dev.a);
            set_tensor(orig.b, dev.b);
        }
    }

    // update number of nodes used
    model.n_lora_nodes += adapter.get_n_nodes();

    LLAMA_LOG_INFO("%s: loaded %zu tensors from lora file\n", __func__, adapter.ab_map.size()*2);
}

llama_adapter_lora * llama_adapter_lora_init(llama_model * model, const char * path_lora) {
    llama_adapter_lora * adapter = new llama_adapter_lora(*model);

    try {
        llama_adapter_lora_init_impl(path_lora, *adapter);
        return adapter;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to apply lora adapter: %s\n", __func__, err.what());

        delete adapter;
    }

    return nullptr;
}

int32_t llama_adapter_meta_val_str(const llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size) {
    const auto & it = adapter->gguf_kv.find(key);
    if (it == adapter->gguf_kv.end()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

int32_t llama_adapter_meta_count(const llama_adapter_lora * adapter) {
    return (int)adapter->gguf_kv.size();
}

int32_t llama_adapter_meta_key_by_index(const llama_adapter_lora * adapter, int i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->first.c_str());
}

int32_t llama_adapter_meta_val_str_by_index(const llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size) {
    if (i < 0 || i >= (int)adapter->gguf_kv.size()) {
        if (buf_size > 0) {
            buf[0] = '\0';
        }
        return -1;
    }
    auto it = adapter->gguf_kv.begin();
    std::advance(it, i);
    return snprintf(buf, buf_size, "%s", it->second.c_str());
}

void llama_adapter_lora_free(llama_adapter_lora * adapter) {
    // update number of nodes used
    GGML_ASSERT(adapter->model.n_lora_nodes >= adapter->get_n_nodes());
    adapter->model.n_lora_nodes -= adapter->get_n_nodes();

    delete adapter;
}

uint64_t llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter) {
    if (!adapter) {
        return 0;
    }
    return adapter->alora_invocation_tokens.size();
}

const llama_token * llama_adapter_get_alora_invocation_tokens(const llama_adapter_lora * adapter) {
    GGML_ASSERT(adapter);
    return adapter->alora_invocation_tokens.data();
}

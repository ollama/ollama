#include "llama.h"

#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-context.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <stdexcept>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

//
// interface implementation
//

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_device_memory_data {
    int64_t total;
    int64_t free;
    llama_memory_breakdown_data mb;
};

static std::vector<llama_device_memory_data> llama_get_device_memory_data(
        const char * path_model, const llama_model_params * mparams, const llama_context_params * cparams,
        std::vector<ggml_backend_dev_t> & devs, uint32_t & hp_ngl, uint32_t & hp_n_ctx_train, uint32_t & hp_n_expert,
        const ggml_log_level log_level) {
    struct user_data_t {
        struct {
            ggml_log_callback callback;
            void * user_data;
        } original_logger;
        ggml_log_level min_level; // prints below this log level go to debug log
    };
    user_data_t ud;
    llama_log_get(&ud.original_logger.callback, &ud.original_logger.user_data);
    ud.min_level = log_level;

    llama_log_set([](ggml_log_level level, const char * text, void * user_data) {
        const user_data_t * ud = (const user_data_t *) user_data;
        const ggml_log_level level_eff = level >= ud->min_level ? level : GGML_LOG_LEVEL_DEBUG;
        ud->original_logger.callback(level_eff, text, ud->original_logger.user_data);
    }, &ud);

    llama_model_params mparams_copy = *mparams;
    mparams_copy.no_alloc = true;
    mparams_copy.use_mmap = false;

    llama_model * model = llama_model_load_from_file(path_model, mparams_copy);
    if (model == nullptr) {
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to load model");
    }

    llama_context * ctx = llama_init_from_model(model, *cparams);
    if (ctx == nullptr) {
        llama_model_free(model);
        llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
        throw std::runtime_error("failed to create llama_context from model");
    }

    std::vector<llama_device_memory_data> ret(model->devices.size());

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown = ctx->memory_breakdown();

    for (const auto & [buft, mb] : memory_breakdown) {
        if (ggml_backend_buft_is_host(buft)) {
            continue;
        }

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
        if (!dev) {
            continue;
        }
        for (size_t i = 0; i < ret.size(); i++) {
            if (model->devices[i] == dev) {
                ret[i].mb.model   += mb.model;
                ret[i].mb.context += mb.context;
                ret[i].mb.compute += mb.compute;
                break;
            }
        }
    }
    for (size_t i = 0; i < ret.size(); i++) {
        size_t free, total;
        ggml_backend_dev_memory(model->devices[i], &free, &total);
        ret[i].free  = free;
        ret[i].total = total;
    }

    devs           = model->devices;
    hp_ngl         = model->hparams.n_layer;
    hp_n_ctx_train = model->hparams.n_ctx_train;
    hp_n_expert    = model->hparams.n_expert;

    llama_memory_breakdown_print(ctx); // goes to debug log

    llama_free(ctx);
    llama_model_free(model);
    llama_log_set(ud.original_logger.callback, ud.original_logger.user_data);
    return ret;
}

// enum to identify part of a layer for distributing its tensors:
enum layer_fraction_t {
    LAYER_FRACTION_NONE = 0, // nothing
    LAYER_FRACTION_ATTN = 1, // attention
    LAYER_FRACTION_UP   = 2, // attention + up
    LAYER_FRACTION_GATE = 3, // attention + up + gate
    LAYER_FRACTION_MOE  = 4, // everything but sparse MoE weights
};
// this enum is only used in llama_params_fit_impl but needs to be defined outside of it to fix a Windows compilation issue

static void llama_params_fit_impl(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t margin_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    constexpr int64_t MiB = 1024*1024;
    const int64_t margin = margin_s; // this function uses int64_t rather than size_t for memory sizes to more conveniently handle deficits
    typedef std::vector<llama_device_memory_data> dmds_t;
    const llama_model_params default_mparams = llama_model_default_params();

    std::vector<ggml_backend_dev_t> devs;
    uint32_t hp_ngl = 0; // hparams.n_gpu_layers
    uint32_t hp_nct = 0; // hparams.n_ctx_train
    uint32_t hp_nex = 0; // hparams.n_expert

    // step 1: get data for default parameters and check whether any changes are necessary in the first place

    LLAMA_LOG_DEBUG("%s: getting device memory data for initial parameters:\n", __func__);
    const dmds_t dmds_full = llama_get_device_memory_data(path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);
    const size_t nd = devs.size(); // number of devices
    if (nd == 0) {
        LLAMA_LOG_INFO("%s: no devices with dedicated memory found\n", __func__);
        return;
    }

    std::vector<std::string> dev_names;
    {
        dev_names.reserve(nd);
        size_t max_length = 0;
        for (ggml_backend_dev_t dev : devs) {
            std::string name = ggml_backend_dev_name(dev);
            name += " (";
            name += ggml_backend_dev_description(dev);
            name += ")";
            dev_names.push_back(name);
            max_length = std::max(max_length, name.length());
        }
        for (std::string & dn : dev_names) {
            dn.insert(dn.end(), max_length - dn.length(), ' ');
        }
    }

    int64_t sum_total          = 0;
    int64_t sum_projected_free = 0;
    int64_t min_projected_free = INT64_MAX;
    int64_t sum_projected_used = 0;
    int64_t sum_projected_ctx  = 0;

    if (nd > 1) {
        LLAMA_LOG_INFO("%s: projected memory use with initial parameters [MiB]:\n", __func__);
    }
    for (size_t id = 0; id < nd; id++) {
        const llama_device_memory_data & dmd = dmds_full[id];

        const int64_t projected_used = dmd.mb.total();
        const int64_t projected_free = dmd.free - projected_used;

        sum_total          += dmd.total;
        sum_projected_used += projected_used;
        sum_projected_free += projected_free;
        min_projected_free  = std::min(min_projected_free, projected_free);
        sum_projected_ctx  += dmd.mb.context;

        if (nd > 1) {
            LLAMA_LOG_INFO("%s:   - %s: %6" PRId64 " total, %6" PRId64 " used, %6" PRId64 " %s\n",
                __func__, dev_names[id].c_str(), dmd.total/MiB, projected_used/MiB, std::abs(projected_free)/MiB,
                projected_free >= 0 ? "surplus" : "deficit");
        }
    }
    assert(sum_total >= 0 && sum_projected_used >= 0 && sum_projected_ctx >= 0);
    assert(sum_projected_used >= sum_projected_ctx);
    LLAMA_LOG_INFO("%s: projected to use %" PRId64 " MiB of device memory vs. %" PRId64 " MiB of free device memory\n",
        __func__, sum_projected_used/MiB, sum_total/MiB);
    if (min_projected_free >= margin) {
        if (nd == 1) {
            LLAMA_LOG_INFO("%s: will leave %" PRId64 " >= %" PRId64 " MiB of free device memory, no changes needed\n",
                __func__, min_projected_free/MiB, margin/MiB);
            return;
        }
        LLAMA_LOG_INFO("%s: will leave at least %" PRId64 " >= %" PRId64 " MiB of free memory on all devices, no changes needed\n",
            __func__, min_projected_free/MiB, margin/MiB);
        return;
    }

    // step 2: try reducing memory use by reducing the context size

    {
        int64_t global_surplus = sum_projected_free - int64_t(nd)*margin;
        if (global_surplus < 0) {
            LLAMA_LOG_INFO(nd == 1 ?
                "%s: cannot fulfill margin of %" PRId64 " MiB, need to reduce device memory by %" PRId64 " MiB\n" :
                "%s: cannot fulfill margin of %" PRId64 " MiB on all devices, need to use %" PRId64 " MiB less in total\n",
                __func__, margin/MiB, -global_surplus/MiB);
            if (cparams->n_ctx == 0) {
                if (hp_nct > n_ctx_min) {
                    const int64_t bytes_per_ctx = sum_projected_ctx / hp_nct;
                    const uint32_t ctx_reduction = std::min(
                        uint32_t((-global_surplus + bytes_per_ctx - 1) / bytes_per_ctx), hp_nct - n_ctx_min);
                    cparams->n_ctx = hp_nct - ctx_reduction;
                    const int64_t memory_reduction = ctx_reduction * bytes_per_ctx;
                    global_surplus += memory_reduction;
                    LLAMA_LOG_INFO("%s: context size reduced from %" PRIu32 " to %" PRIu32 " -> need %" PRId64 " MiB less memory in total\n",
                        __func__, hp_nct, cparams->n_ctx, memory_reduction/MiB);
                    if (global_surplus >= 0) {
                        if (nd == 1) {
                            LLAMA_LOG_INFO("%s: entire model can be fit by reducing context\n", __func__);
                            return;
                        }
                        LLAMA_LOG_INFO("%s: entire model should be fit across devices by reducing context\n", __func__);
                    }
                } else {
                    LLAMA_LOG_INFO("%s: default model context size is %" PRIu32 " which is <= the min. context size of %" PRIu32 " -> no change\n",
                        __func__, hp_nct, n_ctx_min);
                }
            } else {
                LLAMA_LOG_INFO("%s: context size set by user to %" PRIu32 " -> no change\n", __func__, cparams->n_ctx);
            }
        }
    }

    if (mparams->n_gpu_layers != default_mparams.n_gpu_layers) {
        throw std::runtime_error("n_gpu_layers already set by user to " + std::to_string(mparams->n_gpu_layers) + ", abort");
    }
    if (nd > 1) {
        if (!tensor_split) {
            throw std::runtime_error("did not provide a buffer to write the tensor_split to, abort");
        }
        if (mparams->tensor_split) {
            for (size_t id = 0; id < nd; id++) {
                if (mparams->tensor_split[id] != 0.0f) {
                    throw std::runtime_error("model_params::tensor_split already set by user, abort");
                }
            }
        }
        if (mparams->split_mode == LLAMA_SPLIT_MODE_ROW) {
            throw std::runtime_error("changing weight allocation for LLAMA_SPLIT_MODE_ROW not implemented, abort");
        }
        if (hp_ngl < 2*nd) {
            throw std::runtime_error("model has only " + std::to_string(hp_ngl) + " layers but need at least "
                + std::to_string(2*nd) + " to fit memory for " + std::to_string(nd) + " devices, abort");
        }
    }
    if (!tensor_buft_overrides) {
        throw std::runtime_error("did not provide buffer to set tensor_buft_overrides, abort");
    }
    if (mparams->tensor_buft_overrides && (mparams->tensor_buft_overrides->pattern || mparams->tensor_buft_overrides->buft)) {
        throw std::runtime_error("model_params::tensor_buft_overrides already set by user, abort");
    }

    // step 3: iteratively fill the back to front with "dense" layers
    //   - for a dense model simply fill full layers, giving each device a contiguous slice of the model
    //   - for a MoE model, same as dense model but with all MoE tensors in system memory

    // utility function that returns a static C string matching the tensors for a specific layer index and layer fraction:
    auto get_overflow_pattern = [&](const size_t il, const layer_fraction_t lf) -> const char * {
        constexpr size_t n_strings = 1000;
        if (il >= n_strings) {
            throw std::runtime_error("at most " + std::to_string(n_strings) + " model layers are supported");
        }
        switch (lf) {
            case LAYER_FRACTION_ATTN: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|gate|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_UP: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(gate|down).*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_GATE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_down.*";
                }
                return patterns[il].c_str();
            }
            case LAYER_FRACTION_MOE: {
                static std::array<std::string, n_strings> patterns;
                if (patterns[il].empty()) {
                    patterns[il] = "blk\\." + std::to_string(il) + "\\.ffn_(up|down|gate)_(ch|)exps";
                }
                return patterns[il].c_str();
            }
            default:
                GGML_ABORT("fatal error");
        }
    };

    struct ngl_t {
        uint32_t n_layer = 0; // number of total layers
        uint32_t n_part  = 0; // number of partial layers, <= n_layer

        // for the first partial layer varying parts can overflow, all further layers use LAYER_FRACTION_MOE:
        layer_fraction_t overflow_type = LAYER_FRACTION_MOE;
    };

    const size_t ntbo = llama_max_tensor_buft_overrides();

    // utility function to set n_gpu_layers and tensor_split
    auto set_ngl_tensor_split_tbo = [&](
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            llama_model_params & mparams,
            const bool add_nonrepeating) {
        mparams.n_gpu_layers = 0;
        for (size_t id = 0; id < nd; id++) {
            mparams.n_gpu_layers += ngl_per_device[id].n_layer;
            if (nd > 1) {
                tensor_split[id] = ngl_per_device[id].n_layer;
            }
        }
        assert(uint32_t(mparams.n_gpu_layers) <= hp_ngl);
        uint32_t il0 = hp_ngl - mparams.n_gpu_layers; // start index for tensor buft overrides

        if (add_nonrepeating) {
            mparams.n_gpu_layers += 1;
            tensor_split[nd - 1] += 1;
        }
        mparams.tensor_split = tensor_split;

        size_t itbo = 0;
        for (size_t id = 0; id < nd; id++) {
            il0 += ngl_per_device[id].n_layer - ngl_per_device[id].n_part;
            for (uint32_t il = il0; il < il0 + ngl_per_device[id].n_part; il++) {
                if (itbo + 1 >= ntbo) {
                    tensor_buft_overrides[itbo].pattern = nullptr;
                    tensor_buft_overrides[itbo].buft    = nullptr;
                    itbo++;
                    mparams.tensor_buft_overrides = tensor_buft_overrides;
                    throw std::runtime_error("llama_params_fit_n_tensor_buft_overrides() == "
                        + std::to_string(ntbo) + " is insufficient for model\n");
                }
                tensor_buft_overrides[itbo].pattern = get_overflow_pattern(il, il == il0 ? ngl_per_device[id].overflow_type : LAYER_FRACTION_MOE);
                tensor_buft_overrides[itbo].buft = overflow_bufts[id];
                itbo++;
            }
            il0 += ngl_per_device[id].n_part;
        }
        tensor_buft_overrides[itbo].pattern = nullptr;
        tensor_buft_overrides[itbo].buft    = nullptr;
        itbo++;
        mparams.tensor_buft_overrides = tensor_buft_overrides;
    };

    // utility function that returns the memory use per device for given numbers of layers per device
    auto get_memory_for_layers = [&](
            const char * func_name,
            const std::vector<ngl_t> & ngl_per_device,
            const std::vector<ggml_backend_buffer_type_t> & overflow_bufts,
            const bool add_nonrepeating) -> std::vector<int64_t> {
        llama_model_params mparams_copy = *mparams;
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, mparams_copy, add_nonrepeating);

        const dmds_t dmd_nl = llama_get_device_memory_data(
            path_model, &mparams_copy, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        LLAMA_LOG_DEBUG("%s: memory for test allocation by device:\n", func_name);
        for (size_t id = 0; id < nd; id++) {
            const ngl_t & n = ngl_per_device[id];
            LLAMA_LOG_DEBUG(
                "%s: id=%zu, n_layer=%2" PRIu32 ", n_part=%2" PRIu32 ", overflow_type=%d, mem=%6" PRId64 " MiB\n",
                func_name, id, n.n_layer, n.n_part, int(n.overflow_type), dmd_nl[id].mb.total()/MiB);
        }

        std::vector<int64_t> ret;
        ret.reserve(nd);
        for (const llama_device_memory_data & dmd : dmd_nl) {
            ret.push_back(dmd.mb.total());
        }
        return ret;
    };

    int64_t global_surplus_cpu_moe = 0;
    if (hp_nex > 0) {
        const static std::string pattern_moe_all = "blk\\.\\d+\\.ffn_(up|down|gate)_(ch|)exps"; // matches all MoE tensors
        ggml_backend_buffer_type_t cpu_buft = ggml_backend_cpu_buffer_type();
        tensor_buft_overrides[0] = {pattern_moe_all.c_str(), cpu_buft};
        tensor_buft_overrides[1] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;

        LLAMA_LOG_DEBUG("%s: getting device memory data with all MoE tensors moved to system memory:\n", __func__);
        const dmds_t dmds_cpu_moe = llama_get_device_memory_data(
            path_model, mparams, cparams, devs, hp_ngl, hp_nct, hp_nex, log_level);

        for (const llama_device_memory_data & dmd : dmds_cpu_moe) {
            global_surplus_cpu_moe += dmd.free;
            global_surplus_cpu_moe -= int64_t(dmd.mb.total()) + margin;
        }

        if (global_surplus_cpu_moe > 0) {
            LLAMA_LOG_INFO("%s: with only dense weights in device memory there is a total surplus of %" PRId64 " MiB\n",
                __func__, global_surplus_cpu_moe/MiB);
        } else {
            LLAMA_LOG_INFO("%s: with only dense weights in device memory there is still a total deficit of %" PRId64 " MiB\n",
                __func__, -global_surplus_cpu_moe/MiB);
        }

        // reset
        tensor_buft_overrides[0] = {nullptr, nullptr};
        mparams->tensor_buft_overrides = tensor_buft_overrides;
    }

    std::vector<int64_t> targets; // maximum acceptable memory use per device
    targets.reserve(nd);
    for (size_t id = 0; id < nd; id++) {
        targets.push_back(dmds_full[id].free - margin);
        LLAMA_LOG_DEBUG("%s: id=%zu, target=%" PRId64 " MiB\n", __func__, id, targets[id]/MiB);
    }

    // whether for the optimal memory use we expect to load at least some MoE tensors:
    const bool partial_moe = hp_nex > 0 && global_surplus_cpu_moe > 0;

    std::vector<ggml_backend_buffer_type_t> overflow_bufts; // which bufts the partial layers of a device overflow to:
    overflow_bufts.reserve(nd);
    for (size_t id = 0; id < nd - 1; ++id) {
        overflow_bufts.push_back(ggml_backend_dev_buffer_type(devs[id + 1]));
    }
    overflow_bufts.push_back(ggml_backend_cpu_buffer_type());

    std::vector<ngl_t> ngl_per_device(nd);
    std::vector<int64_t> mem = get_memory_for_layers(__func__, ngl_per_device, overflow_bufts, partial_moe);
    if (hp_nex > 0) {
        for (size_t id = 0; id < nd; id++) {
            ngl_per_device[id].overflow_type = LAYER_FRACTION_MOE;
        }
    }

    // optimize the number of layers per device using the method of false position:
    //   - ngl_per_device has 0 layers for each device, lower bound
    //   - try a "high" configuration where a device is given all unassigned layers
    //   - interpolate the memory use / layer between low and high linearly to get a guess where it meets our target
    //   - check memory use of our guess, replace either the low or high bound
    //   - once we only have a difference of a single layer, stop and return the lower bound that just barely still fits
    if (hp_nex == 0) {
        LLAMA_LOG_INFO("%s: filling dense layers back-to-front:\n", __func__);
    } else {
        LLAMA_LOG_INFO("%s: filling dense-only layers back-to-front:\n", __func__);
    }
    uint32_t n_unassigned = hp_ngl;
    for (int id = nd - 1; id >= 0; id--) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        ngl_per_device_high[id].n_layer = n_unassigned;
        if (hp_nex > 0) {
            ngl_per_device_high[id].n_part = ngl_per_device_high[id].n_layer;
        }
        if (ngl_per_device_high[id].n_layer > 0) {
            std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts, partial_moe);
            if (mem_high[id] > targets[id]) {
                uint32_t delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                while (delta > 1) {
                    uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                    step_size = std::max(step_size, uint32_t(1));
                    step_size = std::min(step_size, delta - 1);

                    std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                    ngl_per_device_test[id].n_layer += step_size;
                    if (hp_nex) {
                        ngl_per_device_test[id].n_part += step_size;
                    }
                    const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts, partial_moe);

                    if (mem_test[id] <= targets[id]) {
                        ngl_per_device  = ngl_per_device_test;
                        mem             = mem_test;
                        n_unassigned   -= ngl_per_device[id].n_layer;
                        LLAMA_LOG_DEBUG("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    } else {
                        ngl_per_device_high = ngl_per_device_test;
                        mem_high            = mem_test;
                        LLAMA_LOG_DEBUG("%s: set ngl_per_device_high[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
                    }
                    delta = ngl_per_device_high[id].n_layer - ngl_per_device[id].n_layer;
                }
            } else {
                ngl_per_device  = ngl_per_device_high;
                n_unassigned   -= ngl_per_device[id].n_layer;
                LLAMA_LOG_DEBUG("%s: set ngl_per_device[%d].n_layer=%" PRIu32 "\n", __func__, id, ngl_per_device[id].n_layer);
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LLAMA_LOG_INFO(
            "%s:   - %s: %2" PRIu32 " layers, %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, mem[id]/MiB, projected_margin/MiB);
    }
    if (hp_nex == 0 || global_surplus_cpu_moe <= 0) {
        set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams, partial_moe);
        return;
    }

    // step 4: for a MoE model where all dense tensors fit,
    //     convert the dense-only layers in the back to full layers in the front until all devices are full
    // essentially the same procedure as for the dense-only layers except front-to-back
    // also, try fitting at least part of one more layer to reduce waste for "small" GPUs with e.g. 24 GiB VRAM

    size_t id_dense_start = nd;
    for (int id = nd - 1; id >= 0; id--) {
        if (ngl_per_device[id].n_layer > 0) {
            id_dense_start = id;
            continue;
        }
        break;
    }
    assert(id_dense_start < nd);

    LLAMA_LOG_INFO("%s: converting dense-only layers to full layers and filling them front-to-back with overflow to next device/system memory:\n", __func__);
    for (size_t id = 0; id <= id_dense_start; id++) {
        std::vector<ngl_t> ngl_per_device_high = ngl_per_device;
        for (size_t jd = id_dense_start; jd < nd; jd++) {
            const uint32_t n_layer_move = ngl_per_device_high[jd].n_layer;
            ngl_per_device_high[id].n_layer += n_layer_move;
            ngl_per_device_high[jd].n_layer -= n_layer_move;
            ngl_per_device_high[jd].n_part = 0;
        }
        size_t id_dense_start_high = nd - 1;
        std::vector<int64_t> mem_high = get_memory_for_layers(__func__, ngl_per_device_high, overflow_bufts, partial_moe);

        if (mem_high[id] > targets[id]) {
            assert(ngl_per_device_high[id].n_layer >= ngl_per_device_high[id].n_part);
            assert(ngl_per_device[id].n_layer >= ngl_per_device[id].n_part);
            assert((ngl_per_device_high[id].n_layer - ngl_per_device_high[id].n_part)
                   >= ngl_per_device[id].n_layer - ngl_per_device[id].n_part);
            uint32_t delta = (ngl_per_device_high[id].n_layer - ngl_per_device_high[id].n_part)
                - (ngl_per_device[id].n_layer - ngl_per_device[id].n_part);
            while (delta > 1) {
                uint32_t step_size = int64_t(delta) * (targets[id] - mem[id]) / (mem_high[id] - mem[id]);
                step_size = std::max(step_size, uint32_t(1));
                step_size = std::min(step_size, delta - 1);

                std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
                size_t id_dense_start_test = id_dense_start;
                uint32_t n_converted_test = 0;
                for (;id_dense_start_test < nd; id_dense_start_test++) {
                    const uint32_t n_convert_jd = std::min(step_size - n_converted_test, ngl_per_device_test[id_dense_start_test].n_part);
                    ngl_per_device_test[id_dense_start_test].n_layer -= n_convert_jd;
                    ngl_per_device_test[id_dense_start_test].n_part -= n_convert_jd;
                    ngl_per_device_test[id].n_layer += n_convert_jd;
                    n_converted_test += n_convert_jd;

                    if (ngl_per_device_test[id_dense_start_test].n_layer > 0) {
                        break;
                    }
                }
                const std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts, partial_moe);

                if (mem_test[id] <= targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LLAMA_LOG_DEBUG("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                } else {
                    ngl_per_device_high = ngl_per_device_test;
                    mem_high            = mem_test;
                    id_dense_start_high = id_dense_start_test;
                    LLAMA_LOG_DEBUG("%s: set ngl_per_device_high[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start_high=%zu\n",
                        __func__, id, ngl_per_device_high[id].n_layer, ngl_per_device_high[id].n_part, id_dense_start_high);
                }
                delta = (ngl_per_device_high[id].n_layer - ngl_per_device_high[id].n_part)
                    - (ngl_per_device[id].n_layer - ngl_per_device[id].n_part);
            }
        } else {
            ngl_per_device = ngl_per_device_high;
            id_dense_start = id_dense_start_high;
            LLAMA_LOG_DEBUG("%s: set ngl_per_device[%zu].(n_layer, n_part)=(%" PRIu32 ", %" PRIu32 "), id_dense_start=%zu\n",
                __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
        }

        // try to fit at least part of one more layer
        if (ngl_per_device[id_dense_start].n_layer > 0) {
            std::vector<ngl_t> ngl_per_device_test = ngl_per_device;
            size_t id_dense_start_test = id_dense_start;
            ngl_per_device_test[id_dense_start_test].n_layer--;
            ngl_per_device_test[id_dense_start_test].n_part--;
            ngl_per_device_test[id].n_layer++;
            ngl_per_device_test[id].n_part++;
            if (ngl_per_device_test[id_dense_start_test].n_layer == 0) {
                id_dense_start_test++;
            }
            ngl_per_device_test[id].overflow_type = LAYER_FRACTION_UP;
            LLAMA_LOG_DEBUG("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_UP\n", __func__);
            std::vector<int64_t> mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts, partial_moe);
            if (mem_test[id] < targets[id]) {
                ngl_per_device = ngl_per_device_test;
                mem            = mem_test;
                id_dense_start = id_dense_start_test;
                LLAMA_LOG_DEBUG("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", UP), id_dense_start=%zu\n",
                    __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);

                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_GATE;
                LLAMA_LOG_DEBUG("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_GATE\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts, partial_moe);
                if (mem_test[id] < targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LLAMA_LOG_DEBUG("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", GATE), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            } else {
                ngl_per_device_test[id].overflow_type = LAYER_FRACTION_ATTN;
                LLAMA_LOG_DEBUG("%s: trying to fit one extra layer with overflow_type=LAYER_FRACTION_ATTN\n", __func__);
                mem_test = get_memory_for_layers(__func__, ngl_per_device_test, overflow_bufts, partial_moe);
                if (mem_test[id] < targets[id]) {
                    ngl_per_device = ngl_per_device_test;
                    mem            = mem_test;
                    id_dense_start = id_dense_start_test;
                    LLAMA_LOG_DEBUG("%s: set ngl_per_device[%zu].(n_layer, n_part, overflow_type)=(%" PRIu32 ", %" PRIu32 ", ATTN), id_dense_start=%zu\n",
                        __func__, id, ngl_per_device[id].n_layer, ngl_per_device[id].n_part, id_dense_start);
                }
            }
        }

        const int64_t projected_margin = dmds_full[id].free - mem[id];
        LLAMA_LOG_INFO(
            "%s:   - %s: %2" PRIu32 " layers (%2" PRIu32 " overflowing), %6" PRId64 " MiB used, %6" PRId64 " MiB free\n",
            __func__, dev_names[id].c_str(), ngl_per_device[id].n_layer, ngl_per_device[id].n_part, mem[id]/MiB, projected_margin/MiB);
    }

    set_ngl_tensor_split_tbo(ngl_per_device, overflow_bufts, *mparams, partial_moe);
}

bool llama_params_fit(
        const char * path_model, struct llama_model_params * mparams, struct llama_context_params * cparams,
        float * tensor_split, struct llama_model_tensor_buft_override * tensor_buft_overrides,
        size_t margin_s, uint32_t n_ctx_min, enum ggml_log_level log_level) {
    const int64_t t0_us = llama_time_us();
    bool ok = true;
    try {
        llama_params_fit_impl(path_model, mparams, cparams, tensor_split, tensor_buft_overrides, margin_s, n_ctx_min, log_level);
        LLAMA_LOG_INFO("%s: successfully fit params to free device memory\n", __func__);
    } catch (const std::runtime_error & e) {
        LLAMA_LOG_WARN("%s: failed to fit params to free device memory: %s\n", __func__, e.what());
        ok = false;
    }
    const int64_t t1_us = llama_time_us();
    LLAMA_LOG_INFO("%s: fitting params to free memory took %.2f seconds\n", __func__, (t1_us - t0_us) * 1e-6);
    return ok;
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

size_t llama_max_tensor_buft_overrides() {
    return 4096;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.no_alloc, params.kv_overrides, params.tensor_buft_overrides);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;
        model.hparams.no_alloc   = params.no_alloc;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        if (model.arch == LLM_ARCH_CLIP) {
            throw std::runtime_error("CLIP cannot be used as main model, use it with --mmproj instead");
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        std::vector<ggml_backend_dev_t> rpc_servers;

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU: {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        // check if there is already a GPU with the same device id
                        ggml_backend_dev_props props;
                        ggml_backend_dev_get_props(dev, &props);
                        auto it = std::find_if(gpus.begin(), gpus.end(), [&props](ggml_backend_dev_t d) {
                            ggml_backend_dev_props d_props;
                            ggml_backend_dev_get_props(d, &d_props);
                            if (props.device_id && d_props.device_id) {
                                return strcmp(props.device_id, d_props.device_id) == 0;
                            }
                            return false;
                        });

                        if (it != gpus.end()) {
                            LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                    __func__,
                                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                    props.device_id ? props.device_id : "unknown id",
                                    ggml_backend_dev_name(*it), ggml_backend_dev_description(*it));
                        } else {
                            gpus.push_back(dev);
                        }
                    }
                    break;
                }

                case GGML_BACKEND_DEVICE_TYPE_IGPU:
                    igpus.push_back(dev);
                    break;
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        size_t memory_free, memory_total;
        ggml_backend_dev_memory(dev, &memory_free, &memory_total);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                props.device_id ? props.device_id : "unknown id",
                memory_free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    splits.reserve(n_paths);
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(*model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}


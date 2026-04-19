#include "llama-ollama-compat-util.h"

#include "llama-impl.h"
#include "llama-model-loader.h"

#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace llama_ollama_compat::detail {

// -------------------------------------------------------------------------
// gguf_context KV helpers
// -------------------------------------------------------------------------

bool has_key(const gguf_context * meta, const char * key) {
    return gguf_find_key(meta, key) >= 0;
}

void copy_u32_kv(gguf_context * meta, const char * src, const char * dst) {
    if (has_key(meta, dst)) return;
    const int64_t k = gguf_find_key(meta, src);
    if (k < 0) return;
    gguf_set_val_u32(meta, dst, gguf_get_val_u32(meta, k));
}

void copy_f32_kv(gguf_context * meta, const char * src, const char * dst) {
    if (has_key(meta, dst)) return;
    const int64_t k = gguf_find_key(meta, src);
    if (k < 0) return;
    gguf_set_val_f32(meta, dst, gguf_get_val_f32(meta, k));
}

void copy_kv(gguf_context * meta, const char * src, const char * dst) {
    if (has_key(meta, dst)) return;
    const int64_t kid = gguf_find_key(meta, src);
    if (kid < 0) return;
    const enum gguf_type t = gguf_get_kv_type(meta, kid);
    switch (t) {
        case GGUF_TYPE_UINT8:   gguf_set_val_u8  (meta, dst, gguf_get_val_u8  (meta, kid)); break;
        case GGUF_TYPE_INT8:    gguf_set_val_i8  (meta, dst, gguf_get_val_i8  (meta, kid)); break;
        case GGUF_TYPE_UINT16:  gguf_set_val_u16 (meta, dst, gguf_get_val_u16 (meta, kid)); break;
        case GGUF_TYPE_INT16:   gguf_set_val_i16 (meta, dst, gguf_get_val_i16 (meta, kid)); break;
        case GGUF_TYPE_UINT32:  gguf_set_val_u32 (meta, dst, gguf_get_val_u32 (meta, kid)); break;
        case GGUF_TYPE_INT32:   gguf_set_val_i32 (meta, dst, gguf_get_val_i32 (meta, kid)); break;
        case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (meta, dst, gguf_get_val_f32 (meta, kid)); break;
        case GGUF_TYPE_BOOL:    gguf_set_val_bool(meta, dst, gguf_get_val_bool(meta, kid)); break;
        case GGUF_TYPE_STRING:  gguf_set_val_str (meta, dst, gguf_get_val_str (meta, kid)); break;
        case GGUF_TYPE_UINT64:  gguf_set_val_u64 (meta, dst, gguf_get_val_u64 (meta, kid)); break;
        case GGUF_TYPE_INT64:   gguf_set_val_i64 (meta, dst, gguf_get_val_i64 (meta, kid)); break;
        case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (meta, dst, gguf_get_val_f64 (meta, kid)); break;
        case GGUF_TYPE_ARRAY: {
            const enum gguf_type et = gguf_get_arr_type(meta, kid);
            const size_t n = gguf_get_arr_n(meta, kid);
            if (et == GGUF_TYPE_STRING) {
                std::vector<std::string> owned;
                owned.reserve(n);
                std::vector<const char *> ptrs;
                ptrs.reserve(n);
                for (size_t i = 0; i < n; ++i) owned.emplace_back(gguf_get_arr_str(meta, kid, i));
                for (const auto & s : owned) ptrs.push_back(s.c_str());
                gguf_set_arr_str(meta, dst, ptrs.data(), n);
            } else {
                gguf_set_arr_data(meta, dst, et, gguf_get_arr_data(meta, kid), n);
            }
            break;
        }
        default: break;
    }
}

void rename_kv_prefix(gguf_context * meta, const char * old_prefix,
                      const char * new_prefix) {
    const size_t old_len = std::strlen(old_prefix);
    // Snapshot keys first; copy_kv() invalidates the kv index by appending.
    std::vector<std::string> matches;
    const int64_t n = gguf_get_n_kv(meta);
    for (int64_t i = 0; i < n; ++i) {
        const char * k = gguf_get_key(meta, i);
        if (std::strncmp(k, old_prefix, old_len) == 0) matches.emplace_back(k);
    }
    for (const auto & old_key : matches) {
        copy_kv(meta, old_key.c_str(),
                (std::string(new_prefix) + old_key.substr(old_len)).c_str());
    }
}

void inject_u32_if_missing (gguf_context * meta, const char * key, uint32_t v) {
    if (!has_key(meta, key)) gguf_set_val_u32(meta, key, v);
}
void inject_f32_if_missing (gguf_context * meta, const char * key, float v) {
    if (!has_key(meta, key)) gguf_set_val_f32(meta, key, v);
}
void inject_str_if_missing (gguf_context * meta, const char * key, const char * v) {
    if (!has_key(meta, key)) gguf_set_val_str(meta, key, v);
}
void inject_bool_if_missing(gguf_context * meta, const char * key, bool v) {
    if (!has_key(meta, key)) gguf_set_val_bool(meta, key, v);
}
void inject_f32_arr_if_missing(gguf_context * meta, const char * key,
                               const float * data, size_t n) {
    if (!has_key(meta, key)) gguf_set_arr_data(meta, key, GGUF_TYPE_FLOAT32, data, n);
}

void truncate_str_arr(gguf_context * meta, const char * key, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || new_n >= gguf_get_arr_n(meta, kid)) return;

    std::vector<std::string> owned;
    owned.reserve(new_n);
    std::vector<const char *> ptrs;
    ptrs.reserve(new_n);
    for (size_t i = 0; i < new_n; ++i) owned.emplace_back(gguf_get_arr_str(meta, kid, i));
    for (const auto & s : owned) ptrs.push_back(s.c_str());
    gguf_set_arr_str(meta, key, ptrs.data(), new_n);
}

void truncate_data_arr(gguf_context * meta, const char * key,
                       gguf_type elem_type, size_t elem_size, size_t new_n) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || new_n >= gguf_get_arr_n(meta, kid)) return;

    std::vector<uint8_t> copy(elem_size * new_n);
    std::memcpy(copy.data(), gguf_get_arr_data(meta, kid), elem_size * new_n);
    gguf_set_arr_data(meta, key, elem_type, copy.data(), new_n);
}

// -------------------------------------------------------------------------
// ggml_context tensor scans
// -------------------------------------------------------------------------

bool any_tensor_with_prefix(const ggml_context * ctx, const char * prefix) {
    const size_t plen = std::strlen(prefix);
    for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
        if (std::strncmp(ggml_get_name(t), prefix, plen) == 0) return true;
    }
    return false;
}

// -------------------------------------------------------------------------
// Tensor renaming / reshaping (mutates both contexts)
// -------------------------------------------------------------------------

// gguf_get_tensor_name returns a pointer into a mutable `char[GGML_MAX_NAME]`
// inside a std::vector element; the const on the return type is API
// courtesy, so writing through const_cast is defined.
void rename_tensor(gguf_context * meta, ggml_context * ctx,
                   const char * old_name, const char * new_name) {
    const int64_t id = gguf_find_tensor(meta, old_name);
    if (id < 0) return;
    if (char * p = const_cast<char *>(gguf_get_tensor_name(meta, id))) {
        std::strncpy(p, new_name, GGML_MAX_NAME - 1);
        p[GGML_MAX_NAME - 1] = '\0';
    }
    if (ggml_tensor * t = ggml_get_tensor(ctx, old_name)) ggml_set_name(t, new_name);
}

void rename_tensors_containing(gguf_context * meta, ggml_context * ctx,
                               const char * needle, const char * replacement) {
    std::vector<std::pair<std::string, std::string>> renames;
    const int64_t n = gguf_get_n_tensors(meta);
    const size_t needle_len = std::strlen(needle);
    for (int64_t i = 0; i < n; ++i) {
        std::string s(gguf_get_tensor_name(meta, i));
        const size_t pos = s.find(needle);
        if (pos == std::string::npos) continue;
        std::string ns = s;
        ns.replace(pos, needle_len, replacement);
        renames.emplace_back(std::move(s), std::move(ns));
    }
    for (const auto & [from, to] : renames) rename_tensor(meta, ctx, from.c_str(), to.c_str());
}

void set_tensor_type(ggml_tensor * t, ggml_type type) {
    t->type  = type;
    t->nb[0] = ggml_type_size(type);
    t->nb[1] = t->nb[0] * (t->ne[0] / ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; ++i) t->nb[i] = t->nb[i - 1] * t->ne[i - 1];
}

void set_tensor_shape(ggml_tensor * t, std::initializer_list<int64_t> shape) {
    int i = 0;
    for (auto v : shape) t->ne[i++] = v;
    for (; i < GGML_MAX_DIMS; ++i) t->ne[i] = 1;
    set_tensor_type(t, t->type);
}

// Rename an orphan tensor slot as a new synthesized tensor. See header for
// why this is the workaround of choice (clip's ctx_meta has no spare capacity).
bool reclaim_slot_as(gguf_context * meta, ggml_context * ctx,
                     const char * orphan_name, const char * new_name,
                     std::initializer_list<int64_t> shape, ggml_type type) {
    if (gguf_find_tensor(meta, orphan_name) < 0) return false;
    rename_tensor(meta, ctx, orphan_name, new_name);
    ggml_tensor * t = ggml_get_tensor(ctx, new_name);
    if (!t) return false;
    set_tensor_shape(t, shape);
    set_tensor_type (t, type);
    return true;
}

size_t tensor_file_offset(const gguf_context * meta, const char * name) {
    const int64_t id = gguf_find_tensor(meta, name);
    if (id < 0) return 0;
    return gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id);
}

// -------------------------------------------------------------------------
// Per-loader skip-prefix registry
// -------------------------------------------------------------------------

namespace {
std::mutex g_skip_mutex;
std::unordered_map<const llama_model_loader *, std::vector<std::string>> g_skip_prefixes;
} // anon

void add_skip_prefix(const llama_model_loader * ml, std::string prefix) {
    std::lock_guard<std::mutex> lk(g_skip_mutex);
    g_skip_prefixes[ml].push_back(std::move(prefix));
}

bool should_skip_tensor_prefix(const llama_model_loader * ml, const char * name) {
    std::lock_guard<std::mutex> lk(g_skip_mutex);
    auto it = g_skip_prefixes.find(ml);
    if (it == g_skip_prefixes.end()) return false;
    for (const auto & prefix : it->second) {
        if (std::strncmp(name, prefix.c_str(), prefix.size()) == 0) return true;
    }
    return false;
}

namespace {
std::mutex g_no_mmap_mutex;
std::unordered_set<const llama_model_loader *> g_no_mmap;
} // anon

void disable_mmap_for(const llama_model_loader * ml) {
    std::lock_guard<std::mutex> lk(g_no_mmap_mutex);
    g_no_mmap.insert(ml);
}

bool is_mmap_disabled_for(const llama_model_loader * ml) {
    std::lock_guard<std::mutex> lk(g_no_mmap_mutex);
    return g_no_mmap.count(ml) > 0;
}

// -------------------------------------------------------------------------
// Load-time transform registry
// -------------------------------------------------------------------------

namespace {
std::mutex g_loadop_mutex;
std::unordered_map<std::string, LoadOp> g_loadops;
} // anon

void register_load_op(std::string dest_name, LoadOp op) {
    std::lock_guard<std::mutex> lk(g_loadop_mutex);
    g_loadops[std::move(dest_name)] = std::move(op);
}

bool take_load_op(const char * dest_name, LoadOp & out) {
    std::lock_guard<std::mutex> lk(g_loadop_mutex);
    auto it = g_loadops.find(dest_name);
    if (it == g_loadops.end()) return false;
    out = std::move(it->second);
    g_loadops.erase(it);
    return true;
}

bool read_at(const char * path, size_t offset, void * dst, size_t size) {
    FILE * f = std::fopen(path, "rb");
    if (!f) return false;
    bool ok = (std::fseek(f, (long) offset, SEEK_SET) == 0
               && std::fread(dst, 1, size, f) == size);
    std::fclose(f);
    return ok;
}

// -------------------------------------------------------------------------
// Common high-level transforms
// -------------------------------------------------------------------------

void promote_tensor_to_f32(gguf_context * meta, ggml_context * ctx, const char * name) {
    const int64_t tid = gguf_find_tensor(meta, name);
    if (tid < 0) return;
    ggml_tensor * t = ggml_get_tensor(ctx, name);
    if (!t || t->type != GGML_TYPE_F16) return;

    const size_t src_offset = tensor_file_offset(meta, name);
    const size_t n_elem     = ggml_nelements(t);
    const size_t src_size   = n_elem * sizeof(uint16_t);

    set_tensor_type(t, GGML_TYPE_F32);

    register_load_op(name, LoadOp{
        [src_offset, src_size, n_elem](const char * path, void * dst, size_t dst_size) {
            (void) dst_size;
            std::vector<uint8_t> src(src_size);
            if (!read_at(path, src_offset, src.data(), src_size)) return false;
            const uint16_t * sp = reinterpret_cast<const uint16_t *>(src.data());
            float          * dp = reinterpret_cast<float *>(dst);
            for (size_t i = 0; i < n_elem; ++i) dp[i] = ggml_fp16_to_fp32(sp[i]);
            return true;
        },
        "F16->F32 promote",
    });
}

void register_concat_load(const gguf_context * meta, std::string dest_name,
                          const std::vector<std::string> & src_names) {
    std::vector<std::pair<size_t, size_t>> regions;
    regions.reserve(src_names.size());
    for (const auto & n : src_names) {
        const int64_t id = gguf_find_tensor(meta, n.c_str());
        if (id < 0) return;
        regions.emplace_back(
            gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id),
            gguf_get_tensor_size(meta, id));
    }
    register_load_op(std::move(dest_name), LoadOp{
        [regions](const char * path, void * dst, size_t dst_size) {
            size_t total = 0;
            for (auto & [_, sz] : regions) total += sz;
            if (total != dst_size) return false;
            uint8_t * p = static_cast<uint8_t *>(dst);
            for (auto & [off, sz] : regions) {
                if (!read_at(path, off, p, sz)) return false;
                p += sz;
            }
            return true;
        },
        "concat sources",
    });
}

void register_concat_load_to_f32(const gguf_context * meta,
                                 const ggml_context * ctx,
                                 std::string dest_name,
                                 const std::vector<std::string> & src_names) {
    struct Region { size_t offset; size_t size; ggml_type type; size_t n_elem; };
    std::vector<Region> regions;
    regions.reserve(src_names.size());
    for (const auto & n : src_names) {
        const int64_t id = gguf_find_tensor(meta, n.c_str());
        if (id < 0) return;
        const ggml_tensor * t = ggml_get_tensor(const_cast<ggml_context *>(ctx), n.c_str());
        if (!t) return;
        regions.push_back({
            gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, id),
            gguf_get_tensor_size(meta, id),
            t->type,
            (size_t) ggml_nelements(t),
        });
    }
    register_load_op(std::move(dest_name), LoadOp{
        [regions](const char * path, void * dst, size_t dst_size) {
            size_t total_elems = 0;
            for (auto & r : regions) total_elems += r.n_elem;
            if (total_elems * sizeof(float) != dst_size) return false;

            float * dp = static_cast<float *>(dst);
            for (auto & r : regions) {
                std::vector<uint8_t> src(r.size);
                if (!read_at(path, r.offset, src.data(), r.size)) return false;
                const auto * tt = ggml_get_type_traits(r.type);
                if (!tt || !tt->to_float) return false;
                tt->to_float(src.data(), dp, (int64_t) r.n_elem);
                dp += r.n_elem;
            }
            return true;
        },
        "concat sources (mixed types -> F32)",
    });
}

} // namespace llama_ollama_compat::detail

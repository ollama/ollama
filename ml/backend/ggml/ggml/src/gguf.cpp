#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "gguf.h"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

template <typename T>
struct type_to_gguf_type;

template <>
struct type_to_gguf_type<uint8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
};

template <>
struct type_to_gguf_type<int8_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT8;
};

template <>
struct type_to_gguf_type<uint16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
};

template <>
struct type_to_gguf_type<int16_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT16;
};

template <>
struct type_to_gguf_type<uint32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
};

template <>
struct type_to_gguf_type<int32_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT32;
};

template <>
struct type_to_gguf_type<float> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
struct type_to_gguf_type<bool> {
    static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
};

template <>
struct type_to_gguf_type<std::string> {
    static constexpr enum gguf_type value = GGUF_TYPE_STRING;
};

template <>
struct type_to_gguf_type<uint64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
};

template <>
struct type_to_gguf_type<int64_t> {
    static constexpr enum gguf_type value = GGUF_TYPE_INT64;
};

template <>
struct type_to_gguf_type<double> {
    static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8,   sizeof(uint8_t)},
    {GGUF_TYPE_INT8,    sizeof(int8_t)},
    {GGUF_TYPE_UINT16,  sizeof(uint16_t)},
    {GGUF_TYPE_INT16,   sizeof(int16_t)},
    {GGUF_TYPE_UINT32,  sizeof(uint32_t)},
    {GGUF_TYPE_INT32,   sizeof(int32_t)},
    {GGUF_TYPE_FLOAT32, sizeof(float)},
    {GGUF_TYPE_BOOL,    sizeof(int8_t)},
    {GGUF_TYPE_STRING,  0}, // undefined
    {GGUF_TYPE_ARRAY,   0}, // undefined
    {GGUF_TYPE_UINT64,  sizeof(uint64_t)},
    {GGUF_TYPE_INT64,   sizeof(int64_t)},
    {GGUF_TYPE_FLOAT64, sizeof(double)},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const char *> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8,   "u8"},
    {GGUF_TYPE_INT8,    "i8"},
    {GGUF_TYPE_UINT16,  "u16"},
    {GGUF_TYPE_INT16,   "i16"},
    {GGUF_TYPE_UINT32,  "u32"},
    {GGUF_TYPE_INT32,   "i32"},
    {GGUF_TYPE_FLOAT32, "f32"},
    {GGUF_TYPE_BOOL,    "bool"},
    {GGUF_TYPE_STRING,  "str"},
    {GGUF_TYPE_ARRAY,   "arr"},
    {GGUF_TYPE_UINT64,  "u64"},
    {GGUF_TYPE_INT64,   "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

size_t gguf_type_size(enum gguf_type type) {
    auto it = GGUF_TYPE_SIZE.find(type);
    return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

struct gguf_kv {
    std::string key;

    bool is_array;
    enum gguf_type type;

    std::vector<int8_t>      data;
    std::vector<std::string> data_string;

    template <typename T>
    gguf_kv(const std::string & key, const T value)
            : key(key), is_array(false), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(sizeof(T));
        memcpy(data.data(), &value, sizeof(T));
    }

    template <typename T>
    gguf_kv(const std::string & key, const std::vector<T> & value)
            : key(key), is_array(true), type(type_to_gguf_type<T>::value) {
        GGML_ASSERT(!key.empty());
        data.resize(value.size()*sizeof(T));
        for (size_t i = 0; i < value.size(); ++i) {
            const T tmp = value[i];
            memcpy(data.data() + i*sizeof(T), &tmp, sizeof(T));
        }
    }

    gguf_kv(const std::string & key, const std::string & value)
            : key(key), is_array(false), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string.push_back(value);
    }

    gguf_kv(const std::string & key, const std::vector<std::string> & value)
            : key(key), is_array(true), type(GGUF_TYPE_STRING) {
        GGML_ASSERT(!key.empty());
        data_string = value;
    }

    const std::string & get_key() const {
        return key;
    }

    const enum gguf_type & get_type() const {
        return type;
    }

    size_t get_ne() const {
        if (type == GGUF_TYPE_STRING) {
            const size_t ne = data_string.size();
            GGML_ASSERT(is_array || ne == 1);
            return ne;
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        const size_t ne = data.size() / type_size;
        GGML_ASSERT(is_array || ne == 1);
        return ne;
    }

    template <typename T>
    const T & get_val(const size_t i = 0) const {
        GGML_ASSERT(type_to_gguf_type<T>::value == type);
        if constexpr (std::is_same<T, std::string>::value) {
            GGML_ASSERT(data_string.size() >= i+1);
            return data_string[i];
        }
        const size_t type_size = gguf_type_size(type);
        GGML_ASSERT(data.size() % type_size == 0);
        GGML_ASSERT(data.size() >= (i+1)*type_size);
        return reinterpret_cast<const T *>(data.data())[i];
    }

    void cast(const enum gguf_type new_type) {
        const size_t new_type_size = gguf_type_size(new_type);
        GGML_ASSERT(data.size() % new_type_size == 0);
        type = new_type;
    }
};

struct gguf_tensor_info {
    struct ggml_tensor t; // for holding the equivalent info
    uint64_t offset;      // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context {
    uint32_t version = GGUF_VERSION;

    std::vector<struct gguf_kv> kv;
    std::vector<struct gguf_tensor_info> info;

    size_t alignment = GGUF_DEFAULT_ALIGNMENT;
    size_t offset    = 0; // offset of `data` from beginning of file
    size_t size      = 0; // size of `data` in bytes

    void * data = nullptr;
};

struct gguf_reader {
    FILE * file;

    gguf_reader(FILE * file) : file(file) {}

    template <typename T>
    bool read(T & dst) const {
        return fread(&dst, 1, sizeof(dst), file) == sizeof(dst);
    }

    template <typename T>
    bool read(std::vector<T> & dst, const size_t n) const {
        dst.resize(n);
        for (size_t i = 0; i < dst.size(); ++i) {
            if constexpr (std::is_same<T, bool>::value) {
                bool tmp;
                if (!read(tmp)) {
                    return false;
                }
                dst[i] = tmp;
            } else {
                if (!read(dst[i])) {
                    return false;
                }
            }
        }
        return true;
    }

    bool read(bool & dst) const {
        int8_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = tmp != 0;
        return true;
    }

    bool read(enum ggml_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = ggml_type(tmp);
        return true;
    }

    bool read(enum gguf_type & dst) const {
        int32_t tmp = -1;
        if (!read(tmp)) {
            return false;
        }
        dst = gguf_type(tmp);
        return true;
    }

    bool read(std::string & dst) const {
        uint64_t size = -1;
        if (!read(size)) {
            return false;
        }
        dst.resize(size);
        return fread(dst.data(), 1, dst.length(), file) == dst.length();
    }

    bool read(void * dst, const size_t size) const {
        return fread(dst, 1, size, file) == size;
    }
};

struct gguf_context * gguf_init_empty(void) {
    return new gguf_context;
}

template<typename T>
bool gguf_read_emplace_helper(const struct gguf_reader & gr, std::vector<struct gguf_kv> & kv, const std::string & key, const bool is_array, const size_t n) {
    if (is_array) {
        std::vector<T> value;
        try {
            if (!gr.read(value, n)) {
                return false;
            }
        } catch (std::length_error &) {
            fprintf(stderr, "%s: encountered length_error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        } catch (std::bad_alloc &) {
            fprintf(stderr, "%s: encountered bad_alloc error while reading value for key '%s'\n", __func__, key.c_str());
            return false;
        }
        kv.emplace_back(key, value);
    } else {
        T value;
        if (!gr.read(value)) {
            return false;
        }
        kv.emplace_back(key, value);
    }
    return true;
}

struct gguf_context * gguf_init_from_file_impl(FILE * file, struct gguf_init_params params) {
    const struct gguf_reader gr(file);
    struct gguf_context * ctx = new gguf_context;

    bool ok = true;

    // file magic
    {
        std::vector<char> magic;
        ok = ok && gr.read(magic, 4);

        if (!ok) {
            fprintf(stderr, "%s: failed to read magic\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        for (uint32_t i = 0; i < magic.size(); i++) {
            if (magic[i] != GGUF_MAGIC[i]) {
                fprintf(stderr, "%s: invalid magic characters: '%c%c%c%c', expected 'GGUF'\n", __func__, magic[0], magic[1], magic[2], magic[3]);
                gguf_free(ctx);
                return nullptr;
            }
        }
    }

    // header
    int64_t n_kv      = 0;
    int64_t n_tensors = 0;

    if (ok && gr.read(ctx->version)) {
        if (ctx->version == 1) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported, please use a more up-to-date version\n", __func__);
            ok = false;
        }
        if (ctx->version > GGUF_VERSION) {
            fprintf(stderr, "%s: this GGUF file is version %" PRIu32 " but this software only supports up to version %d\n",
                __func__, ctx->version, GGUF_VERSION);
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_tensors)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_tensors < 0 || n_tensors > int64_t(SIZE_MAX/sizeof(gguf_tensor_info))) {
            fprintf(stderr, "%s: number of tensors is %" PRIi64 " but must be in [0, %zu]\n",
                __func__, n_tensors, SIZE_MAX/sizeof(gguf_tensor_info));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (ok && gr.read(n_kv)) {
        static_assert(sizeof(size_t) <= 8 && sizeof(gguf_tensor_info) >= 2, "int64_t insufficient for indexing");
        if (n_kv < 0 || n_kv > int64_t(SIZE_MAX/sizeof(gguf_kv))) {
            fprintf(stderr, "%s: number of key value pairs is %" PRIi64 " but must be in [0, %zu]\n",
                    __func__, n_kv, SIZE_MAX/sizeof(gguf_kv));
            ok = false;
        }
    } else {
        ok = false;
    }

    if (!ok) {
        fprintf(stderr, "%s: failed to read header\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // KV pairs
    {
        for (int64_t i = 0; ok && i < n_kv; ++i) {
            std::string key;
            gguf_type   type     = gguf_type(-1);
            bool        is_array = false;
            uint64_t    n        = 1;

            try {
                ok = ok && gr.read(key);
            } catch (std::length_error &) {
                fprintf(stderr, "%s: encountered length_error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                fprintf(stderr, "%s: encountered bad_alloc error while reading key %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            for (size_t j = 0; ok && j < ctx->kv.size(); ++j) {
                if (key == ctx->kv[j].key) {
                    fprintf(stderr, "%s: duplicate key '%s' for tensors %zu and %" PRIi64 " \n", __func__, key.c_str(), j, i);
                    ok = false;
                }
            }
            if (!ok) {
                break;
            }

            ok = ok && gr.read(type);
            if (type == GGUF_TYPE_ARRAY) {
                is_array = true;
                ok = ok && gr.read(type);
                ok = ok && gr.read(n);
            }
            if (!ok) {
                break;
            }

            switch (type) {
                case GGUF_TYPE_UINT8:   ok = ok && gguf_read_emplace_helper<uint8_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT8:    ok = ok && gguf_read_emplace_helper<int8_t>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT16:  ok = ok && gguf_read_emplace_helper<uint16_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT16:   ok = ok && gguf_read_emplace_helper<int16_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT32:  ok = ok && gguf_read_emplace_helper<uint32_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT32:   ok = ok && gguf_read_emplace_helper<int32_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT32: ok = ok && gguf_read_emplace_helper<float>      (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_BOOL:    ok = ok && gguf_read_emplace_helper<bool>       (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_STRING:  ok = ok && gguf_read_emplace_helper<std::string>(gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_UINT64:  ok = ok && gguf_read_emplace_helper<uint64_t>   (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_INT64:   ok = ok && gguf_read_emplace_helper<int64_t>    (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_FLOAT64: ok = ok && gguf_read_emplace_helper<double>     (gr, ctx->kv, key, is_array, n); break;
                case GGUF_TYPE_ARRAY:
                default:
                    {
                        fprintf(stderr, "%s: key '%s' has invalid GGUF type %d\n", __func__, key.c_str(), type);
                        ok = false;
                    } break;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }
        GGML_ASSERT(int64_t(ctx->kv.size()) == n_kv);

        const int alignment_idx = gguf_find_key(ctx, GGUF_KEY_GENERAL_ALIGNMENT);
        ctx->alignment = alignment_idx == -1 ? GGUF_DEFAULT_ALIGNMENT : gguf_get_val_u32(ctx, alignment_idx);

        if (ctx->alignment == 0 || (ctx->alignment & (ctx->alignment - 1)) != 0) {
            fprintf(stderr, "%s: alignment %zu is not a power of 2\n", __func__, ctx->alignment);
            gguf_free(ctx);
            return nullptr;
        }
    }

    // read the tensor info
    for (int64_t i = 0; ok && i < n_tensors; ++i) {
        struct gguf_tensor_info info;

        // tensor name
        {
            std::string name;
            try {
                ok = ok && gr.read(name);
            } catch (std::length_error &) {
                fprintf(stderr, "%s: encountered length_error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            } catch (std::bad_alloc &) {
                fprintf(stderr, "%s: encountered bad_alloc error while reading tensor name %" PRIi64 "\n", __func__, i);
                ok = false;
            }
            if (name.length() >= GGML_MAX_NAME) {
                fprintf(stderr, "%s: tensor name %" PRIi64 " is too long: %zu >= %d\n", __func__, i, name.length(), GGML_MAX_NAME);
                ok = false;
                break;
            }
            ggml_set_name(&info.t, name.c_str());

            // make sure there are no duplicate tensor names
            for (int64_t j = 0; ok && j < i; ++j) {
                if (strcmp(info.t.name, ctx->info[j].t.name) == 0) {
                    fprintf(stderr, "%s: duplicate tensor name '%s' for tensors %" PRIi64 " and %" PRIi64 "\n", __func__, info.t.name, j, i);
                    ok = false;
                    break;
                }
            }
        }
        if (!ok) {
            break;
        }

        // tensor shape
        {
            uint32_t n_dims = -1;
            ok = ok && gr.read(n_dims);
            if (n_dims > GGML_MAX_DIMS) {
                fprintf(stderr, "%s: tensor '%s' has invalid number of dimensions: %" PRIu32 " > %" PRIu32 "\n",
                    __func__, info.t.name, n_dims, GGML_MAX_DIMS);
                ok = false;
                break;
            }
            for (uint32_t j = 0; ok && j < GGML_MAX_DIMS; ++j) {
                info.t.ne[j] = 1;
                if (j < n_dims) {
                    ok = ok && gr.read(info.t.ne[j]);
                }

                // check that all ne are non-negative
                if (info.t.ne[j] < 0) {
                    fprintf(stderr, "%s: tensor '%s' dimension %" PRIu32 " has invalid number of elements: %" PRIi64 " < 0\n",
                        __func__, info.t.name, j, info.t.ne[j]);
                    ok = false;
                    break;
                }
            }

            // check that the total number of elements is representable
            if (ok && ((INT64_MAX/info.t.ne[1] <= info.t.ne[0]) ||
                       (INT64_MAX/info.t.ne[2] <= info.t.ne[0]*info.t.ne[1]) ||
                       (INT64_MAX/info.t.ne[3] <= info.t.ne[0]*info.t.ne[1]*info.t.ne[2]))) {

                fprintf(stderr, "%s: total number of elements in tensor '%s' with shape "
                    "(%" PRIi64 ", %" PRIi64 ", %" PRIi64 ", %" PRIi64 ") is >= %" PRIi64 "\n",
                    __func__, info.t.name, info.t.ne[0], info.t.ne[1], info.t.ne[2], info.t.ne[3], INT64_MAX);
                ok = false;
                break;
            }
        }
        if (!ok) {
            break;
        }

        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                fprintf(stderr, "%s: tensor '%s' has invalid ggml type %d (%s)\n",
                    __func__, info.t.name, info.t.type, ggml_type_name(info.t.type));
                ok = false;
                break;
            }
            const size_t  type_size = ggml_type_size(info.t.type);
            const int64_t blck_size = ggml_blck_size(info.t.type);

            // check that row size is divisible by block size
            if (blck_size == 0 || info.t.ne[0] % blck_size != 0) {
                fprintf(stderr, "%s: tensor '%s' of type %d (%s) has %" PRId64 " elements per row, "
                    "not a multiple of block size (%" PRId64 ")\n",
                    __func__, info.t.name, (int) info.t.type, ggml_type_name(info.t.type), info.t.ne[0], blck_size);
                ok = false;
                break;
            }

            // calculate byte offsets given the tensor shape and type
            info.t.nb[0] = type_size;
            info.t.nb[1] = info.t.nb[0]*(info.t.ne[0]/blck_size);
            for (int j = 2; j < GGML_MAX_DIMS; ++j) {
                info.t.nb[j] = info.t.nb[j - 1]*info.t.ne[j - 1];
            }
        }
        if (!ok) {
            break;
        }

        // tensor data offset within buffer
        ok = ok && gr.read(info.offset);

        ctx->info.push_back(info);
    }

    if (!ok) {
        fprintf(stderr, "%s: failed to read tensor info\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }
    GGML_ASSERT(int64_t(ctx->info.size()) == n_tensors);

    // we require the data section to be aligned, so take into account any padding
    if (fseek(file, GGML_PAD(ftell(file), ctx->alignment), SEEK_SET) != 0) {
        fprintf(stderr, "%s: failed to seek to beginning of data section\n", __func__);
        gguf_free(ctx);
        return nullptr;
    }

    // store the current file offset - this is where the data section starts
    ctx->offset = ftell(file);

    // compute the total size of the data section, taking into account the alignment
    {
        ctx->size = 0;
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const gguf_tensor_info & ti = ctx->info[i];
            if (ti.offset != ctx->size) {
                fprintf(stderr, "%s: tensor '%s' has offset %" PRIu64 ", expected %zu\n",
                    __func__, ti.t.name, ti.offset, ctx->size);
                fprintf(stderr, "%s: failed to read tensor data\n", __func__);
                gguf_free(ctx);
                return nullptr;
            }
            ctx->size += GGML_PAD(ggml_nbytes(&ti.t), ctx->alignment);
        }
    }

    // load the tensor data only if requested
    if (params.ctx != nullptr) {
        // if the provided gguf_context is no_alloc, then we create "empty" tensors and do not read the binary blob
        // otherwise, we load the binary blob into the created ggml_context as well, and point the "data" members of
        //   the ggml_tensor structs to the appropriate locations in the binary blob

        // compute the exact size needed for the new ggml_context
        const size_t mem_size =
            params.no_alloc ?
            (n_tensors    )*ggml_tensor_overhead() :
            (n_tensors + 1)*ggml_tensor_overhead() + ctx->size;

        struct ggml_init_params pdata = {
            /*mem_size   =*/ mem_size,
            /*mem_buffer =*/ nullptr,
            /*no_alloc   =*/ params.no_alloc,
        };

        *params.ctx = ggml_init(pdata);
        if (*params.ctx == nullptr) {
            fprintf(stderr, "%s: failed to initialize ggml context for storing tensors\n", __func__);
            gguf_free(ctx);
            return nullptr;
        }

        struct ggml_context * ctx_data = *params.ctx;

        struct ggml_tensor * data = nullptr;

        if (!params.no_alloc) {
            data = ggml_new_tensor_1d(ctx_data, GGML_TYPE_I8, ctx->size);

            ok = ok && data != nullptr;

            if (ok) {
                ggml_set_name(data, "GGUF tensor data binary blob");
            }

            // read the binary blob with the tensor data
            ok = ok && gr.read(data->data, ctx->size);

            if (!ok) {
                fprintf(stderr, "%s: failed to read tensor data binary blob\n", __func__);
                ggml_free(ctx_data);
                *params.ctx = nullptr;
                gguf_free(ctx);
                return nullptr;
            }

            ctx->data = data->data;
        }

        ggml_set_no_alloc(ctx_data, true);

        // create the tensors
        for (size_t i = 0; i < ctx->info.size(); ++i) {
            const struct gguf_tensor_info & info = ctx->info[i];

            struct ggml_tensor * cur = ggml_new_tensor(ctx_data, info.t.type, GGML_MAX_DIMS, info.t.ne);

            ok = ok && cur != nullptr;

            if (!ok) {
                break;
            }

            ggml_set_name(cur, info.t.name);

            // point the data member to the appropriate location in the binary blob using the tensor info
            if (!params.no_alloc) {
                cur->data = (char *) data->data + info.offset;
            }
        }

        if (!ok) {
            fprintf(stderr, "%s: failed to create tensors\n", __func__);
            ggml_free(ctx_data);
            *params.ctx = nullptr;
            gguf_free(ctx);
            return nullptr;
        }

        ggml_set_no_alloc(ctx_data, params.no_alloc);
    }

    return ctx;
}

struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params) {
    FILE * file = ggml_fopen(fname, "rb");

    if (!file) {
        fprintf(stderr, "%s: failed to open GGUF file '%s'\n", __func__, fname);
        return nullptr;
    }

    struct gguf_context * result = gguf_init_from_file_impl(file, params);
    fclose(file);
    return result;
}

void gguf_free(struct gguf_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    delete ctx;
}

const char * gguf_type_name(enum gguf_type type) {
    auto it = GGUF_TYPE_NAME.find(type);
    return it == GGUF_TYPE_NAME.end() ? nullptr : it->second;
}

uint32_t gguf_get_version(const struct gguf_context * ctx) {
    return ctx->version;
}

size_t gguf_get_alignment(const struct gguf_context * ctx) {
    return ctx->alignment;
}

size_t gguf_get_data_offset(const struct gguf_context * ctx) {
    return ctx->offset;
}

int64_t gguf_get_n_kv(const struct gguf_context * ctx) {
    return ctx->kv.size();
}

int64_t gguf_find_key(const struct gguf_context * ctx, const char * key) {
    // return -1 if key not found
    int64_t keyfound = -1;

    const int64_t n_kv = gguf_get_n_kv(ctx);

    for (int64_t i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

const char * gguf_get_key(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].get_key().c_str();
}

enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].is_array ? GGUF_TYPE_ARRAY : ctx->kv[key_id].get_type();
}

enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].is_array);
    return ctx->kv[key_id].get_type();
}

const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

const char * gguf_get_arr_str(const struct gguf_context * ctx, int64_t key_id, size_t i) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_type() == GGUF_TYPE_STRING);
    return ctx->kv[key_id].data_string[i].c_str();
}

size_t gguf_get_arr_n(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));

    if (ctx->kv[key_id].type == GGUF_TYPE_STRING) {
        return ctx->kv[key_id].data_string.size();
    }

    const size_t type_size = gguf_type_size(ctx->kv[key_id].type);
    GGML_ASSERT(ctx->kv[key_id].data.size() % type_size == 0);
    return ctx->kv[key_id].data.size() / type_size;
}

uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint8_t>();
}

int8_t gguf_get_val_i8(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int8_t>();
}

uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint16_t>();
}

int16_t gguf_get_val_i16(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int16_t>();
}

uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint32_t>();
}

int32_t gguf_get_val_i32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int32_t>();
}

float gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<float>();
}

uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<uint64_t>();
}

int64_t gguf_get_val_i64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<int64_t>();
}

double gguf_get_val_f64(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<double>();
}

bool gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<bool>();
}

const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    return ctx->kv[key_id].get_val<std::string>().c_str();
}

const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id) {
    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    GGML_ASSERT(ctx->kv[key_id].get_ne() == 1);
    GGML_ASSERT(ctx->kv[key_id].get_type() != GGUF_TYPE_STRING);
    return ctx->kv[key_id].data.data();
}

int64_t gguf_get_n_tensors(const struct gguf_context * ctx) {
    return ctx->info.size();
}

int64_t gguf_find_tensor(const struct gguf_context * ctx, const char * name) {
    // return -1 if tensor not found
    int64_t tensor_id = -1;

    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    for (int64_t i = 0; i < n_tensors; ++i) {
        if (strcmp(name, gguf_get_tensor_name(ctx, i)) == 0) {
            tensor_id = i;
            break;
        }
    }

    return tensor_id;
}

size_t gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].offset;
}

const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.name;
}

enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ctx->info[tensor_id].t.type;
}

size_t gguf_get_tensor_size(const struct gguf_context * ctx, int64_t tensor_id) {
    GGML_ASSERT(tensor_id >= 0 && tensor_id < gguf_get_n_tensors(ctx));
    return ggml_nbytes(&ctx->info[tensor_id].t);
}

int64_t gguf_remove_key(struct gguf_context * ctx, const char * key) {
    const int64_t key_id = gguf_find_key(ctx, key);
    if (key_id >= 0) {
        ctx->kv.erase(ctx->kv.begin() + key_id);
    }
    return key_id;
}

template<typename T>
static void gguf_check_reserved_keys(const std::string & key, const T val) {
    if (key == GGUF_KEY_GENERAL_ALIGNMENT) {
        if constexpr (std::is_same<T, uint32_t>::value) {
            GGML_ASSERT(val > 0 && (val & (val - 1)) == 0 && GGUF_KEY_GENERAL_ALIGNMENT " must be power of 2");
        } else {
            GGML_ABORT(GGUF_KEY_GENERAL_ALIGNMENT " must be type u32");
        }
    }
}

void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, val);
}

void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val) {
    gguf_check_reserved_keys(key, val);
    gguf_remove_key(ctx, key);
    ctx->kv.emplace_back(key, std::string(val));
}

void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    const size_t nbytes = n*gguf_type_size(type);
    std::vector<int8_t> tmp(nbytes);
    if (!tmp.empty()) {
        memcpy(tmp.data(), data, nbytes);
    }
    ctx->kv.emplace_back(key, tmp);
    ctx->kv.back().cast(type);
}

void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, size_t n) {
    gguf_check_reserved_keys(key, data);
    gguf_remove_key(ctx, key);

    std::vector<std::string> tmp(n);
    for (size_t i = 0; i < n; ++i) {
        tmp[i] = data[i];
    }
    ctx->kv.emplace_back(key, tmp);
}

// set or add KV pairs from another context
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src) {
    const int64_t n_kv = gguf_get_n_kv(src);
    for (int64_t i = 0; i < n_kv; ++i) {
        const struct gguf_kv & kv = src->kv[i];

        if (!kv.is_array) {
            switch (kv.get_type()) {
                case GGUF_TYPE_UINT8:   gguf_set_val_u8  (ctx, kv.get_key().c_str(), kv.get_val<uint8_t>());             break;
                case GGUF_TYPE_INT8:    gguf_set_val_i8  (ctx, kv.get_key().c_str(), kv.get_val<int8_t>());              break;
                case GGUF_TYPE_UINT16:  gguf_set_val_u16 (ctx, kv.get_key().c_str(), kv.get_val<uint16_t>());            break;
                case GGUF_TYPE_INT16:   gguf_set_val_i16 (ctx, kv.get_key().c_str(), kv.get_val<int16_t>());             break;
                case GGUF_TYPE_UINT32:  gguf_set_val_u32 (ctx, kv.get_key().c_str(), kv.get_val<uint32_t>());            break;
                case GGUF_TYPE_INT32:   gguf_set_val_i32 (ctx, kv.get_key().c_str(), kv.get_val<int32_t>());             break;
                case GGUF_TYPE_FLOAT32: gguf_set_val_f32 (ctx, kv.get_key().c_str(), kv.get_val<float>());               break;
                case GGUF_TYPE_UINT64:  gguf_set_val_u64 (ctx, kv.get_key().c_str(), kv.get_val<uint64_t>());            break;
                case GGUF_TYPE_INT64:   gguf_set_val_i64 (ctx, kv.get_key().c_str(), kv.get_val<int64_t>());             break;
                case GGUF_TYPE_FLOAT64: gguf_set_val_f64 (ctx, kv.get_key().c_str(), kv.get_val<double>());              break;
                case GGUF_TYPE_BOOL:    gguf_set_val_bool(ctx, kv.get_key().c_str(), kv.get_val<bool>());                break;
                case GGUF_TYPE_STRING:  gguf_set_val_str (ctx, kv.get_key().c_str(), kv.get_val<std::string>().c_str()); break;
                case GGUF_TYPE_ARRAY:
                default: GGML_ABORT("invalid type");
            }
            continue;
        }

        const size_t ne = kv.get_ne();

        switch (kv.get_type()) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64:
            case GGUF_TYPE_BOOL: {
                gguf_set_arr_data(ctx, kv.get_key().c_str(), kv.get_type(), kv.data.data(), ne);
            } break;
            case GGUF_TYPE_STRING: {
                std::vector<const char *> tmp(ne);
                for (size_t j = 0; j < ne; ++j) {
                    tmp[j] = kv.data_string[j].c_str();
                }
                gguf_set_arr_str(ctx, kv.get_key().c_str(), tmp.data(), ne);
            } break;
            case GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }
}

void gguf_add_tensor(
             struct gguf_context * ctx,
        const struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    if (gguf_find_tensor(ctx, tensor->name) != -1) {
        GGML_ABORT("duplicate tensor name: %s", tensor->name);
    }

    struct gguf_tensor_info ti;
    ti.t = *tensor;
    ti.offset = ctx->info.empty() ? 0 :
        ctx->info.back().offset + GGML_PAD(ggml_nbytes(&ctx->info.back().t), ctx->alignment);
    ctx->info.push_back(ti);
}

void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }
    struct ggml_tensor * tensor = &ctx->info[tensor_id].t;
    const size_t  type_size = ggml_type_size(type);
    const int64_t blck_size = ggml_blck_size(type);

    tensor->type = type;
    GGML_ASSERT(tensor->ne[0] % blck_size == 0 && "tensor row size not divisible by block size of new type");

    tensor->nb[0] = type_size;
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/blck_size);
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }

    // update offsets
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = tensor_id + 1; i < n_tensors; ++i) {
        ctx->info[i].offset = ctx->info[i - 1].offset + GGML_PAD(ggml_nbytes(&ctx->info[i - 1].t), ctx->alignment);
    }
}

void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data) {
    const int64_t tensor_id = gguf_find_tensor(ctx, name);
    if (tensor_id < 0) {
        GGML_ABORT("tensor not found: %s", name);
    }

    ctx->info[tensor_id].t.data = (void *)(uintptr_t)data; // double cast suppresses warning about casting away const
}

struct gguf_writer {
    std::vector<int8_t> & buf;

    gguf_writer(std::vector<int8_t> & buf) : buf(buf) {}

    template <typename T>
    void write(const T & val) const {
        for (size_t i = 0; i < sizeof(val); ++i) {
            buf.push_back(reinterpret_cast<const int8_t *>(&val)[i]);
        }
    }

    void write(const std::vector<int8_t> & val) const {
        buf.insert(buf.end(), val.begin(), val.end());
    }

    void write(const bool & val) const {
        const int8_t val8 = val ? 1 : 0;
        write(val8);
    }

    void write(const std::string & val) const {
        {
            const uint64_t n = val.length();
            write(n);
        }
        for (size_t i = 0; i < val.length(); ++i) {
            buf.push_back(reinterpret_cast<const int8_t *>(val.data())[i]);
        }
    }

    void write(const char * val) const {
        write(std::string(val));
    }

    void write(const enum ggml_type & val) const {
        write(int32_t(val));
    }

    void write(const enum gguf_type & val) const {
        write(int32_t(val));
    }

    void write(const struct gguf_kv & kv) const {
        const uint64_t ne = kv.get_ne();

        write(kv.get_key());

        if (kv.is_array) {
            write(GGUF_TYPE_ARRAY);
            write(kv.get_type());
            write(ne);
        } else {
            write(kv.get_type());
        }

        switch (kv.get_type()) {
            case GGUF_TYPE_UINT8:
            case GGUF_TYPE_INT8:
            case GGUF_TYPE_UINT16:
            case GGUF_TYPE_INT16:
            case GGUF_TYPE_UINT32:
            case GGUF_TYPE_INT32:
            case GGUF_TYPE_FLOAT32:
            case GGUF_TYPE_UINT64:
            case GGUF_TYPE_INT64:
            case GGUF_TYPE_FLOAT64: {
                write(kv.data);
            } break;
            case GGUF_TYPE_BOOL: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<bool>(i));
                }
            } break;
            case GGUF_TYPE_STRING: {
                for (size_t i = 0; i < ne; ++i) {
                    write(kv.get_val<std::string>(i));
                }
            } break;
            case GGUF_TYPE_ARRAY:
            default: GGML_ABORT("invalid type");
        }
    }

    void write_tensor_meta(const struct gguf_tensor_info & info) const {
        write(info.t.name);

        const uint32_t n_dims = ggml_n_dims(&info.t);
        write(n_dims);

        for (uint32_t j = 0; j < n_dims; ++j) {
            write(info.t.ne[j]);
        }
        write(info.t.type);
        write(info.offset);
    }

    void pad(const size_t alignment) const {
        while (buf.size() % alignment != 0) {
            const int8_t zero = 0;
            write(zero);
        }
    }

    void write_tensor_data(const struct gguf_tensor_info & info, const size_t offset_data, const size_t alignment) const {
        GGML_ASSERT(buf.size() - offset_data == info.offset);

        GGML_ASSERT(ggml_is_contiguous(&info.t));
        const size_t offset = buf.size();
        const size_t nbytes = ggml_nbytes(&info.t);

        buf.resize(offset + nbytes);
        if (info.t.buffer) {
            ggml_backend_tensor_get(&info.t, buf.data() + offset, 0, nbytes);
        } else {
            GGML_ASSERT(info.t.data);
            memcpy(buf.data() + offset, info.t.data, nbytes);
        }

        pad(alignment);
    }
};

void gguf_write_to_buf(const struct gguf_context * ctx, std::vector<int8_t> & buf, bool only_meta) {
    const struct gguf_writer gw(buf);

    const int64_t n_kv      = gguf_get_n_kv(ctx);
    const int64_t n_tensors = gguf_get_n_tensors(ctx);

    // write header
    gw.write(GGUF_MAGIC[0]);
    gw.write(GGUF_MAGIC[1]);
    gw.write(GGUF_MAGIC[2]);
    gw.write(GGUF_MAGIC[3]);
    gw.write(ctx->version);
    gw.write(n_tensors);
    gw.write(n_kv);

    // write key-value pairs
    for (int64_t i = 0; i < n_kv; ++i) {
        gw.write(ctx->kv[i]);
    }

    // write tensor info
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_meta(ctx->info[i]);
    }

    // we require the data section to be aligned
    gw.pad(ctx->alignment);

    if (only_meta) {
        return;
    }

    const size_t offset_data = gw.buf.size();

    // write tensor data
    for (int64_t i = 0; i < n_tensors; ++i) {
        gw.write_tensor_data(ctx->info[i], offset_data, ctx->alignment);
    }
}

bool gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta) {
    FILE * file = ggml_fopen(fname, "wb");

    if (!file) {
        fprintf(stderr, "%s: failed to open file '%s' for writing GGUF data\n", __func__, fname);
        return false;
    }

    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, only_meta);
    const bool ok = fwrite(buf.data(), 1, buf.size(), file) == buf.size();
    fclose(file);
    return ok;
}

size_t gguf_get_meta_size(const struct gguf_context * ctx) {
    // only return size
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    return buf.size();
}

void gguf_get_meta_data(const struct gguf_context * ctx, void * data) {
    std::vector<int8_t> buf;
    gguf_write_to_buf(ctx, buf, /*only_meta =*/ true);
    memcpy(data, buf.data(), buf.size());
}

#include "llama-ollama-compat.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace {

int fail(const char * msg) {
    std::fprintf(stderr, "%s\n", msg);
    return 1;
}

bool expect_u32(const gguf_context * meta, const char * key, uint32_t want) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0) {
        std::fprintf(stderr, "missing key: %s\n", key);
        return false;
    }
    if (gguf_get_kv_type(meta, kid) != GGUF_TYPE_UINT32) {
        std::fprintf(stderr, "%s has type %d, want %d\n", key, gguf_get_kv_type(meta, kid), GGUF_TYPE_UINT32);
        return false;
    }
    const uint32_t got = gguf_get_val_u32(meta, kid);
    if (got != want) {
        std::fprintf(stderr, "%s = %u, want %u\n", key, got, want);
        return false;
    }
    return true;
}

bool expect_str(const gguf_context * meta, const char * key, const char * want) {
    const int64_t kid = gguf_find_key(meta, key);
    if (kid < 0 || gguf_get_kv_type(meta, kid) != GGUF_TYPE_STRING) {
        std::fprintf(stderr, "missing string key: %s\n", key);
        return false;
    }
    const char * got = gguf_get_val_str(meta, kid);
    if (!got || std::string(got) != want) {
        std::fprintf(stderr, "%s = %s, want %s\n", key, got ? got : "<null>", want);
        return false;
    }
    return true;
}

bool expect_missing(const gguf_context * meta, const char * key) {
    if (gguf_find_key(meta, key) >= 0) {
        std::fprintf(stderr, "unexpected key: %s\n", key);
        return false;
    }
    return true;
}

void set_minimal_glmocr_kvs(gguf_context * meta) {
    const int32_t mrope_section[] = { 16, 24, 24 };

    gguf_set_val_str(meta, "general.architecture", "glmocr");
    gguf_set_val_u32(meta, "glmocr.attention.key_length", 128);
    gguf_set_val_u32(meta, "glmocr.block_count", 0);
    gguf_set_arr_data(meta, "glmocr.rope.mrope_section", GGUF_TYPE_INT32, mrope_section, 3);
    gguf_set_val_str(meta, "tokenizer.ggml.pre", "llama-bpe");
}

ggml_context * init_ggml_context() {
    ggml_init_params params = { 16 * 1024, nullptr, false };
    return ggml_init(params);
}

} // namespace

int main() {
    bool ok = true;

    {
        gguf_context * meta = gguf_init_empty();
        if (!meta) return fail("gguf_init_empty failed");
        ggml_context * ctx = init_ggml_context();
        if (!ctx) {
            gguf_free(meta);
            return fail("ggml_init failed");
        }

        const char * tokens[] = {
            "<|endoftext|>",
            "x",
            "<|user|>",
        };

        set_minimal_glmocr_kvs(meta);
        gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens, 3);

        // Same numeric value but wrong GGUF type: must be rewritten as UINT32.
        gguf_set_val_i32(meta, "tokenizer.ggml.bos_token_id", 0);
        // Wrong value: must be rewritten to <|endoftext|>.
        gguf_set_val_u32(meta, "tokenizer.ggml.unknown_token_id", 99);
        // Wrong type and value: must be rewritten to <|user|>.
        gguf_set_val_i32(meta, "tokenizer.ggml.eot_token_id", -1);

        std::string arch = "glmocr";
        llama_ollama_compat::translate_metadata(nullptr, meta, ctx, arch, "test.gguf");

        ok = expect_str(meta, "general.architecture", "glm4") && ok;
        ok = arch == "glm4" && ok;
        ok = expect_str(meta, "tokenizer.ggml.pre", "chatglm-bpe") && ok;
        ok = expect_u32(meta, "tokenizer.ggml.bos_token_id", 0) && ok;
        ok = expect_u32(meta, "tokenizer.ggml.unknown_token_id", 0) && ok;
        ok = expect_u32(meta, "tokenizer.ggml.eot_token_id", 2) && ok;
        ok = expect_u32(meta, "glm4.rope.dimension_count", 128) && ok;

        const int64_t mrope_kid = gguf_find_key(meta, "glm4.rope.dimension_sections");
        if (mrope_kid < 0 || gguf_get_arr_n(meta, mrope_kid) != 4) {
            std::fprintf(stderr, "glm4.rope.dimension_sections was not padded to 4 entries\n");
            ok = false;
        }

        ggml_free(ctx);
        gguf_free(meta);
    }

    {
        gguf_context * meta = gguf_init_empty();
        if (!meta) return fail("gguf_init_empty failed");
        ggml_context * ctx = init_ggml_context();
        if (!ctx) {
            gguf_free(meta);
            return fail("ggml_init failed");
        }

        const char * tokens[] = {
            "<|endoftext|>",
            "x",
        };

        set_minimal_glmocr_kvs(meta);
        gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens, 2);

        std::string arch = "glmocr";
        llama_ollama_compat::translate_metadata(nullptr, meta, ctx, arch, "test-missing-user.gguf");

        ok = expect_u32(meta, "tokenizer.ggml.bos_token_id", 0) && ok;
        ok = expect_u32(meta, "tokenizer.ggml.unknown_token_id", 0) && ok;
        ok = expect_missing(meta, "tokenizer.ggml.eot_token_id") && ok;

        ggml_free(ctx);
        gguf_free(meta);
    }

    {
        gguf_context * meta = gguf_init_empty();
        if (!meta) return fail("gguf_init_empty failed");
        ggml_context * ctx = init_ggml_context();
        if (!ctx) {
            gguf_free(meta);
            return fail("ggml_init failed");
        }

        const char * tokens[] = {
            "<|endoftext|>",
            "<|user|>",
        };

        gguf_set_val_str(meta, "general.architecture", "glm4");
        gguf_set_val_str(meta, "tokenizer.ggml.pre", "llama-bpe");
        gguf_set_arr_str(meta, "tokenizer.ggml.tokens", tokens, 2);

        std::string arch = "glm4";
        llama_ollama_compat::translate_metadata(nullptr, meta, ctx, arch, "test-native-glm4.gguf");

        ok = expect_str(meta, "general.architecture", "glm4") && ok;
        ok = arch == "glm4" && ok;
        ok = expect_str(meta, "tokenizer.ggml.pre", "llama-bpe") && ok;
        ok = expect_missing(meta, "tokenizer.ggml.bos_token_id") && ok;
        ok = expect_missing(meta, "tokenizer.ggml.unknown_token_id") && ok;
        ok = expect_missing(meta, "tokenizer.ggml.eot_token_id") && ok;

        ggml_free(ctx);
        gguf_free(meta);
    }

    return ok ? 0 : 1;
}

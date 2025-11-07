#pragma once

#include "llama.h"

#include <string>
#include <vector>
#include <memory>

// pre-tokenization types
enum llama_vocab_pre_type {
    LLAMA_VOCAB_PRE_TYPE_DEFAULT         = 0,
    LLAMA_VOCAB_PRE_TYPE_LLAMA3          = 1,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM    = 2,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER  = 3,
    LLAMA_VOCAB_PRE_TYPE_FALCON          = 4,
    LLAMA_VOCAB_PRE_TYPE_MPT             = 5,
    LLAMA_VOCAB_PRE_TYPE_STARCODER       = 6,
    LLAMA_VOCAB_PRE_TYPE_GPT2            = 7,
    LLAMA_VOCAB_PRE_TYPE_REFACT          = 8,
    LLAMA_VOCAB_PRE_TYPE_COMMAND_R       = 9,
    LLAMA_VOCAB_PRE_TYPE_STABLELM2       = 10,
    LLAMA_VOCAB_PRE_TYPE_QWEN2           = 11,
    LLAMA_VOCAB_PRE_TYPE_OLMO            = 12,
    LLAMA_VOCAB_PRE_TYPE_DBRX            = 13,
    LLAMA_VOCAB_PRE_TYPE_SMAUG           = 14,
    LLAMA_VOCAB_PRE_TYPE_PORO            = 15,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM3        = 16,
    LLAMA_VOCAB_PRE_TYPE_CHATGLM4        = 17,
    LLAMA_VOCAB_PRE_TYPE_VIKING          = 18,
    LLAMA_VOCAB_PRE_TYPE_JAIS            = 19,
    LLAMA_VOCAB_PRE_TYPE_TEKKEN          = 20,
    LLAMA_VOCAB_PRE_TYPE_SMOLLM          = 21,
    LLAMA_VOCAB_PRE_TYPE_CODESHELL       = 22,
    LLAMA_VOCAB_PRE_TYPE_BLOOM           = 23,
    LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH    = 24,
    LLAMA_VOCAB_PRE_TYPE_EXAONE          = 25,
    LLAMA_VOCAB_PRE_TYPE_CHAMELEON       = 26,
    LLAMA_VOCAB_PRE_TYPE_MINERVA         = 27,
    LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM   = 28,
    LLAMA_VOCAB_PRE_TYPE_GPT4O           = 29,
    LLAMA_VOCAB_PRE_TYPE_SUPERBPE        = 30,
    LLAMA_VOCAB_PRE_TYPE_TRILLION        = 31,
    LLAMA_VOCAB_PRE_TYPE_BAILINGMOE      = 32,
    LLAMA_VOCAB_PRE_TYPE_LLAMA4          = 33,
    LLAMA_VOCAB_PRE_TYPE_PIXTRAL         = 34,
    LLAMA_VOCAB_PRE_TYPE_SEED_CODER      = 35,
    LLAMA_VOCAB_PRE_TYPE_HUNYUAN         = 36,
    LLAMA_VOCAB_PRE_TYPE_KIMI_K2         = 37,
    LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE   = 38,
    LLAMA_VOCAB_PRE_TYPE_GROK_2          = 39,
    LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING = 40,
    LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2      = 41,
};

struct LLM_KV;
struct llama_model_loader;

struct llama_vocab {
    struct token_data {
        std::string      text;
        float            score;
        llama_token_attr attr;
    };

    llama_vocab();
    ~llama_vocab();

    void load(llama_model_loader & ml, const LLM_KV & kv);

    std::string get_tokenizer_model() const;
    std::string get_tokenizer_pre() const;

    enum llama_vocab_type     get_type()     const;
    enum llama_vocab_pre_type get_pre_type() const;

    uint32_t n_tokens() const;
    uint32_t n_token_types() const;

    std::string type_name() const;

    bool is_normal      (llama_token id) const;
    bool is_unknown     (llama_token id) const;
    bool is_control     (llama_token id) const;
    bool is_byte        (llama_token id) const;
    bool is_user_defined(llama_token id) const;
    bool is_unused      (llama_token id) const;
    bool is_eog         (llama_token id) const;

    uint8_t     token_to_byte(llama_token id) const;
    llama_token byte_to_token(uint8_t ch)     const;

    llama_token text_to_token(const std::string & text) const;

    const token_data & get_token_data(llama_token id) const;

    const char *     token_get_text (llama_token id) const;
    float            token_get_score(llama_token id) const;
    llama_token_attr token_get_attr (llama_token id) const;

    llama_token token_bos() const;
    llama_token token_eos() const;
    llama_token token_eot() const;
    llama_token token_eom() const;
    llama_token token_unk() const;
    llama_token token_sep() const;
    llama_token token_nl () const;
    llama_token token_pad() const;
    llama_token token_mask() const;

    llama_token token_prefix() const;
    llama_token token_middle() const;
    llama_token token_suffix() const;

    llama_token token_fim_pre() const;
    llama_token token_fim_suf() const;
    llama_token token_fim_mid() const;
    llama_token token_fim_pad() const;
    llama_token token_fim_rep() const;
    llama_token token_fim_sep() const;

    bool get_add_space_prefix          () const;
    bool get_add_bos                   () const;
    bool get_add_eos                   () const;
    bool get_add_sep                   () const;
    bool get_ignore_merges             () const;
    bool get_clean_spaces              () const;
    bool get_remove_extra_whitespaces  () const;
    bool get_escape_whitespaces        () const;
    bool get_treat_whitespace_as_suffix() const;

    int max_token_len() const;

    int find_bpe_rank(const std::string & token_left, const std::string & token_right) const;
    std::vector<std::string> get_bpe_merges() const;

    std::vector<char> get_precompiled_charsmap() const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    // does not write null-terminator to buf
    int32_t token_to_piece(
                  llama_token   token,
                         char * buf,
                      int32_t   length,
                      int32_t   lstrip,
                         bool   special) const;

    // use cached data
    const std::string & token_to_piece(llama_token token) const;

    int32_t detokenize(
            const llama_token * tokens,
                      int32_t   n_tokens,
                         char * text,
                      int32_t   text_len_max,
                         bool   remove_special,
                         bool   unparse_special) const;

    std::string detokenize(
            const std::vector<llama_token> & tokens,
                                      bool   special) const;

    void print_info() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

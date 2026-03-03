#include "llama-vocab.h"

#include "ggml.h"
#include "gguf.h"
#include "llama-impl.h"
#include "llama-model-loader.h"

#include "unicode.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstdarg>
#include <cstring>
#include <forward_list>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <unordered_map>

//
// helpers
//

struct naive_trie {
    naive_trie() : has_value(false), value(0) {
    }
    void insert(const char * key, size_t len, int32_t value = 0) {
        if (len == 0) {
            this->has_value = true;
            this->value = value;
            return;
        }
        char c = key[0];
        auto res = children.find(c);
        if (res != children.end()) {
            res->second.insert(key + 1, len - 1, value);
        } else {
            auto res = children.insert(std::make_pair(c, naive_trie()));
            res.first->second.insert(key + 1, len - 1, value);
        }
    }
    std::pair<const char *, size_t> get_longest_prefix(const char * key, size_t len, size_t offset = 0) const {
        if (len == 0 || offset == len) {
            return std::make_pair(key, offset);
        }
        char c = key[offset];
        auto res = children.find(c);
        if (res != children.end()) {
            return res->second.get_longest_prefix(key, len, offset + 1);
        }

        return std::make_pair(key, offset);
    }
    const struct naive_trie * traverse(const char c) const {
        auto res = children.find(c);
        if (res != children.end()) {
            return &res->second;
        }

        return NULL;
    }
    std::map<char, struct naive_trie> children;
    bool has_value;
    llama_token value;
};

//
// tokenizers
//

struct llm_tokenizer {
    llm_tokenizer() {}
    virtual ~llm_tokenizer() = default;
};

struct llm_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

static_assert(std::is_trivially_copyable<llm_symbol>::value, "llm_symbol is not trivially copyable");

//
// SPM tokenizer
// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
//

struct llm_bigram_spm {
    struct comparator {
        bool operator()(llm_bigram_spm & l, llm_bigram_spm & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llm_bigram_spm>;
    using queue = std::priority_queue<llm_bigram_spm, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    float score;
    size_t size;
};

struct llm_tokenizer_spm : llm_tokenizer {
    llm_tokenizer_spm(const llama_vocab & /*vocab*/) {}
};

struct llm_tokenizer_spm_session {
    llm_tokenizer_spm_session(const llama_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llm_symbol sym;
            size_t len = unicode_len_utf8(text[offs]);
            sym.text = text.c_str() + offs;
            sym.n = std::min(len, text.size() - offs);
            offs += sym.n;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols.emplace_back(sym);
        }

        // seed the work queue with all possible 2-character tokens.
        for (int i = 1; i < (int) symbols.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue.empty()) {
            auto bigram = work_queue.top();
            work_queue.pop();

            auto & left_sym = symbols[bigram.left];
            auto & right_sym = symbols[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //LLAMA_LOG_INFO("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols[i].next) {
            auto & symbol = symbols[i];
            resegment(symbol, output);
        }
    }

private:
    void resegment(llm_symbol & symbol, std::vector<llama_token> & output) {
        auto text = std::string(symbol.text, symbol.n);
        auto token = vocab.text_to_token(text);

        // Do we need to support is_unused?
        if (token != LLAMA_TOKEN_NULL) {
            output.push_back(token);
            return;
        }

        const auto p = rev_merge.find(text);

        if (p == rev_merge.end()) {
            // output any symbols that did not form tokens as bytes.
            output.reserve(output.size() + symbol.n);
            for (int j = 0; j < (int)symbol.n; ++j) {
                llama_token id = vocab.byte_to_token(symbol.text[j]);
                output.push_back(id);
            }
            return;
        }

        resegment(symbols[p->second.first], output);
        resegment(symbols[p->second.second], output);
    }

    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        const std::string text = std::string(symbols[left].text, symbols[left].n + symbols[right].n);
        auto token = vocab.text_to_token(text);

        if (token == LLAMA_TOKEN_NULL) {
            return;
        }

        if (static_cast<uint32_t>(token) >= vocab.n_tokens()) {
            return;
        }

        const auto & tok_data = vocab.get_token_data(token);

        llm_bigram_spm bigram;
        bigram.left  = left;
        bigram.right = right;
        bigram.score = tok_data.score;
        bigram.size  = text.size();

        work_queue.push(bigram);

        // Do we need to support is_unused?
        rev_merge[text] = std::make_pair(left, right);
    }

    const llama_vocab & vocab;
    // currently unused
    // const llm_tokenizer_spm * spm_tokenizer;

    std::vector<llm_symbol> symbols;
    llm_bigram_spm::queue work_queue;
    std::map<std::string, std::pair<int, int>> rev_merge;
};

//
// BPE tokenizer
// adapted from https://github.com/cmp-nct/ggllm.cpp [MIT License]
// tried to simplify unicode stuff, so most likely does not work 100% correctly!
//

// TODO: there are a lot of common parts between spm and bpe tokenizers, should be refactored and reused

template<typename T, typename Container = std::vector<T>, typename Compare = std::less<typename Container::value_type>>
class llama_priority_queue : public std::priority_queue<T, Container, Compare> {
public:
    using std::priority_queue<T, Container, Compare>::priority_queue;

    T pop_move() {
        T item = std::move(this->c.front());
        std::pop_heap(this->c.begin(), this->c.end(), this->comp);
        this->c.pop_back();
        return item;
    }

    void pop() =  delete;
};

struct llm_bigram_bpe {
    struct comparator {
        bool operator()(const llm_bigram_bpe & l, const llm_bigram_bpe & r) const {
            return l.rank > r.rank || (l.rank == r.rank && l.left > r.left);
        }
    };

    using queue_storage = std::vector<llm_bigram_bpe>;
    using queue = llama_priority_queue<llm_bigram_bpe, queue_storage, comparator>;
    llm_symbol::index left;
    llm_symbol::index right;
    std::string text;
    int rank;
    size_t size;
};

struct llm_tokenizer_bpe : llm_tokenizer {
    llm_tokenizer_bpe(const llama_vocab & vocab) {
        GGML_ASSERT(vocab.get_type() == LLAMA_VOCAB_TYPE_BPE);
        switch (vocab.get_pre_type()) {
            case LLAMA_VOCAB_PRE_TYPE_LLAMA3:
                regex_exprs = {
                    // original regex from tokenizer.json
                    //"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",

                    // adapted: https://github.com/ggerganov/llama.cpp/pull/6920#issuecomment-2080233989
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DBRX:
            case LLAMA_VOCAB_PRE_TYPE_SMAUG:
                regex_exprs = {
                    // same as llama3
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ\U00010400-\U0001044f𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+",
                    "\\s?[!-/:-~！-／：-～‘-‟　-。]+",
                    "\\s+$",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM:
            case LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE:
                regex_exprs = {
                    "\\p{N}{1,3}",
                    "[一-龥぀-ゟ゠-ヿ]+",
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER:
                regex_exprs = {
                    "[\r\n]",
                    "\\s?\\p{L}+",
                    "\\s?\\p{P}+",
                    "[一-龥ࠀ-一가-퟿]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_FALCON:
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|`]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "[0-9][0-9][0-9]",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STARCODER:
            case LLAMA_VOCAB_PRE_TYPE_REFACT:
            case LLAMA_VOCAB_PRE_TYPE_COMMAND_R:
            case LLAMA_VOCAB_PRE_TYPE_SMOLLM:
            case LLAMA_VOCAB_PRE_TYPE_CODESHELL:
            case LLAMA_VOCAB_PRE_TYPE_EXAONE:
            case LLAMA_VOCAB_PRE_TYPE_MINERVA:
                regex_exprs = {
                    "\\p{N}",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_GPT2:
            case LLAMA_VOCAB_PRE_TYPE_MPT:
            case LLAMA_VOCAB_PRE_TYPE_OLMO:
            case LLAMA_VOCAB_PRE_TYPE_JAIS:
            case LLAMA_VOCAB_PRE_TYPE_TRILLION:
            case LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING:
                regex_exprs = {
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_STABLELM2:
            case LLAMA_VOCAB_PRE_TYPE_QWEN2:
            case LLAMA_VOCAB_PRE_TYPE_HUNYUAN:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_PORO:
            case LLAMA_VOCAB_PRE_TYPE_BLOOM:
            case LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_CHATGLM4:
                regex_exprs = {
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_VIKING:
                regex_exprs = {
                    " ?[^(\\s|.,!?…。，、।۔،)]+",
                    "\\p{N}",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_TEKKEN:
                // original regex from tokenizer.json
                // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                regex_exprs = {
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_CHAMELEON:
                // Note: in theory, the special token (sentinel and image token) regex_exprs below
                // are unnecessary, as they are split in `tokenizer_st_partition` anyway.
                // However, since the upstream pre-tokenizer uses them, they are also
                // included here (see https://huggingface.co/facebook/chameleon-7b).
                regex_exprs = {
                    "<sentinel:[0-9]+>",  // Sentinel tokens
                    "(IMGIMG)((A|B|C|D|E|F|G|H|I){1,4})Z",  // Image tokens
                    "([\\t\\n]|    |  )",  // directly from tokenizer.json
                    "\\p{N}", // Individual digits
                    "[\\p{P}!-/:-@\\[-`{-~]",  // Punctuation, Isolated
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_GPT4O:
            case LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                    "[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))*((?=[\\p{L}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\\r\\n\\p{L}\\p{N}]?((?=[\\p{L}])([^a-z]))+((?=[\\p{L}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_KIMI_K2:
                regex_exprs = {
                    // K2 trigger pattern - this will activate the custom K2 handler in unicode.cpp
                    // The custom handler implements all K2 patterns with proper Han character exclusion
                    "\\p{Han}+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_SUPERBPE:
                regex_exprs = {
                    "\\p{N}+",
                    "(?=(\\d{3})+(?!\\d))",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_BAILINGMOE:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"
                    // FIXME? Changed possessive quantifiers (?+ and ++) to greedy to avoid errors and imatrix hanging (tried atomic grouping but it's not supported?)
                    "'(?:[sSdDmMtT]|[lL][lL]|[vV][eE]|[rR][eE])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_SEED_CODER:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\r\n]+|\\s*[\r\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1}| ?[^\\s\\p{L}\\p{N}\\r\\n]+|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_GROK_2:
                regex_exprs = {
                    // original regex from tokenizer.json
                    // "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
                    "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            case LLAMA_VOCAB_PRE_TYPE_AFMOE:
                regex_exprs = {
                    // Digit handling - uses custom implementation in unicode.cpp
                    // Groups digits with leading 1-2 based on total length modulo 3
                    "\\p{AFMoE_digits}",
                    // CJK and Asian scripts (using direct Unicode literals)
                    "[一-鿿㐀-䶿豈-﫿぀-ゟ゠-ヿ･-ﾟ⼀-⿟เ-๿຀-໿ក-៿က-႟ꩠ-ꩿꧠ-꧿가-힯ᄀ-ᇿ]+",
                    // Main BPE pattern
                    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\\r\\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
                };
                break;
            default:
                // default regex for BPE tokenization pre-processing
                regex_exprs = {
                    "[\\p{P}\\$\\+<=>\\^~\\|]+",
                    "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
                    "\\p{N}+",
                    "[0-9][0-9][0-9]",
                };
                break;
        }
    }

    std::vector<std::string> regex_exprs;
};

struct llm_tokenizer_bpe_session {
    llm_tokenizer_bpe_session(const llama_vocab & vocab, const llm_tokenizer_bpe & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    static void append(const llama_token token_id, std::vector<llama_token> & output)  {
        output.push_back(token_id);
    }

    bool append_bos(std::vector<llama_token> & output) const {
        if (vocab.get_add_bos()) {
            GGML_ASSERT(vocab.token_bos() != LLAMA_TOKEN_NULL);
            output.push_back(vocab.token_bos());
            return true;
        }
        return false;
    }

    bool append_eos(std::vector<llama_token> & output) const {
        if (vocab.get_add_eos()) {
            GGML_ASSERT(vocab.token_eos() != LLAMA_TOKEN_NULL);
            output.push_back(vocab.token_eos());
            return true;
        }
        return false;
    }

    void check_double_bos_eos(const std::vector<llama_token> & output) const {
        if (vocab.get_add_bos() && output.size() >= 2 && output[1] == vocab.token_bos()) {
            LLAMA_LOG_WARN(
                "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
        if (vocab.get_add_eos() && output.size() >= 2 && *(output.end()-2) == vocab.token_eos()) {
            LLAMA_LOG_WARN(
                "%s: Added a EOS token to the prompt as specified by the model but the prompt "
                "also ends with a EOS token. So now the final prompt ends with 2 EOS tokens. "
                "Are you sure this is what you want?\n", __FUNCTION__);
        }
    }

    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        int final_prev_index = -1;
        const auto word_collection = unicode_regex_split(text, tokenizer.regex_exprs);

        symbols_final.clear();

        for (const auto & word : word_collection) {
            work_queue = llm_bigram_bpe::queue();
            symbols.clear();

            int index = 0;
            size_t offset = 0;

            //if (vocab.tokenizer_ignore_merges && vocab.token_to_id.find(word) != vocab.token_to_id.end()) {
            if (vocab.get_ignore_merges() && vocab.text_to_token(word) != LLAMA_TOKEN_NULL) {
                symbols.emplace_back(llm_symbol{-1, -1, word.c_str(), word.size()});
                offset = word.size();
            }

            while (offset < word.size()) {
                llm_symbol sym;
                size_t char_len = std::min(word.size() - offset, (size_t) unicode_len_utf8(word[offset]));
                sym.text = word.c_str() + offset;
                sym.n = char_len;
                offset += sym.n;
                sym.prev = index - 1;
                sym.next = offset == word.size() ? -1 : index + 1;
                index++;
                symbols.emplace_back(sym);
            }
            for (int i = 1; i < (int) symbols.size(); ++i) {
                add_new_bigram(i - 1, i);
            }

            // build token(s)
            while (!work_queue.empty()) {
                auto bigram = work_queue.pop_move();

                auto & left_symbol = symbols[bigram.left];
                auto & right_symbol = symbols[bigram.right];

                if (left_symbol.n == 0 || right_symbol.n == 0) {
                    continue;
                }
                std::string left_token = std::string(left_symbol.text, left_symbol.n);
                std::string right_token = std::string(right_symbol.text, right_symbol.n);
                if (left_token + right_token != bigram.text) {
                    continue;  // Skip this bigram if it's outdated
                }

                // merge the right sym into the left one
                left_symbol.n += right_symbol.n;
                right_symbol.n = 0;

                // remove the right sym from the chain
                left_symbol.next = right_symbol.next;
                if (right_symbol.next >= 0) {
                    symbols[right_symbol.next].prev = bigram.left;
                }

                add_new_bigram(left_symbol.prev, bigram.left);  // left side of current symbol
                add_new_bigram(bigram.left, left_symbol.next);  // right side of current symbol
            }

            // add the finished tokens to the final list keeping correct order for next and prev
            for (auto & sym : symbols) {
                if (sym.n > 0) {
                    sym.prev = final_prev_index;
                    sym.next = -1;
                    if (final_prev_index != -1) {
                        symbols_final[final_prev_index].next = symbols_final.size();
                    }
                    symbols_final.emplace_back(sym);
                    final_prev_index = symbols_final.size() - 1;
                }
            }
        }

        symbols = symbols_final;

        if (!symbols.empty()) {
            for (int i = 0; i != -1; i = symbols[i].next) {
                auto & symbol = symbols[i];
                if (symbol.n == 0) {
                    continue;
                }

                const std::string str = std::string(symbol.text, symbol.n);
                const auto token = vocab.text_to_token(str);

                if (token == LLAMA_TOKEN_NULL) {
                    for (auto j = str.begin(); j != str.end(); ++j) {
                        std::string byte_str(1, *j);
                        auto token_multibyte = vocab.text_to_token(byte_str);
                        if (token_multibyte != LLAMA_TOKEN_NULL) {
                            output.push_back(token_multibyte);
                        }
                    }
                } else {
                    output.push_back(token);
                }
            }
        }
    }

private:
    void add_new_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }
        std::string left_token  = std::string(symbols[left].text,  symbols[left].n);
        std::string right_token = std::string(symbols[right].text, symbols[right].n);

        int rank_found = -1;

        rank_found = vocab.find_bpe_rank(left_token, right_token);

        if (rank_found < 0) {
            return;
        }

        llm_bigram_bpe bigram;

        bigram.left  = left;
        bigram.right = right;
        bigram.text  = left_token + right_token;
        bigram.size  = left_token.size() + right_token.size();
        bigram.rank  = rank_found;

        work_queue.push(bigram);
    }

    const llama_vocab & vocab;
    const llm_tokenizer_bpe & tokenizer;

    std::vector<llm_symbol> symbols;
    std::vector<llm_symbol> symbols_final;
    llm_bigram_bpe::queue work_queue;
};

//
// WPM tokenizer
//

struct llm_tokenizer_wpm : llm_tokenizer {
    llm_tokenizer_wpm(const llama_vocab & /*vocab*/) {}
};

struct llm_tokenizer_wpm_session {
    llm_tokenizer_wpm_session(const llama_vocab & vocab) : vocab(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        // normalize and split by whitespace
        std::vector<std::string> words = preprocess(text);
        // bos token prepended already

        // find the longest tokens that form the words
        for (const std::string & word : words) {
            // skip empty words
            if (word.size() == 0) {
                continue;
            }

            // prepend phantom space
            const std::string word1 = "\xe2\x96\x81" + word;
            const int n = word1.size();

            const size_t current_tokens = output.size();

            // we're at the start of a new word
            // move through character position in word
            for (int i = 0; i < n; ++i) {
                // loop through possible match length
                bool match = false;
                for (int j = std::min(n, i + vocab.max_token_len() + 1); j > i; j--) {
                    auto id = vocab.text_to_token(word1.substr(i, j - i));
                    if (id != LLAMA_TOKEN_NULL) {
                        output.push_back(id);
                        match = true;
                        i = j - 1;
                        break;
                    }
                }

                if (!match) { // discard all
                    output.resize(current_tokens);
                    break;  // and discard next tokens
                }
            }

            // we didn't find any matches for this word
            if (current_tokens == output.size()) {
                output.push_back(vocab.token_unk());
            }
        }
    }

    // TODO: reduce string copies by using cpts_offs array
    static std::vector<std::string> preprocess(const std::string & text)  {
        const std::vector<uint32_t> cpts_nfd = unicode_cpts_normalize_nfd(unicode_cpts_from_utf8(text));
        std::vector<std::string> words(1, "");

        for (const uint32_t cpt : cpts_nfd) {
            const auto flags = unicode_cpt_flags_from_cpt(cpt);

            if (flags.is_whitespace) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                continue;
            }

            assert (!flags.is_separator);
            if (cpt == 0 || cpt == 0xFFFD || flags.is_control) {
                continue;
            }

            const std::string s = unicode_cpt_to_utf8(unicode_tolower(cpt));
            if (flags.is_punctuation || ( cpt < 0x7F && flags.is_symbol ) || is_chinese_char(cpt)) {
                if (words.back().size()) {  // finish previous word if any
                    words.emplace_back();
                }
                words.back() = s;       // single char word
                words.emplace_back();   // start a new word
            } else {
                words.back() += s;  // append char to word
            }
        }

        if (!words.back().size()) {
            words.pop_back();
        }

        return words;
    }

    static bool is_chinese_char(uint32_t cpt) {
        return
            (cpt >= 0x04E00 && cpt <= 0x09FFF) ||
            (cpt >= 0x03400 && cpt <= 0x04DBF) ||
            (cpt >= 0x20000 && cpt <= 0x2A6DF) ||
            (cpt >= 0x2A700 && cpt <= 0x2B73F) ||
            (cpt >= 0x2B740 && cpt <= 0x2B81F) ||
            (cpt >= 0x2B920 && cpt <= 0x2CEAF) || // this should be 0x2B820 but in hf rust code it is 0x2B920
            (cpt >= 0x0F900 && cpt <= 0x0FAFF) ||
            (cpt >= 0x2F800 && cpt <= 0x2FA1F);
            //(cpt >= 0x3000  && cpt <= 0x303F)  ||
            //(cpt >= 0xFF00  && cpt <= 0xFFEF);
    }

private:
    const llama_vocab & vocab;
    // currently unused
    // const llm_tokenizer_wpm * wpm_tokenizer;
};

//
// UGM tokenizer
//

struct llm_tokenizer_ugm : llm_tokenizer {
    llm_tokenizer_ugm(const llama_vocab & vocab, const std::vector<char> & precompiled_charsmap) {
        if (precompiled_charsmap.size() > 0) {
            size_t charsmap_offset = 0;

            // First four bytes of precompiled_charsmap contains length of binary
            // blob containing XOR-compressed compact double array (XCDA) entries
            uint32_t xcda_blob_size = *(const uint32_t *) &precompiled_charsmap[0];
            charsmap_offset += sizeof(xcda_blob_size);
            if (xcda_blob_size + charsmap_offset >= precompiled_charsmap.size()) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }

            // Next xcda_blob_size bytes contain entries of XOR-compressed compact
            // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
            xcda_array = (const uint32_t *) &precompiled_charsmap[charsmap_offset];
            xcda_array_size = xcda_blob_size / sizeof(uint32_t);
            charsmap_offset += xcda_blob_size;

            // Remaining bytes of precompiled charsmap contain null-terminated
            // replacement strings for prefixes matched by the XCDA.
            prefix_replacements = &precompiled_charsmap[charsmap_offset];
            prefix_replacements_size = precompiled_charsmap.size() - charsmap_offset;
        }

        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & token_data = vocab.get_token_data(id);

            if (vocab.is_normal(id)) {
                min_score = std::min<float>(min_score, token_data.score);
                max_score = std::max<float>(max_score, token_data.score);
            }

            if (vocab.is_normal(id) ||
                vocab.is_user_defined(id) ||
                vocab.is_unused(id)) {
                token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
            }

            if (vocab.is_user_defined(id)) {
                user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
            }
        }

        unknown_token_score = min_score - unknown_token_score_penalty;
    }

    // escaped space symbol - U+2581 (Lower One Eighth Block)
    const std::string escaped_space = "\xE2\x96\x81";

    const char * prefix_replacements = NULL;
    size_t prefix_replacements_size = 0;

    const uint32_t * xcda_array = NULL;
    size_t xcda_array_size = 0;

    struct naive_trie user_defined_token_matcher;

    float min_score = FLT_MAX;
    float max_score = -FLT_MAX;

    float unknown_token_score_penalty = 10.0;
    float unknown_token_score;

    struct naive_trie token_matcher;
};

struct llm_tokenizer_ugm_session {
    llm_tokenizer_ugm_session(const llama_vocab & vocab, const llm_tokenizer_ugm & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    /* This implementation is based on SentencePiece optimized Viterbi algorithm for
     * unigram language models. The general idea is to:
     * - move along the input sequence in steps of one UTF code point,
     * - at each step find all possible tokenizations of the prefix by
     *   traversing the tokens trie,
     * - for each tokenization store the best one so far (by higher score)
     * - use the position in sequence after given token as an index to store
     *   results
     * - if there was no valid tokenization of the current UTF code point
     *   then use unknown token with additional score penalty
     * After processing the whole sequence we backtrack from the end to get
     * the best tokenization.
    */
    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        // get current size of output (for reversal later)
        size_t output_size = output.size();

        // normalize the input first
        std::string normalized;
        normalize(text, &normalized);
        size_t input_len = normalized.size();
        if (input_len == 0) {
            return;
        }

        // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
        std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.token_unk(), 0, -DBL_MAX});
        // at the beginning tokenization score is zero
        tokenization_results[0] = { vocab.token_unk(), 0, 0 };

        for (size_t input_offset = 0; input_offset < input_len;) {
            size_t prefix_offset = input_offset;
            // calculate how many code units are in the currently processed UTF code point
            size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

            // traverse the token matcher trie to find a matching token
            bool single_codepoint_token_found = false;
            const struct best_tokenization & current_best = tokenization_results[input_offset];
            const struct naive_trie * node = tokenizer.token_matcher.traverse(normalized[prefix_offset++]);

            while (prefix_offset <= input_len && node != NULL) {
                // check if we found valid token in prefix
                if (node->has_value) {
                    // check if it corresponds to the whole UTF code point
                    if (prefix_offset - input_offset == n_utf8_code_units) {
                        single_codepoint_token_found = true;
                    }
                    llama_token token_id = node->value;
                    const auto & token_data = vocab.get_token_data(token_id);

                    // we set the user-defined token scores to 0 to make them more likely to be selected
                    // (normal token scores are log probabilities, so they are negative)
                    // score type is double here to make tokenization results exactly
                    // the same as in the HF tokenizer using SentencePiece
                    const double token_score = vocab.is_user_defined(token_id) ? 0.0 : token_data.score;
                    const double challenger_score = current_best.score_sum + token_score;
                    struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                    if (challenger_score > current_champ.score_sum) {
                        struct best_tokenization challenger = { token_id, input_offset, challenger_score };
                        current_champ = challenger;
                    }
                }
                node = node->traverse(normalized[prefix_offset++]);
            }

            // if we didn't find a valid token corresponding to the whole UTF code point
            // then use unknown token as the tokenization of this UTF code point
            if (!single_codepoint_token_found) {
                const double challenger_score = current_best.score_sum + tokenizer.unknown_token_score;
                prefix_offset = input_offset + n_utf8_code_units;
                struct best_tokenization & current_champ = tokenization_results[prefix_offset];
                if (challenger_score > current_champ.score_sum) {
                    struct best_tokenization challenger = { vocab.token_unk(), input_offset, challenger_score };
                    current_champ = challenger;
                }
            }

            // move to the next UTF code point
            input_offset += n_utf8_code_units;
        }

        // now backtrack from the end to gather token ids of the best tokenization
        // merge sequences of consecutive unknown tokens into single unknown tokens
        bool is_prev_unknown = false;
        for (struct best_tokenization & tokenization = tokenization_results[input_len]; ; tokenization = tokenization_results[tokenization.input_offset]) {
            bool is_unknown = tokenization.token_id == vocab.token_unk();
            if (!(is_prev_unknown && is_unknown)) {
                output.push_back(tokenization.token_id);
            }
            if (tokenization.input_offset == 0) {
                break;
            }
            is_prev_unknown = is_unknown;
        }

        // reverse the output since we added tokens starting from the end of the input
        std::reverse(output.begin() + output_size, output.end());
    }

private:

    // helper structure for returning normalization results
    struct normalization_result {
        const char * normalized;
        size_t normalized_len;
        size_t consumed_input;
    };

    void normalize(const std::string& input, std::string * normalized) {
        normalized->clear();
        normalized->reserve(input.size() * 3);

        const std::string space = vocab.get_escape_whitespaces() ? tokenizer.escaped_space : " ";

        const bool shall_prepend_space = !vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_append_space  =  vocab.get_treat_whitespace_as_suffix() && vocab.get_add_space_prefix();
        const bool shall_merge_spaces  =  vocab.get_remove_extra_whitespaces();

        bool is_space_prepended = false;
        bool processing_non_ws = false;

        size_t input_len = input.size();

        for (size_t input_offset = 0; input_offset < input_len; ) {
            auto norm_res = normalize_prefix(input, input_offset);
            for (size_t i = 0; i < norm_res.normalized_len; i++) {
                char c = norm_res.normalized[i];
                if (c != ' ') {
                    if (!processing_non_ws) {
                        processing_non_ws = true;
                        if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
                            normalized->append(space);
                            is_space_prepended = true;
                        }
                    }
                    normalized->push_back(c);
                } else {
                    if (processing_non_ws) {
                        processing_non_ws = false;
                    }
                    if (!shall_merge_spaces) {
                        normalized->append(space);
                    }
                }
            }

            input_offset += norm_res.consumed_input;
        }

        if (shall_append_space) {
            normalized->append(space);
        }
    }

    /*
     * This structure is a view wrapper for XOR-compressed double array (XCDA)
     * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
     * Each bit-packed entry contains:
     * - BASE array value in bits 10-30
     * - LCHECK array value in bits 0-7
     * - LEAF array value in bit 9
     * Entries containing indexes of replacement sequences have set bit 31
     */
    struct xcda_array_view {
    public:
        xcda_array_view(const uint32_t * xcda_array, size_t xcda_array_size) : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {
        }
        uint32_t get_base(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
        }
        uint32_t get_lcheck(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) | 0xff);
        }
        bool get_leaf(size_t index) {
            uint32_t packed_node = get_node(index);
            return (packed_node >> 8) & 1;
        }
        uint32_t get_value(size_t index) {
            uint32_t packed_node = get_node(index);
            return packed_node & ((1U << 31) - 1);
        }
    private:
        uint32_t get_node(size_t index) {
            if (index >= xcda_array_size) {
                throw std::runtime_error("Index out of array bounds in XCDA array!");
            }
            return xcda_array[index];
        }
        const uint32_t * xcda_array;
        size_t xcda_array_size;
    };

    // this structure stores the best tokenization so far at input_offset
    struct best_tokenization {
        llama_token token_id;
        size_t input_offset;
        double score_sum;
    };

    struct normalization_result normalize_prefix(const std::string & input, size_t input_offset) {
        if (input_offset == input.size()) {
            return { &input[input_offset], 0, 0 };
        }

        // if input prefix matches some user-defined token return this token as normalization result
        auto user_defined_token_match =
           tokenizer.user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
        if (user_defined_token_match.second > 0) {
            return { &input[input_offset], user_defined_token_match.second, user_defined_token_match.second };
        }

        size_t longest_prefix_length = 0;
        size_t longest_prefix_offset = 0;

        if (tokenizer.xcda_array_size > 0) {
            struct xcda_array_view xcda_view(tokenizer.xcda_array, tokenizer.xcda_array_size);

            // Find the longest normalized sequence matching the input prefix by walking
            // the XOR-compressed compact double array (XCDA) starting from the root node
            // We find the index of the next node by calculating BASE[s] ^ c where s is
            // the index of the previous node and c is a numerical character value
            uint32_t node_index = 0;
            // get BASE of the root node
            node_index = xcda_view.get_base(node_index);
            for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
                unsigned char c = input[prefix_offset];
                if (c == 0) {
                    break;
                }
                node_index ^= c;
                // if value of LCHECK is not c it means that this is not a child of
                // the previous node, so we stop matching
                if (xcda_view.get_lcheck(node_index) != c) {
                    break;
                }
                bool is_leaf = xcda_view.get_leaf(node_index);
                // get BASE of the current node
                node_index ^= xcda_view.get_base(node_index);
                // if LEAF of the current node is true, it means that its BASE points to the node
                // containing index of replacement sequence for currently matched input prefix
                if (is_leaf)
                {
                    longest_prefix_length = prefix_offset - input_offset + 1;
                    // get index of replacement sequence for currently matched input prefix
                    longest_prefix_offset = xcda_view.get_value(node_index);
                }
            }
        }

        if (longest_prefix_length > 0) {
            // we have a match, so return the replacement sequence
            if (longest_prefix_offset >= tokenizer.prefix_replacements_size) {
                throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
            }
            const char * prefix_replacement = &(tokenizer.prefix_replacements)[longest_prefix_offset];
            return { prefix_replacement, strlen(prefix_replacement), longest_prefix_length };
        }

        // check if the input prefix contains a valid sequence of UTF-8 code units
        try {
            // if yes, return this sequence unmodified
            size_t prefix_offset = input_offset;
            unicode_cpt_from_utf8(input, prefix_offset);
            return { &input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset };
        } catch (std::invalid_argument & /*ex*/) {
            // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
            return { "\xEF\xBF\xBD", 3, 1 };
        }
    }

    const llama_vocab & vocab;
    const llm_tokenizer_ugm & tokenizer;
};

//
// RWKV tokenizer
//

static std::vector<uint8_t> llama_unescape_rwkv_token(const std::string & escaped) {
    std::vector<uint8_t> output;
    output.reserve(escaped.size());

    // Parser state
    bool escaping = false;
    uint8_t hex_remaining = 0;
    uint8_t hex_acc = 0;

    // Step through characters, performing parsing
    for (const char & c : escaped) {
        // If we're parsing a hex code, interpret the next character
        if (hex_remaining != 0) {
            uint8_t value = (c >= 'a') ? (c - 'a' + 10) : (c - '0');
            hex_acc = (hex_acc << 4) + value;

            hex_remaining -= 1;
            if (hex_remaining == 0) {
                output.push_back(hex_acc);
                hex_acc = 0;
            }

            continue;
        }

        // If we got an escape character, interpret it
        if (escaping) {
            if (c == 't') {
                output.push_back('\t');
            } else if (c == 'n') {
                output.push_back('\n');
            } else if (c == 'r') {
                output.push_back('\r');
            } else if (c == 'x') {
                hex_remaining = 2;
            } else {
                output.push_back(c);
            }

            escaping = false;
            continue;
        }

        if (c == '\\') {
            escaping = true;
            continue;
        }

        output.push_back(c);
    }

    return output;
}

struct llm_tokenizer_rwkv : llm_tokenizer {
    llm_tokenizer_rwkv(const llama_vocab & vocab) {
        // RWKV supports arbitrary byte tokens, but the vocab struct only supports string tokens.
        // For now, we decode the vocab here into the lookup we'll use for tokenization.

        // build trie
        for (uint32_t id = 0; id < vocab.n_tokens(); ++id) {
            const auto & data = vocab.get_token_data(id);
            const auto text = llama_unescape_rwkv_token(data.text);
            token_matcher.insert((const char *) text.data(), text.size(), id);
        }
    }

    struct naive_trie token_matcher;
};

struct llm_tokenizer_rwkv_session {
    llm_tokenizer_rwkv_session(const llama_vocab & vocab, const llm_tokenizer_rwkv & tokenizer) : vocab(vocab), tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        uint32_t position = 0;
        while (position < text.size()) {
            const struct naive_trie * node = tokenizer.token_matcher.traverse(text[position]);
            if (node == NULL) {
                // no matching token found, add unknown token
                output.push_back(vocab.token_unk());
                position += 1;
                continue;
            }

            // traverse the trie to find the longest matching token
            uint32_t token_id = 0;
            uint32_t token_length = 0;
            while (node != NULL) {
                if (node->has_value) {
                    token_id = node->value;
                    token_length = position + 1;
                }
                node = node->traverse(text[++position]);
            }

            // add the longest matching token
            output.push_back(token_id);
            position = token_length;
        }
    }

private:
    const llama_vocab & vocab;
    const llm_tokenizer_rwkv & tokenizer;
};

struct llm_tokenizer_plamo2 : llm_tokenizer {
    llm_tokenizer_plamo2(const llama_vocab & vocab) {
        build(vocab);
    }

    void build(const llama_vocab & vocab) {
        // Reset internal structures
        tokens_.clear();
        bytes_.assign(256, 0);
        to_suffix_id_.clear();
        table_.clear();

        // Build token list and byte mapping
        std::unordered_map<std::string, float> suffix_to_score;
        std::unordered_map<std::string, llama_token> token_to_id;

        for (size_t token_id = 0; token_id < vocab.n_tokens(); ++token_id) {
            const auto & entry = vocab.get_token_data(token_id);
            tokens_.push_back(entry.text);
            token_to_id[entry.text] = static_cast<llama_token>(token_id);

            // Handle byte tokens
            if (vocab.is_byte(token_id)) {
                if (entry.text.length() == 6 && entry.text.substr(0, 3) == "<0x" && entry.text.back() == '>') {
                    std::string hex_str = entry.text.substr(3, 2);
                    int byte_val = std::stoi(hex_str, nullptr, 16);
                    bytes_[byte_val] = static_cast<llama_token>(token_id);
                }
                continue;
            }

            // Add token and all its suffixes to suffix_to_score
            suffix_to_score[entry.text] = entry.score;

            // Extract suffixes character by character (UTF-8 aware)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(entry.text);
            for (size_t i = 1; i < cpts.size(); ++i) {
                std::string suffix;
                for (size_t j = i; j < cpts.size(); ++j) {
                    suffix += unicode_cpt_to_utf8(cpts[j]);
                }
                if (suffix_to_score.find(suffix) == suffix_to_score.end()) {
                    suffix_to_score[suffix] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }

        // Check that all byte tokens are set
        for (int i = 0; i < 256; ++i) {
            if (bytes_[i] == 0) {
                throw std::runtime_error("Byte token for <0x" + std::to_string(i) + "> is not set");
            }
        }

        // Build suffix list in lexicographical order of reversed strings
        std::vector<std::string> suffixes;
        suffixes.reserve(suffix_to_score.size() + 1);
        for (const auto & pair : suffix_to_score) {
            suffixes.push_back(pair.first);
        }
        suffixes.push_back("");  // Empty suffix

        std::sort(suffixes.begin(), suffixes.end(), [](const std::string & a, const std::string & b) {
            std::string rev_a(a.rbegin(), a.rend());
            std::string rev_b(b.rbegin(), b.rend());
            return rev_a < rev_b;
        });

        // Build suffix_to_id and to_suffix_id_
        std::unordered_map<std::string, int32_t> suffix_to_id;
        int32_t num_pieces = 0;

        for (const auto & suffix : suffixes) {
            suffix_to_id[suffix] = num_pieces;
            if (!suffix.empty()) {
                std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);

                std::string remaining;
                for (size_t i = 1; i < cpts.size(); ++i) {
                    remaining += unicode_cpt_to_utf8(cpts[i]);
                }

                int64_t piece_code = (static_cast<int64_t>(cpts[0]) << 32) | suffix_to_id[remaining];
                to_suffix_id_[piece_code] = num_pieces;

                // Count number of pieces for this suffix
                int32_t pieces_for_suffix = 1; // sentinel row
                for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                    std::string piece;
                    for (int32_t i = 0; i < piece_length; ++i) {
                        piece += unicode_cpt_to_utf8(cpts[i]);
                    }
                    if (suffix_to_score.find(piece) != suffix_to_score.end()) {
                        pieces_for_suffix++;
                    }
                }
                num_pieces += pieces_for_suffix;
            } else {
                num_pieces++;  // Empty suffix contributes one piece (sentinel row)
            }
        }

        // Build flattened table
        table_.resize(num_pieces, std::vector<int32_t>(4, 0));
        int32_t table_idx = 0;

        for (const auto & suffix : suffixes) {
            // Add all prefixes of the suffix to the table (in decreasing order of length)
            std::vector<uint32_t> cpts = unicode_cpts_from_utf8(suffix);
            for (int32_t piece_length = static_cast<int32_t>(cpts.size()); piece_length > 0; --piece_length) {
                std::string piece;
                for (int32_t i = 0; i < piece_length; ++i) {
                    piece += unicode_cpt_to_utf8(cpts[i]);
                }

                auto score_it = suffix_to_score.find(piece);
                if (score_it == suffix_to_score.end()) {
                    continue;
                }

                table_[table_idx][TABLE_PIECE_LENGTH] = piece_length;
                auto token_it = token_to_id.find(piece);
                table_[table_idx][TABLE_TOKEN_ID] = (token_it != token_to_id.end()) ? token_it->second : -1;

                float score = score_it->second;
                table_[table_idx][TABLE_SCORE] = std::isfinite(score) ?
                    static_cast<int32_t>(std::round(score * 1e4)) : INVALID_SCORE;
                table_[table_idx][TABLE_PIECE_ID] = suffix_to_id[piece];

                table_idx++;
            }

            // Add sentinel row
            table_[table_idx][TABLE_PIECE_LENGTH] = 1;
            table_[table_idx][TABLE_TOKEN_ID] = -1;
            table_[table_idx][TABLE_SCORE] = UNKNOWN_SCORE;
            table_idx++;
        }
    }

    std::vector<llama_token> encode(const std::string & text) const {
        std::vector<uint32_t> unicode_data = unicode_cpts_from_utf8(text);
        // Skip the first code point if it is a BOM (Byte Order Mark)
        if (!unicode_data.empty() && unicode_data[0] == 0xFEFF) {
            unicode_data.erase(unicode_data.begin());
        }

        if (unicode_data.empty()) {
            return {};
        }

        const size_t data_len = unicode_data.size();

        // Initialize scores array (dynamic programming)
        std::vector<int64_t> scores(data_len + 1, static_cast<int64_t>(1) << 60);
        scores[data_len] = 0;

        // Path array to track best tokenization
        std::vector<std::vector<int32_t>> path(data_len + 1, std::vector<int32_t>(3, 0));

        int32_t suffix_id = 0;

        // Process from end to beginning
        for (int i = static_cast<int>(data_len) - 1; i >= 0; --i) {
            uint32_t c = unicode_data[i];

            // Find next suffix ID
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int64_t piece_code = (static_cast<int64_t>(c) << 32) | table_[p][TABLE_PIECE_ID];
                auto it = to_suffix_id_.find(piece_code);
                suffix_id = (it != to_suffix_id_.end()) ? it->second : 0;

                if (suffix_id > 0 || table_[p][TABLE_SCORE] == UNKNOWN_SCORE) {
                    break;
                }
            }

            // Update best path
            for (size_t p = suffix_id; p < table_.size(); ++p) {
                int32_t score = table_[p][TABLE_SCORE];
                if (score > INVALID_SCORE) {
                    int32_t piece_length = table_[p][TABLE_PIECE_LENGTH];
                    int64_t s = scores[i + piece_length] - score;

                    if (s < scores[i]) {
                        scores[i] = s;
                        path[i][PATH_TOKEN_LENGTH] = piece_length;
                        path[i][PATH_TOKEN_ID] = table_[p][TABLE_TOKEN_ID];
                        path[i][PATH_NUM_TOKENS] = path[i + piece_length][PATH_NUM_TOKENS] + 1;

                        if (score == UNKNOWN_SCORE) {
                            // Add UTF-8 byte count
                            path[i][PATH_NUM_TOKENS] += (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);
                        }
                    }
                }

                if (score == UNKNOWN_SCORE) {
                    break;
                }
            }
        }

        // Decode the best path
        std::vector<llama_token> token_ids;
        token_ids.reserve(path[0][PATH_NUM_TOKENS]);

        int pos = 0;
        while (pos < static_cast<int>(data_len)) {
            if (path[pos][PATH_TOKEN_ID] >= 0) {
                token_ids.push_back(path[pos][PATH_TOKEN_ID]);
            } else {
                // Fall back to byte tokens
                uint32_t c = unicode_data[pos];
                int s = 1 + (c >= 0x80) + (c >= 0x800) + (c >= 0x10000);

                for (int i = 0; i < s; ++i) {
                    uint8_t b;
                    if (s == 1) {
                        b = c;
                    } else {
                        if (i == 0) {
                            b = (0xF00 >> s) & 0xFF;
                        } else {
                            b = 0x80;
                        }
                    }
                    token_ids.push_back(bytes_[b | ((c >> ((s - i - 1) * 6)) & 0x3F)]);
                }
            }

            assert(path[pos][PATH_TOKEN_LENGTH] > 0);
            pos += path[pos][PATH_TOKEN_LENGTH];
        }

        return token_ids;
    }
private:
    // Constants for table structure
    static constexpr int32_t TABLE_PIECE_LENGTH = 0;
    static constexpr int32_t TABLE_TOKEN_ID     = 1;
    static constexpr int32_t TABLE_SCORE        = 2;
    static constexpr int32_t TABLE_PIECE_ID     = 3;

    // Constants for path array
    static constexpr int32_t PATH_TOKEN_LENGTH  = 0;
    static constexpr int32_t PATH_TOKEN_ID      = 1;
    static constexpr int32_t PATH_NUM_TOKENS    = 2;

    // Score constants
    static constexpr int32_t INVALID_SCORE = -20000000;
    static constexpr int32_t UNKNOWN_SCORE = -10000000;

    // List of tokens in the vocabulary
    std::vector<std::string> tokens_;

    // Mapping from byte code point to token ID (for byte fallback)
    std::vector<llama_token> bytes_;

    // Mapping from piece code to suffix ID
    std::unordered_map<int64_t, int32_t> to_suffix_id_;

    // Flattened table representing the Trie structure
    // Each row contains: [piece_length, token_id, score, piece_id]
    std::vector<std::vector<int32_t>> table_;
};

struct llm_tokenizer_plamo2_session {
    llm_tokenizer_plamo2_session(const llm_tokenizer_plamo2 & tokenizer) : tokenizer(tokenizer) {}

    void tokenize(const std::string & text, std::vector<llama_token> & output) {
        std::vector<llama_token> tokens = tokenizer.encode(text);
        output.insert(output.end(), tokens.begin(), tokens.end());
    }

private:
    const llm_tokenizer_plamo2 & tokenizer;
};

//
// impl
//

typedef enum FRAGMENT_BUFFER_VARIANT_TYPE {
    FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN,
    FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT
} FRAGMENT_BUFFER_VARIANT_TYPE;

struct fragment_buffer_variant {
    fragment_buffer_variant(llama_token _token)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN),
        token(_token),
        raw_text(_dummy),
        offset(0),
        length(0) {}

    fragment_buffer_variant(const std::string & _raw_text, int64_t _offset, int64_t _length)
    :
        type(FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT),
        token((llama_token) - 1),
        raw_text(_raw_text),
        offset(_offset),
        length(_length){
            GGML_ASSERT(_offset >= 0);
            GGML_ASSERT(_length >= 1);
            GGML_ASSERT(offset + length <= raw_text.length());
        }

    const FRAGMENT_BUFFER_VARIANT_TYPE type;
    const llama_token token;
    const std::string _dummy;
    const std::string & raw_text;
    const uint64_t offset;
    const uint64_t length;
};

struct llama_vocab::impl {
    uint32_t n_token_types = 0; // for BERT-style token types

    std::string tokenizer_model;
    std::string tokenizer_pre;

    enum llama_vocab_type     type     = LLAMA_VOCAB_TYPE_SPM;
    enum llama_vocab_pre_type pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;

    int max_token_len = 0; // used for optimizing longest token search

    // default LLaMA special tokens
    // TODO: should we set all of these to LLAMA_TOKEN_NULL?
    llama_token special_bos_id  = 1;
    llama_token special_eos_id  = 2;
    llama_token special_eot_id  = LLAMA_TOKEN_NULL;
    llama_token special_eom_id  = LLAMA_TOKEN_NULL;
    llama_token special_unk_id  = 0;
    llama_token special_sep_id  = LLAMA_TOKEN_NULL;
    llama_token special_pad_id  = LLAMA_TOKEN_NULL;
    llama_token special_mask_id = LLAMA_TOKEN_NULL;

    llama_token linefeed_id = 13;

    // fim tokens
    llama_token special_fim_pre_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_suf_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_mid_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_pad_id = LLAMA_TOKEN_NULL;
    llama_token special_fim_rep_id = LLAMA_TOKEN_NULL; // repo
    llama_token special_fim_sep_id = LLAMA_TOKEN_NULL; // file separator

    // tokenizer flags
    bool add_space_prefix           = false;
    bool add_bos                    = false;
    bool add_eos                    = false;
    bool add_sep                    = false;
    bool ignore_merges              = false;
    bool clean_spaces               = false;  // clean_up_tokenization_spaces
    bool remove_extra_whitespaces   = false;
    bool escape_whitespaces         = true;
    bool treat_whitespace_as_suffix = false;

    std::unordered_map<std::string, llama_token> token_to_id;
    std::vector<token_data>                      id_to_token;

    std::vector<llama_token> cache_special_tokens;
    std::vector<std::string> cache_token_to_piece; // llama_token_to_piece(special = true);
    struct pair_hash {
        size_t operator()(const std::pair<std::string, std::string> & p) const {
            return std::hash<std::string>{}(p.first) ^  //create some hash for pair
                   (std::hash<std::string>{}(p.second) << 1);
        }
    };
    std::unordered_map<std::pair<std::string, std::string>, int, pair_hash> bpe_ranks;

    // set of all tokens that cause "end of generation"
    std::set<llama_token> special_eog_ids;

    std::unique_ptr<llm_tokenizer> tokenizer;

    std::vector<char> precompiled_charsmap;

    impl(const llama_vocab & vocab) : vocab(vocab) {
    }

    ~impl() = default;

    void load(llama_model_loader & ml, const LLM_KV & kv);

    enum llama_vocab_type get_type() const;

    std::string type_name() const;

    bool is_normal      (llama_token id) const;
    bool is_unknown     (llama_token id) const;
    bool is_control     (llama_token id) const;
    bool is_byte        (llama_token id) const;
    bool is_user_defined(llama_token id) const;
    bool is_unused      (llama_token id) const;
    bool is_eog         (llama_token id) const;

    uint8_t token_to_byte(llama_token id) const;

    llama_token_attr token_get_attr(llama_token id) const;

    void init_tokenizer(enum llama_vocab_type type);

    void tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const;

    std::string token_to_piece_for_cache(
                  llama_token   token,
                         bool   special) const;


    std::vector<llama_token> tokenize(
            const std::string & raw_text,
                         bool   add_special,
                         bool   parse_special = false) const;

    int32_t tokenize(
                   const char * text,
                      int32_t   text_len,
                  llama_token * tokens,
                      int32_t   n_tokens_max,
                         bool   add_special,
                         bool   parse_special) const;

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
    const llama_vocab & vocab;
};

void llama_vocab::impl::load(llama_model_loader & ml, const LLM_KV & kv) {
    struct gguf_context * ctx = ml.meta.get();

    // determine vocab type
    {
        ml.get_key(LLM_KV_TOKENIZER_MODEL, tokenizer_model);
        ml.get_key(LLM_KV_TOKENIZER_PRE,   tokenizer_pre, false);

        ml.get_key(LLM_KV_TOKENIZER_TOKEN_TYPE_COUNT, n_token_types, false);

        if (tokenizer_model == "no_vocab" || tokenizer_model == "none") {
            type = LLAMA_VOCAB_TYPE_NONE;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
            linefeed_id     = LLAMA_TOKEN_NULL;

            // read vocab size from metadata
            uint32_t n_tokens = 0;
            if (ml.get_key(LLM_KV_VOCAB_SIZE, n_tokens, false)) {
                LLAMA_LOG_WARN("%s: adding %u dummy tokens\n", __func__, n_tokens);
                id_to_token.resize(n_tokens);
            }

            return;
        }

        if (tokenizer_model == "llama") {
            type = LLAMA_VOCAB_TYPE_SPM;

            // default special tokens
            special_bos_id  = 1;
            special_eos_id  = 2;
            special_unk_id  = 0;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "bert") {
            type = LLAMA_VOCAB_TYPE_WPM;

            // default special tokens
            special_bos_id  = 101;
            special_eos_id  = LLAMA_TOKEN_NULL;
            special_unk_id  = 100;
            special_sep_id  = 102;
            special_pad_id  = 0;
            special_mask_id = 103;

            add_sep = true;
        } else if (tokenizer_model == "gpt2") {
            type = LLAMA_VOCAB_TYPE_BPE;

            // read bpe merges and populate bpe ranks
            const int merges_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_MERGES).c_str());
            if (merges_keyidx == -1) {
                throw std::runtime_error("cannot find tokenizer merges in model file\n");
            }

            const int n_merges = gguf_get_arr_n(ctx, merges_keyidx);
            for (int i = 0; i < n_merges; i++) {
                const std::string word = gguf_get_arr_str(ctx, merges_keyidx, i);
                //GGML_ASSERT(unicode_cpts_from_utf8(word).size() > 0);

                std::string first;
                std::string second;

                const size_t pos = word.find(' ', 1);

                if (pos != std::string::npos) {
                    first  = word.substr(0, pos);
                    second = word.substr(pos + 1);
                }

                bpe_ranks.emplace(std::make_pair(first, second), i);
            }

            // default special tokens
            special_bos_id  = 11;
            special_eos_id  = 11;
            special_unk_id  = LLAMA_TOKEN_NULL;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = LLAMA_TOKEN_NULL;
            special_mask_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "t5") {
            type = LLAMA_VOCAB_TYPE_UGM;

            // default special tokens
            special_bos_id  = LLAMA_TOKEN_NULL;
            special_eos_id  = 1;
            special_unk_id  = 2;
            special_sep_id  = LLAMA_TOKEN_NULL;
            special_pad_id  = 0;
            special_mask_id = LLAMA_TOKEN_NULL;

            const int precompiled_charsmap_keyidx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_PRECOMPILED_CHARSMAP).c_str());
            if (precompiled_charsmap_keyidx != -1) {
                const gguf_type pc_type = gguf_get_arr_type(ctx, precompiled_charsmap_keyidx);
                const size_t n_precompiled_charsmap = gguf_get_arr_data_n(ctx, precompiled_charsmap_keyidx);
                const char * pc = (const char *) gguf_get_arr_data(ctx, precompiled_charsmap_keyidx);
                precompiled_charsmap.assign(pc, pc + n_precompiled_charsmap);
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                // correct endiannes of data in precompiled_charsmap binary blob
                uint32_t * xcda_blob_size = (uint32_t *) &precompiled_charsmap[0];
                *xcda_blob_size = __builtin_bswap32(*xcda_blob_size);
                assert(*xcda_blob_size + sizeof(uint32_t) < n_precompiled_charsmap);
                size_t xcda_array_size = *xcda_blob_size / sizeof(uint32_t);
                uint32_t * xcda_array = (uint32_t *) &precompiled_charsmap[sizeof(uint32_t)];
                for (size_t i = 0; i < xcda_array_size; ++i) {
                    xcda_array[i] = __builtin_bswap32(xcda_array[i]);
                }
#endif
            }
        } else if (tokenizer_model == "rwkv") {
            type = LLAMA_VOCAB_TYPE_RWKV;

            // default special tokens
            special_bos_id = LLAMA_TOKEN_NULL;
            special_eos_id = LLAMA_TOKEN_NULL;
            special_unk_id = LLAMA_TOKEN_NULL;
            special_sep_id = LLAMA_TOKEN_NULL;
            special_pad_id = LLAMA_TOKEN_NULL;
        } else if (tokenizer_model == "plamo2") {
            type = LLAMA_VOCAB_TYPE_PLAMO2;

            // PLaMo-2 default special tokens (these will be overridden by model config)
            special_bos_id = 1;  // <|plamo:bos|>
            special_eos_id = 2;  // <|plamo:eos|>
            special_unk_id = 0;  // <|plamo:unk|>
            special_sep_id = LLAMA_TOKEN_NULL;
            special_pad_id = 3;  // <|plamo:pad|>
            special_mask_id = LLAMA_TOKEN_NULL;
        } else {
            throw std::runtime_error(format("unknown tokenizer: '%s'", tokenizer_model.c_str()));
        }

        // for now, only BPE models have pre-tokenizers
        if (type == LLAMA_VOCAB_TYPE_BPE) {
            add_space_prefix = false;
            clean_spaces = true;
            if (tokenizer_pre == "default") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            } else if (
                    tokenizer_pre == "llama3"   ||
                    tokenizer_pre == "llama-v3" ||
                    tokenizer_pre == "llama-bpe"||
                    tokenizer_pre == "falcon3"  ||
                    tokenizer_pre == "falcon-h1" ||
                    tokenizer_pre == "pixtral"  ||
                    tokenizer_pre == "midm-2.0" ||
                    tokenizer_pre == "lfm2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_LLAMA3;
                ignore_merges = true;
                add_bos = true;
            } else if (
                    tokenizer_pre == "deepseek-llm") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-coder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "deepseek-v3") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "falcon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_FALCON;
            } else if (
                    tokenizer_pre == "mpt") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MPT;
            } else if (
                    tokenizer_pre == "starcoder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_STARCODER;
            } else if (
                    tokenizer_pre == "gpt-2"   ||
                    tokenizer_pre == "phi-2"   ||
                    tokenizer_pre == "jina-es" ||
                    tokenizer_pre == "jina-de" ||
                    tokenizer_pre == "gigachat"   ||
                    tokenizer_pre == "jina-v2-es" ||
                    tokenizer_pre == "jina-v2-de" ||
                    tokenizer_pre == "a.x-4.0" ||
                    tokenizer_pre == "mellum") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                    tokenizer_pre == "jina-v1-en" ||
                    tokenizer_pre == "jina-v2-code" ||
                    tokenizer_pre == "roberta-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
                add_sep = true;
            } else if (
                    tokenizer_pre == "refact") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_REFACT;
            } else if (
                tokenizer_pre == "command-r") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_COMMAND_R;
                clean_spaces = false;
            } else if (
                    tokenizer_pre == "qwen2" ||
                    tokenizer_pre == "deepseek-r1-qwen" ||
                    tokenizer_pre == "kormo") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "stablelm2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_STABLELM2;
            } else if (
                tokenizer_pre == "olmo") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_OLMO;
            } else if (
                tokenizer_pre == "dbrx") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_DBRX;
            } else if (
                tokenizer_pre == "smaug-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SMAUG;
            } else if (
                tokenizer_pre == "poro-chat") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_PORO;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "glm4" ||
                tokenizer_pre == "chatglm-bpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CHATGLM4;
                special_bos_id = LLAMA_TOKEN_NULL;
            } else if (
                tokenizer_pre == "viking") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_VIKING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "jais") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_JAIS;
            } else if (
                tokenizer_pre == "tekken") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_TEKKEN;
                clean_spaces = false;
                ignore_merges = true;
                add_bos = true;
            } else if (
                tokenizer_pre == "smollm") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SMOLLM;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "codeshell") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CODESHELL;
            } else if (
                tokenizer_pre == "bloom") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_BLOOM;
            } else if (
                tokenizer_pre == "gpt3-finnish") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH;
            } else if (
                tokenizer_pre == "exaone") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_EXAONE;
            } else if (
                tokenizer_pre == "exaone4") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT2;
            } else if (
                tokenizer_pre == "chameleon") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_CHAMELEON;
                add_bos = true;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minerva-7b") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINERVA;
            } else if (
                tokenizer_pre == "megrez") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_QWEN2;
            } else if (
                    tokenizer_pre == "gpt-4o" ||
                    tokenizer_pre == "llama4") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GPT4O;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "superbpe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SUPERBPE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "trillion") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_TRILLION;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "granite-docling") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GRANITE_DOCLING;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "bailingmoe" ||
                tokenizer_pre == "bailingmoe2" ||
                tokenizer_pre == "llada-moe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_BAILINGMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "seed-coder") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_SEED_CODER;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_HUNYUAN;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "hunyuan-dense") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_HUNYUAN_DENSE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "kimi-k2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_KIMI_K2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "grok-2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_GROK_2;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "afmoe") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_AFMOE;
                clean_spaces = false;
            } else if (
                tokenizer_pre == "minimax-m2") {
                pre_type = LLAMA_VOCAB_PRE_TYPE_MINIMAX_M2;
                clean_spaces = false;
            } else {
                LLAMA_LOG_WARN("%s: missing or unrecognized pre-tokenizer type, using: 'default'\n", __func__);
                pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            }
        } else if (type == LLAMA_VOCAB_TYPE_SPM) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = true;
            clean_spaces = false;
            add_bos = true;
            add_eos = false;
        } else if (type == LLAMA_VOCAB_TYPE_WPM) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = true;
            add_bos = true;
            add_eos = false;
            add_sep = true;
        } else if (type == LLAMA_VOCAB_TYPE_UGM) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_bos = false;
            add_eos = true;
        } else if (type == LLAMA_VOCAB_TYPE_RWKV) {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
            add_space_prefix = false;
            clean_spaces = false;
            add_bos = false;
            add_eos = false;
        } else {
            pre_type = LLAMA_VOCAB_PRE_TYPE_DEFAULT;
        }

        ml.get_key(LLM_KV_TOKENIZER_ADD_PREFIX,      add_space_prefix,         false);
        ml.get_key(LLM_KV_TOKENIZER_REMOVE_EXTRA_WS, remove_extra_whitespaces, false);
    }

    const int token_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_LIST).c_str());
    if (token_idx == -1) {
        throw std::runtime_error("cannot find tokenizer vocab in model file\n");
    }

    const float * scores = nullptr;
    const int score_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_SCORES).c_str());
    if (score_idx != -1) {
        scores = (const float * ) gguf_get_arr_data(ctx, score_idx);
    }

    const int * toktypes = nullptr;
    const int toktype_idx = gguf_find_key(ctx, kv(LLM_KV_TOKENIZER_TOKEN_TYPE).c_str());
    if (toktype_idx != -1) {
        toktypes = (const int * ) gguf_get_arr_data(ctx, toktype_idx);
    }

    uint32_t n_tokens = gguf_get_arr_n(ctx, token_idx);
    id_to_token.resize(n_tokens);

    for (uint32_t i = 0; i < n_tokens; i++) {
        std::string word = gguf_get_arr_str(ctx, token_idx, i);
        if (word.empty()) {
            LLAMA_LOG_WARN("%s: empty token at index %u\n", __func__, i);
            word = "[EMPTY_" + std::to_string(i) + "]";
        }

        token_to_id[word] = i;
        max_token_len = std::max(max_token_len, (int) word.size());

        auto & token_data = id_to_token[i];
        token_data.text  = std::move(word);
        token_data.score = scores ? scores[i] : 0.0f;
        token_data.attr  = LLAMA_TOKEN_ATTR_NORMAL;

        if (toktypes) {  //TODO: remove, required until per token attributes are available from GGUF file
            switch(toktypes[i]) {
                case LLAMA_TOKEN_TYPE_UNKNOWN:      token_data.attr = LLAMA_TOKEN_ATTR_UNKNOWN;      break;
                case LLAMA_TOKEN_TYPE_UNUSED:       token_data.attr = LLAMA_TOKEN_ATTR_UNUSED;       break;
                case LLAMA_TOKEN_TYPE_NORMAL:       token_data.attr = LLAMA_TOKEN_ATTR_NORMAL;       break;
                case LLAMA_TOKEN_TYPE_CONTROL:      token_data.attr = LLAMA_TOKEN_ATTR_CONTROL;      break;
                case LLAMA_TOKEN_TYPE_USER_DEFINED: token_data.attr = LLAMA_TOKEN_ATTR_USER_DEFINED; break;
                case LLAMA_TOKEN_TYPE_BYTE:         token_data.attr = LLAMA_TOKEN_ATTR_BYTE;         break;
                case LLAMA_TOKEN_TYPE_UNDEFINED:    token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
                default:                            token_data.attr = LLAMA_TOKEN_ATTR_UNDEFINED;    break;
            }
        }
    }
    GGML_ASSERT(id_to_token.size() == token_to_id.size());

    init_tokenizer(type);

    // determine the newline token: LLaMA "<0x0A>" == 10 == '\n', Falcon 193 == '\n'
    if (type == LLAMA_VOCAB_TYPE_SPM) {
        try {
            linefeed_id = vocab.byte_to_token('\n');
        } catch (const std::exception & e) {
            LLAMA_LOG_WARN("%s: SPM vocabulary, but newline token not found: %s! Using special_pad_id instead.", __func__, e.what());
            linefeed_id = special_pad_id;
        }
    } else if (type == LLAMA_VOCAB_TYPE_WPM) {
        linefeed_id = special_pad_id;
    } else if (type == LLAMA_VOCAB_TYPE_RWKV) {
        const std::vector<int> ids = tokenize("\n", false);
        GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        linefeed_id = ids[0];
    } else {
        const std::vector<int> ids = tokenize("\n", false);

        //GGML_ASSERT(!ids.empty() && "model vocab missing newline token");
        if (ids.empty()) {
            LLAMA_LOG_WARN("%s: model vocab missing newline token, using special_pad_id instead\n", __func__);
            linefeed_id = special_pad_id;
        } else {
            linefeed_id = ids[0];
        }
    }

    // special tokens
    {
        const std::vector<std::pair<enum llm_kv, int32_t &>> special_token_types = {
            { LLM_KV_TOKENIZER_BOS_ID,     special_bos_id     },
            { LLM_KV_TOKENIZER_EOS_ID,     special_eos_id     },
            { LLM_KV_TOKENIZER_EOT_ID,     special_eot_id     },
            { LLM_KV_TOKENIZER_EOM_ID,     special_eom_id     },
            { LLM_KV_TOKENIZER_UNK_ID,     special_unk_id     },
            { LLM_KV_TOKENIZER_SEP_ID,     special_sep_id     },
            { LLM_KV_TOKENIZER_PAD_ID,     special_pad_id     },
            { LLM_KV_TOKENIZER_MASK_ID,    special_mask_id    },
            { LLM_KV_TOKENIZER_FIM_PRE_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_FIM_SUF_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_FIM_MID_ID, special_fim_mid_id },
            { LLM_KV_TOKENIZER_FIM_PAD_ID, special_fim_pad_id },
            { LLM_KV_TOKENIZER_FIM_REP_ID, special_fim_rep_id },
            { LLM_KV_TOKENIZER_FIM_SEP_ID, special_fim_sep_id },

            // deprecated
            { LLM_KV_TOKENIZER_PREFIX_ID, special_fim_pre_id },
            { LLM_KV_TOKENIZER_SUFFIX_ID, special_fim_suf_id },
            { LLM_KV_TOKENIZER_MIDDLE_ID, special_fim_mid_id },
        };

        for (const auto & it : special_token_types) {
            const std::string & key = kv(std::get<0>(it));
            int32_t & id = std::get<1>(it);

            uint32_t new_id;
            if (!ml.get_key(std::get<0>(it), new_id, false)) {
                continue;
            }
            if (new_id >= id_to_token.size()) {
                LLAMA_LOG_WARN("%s: bad special token: '%s' = %u, using default id %d\n",
                    __func__, key.c_str(), new_id, id);
            } else {
                id = new_id;
            }
        }

        // Handle add_bos, add_eos and add_sep
        {
            bool temp = true;

            if (ml.get_key(LLM_KV_TOKENIZER_ADD_BOS, temp, false)) {
                add_bos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false)) {
                add_eos = temp;
            }
            if (ml.get_key(LLM_KV_TOKENIZER_ADD_SEP, temp, false)) {
                add_sep = temp;
            }
        }

        // auto-detect special tokens by text
        // TODO: convert scripts should provide these tokens through the KV metadata LLM_KV_TOKENIZER_...
        //       for now, we apply this workaround to find the tokens based on their text

        for (const auto & t : token_to_id) {
            // find EOT token: "<|eot_id|>", "<|im_end|>", "<end_of_turn>", etc.
            if (special_eot_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eot_id|>"
                        || t.first == "<|im_end|>"
                        || t.first == "<|end|>"
                        || t.first == "<end_of_turn>"
                        || t.first == "<|endoftext|>"
                        || t.first == "<|end_of_text|>" // granite
                        || t.first == "<EOT>"
                        || t.first == "_<EOT>"
                        || t.first == "<｜end▁of▁sentence｜>" // DeepSeek
                        || t.first == "<end_of_utterance>" // smoldocling
                   ) {
                    special_eot_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find EOM token: "<|eom_id|>"
            if (special_eom_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|eom_id|>"
                        ) {
                    special_eom_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PRE token: "<|fim_prefix|>", "<fim-prefix>", "<PRE>", etc.
            if (special_fim_pre_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_prefix|>"  // Qwen
                        || t.first == "<fim-prefix>"
                        || t.first == "<fim_prefix>"    // Granite
                        || t.first == "<｜fim▁begin｜>" // DeepSeek
                        || t.first == "<PRE>"
                        || t.first == "▁<PRE>"          // CodeLlama
                        || t.first == "<|code_prefix|>" // GLM-4.5
                        ) {
                    special_fim_pre_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SUF token: "<|fim_suffix|>", "<fim-suffix>", "<SUF>", etc.
            if (special_fim_suf_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_suffix|>" // Qwen
                        || t.first == "<fim-suffix>"
                        || t.first == "<fim_suffix>"   // Granite
                        || t.first == "<｜fim▁hole｜>" // DeepSeek
                        || t.first == "<SUF>"
                        || t.first == "▁<SUF>"         // CodeLlama
                        || t.first == "<|code_suffix|>" // GLM-4.5
                        ) {
                    special_fim_suf_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_MID token: "<|fim_middle|>", "<fim-middle>", "<MID>", etc.
            if (special_fim_mid_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_middle|>" // Qwen
                        || t.first == "<fim-middle>"
                        || t.first == "<fim_middle>"   // Granite
                        || t.first == "<｜fim▁end｜>"  // DeepSeek
                        || t.first == "<MID>"
                        || t.first == "▁<MID>"         // CodeLlama
                        || t.first == "<|code_middle|>" // GLM-4.5
                        ) {
                    special_fim_mid_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_PAD token: "<|fim_pad|>", "<fim-pad>", "<PAD>", etc.
            if (special_fim_pad_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_pad|>" // Qwen
                        || t.first == "<fim-pad>"
                        || t.first == "<fim_pad>"   // Granite
                        || t.first == "<PAD>"
                        ) {
                    special_fim_pad_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_REP token: "<|fim_repo|>", "<fim-repo>", "<REP>", etc.
            if (special_fim_rep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|fim_repo|>"  // Qwen
                        || t.first == "<|repo_name|>"
                        || t.first == "<fim-repo>"
                        || t.first == "<REPO>"
                        || t.first == "<reponame>"    // Granite
                        ) {
                    special_fim_rep_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }

            // find FIM_SEP token: "<|file_sep|>"
            if (special_fim_sep_id == LLAMA_TOKEN_NULL) {
                if (false
                        || t.first == "<|file_sep|>" // Qwen
                        ) {
                    special_fim_sep_id = t.second;
                    if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                        LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                                __func__, t.second, t.first.c_str());
                        id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                    }
                }
            }
        }

        // maintain a list of tokens that cause end-of-generation
        // this is currently determined based on the token text, which is obviously not ideal
        // ref: https://github.com/ggerganov/llama.cpp/issues/9606
        special_eog_ids.clear();

        if (special_fim_pad_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_pad_id) == 0) {
            special_eog_ids.insert(special_fim_pad_id);
        }

        if (special_fim_rep_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_rep_id) == 0) {
            special_eog_ids.insert(special_fim_rep_id);
        }

        if (special_fim_sep_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_fim_sep_id) == 0) {
            special_eog_ids.insert(special_fim_sep_id);
        }

        for (const auto & t : token_to_id) {
            if (false
                    || t.first == "<|eot_id|>"
                    || t.first == "<|im_end|>"
                    || t.first == "<|end|>"
                    || t.first == "<|return|>" // o200k_harmony
                    || t.first == "<|call|>"   // o200k_harmony
                    || t.first == "<end_of_turn>"
                    || t.first == "<|endoftext|>"
                    || t.first == "<|eom_id|>"
                    || t.first == "<EOT>"
                    || t.first == "_<EOT>"
                    || t.first == "<|end_of_text|>"
                    || t.first == "<end_of_utterance>" // smoldocling
               ) {
                special_eog_ids.insert(t.second);
                if ((id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL) == 0) {
                    LLAMA_LOG_WARN("%s: control-looking token: %6d '%s' was not control-type; this is probably a bug in the model. its type will be overridden\n",
                            __func__, t.second, t.first.c_str());
                    id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_CONTROL;
                }
            } else {
                // token is control, but not marked as EOG -> print a debug log
                if (id_to_token[t.second].attr & LLAMA_TOKEN_ATTR_CONTROL && special_eog_ids.count(t.second) == 0) {
                    LLAMA_LOG_DEBUG("%s: control token: %6d '%s' is not marked as EOG\n",
                            __func__, t.second, t.first.c_str());
                }
            }
        }

        // @ngxson : quick hack for gpt-oss, always render these tokens
        for (const auto & t : token_to_id) {
            if (t.first == "<|channel|>" || t.first == "<|message|>" || t.first == "<|start|>" || t.first == "<|constrain|>") {
                id_to_token[t.second].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
            }
        }

        // sanity checks
        if (special_eos_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eos_id) == 0) {
            special_eog_ids.insert(special_eos_id);
            LLAMA_LOG_WARN("%s: special_eos_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eot_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eot_id) == 0) {
            special_eog_ids.insert(special_eot_id);
            LLAMA_LOG_WARN("%s: special_eot_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        if (special_eom_id != LLAMA_TOKEN_NULL && special_eog_ids.count(special_eom_id) == 0) {
            special_eog_ids.insert(special_eom_id);
            LLAMA_LOG_WARN("%s: special_eom_id is not in special_eog_ids - the tokenizer config may be incorrect\n", __func__);
        }

        // TODO: workaround for o200k_harmony tokenizer: the "<|end|>" token should not be EOG
        //       we don't have a good way to detect this, so for now, if we have "<|return|>" and "<|call|>" tokens,
        //       we remove the "<|end|>" token from the EOG list
        {
            bool has_return = false;
            bool has_call   = false;
            bool has_end    = false;

            llama_token end_id = LLAMA_TOKEN_NULL;

            LLAMA_LOG_INFO("%s: printing all EOG tokens:\n", __func__);
            for (auto tid : special_eog_ids) {
                LLAMA_LOG_INFO("%s:   - %d ('%s')\n", __func__, tid, id_to_token[tid].text.c_str());

                if (id_to_token[tid].text == "<|return|>") {
                    has_return = true;
                } else if (id_to_token[tid].text == "<|call|>") {
                    has_call = true;
                } else if (id_to_token[tid].text == "<|end|>") {
                    has_end = true;
                    end_id = tid;
                }
            }

            if (has_return && has_call && has_end) {
                special_eog_ids.erase(end_id);
                id_to_token[end_id].attr = LLAMA_TOKEN_ATTR_USER_DEFINED;
                LLAMA_LOG_WARN("%s: special_eog_ids contains both '<|return|>' and '<|call|>' tokens, removing '<|end|>' token from EOG list\n", __func__);
            }
        }
    }

    // build special tokens cache
    {
        for (llama_token id = 0; id < (llama_token) n_tokens; ++id) {
            if (id_to_token[id].attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED | LLAMA_TOKEN_ATTR_UNKNOWN)) {
                cache_special_tokens.push_back(id);
            }
        }

        std::sort(cache_special_tokens.begin(), cache_special_tokens.end(),
            [&] (const llama_token a, const llama_token b) {
                return id_to_token[a].text.size() > id_to_token[b].text.size();
            }
        );

        LLAMA_LOG_INFO("%s: special tokens cache size = %u\n", __func__, (uint32_t) cache_special_tokens.size());
    }

    // build token to piece cache
    {
        size_t size_cache = 0;

        std::vector<std::string> cache(n_tokens);

        for (uint32_t id = 0; id < n_tokens; ++id) {
            cache[id] = token_to_piece_for_cache(id, true);

            size_cache += cache[id].size();
        }

        std::swap(cache_token_to_piece, cache);

        LLAMA_LOG_INFO("%s: token to piece cache size = %.4f MB\n", __func__, size_cache / 1024.0 / 1024.0);
    }

    // Handle per token attributes
    //NOTE: Each model customizes per token attributes.
    //NOTE: Per token attributes are missing from the GGUF file.
    //TODO: Extract attributes from GGUF file.
    {
        auto _contains_any = [] (const std::string & str, const std::vector<std::string_view> & substrs) -> bool {
            for (const auto & substr : substrs) {
                if (str.find(substr) != std::string::npos) {
                    return true;
                }
            }
            return false;
        };

        auto _set_tokenid_attr = [&] (const llama_token id, llama_token_attr attr, bool value) {
            uint32_t current = id_to_token.at(id).attr;
            current = value ? (current | attr) : (current & ~attr);
            id_to_token[id].attr = (llama_token_attr) current;
        };

        auto _set_token_attr = [&] (const std::string & token, llama_token_attr attr, bool value) {
            _set_tokenid_attr(token_to_id.at(token), attr, value);
        };

        std::string model_name;
        std::string tokenizer_pre;
        std::string general_arch;

        ml.get_key(LLM_KV_GENERAL_NAME,  model_name,    false);
        ml.get_key(LLM_KV_TOKENIZER_PRE, tokenizer_pre, false);
        ml.get_key(LLM_KV_GENERAL_ARCHITECTURE, general_arch, false);

        // model name to lowercase
        std::transform(model_name.begin(), model_name.end(), model_name.begin(),
            [] (const std::string::value_type x) {
                return std::tolower(x);
            }
        );

        // set attributes by model/tokenizer/architecture name
        if (false
                || _contains_any(tokenizer_pre, {"jina-v2-de", "jina-v2-es", "jina-v2-code"})
                || _contains_any(general_arch, {"nomic-bert-moe", "jina-bert-v3"})
           ) {
            if (token_to_id.count("<mask>") == 0) {
                LLAMA_LOG_WARN("%s: Mask token is missing in vocab, please reconvert model!\n", __func__);
            } else {
                _set_token_attr("<mask>", LLAMA_TOKEN_ATTR_LSTRIP, true);
            }
        } else if (_contains_any(model_name, {"phi-3", "phi3"})) {
            for (auto id : cache_special_tokens) {
                _set_tokenid_attr(id, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"</s>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, true);
            }
            for (const auto * token : {"<unk>", "<s>", "<|endoftext|>"}) {
                _set_token_attr(token, LLAMA_TOKEN_ATTR_RSTRIP, false);
            }
        }
    }
}

enum llama_vocab_type llama_vocab::impl::get_type() const {
    return type;
}

std::string llama_vocab::impl::type_name() const{
    switch (type) {
        case LLAMA_VOCAB_TYPE_NONE:   return "no vocab";
        case LLAMA_VOCAB_TYPE_SPM:    return "SPM";
        case LLAMA_VOCAB_TYPE_BPE:    return "BPE";
        case LLAMA_VOCAB_TYPE_WPM:    return "WPM";
        case LLAMA_VOCAB_TYPE_UGM:    return "UGM";
        case LLAMA_VOCAB_TYPE_RWKV:   return "RWKV";
        case LLAMA_VOCAB_TYPE_PLAMO2: return "PLaMo2";
        default:                      return "unknown";
    }
}

bool llama_vocab::impl::is_normal(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_NORMAL;
}

bool llama_vocab::impl::is_unknown(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNKNOWN;
}

bool llama_vocab::impl::is_control(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_CONTROL;
}

bool llama_vocab::impl::is_byte(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_BYTE;
}

bool llama_vocab::impl::is_user_defined(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_USER_DEFINED;
}

bool llama_vocab::impl::is_unused(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token[id].attr & LLAMA_TOKEN_ATTR_UNUSED;
}

bool llama_vocab::impl::is_eog(llama_token id) const {
    return id != LLAMA_TOKEN_NULL && special_eog_ids.count(id) > 0;
}

uint8_t llama_vocab::impl::token_to_byte(llama_token id) const {
    GGML_ASSERT(get_type() != LLAMA_VOCAB_TYPE_NONE);
    GGML_ASSERT(is_byte(id));
    const auto & token_data = id_to_token.at(id);
    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            auto buf = token_data.text.substr(3, 2);
            return strtol(buf.c_str(), NULL, 16);
        }
        case LLAMA_VOCAB_TYPE_BPE: {
            GGML_ABORT("fatal error");
        }
        case LLAMA_VOCAB_TYPE_WPM: {
            GGML_ABORT("fatal error");
        }
        default:
            GGML_ABORT("fatal error");
    }
}

llama_token_attr llama_vocab::impl::token_get_attr(llama_token id) const {
    GGML_ASSERT(type != LLAMA_VOCAB_TYPE_NONE);
    return id_to_token.at(id).attr;
}

void llama_vocab::impl::init_tokenizer(enum llama_vocab_type type) {
    LLAMA_LOG_DEBUG("%s: initializing tokenizer for type %d\n", __func__, type);

    switch (type) {
        case LLAMA_VOCAB_TYPE_SPM:
            tokenizer = std::make_unique<llm_tokenizer_spm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_BPE:
            tokenizer = std::make_unique<llm_tokenizer_bpe>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_WPM:
            tokenizer = std::make_unique<llm_tokenizer_wpm>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_UGM:
            tokenizer = std::make_unique<llm_tokenizer_ugm>(vocab, precompiled_charsmap);
            break;
        case LLAMA_VOCAB_TYPE_RWKV:
            tokenizer = std::make_unique<llm_tokenizer_rwkv>(vocab);
            break;
        case LLAMA_VOCAB_TYPE_PLAMO2:
            tokenizer = std::make_unique<llm_tokenizer_plamo2>(vocab);
            break;
        default:
            GGML_ABORT("unsupported vocab type");
    }
}

//
// (de-) tokenize
//

// #define PRETOKENIZERDEBUG

void llama_vocab::impl::tokenizer_st_partition(std::forward_list<fragment_buffer_variant> & buffer, bool parse_special) const {
    // for each special token
    for (const llama_token special_id : cache_special_tokens) {
        const auto & data = vocab.get_token_data(special_id);
        const auto & text = data.text;

        if (!parse_special && (data.attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN))) {
            // Ignore control and unknown tokens when parse_special == false
            continue;
            // User-defined tokens are still pre-tokenized before everything else
            // ref: https://github.com/huggingface/tokenizers/blob/fdd26ba9a3f0c133427aab0423888cbde91362d7/tokenizers/src/tokenizer/mod.rs#L726
            // This is mostly relevant for neox-style tokenizers (mpt, olmo, stablelm, etc.)
        }

        // for each text fragment
        std::forward_list<fragment_buffer_variant>::iterator it = buffer.begin();
        while (it != buffer.end()) {
            auto & fragment = (*it);

            // if a fragment is text ( not yet processed )
            if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                const auto & raw_text = fragment.raw_text;

                auto raw_text_base_offset = fragment.offset;
                auto raw_text_base_length = fragment.length;

                // loop over the text
                while (true) {
                    // find the first occurrence of a given special token in this fragment
                    //  passing offset argument only limit the "search area" but match coordinates
                    //  are still relative to the source full raw_text
                    //  string_view begins at pos 0 for the same reason
                    auto match = std::string_view(raw_text.data(), raw_text_base_offset + raw_text_base_length).find(text, raw_text_base_offset);

                    // no occurrences found, stop processing this fragment for a given special token
                    if (match == std::string::npos) break;

#ifdef PRETOKENIZERDEBUG
                    LLAMA_LOG_WARN("FF: (%ld %ld %ld) '%s'\n", raw_text->length(), raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    auto source = std::distance(buffer.begin(), it);

                    // if match is further than base offset
                    //  then we have some text to the left of it
                    if (match > raw_text_base_offset) {
                        // left
                        const int64_t left_reminder_offset = raw_text_base_offset + 0;
                        int64_t left_reminder_length = match - raw_text_base_offset;

                        if (data.attr & LLAMA_TOKEN_ATTR_LSTRIP) {
                            while (left_reminder_length > 0 && isspace(raw_text[left_reminder_offset + left_reminder_length - 1])) {
                                left_reminder_length--;
                            }
                        }

                        if (left_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, left_reminder_offset, left_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FL: (%ld %ld) '%s'\n", left_reminder_offset, left_reminder_length, raw_text->substr(left_reminder_offset, left_reminder_length).c_str());
#endif
                    }

                    // special token
                    buffer.emplace_after(it, special_id);
                    it++;

                    // right
                    if (match + text.length() < raw_text_base_offset + raw_text_base_length) {
                        int64_t right_reminder_offset = match + text.length();
                        int64_t right_reminder_length = raw_text_base_length - ((match - raw_text_base_offset) + text.length());

                        if (data.attr & LLAMA_TOKEN_ATTR_RSTRIP) {
                            while (right_reminder_length > 0 && isspace(raw_text[right_reminder_offset])) {
                                right_reminder_offset++;
                                right_reminder_length--;
                            }
                        }

                        if (right_reminder_length > 0) {
                            buffer.emplace_after(it, raw_text, right_reminder_offset, right_reminder_length);
                            it++;
                        }

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("FR: (%ld %ld) '%s'\n", right_reminder_offset, right_reminder_length, raw_text->substr(right_reminder_offset, right_reminder_length).c_str());
#endif

                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }

                        // repeat for the right side
                        raw_text_base_offset = right_reminder_offset;
                        raw_text_base_length = right_reminder_length;

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("RR: (%ld %ld) '%s'\n", raw_text_base_offset, raw_text_base_length, raw_text->substr(raw_text_base_offset, raw_text_base_length).c_str());
#endif
                    } else {
                        if (source == 0) {
                            buffer.erase_after(buffer.before_begin());
                        } else {
                            buffer.erase_after(std::next(buffer.begin(), (source - 1)));
                        }
                        break;
                    }
                }
            }
            it++;
        }
    }
}

// NOTE: avoid ever using this except for building the token_to_piece caches
std::string llama_vocab::impl::token_to_piece_for_cache(llama_token token, bool special) const {
    std::string piece;
    piece.resize(piece.capacity());  // using string internal cache
    const int n_chars = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
    if (n_chars < 0) {
        piece.resize(-n_chars);
        int check = vocab.token_to_piece(token, &piece[0], piece.size(), 0, special);
        GGML_ASSERT(check == -n_chars);
    }
    else {
        piece.resize(n_chars);
    }

    return piece;
}

static void llama_escape_whitespace(std::string & text) {
    replace_all(text, " ", "\xe2\x96\x81");
}

static void llama_unescape_whitespace(std::string & word) {
    replace_all(word, "\xe2\x96\x81", " ");
}

static std::string llama_decode_text(const std::string & text) {
    std::string decoded_text;

    const auto cpts = unicode_cpts_from_utf8(text);
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        try {
            decoded_text += unicode_utf8_to_byte(utf8);
        } catch (const std::out_of_range & /*e*/) {
            decoded_text += "[UNK_BYTE_0x";
            for (const auto c : utf8) {
                decoded_text += format("%02x", (uint8_t) c);
            }
            decoded_text += text + "]";
        }
    }

    return decoded_text;
}

std::vector<llama_token> llama_vocab::impl::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    std::vector<llama_token> output;
    std::forward_list<fragment_buffer_variant> fragment_buffer;

    if (!raw_text.empty()) {
        fragment_buffer.emplace_front(raw_text, 0, raw_text.length());
        tokenizer_st_partition(fragment_buffer, parse_special);
    }

    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_SPM:
            {
                // OG tokenizer behavior:
                //
                // tokenizer.encode('', add_special_tokens=True)  returns [1]
                // tokenizer.encode('', add_special_tokens=False) returns []

                bool is_prev_special = true;  // prefix with space if first token

                if (add_special && add_bos) {
                    GGML_ASSERT(special_bos_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_bos_id);
                    is_prev_special = true;
                }

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text;

                        // prefix with space if previous is special
                        if (add_space_prefix && is_prev_special) {
                            text = ' ';
                        }

                        text += fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        llama_escape_whitespace(text);
                        llm_tokenizer_spm_session session(vocab);
                        session.tokenize(text, output);
                        is_prev_special = false;
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                        is_prev_special = true;
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    LLAMA_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    GGML_ASSERT(special_eos_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_BPE:
            {
                llm_tokenizer_bpe_session session(vocab, *static_cast<const llm_tokenizer_bpe *>(tokenizer.get()));
                // it calls some other methods that are not exist in llm_tokenizer,
                // here just cast it to bpe tokenizer object
                if (add_special) {
                    session.append_bos(output);
                }
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        session.append(fragment.token, output);
                    }
                }

                if (add_special) {
                    session.append_eos(output);
                    session.check_double_bos_eos(output);
                }
            } break;
        case LLAMA_VOCAB_TYPE_WPM:
            {
                if (add_special) {
                    GGML_ASSERT(special_bos_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_bos_id);
                }

                llm_tokenizer_wpm_session session(vocab);

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special) {
                    GGML_ASSERT(special_sep_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_sep_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_UGM:
            {
                if (add_special && add_bos) {
                    GGML_ASSERT(special_bos_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_bos_id);
                }
                llm_tokenizer_ugm_session session(vocab, *static_cast<const llm_tokenizer_ugm *>(tokenizer.get()));

                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);
#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif
                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }

                if (add_special && add_bos && output.size() >= 2 && output[1] == special_bos_id) {
                    LLAMA_LOG_WARN(
                        "%s: Added a BOS token to the prompt as specified by the model but the prompt "
                        "also starts with a BOS token. So now the final prompt starts with 2 BOS tokens. "
                        "Are you sure this is what you want?\n", __FUNCTION__);
                }

                if (add_special && add_eos) {
                    GGML_ASSERT(special_eos_id != LLAMA_TOKEN_NULL);
                    output.push_back(special_eos_id);
                }
            } break;
        case LLAMA_VOCAB_TYPE_RWKV:
            {
                llm_tokenizer_rwkv_session session(vocab, *static_cast<const llm_tokenizer_rwkv *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_PLAMO2:
            {
                llm_tokenizer_plamo2_session session(*static_cast<const llm_tokenizer_plamo2 *>(tokenizer.get()));
                for (const auto & fragment : fragment_buffer) {
                    if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_RAW_TEXT) {
                        std::string text = fragment.raw_text.substr(fragment.offset, fragment.length);

#ifdef PRETOKENIZERDEBUG
                        LLAMA_LOG_WARN("TT: (%ld %ld %ld) '%s'\n", text.length(), fragment.offset, fragment.length, text.c_str());
#endif

                        session.tokenize(text, output);
                    } else { // if (fragment.type == FRAGMENT_BUFFER_VARIANT_TYPE_TOKEN)
                        output.push_back(fragment.token);
                    }
                }
            } break;
        case LLAMA_VOCAB_TYPE_NONE:
            GGML_ABORT("fatal error");
    }

    return output;
}

int32_t llama_vocab::impl::token_to_piece(llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    // ref: https://github.com/ggerganov/llama.cpp/pull/7587#discussion_r1620983843
    static const int attr_special = LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_CONTROL;
    const llama_token_attr attr = token_get_attr(token);
    if (!special && (attr & attr_special)) {
        return 0;
    }

    // copy piece chars to output text buffer
    // skip up to 'lstrip' leading spaces before copying
    auto _try_copy = [=] (const char * token, size_t size) -> int32_t {
        if (size >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
            GGML_ABORT("invalid token size: %zu exceeds int32_t limit", size);
        }

        for (int32_t i = 0; i < lstrip && size && *token == ' '; ++i) {
            token++;
            size--;
        }
        if (length < (int32_t)size) {
            return -(int32_t) size;
        }
        memcpy(buf, token, size);
        return (int32_t) size;
    };

    // if we have a cache - use it
    {
        const auto & cache = cache_token_to_piece;

        if (!cache.empty()) {
            const auto & result = cache.at(token);
            return _try_copy(result.data(), result.size());
        }
    }

    if (0 <= token && token < (int32_t) id_to_token.size()) {
        const std::string & token_text = id_to_token[token].text;
        switch (get_type()) {
            case LLAMA_VOCAB_TYPE_WPM:
            case LLAMA_VOCAB_TYPE_SPM:
            case LLAMA_VOCAB_TYPE_UGM: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = token_text;
                    llama_unescape_whitespace(result);
                    return _try_copy(result.data(), result.size());
                }
                if (attr & LLAMA_TOKEN_ATTR_BYTE) {
                    char byte = (char) token_to_byte(token);
                    return _try_copy((char*) &byte, 1);
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_BPE: {
                // NOTE: we accept all unsupported token types,
                // suppressing them like CONTROL tokens.
                if (attr & (attr_special | LLAMA_TOKEN_ATTR_USER_DEFINED)) {
                    return _try_copy(token_text.data(), token_text.size());
                }
                if (attr & LLAMA_TOKEN_ATTR_NORMAL) {
                    std::string result = llama_decode_text(token_text);
                    return _try_copy(result.data(), result.size());
                }
                break;
            }
            case LLAMA_VOCAB_TYPE_RWKV: {
                std::vector<uint8_t> result = llama_unescape_rwkv_token(token_text);

                // If we don't have enough space, return an error
                if (result.size() > (size_t)length) {
                    return -(int)result.size();
                }

                memcpy(buf, result.data(), result.size());
                return (int)result.size();
            }
            case LLAMA_VOCAB_TYPE_PLAMO2: {
                // PLaMo-2 uses similar token handling as BPE/SPM
                if (vocab.is_byte(token)) {
                    // Handle byte tokens like <0xXX>
                    if (token_text.length() == 6 && token_text.substr(0, 3) == "<0x" && token_text.back() == '>') {
                        int hex_val = std::stoi(token_text.substr(3, 2), nullptr, 16);
                        if (length < 1) {
                            return -1;
                        }
                        buf[0] = static_cast<char>(hex_val);
                        return 1;
                    }
                }

                // Normal token - just copy the text
                std::string result = token_text;
                return _try_copy(result.data(), result.size());
            }
            default:
                GGML_ABORT("fatal error");
        }
    }

    return 0;
}

const std::string & llama_vocab::impl::token_to_piece(llama_token token) const {
    return cache_token_to_piece.at(token);
}

int32_t llama_vocab::impl::detokenize(
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    if (type == LLAMA_VOCAB_TYPE_NONE) {
        return 0;
    }

    GGML_ASSERT(tokenizer && "Tokenizer not initialized. Call llama_vocab::init_tokenizer() first.");

    int32_t avail = text_len_max;
    int32_t total = 0;

    // remove the leading space
    bool remove_space = add_space_prefix;

    if (remove_special && add_bos) {
        if (n_tokens > 0 && tokens[0] == special_bos_id) {
            remove_space = false;
            n_tokens--;
            tokens++;
        }
    }

    if (remove_special && add_eos) {
        if (n_tokens > 0 && tokens[n_tokens - 1] == special_eos_id) {
            n_tokens--;
        }
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        GGML_ASSERT(avail >= 0);
        int32_t n_chars = token_to_piece(tokens[i], text, avail, remove_space, unparse_special);
        remove_space = false;
        if (n_chars < 0) {
            avail = 0;
            total -= n_chars;
        } else if (n_chars > 0) {
            avail -= n_chars;
            text  += n_chars;
            total += n_chars;
        }
    }

    if (total > text_len_max) {
        return -total;
    }

    if (clean_spaces) {
        text -= total;  // restart text

        // first pass: characters ?!.,  //TODO: where do these characters come from?
        const int32_t total1 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total1; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '?' || x == '!' || x == '.' || x == ',') {  // " ?", " !", " .", " ,"
                    total--;  // remove space
                }
            }
            text[total++] = x;
        }

        // second pass: strip single apostrophe between spaces
        const int32_t total2 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total2; ++i) {
            const char x = text[i];
            if (x == '\'' && i + 1 < total2 && text[i - 1] == ' ' && text[i + 1] == ' ') {  // " ' "
                total--;           // remove prev space
                text[++i] = '\0';  // remove next space
            }
            text[total++] = x;
        }

        // third pass: apostrophe contractions  //NOTE: this makes sense?
        const int32_t total3 = total;
        total = total ? 1 : 0;
        for (int32_t i = 1; i < total3; ++i) {
            const char x = text[i];
            if (text[i - 1] == ' ') {
                if (x == '\'' && i + 1 < total3) {
                    const char x1 = text[i + 1];
                    if (x1 == 't' || x1 == 'd') {  // " 't", " 'd"
                        //total--;  // remove space
                    } else if (x1 == 's' || x1 == 'm') {  // " 's", " 'm"
                        total--;  // remove space
                    } else if (i + 2 < total3) {
                        const char x2 = text[i + 2];
                        if ((x1 == 'l' && x2 == 'l')) {  // " 'll"
                            //total--;  // remove space
                        } else if ((x1 == 'r' && x2 == 'e') || (x1 == 'v' && x2 == 'e')) {  // " 're", " 've"
                            total--;  // remove space
                        } else {
                            //total--;  // remove space
                        }
                    } else {
                        //total--;  // remove space
                    }
                }
            }
            text[total++] = x;
        }
    }

    return total <= text_len_max ? total : -total;
}

void llama_vocab::impl::print_info() const {
    LLAMA_LOG_INFO("%s: vocab type       = %s\n",     __func__, type_name().c_str());
    LLAMA_LOG_INFO("%s: n_vocab          = %u\n",     __func__, vocab.n_tokens());
    LLAMA_LOG_INFO("%s: n_merges         = %u\n",     __func__, (uint32_t) bpe_ranks.size());

    // special tokens
    if (special_bos_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: BOS token        = %d '%s'\n", __func__, special_bos_id,     id_to_token.at(special_bos_id).text.c_str() );  }
    if (special_eos_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOS token        = %d '%s'\n", __func__, special_eos_id,     id_to_token.at(special_eos_id).text.c_str() );  }
    if (special_eot_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOT token        = %d '%s'\n", __func__, special_eot_id,     id_to_token.at(special_eot_id).text.c_str() );  }
    if (special_eom_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: EOM token        = %d '%s'\n", __func__, special_eom_id,     id_to_token.at(special_eom_id).text.c_str() );  }
    if (special_unk_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: UNK token        = %d '%s'\n", __func__, special_unk_id,     id_to_token.at(special_unk_id).text.c_str() );  }
    if (special_sep_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: SEP token        = %d '%s'\n", __func__, special_sep_id,     id_to_token.at(special_sep_id).text.c_str() );  }
    if (special_pad_id  != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: PAD token        = %d '%s'\n", __func__, special_pad_id,     id_to_token.at(special_pad_id).text.c_str() );  }
    if (special_mask_id != LLAMA_TOKEN_NULL)    { LLAMA_LOG_INFO( "%s: MASK token       = %d '%s'\n", __func__, special_mask_id,    id_to_token.at(special_mask_id).text.c_str() ); }

    if (linefeed_id != LLAMA_TOKEN_NULL)        { LLAMA_LOG_INFO( "%s: LF token         = %d '%s'\n", __func__, linefeed_id,        id_to_token.at(linefeed_id).text.c_str() ); }

    if (special_fim_pre_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM PRE token    = %d '%s'\n", __func__, special_fim_pre_id, id_to_token.at(special_fim_pre_id).text.c_str() ); }
    if (special_fim_suf_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM SUF token    = %d '%s'\n", __func__, special_fim_suf_id, id_to_token.at(special_fim_suf_id).text.c_str() ); }
    if (special_fim_mid_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM MID token    = %d '%s'\n", __func__, special_fim_mid_id, id_to_token.at(special_fim_mid_id).text.c_str() ); }
    if (special_fim_pad_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM PAD token    = %d '%s'\n", __func__, special_fim_pad_id, id_to_token.at(special_fim_pad_id).text.c_str() ); }
    if (special_fim_rep_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM REP token    = %d '%s'\n", __func__, special_fim_rep_id, id_to_token.at(special_fim_rep_id).text.c_str() ); }
    if (special_fim_sep_id != LLAMA_TOKEN_NULL) { LLAMA_LOG_INFO( "%s: FIM SEP token    = %d '%s'\n", __func__, special_fim_sep_id, id_to_token.at(special_fim_sep_id).text.c_str() ); }

    for (const auto & id : special_eog_ids) {
        LLAMA_LOG_INFO( "%s: EOG token        = %d '%s'\n", __func__, id, id_to_token.at(id).text.c_str() );
    }

    LLAMA_LOG_INFO("%s: max token length = %d\n", __func__, max_token_len);
}

llama_vocab::llama_vocab() : pimpl(new impl(*this)) {
}

llama_vocab::~llama_vocab() = default;

void llama_vocab::load(llama_model_loader & ml, const LLM_KV & kv) {
    pimpl->load(ml, kv);
}

std::string llama_vocab::get_tokenizer_model() const {
    return pimpl->tokenizer_model;
}

std::string llama_vocab::get_tokenizer_pre() const {
    return pimpl->tokenizer_pre;
}

enum llama_vocab_type llama_vocab::get_type() const {
    return pimpl->type;
}

enum llama_vocab_pre_type llama_vocab::get_pre_type() const {
    return pimpl->pre_type;
}

uint32_t llama_vocab::n_tokens() const {
    return (uint32_t) pimpl->id_to_token.size();
}

uint32_t llama_vocab::n_token_types() const {
    return (uint32_t) pimpl->n_token_types;
}

std::string llama_vocab::type_name() const{
    return pimpl->type_name();
}

bool llama_vocab::is_normal(llama_token id) const {
    return pimpl->is_normal(id);
}

bool llama_vocab::is_unknown(llama_token id) const {
    return pimpl->is_unknown(id);
}

bool llama_vocab::is_control(llama_token id) const {
    return pimpl->is_control(id);
}

bool llama_vocab::is_byte(llama_token id) const {
    return pimpl->is_byte(id);
}

bool llama_vocab::is_user_defined(llama_token id) const {
    return pimpl->is_user_defined(id);
}

bool llama_vocab::is_unused(llama_token id) const {
    return pimpl->is_unused(id);
}

bool llama_vocab::is_eog(llama_token id) const {
    return pimpl->is_eog(id);
}

uint8_t llama_vocab::token_to_byte(llama_token id) const {
    return pimpl->token_to_byte(id);
}

llama_token llama_vocab::byte_to_token(uint8_t ch) const {
    GGML_ASSERT(get_type() != LLAMA_VOCAB_TYPE_NONE);
    static const char * hex = "0123456789ABCDEF";
    switch (get_type()) {
        case LLAMA_VOCAB_TYPE_SPM:
        case LLAMA_VOCAB_TYPE_UGM: {
            const char buf[7] = { '<', '0', 'x', hex[ch >> 4], hex[ch & 15], '>', 0 };
            auto token = pimpl->token_to_id.find(buf);
            if (token != pimpl->token_to_id.end()) {
                return (*token).second;
            }
            // Try to fall back to just the byte as a string
            const char buf2[2] = { (char)ch, 0 };
            return pimpl->token_to_id.at(buf2);
        }
        case LLAMA_VOCAB_TYPE_WPM:
        case LLAMA_VOCAB_TYPE_BPE: {
            return pimpl->token_to_id.at(unicode_byte_to_utf8(ch));
        }
        case LLAMA_VOCAB_TYPE_PLAMO2: {
            // PLaMo-2 uses byte tokens in format <0xXX>
            char hex_str[8];
            snprintf(hex_str, sizeof(hex_str), "<0x%02X>", ch);
            return pimpl->token_to_id.at(hex_str);
        }
        default:
            GGML_ABORT("fatal error");
    }
}

llama_token llama_vocab::text_to_token(const std::string & text) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    auto it = pimpl->token_to_id.find(text);
    if (it != pimpl->token_to_id.end()) {
        return (*it).second;
    }
    return LLAMA_TOKEN_NULL;
}

const llama_vocab::token_data & llama_vocab::get_token_data(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id);
}

const char * llama_vocab::token_get_text(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id).text.c_str();
}

float llama_vocab::token_get_score(llama_token id) const {
    GGML_ASSERT(pimpl->type != LLAMA_VOCAB_TYPE_NONE);
    return pimpl->id_to_token.at(id).score;
}

llama_token_attr llama_vocab::token_get_attr(llama_token id) const {
    return pimpl->token_get_attr(id);
}

llama_token llama_vocab::token_bos() const {
    return pimpl->special_bos_id;
}

llama_token llama_vocab::token_eos() const {
    return pimpl->special_eos_id;
}

llama_token llama_vocab::token_eot() const {
    return pimpl->special_eot_id;
}

llama_token llama_vocab::token_eom() const {
    return pimpl->special_eom_id;
}

llama_token llama_vocab::token_unk() const {
    return pimpl->special_unk_id;
}

llama_token llama_vocab::token_sep() const {
    return pimpl->special_sep_id;
}

llama_token llama_vocab::token_nl() const {
    return pimpl->linefeed_id;
}

llama_token llama_vocab::token_pad() const {
    return pimpl->special_pad_id;
}

llama_token llama_vocab::token_prefix() const {
    return pimpl->special_fim_pre_id;
}

llama_token llama_vocab::token_middle() const {
    return pimpl->special_fim_mid_id;
}

llama_token llama_vocab::token_suffix() const {
    return pimpl->special_fim_suf_id;
}

llama_token llama_vocab::token_fim_pre() const {
    return pimpl->special_fim_pre_id;
}

llama_token llama_vocab::token_fim_suf() const {
    return pimpl->special_fim_suf_id;
}

llama_token llama_vocab::token_fim_mid() const {
    return pimpl->special_fim_mid_id;
}

llama_token llama_vocab::token_fim_pad() const {
    return pimpl->special_fim_pad_id;
}

llama_token llama_vocab::token_fim_rep() const {
    return pimpl->special_fim_rep_id;
}

llama_token llama_vocab::token_fim_sep() const {
    return pimpl->special_fim_sep_id;
}

llama_token llama_vocab::token_mask() const {
    return pimpl->special_mask_id;
}

bool llama_vocab::get_add_space_prefix() const {
    return pimpl->add_space_prefix;
}

bool llama_vocab::get_add_bos() const {
    return pimpl->add_bos;
}

bool llama_vocab::get_add_eos() const {
    return pimpl->add_eos;
}

bool llama_vocab::get_add_sep() const {
    return pimpl->add_sep;
}

bool llama_vocab::get_ignore_merges() const {
    return pimpl->ignore_merges;
}

bool llama_vocab::get_clean_spaces() const {
    return pimpl->clean_spaces;
}

bool llama_vocab::get_remove_extra_whitespaces() const {
    return pimpl->remove_extra_whitespaces;
}

bool llama_vocab::get_escape_whitespaces() const {
    return pimpl->escape_whitespaces;
}

bool llama_vocab::get_treat_whitespace_as_suffix() const {
    return pimpl->treat_whitespace_as_suffix;
}

int llama_vocab::max_token_len() const {
    return pimpl->max_token_len;
}

int llama_vocab::find_bpe_rank(const std::string & token_left, const std::string & token_right) const {
    GGML_ASSERT(token_left.find(' ')   == std::string::npos);
    GGML_ASSERT(token_left.find('\n')  == std::string::npos);
    GGML_ASSERT(token_right.find(' ')  == std::string::npos);
    GGML_ASSERT(token_right.find('\n') == std::string::npos);

    auto it = pimpl->bpe_ranks.find(std::make_pair(token_left, token_right));
    if (it == pimpl->bpe_ranks.end()) {
        return -1;
    }

    return it->second;
}

std::vector<std::string> llama_vocab::get_bpe_merges() const {
    std::vector<std::string> result(pimpl->bpe_ranks.size());

    for (const auto & pair : pimpl->bpe_ranks) {
        result[pair.second] = pair.first.first + " " + pair.first.second;
    }

    return result;
}

std::vector<char> llama_vocab::get_precompiled_charsmap() const {
    return pimpl->precompiled_charsmap;
}

int32_t llama_vocab::tokenize(
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) const {
    auto res = tokenize(std::string(text, text_len), add_special, parse_special);
    if (res.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LLAMA_LOG_ERROR("%s: tokenization result size %zu exceeds int32_t limit\n", __func__, res.size());
        return std::numeric_limits<int32_t>::min();
    }

    if (n_tokens_max < (int) res.size()) {
        // LLAMA_LOG_ERROR("%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

std::vector<llama_token> llama_vocab::tokenize(
        const std::string & raw_text,
        bool add_special,
        bool parse_special) const {
    return pimpl->tokenize(raw_text, add_special, parse_special);
}

const std::string & llama_vocab::token_to_piece(llama_token token) const {
    return pimpl->token_to_piece(token);
}

int32_t llama_vocab::token_to_piece(llama_token token, char * buf, int32_t length, int32_t lstrip, bool special) const {
    return pimpl->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llama_vocab::detokenize(
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special) const {
    return pimpl->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

std::string llama_vocab::detokenize(const std::vector<llama_token> & tokens, bool special) const {
    std::string text;
    text.resize(std::max(text.capacity(), tokens.size()));
    int32_t n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
    if (n_chars < 0) {
        text.resize(-n_chars);
        n_chars = detokenize(tokens.data(), (int32_t)tokens.size(), &text[0], (int32_t)text.size(), false, special);
        GGML_ASSERT(n_chars <= (int32_t)text.size());  // whitespace trimming is performed after per-token detokenization
    }

    text.resize(n_chars);

    // NOTE: the original tokenizer decodes bytes after collecting the pieces.
    return text;
}

void llama_vocab::print_info() const {
    pimpl->print_info();
}

//
// interface implementation
//

int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab) {
    return vocab->n_tokens();
}

// deprecated
int32_t llama_n_vocab(const struct llama_vocab * vocab) {
    return llama_vocab_n_tokens(vocab);
}

enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab) {
    return vocab->get_type();
}

const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_text(token);
}

float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_score(token);
}

enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token) {
    return vocab->token_get_attr(token);
}

bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token) {
    return vocab->is_eog(token);
}

bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token) {
    return vocab->is_control(token);
}

llama_token llama_vocab_bos(const struct llama_vocab * vocab) {
    return vocab->token_bos();
}

llama_token llama_vocab_eos(const struct llama_vocab * vocab) {
    return vocab->token_eos();
}

llama_token llama_vocab_eot(const struct llama_vocab * vocab) {
    return vocab->token_eot();
}

// deprecated
llama_token llama_vocab_cls(const struct llama_vocab * vocab) {
    return vocab->token_bos();
}

llama_token llama_vocab_sep(const struct llama_vocab * vocab) {
    return vocab->token_sep();
}

llama_token llama_vocab_nl (const struct llama_vocab * vocab) {
    return vocab->token_nl();
}

llama_token llama_vocab_pad(const struct llama_vocab * vocab) {
    return vocab->token_pad();
}

bool llama_vocab_get_add_bos(const struct llama_vocab * vocab) {
    return vocab->get_add_bos();
}

bool llama_vocab_get_add_eos(const struct llama_vocab * vocab) {
    return vocab->get_add_eos();
}

bool llama_vocab_get_add_sep(const struct llama_vocab * vocab) {
    return vocab->get_add_sep();
}

llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab) {
    return vocab->token_fim_pre();
}

llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab) {
    return vocab->token_fim_suf();
}

llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab) {
    return vocab->token_fim_mid();
}

llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab) {
    return vocab->token_fim_pad();
}

llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab) {
    return vocab->token_fim_rep();
}

llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab) {
    return vocab->token_fim_sep();
}

llama_token llama_vocab_mask(const struct llama_vocab* vocab) {
    return vocab->token_mask();
}

// deprecated
const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_text(vocab, token);
}

// deprecated
float llama_token_get_score(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_score(vocab, token);
}

// deprecated
enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_get_attr(vocab, token);
}

// deprecated
bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_is_eog(vocab, token);
}

// deprecated
bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token) {
    return llama_vocab_is_control(vocab, token);
}

// deprecated
llama_token llama_token_bos(const struct llama_vocab * vocab) {
    return llama_vocab_bos(vocab);
}

// deprecated
llama_token llama_token_eos(const struct llama_vocab * vocab) {
    return llama_vocab_eos(vocab);
}

// deprecated
llama_token llama_token_eot(const struct llama_vocab * vocab) {
    return llama_vocab_eot(vocab);
}

// deprecated
llama_token llama_token_cls(const struct llama_vocab * vocab) {
    //return llama_vocab_cls(vocab);
    return llama_vocab_bos(vocab); // avoid deprecation warning
}

// deprecated
llama_token llama_token_sep(const struct llama_vocab * vocab) {
    return llama_vocab_sep(vocab);
}

// deprecated
llama_token llama_token_nl (const struct llama_vocab * vocab) {
    return llama_vocab_nl(vocab);
}

// deprecated
llama_token llama_token_pad(const struct llama_vocab * vocab) {
    return llama_vocab_pad(vocab);
}

// deprecated
bool llama_add_bos_token(const struct llama_vocab * vocab) {
    return llama_vocab_get_add_bos(vocab);
}

// deprecated
bool llama_add_eos_token(const struct llama_vocab * vocab) {
    return llama_vocab_get_add_eos(vocab);
}

// deprecated
llama_token llama_token_fim_pre(const struct llama_vocab * vocab) {
    return llama_vocab_fim_pre(vocab);
}

// deprecated
llama_token llama_token_fim_suf(const struct llama_vocab * vocab) {
    return llama_vocab_fim_suf(vocab);
}

// deprecated
llama_token llama_token_fim_mid(const struct llama_vocab * vocab) {
    return llama_vocab_fim_mid(vocab);
}

// deprecated
llama_token llama_token_fim_pad(const struct llama_vocab * vocab) {
    return llama_vocab_fim_pad(vocab);
}

// deprecated
llama_token llama_token_fim_rep(const struct llama_vocab * vocab) {
    return llama_vocab_fim_rep(vocab);
}

// deprecated
llama_token llama_token_fim_sep(const struct llama_vocab * vocab) {
    return llama_vocab_fim_sep(vocab);
}

//
// tokenization
//

int32_t llama_tokenize(
    const struct llama_vocab * vocab,
                  const char * text,
                     int32_t   text_len,
                 llama_token * tokens,
                     int32_t   n_tokens_max,
                        bool   add_special,
                        bool   parse_special) {
    return vocab->tokenize(text, text_len, tokens, n_tokens_max, add_special, parse_special);
}

int32_t llama_token_to_piece(
    const struct llama_vocab * vocab,
                 llama_token   token,
                        char * buf,
                     int32_t   length,
                     int32_t   lstrip,
                        bool   special) {
    return vocab->token_to_piece(token, buf, length, lstrip, special);
}

int32_t llama_detokenize(
    const struct llama_vocab * vocab,
           const llama_token * tokens,
                     int32_t   n_tokens,
                        char * text,
                     int32_t   text_len_max,
                        bool   remove_special,
                        bool   unparse_special) {
    return vocab->detokenize(tokens, n_tokens, text, text_len_max, remove_special, unparse_special);
}

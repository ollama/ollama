#pragma once

#include "llama.h"

#include <map>
#include <string>
#include <vector>

struct ollama_vocab {
    std::map<std::string, uint32_t> symbol_ids;
    std::map<uint32_t, std::string> token_to_piece;
    uint32_t eog_token;

    void add_symbol_id(const std::string & symbol, uint32_t id);
    void add_token_piece(uint32_t token, const std::string & piece);
    void set_eog_token(uint32_t token);
};

// grammar element type
enum gretype {
    // end of rule definition
    GRETYPE_END            = 0,

    // start of alternate definition for rule
    GRETYPE_ALT            = 1,

    // non-terminal element: reference to rule
    GRETYPE_RULE_REF       = 2,

    // terminal element: character (code point)
    GRETYPE_CHAR           = 3,

    // inverse char(s) ([^a], [^a-b] [^abc])
    GRETYPE_CHAR_NOT       = 4,

    // modifies a preceding GRETYPE_CHAR or GRETYPE_CHAR_ALT to
    // be an inclusive range ([a-z])
    GRETYPE_CHAR_RNG_UPPER = 5,

    // modifies a preceding GRETYPE_CHAR or
    // GRETYPE_CHAR_RNG_UPPER to add an alternate char to match ([ab], [a-zA])
    GRETYPE_CHAR_ALT       = 6,

    // any character (.)
    GRETYPE_CHAR_ANY       = 7,
};

typedef struct grammar_element {
    enum gretype type;
    uint32_t     value; // Unicode code point or rule ID
} grammar_element;

struct partial_utf8 {
    uint32_t value;    // bit value so far (unshifted)
    int      n_remain; // num bytes remaining; -1 indicates invalid sequence
};

struct grammar_candidate {
    size_t           index;
    const uint32_t * code_points;
    partial_utf8     partial_utf8;
};

using grammar_rule  = std::vector<      grammar_element>;
using grammar_stack = std::vector<const grammar_element *>;

using grammar_rules      = std::vector<grammar_rule>;
using grammar_stacks     = std::vector<grammar_stack>;
using grammar_candidates = std::vector<grammar_candidate>;

// TODO: remove, needed for tests atm
const grammar_rules  & grammar_get_rules (const struct grammar * grammar);
      grammar_stacks & grammar_get_stacks(      struct grammar * grammar);

// takes a set of possible pushdown stacks on a grammar, which are required to
// be positioned at a character range (see `grammar_advance_stack`), and
// produces the N possible stacks if the given char is accepted at those
// positions
void grammar_accept(struct grammar * grammar, uint32_t chr);

std::vector<grammar_candidate> grammar_reject_candidates_for_stack(
        const grammar_rules      & rules,
        const grammar_stack      & stack,
        const grammar_candidates & candidates);

struct grammar_parser {
    std::map<std::string, uint32_t> symbol_ids;

    grammar_rules rules;

    grammar_stack c_rules() const;

    uint32_t get_symbol_id(const char * src, size_t len);
    uint32_t generate_symbol_id(const std::string & base_name);

    void add_rule(uint32_t rule_id, const grammar_rule & rule);

    const char * parse_alternates(
            const char        * src,
            const std::string & rule_name,
            uint32_t            rule_id,
            bool                is_nested);

    const char * parse_sequence(
            const char         * src,
            const std::string  & rule_name,
            grammar_rule       & rule,
            bool               is_nested);

    const char * parse_rule(const char * src);

    bool parse(const char * src);
    void print(FILE * file);
};

struct grammar {
    // note: allow null vocab for testing (not great)
    ollama_vocab * vocab;
    const grammar_rules  rules;  // TODO: shared ptr
          grammar_stacks stacks;

    // buffer for partially generated UTF-8 sequence from accepted tokens
    partial_utf8 partial_utf8;

    // lazy grammars wait for trigger words or tokens before constraining the sampling.
    // we still have trigger_tokens for non-lazy grammars to force printing of special trigger tokens.
    // (useful e.g. for tool_choice=required)
    bool                     lazy             = false;
    bool                     awaiting_trigger = false; // Initialized to true for lazy grammars only
    std::string              trigger_buffer;           // Output buffered by lazy grammar. Will be cleared once trigger is found.
    std::vector<llama_token> trigger_tokens;           // Tokens that trigger a lazy grammar, or tokens to force printing of (even if special).
    std::vector<std::string> trigger_words;
};

//
// internal API
//

// note: needed for tests (not great)
struct grammar * grammar_init_impl(
        struct ollama_vocab * ollama_vocab,
        const grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index);

struct grammar * grammar_init_impl(
        struct ollama_vocab * ollama_vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);

void grammar_free_impl(struct grammar * grammar);

struct grammar * grammar_clone_impl(const struct grammar & grammar);

// TODO: move the API below as member functions of grammar
void grammar_apply_impl(
        const struct grammar & grammar,
            llama_token_data_array * cur_p);

void grammar_accept_impl(
              struct grammar & grammar,
                       llama_token   token);

void grammar_accept_str(
              struct grammar & grammar,
                 const std::string & piece);

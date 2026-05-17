#include "llama-grammar.h"

#include "llama-impl.h"
#include "llama-vocab.h"
#include "llama-sampling.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#define MAX_REPETITION_THRESHOLD 2000
//
// helpers
//

// NOTE: assumes valid utf8 (but checks for overrun)
static std::pair<uint32_t, const char *> decode_utf8(const char * src) {
    static const int lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t  first_byte = static_cast<uint8_t>(*src);
    uint8_t  highbits   = first_byte >> 4;
    int      len        = lookup[highbits];
    uint8_t  mask       = (1 << (8 - len)) - 1;
    uint32_t value      = first_byte & mask;
    const char * end    = src + len; // may overrun!
    const char * pos    = src + 1;
    for ( ; pos < end && *pos; pos++) {
        value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
    }
    return std::make_pair(value, pos);
}

static std::pair<std::vector<uint32_t>, llama_partial_utf8> decode_utf8(
        const std::string & src,
        llama_partial_utf8 partial_start) {
    static const int      lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4 };
    const char          * pos      = src.c_str();
    std::vector<uint32_t> code_points;

    // common english strings have the same number of codepoints and bytes. `+ 1` for the terminating 0.
    code_points.reserve(src.size() + 1);
    uint32_t value    = partial_start.value;
    int      n_remain = partial_start.n_remain;

    // continue previous decode, if applicable
    while (*pos != 0 && n_remain > 0) {
        uint8_t next_byte = static_cast<uint8_t>(*pos);
        if ((next_byte >> 6) != 2) {
            // invalid sequence, abort
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, -1 });
        }
        value = (value << 6) + (next_byte & 0x3F);
        ++pos;
        --n_remain;
    }

    if (partial_start.n_remain > 0 && n_remain == 0) {
        code_points.push_back(value);
    }

    // decode any subsequent utf-8 sequences, which may end in an incomplete one
    while (*pos != 0) {
        uint8_t first_byte = static_cast<uint8_t>(*pos);
        uint8_t highbits   = first_byte >> 4;
        n_remain   = lookup[highbits] - 1;

        if (n_remain < 0) {
            // invalid sequence, abort
            code_points.clear();
            code_points.push_back(0);
            return std::make_pair(std::move(code_points), llama_partial_utf8{ 0, n_remain });
        }

        uint8_t mask  = (1 << (7 - n_remain)) - 1;
        value = first_byte & mask;

        ++pos;
        while (*pos != 0 && n_remain > 0) {
            value = (value << 6) + (static_cast<uint8_t>(*pos) & 0x3F);
            ++pos;
            --n_remain;
        }
        if (n_remain == 0) {
            code_points.push_back(value);
        }
    }
    code_points.push_back(0);

    return std::make_pair(std::move(code_points), llama_partial_utf8{ value, n_remain });
}

static bool is_digit_char(char c) {
    return '0' <= c && c <= '9';
}

static bool is_word_char(char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '-' || is_digit_char(c);
}

static std::pair<uint32_t, const char *> parse_hex(const char * src, int size) {
    const char * pos   = src;
    const char * end   = src + size;
    uint32_t     value = 0;
    for ( ; pos < end && *pos; pos++) {
        value <<= 4;
        char c = *pos;
        if ('a' <= c && c <= 'f') {
            value += c - 'a' + 10;
        } else if ('A' <= c && c <= 'F') {
            value += c - 'A' + 10;
        } else if ('0' <= c && c <= '9') {
            value += c - '0';
        } else {
            break;
        }
    }
    if (pos != end) {
        throw std::runtime_error("expecting " + std::to_string(size) + " hex chars at " + src);
    }
    return std::make_pair(value, pos);
}

static const char * parse_space(const char * src, bool newline_ok) {
    const char * pos = src;
    while (*pos == ' ' || *pos == '\t' || *pos == '#' ||
            (newline_ok && (*pos == '\r' || *pos == '\n'))) {
        if (*pos == '#') {
            while (*pos && *pos != '\r' && *pos != '\n') {
                pos++;
            }
        } else {
            pos++;
        }
    }
    return pos;
}

static const char * parse_name(const char * src) {
    const char * pos = src;
    while (is_word_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting name at ") + src);
    }
    return pos;
}

static const char * parse_int(const char * src) {
    const char * pos = src;
    while (is_digit_char(*pos)) {
        pos++;
    }
    if (pos == src) {
        throw std::runtime_error(std::string("expecting integer at ") + src);
    }
    return pos;
}

static std::pair<uint32_t, const char *> parse_char(const char * src) {
    if (*src == '\\') {
        switch (src[1]) {
            case 'x': return parse_hex(src + 2, 2);
            case 'u': return parse_hex(src + 2, 4);
            case 'U': return parse_hex(src + 2, 8);
            case 't': return std::make_pair('\t', src + 2);
            case 'r': return std::make_pair('\r', src + 2);
            case 'n': return std::make_pair('\n', src + 2);
            case '\\':
            case '"':
            case '[':
            case ']':
                      return std::make_pair(src[1], src + 2);
            default:
                      throw std::runtime_error(std::string("unknown escape at ") + src);
        }
    } else if (*src) {
        return decode_utf8(src);
    }
    throw std::runtime_error("unexpected end of input");
}

static std::pair<uint32_t, const char *> parse_token(const llama_vocab * vocab, const char * src) {
    const char * pos = src;
    if (*pos != '<') {
        throw std::runtime_error(std::string("expecting '<' at ") + pos);
    }
    pos++;

    // Parse <[id]>
    if (*pos == '[') {
        pos++;
        const char * int_end = parse_int(pos);
        uint32_t token_id = std::stoul(std::string(pos, int_end - pos));
        pos = int_end;
        if (*pos != ']') {
            throw std::runtime_error(std::string("expecting ']' at ") + pos);
        }
        pos++;
        if (*pos != '>') {
            throw std::runtime_error(std::string("expecting '>' at ") + pos);
        }
        pos++;
        return std::make_pair(token_id, pos);
    }

    if (vocab == nullptr) {
        throw std::runtime_error(std::string("no vocab to parse token at ") + src);
    }

    // Parse <token> and tokenize to obtain the token id
    while (*pos != 0 && *pos != '>') {
        pos++;
    }
    if (*pos != '>') {
        throw std::runtime_error(std::string("expecting '>' at ") + pos);
    }
    pos++;

    llama_token tokens[2];
    int32_t n_tokens = vocab->tokenize(src, static_cast<int32_t>(pos - src), tokens, 2, false, true);
    if (n_tokens != 1) {
        // must tokenize to exactly 1 token
        throw std::runtime_error("invalid token '" + std::string(src, pos - src) + "'");
    }
    return std::make_pair(tokens[0], pos);
}

static void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        // cop out of encoding UTF-8
        fprintf(file, "<U+%04X>", c);
    }
}

static bool is_char_element(llama_grammar_element elem) {
    switch (elem.type) {
        case LLAMA_GRETYPE_CHAR:           return true;
        case LLAMA_GRETYPE_CHAR_NOT:       return true;
        case LLAMA_GRETYPE_CHAR_ALT:       return true;
        case LLAMA_GRETYPE_CHAR_RNG_UPPER: return true;
        case LLAMA_GRETYPE_CHAR_ANY:       return true;
        default:                           return false;
    }
}

static void print_rule_binary(FILE * file, const llama_grammar_rule & rule) {
    for (auto elem : rule) {
        switch (elem.type) {
            case LLAMA_GRETYPE_END:            fprintf(file, "END");            break;
            case LLAMA_GRETYPE_ALT:            fprintf(file, "ALT");            break;
            case LLAMA_GRETYPE_RULE_REF:       fprintf(file, "RULE_REF");       break;
            case LLAMA_GRETYPE_CHAR:           fprintf(file, "CHAR");           break;
            case LLAMA_GRETYPE_CHAR_NOT:       fprintf(file, "CHAR_NOT");       break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER: fprintf(file, "CHAR_RNG_UPPER"); break;
            case LLAMA_GRETYPE_CHAR_ALT:       fprintf(file, "CHAR_ALT");       break;
            case LLAMA_GRETYPE_CHAR_ANY:       fprintf(file, "CHAR_ANY");       break;
            case LLAMA_GRETYPE_TOKEN:          fprintf(file, "TOKEN");          break;
            case LLAMA_GRETYPE_TOKEN_NOT:      fprintf(file, "TOKEN_NOT");      break;
        }
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
            case LLAMA_GRETYPE_ALT:
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "(%u) ", elem.value);
                break;
            case LLAMA_GRETYPE_CHAR:
            case LLAMA_GRETYPE_CHAR_NOT:
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
            case LLAMA_GRETYPE_CHAR_ALT:
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, "(\"");
                print_grammar_char(file, elem.value);
                fprintf(file, "\") ");
                break;
            case LLAMA_GRETYPE_TOKEN:
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
            case LLAMA_GRETYPE_TOKEN_NOT:
                fprintf(file, "!");
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
        }
    }
    fprintf(file, "\n");
}

static void print_rule(
        FILE     * file,
        uint32_t   rule_id,
        const llama_grammar_rule & rule,
        const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != LLAMA_GRETYPE_END) {
        throw std::runtime_error(
            "malformed rule, does not end with LLAMA_GRETYPE_END: " + std::to_string(rule_id));
    }
    fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        llama_grammar_element elem = rule[i];
        switch (elem.type) {
            case LLAMA_GRETYPE_END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," +
                    std::to_string(i));
            case LLAMA_GRETYPE_ALT:
                fprintf(file, "| ");
                break;
            case LLAMA_GRETYPE_RULE_REF:
                fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case LLAMA_GRETYPE_CHAR:
                fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_NOT:
                fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "LLAMA_GRETYPE_CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case LLAMA_GRETYPE_CHAR_ANY:
                fprintf(file, ".");
                break;
            case LLAMA_GRETYPE_TOKEN:
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
            case LLAMA_GRETYPE_TOKEN_NOT:
                fprintf(file, "!");
                fprintf(file, "<[");
                fprintf(file, "%u", elem.value);
                fprintf(file, "]> ");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case LLAMA_GRETYPE_CHAR_ALT:
                case LLAMA_GRETYPE_CHAR_RNG_UPPER:
                case LLAMA_GRETYPE_CHAR_ANY:
                    break;
                default:
                    fprintf(file, "] ");
            }
        }
    }
    fprintf(file, "\n");
}

//
// implementation
//

uint32_t llama_grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t llama_grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void llama_grammar_parser::add_rule(uint32_t rule_id, const llama_grammar_rule & rule) {
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = rule;
}

const char * llama_grammar_parser::parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested) {
    llama_grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({LLAMA_GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({LLAMA_GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}

const char * llama_grammar_parser::parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        llama_grammar_rule & rule,
        bool               is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    // use UINT64_MAX as the empty value because we aligned to the proper uint64_t type so -1 can't be used
    // (though it's technically the same as -1 now)
    auto handle_repetitions = [&](uint64_t min_times, uint64_t max_times) {
        bool no_max = max_times == UINT64_MAX;
        if (last_sym_start == rule.size()) {
            throw std::runtime_error(std::string("expecting preceding item to */+/?/{ at ") + pos);
        }

        // apply transformation to previous symbol (last_sym_start to end) according to
        // the following rewrite rules:
        // S{m,n} --> S S S (m times) S'(n-m)
        //            S'(x)   ::= S S'(x-1) |
        //            (... n-m definitions of these S' rules ...)
        //            S'(1)   ::= S |
        // S{m,} -->  S S S (m times) S'
        //            S'     ::= S S' |
        // S*     --> S{0,}
        //        --> S'     ::= S S' |
        // S+     --> S{1,}
        //        --> S S'
        //            S'     ::= S S' |
        // S?     --> S{0,1}
        //        --> S'
        //            S'     ::= S |

        llama_grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // Repeat the previous elements (min_times - 1) times
            for (uint64_t i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = no_max ? 1 : max_times - min_times;

        llama_grammar_rule rec_rule(prev_rule);
        for (uint64_t i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id( rule_name);
            if (i > 0 || no_max) {
                rec_rule.push_back({LLAMA_GRETYPE_RULE_REF, no_max ? rec_rule_id : last_rec_rule_id});
            }
            rec_rule.push_back({LLAMA_GRETYPE_ALT, 0});
            rec_rule.push_back({LLAMA_GRETYPE_END, 0});
            add_rule( rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({LLAMA_GRETYPE_RULE_REF, last_rec_rule_id});
        }
    };

    while (*pos) {
        if (*pos == '"') { // literal string
            pos++;
            last_sym_start = rule.size();
            while (*pos != '"') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                rule.push_back({LLAMA_GRETYPE_CHAR, char_pair.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') { // char range(s)
            pos++;
            enum llama_gretype start_type = LLAMA_GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = LLAMA_GRETYPE_CHAR_NOT;
            }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                enum llama_gretype type = last_sym_start < rule.size()
                    ? LLAMA_GRETYPE_CHAR_ALT
                    : start_type;

                rule.push_back({type, char_pair.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto endchar_pair = parse_char(pos + 1);
                         pos          = endchar_pair.second;
                    rule.push_back({LLAMA_GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '<' || *pos == '!') { // token
            auto type = LLAMA_GRETYPE_TOKEN;
            if (*pos == '!') { // token inverse
                type = LLAMA_GRETYPE_TOKEN_NOT;
                pos++;
            }
            auto token_pair = parse_token(vocab, pos);
            const char * token_end  = token_pair.second;
            last_sym_start = rule.size();
            rule.push_back({type, token_pair.first});
            pos = parse_space(token_end, is_nested);
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end    = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({LLAMA_GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') { // grouping
            // parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            last_sym_start = rule.size();
            // output reference to synthesized rule
            rule.push_back({LLAMA_GRETYPE_RULE_REF, sub_rule_id});
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') { // any char
            last_sym_start = rule.size();
            rule.push_back({LLAMA_GRETYPE_CHAR_ANY, 0});
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '*') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, -1);
        } else if (*pos == '+') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(1, -1);
        } else if (*pos == '?') {
            pos = parse_space(pos + 1, is_nested);
            handle_repetitions(0, 1);
        } else if (*pos == '{') {
            pos = parse_space(pos + 1, is_nested);

            if (!is_digit_char(*pos)) {
                throw std::runtime_error(std::string("expecting an int at ") + pos);
            }
            const char * int_end = parse_int(pos);
            uint64_t min_times = std::stoul(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            uint64_t max_times = UINT64_MAX; // default: no max limit

            if (*pos == '}') {
                max_times = min_times;
                pos = parse_space(pos + 1, is_nested);
            } else if (*pos == ',') {
                pos = parse_space(pos + 1, is_nested);

                if (is_digit_char(*pos)) {
                    const char * int_end = parse_int(pos);
                    max_times = std::stoul(std::string(pos, int_end - pos));
                    pos = parse_space(int_end, is_nested);
                }

                if (*pos != '}') {
                    throw std::runtime_error(std::string("expecting '}' at ") + pos);
                }
                pos = parse_space(pos + 1, is_nested);
            } else {
                throw std::runtime_error(std::string("expecting ',' at ") + pos);
            }
            bool has_max = max_times != UINT64_MAX;
            if (min_times > MAX_REPETITION_THRESHOLD || (has_max && max_times > MAX_REPETITION_THRESHOLD)) {
                throw std::runtime_error(std::string("number of repetitions exceeds sane defaults, please reduce the number of repetitions"));
            }
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char * llama_grammar_parser::parse_rule(const char * src) {
    const char * name_end = parse_name(src);
    const char * pos      = parse_space(name_end, false);
    size_t       name_len = name_end - src;
    uint32_t     rule_id  = get_symbol_id(src, name_len);
    const std::string name(src, name_len);

    if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
        throw std::runtime_error(std::string("expecting ::= at ") + pos);
    }
    pos = parse_space(pos + 3, true);

    pos = parse_alternates(pos, name, rule_id, false);

    if (*pos == '\r') {
        pos += pos[1] == '\n' ? 2 : 1;
    } else if (*pos == '\n') {
        pos++;
    } else if (*pos) {
        throw std::runtime_error(std::string("expecting newline or end at ") + pos);
    }
    return parse_space(pos, true);
}

bool llama_grammar_parser::parse(const char * src) {
    try {
        const char * pos = parse_space(src, true);
        while (*pos) {
            pos = parse_rule(pos);
        }
        // Validate the state to ensure that all rules are defined
        for (const auto & rule : rules) {
            if (rule.empty()) {
                throw std::runtime_error("Undefined rule");
            }
            for (const auto & elem : rule) {
                if (elem.type == LLAMA_GRETYPE_RULE_REF) {
                    // Ensure that the rule at that location exists
                    if (elem.value >= rules.size() || rules[elem.value].empty()) {
                        // Get the name of the rule that is missing
                        for (const auto & kv : symbol_ids) {
                            if (kv.second == elem.value) {
                                throw std::runtime_error("Undefined rule identifier '" + kv.first + "'");
                            }
                        }
                    }
                }
            }
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: error parsing grammar: %s\n\n%s\n", __func__, err.what(), src);
        rules.clear();
        return false;
    }

    return true;
}

void llama_grammar_parser::print(FILE * file) {
    try {
        std::map<uint32_t, std::string> symbol_id_names;
        for (const auto & kv : symbol_ids) {
            symbol_id_names[kv.second] = kv.first;
        }
        for (size_t i = 0, end = rules.size(); i < end; i++) {
            // fprintf(file, "%zu: ", i);
            // print_rule_binary(file, rules[i]);
            print_rule(file, uint32_t(i), rules[i], symbol_id_names);
            // fprintf(file, "\n");
        }
    } catch (const std::exception & err) {
        fprintf(stderr, "\n%s: error printing grammar: %s\n", __func__, err.what());
    }
}

llama_grammar_stack llama_grammar_parser::c_rules() const {
    llama_grammar_stack ret;
    ret.reserve(rules.size());
    for (const auto & rule : rules) {
        ret.push_back(rule.data());
    }
    return ret;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool llama_grammar_is_end_of_sequence(const llama_grammar_element * pos) {
    switch (pos->type) {
        case LLAMA_GRETYPE_END: return true;  // NOLINT
        case LLAMA_GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const llama_grammar_element *> llama_grammar_match_char(
        const llama_grammar_element * pos,
        const uint32_t                chr) {
    bool found            = false;
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;

    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            // Any character matches "."
            found = true;
            pos += 1;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool llama_grammar_match_partial_char(
        const llama_grammar_element * pos,
        const llama_partial_utf8      partial_utf8) {
    bool is_positive_char = pos->type == LLAMA_GRETYPE_CHAR || pos->type == LLAMA_GRETYPE_CHAR_ANY;
    GGML_ASSERT(is_positive_char || pos->type == LLAMA_GRETYPE_CHAR_NOT);

    uint32_t partial_value = partial_utf8.value;
    int      n_remain      = partial_utf8.n_remain;

    // invalid sequence or 7-bit char split across 2 bytes (overlong)
    if (n_remain < 0 || (n_remain == 1 && partial_value < 2)) {
        return false;
    }

    // range of possible code points this partial UTF-8 sequence could complete to
    uint32_t low  = partial_value << (n_remain * 6);
    uint32_t high = low | ((1 << (n_remain * 6)) - 1);

    if (low == 0) {
        if (n_remain == 2) {
            low = 1 << 11;
        } else if (n_remain == 3) {
            low = 1 << 16;
        }
    }

    do {
        if (pos[1].type == LLAMA_GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else if (pos->type == LLAMA_GRETYPE_CHAR_ANY) {
            // Any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == LLAMA_GRETYPE_CHAR_ALT);

    return !is_positive_char;
}

// returns true iff token matches the rule at pos (regular or inverse)
// asserts that pos is pointing to a token element
static bool llama_grammar_match_token(
    const llama_grammar_element * pos,
    const llama_token             token) {
    GGML_ASSERT(pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT);
    if (pos->type == LLAMA_GRETYPE_TOKEN) {
        return pos->value == static_cast<uint32_t>(token);
    }
    if (pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        return pos->value != static_cast<uint32_t>(token);
    }
    return false;
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void llama_grammar_advance_stack(
        const llama_grammar_rules  & rules,
        const llama_grammar_stack  & stack,
              llama_grammar_stacks & new_stacks) {
    if (stack.empty()) {
        if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
            new_stacks.emplace_back(stack);
        }
        return;
    }

    const llama_grammar_element * pos = stack.back();

    switch (pos->type) {
        case LLAMA_GRETYPE_RULE_REF: {
            const size_t                  rule_id = static_cast<size_t>(pos->value);
            const llama_grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!llama_grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                llama_grammar_advance_stack(rules, new_stack, new_stacks);
                while (!llama_grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == LLAMA_GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case LLAMA_GRETYPE_CHAR:
        case LLAMA_GRETYPE_CHAR_NOT:
        case LLAMA_GRETYPE_CHAR_ANY:
        case LLAMA_GRETYPE_TOKEN:
        case LLAMA_GRETYPE_TOKEN_NOT:
            if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
                // only add the stack if it's not a duplicate of one we already have
                new_stacks.emplace_back(stack);
            }
            break;
        default:
            // end of alternate (LLAMA_GRETYPE_END, LLAMA_GRETYPE_ALT) or middle of char range
            // (LLAMA_GRETYPE_CHAR_ALT, LLAMA_GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ABORT("fatal error");
    }
}

static llama_grammar_candidates llama_grammar_reject_candidates(
        const llama_grammar_rules      & rules,
        const llama_grammar_stacks     & stacks,
        const llama_grammar_candidates & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return {};
    }

    auto rejects = llama_grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = llama_grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }

    return rejects;
}

static bool llama_grammar_detect_left_recursion(
        const llama_grammar_rules & rules,
        size_t rule_index,
        std::vector<bool> * rules_visited,
        std::vector<bool> * rules_in_progress,
        std::vector<bool> * rules_may_be_empty) {
    if ((*rules_in_progress)[rule_index]) {
        return true;
    }

    (*rules_in_progress)[rule_index] = true;

    const llama_grammar_rule & rule = rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    bool at_rule_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (llama_grammar_is_end_of_sequence(&rule[i])) {
            if (at_rule_start) {
                (*rules_may_be_empty)[rule_index] = true;
                break;
            }
            at_rule_start = true;
        } else {
            at_rule_start = false;
        }
    }

    // Second, recurse into leftmost nonterminals (or next-leftmost as long as the previous nonterminal may
    // be empty)
    bool recurse_into_nonterminal = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (rule[i].type == LLAMA_GRETYPE_RULE_REF && recurse_into_nonterminal) {
            if (llama_grammar_detect_left_recursion(rules, (size_t)rule[i].value, rules_visited, rules_in_progress, rules_may_be_empty)) {
                return true;
            }
            if (!((*rules_may_be_empty)[(size_t)rule[i].value])) {
                recurse_into_nonterminal = false;
            }
        } else if (llama_grammar_is_end_of_sequence(&rule[i])) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    (*rules_in_progress)[rule_index] = false;
    (*rules_visited)[rule_index] = true;

    return false;
}

const llama_grammar_rules & llama_grammar_get_rules(const struct llama_grammar * grammar) {
    return grammar->rules;
}

llama_grammar_stacks & llama_grammar_get_stacks(struct llama_grammar * grammar) {
    return grammar->stacks;
}

static void llama_grammar_accept_chr(
        struct llama_grammar       & grammar,
        const llama_grammar_stack  & stack,
              uint32_t               chr,
              llama_grammar_stacks & new_stacks) {
    if (stack.empty()) {
        return;
    }

    const llama_grammar_element * pos = stack.back();

    // ignore if this turns into a token
    if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        return;
    }

    auto match = llama_grammar_match_char(pos, chr);
    if (match.first) {
        llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
        if (!llama_grammar_is_end_of_sequence(match.second)) {
            new_stack.push_back(match.second);
        }
        llama_grammar_advance_stack(grammar.rules, new_stack, new_stacks);
    }
}

void llama_grammar_accept(struct llama_grammar * grammar, uint32_t chr) {
    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        llama_grammar_accept_chr(*grammar, stack, chr, stacks_new);
    }

    grammar->stacks = std::move(stacks_new);
}

llama_grammar_candidates llama_grammar_reject_candidates_for_stack(
        const llama_grammar_rules      & rules,
        const llama_grammar_stack      & stack,
        const llama_grammar_candidates & candidates) {

    llama_grammar_candidates rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.partial_utf8.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const llama_grammar_element * stack_pos = stack.back();

    // if the top of the stack is a token rule, then we only need to check the token id
    if (stack_pos->type == LLAMA_GRETYPE_TOKEN || stack_pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
        for (const auto & tok : candidates) {
            if (*tok.code_points == 0) {
                // reached the end of a token consumed by char rules, reject iff it ended
                // in a partial response
                if (tok.partial_utf8.n_remain != 0) {
                    rejects.push_back(tok);
                }
            } else if (!llama_grammar_match_token(stack_pos, tok.id)) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    llama_grammar_candidates next_candidates;
    next_candidates.reserve(candidates.size());

    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.partial_utf8.n_remain != 0 &&
                    !llama_grammar_match_partial_char(stack_pos, tok.partial_utf8)) {
                rejects.push_back(tok);
            }
        } else if (llama_grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.partial_utf8, tok.id });
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * stack_pos_after = llama_grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    llama_grammar_stack stack_after(stack.begin(), stack.end() - 1);
    if (!llama_grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    llama_grammar_stacks next_stacks;
    llama_grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = llama_grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.partial_utf8, tok.id });
    }

    return rejects;
}

////////////////////

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const struct ollama_vocab * ollama_vocab,
        const llama_grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index) {
    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    // loop over alternates of start rule to build initial stacks
    llama_grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        llama_grammar_stack stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into llama_grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    return new llama_grammar {
        vocab,
        ollama_vocab,
        std::move(vec_rules),
        std::move(stacks),
        /* .partial_utf8 = */             {},
        /* .lazy = */                     false,
        /* .awaiting_trigger = */         false,
        /* .trigger_buffer = */           "",
        /* .trigger_buffer_positions = */ {},
        /* .trigger_tokens = */           {},
        /* .trigger_patterns = */         {},
    };
}

struct llama_grammar * llama_grammar_init_impl(
        const struct llama_vocab * vocab,
        const struct ollama_vocab * ollama_vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    llama_grammar_parser parser(vocab);

    // if there is a grammar, parse it
    // rules will be empty (default) if there are parse errors
    if (!parser.parse(grammar_str) || parser.rules.empty()) {
        fprintf(stderr, "%s: failed to parse grammar\n", __func__);
        return nullptr;
    }

    // Ensure that there is a "root" node.
    if (parser.symbol_ids.find("root") == parser.symbol_ids.end()) {
        fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return nullptr;
    }

    std::vector<const llama_grammar_element *> grammar_rules(parser.c_rules());

    const size_t n_rules = grammar_rules.size();
    const size_t start_rule_index = parser.symbol_ids.at(grammar_root);

    const llama_grammar_element * pos;

    // copy rule definitions into vectors
    llama_grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = grammar_rules[i]; pos->type != LLAMA_GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({LLAMA_GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (llama_grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    // loop over alternates of start rule to build initial stacks
    llama_grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        llama_grammar_stack stack;
        if (!llama_grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        llama_grammar_advance_stack(vec_rules, stack, stacks);
        while (!llama_grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == LLAMA_GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    std::vector<llama_token>    vec_trigger_tokens;
    std::vector<llama_grammar_trigger_pattern> vec_trigger_patterns;
    for (size_t i = 0; i < num_trigger_tokens; i++) {
        GGML_ASSERT(trigger_tokens != nullptr);
        vec_trigger_tokens.push_back(trigger_tokens[i]);
    }
    for (size_t i = 0; i < num_trigger_patterns; i++) {
        GGML_ASSERT(trigger_patterns != nullptr);
        auto & trigger = vec_trigger_patterns.emplace_back();
        trigger.pattern = trigger_patterns[i];
        trigger.regex = std::regex(trigger.pattern);
    }

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into llama_grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    return new llama_grammar {
        vocab,
        ollama_vocab,
        std::move(vec_rules),
        std::move(stacks),
        /* .partial_utf8 = */             {},
        /* .lazy = */                     lazy,
        /* .awaiting_trigger = */         lazy,
        /* .trigger_buffer = */           "",
        /* .trigger_buffer_positions = */ {},
        std::move(vec_trigger_tokens),
        std::move(vec_trigger_patterns),
    };
}

void llama_grammar_free_impl(struct llama_grammar * grammar) {
    if (grammar == nullptr) {
        return;
    }

    delete grammar;
}

struct llama_grammar * llama_grammar_clone_impl(const struct llama_grammar & grammar) {
    auto * result = new llama_grammar {
        grammar.vocab,
        grammar.o_vocab,
        grammar.rules,
        grammar.stacks,
        grammar.partial_utf8,
        grammar.lazy,
        grammar.awaiting_trigger,
        grammar.trigger_buffer,
        grammar.trigger_buffer_positions,
        grammar.trigger_tokens,
        grammar.trigger_patterns,
    };

    // redirect elements in stacks to point to new rules
    for (size_t is = 0; is < result->stacks.size(); is++) {
        for (size_t ie = 0; ie < result->stacks[is].size(); ie++) {
            for (size_t ir0 = 0; ir0 < grammar.rules.size(); ir0++) {
                for (size_t ir1 = 0; ir1 < grammar.rules[ir0].size(); ir1++) {
                    if (grammar.stacks[is][ie] == &grammar.rules[ir0][ir1]) {
                        result->stacks[is][ie] =  &result->rules[ir0][ir1];
                    }
                }
            }
        }
    }

    return result;
}

void llama_grammar_apply_impl(const struct llama_grammar & grammar, llama_token_data_array * cur_p) {

    if (grammar.awaiting_trigger) {
        return;
    }

    bool allow_eog = false;
    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            allow_eog = true;
            break;
        }
    }

    std::vector<std::pair<std::vector<uint32_t>, llama_partial_utf8>> candidates_decoded;
    candidates_decoded.reserve(cur_p->size);

    llama_grammar_candidates candidates_grammar;
    candidates_grammar.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id      = cur_p->data[i].id;
        const std::string piece = grammar.o_vocab ?
            grammar.o_vocab->token_to_piece(id) :
            grammar.vocab->token_to_piece(id);

        const bool is_eog = grammar.o_vocab ? grammar.o_vocab->is_eog(id) : grammar.vocab->is_eog(id);

        if (is_eog) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece, grammar.partial_utf8));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second, id });
        }
    }

    const auto rejects = llama_grammar_reject_candidates(grammar.rules, grammar.stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        cur_p->data[reject.index].logit = -INFINITY;
    }
}

void llama_grammar_accept_impl(struct llama_grammar & grammar, llama_token token) {

    const std::string piece = grammar.o_vocab ?
        grammar.o_vocab->token_to_piece(token) :
        grammar.vocab->token_to_piece(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            llama_grammar_accept_token(grammar, token, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            auto position = std::make_pair(grammar.trigger_buffer.size(), grammar.trigger_buffer.size() + piece.size());
            grammar.trigger_buffer_positions.push_back(std::make_pair(token, position));
            grammar.trigger_buffer += piece;

            std::smatch match;
            for (const auto & trigger_pattern : grammar.trigger_patterns) {
                if (std::regex_match(grammar.trigger_buffer, match, trigger_pattern.regex)) {
                    grammar.awaiting_trigger = false;
                    // get from the first matched capturing group to the end of the string
                    size_t start = std::string::npos;
                    for (auto i = 1u; i < match.size(); i++) {
                        if (match.length(i) > 0) {
                            start = match.position(i);
                            break;
                        }
                    }
                    if (start == std::string::npos) {
                        start = match.position(0);
                    }

                    // replay tokens that overlap with [start, end)
                    for (const auto & [tok, tok_pos] : grammar.trigger_buffer_positions) {
                        auto [tok_start, tok_end] = tok_pos;
                        if (tok_end <= start) {
                            continue;
                        }

                        size_t piece_start = (tok_start < start) ? start : tok_start; // allow for partial token pieces
                        size_t piece_len = tok_end - piece_start;
                        auto tok_piece = grammar.trigger_buffer.substr(piece_start, piece_len);
                        llama_grammar_accept_token(grammar, tok, tok_piece);
                    }

                    auto constrained_str = grammar.trigger_buffer.substr(start);
                    grammar.trigger_buffer.clear();
                    grammar.trigger_buffer_positions.clear();
                    LLAMA_LOG_DEBUG("Grammar triggered on regex: '%s'\n", constrained_str.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", token, piece.c_str());
            return;
        }
    }

    const bool is_eog = grammar.o_vocab ? grammar.o_vocab->is_eog(token) : grammar.vocab->is_eog(token);
    if (is_eog) {
        for (const auto & stack : grammar.stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ABORT("grammar error: end of grammar token received but grammar stack is not empty");
    }

    llama_grammar_accept_token(grammar, token, piece);
}

void llama_grammar_accept_str(struct llama_grammar & grammar, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.partial_utf8);
    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        llama_grammar_accept(&grammar, *it);
    }

    grammar.partial_utf8 = decoded.second;
    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece);
    }
}

void llama_grammar_accept_token(struct llama_grammar & grammar, llama_token token, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.partial_utf8);
    const auto & code_points = decoded.first;

    llama_grammar_stacks stacks_new;
    stacks_new.reserve(grammar.stacks.size());

    for (const auto & stack : grammar.stacks) {
        if (stack.empty()) {
            continue;
        }

        const llama_grammar_element * pos = stack.back();

        if (pos->type == LLAMA_GRETYPE_TOKEN || pos->type == LLAMA_GRETYPE_TOKEN_NOT) {
            if (llama_grammar_match_token(pos, token)) {
                llama_grammar_stack new_stack(stack.begin(), stack.end() - 1);
                if (!llama_grammar_is_end_of_sequence(pos + 1)) {
                    new_stack.push_back(pos + 1);
                }
                llama_grammar_advance_stack(grammar.rules, new_stack, stacks_new);
            }
        } else {
            llama_grammar_stacks current_stacks = {stack};

            for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
                llama_grammar_stacks next_stacks;

                for (const auto & cur_stack : current_stacks) {
                    llama_grammar_accept_chr(grammar, cur_stack, *it, next_stacks);
                }

                current_stacks = std::move(next_stacks);
                if (current_stacks.empty()) {
                    break;
                }
            }

            for (auto & surviving_stack : current_stacks) {
                if (std::find(stacks_new.begin(), stacks_new.end(), surviving_stack) == stacks_new.end()) {
                    stacks_new.emplace_back(surviving_stack);
                }
            }
        }
    }

    grammar.stacks = std::move(stacks_new);
    grammar.partial_utf8 = decoded.second;

    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece + " (" + std::to_string(token) + ")");
    }
}


const std::string & ollama_vocab::token_to_piece(const uint32_t token) const {
    try {
        return token_to_piece_map.at(token);
    } catch (const std::out_of_range&) {
        throw std::runtime_error("Token not found in vocabulary: " + std::to_string(token));
    }
}

void ollama_vocab::add_token_pieces(const uint32_t* tokens, size_t n_tokens, const char** pieces) {
    for (size_t i = 0; i < n_tokens; i++) {
        token_to_piece_map[tokens[i]] = pieces[i];
    }
}

bool ollama_vocab::is_eog(const uint32_t token) const {
    return special_eog_ids.count(token) > 0;
}

void ollama_vocab::set_eog_tokens(const uint32_t* tokens, size_t n_tokens) {
    for (size_t i = 0; i < n_tokens; i++) {
        special_eog_ids.insert(tokens[i]);
    }
}

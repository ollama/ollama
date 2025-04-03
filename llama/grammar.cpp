// #include "llama-grammar.h"
#include "grammar.h"

#include "llama-impl.h"
// #include "llama-vocab.h"
// #include "llama-sampling.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

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

static std::pair<std::vector<uint32_t>, partial_utf8> decode_utf8(
        const std::string & src,
        partial_utf8 partial_start) {
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
            return std::make_pair(std::move(code_points), partial_utf8{ 0, -1 });
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
            return std::make_pair(std::move(code_points), partial_utf8{ 0, n_remain });
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

    return std::make_pair(std::move(code_points), partial_utf8{ value, n_remain });
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

static void print_grammar_char(FILE * file, uint32_t c) {
    if (0x20 <= c && c <= 0x7f) {
        fprintf(file, "%c", static_cast<char>(c));
    } else {
        // cop out of encoding UTF-8
        fprintf(file, "<U+%04X>", c);
    }
}

static bool is_char_element(grammar_element elem) {
    switch (elem.type) {
        case GRETYPE_CHAR:           return true;
        case GRETYPE_CHAR_NOT:       return true;
        case GRETYPE_CHAR_ALT:       return true;
        case GRETYPE_CHAR_RNG_UPPER: return true;
        case GRETYPE_CHAR_ANY:       return true;
        default:                           return false;
    }
}

static void print_rule_binary(FILE * file, const grammar_rule & rule) {
    for (auto elem : rule) {
        switch (elem.type) {
            case GRETYPE_END:            fprintf(file, "END");            break;
            case GRETYPE_ALT:            fprintf(file, "ALT");            break;
            case GRETYPE_RULE_REF:       fprintf(file, "RULE_REF");       break;
            case GRETYPE_CHAR:           fprintf(file, "CHAR");           break;
            case GRETYPE_CHAR_NOT:       fprintf(file, "CHAR_NOT");       break;
            case GRETYPE_CHAR_RNG_UPPER: fprintf(file, "CHAR_RNG_UPPER"); break;
            case GRETYPE_CHAR_ALT:       fprintf(file, "CHAR_ALT");       break;
            case GRETYPE_CHAR_ANY:       fprintf(file, "CHAR_ANY");       break;
        }
        switch (elem.type) {
            case GRETYPE_END:
            case GRETYPE_ALT:
            case GRETYPE_RULE_REF:
                fprintf(file, "(%u) ", elem.value);
                break;
            case GRETYPE_CHAR:
            case GRETYPE_CHAR_NOT:
            case GRETYPE_CHAR_RNG_UPPER:
            case GRETYPE_CHAR_ALT:
            case GRETYPE_CHAR_ANY:
                fprintf(file, "(\"");
                print_grammar_char(file, elem.value);
                fprintf(file, "\") ");
                break;
        }
    }
    fprintf(file, "\n");
}

static void print_rule(
        FILE     * file,
        uint32_t   rule_id,
        const grammar_rule & rule,
        const std::map<uint32_t, std::string> & symbol_id_names) {
    if (rule.empty() || rule.back().type != GRETYPE_END) {
        throw std::runtime_error(
            "malformed rule, does not end with GRETYPE_END: " + std::to_string(rule_id));
    }
    fprintf(file, "%s ::= ", symbol_id_names.at(rule_id).c_str());
    for (size_t i = 0, end = rule.size() - 1; i < end; i++) {
        grammar_element elem = rule[i];
        switch (elem.type) {
            case GRETYPE_END:
                throw std::runtime_error(
                    "unexpected end of rule: " + std::to_string(rule_id) + "," +
                    std::to_string(i));
            case GRETYPE_ALT:
                fprintf(file, "| ");
                break;
            case GRETYPE_RULE_REF:
                fprintf(file, "%s ", symbol_id_names.at(elem.value).c_str());
                break;
            case GRETYPE_CHAR:
                fprintf(file, "[");
                print_grammar_char(file, elem.value);
                break;
            case GRETYPE_CHAR_NOT:
                fprintf(file, "[^");
                print_grammar_char(file, elem.value);
                break;
            case GRETYPE_CHAR_RNG_UPPER:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "GRETYPE_CHAR_RNG_UPPER without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                fprintf(file, "-");
                print_grammar_char(file, elem.value);
                break;
            case GRETYPE_CHAR_ALT:
                if (i == 0 || !is_char_element(rule[i - 1])) {
                    throw std::runtime_error(
                        "GRETYPE_CHAR_ALT without preceding char: " +
                        std::to_string(rule_id) + "," + std::to_string(i));
                }
                print_grammar_char(file, elem.value);
                break;
            case GRETYPE_CHAR_ANY:
                fprintf(file, ".");
                break;
        }
        if (is_char_element(elem)) {
            switch (rule[i + 1].type) {
                case GRETYPE_CHAR_ALT:
                case GRETYPE_CHAR_RNG_UPPER:
                case GRETYPE_CHAR_ANY:
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

uint32_t grammar_parser::get_symbol_id(const char * src, size_t len) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    auto result = symbol_ids.emplace(std::string(src, len), next_id);
    return result.first->second;
}

uint32_t grammar_parser::generate_symbol_id(const std::string & base_name) {
    uint32_t next_id = static_cast<uint32_t>(symbol_ids.size());
    symbol_ids[base_name + '_' + std::to_string(next_id)] = next_id;
    return next_id;
}

void grammar_parser::add_rule(uint32_t rule_id, const grammar_rule & rule) {
    if (rules.size() <= rule_id) {
        rules.resize(rule_id + 1);
    }
    rules[rule_id] = rule;
}

const char * grammar_parser::parse_alternates(
        const char        * src,
        const std::string & rule_name,
        uint32_t            rule_id,
        bool                is_nested) {
    grammar_rule rule;
    const char * pos = parse_sequence(src, rule_name, rule, is_nested);
    while (*pos == '|') {
        rule.push_back({GRETYPE_ALT, 0});
        pos = parse_space(pos + 1, true);
        pos = parse_sequence(pos, rule_name, rule, is_nested);
    }
    rule.push_back({GRETYPE_END, 0});
    add_rule(rule_id, rule);
    return pos;
}

const char * grammar_parser::parse_sequence(
        const char         * src,
        const std::string  & rule_name,
        grammar_rule & rule,
        bool               is_nested) {
    size_t last_sym_start = rule.size();
    const char * pos = src;

    auto handle_repetitions = [&](int min_times, int max_times) {

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

        grammar_rule prev_rule(rule.begin() + last_sym_start, rule.end());
        if (min_times == 0) {
            rule.resize(last_sym_start);
        } else {
            // Repeat the previous elements (min_times - 1) times
            for (int i = 1; i < min_times; i++) {
                rule.insert(rule.end(), prev_rule.begin(), prev_rule.end());
            }
        }

        uint32_t last_rec_rule_id = 0;
        auto n_opt = max_times < 0 ? 1 : max_times - min_times;

        grammar_rule rec_rule(prev_rule);
        for (int i = 0; i < n_opt; i++) {
            rec_rule.resize(prev_rule.size());
            uint32_t rec_rule_id = generate_symbol_id( rule_name);
            if (i > 0 || max_times < 0) {
                rec_rule.push_back({GRETYPE_RULE_REF, max_times < 0 ? rec_rule_id : last_rec_rule_id});
            }
            rec_rule.push_back({GRETYPE_ALT, 0});
            rec_rule.push_back({GRETYPE_END, 0});
            add_rule( rec_rule_id, rec_rule);
            last_rec_rule_id = rec_rule_id;
        }
        if (n_opt > 0) {
            rule.push_back({GRETYPE_RULE_REF, last_rec_rule_id});
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
                rule.push_back({GRETYPE_CHAR, char_pair.first});
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '[') { // char range(s)
            pos++;
            enum gretype start_type = GRETYPE_CHAR;
            if (*pos == '^') {
                pos++;
                start_type = GRETYPE_CHAR_NOT;
            }
            last_sym_start = rule.size();
            while (*pos != ']') {
                if (!*pos) {
                    throw std::runtime_error("unexpected end of input");
                }
                auto char_pair = parse_char(pos);
                     pos       = char_pair.second;
                enum gretype type = last_sym_start < rule.size()
                    ? GRETYPE_CHAR_ALT
                    : start_type;

                rule.push_back({type, char_pair.first});
                if (pos[0] == '-' && pos[1] != ']') {
                    if (!pos[1]) {
                        throw std::runtime_error("unexpected end of input");
                    }
                    auto endchar_pair = parse_char(pos + 1);
                         pos          = endchar_pair.second;
                    rule.push_back({GRETYPE_CHAR_RNG_UPPER, endchar_pair.first});
                }
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (is_word_char(*pos)) { // rule reference
            const char * name_end    = parse_name(pos);
            uint32_t ref_rule_id = get_symbol_id(pos, name_end - pos);
            pos = parse_space(name_end, is_nested);
            last_sym_start = rule.size();
            rule.push_back({GRETYPE_RULE_REF, ref_rule_id});
        } else if (*pos == '(') { // grouping
            // parse nested alternates into synthesized rule
            pos = parse_space(pos + 1, true);
            uint32_t sub_rule_id = generate_symbol_id(rule_name);
            pos = parse_alternates(pos, rule_name, sub_rule_id, true);
            last_sym_start = rule.size();
            // output reference to synthesized rule
            rule.push_back({GRETYPE_RULE_REF, sub_rule_id});
            if (*pos != ')') {
                throw std::runtime_error(std::string("expecting ')' at ") + pos);
            }
            pos = parse_space(pos + 1, is_nested);
        } else if (*pos == '.') { // any char
            last_sym_start = rule.size();
            rule.push_back({GRETYPE_CHAR_ANY, 0});
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
            int min_times = std::stoul(std::string(pos, int_end - pos));
            pos = parse_space(int_end, is_nested);

            int max_times = -1;

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
            handle_repetitions(min_times, max_times);
        } else {
            break;
        }
    }
    return pos;
}

const char * grammar_parser::parse_rule(const char * src) {
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

bool grammar_parser::parse(const char * src) {
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
                if (elem.type == GRETYPE_RULE_REF) {
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

void grammar_parser::print(FILE * file) {
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

grammar_stack grammar_parser::c_rules() const {
    grammar_stack ret;
    ret.reserve(rules.size());
    
    // Only add non-empty rules
    for (const auto & rule : rules) {
        if (!rule.empty()) {
            ret.push_back(rule.data());
        } else {
            LLAMA_LOG_ERROR("empty rule found in grammar_parser::c_rules()");
            return grammar_stack();  // Return empty stack on error
        }
    }
    
    if (ret.empty()) {
        LLAMA_LOG_ERROR("no valid rules found in grammar_parser::c_rules()");
    }
    
    return ret;
}

// returns true iff pos points to the end of one of the definitions of a rule
static bool grammar_is_end_of_sequence(const grammar_element * pos) {
    switch (pos->type) {
        case GRETYPE_END: return true;  // NOLINT
        case GRETYPE_ALT: return true;  // NOLINT
        default:                return false;
    }
}

// returns true iff chr satisfies the char range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static std::pair<bool, const grammar_element *> grammar_match_char(
        const grammar_element * pos,
        const uint32_t                chr) {
    bool found            = false;
    bool is_positive_char = pos->type == GRETYPE_CHAR || pos->type == GRETYPE_CHAR_ANY;

    GGML_ASSERT(is_positive_char || pos->type == GRETYPE_CHAR_NOT); // NOLINT

    do {
        if (pos[1].type == GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            found = found || (pos->value <= chr && chr <= pos[1].value);
            pos += 2;
        } else if (pos->type == GRETYPE_CHAR_ANY) {
            // Any character matches "."
            found = true;
            pos += 1;
        } else {
            // exact char match, e.g. [a] or "a"
            found = found || pos->value == chr;
            pos += 1;
        }
    } while (pos->type == GRETYPE_CHAR_ALT);

    return std::make_pair(found == is_positive_char, pos);
}

// returns true iff some continuation of the given partial UTF-8 sequence could satisfy the char
// range at pos (regular or inverse range)
// asserts that pos is pointing to a char range element
static bool grammar_match_partial_char(
        const grammar_element * pos,
        const partial_utf8      partial_utf8) {
    bool is_positive_char = pos->type == GRETYPE_CHAR || pos->type == GRETYPE_CHAR_ANY;
    GGML_ASSERT(is_positive_char || pos->type == GRETYPE_CHAR_NOT);

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
        if (pos[1].type == GRETYPE_CHAR_RNG_UPPER) {
            // inclusive range, e.g. [a-z]
            if (pos->value <= high && low <= pos[1].value) {
                return is_positive_char;
            }
            pos += 2;
        } else if (pos->type == GRETYPE_CHAR_ANY) {
            // Any character matches "."
            return true;
        } else {
            // exact char match, e.g. [a] or "a"
            if (low <= pos->value && pos->value <= high) {
                return is_positive_char;
            }
            pos += 1;
        }
    } while (pos->type == GRETYPE_CHAR_ALT);

    return !is_positive_char;
}

// transforms a grammar pushdown stack into N possible stacks, all ending
// at a character range (terminal element)
static void grammar_advance_stack(
        const grammar_rules  & rules,
        const grammar_stack  & stack,
              grammar_stacks & new_stacks) {
    if (stack.empty()) {
        if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
            new_stacks.emplace_back(stack);
        }
        return;
    }

    const grammar_element * pos = stack.back();

    switch (pos->type) {
        case GRETYPE_RULE_REF: {
            const size_t             rule_id = static_cast<size_t>(pos->value);
            const grammar_element * subpos  = rules[rule_id].data();
            do {
                // init new stack without the top (pos)
                grammar_stack new_stack(stack.begin(), stack.end() - 1);
                if (!grammar_is_end_of_sequence(pos + 1)) {
                    // if this rule ref is followed by another element, add that to stack
                    new_stack.push_back(pos + 1);
                }
                if (!grammar_is_end_of_sequence(subpos)) {
                    // if alternate is nonempty, add to stack
                    new_stack.push_back(subpos);
                }
                grammar_advance_stack(rules, new_stack, new_stacks);
                while (!grammar_is_end_of_sequence(subpos)) {
                    // scan to end of alternate def
                    subpos++;
                }
                if (subpos->type == GRETYPE_ALT) {
                    // there's another alternate def of this rule to process
                    subpos++;
                } else {
                    break;
                }
            } while (true);
            break;
        }
        case GRETYPE_CHAR:
        case GRETYPE_CHAR_NOT:
        case GRETYPE_CHAR_ANY:
            if (std::find(new_stacks.begin(), new_stacks.end(), stack) == new_stacks.end()) {
                // only add the stack if it's not a duplicate of one we already have
                new_stacks.emplace_back(stack);
            }
            break;
        default:
            // end of alternate (GRETYPE_END, GRETYPE_ALT) or middle of char range
            // (GRETYPE_CHAR_ALT, GRETYPE_CHAR_RNG_UPPER); stack should never be left on
            // those
            GGML_ABORT("fatal error");
    }
}

static grammar_candidates grammar_reject_candidates(
        const grammar_rules      & rules,
        const grammar_stacks     & stacks,
        const grammar_candidates & candidates) {
    GGML_ASSERT(!stacks.empty()); // REVIEW

    if (candidates.empty()) {
        return {};
    }

    auto rejects = grammar_reject_candidates_for_stack(rules, stacks.front(), candidates);

    for (size_t i = 1, size = stacks.size(); i < size; ++i) {
        rejects = grammar_reject_candidates_for_stack(rules, stacks[i], rejects);
    }

    return rejects;
}

static bool grammar_detect_left_recursion(
        const grammar_rules & rules,
        size_t rule_index,
        std::vector<bool> * rules_visited,
        std::vector<bool> * rules_in_progress,
        std::vector<bool> * rules_may_be_empty) {
    if ((*rules_in_progress)[rule_index]) {
        return true;
    }

    (*rules_in_progress)[rule_index] = true;

    const grammar_rule & rule = rules[rule_index];

    // First check if the rule might produce the empty string. This could be done combined with the second
    // step but it's more readable as two steps.
    bool at_rule_start = true;
    for (size_t i = 0; i < rule.size(); i++) {
        if (grammar_is_end_of_sequence(&rule[i])) {
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
        if (rule[i].type == GRETYPE_RULE_REF && recurse_into_nonterminal) {
            if (grammar_detect_left_recursion(rules, (size_t)rule[i].value, rules_visited, rules_in_progress, rules_may_be_empty)) {
                return true;
            }
            if (!((*rules_may_be_empty)[(size_t)rule[i].value])) {
                recurse_into_nonterminal = false;
            }
        } else if (grammar_is_end_of_sequence(&rule[i])) {
            recurse_into_nonterminal = true;
        } else {
            recurse_into_nonterminal = false;
        }
    }

    (*rules_in_progress)[rule_index] = false;
    (*rules_visited)[rule_index] = true;

    return false;
}

const grammar_rules & grammar_get_rules(const struct grammar * grammar) {
    return grammar->rules;
}

grammar_stacks & grammar_get_stacks(struct grammar * grammar) {
    return grammar->stacks;
}

void grammar_accept(struct grammar * grammar, uint32_t chr) {
    grammar_stacks stacks_new;
    stacks_new.reserve(grammar->stacks.size());

    for (const auto & stack : grammar->stacks) {
        if (stack.empty()) {
            continue;
        }

        auto match = grammar_match_char(stack.back(), chr);
        if (match.first) {
            const grammar_element * pos = match.second;

            // update top of stack to next element, if any
            grammar_stack new_stack(stack.begin(), stack.end() - 1);
            if (!grammar_is_end_of_sequence(pos)) {
                new_stack.push_back(pos);
            }
            grammar_advance_stack(grammar->rules, new_stack, stacks_new);
        }
    }

    grammar->stacks = std::move(stacks_new);
}

grammar_candidates grammar_reject_candidates_for_stack(
        const grammar_rules      & rules,
        const grammar_stack      & stack,
        const grammar_candidates & candidates) {

    grammar_candidates rejects;
    rejects.reserve(candidates.size());

    if (stack.empty()) {
        for (const auto & tok : candidates) {
            if (*tok.code_points != 0 || tok.utf8_state.n_remain != 0) {
                rejects.push_back(tok);
            }
        }
        return rejects;
    }

    const grammar_element * stack_pos = stack.back();

    grammar_candidates next_candidates;
    next_candidates.reserve(candidates.size());

    for (const auto & tok : candidates) {
        if (*tok.code_points == 0) {
            // reached end of full codepoints in token, reject iff it ended in a partial sequence
            // that cannot satisfy this position in grammar
            if (tok.utf8_state.n_remain != 0 &&
                    !grammar_match_partial_char(stack_pos, tok.utf8_state)) {
                rejects.push_back(tok);
            }
        } else if (grammar_match_char(stack_pos, *tok.code_points).first) {
            next_candidates.push_back({ tok.index, tok.code_points + 1, tok.utf8_state });
        } else {
            rejects.push_back(tok);
        }
    }

    const auto * stack_pos_after = grammar_match_char(stack_pos, 0).second;

    // update top of stack to next element, if any
    grammar_stack stack_after(stack.begin(), stack.end() - 1);
    if (!grammar_is_end_of_sequence(stack_pos_after)) {
        stack_after.push_back(stack_pos_after);
    }
    grammar_stacks next_stacks;
    grammar_advance_stack(rules, stack_after, next_stacks);

    auto next_rejects = grammar_reject_candidates(rules, next_stacks, next_candidates);
    for (const auto & tok : next_rejects) {
        rejects.push_back({ tok.index, tok.code_points - 1, tok.utf8_state });
    }

    return rejects;
}

////////////////////

struct grammar * grammar_init_impl(
        struct ollama_vocab * ollama_vocab,
        const grammar_element ** rules,
        size_t n_rules,
        size_t start_rule_index) {
    const grammar_element * pos;

    // copy rule definitions into vectors
    grammar_rules vec_rules(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        for (pos = rules[i]; pos->type != GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        vec_rules[i].push_back({GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    // loop over alternates of start rule to build initial stacks
    grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        grammar_stack stack;
        if (!grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        grammar_advance_stack(vec_rules, stack, stacks);
        while (!grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    return new grammar {
        ollama_vocab,
        std::move(vec_rules),
        std::move(stacks),
        /* .partial_utf8 = */     {},
        /* .lazy =*/              false,
        /* .awaiting_trigger = */ false,
        /* .trigger_buffer = */   "",
        /* .trigger_tokens   = */ {},
        /* .trigger_words    = */ {},
    };
}

struct grammar * grammar_init_impl(
                    struct ollama_vocab * ollama_vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                              bool lazy,
                     const char ** trigger_words,
                            size_t num_trigger_words,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens) {
    grammar_parser parser;

    // if there is a grammar, parse it
    if (!parser.parse(grammar_str)) {
        LLAMA_LOG_ERROR("failed to parse grammar");
        return nullptr;
    }

    // will be empty (default) if there are parse errors
    if (parser.rules.empty()) {
        fprintf(stderr, "%s: failed to parse grammar\n", __func__);
        return nullptr;
    }

    // Ensure that there is a "root" node.
    if (parser.symbol_ids.find("root") == parser.symbol_ids.end()) {
        fprintf(stderr, "%s: grammar does not contain a 'root' symbol\n", __func__);
        return nullptr;
    }

    std::vector<const grammar_element *> grammar_rules_vec(parser.c_rules());

    const size_t n_rules = grammar_rules_vec.size();
    const size_t start_rule_index = parser.symbol_ids.at(grammar_root);

    const grammar_element * pos;

    // copy rule definitions into vectors
    grammar_rules vec_rules;
    vec_rules.resize(n_rules);  // Pre-allocate space for all rules
    
    for (size_t i = 0; i < n_rules; i++) {
        if (grammar_rules_vec[i] == nullptr) {
            LLAMA_LOG_ERROR("null rule pointer at index %zu", i);
            return nullptr;
        }
        
        // Reserve space for the rule to avoid reallocation
        vec_rules[i].reserve(16);  // Start with reasonable capacity
        
        for (pos = grammar_rules_vec[i]; pos != nullptr && pos->type != GRETYPE_END; pos++) {
            vec_rules[i].push_back(*pos);
        }
        
        if (pos == nullptr) {
            LLAMA_LOG_ERROR("unexpected null pointer while copying rule %zu", i);
            return nullptr;
        }
        
        vec_rules[i].push_back({GRETYPE_END, 0});
    }

    // Check for left recursion
    std::vector<bool> rules_visited(n_rules);
    std::vector<bool> rules_in_progress(n_rules);
    std::vector<bool> rules_may_be_empty(n_rules);
    for (size_t i = 0; i < n_rules; i++) {
        if (rules_visited[i]) {
            continue;
        }
        if (grammar_detect_left_recursion(vec_rules, i, &rules_visited, &rules_in_progress, &rules_may_be_empty)) {
            LLAMA_LOG_ERROR("unsupported grammar, left recursion detected for nonterminal at index %zu", i);
            return nullptr;
        }
    }

    // loop over alternates of start rule to build initial stacks
    grammar_stacks stacks;
    pos = vec_rules[start_rule_index].data();
    do {
        grammar_stack stack;
        if (!grammar_is_end_of_sequence(pos)) {
            // if alternate is nonempty, add to stack
            stack.push_back(pos);
        }
        grammar_advance_stack(vec_rules, stack, stacks);
        while (!grammar_is_end_of_sequence(pos)) {
            // scan to end of alternate def
            pos++;
        }
        if (pos->type == GRETYPE_ALT) {
            // there's another alternate def of this rule to process
            pos++;
        } else {
            break;
        }
    } while (true);

    std::vector<llama_token>    vec_trigger_tokens;
    std::vector<std::string> vec_trigger_words;
    for (size_t i = 0; i < num_trigger_tokens; i++) {
        GGML_ASSERT(trigger_tokens != nullptr);
        vec_trigger_tokens.push_back(trigger_tokens[i]);
    }
    for (size_t i = 0; i < num_trigger_words; i++) {
        GGML_ASSERT(trigger_words != nullptr);
        vec_trigger_words.push_back(trigger_words[i]);
    }

    // Important: vec_rules has to be moved here, not copied, because stacks contains
    // pointers to elements of vec_rules. If vec_rules were copied into grammar
    // then the pointers would be invalidated when the local vec_rules goes out of scope.
    return new grammar {
        ollama_vocab,
        std::move(vec_rules),
        std::move(stacks),
        /* .partial_utf8 = */     {},
        /* .lazy = */             lazy,
        /* .awaiting_trigger = */ lazy,
        /* .trigger_buffer = */   "",
        std::move(vec_trigger_tokens),
        std::move(vec_trigger_words),
    };
}

void grammar_free_impl(struct grammar * grammar) {
    if (grammar == nullptr) {
        return;
    }

    delete grammar;
}

struct grammar * grammar_clone_impl(const struct grammar & g) {
    grammar * result = new grammar {
        g.vocab,
        g.rules,
        g.stacks,
        g.utf8_state,
        g.lazy,
        g.awaiting_trigger,
        g.trigger_buffer,
        g.trigger_tokens,
        g.trigger_words,
    };

    // redirect elements in stacks to point to new rules
    for (size_t is = 0; is < result->stacks.size(); is++) {
        for (size_t ie = 0; ie < result->stacks[is].size(); ie++) {
            for (size_t ir0 = 0; ir0 < g.rules.size(); ir0++) {
                for (size_t ir1 = 0; ir1 < g.rules[ir0].size(); ir1++) {
                    if (g.stacks[is][ie] == &g.rules[ir0][ir1]) {
                        result->stacks[is][ie] =  &result->rules[ir0][ir1];
                    }
                }
            }
        }
    }

    return result;
}

void grammar_apply_impl(const struct grammar & grammar, llama_token_data_array * cur_p) {
    GGML_ASSERT(grammar.vocab != nullptr);

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

    std::vector<std::pair<std::vector<uint32_t>, partial_utf8>> candidates_decoded;
    candidates_decoded.reserve(cur_p->size);

    grammar_candidates candidates_grammar;
    candidates_grammar.reserve(cur_p->size);

    for (size_t i = 0; i < cur_p->size; ++i) {
        const llama_token id      = cur_p->data[i].id;
        const std::string & piece = grammar.vocab->token_to_piece.at(id);

        if (id == grammar.vocab->eog_token) {
            if (!allow_eog) {
                cur_p->data[i].logit = -INFINITY;
            }
        } else if (piece.empty() || piece[0] == 0) {
            cur_p->data[i].logit = -INFINITY;
        } else {
            candidates_decoded.push_back(decode_utf8(piece, grammar.utf8_state));
            candidates_grammar.push_back({ i, candidates_decoded.back().first.data(), candidates_decoded.back().second });
        }
    }

    const auto rejects = grammar_reject_candidates(grammar.rules, grammar.stacks, candidates_grammar);
    for (const auto & reject : rejects) {
        cur_p->data[reject.index].logit = -INFINITY;
    }
}

void grammar_accept_impl(struct grammar & grammar, llama_token token) {
    GGML_ASSERT(grammar.vocab != nullptr);

    const auto & piece = grammar.vocab->token_to_piece.at(token);

    if (grammar.awaiting_trigger) {
        if (std::find(grammar.trigger_tokens.begin(), grammar.trigger_tokens.end(), token) != grammar.trigger_tokens.end()) {
            grammar.awaiting_trigger = false;
            grammar.trigger_buffer.clear();
            grammar_accept_str(grammar, piece);
            LLAMA_LOG_DEBUG("Grammar triggered on token %u (`%s`)", token, piece.c_str());
            return;
        } else {
            // TODO: consider a smarter incremental substring search algorithm (store last position to search from).
            grammar.trigger_buffer += piece;
            for (const auto & word : grammar.trigger_words) {
                auto pos = grammar.trigger_buffer.find(word);
                if (pos != std::string::npos) {
                    grammar.awaiting_trigger = false;
                    auto constrained_str = grammar.trigger_buffer.substr(pos);
                    grammar.trigger_buffer.clear();
                    grammar_accept_str(grammar, constrained_str);
                    LLAMA_LOG_DEBUG("Grammar triggered on word `%s`", word.c_str());
                    return;
                }
            }
            LLAMA_LOG_DEBUG("Grammar still awaiting trigger after token %d (`%s`)\n", token, piece.c_str());
            return;
        }
    }

    if (token == grammar.vocab->eog_token) {
        for (const auto & stack : grammar.stacks) {
            if (stack.empty()) {
                return;
            }
        }
        GGML_ABORT("fatal error");
    }

    grammar_accept_str(grammar, piece);
}

void grammar_accept_str(struct grammar & grammar, const std::string & piece) {
    // Note terminating 0 in decoded string
    const auto   decoded     = decode_utf8(piece, grammar.utf8_state);
    const auto & code_points = decoded.first;

    for (auto it = code_points.begin(), end = code_points.end() - 1; it != end; ++it) {
        grammar_accept(&grammar, *it);
    }

    grammar.utf8_state = decoded.second;
    if (grammar.stacks.empty()) {
        throw std::runtime_error("Unexpected empty grammar stack after accepting piece: " + piece);
    }
}

void ollama_vocab::set_eog_token(uint32_t token){
    eog_token = token;
}

void ollama_vocab::add_symbol_id(const std::string & symbol, uint32_t id){
    symbol_ids[symbol] = id;
}

void ollama_vocab::add_token_piece(uint32_t token, const std::string & piece){
    token_to_piece[token] = piece;
}

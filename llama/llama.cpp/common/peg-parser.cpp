#include "common.h"
#include "peg-parser.h"
#include "json-schema-to-grammar.h"
#include "unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <initializer_list>
#include <map>
#include <memory>
#include <regex>
#include <stdexcept>
#include <unordered_set>

// Trick to catch missing branches
template <typename T>
inline constexpr bool is_always_false_v = false;

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type) {
    switch (type) {
        case COMMON_PEG_PARSE_RESULT_FAIL:            return "fail";
        case COMMON_PEG_PARSE_RESULT_SUCCESS:         return "success";
        case COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT: return "need_more_input";
        default:                                      return "unknown";
    }
}

static bool is_hex_digit(const char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

// Trie for matching multiple literals.
// This is used in common_peg_until_parser and to build a GBNF exclusion grammar
struct trie {
    struct node {
        size_t depth = 0;
        std::map<unsigned char, size_t> children;
        bool is_word;
    };

    std::vector<node> nodes;

    trie(const std::vector<std::string> & words) {
      create_node(); // root node
      for (const auto & w : words) {
          insert(w);
      }
    }

    enum match_result { NO_MATCH, PARTIAL_MATCH, COMPLETE_MATCH };

    // Check if a delimiter starts at the given position
    match_result check_at(std::string_view sv, size_t start_pos) const {
        size_t current = 0; // Start at root
        size_t pos = start_pos;

        while (pos < sv.size()) {
            auto it = nodes[current].children.find(sv[pos]);
            if (it == nodes[current].children.end()) {
                // Can't continue matching
                return match_result{match_result::NO_MATCH};
            }

            current = it->second;
            pos++;

            // Check if we've matched a complete word
            if (nodes[current].is_word) {
                return match_result{match_result::COMPLETE_MATCH};
            }
        }

        // Reached end of input while still in the trie (not at root)
        if (current != 0) {
            // We're in the middle of a potential match
            return match_result{match_result::PARTIAL_MATCH};
        }

        // Reached end at root (no match)
        return match_result{match_result::NO_MATCH};
    }

    struct prefix_and_next {
        std::string prefix;
        std::string next_chars;
    };

    std::vector<prefix_and_next> collect_prefix_and_next() {
        std::string prefix;
        std::vector<prefix_and_next> result;
        collect_prefix_and_next(0, prefix, result);
        return result;
    }

  private:
    void collect_prefix_and_next(size_t index, std::string & prefix, std::vector<prefix_and_next> & out) {
        if (!nodes[index].is_word) {
            if (!nodes[index].children.empty()) {
                std::string chars;
                chars.reserve(nodes[index].children.size());
                for (const auto & p : nodes[index].children) {
                    chars.push_back(p.first);
                }
                out.emplace_back(prefix_and_next{prefix, chars});
            }
        }

        for (const auto & p : nodes[index].children) {
            unsigned char ch = p.first;
            auto child = p.second;
            prefix.push_back(ch);
            collect_prefix_and_next(child, prefix, out);
            prefix.pop_back();
        }
    }

    size_t create_node() {
        size_t index = nodes.size();
        nodes.emplace_back();
        return index;
    }

    void insert(const std::string & word) {
        size_t current = 0;
        for (unsigned char ch : word) {
            auto it = nodes[current].children.find(ch);
            if (it == nodes[current].children.end()) {
                size_t child = create_node();
                nodes[child].depth = nodes[current].depth + 1;
                nodes[current].children[ch] = child;
                current = child;
            } else {
                current = it->second;
            }
        }
        nodes[current].is_word = true;
    }
};

static std::pair<uint32_t, size_t> parse_hex_escape(const std::string & str, size_t pos, int hex_count) {
    if (pos + hex_count > str.length()) {
        return {0, 0};
    }

    uint32_t value = 0;
    for (int i = 0; i < hex_count; i++) {
        char c = str[pos + i];
        if (!is_hex_digit(c)) {
            return {0, 0};
        }
        value <<= 4;
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
    return {value, static_cast<size_t>(hex_count)};
}

static std::pair<uint32_t, size_t> parse_char_class_char(const std::string & content, size_t pos) {
    if (content[pos] == '\\' && pos + 1 < content.length()) {
        switch (content[pos + 1]) {
            case 'x': {
                auto result = parse_hex_escape(content, pos + 2, 2);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'x'
                return {static_cast<uint32_t>('x'), 2};
            }
            case 'u': {
                auto result = parse_hex_escape(content, pos + 2, 4);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'u'
                return {static_cast<uint32_t>('u'), 2};
            }
            case 'U': {
                auto result = parse_hex_escape(content, pos + 2, 8);
                if (result.second > 0) {
                    return {result.first, 2 + result.second};
                }
                // Invalid escape, treat as literal 'U'
                return {static_cast<uint32_t>('U'), 2};
            }
            case 'n':  return {'\n', 2};
            case 't':  return {'\t', 2};
            case 'r':  return {'\r', 2};
            case '\\': return {'\\', 2};
            case ']':  return {']', 2};
            case '[':  return {'[', 2};
            default:   return {static_cast<uint32_t>(content[pos + 1]), 2};
        }
    }

    // Regular character - return as codepoint
    return {static_cast<uint32_t>(static_cast<unsigned char>(content[pos])), 1};
}

static std::pair<std::vector<common_peg_chars_parser::char_range>, bool> parse_char_classes(const std::string & classes) {
    std::vector<common_peg_chars_parser::char_range> ranges;
    bool negated = false;

    std::string content = classes;
    if (content.front() == '[') {
        content = content.substr(1);
    }

    if (content.back() == ']') {
        content.pop_back();
    }

    // Check for negation
    if (!content.empty() && content.front() == '^') {
        negated = true;
        content = content.substr(1);
    }

    size_t i = 0;
    while (i < content.length()) {
        auto [start, start_len] = parse_char_class_char(content, i);
        i += start_len;

        if (i + 1 < content.length() && content[i] == '-') {
            // Range detected
            auto [end, end_len] = parse_char_class_char(content, i + 1);
            ranges.push_back(common_peg_chars_parser::char_range{start, end});
            i += 1 + end_len;
        } else {
            ranges.push_back(common_peg_chars_parser::char_range{start, start});
        }
    }

    return {ranges, negated};
}

void common_peg_ast_arena::visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const {
    if (id == COMMON_PEG_INVALID_AST_ID) {
        return;
    }
    const auto & node = get(id);
    visitor(node);
    for (const auto & child : node.children) {
        visit(child, visitor);
    }
}

void common_peg_ast_arena::visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const {
    for (const auto & node : result.nodes) {
        visit(node, visitor);
    }
}

struct parser_executor;

common_peg_parser_id common_peg_arena::add_parser(common_peg_parser_variant parser) {
    common_peg_parser_id id = parsers_.size();
    parsers_.push_back(std::move(parser));
    return id;
}

void common_peg_arena::add_rule(const std::string & name, common_peg_parser_id id) {
    rules_[name] = id;
}

common_peg_parser_id common_peg_arena::get_rule(const std::string & name) const {
    auto it = rules_.find(name);
    if (it == rules_.end()) {
        throw std::runtime_error("Rule not found: " + name);
    }
    return it->second;
}

struct parser_executor {
    const common_peg_arena & arena;
    common_peg_parse_context & ctx;
    size_t start_pos;

    parser_executor(const common_peg_arena & arena, common_peg_parse_context & ctx, size_t start)
        : arena(arena), ctx(ctx), start_pos(start) {}

    common_peg_parse_result operator()(const common_peg_epsilon_parser & /* p */) const {
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_start_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos == 0 ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_end_parser & /* p */) const {
        return common_peg_parse_result(
            start_pos >= ctx.input.size() ? COMMON_PEG_PARSE_RESULT_SUCCESS : COMMON_PEG_PARSE_RESULT_FAIL,
            start_pos
        );
    }

    common_peg_parse_result operator()(const common_peg_literal_parser & p) {
        auto pos = start_pos;
        for (auto i = 0u; i < p.literal.size(); ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            if (ctx.input[pos] != p.literal[i]) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            ++pos;
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_sequence_parser & p) {
        auto pos = start_pos;
        std::vector<common_peg_ast_id> nodes;

        for (const auto & child_id : p.children) {
            auto result = arena.parse(child_id, ctx, pos);
            if (result.fail()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, result.end);
            }

            if (!result.nodes.empty()) {
                nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
            }

            if (result.need_more_input()) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            pos = result.end;
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_choice_parser & p) {
        auto pos = start_pos;
        for (const auto & child_id : p.children) {
            auto result = arena.parse(child_id, ctx, pos);
            if (!result.fail()) {
                return result;
            }
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_repetition_parser & p) {
        auto pos = start_pos;
        int match_count = 0;
        std::vector<common_peg_ast_id> nodes;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            if (pos >= ctx.input.size()) {
                break;
            }

            auto result = arena.parse(p.child, ctx, pos);

            if (result.success()) {
                // Prevent infinite loop on empty matches
                if (result.end == pos) {
                    break;
                }

                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                pos = result.end;
                match_count++;
                continue;
            }

            if (result.need_more_input()) {
                if (!result.nodes.empty()) {
                    nodes.insert(nodes.end(), result.nodes.begin(), result.nodes.end());
                }

                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, result.end, std::move(nodes));
            }

            // Child failed - stop trying
            break;
        }

        // Check if we got enough matches
        if (p.min_count > 0 && match_count < p.min_count) {
            if (pos >= ctx.input.size() && ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos, std::move(nodes));
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos, std::move(nodes));
    }

    common_peg_parse_result operator()(const common_peg_and_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        // Pass result but don't consume input
        return common_peg_parse_result(result.type, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_not_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);

        if (result.success()) {
            // Fail if the underlying parser matches
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }

        if (result.need_more_input()) {
            // Propagate - need to know what child would match before negating
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }

        // Child failed, so negation succeeds
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_any_parser & /* p */) const {
        // Parse a single UTF-8 codepoint (not just a single byte)
        auto result = parse_utf8_codepoint(ctx.input, start_pos);

        if (result.status == utf8_parse_result::INCOMPLETE) {
            if (!ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos);
        }
        if (result.status == utf8_parse_result::INVALID) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, start_pos + result.bytes_consumed);
    }

    common_peg_parse_result operator()(const common_peg_space_parser & /* p */) {
        auto pos = start_pos;
        while (pos < ctx.input.size()) {
            auto c = static_cast<unsigned char>(ctx.input[pos]);
            if (std::isspace(c)) {
                ++pos;
            } else {
                break;
            }
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_chars_parser & p) const {
        auto pos = start_pos;
        int match_count = 0;

        // Try to match up to max_count times (or unlimited if max_count is -1)
        while (p.max_count == -1 || match_count < p.max_count) {
            auto result = parse_utf8_codepoint(ctx.input, pos);

            if (result.status == utf8_parse_result::INCOMPLETE) {
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed with what we have
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches yet
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }

            if (result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8 in input
                if (match_count >= p.min_count) {
                    // We have enough matches, succeed up to here
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
                }
                // Not enough matches, fail
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if this codepoint matches our character class
            bool matches = false;
            for (const auto & range : p.ranges) {
                if (range.contains(result.codepoint)) {
                    matches = true;
                    break;
                }
            }

            // If negated, invert the match result
            if (p.negated) {
                matches = !matches;
            }

            if (matches) {
                pos += result.bytes_consumed;
                ++match_count;
            } else {
                // Character doesn't match, stop matching
                break;
            }
        }

        // Check if we got enough matches
        if (match_count < p.min_count) {
            if (pos >= ctx.input.size() && ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }

        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
    }

    static common_peg_parse_result handle_escape_sequence(common_peg_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume '\'
        if (pos >= ctx.input.size()) {
            if (!ctx.is_partial) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
        }

        switch (ctx.input[pos]) {
            case '"':
            case '\\':
            case '/':
            case 'b':
            case 'f':
            case 'n':
            case 'r':
            case 't':
                ++pos;
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
            case 'u':
                return handle_unicode_escape(ctx, start, pos);
            default:
                // Invalid escape sequence
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
        }
    }

    static common_peg_parse_result handle_unicode_escape(common_peg_parse_context & ctx, size_t start, size_t & pos) {
        ++pos; // consume 'u'
        for (int i = 0; i < 4; ++i) {
            if (pos >= ctx.input.size()) {
                if (!ctx.is_partial) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
                }
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start, pos);
            }
            if (!is_hex_digit(ctx.input[pos])) {
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start);
            }
            ++pos;
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start, pos);
    }

    common_peg_parse_result operator()(const common_peg_json_string_parser & /* p */) {
        auto pos = start_pos;

        // Parse string content (without quotes)
        while (pos < ctx.input.size()) {
            char c = ctx.input[pos];

            if (c == '"') {
                // Found closing quote - success (don't consume it)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (c == '\\') {
                auto result = handle_escape_sequence(ctx, start_pos, pos);
                if (!result.success()) {
                    return result;
                }
            } else {
                auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

                if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                    if (!ctx.is_partial) {
                        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                    }
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
                }

                if (utf8_result.status == utf8_parse_result::INVALID) {
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }

                pos += utf8_result.bytes_consumed;
            }
        }

        // Reached end without finding closing quote
        if (!ctx.is_partial) {
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos, pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, pos);
    }

    common_peg_parse_result operator()(const common_peg_until_parser & p) const {
        trie matcher(p.delimiters);

        // Scan input and check for delimiters
        size_t pos = start_pos;
        size_t last_valid_pos = start_pos;

        while (pos < ctx.input.size()) {
            auto utf8_result = parse_utf8_codepoint(ctx.input, pos);

            if (utf8_result.status == utf8_parse_result::INCOMPLETE) {
                // Incomplete UTF-8 sequence
                if (!ctx.is_partial) {
                    // Input is complete but UTF-8 is incomplete = malformed
                    return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
                }
                // Return what we have so far (before incomplete sequence)
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
            }

            if (utf8_result.status == utf8_parse_result::INVALID) {
                // Malformed UTF-8
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_FAIL, start_pos);
            }

            // Check if a delimiter starts at this position
            auto match = matcher.check_at(ctx.input, pos);

            if (match == trie::COMPLETE_MATCH) {
                // Found a complete delimiter, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            if (match == trie::PARTIAL_MATCH) {
                // Found a partial match extending to end of input, return everything before it
                return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, pos);
            }

            pos += utf8_result.bytes_consumed;
            last_valid_pos = pos;
        }

        if (last_valid_pos == ctx.input.size() && ctx.is_partial) {
            // Reached the end of a partial stream, there might still be more input that we need to consume.
            return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT, start_pos, last_valid_pos);
        }
        return common_peg_parse_result(COMMON_PEG_PARSE_RESULT_SUCCESS, start_pos, last_valid_pos);
    }

    common_peg_parse_result operator()(const common_peg_schema_parser & p) {
        return arena.parse(p.child, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_rule_parser & p) {
        // Parse the child
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                p.name,
                "",
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_tag_parser & p) {
        // Parse the child
        auto result = arena.parse(p.child, ctx, start_pos);

        if (!result.fail()) {
            std::string_view text;
            if (result.start < ctx.input.size()) {
                text = std::string_view(ctx.input).substr(result.start, result.end - result.start);
            }

            auto node_id = ctx.ast.add_node(
                "",
                p.tag,
                result.start,
                result.end,
                text,
                std::move(result.nodes),
                result.need_more_input()
            );

            return common_peg_parse_result(result.type, result.start, result.end, { node_id });
        }

        return result;
    }

    common_peg_parse_result operator()(const common_peg_ref_parser & p) {
        auto rule_id = arena.get_rule(p.name);
        return arena.parse(rule_id, ctx, start_pos);
    }

    common_peg_parse_result operator()(const common_peg_atomic_parser & p) {
        auto result = arena.parse(p.child, ctx, start_pos);
        if (result.need_more_input()) {
            // Clear nodes so they don't propagate up.
            result.nodes.clear();
        }
        return result;
    }
};

common_peg_parse_result common_peg_arena::parse(common_peg_parse_context & ctx, size_t start) const {
    if (root_ == COMMON_PEG_INVALID_PARSER_ID) {
        throw std::runtime_error("No root parser set");
    }
    return parse(root_, ctx, start);
}

common_peg_parse_result common_peg_arena::parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const {
    // Execute parser
    const auto & parser = parsers_.at(id);
    parser_executor exec(*this, ctx, start);
    return std::visit(exec, parser);
}

common_peg_parser_id common_peg_arena::resolve_ref(common_peg_parser_id id) {
    const auto & parser = parsers_.at(id);
    if (auto ref = std::get_if<common_peg_ref_parser>(&parser)) {
        return get_rule(ref->name);
    }
    return id;
}

void common_peg_arena::resolve_refs() {
    // Walk through all parsers and replace refs with their corresponding rule IDs
    for (auto & parser : parsers_) {
        std::visit([this](auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto & child : p.children) {
                    child = resolve_ref(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                p.child = resolve_ref(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                                 std::is_same_v<T, common_peg_start_parser> ||
                                 std::is_same_v<T, common_peg_end_parser> ||
                                 std::is_same_v<T, common_peg_ref_parser> ||
                                 std::is_same_v<T, common_peg_until_parser> ||
                                 std::is_same_v<T, common_peg_literal_parser> ||
                                 std::is_same_v<T, common_peg_json_string_parser> ||
                                 std::is_same_v<T, common_peg_chars_parser> ||
                                 std::is_same_v<T, common_peg_any_parser> ||
                                 std::is_same_v<T, common_peg_space_parser>) {
                // These rules do not have children
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    }

    // Also flatten root if it's a ref
    if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        root_ = resolve_ref(root_);
    }
}

std::string common_peg_arena::dump(common_peg_parser_id id) const {
    const auto & parser = parsers_.at(id);

    return std::visit([this](const auto & p) -> std::string {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return "Epsilon";
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return "Start";
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return "End";
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return "Literal(" + p.literal + ")";
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump(child));
            }
            return "Sequence(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            std::vector<std::string> parts;
            for (const auto & child : p.children) {
                parts.push_back(dump(child));
            }
            return "Choice(" + string_join(parts, ", ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            if (p.max_count == -1) {
                return "Repetition(" + dump(p.child) + ", " + std::to_string(p.min_count) + ", unbounded)";
            }
            return "Repetition(" + dump(p.child) + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return "And(" + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return "Not(" + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return "Any";
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return "Space";
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            if (p.max_count == -1) {
                return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", unbounded)";
            }
            return "CharRepeat(" + p.pattern + ", " + std::to_string(p.min_count) + ", " + std::to_string(p.max_count) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
            return "JsonString()";
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return "Until(" + string_join(p.delimiters, " | ") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return "Schema(" + dump(p.child) + ", " + (p.schema ? p.schema->dump() : "null") + ")";
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return "Rule(" + p.name + ", " + dump(p.child) + ")";
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return "Ref(" + p.name + ")";
        } else {
            return "Unknown";
        }
    }, parser);
}

common_peg_parser & common_peg_parser::operator=(const common_peg_parser & other) {
    id_ = other.id_;
    return *this;
}

common_peg_parser & common_peg_parser::operator+=(const common_peg_parser & other) {
    id_ = builder_.sequence({id_, other.id_});
    return *this;
}

common_peg_parser & common_peg_parser::operator|=(const common_peg_parser & other) {
    id_ = builder_.choice({id_, other.id_});
    return *this;
}

common_peg_parser common_peg_parser::operator+(const common_peg_parser & other) const {
    return builder_.sequence({id_, other.id_});
}

common_peg_parser common_peg_parser::operator|(const common_peg_parser & other) const {
    return builder_.choice({id_, other.id_});
}

common_peg_parser common_peg_parser::operator<<(const common_peg_parser & other) const {
    return builder_.sequence({id_, builder_.space(), other.id_});
}

common_peg_parser common_peg_parser::operator+(const char * str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator+(const std::string & str) const {
    return *this + builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const char * str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator<<(const std::string & str) const {
    return *this << builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const char * str) const {
    return *this | builder_.literal(str);
}

common_peg_parser common_peg_parser::operator|(const std::string & str) const {
    return *this | builder_.literal(str);
}

common_peg_parser operator+(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) + p;
}

common_peg_parser operator+(const std::string & str, const common_peg_parser & p) {
    return operator+(str.c_str(), p);
}

common_peg_parser operator<<(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) << p;
}

common_peg_parser operator<<(const std::string & str, const common_peg_parser & p) {
    return operator<<(str.c_str(), p);
}

common_peg_parser operator|(const char * str, const common_peg_parser & p) {
    return p.builder().literal(str) | p;
}

common_peg_parser operator|(const std::string & str, const common_peg_parser & p) {
    return operator|(str.c_str(), p);
}

static std::string rule_name(const std::string & name) {
    static const std::regex invalid_rule_chars_re("[^a-zA-Z0-9-]+");
    return std::regex_replace(name, invalid_rule_chars_re, "-");
}

common_peg_parser_builder::common_peg_parser_builder() {}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested sequences
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto seq = std::get_if<common_peg_sequence_parser>(&parser)) {
            flattened.insert(flattened.end(), seq->children.begin(), seq->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_sequence_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::sequence(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::sequence(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return sequence(ids);
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser_id> & parsers) {
    // Flatten nested choices
    std::vector<common_peg_parser_id> flattened;
    for (const auto & p : parsers) {
        const auto & parser = arena_.get(p);
        if (auto choice = std::get_if<common_peg_choice_parser>(&parser)) {
            flattened.insert(flattened.end(), choice->children.begin(), choice->children.end());
        } else {
            flattened.push_back(p);
        }
    }
    return wrap(arena_.add_parser(common_peg_choice_parser{flattened}));
}

common_peg_parser common_peg_parser_builder::choice(const std::vector<common_peg_parser> & parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::choice(std::initializer_list<common_peg_parser> parsers) {
    std::vector<common_peg_parser_id> ids;
    ids.reserve(parsers.size());
    for (const auto & p : parsers) {
        ids.push_back(p.id());
    }
    return choice(ids);
}

common_peg_parser common_peg_parser_builder::chars(const std::string & classes, int min, int max) {
    auto [ranges, negated] = parse_char_classes(classes);
    return wrap(arena_.add_parser(common_peg_chars_parser{classes, ranges, negated, min, max}));
}

common_peg_parser common_peg_parser_builder::schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw) {
    return wrap(arena_.add_parser(common_peg_schema_parser{p.id(), name, std::make_shared<nlohmann::ordered_json>(schema), raw}));
}

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const common_peg_parser & p, bool trigger) {
    auto clean_name = rule_name(name);
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, p.id(), trigger});
    arena_.add_rule(clean_name, rule_id);
    return ref(clean_name);
}

common_peg_parser common_peg_parser_builder::rule(const std::string & name, const std::function<common_peg_parser()> & builder_fn, bool trigger) {
    auto clean_name = rule_name(name);
    if (arena_.has_rule(clean_name)) {
        return ref(clean_name);
    }

    // Create placeholder rule to allow recursive references
    auto placeholder = any();  // Temporary placeholder
    auto placeholder_rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, placeholder.id(), trigger});
    arena_.add_rule(clean_name, placeholder_rule_id);

    // Build the actual parser
    auto parser = builder_fn();

    // Replace placeholder with actual rule
    auto rule_id = arena_.add_parser(common_peg_rule_parser{clean_name, parser.id(), trigger});
    arena_.rules_[clean_name] = rule_id;

    return ref(clean_name);
}

void common_peg_parser_builder::set_root(const common_peg_parser & p) {
    arena_.set_root(p.id());
}

common_peg_arena common_peg_parser_builder::build() {
    arena_.resolve_refs();
    return std::move(arena_);
}

// JSON parsers
common_peg_parser common_peg_parser_builder::json_number() {
   return rule("json-number", [this]() {
        auto digit1_9 = chars("[1-9]", 1, 1);
        auto digits = chars("[0-9]");
        auto int_part = choice({literal("0"), sequence({digit1_9, chars("[0-9]", 0, -1)})});
        auto frac = sequence({literal("."), digits});
        auto exp = sequence({choice({literal("e"), literal("E")}), optional(chars("[+-]", 1, 1)), digits});
        return sequence({optional(literal("-")), int_part, optional(frac), optional(exp), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_string() {
    return rule("json-string", [this]() {
        return sequence({literal("\""), json_string_content(), literal("\""), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_bool() {
    return rule("json-bool", [this]() {
        return sequence({choice({literal("true"), literal("false")}), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_null() {
    return rule("json-null", [this]() {
        return sequence({literal("null"), space()});
    });
}

common_peg_parser common_peg_parser_builder::json_object() {
    return rule("json-object", [this]() {
        auto ws = space();
        auto member = sequence({json_string(), ws, literal(":"), ws, json()});
        auto members = sequence({member, zero_or_more(sequence({ws, literal(","), ws, member}))});
        return sequence({
            literal("{"),
            ws,
            choice({
                literal("}"),
                sequence({members, ws, literal("}")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json_array() {
    return rule("json-array", [this]() {
        auto ws = space();
        auto elements = sequence({json(), zero_or_more(sequence({literal(","), ws, json()}))});
        return sequence({
            literal("["),
            ws,
            choice({
                literal("]"),
                sequence({elements, ws, literal("]")})
            }),
            ws
        });
    });
}

common_peg_parser common_peg_parser_builder::json() {
    return rule("json-value", [this]() {
        return choice({
            json_object(),
            json_array(),
            json_string(),
            json_number(),
            json_bool(),
            json_null()
        });
    });
}

common_peg_parser common_peg_parser_builder::json_string_content() {
    return wrap(arena_.add_parser(common_peg_json_string_parser{}));
}

common_peg_parser common_peg_parser_builder::json_member(const std::string & key, const common_peg_parser & p) {
    auto ws = space();
    return sequence({
        literal("\"" + key + "\""),
        ws,
        literal(":"),
        ws,
        p,
    });
}


static std::string gbnf_escape_char_class(char c) {
    switch (c) {
        case '\n': return "\\n";
        case '\t': return "\\t";
        case '\r': return "\\r";
        case '\\': return "\\\\";
        case ']':  return "\\]";
        case '[':  return "\\[";
        default:   return std::string(1, c);
    }
}

static std::string gbnf_excluding_pattern(const std::vector<std::string> & strings) {
    trie matcher(strings);
    auto pieces = matcher.collect_prefix_and_next();

    std::string pattern;
    for (size_t i = 0; i < pieces.size(); ++i) {
        if (i > 0) {
            pattern += " | ";
        }

        const auto & pre = pieces[i].prefix;
        const auto & chars = pieces[i].next_chars;

        std::string cls;
        cls.reserve(chars.size());
        for (const auto & ch : chars) {
            cls += gbnf_escape_char_class(ch);
        }

        if (!pre.empty()) {
            pattern += gbnf_format_literal(pre) + " [^" + cls + "]";
        } else {
            pattern += "[^" + cls + "]";
        }
    }

    return "(" + pattern + ")*";
}

static std::unordered_set<std::string> collect_reachable_rules(
    const common_peg_arena & arena,
    const common_peg_parser_id & rule
) {
    std::unordered_set<std::string> reachable;
    std::unordered_set<std::string> visited;

    std::function<void(common_peg_parser_id)> visit = [&](common_peg_parser_id id) {
        const auto & parser = arena.get(id);

        std::visit([&](const auto & p) {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser> ||
                          std::is_same_v<T, common_peg_until_parser> ||
                          std::is_same_v<T, common_peg_literal_parser> ||
                          std::is_same_v<T, common_peg_chars_parser> ||
                          std::is_same_v<T, common_peg_space_parser> ||
                          std::is_same_v<T, common_peg_any_parser> ||
                          std::is_same_v<T, common_peg_json_string_parser>) {
                // These parsers do not have any children
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                for (auto child : p.children) {
                    visit(child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser> ||
                                 std::is_same_v<T, common_peg_and_parser> ||
                                 std::is_same_v<T, common_peg_not_parser> ||
                                 std::is_same_v<T, common_peg_tag_parser> ||
                                 std::is_same_v<T, common_peg_atomic_parser> ||
                                 std::is_same_v<T, common_peg_schema_parser>) {
                visit(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                if (visited.find(p.name) == visited.end()) {
                    visited.insert(p.name);
                    reachable.insert(p.name);
                    visit(p.child);
                }
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Traverse rules so we pick up everything
                auto referenced_rule = arena.get_rule(p.name);
                visit(referenced_rule);
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    visit(rule);
    return reachable;
}

// GBNF generation implementation
void common_peg_arena::build_grammar(const common_grammar_builder & builder, bool lazy) const {
    // Generate GBNF for a parser
    std::function<std::string(common_peg_parser_id)> to_gbnf = [&](common_peg_parser_id id) -> std::string {
        const auto & parser = parsers_.at(id);

        return std::visit([&](const auto & p) -> std::string {
            using T = std::decay_t<decltype(p)>;

            if constexpr (std::is_same_v<T, common_peg_epsilon_parser> ||
                          std::is_same_v<T, common_peg_start_parser> ||
                          std::is_same_v<T, common_peg_end_parser>) {
                return "";
            } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
                return gbnf_format_literal(p.literal);
            } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    if (!s.empty()) {
                        s += " ";
                    }
                    auto child_gbnf = to_gbnf(child);
                    const auto & child_parser = parsers_.at(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                        std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
                std::string s;
                for (const auto & child : p.children) {
                    if (!s.empty()) {
                        s += " | ";
                    }
                    auto child_gbnf = to_gbnf(child);
                    const auto & child_parser = parsers_.at(child);
                    if (std::holds_alternative<common_peg_choice_parser>(child_parser)) {
                        s += "(" + child_gbnf + ")";
                    } else {
                        s += child_gbnf;
                    }
                }
                return s;
            } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
                auto child_gbnf = to_gbnf(p.child);
                const auto & child_parser = parsers_.at(p.child);
                if (std::holds_alternative<common_peg_choice_parser>(child_parser) ||
                    std::holds_alternative<common_peg_sequence_parser>(child_parser)) {
                    child_gbnf = "(" + child_gbnf + ")";
                }
                if (p.min_count == 0 && p.max_count == 1) {
                    return child_gbnf + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return child_gbnf + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return child_gbnf + "+";
                }
                if (p.max_count == -1) {
                    return child_gbnf + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return child_gbnf;
                    }
                    return child_gbnf + "{" + std::to_string(p.min_count) + "}";
                }
                return child_gbnf + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_and_parser> || std::is_same_v<T, common_peg_not_parser>) {
                return "";  // Lookahead not supported in GBNF
            } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
                return ".";
            } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
                return "space";
            } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
                std::string result = p.pattern;
                if (p.min_count == 0 && p.max_count == 1) {
                    return result + "?";
                }
                if (p.min_count == 0 && p.max_count == -1) {
                    return result + "*";
                }
                if (p.min_count == 1 && p.max_count == -1) {
                    return result + "+";
                }
                if (p.max_count == -1) {
                    return result + "{" + std::to_string(p.min_count) + ",}";
                }
                if (p.min_count == p.max_count) {
                    if (p.min_count == 1) {
                        return result;
                    }
                    return result + "{" + std::to_string(p.min_count) + "}";
                }
                return result + "{" + std::to_string(p.min_count) + "," + std::to_string(p.max_count) + "}";
            } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
                return R"(( [^"\\] | "\\" ( ["\\/ bfnrt] | "u" [0-9a-fA-F]{4} ) )*)";
            } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
                if (p.delimiters.empty()) {
                    return ".*";
                }
                return gbnf_excluding_pattern(p.delimiters);
            } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
                if (p.schema) {
                    if (p.raw && p.schema->contains("type") && p.schema->at("type").is_string() && p.schema->at("type") == "string") {
                        // TODO: Implement more comprehensive grammar generation for raw strings.
                        // For now, use the grammar emitted from the underlying parser.
                        return to_gbnf(p.child);
                    }
                    return builder.add_schema(p.name, *p.schema);
                }
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
                // Refs should not exist after flattening, but kept just in case
                return p.name;
            } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
                return to_gbnf(p.child);
            } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
                return to_gbnf(p.child);
            } else {
                static_assert(is_always_false_v<T>);
            }
        }, parser);
    };

    // Collect reachable rules
    std::unordered_set<std::string> reachable_rules;

    if (lazy) {
        // Collect rules reachable from trigger rules
        for (const auto & [name, id] : rules_) {
            const auto & parser = parsers_.at(id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    // Mark trigger as reachable and visit it
                    reachable_rules.insert(name);
                    auto add_rules = collect_reachable_rules(*this, id);
                    reachable_rules.insert(add_rules.begin(), add_rules.end());
                }
            }
        }
    } else {
        // Collect rules reachable from root
        reachable_rules = collect_reachable_rules(*this, root_);
    }

    // Create GBNF rules for all reachable rules
    for (const auto & [name, rule_id] : rules_) {
        if (reachable_rules.find(name) == reachable_rules.end()) {
            continue;
        }

        const auto & parser = parsers_.at(rule_id);
        if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
            builder.add_rule(rule->name, to_gbnf(rule->child));
        }
    }

    if (lazy) {
        // Generate root rule from trigger rules only
        std::vector<std::string> trigger_names;
        for (const auto & [name, rule_id] : rules_) {
            const auto & parser = parsers_.at(rule_id);
            if (auto rule = std::get_if<common_peg_rule_parser>(&parser)) {
                if (rule->trigger) {
                    trigger_names.push_back(rule->name);
                }
            }
        }

        // Sort for predictable order
        std::sort(trigger_names.begin(), trigger_names.end());
        builder.add_rule("root", string_join(trigger_names, " | "));
    } else if (root_ != COMMON_PEG_INVALID_PARSER_ID) {
        builder.add_rule("root", to_gbnf(root_));
    }
}

static nlohmann::json serialize_parser_variant(const common_peg_parser_variant & variant) {
    using json = nlohmann::json;

    return std::visit([](const auto & p) -> json {
        using T = std::decay_t<decltype(p)>;

        if constexpr (std::is_same_v<T, common_peg_epsilon_parser>) {
            return json{{"type", "epsilon"}};
        } else if constexpr (std::is_same_v<T, common_peg_start_parser>) {
            return json{{"type", "start"}};
        } else if constexpr (std::is_same_v<T, common_peg_end_parser>) {
            return json{{"type", "end"}};
        } else if constexpr (std::is_same_v<T, common_peg_literal_parser>) {
            return json{{"type", "literal"}, {"literal", p.literal}};
        } else if constexpr (std::is_same_v<T, common_peg_sequence_parser>) {
            return json{{"type", "sequence"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_choice_parser>) {
            return json{{"type", "choice"}, {"children", p.children}};
        } else if constexpr (std::is_same_v<T, common_peg_repetition_parser>) {
            return json{
                {"type", "repetition"},
                {"child", p.child},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_and_parser>) {
            return json{{"type", "and"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_not_parser>) {
            return json{{"type", "not"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_any_parser>) {
            return json{{"type", "any"}};
        } else if constexpr (std::is_same_v<T, common_peg_space_parser>) {
            return json{{"type", "space"}};
        } else if constexpr (std::is_same_v<T, common_peg_chars_parser>) {
            json ranges = json::array();
            for (const auto & range : p.ranges) {
                ranges.push_back({{"start", range.start}, {"end", range.end}});
            }
            return json{
                {"type", "chars"},
                {"pattern", p.pattern},
                {"ranges", ranges},
                {"negated", p.negated},
                {"min_count", p.min_count},
                {"max_count", p.max_count}
            };
        } else if constexpr (std::is_same_v<T, common_peg_json_string_parser>) {
            return json{{"type", "json_string"}};
        } else if constexpr (std::is_same_v<T, common_peg_until_parser>) {
            return json{{"type", "until"}, {"delimiters", p.delimiters}};
        } else if constexpr (std::is_same_v<T, common_peg_schema_parser>) {
            return json{
                {"type", "schema"},
                {"child", p.child},
                {"name", p.name},
                {"schema", p.schema ? *p.schema : nullptr},
                {"raw", p.raw}
            };
        } else if constexpr (std::is_same_v<T, common_peg_rule_parser>) {
            return json{
                {"type", "rule"},
                {"name", p.name},
                {"child", p.child},
                {"trigger", p.trigger}
            };
        } else if constexpr (std::is_same_v<T, common_peg_ref_parser>) {
            return json{{"type", "ref"}, {"name", p.name}};
        } else if constexpr (std::is_same_v<T, common_peg_atomic_parser>) {
            return json{{"type", "atomic"}, {"child", p.child}};
        } else if constexpr (std::is_same_v<T, common_peg_tag_parser>) {
            return json{
                {"type", "tag"},
                {"child", p.child},
                {"tag", p.tag}
            };
        }
    }, variant);
}

nlohmann::json common_peg_arena::to_json() const {
    auto parsers = nlohmann::json::array();
    for (const auto & parser : parsers_) {
        parsers.push_back(serialize_parser_variant(parser));
    }
    return nlohmann::json{
        {"parsers", parsers},
        {"rules", rules_},
        {"root", root_}
    };
}

static common_peg_parser_variant deserialize_parser_variant(const nlohmann::json & j) {
    if (!j.contains("type") || !j["type"].is_string()) {
        throw std::runtime_error("Parser variant JSON missing or invalid 'type' field");
    }

    std::string type = j["type"];

    if (type == "epsilon") {
        return common_peg_epsilon_parser{};
    }
    if (type == "start") {
        return common_peg_start_parser{};
    }
    if (type == "end") {
        return common_peg_end_parser{};
    }
    if (type == "literal") {
        if (!j.contains("literal") || !j["literal"].is_string()) {
            throw std::runtime_error("literal parser missing or invalid 'literal' field");
        }
        return common_peg_literal_parser{j["literal"]};
    }
    if (type == "sequence") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("sequence parser missing or invalid 'children' field");
        }
        return common_peg_sequence_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "choice") {
        if (!j.contains("children") || !j["children"].is_array()) {
            throw std::runtime_error("choice parser missing or invalid 'children' field");
        }
        return common_peg_choice_parser{j["children"].get<std::vector<common_peg_parser_id>>()};
    }
    if (type == "repetition") {
        if (!j.contains("child") || !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("repetition parser missing required fields");
        }
        return common_peg_repetition_parser{
            j["child"].get<common_peg_parser_id>(),
            j["min_count"].get<int>(),
            j["max_count"].get<int>()
        };
    }
    if (type == "and") {
        if (!j.contains("child")) {
            throw std::runtime_error("and parser missing 'child' field");
        }
        return common_peg_and_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "not") {
        if (!j.contains("child")) {
            throw std::runtime_error("not parser missing 'child' field");
        }
        return common_peg_not_parser{j["child"].get<common_peg_parser_id>()};
    }
    if (type == "any") {
        return common_peg_any_parser{};
    }
    if (type == "space") {
        return common_peg_space_parser{};
    }
    if (type == "chars") {
        if (!j.contains("pattern") || !j.contains("ranges") || !j.contains("negated") ||
            !j.contains("min_count") || !j.contains("max_count")) {
            throw std::runtime_error("chars parser missing required fields");
        }
        common_peg_chars_parser parser;
        parser.pattern = j["pattern"];
        parser.negated = j["negated"];
        parser.min_count = j["min_count"];
        parser.max_count = j["max_count"];
        for (const auto & range_json : j["ranges"]) {
            if (!range_json.contains("start") || !range_json.contains("end")) {
                throw std::runtime_error("char_range missing 'start' or 'end' field");
            }
            parser.ranges.push_back({
                range_json["start"].get<uint32_t>(),
                range_json["end"].get<uint32_t>()
            });
        }
        return parser;
    }
    if (type == "json_string") {
        return common_peg_json_string_parser{};
    }
    if (type == "until") {
        if (!j.contains("delimiters") || !j["delimiters"].is_array()) {
            throw std::runtime_error("until parser missing or invalid 'delimiters' field");
        }
        return common_peg_until_parser{j["delimiters"].get<std::vector<std::string>>()};
    }
    if (type == "schema") {
        if (!j.contains("child") || !j.contains("name") || !j.contains("schema") || !j.contains("raw")) {
            throw std::runtime_error("schema parser missing required fields");
        }
        common_peg_schema_parser parser;
        parser.child = j["child"].get<common_peg_parser_id>();
        parser.name = j["name"];
        if (!j["schema"].is_null()) {
            parser.schema = std::make_shared<nlohmann::ordered_json>(j["schema"]);
        }
        parser.raw = j["raw"].get<bool>();
        return parser;
    }
    if (type == "rule") {
        if (!j.contains("name") || !j.contains("child") || !j.contains("trigger")) {
            throw std::runtime_error("rule parser missing required fields");
        }
        return common_peg_rule_parser{
            j["name"].get<std::string>(),
            j["child"].get<common_peg_parser_id>(),
            j["trigger"].get<bool>()
        };
    }
    if (type == "ref") {
        if (!j.contains("name") || !j["name"].is_string()) {
            throw std::runtime_error("ref parser missing or invalid 'name' field");
        }
        return common_peg_ref_parser{j["name"]};
    }
    if (type == "atomic") {
        if (!j.contains("child")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_atomic_parser{
            j["child"].get<common_peg_parser_id>(),
        };
    }
    if (type == "tag") {
        if (!j.contains("child") || !j.contains("tag")) {
            throw std::runtime_error("tag parser missing required fields");
        }
        return common_peg_tag_parser{
            j["child"].get<common_peg_parser_id>(),
            j["tag"].get<std::string>(),
        };
    }

    throw std::runtime_error("Unknown parser type: " + type);
}

common_peg_arena common_peg_arena::from_json(const nlohmann::json & j) {
    if (!j.contains("parsers") || !j["parsers"].is_array()) {
        throw std::runtime_error("JSON missing or invalid 'parsers' array");
    }
    if (!j.contains("rules") || !j["rules"].is_object()) {
        throw std::runtime_error("JSON missing or invalid 'rules' object");
    }
    if (!j.contains("root")) {
        throw std::runtime_error("JSON missing 'root' field");
    }

    common_peg_arena arena;

    const auto & parsers_json = j["parsers"];
    arena.parsers_.reserve(parsers_json.size());
    for (const auto & parser_json : parsers_json) {
        arena.parsers_.push_back(deserialize_parser_variant(parser_json));
    }

    arena.rules_ = j["rules"].get<std::unordered_map<std::string, common_peg_parser_id>>();

    for (const auto & [name, id] : arena.rules_) {
        if (id >= arena.parsers_.size()) {
            throw std::runtime_error("Rule '" + name + "' references invalid parser ID: " + std::to_string(id));
        }
    }

    arena.root_ = j["root"].get<common_peg_parser_id>();
    if (arena.root_ != COMMON_PEG_INVALID_PARSER_ID && arena.root_ >= arena.parsers_.size()) {
        throw std::runtime_error("Root references invalid parser ID: " + std::to_string(arena.root_));
    }

    return arena;
}

std::string common_peg_arena::save() const {
    return to_json().dump();
}

void common_peg_arena::load(const std::string & data) {
    *this = from_json(nlohmann::json::parse(data));
}

common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn) {
    common_peg_parser_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

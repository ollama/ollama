#pragma once

#include <regex>
#include <string>

enum common_regex_match_type {
    COMMON_REGEX_MATCH_TYPE_NONE,
    COMMON_REGEX_MATCH_TYPE_PARTIAL,
    COMMON_REGEX_MATCH_TYPE_FULL,
};

struct common_string_range {
    size_t begin;
    size_t end;
    common_string_range(size_t begin, size_t end) : begin(begin), end(end) {
        if (begin > end) {
            throw std::runtime_error("Invalid range");
        }
    }
    // prevent default ctor
    common_string_range() = delete;
    bool empty() const {
        return begin == end;
    }
    bool operator==(const common_string_range & other) const {
        return begin == other.begin && end == other.end;
    }
};

struct common_regex_match {
    common_regex_match_type type = COMMON_REGEX_MATCH_TYPE_NONE;
    std::vector<common_string_range> groups;

    bool operator==(const common_regex_match & other) const {
        return type == other.type && groups == other.groups;
    }
    bool operator!=(const common_regex_match & other) const {
        return !(*this == other);
    }
};

class common_regex {
    std::string pattern;
    std::regex rx;
    std::regex rx_reversed_partial;

  public:
    explicit common_regex(const std::string & pattern);

    common_regex_match search(const std::string & input, size_t pos, bool as_match = false) const;

    const std::string & str() const { return pattern; }
};

// For testing only (pretty print of failures).
std::string regex_to_reversed_partial_regex(const std::string & pattern);

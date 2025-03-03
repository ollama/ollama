#pragma once

#include "ggml.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

std::string json_schema_to_grammar(const nlohmann::ordered_json & schema,
                                   bool force_gbnf = false);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
    bool compact_spaces = false;
};

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});

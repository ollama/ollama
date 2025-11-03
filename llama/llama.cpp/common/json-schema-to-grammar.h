#pragma once

#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <string>

std::string json_schema_to_grammar(const nlohmann::ordered_json & schema,
                                   bool force_gbnf = false);

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});

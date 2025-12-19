#pragma once

#include <nlohmann/json_fwd.hpp>

#include <functional>
#include <memory>
#include <string>

std::string json_schema_to_grammar(const nlohmann::ordered_json & schema,
                                   bool force_gbnf = false);

class common_schema_converter;

// Probes a JSON schema to extract information about its structure and type constraints.
class common_schema_info {
    std::unique_ptr<common_schema_converter> impl_;

  public:
    common_schema_info();
    ~common_schema_info();

    common_schema_info(const common_schema_info &) = delete;
    common_schema_info & operator=(const common_schema_info &) = delete;
    common_schema_info(common_schema_info &&) noexcept;
    common_schema_info & operator=(common_schema_info &&) noexcept;

    void resolve_refs(nlohmann::ordered_json & schema);
    bool resolves_to_string(const nlohmann::ordered_json & schema);
};

struct common_grammar_builder {
    std::function<std::string(const std::string &, const std::string &)> add_rule;
    std::function<std::string(const std::string &, const nlohmann::ordered_json &)> add_schema;
    std::function<void(nlohmann::ordered_json &)> resolve_refs;
};

struct common_grammar_options {
    bool dotall = false;
};

std::string gbnf_format_literal(const std::string & literal);

std::string build_grammar(const std::function<void(const common_grammar_builder &)> & cb, const common_grammar_options & options = {});

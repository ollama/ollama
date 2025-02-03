#pragma once

#include "ggml.h"
// Change JSON_ASSERT from assert() to GGML_ASSERT:
#define JSON_ASSERT GGML_ASSERT
#include "json.hpp"

std::string json_schema_to_grammar(const nlohmann::ordered_json& schema);

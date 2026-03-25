#pragma once

#include <nlohmann/json.hpp>

// Healing marker (empty if the JSON was fully parsed / wasn't healed).
struct common_healing_marker {
    // Raw marker.
    std::string marker;

    // Cutting the `common_json.json.dump()` string at the (only) occurrence of this marker should yield the original partial JSON string (modulo spaces / if it had the same dump format).
    std::string json_dump_marker;
};

// Represents a parsed JSON object, with its optional healing marker (a JSON dump fragment that can be used to find the position of healing in the JSON dump string)
struct common_json {
    nlohmann::ordered_json json;

    common_healing_marker healing_marker;
};

// Parse the JSON string, healing (closing) any partial JSON if `healing_marker` is not empty.
//
// Healing completes partial JSON strings by adding a (possibly modified) healing marker, then whatever is needed to close the JSON.
// This allows to parse the resulting healed JSON string, yet be able to cut it again if needed at the healing marker.
// (this is used when parsing JSON outputs from the models, then crafting partial JSONs for the partial tool calls in OAI format).
//
// For instance, parsing `{` with a healing marker `foo` will produce a healed JSON `{"foo":1}`, w/ json_dump_marker = `"foo"` (which can be used to break the JSON again).
bool common_json_parse(
    const std::string & input,
    const std::string & healing_marker,
    common_json & out);

// Parse the JSON string (see overload above), but advancing an iterator to the end of the input when the (potentially partial) parsing succeeds.
bool common_json_parse(
    std::string::const_iterator & it,
    const std::string::const_iterator & end,
    const std::string & healing_marker,
    common_json & out);

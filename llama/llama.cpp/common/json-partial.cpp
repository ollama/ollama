#include "json-partial.h"

#include "log.h"

#include <nlohmann/json.hpp>

#include <string>

using json = nlohmann::ordered_json;

enum common_json_stack_element_type {
    COMMON_JSON_STACK_ELEMENT_OBJECT,
    COMMON_JSON_STACK_ELEMENT_KEY,
    COMMON_JSON_STACK_ELEMENT_ARRAY,
};

struct common_json_stack_element {
    common_json_stack_element_type type;
    std::string key;
};

bool common_json_parse(
    const std::string & input,
    const std::string & healing_marker,
    common_json & out)
{
    std::string::const_iterator it = input.begin();
    const auto end = input.end();
    return common_json_parse(it, end, healing_marker, out);
}

bool common_json_parse(
    std::string::const_iterator & it,
    const std::string::const_iterator & end,
    const std::string & healing_marker,
    common_json & out)
{
    // // https://json.nlohmann.me/features/parsing/sax_interface/
    struct json_error_locator : public nlohmann::json_sax<json> {
        std::size_t position;
        bool found_error;
        std::string last_token;
        std::string exception_message;
        std::vector<common_json_stack_element> stack;

        json_error_locator() : position(0), found_error(false) {}

        bool parse_error(std::size_t position, const std::string & last_token, const json::exception & ex) override { // NOLINT
            this->position = position - 1;
            this->found_error = true;
            this->last_token = last_token;
            this->exception_message = ex.what();
            return false;
        }
        void close_value() {
            if (!stack.empty() && (stack.back().type == COMMON_JSON_STACK_ELEMENT_KEY)) {
                stack.pop_back();
            }
        }
        bool null() override { // NOLINT
            close_value();
            return true;
        }
        bool boolean(bool) override { // NOLINT
            close_value();
            return true;
        }
        bool number_integer(number_integer_t) override { // NOLINT
            close_value();
            return true;
        }
        bool number_unsigned(number_unsigned_t) override { // NOLINT
            close_value();
            return true;
        }
        bool number_float(number_float_t, const string_t &) override { // NOLINT
            close_value();
            return true;
        }
        bool string(string_t &) override { // NOLINT
            close_value();
            return true;
        }
        bool binary(binary_t &) override { // NOLINT
            close_value();
            return true;
        }
        bool start_object(std::size_t) override { // NOLINT
            stack.push_back({COMMON_JSON_STACK_ELEMENT_OBJECT, ""});
            return true;
        }
        bool end_object() override {
            GGML_ASSERT(!stack.empty() && stack.back().type == COMMON_JSON_STACK_ELEMENT_OBJECT);
            stack.pop_back();
            close_value();
            return true;
        }
        bool key(string_t & key) override { // NOLINT
            stack.push_back({COMMON_JSON_STACK_ELEMENT_KEY, key});
            return true;
        }
        bool start_array(std::size_t) override { // NOLINT
            stack.push_back({COMMON_JSON_STACK_ELEMENT_ARRAY, ""});
            return true;
        }
        bool end_array() override {
            GGML_ASSERT(!stack.empty() && stack.back().type == COMMON_JSON_STACK_ELEMENT_ARRAY);
            stack.pop_back();
            close_value();
            return true;
        }
    };
    json_error_locator err_loc;
    auto start = it;
    json::sax_parse(it, end, &err_loc);

    if (err_loc.found_error) {
        it = start;
        auto temptative_end = it + err_loc.position;
        // LOG_DBG("Error at position %zu (is_end = %s): %s\n", err_loc.position, temptative_end == end ? "true" : "false", err_loc.exception_message.c_str());

        auto input = std::string(it, temptative_end);
        try {
            out.json = json::parse(input);
            // out.json = json::parse(it, temptative_end);
            it = temptative_end;
            return true;
        } catch (const std::exception & ex) {
            // No, needs healing.
            LOG_DBG("Failed to parse up to error: %s: <<<%s>>>\n", ex.what(), std::string(it, temptative_end).c_str());
        }
        auto can_parse = [](const std::string & str) {
            try {
                auto _ = json::parse(str); // NOLINT
                return true;
            } catch (const std::exception &) {
                return false;
            }
        };
        if (!healing_marker.empty() && !err_loc.stack.empty()) {
            std::string str(it, temptative_end);
            auto last_non_sp_pos = str.find_last_not_of(" \n\r\t");
            if (last_non_sp_pos == std::string::npos) {
                throw std::runtime_error("Cannot heal a truncated JSON that stopped in an unknown location");
            }
            auto last_non_sp_char = str[last_non_sp_pos];
            // Used to detect stops on a number, which may not be complete.
            auto was_maybe_number = [&]() {
                if (!str.empty() && std::isspace(str.back())) {
                    return false;
                }
                return std::isdigit(last_non_sp_char) ||
                    last_non_sp_char == '.' ||
                    last_non_sp_char == 'e' ||
                    last_non_sp_char == 'E' ||
                    last_non_sp_char == '-';
            };

            std::string closing;
            for (size_t i = err_loc.stack.size(); i > 0; i--) {
                auto & el = err_loc.stack[i - 1];
                if (el.type == COMMON_JSON_STACK_ELEMENT_OBJECT) {
                    closing += "}";
                } else if (el.type == COMMON_JSON_STACK_ELEMENT_ARRAY) {
                    closing += "]";
                } else if (el.type != COMMON_JSON_STACK_ELEMENT_KEY) {
                    throw std::runtime_error("Unexpected stack element type");
                }
            }

            const auto & magic_seed = out.healing_marker.marker = healing_marker;//"$llama.cpp.json$";

            if (err_loc.stack.back().type == COMMON_JSON_STACK_ELEMENT_KEY) {
                // We're inside an object value
                if (last_non_sp_char == ':' && can_parse(str + "1" + closing)) {
                    // Was about to create an object value
                    str += (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\"" + closing;
                } else if (can_parse(str + ": 1" + closing)) {
                    str += (out.healing_marker.json_dump_marker = ":\"" + magic_seed) + "\"" + closing;
                } else if (last_non_sp_char == '{' && can_parse(str + closing)) {
                    // Was about to create an object
                    str += (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\": 1" + closing;
                } else if (can_parse(str + "\"" + closing)) {
                    // Was inside an object value string
                    str += (out.healing_marker.json_dump_marker = magic_seed) + "\"" + closing;
                } else if (str[str.length() - 1] == '\\' && can_parse(str + "\\\"" + closing)) {
                    // Was inside an object value string after an escape
                    str += (out.healing_marker.json_dump_marker = "\\" + magic_seed) + "\"" + closing;
                } else {
                    // find last :
                    auto last_pos = str.find_last_of(':');
                    if (last_pos == std::string::npos) {
                        throw std::runtime_error("Cannot heal a truncated JSON that stopped in an unknown location");
                    }
                    // Cutting back to opening : for object value
                    str = str.substr(0, last_pos + 1) + (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\"" + closing;
                }
            } else if (err_loc.stack.back().type == COMMON_JSON_STACK_ELEMENT_ARRAY) {
                if ((last_non_sp_char == ',' || last_non_sp_char == '[') && can_parse(str + "1" + closing)) {
                    // Was about to create an array value
                    str += (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\"" + closing;
                } else if (can_parse(str + "\"" + closing)) {
                    // Was inside an array value string
                    str += (out.healing_marker.json_dump_marker = magic_seed) + "\"" + closing;
                } else if (str[str.length() - 1] == '\\' && can_parse(str + "\\\"" + closing)) {
                    // Was inside an array value string after an escape
                    str += (out.healing_marker.json_dump_marker = "\\" + magic_seed) + "\"" + closing;
                } else if (!was_maybe_number() && can_parse(str + ", 1" + closing)) {
                    // Had just finished a value
                    str += (out.healing_marker.json_dump_marker = ",\"" + magic_seed) + "\"" + closing;
                } else {
                    auto last_pos = str.find_last_of("[,");
                    if (last_pos == std::string::npos) {
                        throw std::runtime_error("Cannot heal a truncated JSON array stopped in an unknown location");
                    }
                    // Cutting back to last [ or , for array value
                    str = str.substr(0, last_pos + 1) + (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\"" + closing;
                }
            } else if (err_loc.stack.back().type == COMMON_JSON_STACK_ELEMENT_OBJECT) {
                if ((last_non_sp_char == '{' && can_parse(str + closing)) ||
                        (last_non_sp_char == ',' && can_parse(str + "\"\": 1" + closing))) {
                    // Was about to create an object key+value
                    str += (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\": 1" + closing;
                } else if (!was_maybe_number() && can_parse(str + ",\"\": 1" + closing)) {
                    // Was about to create an object key+value
                    str += (out.healing_marker.json_dump_marker = ",\"" + magic_seed) + "\": 1" + closing;
                } else if (can_parse(str + "\": 1" + closing)) {
                    // Was inside an object key string
                    str += (out.healing_marker.json_dump_marker = magic_seed) + "\": 1" + closing;
                } else if (str[str.length() - 1] == '\\' && can_parse(str + "\\\": 1" + closing)) {
                    // Was inside an object key string after an escape
                    str += (out.healing_marker.json_dump_marker = "\\" + magic_seed) + "\": 1" + closing;
                } else {
                    auto last_pos = str.find_last_of(':');
                    if (last_pos == std::string::npos) {
                        throw std::runtime_error("Cannot heal a truncated JSON object stopped in an unknown location");
                    }
                    // fprintf(stderr, "Cutting back to last : for object key+value\n");
                    str = str.substr(0, last_pos + 1) + (out.healing_marker.json_dump_marker = "\"" + magic_seed) + "\"" + closing;
                }
            } else {
                throw std::runtime_error("Cannot heal a truncated JSON object stopped in an unknown location");
            }
            // fprintf(stderr, "HEALED:\nSTRING <<<\n%s\n>>>\n\nmagic_cut: <<<\n%s\n>>>\n\n", str.c_str(), out.healing_marker.json_dump_marker.c_str());
            out.json = json::parse(str);
            it = temptative_end;
            return true;
        }
        // TODO: handle unclosed top-level primitive if the stack was empty but we got an error (e.g. "tru", "\"", etc...)
        // fprintf(stderr, "Closing: TODO\n");
        return false;
    }
    out.json = json::parse(it, end);
    it = end;
    return true;
}

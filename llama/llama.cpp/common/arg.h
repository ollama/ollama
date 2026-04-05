#pragma once

#include "common.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include <cstring>

//
// CLI argument parsing
//

struct common_arg {
    std::set<enum llama_example> examples = {LLAMA_EXAMPLE_COMMON};
    std::set<enum llama_example> excludes = {};
    std::vector<const char *> args;
    std::vector<const char *> args_neg;  // for negated args like --no-xxx
    const char * value_hint   = nullptr; // help text or example for arg value
    const char * value_hint_2 = nullptr; // for second arg value
    const char * env          = nullptr;
    std::string help;
    bool is_sparam = false; // is current arg a sampling param?
    void (*handler_void)   (common_params & params) = nullptr;
    void (*handler_string) (common_params & params, const std::string &) = nullptr;
    void (*handler_str_str)(common_params & params, const std::string &, const std::string &) = nullptr;
    void (*handler_int)    (common_params & params, int) = nullptr;
    void (*handler_bool)   (common_params & params, bool) = nullptr;

    common_arg() = default;

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &)
    ) : args(args), value_hint(value_hint), help(help), handler_string(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const std::string & help,
        void (*handler)(common_params & params, int)
    ) : args(args), value_hint(value_hint), help(help), handler_int(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::string & help,
        void (*handler)(common_params & params)
    ) : args(args), help(help), handler_void(handler) {}

    common_arg(
        const std::initializer_list<const char *> & args,
        const std::initializer_list<const char *> & args_neg,
        const std::string & help,
        void (*handler)(common_params & params, bool)
    ) : args(args), args_neg(args_neg), help(help), handler_bool(handler) {}

    // support 2 values for arg
    common_arg(
        const std::initializer_list<const char *> & args,
        const char * value_hint,
        const char * value_hint_2,
        const std::string & help,
        void (*handler)(common_params & params, const std::string &, const std::string &)
    ) : args(args), value_hint(value_hint), value_hint_2(value_hint_2), help(help), handler_str_str(handler) {}

    common_arg & set_examples(std::initializer_list<enum llama_example> examples);
    common_arg & set_excludes(std::initializer_list<enum llama_example> excludes);
    common_arg & set_env(const char * env);
    common_arg & set_sparam();
    bool in_example(enum llama_example ex);
    bool is_exclude(enum llama_example ex);
    bool get_value_from_env(std::string & output) const;
    bool has_value_from_env() const;
    std::string to_string() const;

    // for using as key in std::map
    bool operator<(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) < 0;
    }
    bool operator==(const common_arg& other) const {
        if (args.empty() || other.args.empty()) {
            return false;
        }
        return strcmp(args[0], other.args[0]) == 0;
    }

    // get all args and env vars (including negated args/env)
    std::vector<std::string> get_args() const;
    std::vector<std::string> get_env() const;
};

namespace common_arg_utils {
    bool is_truthy(const std::string & value);
    bool is_falsey(const std::string & value);
    bool is_autoy(const std::string & value);
}

struct common_params_context {
    enum llama_example ex = LLAMA_EXAMPLE_COMMON;
    common_params & params;
    std::vector<common_arg> options;
    void(*print_usage)(int, char **) = nullptr;
    common_params_context(common_params & params) : params(params) {}
};

// parse input arguments from CLI
// if one argument has invalid value, it will automatically display usage of the specific argument (and not the full usage message)
bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

// parse input arguments from CLI into a map
// TODO: support repeated args in the future
bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map);

// initialize argument parser context - used by test-arg-parser and preset
common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **) = nullptr);

struct common_remote_params {
    std::vector<std::string> headers;
    long timeout = 0; // CURLOPT_TIMEOUT, in seconds ; 0 means no timeout
    long max_size = 0; // max size of the response ; unlimited if 0 ; max is 2GB
};
// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);

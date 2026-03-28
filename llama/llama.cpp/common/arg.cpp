#include "arg.h"

#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "log.h"
#include "sampling.h"
#include "download.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <cstdarg>
#include <fstream>
#include <list>
#include <regex>
#include <set>
#include <string>
#include <thread> // for hardware_concurrency
#include <vector>

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

#define LLAMA_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

using json = nlohmann::ordered_json;
using namespace common_arg_utils;

static std::initializer_list<enum llama_example> mmproj_examples = {
    LLAMA_EXAMPLE_MTMD,
    LLAMA_EXAMPLE_SERVER,
    LLAMA_EXAMPLE_CLI,
};

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

static const std::vector<common_arg> & get_common_arg_defs() {
    static const std::vector<common_arg> options = [] {
        common_params params;
        auto ctx = common_params_parser_init(params, LLAMA_EXAMPLE_SERVER, nullptr);
        return ctx.options;
    }();
    return options;
}

common_arg & common_arg::set_examples(std::initializer_list<enum llama_example> examples) {
    this->examples = examples;
    return *this;
}

common_arg & common_arg::set_excludes(std::initializer_list<enum llama_example> excludes) {
    this->excludes = excludes;
    return *this;
}

common_arg & common_arg::set_env(const char * env) {
    help = help + "\n(env: " + env + ")";
    this->env = env;
    return *this;
}

common_arg & common_arg::set_sparam() {
    is_sparam = true;
    return *this;
}

bool common_arg::in_example(enum llama_example ex) {
    return examples.find(ex) != examples.end();
}

bool common_arg::is_exclude(enum llama_example ex) {
    return excludes.find(ex) != excludes.end();
}

bool common_arg::get_value_from_env(std::string & output) const {
    if (env == nullptr) return false;
    if (!args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        char * neg_value = std::getenv(neg_env.c_str());
        if (neg_value) {
            output = "0"; // falsey
            return true;
        }
    }
    char * value = std::getenv(env);
    if (value) {
        output = value;
        return true;
    }
    return false;
}

bool common_arg::has_value_from_env() const {
    if (env != nullptr && !args_neg.empty()) {
        // for compatibility, we need to check LLAMA_ARG_NO_ env as well
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        if (std::getenv(neg_env.c_str())) {
            return true;
        }
    }
    return env != nullptr && std::getenv(env);
}

static std::vector<std::string> break_str_into_lines(std::string input, size_t max_char_per_line) {
    std::vector<std::string> result;
    std::istringstream iss(input);
    std::string line;
    auto add_line = [&](const std::string& l) {
        if (l.length() <= max_char_per_line) {
            result.push_back(l);
        } else {
            std::istringstream line_stream(l);
            std::string word, current_line;
            while (line_stream >> word) {
                if (current_line.length() + !current_line.empty() + word.length() > max_char_per_line) {
                    if (!current_line.empty()) result.push_back(current_line);
                    current_line = word;
                } else {
                    current_line += (!current_line.empty() ? " " : "") + word;
                }
            }
            if (!current_line.empty()) result.push_back(current_line);
        }
    };
    while (std::getline(iss, line)) {
        add_line(line);
    }
    return result;
}

std::string common_arg::to_string() const {
    // params for printing to console
    const static int n_leading_spaces = 40;
    const static int n_char_per_line_help = 70; // TODO: detect this based on current console
    std::string leading_spaces(n_leading_spaces, ' ');

    std::ostringstream ss;
    auto all_args = get_args(); // also contains args_neg
    for (const auto & arg : all_args) {
        if (arg == all_args.front()) {
            if (all_args.size() == 1) {
                ss << arg;
            } else {
                // first arg is usually abbreviation, we need padding to make it more beautiful
                auto tmp = std::string(arg) + ", ";
                auto spaces = std::string(std::max(0, 7 - (int)tmp.size()), ' ');
                ss << tmp << spaces;
            }
        } else {
            ss << arg << (arg != all_args.back() ? ", " : "");
        }
    }
    if (value_hint) ss << " " << value_hint;
    if (value_hint_2) ss << " " << value_hint_2;
    if (ss.tellp() > n_leading_spaces - 3) {
        // current line is too long, add new line
        ss << "\n" << leading_spaces;
    } else {
        // padding between arg and help, same line
        ss << std::string(leading_spaces.size() - ss.tellp(), ' ');
    }
    const auto help_lines = break_str_into_lines(help, n_char_per_line_help);
    for (const auto & line : help_lines) {
        ss << (&line == &help_lines.front() ? "" : leading_spaces) << line << "\n";
    }
    return ss.str();
}

std::vector<std::string> common_arg::get_args() const {
    std::vector<std::string> result;
    for (const auto & arg : args) {
        result.push_back(std::string(arg));
    }
    for (const auto & arg : args_neg) {
        result.push_back(std::string(arg));
    }
    return result;
}

std::vector<std::string> common_arg::get_env() const {
    std::vector<std::string> result;
    if (env) {
        result.push_back(std::string(env));
    }
    if (!args_neg.empty() && env) {
        // for compatibility, we need to add LLAMA_ARG_NO_ variant
        std::string neg_env = env;
        string_replace_all(neg_env, "LLAMA_ARG_", "LLAMA_ARG_NO_");
        result.push_back(neg_env);
    }
    return result;
}

//
// utils
//

// Helper function to parse tensor buffer override strings
static void parse_tensor_buffer_overrides(const std::string & value, std::vector<llama_model_tensor_buft_override> & overrides) {
    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * dev = ggml_backend_dev_get(i);
        auto * buft = ggml_backend_dev_buffer_type(dev);
        if (buft) {
            buft_list[ggml_backend_buft_name(buft)] = buft;
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);

        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto & it : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(it.second));
            }
            throw std::invalid_argument("unknown buffer type");
        }
        // keep strings alive and avoid leaking memory by storing them in a static vector
        static std::list<std::string> buft_overrides;
        buft_overrides.push_back(tensor_name);
        overrides.push_back({buft_overrides.back().c_str(), buft_list.at(buffer_type)});
    }
}

struct handle_model_result {
    bool found_mmproj = false;
    common_params_model mmproj;
};

static handle_model_result common_params_handle_model(
        struct common_params_model & model,
        const std::string & bearer_token,
        bool offline) {
    handle_model_result result;
    // handle pre-fill default model path and url based on hf_repo and hf_file
    {
        if (!model.docker_repo.empty()) {  // Handle Docker URLs by resolving them to local paths
            model.path = common_docker_resolve_model(model.docker_repo);
            model.name = model.docker_repo; // set name for consistency
        } else if (!model.hf_repo.empty()) {
            // short-hand to avoid specifying --hf-file -> default it to --model
            if (model.hf_file.empty()) {
                if (model.path.empty()) {
                    auto auto_detected = common_get_hf_file(model.hf_repo, bearer_token, offline);
                    if (auto_detected.repo.empty() || auto_detected.ggufFile.empty()) {
                        exit(1); // built without CURL, error message already printed
                    }
                    model.name    = model.hf_repo;      // repo name with tag
                    model.hf_repo = auto_detected.repo; // repo name without tag
                    model.hf_file = auto_detected.ggufFile;
                    if (!auto_detected.mmprojFile.empty()) {
                        result.found_mmproj   = true;
                        result.mmproj.hf_repo = model.hf_repo;
                        result.mmproj.hf_file = auto_detected.mmprojFile;
                    }
                } else {
                    model.hf_file = model.path;
                }
            }

            std::string model_endpoint = get_model_endpoint();
            model.url = model_endpoint + model.hf_repo + "/resolve/main/" + model.hf_file;
            // make sure model path is present (for caching purposes)
            if (model.path.empty()) {
                // this is to avoid different repo having same file name, or same file name in different subdirs
                std::string filename = model.hf_repo + "_" + model.hf_file;
                // to make sure we don't have any slashes in the filename
                string_replace_all(filename, "/", "_");
                model.path = fs_get_cache_file(filename);
            }

        } else if (!model.url.empty()) {
            if (model.path.empty()) {
                auto f = string_split<std::string>(model.url, '#').front();
                f = string_split<std::string>(f, '?').front();
                model.path = fs_get_cache_file(string_split<std::string>(f, '/').back());
            }

        }
    }

    // then, download it if needed
    if (!model.url.empty()) {
        bool ok = common_download_model(model, bearer_token, offline);
        if (!ok) {
            LOG_ERR("error: failed to download model from %s\n", model.url.c_str());
            exit(1);
        }
    }

    return result;
}

const std::vector<ggml_type> kv_cache_types = {
    GGML_TYPE_F32,
    GGML_TYPE_F16,
    GGML_TYPE_BF16,
    GGML_TYPE_Q8_0,
    GGML_TYPE_Q4_0,
    GGML_TYPE_Q4_1,
    GGML_TYPE_IQ4_NL,
    GGML_TYPE_Q5_0,
    GGML_TYPE_Q5_1,
};

static ggml_type kv_cache_type_from_str(const std::string & s) {
    for (const auto & type : kv_cache_types) {
        if (ggml_type_name(type) == s) {
            return type;
        }
    }
    throw std::runtime_error("Unsupported cache type: " + s);
}

static std::string get_all_kv_cache_types() {
    std::ostringstream msg;
    for (const auto & type : kv_cache_types) {
        msg << ggml_type_name(type) << (&type == &kv_cache_types.back() ? "" : ", ");
    }
    return msg.str();
}

static bool parse_bool_value(const std::string & value) {
    if (is_truthy(value)) {
        return true;
    } else if (is_falsey(value)) {
        return false;
    } else {
        throw std::invalid_argument("invalid boolean value");
    }
}

//
// CLI argument parsing functions
//

static bool common_params_parse_ex(int argc, char ** argv, common_params_context & ctx_arg) {
    common_params & params = ctx_arg.params;

    std::unordered_map<std::string, std::pair<common_arg *, bool>> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = {&opt, /* is_positive */ true};
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = {&opt, /* is_positive */ false};
        }
    }

    // handle environment variables
    for (auto & opt : ctx_arg.options) {
        std::string value;
        if (opt.get_value_from_env(value)) {
            try {
                if (opt.handler_void && is_truthy(value)) {
                    opt.handler_void(params);
                }
                if (opt.handler_int) {
                    opt.handler_int(params, std::stoi(value));
                }
                if (opt.handler_bool) {
                    opt.handler_bool(params, parse_bool_value(value));
                }
                if (opt.handler_string) {
                    opt.handler_string(params, value);
                    continue;
                }
            } catch (std::exception & e) {
                throw std::invalid_argument(string_format(
                    "error while handling environment variable \"%s\": %s\n\n", opt.env, e.what()));
            }
        }
    }

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        auto & tmp = arg_to_options[arg];
        auto opt = *tmp.first;
        bool is_positive = tmp.second;
        if (opt.has_value_from_env()) {
            fprintf(stderr, "warn: %s environment variable is set, but will be overwritten by command line argument %s\n", opt.env, arg.c_str());
        }
        try {
            if (opt.handler_void) {
                opt.handler_void(params);
                continue;
            }
            if (opt.handler_bool) {
                opt.handler_bool(params, is_positive);
                continue;
            }

            // arg with single value
            check_arg(i);
            std::string val = argv[++i];
            if (opt.handler_int) {
                opt.handler_int(params, std::stoi(val));
                continue;
            }
            if (opt.handler_string) {
                opt.handler_string(params, val);
                continue;
            }

            // arg with 2 values
            check_arg(i);
            std::string val2 = argv[++i];
            if (opt.handler_str_str) {
                opt.handler_str_str(params, val, val2);
                continue;
            }
        } catch (std::exception & e) {
            throw std::invalid_argument(string_format(
                "error while handling argument \"%s\": %s\n\n"
                "usage:\n%s\n\nto show complete usage, run with -h",
                arg.c_str(), e.what(), opt.to_string().c_str()));
        }
    }

    postprocess_cpu_params(params.cpuparams,       nullptr);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    postprocess_cpu_params(params.speculative.cpuparams,       &params.cpuparams);
    postprocess_cpu_params(params.speculative.cpuparams_batch, &params.cpuparams_batch);

    if (params.prompt_cache_all && (params.interactive || params.interactive_first)) {
        throw std::invalid_argument("error: --prompt-cache-all not supported in interactive mode yet\n");
    }

    // handle model and download
    {
        auto res = common_params_handle_model(params.model, params.hf_token, params.offline);
        if (params.no_mmproj) {
            params.mmproj = {};
        } else if (res.found_mmproj && params.mmproj.path.empty() && params.mmproj.url.empty()) {
            // optionally, handle mmproj model when -hf is specified
            params.mmproj = res.mmproj;
        }
        // only download mmproj if the current example is using it
        for (auto & ex : mmproj_examples) {
            if (ctx_arg.ex == ex) {
                common_params_handle_model(params.mmproj,    params.hf_token, params.offline);
                break;
            }
        }
        common_params_handle_model(params.speculative.model, params.hf_token, params.offline);
        common_params_handle_model(params.vocoder.model,     params.hf_token, params.offline);
    }

    // model is required (except for server)
    // TODO @ngxson : maybe show a list of available models in CLI in this case
    if (params.model.path.empty() && ctx_arg.ex != LLAMA_EXAMPLE_SERVER && !params.usage && !params.completion) {
        throw std::invalid_argument("error: --model is required\n");
    }

    if (params.escape) {
        string_process_escapes(params.prompt);
        string_process_escapes(params.input_prefix);
        string_process_escapes(params.input_suffix);
        for (auto & antiprompt : params.antiprompt) {
            string_process_escapes(antiprompt);
        }
        for (auto & seq_breaker : params.sampling.dry_sequence_breakers) {
            string_process_escapes(seq_breaker);
        }
        for (auto & pair : params.speculative.replacements) {
            string_process_escapes(pair.first);
            string_process_escapes(pair.second);
        }
    }

    if (!params.kv_overrides.empty()) {
        params.kv_overrides.emplace_back();
        params.kv_overrides.back().key[0] = 0;
    }

    // pad tensor_buft_overrides for llama_params_fit:
    const size_t ntbo = llama_max_tensor_buft_overrides();
    while (params.tensor_buft_overrides.size() < ntbo) {
        params.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.speculative.tensor_buft_overrides.empty()) {
        params.speculative.tensor_buft_overrides.push_back({nullptr, nullptr});
    }

    if (!params.chat_template.empty() && !common_chat_verify_template(params.chat_template, params.use_jinja)) {
        throw std::runtime_error(string_format(
            "error: the supplied chat template is not supported: %s%s\n",
            params.chat_template.c_str(),
            params.use_jinja ? "" : "\nnote: llama.cpp was started without --jinja, we only support commonly used templates"
        ));
    }

    common_log_set_verbosity_thold(params.verbosity);

    return true;
}

static void common_params_print_usage(common_params_context & ctx_arg) {
    auto print_options = [](std::vector<common_arg *> & options) {
        for (common_arg * opt : options) {
            printf("%s", opt->to_string().c_str());
        }
    };

    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sparam_options;
    std::vector<common_arg *> specific_options;
    for (auto & opt : ctx_arg.options) {
        // in case multiple LLAMA_EXAMPLE_* are set, we prioritize the LLAMA_EXAMPLE_* matching current example
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }
    printf("----- common params -----\n\n");
    print_options(common_options);
    printf("\n\n----- sampling params -----\n\n");
    print_options(sparam_options);
    // TODO: maybe convert enum llama_example to string
    printf("\n\n----- example-specific params -----\n\n");
    print_options(specific_options);
}

static void common_params_print_completion(common_params_context & ctx_arg) {
    std::vector<common_arg *> common_options;
    std::vector<common_arg *> sparam_options;
    std::vector<common_arg *> specific_options;

    for (auto & opt : ctx_arg.options) {
        if (opt.is_sparam) {
            sparam_options.push_back(&opt);
        } else if (opt.in_example(ctx_arg.ex)) {
            specific_options.push_back(&opt);
        } else {
            common_options.push_back(&opt);
        }
    }

    printf("_llama_completions() {\n");
    printf("    local cur prev opts\n");
    printf("    COMPREPLY=()\n");
    printf("    cur=\"${COMP_WORDS[COMP_CWORD]}\"\n");
    printf("    prev=\"${COMP_WORDS[COMP_CWORD-1]}\"\n\n");

    printf("    opts=\"");
    auto print_options = [](const std::vector<common_arg *> & options) {
        for (const common_arg * opt : options) {
            for (const char * arg : opt->args) {
                printf("%s ", arg);
            }
        }
    };

    print_options(common_options);
    print_options(sparam_options);
    print_options(specific_options);
    printf("\"\n\n");

    printf("    case \"$prev\" in\n");
    printf("        --model|-m)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gguf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --grammar-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.gbnf' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        --chat-template-file)\n");
    printf("            COMPREPLY=( $(compgen -f -X '!*.jinja' -- \"$cur\") $(compgen -d -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("        *)\n");
    printf("            COMPREPLY=( $(compgen -W \"${opts}\" -- \"$cur\") )\n");
    printf("            return 0\n");
    printf("            ;;\n");
    printf("    esac\n");
    printf("}\n\n");

    std::set<std::string> executables = {
        "llama-batched",
        "llama-batched-bench",
        "llama-bench",
        "llama-cli",
        "llama-completion",
        "llama-convert-llama2c-to-ggml",
        "llama-cvector-generator",
        "llama-embedding",
        "llama-eval-callback",
        "llama-export-lora",
        "llama-gen-docs",
        "llama-gguf",
        "llama-gguf-hash",
        "llama-gguf-split",
        "llama-gritlm",
        "llama-imatrix",
        "llama-infill",
        "llama-mtmd-cli",
        "llama-llava-clip-quantize-cli",
        "llama-lookahead",
        "llama-lookup",
        "llama-lookup-create",
        "llama-lookup-merge",
        "llama-lookup-stats",
        "llama-parallel",
        "llama-passkey",
        "llama-perplexity",
        "llama-q8dot",
        "llama-quantize",
        "llama-qwen2vl-cli",
        "llama-retrieval",
        "llama-run",
        "llama-save-load-state",
        "llama-server",
        "llama-simple",
        "llama-simple-chat",
        "llama-speculative",
        "llama-speculative-simple",
        "llama-tokenize",
        "llama-tts",
        "llama-vdot"
    };

    for (const auto& exe : executables) {
        printf("complete -F _llama_completions %s\n", exe.c_str());
    }
}

static std::vector<ggml_backend_dev_t> parse_device_list(const std::string & value) {
    std::vector<ggml_backend_dev_t> devices;
    auto dev_names = string_split<std::string>(value, ',');
    if (dev_names.empty()) {
        throw std::invalid_argument("no devices specified");
    }
    if (dev_names.size() == 1 && dev_names[0] == "none") {
        devices.push_back(nullptr);
    } else {
        for (const auto & device : dev_names) {
            auto * dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
                throw std::invalid_argument(string_format("invalid device: %s", device.c_str()));
            }
            devices.push_back(dev);
        }
        devices.push_back(nullptr);
    }
    return devices;
}

static void add_rpc_devices(const std::string & servers) {
    auto rpc_servers = string_split<std::string>(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_reg_t (*ggml_backend_rpc_add_server_t)(const char * endpoint);
    ggml_backend_rpc_add_server_t ggml_backend_rpc_add_server_fn = (ggml_backend_rpc_add_server_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_server");
    if (!ggml_backend_rpc_add_server_fn) {
        throw std::invalid_argument("failed to find RPC add server function");
    }
    for (const auto & server : rpc_servers) {
        auto reg = ggml_backend_rpc_add_server_fn(server.c_str());
        ggml_backend_register(reg);
    }
}

bool common_params_to_map(int argc, char ** argv, llama_example ex, std::map<common_arg, std::string> & out_map) {
    common_params dummy_params;
    common_params_context ctx_arg = common_params_parser_init(dummy_params, ex, nullptr);

    std::unordered_map<std::string, common_arg *> arg_to_options;
    for (auto & opt : ctx_arg.options) {
        for (const auto & arg : opt.args) {
            arg_to_options[arg] = &opt;
        }
        for (const auto & arg : opt.args_neg) {
            arg_to_options[arg] = &opt;
        }
    }

    // TODO @ngxson : find a way to deduplicate this code

    // handle command line arguments
    auto check_arg = [&](int i) {
        if (i+1 >= argc) {
            throw std::invalid_argument("expected value for argument");
        }
    };

    for (int i = 1; i < argc; i++) {
        const std::string arg_prefix = "--";

        std::string arg = argv[i];
        if (arg.compare(0, arg_prefix.size(), arg_prefix) == 0) {
            std::replace(arg.begin(), arg.end(), '_', '-');
        }
        if (arg_to_options.find(arg) == arg_to_options.end()) {
            throw std::invalid_argument(string_format("error: invalid argument: %s", arg.c_str()));
        }
        auto opt = *arg_to_options[arg];
        std::string val;
        if (opt.value_hint != nullptr) {
            // arg with single value
            check_arg(i);
            val = argv[++i];
        }
        if (opt.value_hint_2 != nullptr) {
            // TODO: support arg with 2 values
            throw std::invalid_argument("error: argument with 2 values is not yet supported\n");
        }
        out_map[opt] = val;
    }

    return true;
}

bool common_params_parse(int argc, char ** argv, common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    auto ctx_arg = common_params_parser_init(params, ex, print_usage);
    const common_params params_org = ctx_arg.params; // the example can modify the default params

    try {
        if (!common_params_parse_ex(argc, argv, ctx_arg)) {
            ctx_arg.params = params_org;
            return false;
        }
        if (ctx_arg.params.usage) {
            common_params_print_usage(ctx_arg);
            if (ctx_arg.print_usage) {
                ctx_arg.print_usage(argc, argv);
            }
            exit(0);
        }
        if (ctx_arg.params.completion) {
            common_params_print_completion(ctx_arg);
            exit(0);
        }
        params.lr.init();
    } catch (const std::invalid_argument & ex) {
        fprintf(stderr, "%s\n", ex.what());
        ctx_arg.params = params_org;
        return false;
    } catch (std::exception & ex) {
        fprintf(stderr, "%s\n", ex.what());
        exit(1); // for other exceptions, we exit with status code 1
    }

    return true;
}

static std::string list_builtin_chat_templates() {
    std::vector<const char *> supported_tmpl;
    int32_t res = llama_chat_builtin_templates(nullptr, 0);
    supported_tmpl.resize(res);
    res = llama_chat_builtin_templates(supported_tmpl.data(), supported_tmpl.size());
    std::ostringstream msg;
    for (auto & tmpl : supported_tmpl) {
        msg << tmpl << (&tmpl == &supported_tmpl.back() ? "" : ", ");
    }
    return msg.str();
}

bool common_arg_utils::is_truthy(const std::string & value) {
    return value == "on" || value == "enabled" || value == "true" || value == "1";
}

bool common_arg_utils::is_falsey(const std::string & value) {
    return value == "off" || value == "disabled" || value == "false" || value == "0";
}

bool common_arg_utils::is_autoy(const std::string & value) {
    return value == "auto" || value == "-1";
}

common_params_context common_params_parser_init(common_params & params, llama_example ex, void(*print_usage)(int, char **)) {
    // per-example default params
    // we define here to make sure it's included in llama-gen-docs
    if (ex == LLAMA_EXAMPLE_COMPLETION) {
        params.use_jinja = false;   // disable jinja by default

    } else if (ex == LLAMA_EXAMPLE_MTMD) {
        params.use_jinja = false;   // disable jinja by default
        params.sampling.temp = 0.2; // lower temp by default for better quality

    } else if (ex == LLAMA_EXAMPLE_SERVER) {
        params.n_parallel = -1;     // auto by default
    }

    params.use_color = tty_can_use_colors();

    // load dynamic backends
    ggml_backend_load_all();

    common_params_context ctx_arg(params);
    ctx_arg.print_usage = print_usage;
    ctx_arg.ex          = ex;

    std::string sampler_type_chars;
    std::string sampler_type_names;
    for (const auto & sampler : params.sampling.samplers) {
        sampler_type_chars += common_sampler_type_to_chr(sampler);
        sampler_type_names += common_sampler_type_to_str(sampler) + ";";
    }
    sampler_type_names.pop_back();


    /**
     * filter options by example
     * rules:
     * - all examples inherit options from LLAMA_EXAMPLE_COMMON
     * - if LLAMA_EXAMPLE_* is set (other than COMMON), we only show the option in the corresponding example
     * - if both {LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_*,} are set, we will prioritize the LLAMA_EXAMPLE_* matching current example
     */
    auto add_opt = [&](common_arg arg) {
        if ((arg.in_example(ex) || arg.in_example(LLAMA_EXAMPLE_COMMON)) && !arg.is_exclude(ex)) {
            ctx_arg.options.push_back(std::move(arg));
        }
    };


    add_opt(common_arg(
        {"-h", "--help", "--usage"},
        "print usage and exit",
        [](common_params & params) {
            params.usage = true;
        }
    ));
    add_opt(common_arg(
        {"--version"},
        "show version and build info",
        [](common_params &) {
            fprintf(stderr, "version: %d (%s)\n", LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
            fprintf(stderr, "built with %s for %s\n", LLAMA_COMPILER, LLAMA_BUILD_TARGET);
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"-cl", "--cache-list"},
        "show list of models in cache",
        [](common_params &) {
            printf("model cache directory: %s\n", fs_get_cache_directory().c_str());
            auto models = common_list_cached_models();
            printf("number of models in cache: %zu\n", models.size());
            for (size_t i = 0; i < models.size(); i++) {
                auto & model = models[i];
                printf("%4d. %s\n", (int) i + 1, model.to_string().c_str());
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--completion-bash"},
        "print source-able bash completion script for llama.cpp",
        [](common_params & params) {
            params.completion = true;
        }
    ));
    add_opt(common_arg(
        {"--verbose-prompt"},
        string_format("print a verbose prompt before generation (default: %s)", params.verbose_prompt ? "true" : "false"),
        [](common_params & params) {
            params.verbose_prompt = true;
        }
    ));
    add_opt(common_arg(
        {"--display-prompt"},
        {"--no-display-prompt"},
        string_format("whether to print prompt at generation (default: %s)", params.display_prompt ? "true" : "false"),
        [](common_params & params, bool value) {
            params.display_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-co", "--color"}, "[on|off|auto]",
        "Colorize output to distinguish prompt and user input from generations ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.use_color = true;
            } else if (is_falsey(value)) {
                params.use_color = false;
            } else if (is_autoy(value)) {
                params.use_color = tty_can_use_colors();
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --color: '%s'\n", value.c_str()));
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-t", "--threads"}, "N",
        string_format("number of CPU threads to use during generation (default: %d)", params.cpuparams.n_threads),
        [](common_params & params, int value) {
            params.cpuparams.n_threads = value;
            if (params.cpuparams.n_threads <= 0) {
                params.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_env("LLAMA_ARG_THREADS"));
    add_opt(common_arg(
        {"-tb", "--threads-batch"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads)",
        [](common_params & params, int value) {
            params.cpuparams_batch.n_threads = value;
            if (params.cpuparams_batch.n_threads <= 0) {
                params.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ));
    add_opt(common_arg(
        {"-C", "--cpu-mask"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range (default: \"\")",
        [](common_params & params, const std::string & mask) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Cr", "--cpu-range"}, "lo-hi",
        "range of CPUs for affinity. Complements --cpu-mask",
        [](common_params & params, const std::string & range) {
            params.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict"}, "<0|1>",
        string_format("use strict CPU placement (default: %u)\n", (unsigned) params.cpuparams.strict_cpu),
        [](common_params & params, const std::string & value) {
            params.cpuparams.strict_cpu = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"--prio"}, "N",
        string_format("set process/thread priority : low(-1), normal(0), medium(1), high(2), realtime(3) (default: %d)\n", params.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < GGML_SCHED_PRIO_LOW || prio > GGML_SCHED_PRIO_REALTIME) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll"}, "<0...100>",
        string_format("use polling level to wait for work (0 - no polling, default: %u)\n", (unsigned) params.cpuparams.poll),
        [](common_params & params, const std::string & value) {
            params.cpuparams.poll = std::stoul(value);
        }
    ));
    add_opt(common_arg(
        {"-Cb", "--cpu-mask-batch"}, "M",
        "CPU affinity mask: arbitrarily long hex. Complements cpu-range-batch (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ));
    add_opt(common_arg(
        {"-Crb", "--cpu-range-batch"}, "lo-hi",
        "ranges of CPUs for affinity. Complements --cpu-mask-batch",
        [](common_params & params, const std::string & range) {
            params.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ));
    add_opt(common_arg(
        {"--cpu-strict-batch"}, "<0|1>",
        "use strict CPU placement (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.cpuparams_batch.strict_cpu = value;
        }
    ));
    add_opt(common_arg(
        {"--prio-batch"}, "N",
        string_format("set process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ));
    add_opt(common_arg(
        {"--poll-batch"}, "<0|1>",
        "use polling to wait for work (default: same as --poll)",
        [](common_params & params, int value) {
            params.cpuparams_batch.poll = value;
        }
    ));
    add_opt(common_arg(
        {"-lcs", "--lookup-cache-static"}, "FNAME",
        "path to static lookup cache to use for lookup decoding (not updated by generation)",
        [](common_params & params, const std::string & value) {
            params.lookup_cache_static = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-lcd", "--lookup-cache-dynamic"}, "FNAME",
        "path to dynamic lookup cache to use for lookup decoding (updated by generation)",
        [](common_params & params, const std::string & value) {
            params.lookup_cache_dynamic = value;
        }
    ).set_examples({LLAMA_EXAMPLE_LOOKUP}));
    add_opt(common_arg(
        {"-c", "--ctx-size"}, "N",
        string_format("size of the prompt context (default: %d, 0 = loaded from model)", params.n_ctx),
        [](common_params & params, int value) {
            params.n_ctx = value;
        }
    ).set_env("LLAMA_ARG_CTX_SIZE"));
    add_opt(common_arg(
        {"-n", "--predict", "--n-predict"}, "N",
        string_format(
            ex == LLAMA_EXAMPLE_COMPLETION
                ? "number of tokens to predict (default: %d, -1 = infinity, -2 = until context filled)"
                : "number of tokens to predict (default: %d, -1 = infinity)",
            params.n_predict),
        [](common_params & params, int value) {
            params.n_predict = value;
        }
    ).set_env("LLAMA_ARG_N_PREDICT"));
    add_opt(common_arg(
        {"-b", "--batch-size"}, "N",
        string_format("logical maximum batch size (default: %d)", params.n_batch),
        [](common_params & params, int value) {
            params.n_batch = value;
        }
    ).set_env("LLAMA_ARG_BATCH"));
    add_opt(common_arg(
        {"-ub", "--ubatch-size"}, "N",
        string_format("physical maximum batch size (default: %d)", params.n_ubatch),
        [](common_params & params, int value) {
            params.n_ubatch = value;
        }
    ).set_env("LLAMA_ARG_UBATCH"));
    add_opt(common_arg(
        {"--keep"}, "N",
        string_format("number of tokens to keep from the initial prompt (default: %d, -1 = all)", params.n_keep),
        [](common_params & params, int value) {
            params.n_keep = value;
        }
    ));
    add_opt(common_arg(
        {"--swa-full"},
        string_format("use full-size SWA cache (default: %s)\n"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)", params.swa_full ? "true" : "false"),
        [](common_params & params) {
            params.swa_full = true;
        }
    ).set_env("LLAMA_ARG_SWA_FULL"));
    add_opt(common_arg(
        {"--ctx-checkpoints", "--swa-checkpoints"}, "N",
        string_format("max number of context checkpoints to create per slot (default: %d)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/15293)", params.n_ctx_checkpoints),
        [](common_params & params, int value) {
            params.n_ctx_checkpoints = value;
        }
    ).set_env("LLAMA_ARG_CTX_CHECKPOINTS").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--cache-ram", "-cram"}, "N",
        string_format("set the maximum cache size in MiB (default: %d, -1 - no limit, 0 - disable)"
            "[(more info)](https://github.com/ggml-org/llama.cpp/pull/16391)", params.cache_ram_mib),
        [](common_params & params, int value) {
            params.cache_ram_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--kv-unified", "-kvu"},
        "use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)",
        [](common_params & params) {
            params.kv_unified = true;
        }
    ).set_env("LLAMA_ARG_KV_UNIFIED").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--context-shift"},
        {"--no-context-shift"},
        string_format("whether to use context shift on infinite text generation (default: %s)", params.ctx_shift ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.ctx_shift = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY}).set_env("LLAMA_ARG_CONTEXT_SHIFT"));
    add_opt(common_arg(
        {"--chunks"}, "N",
        string_format("max number of chunks to process (default: %d, -1 = all)", params.n_chunks),
        [](common_params & params, int value) {
            params.n_chunks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_PERPLEXITY, LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg({ "-fa", "--flash-attn" }, "[on|off|auto]",
                       string_format("set Flash Attention use ('on', 'off', or 'auto', default: '%s')",
                                     llama_flash_attn_type_name(params.flash_attn_type)),
                       [](common_params & params, const std::string & value) {
                           if (is_truthy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
                           } else if (is_falsey(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
                           } else if (is_autoy(value)) {
                               params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
                           } else {
                               throw std::runtime_error(
                                   string_format("error: unknown value for --flash-attn: '%s'\n", value.c_str()));
                           }
                       }).set_env("LLAMA_ARG_FLASH_ATTN"));
    add_opt(common_arg(
        {"-p", "--prompt"}, "PROMPT",
        "prompt to start generation with; for system message, use -sys",
        [](common_params & params, const std::string & value) {
            params.prompt = value;
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sys", "--system-prompt"}, "PROMPT",
        "system prompt to use with model (if applicable, depending on chat template)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION}));
    add_opt(common_arg(
        {"--perf"},
        {"--no-perf"},
        string_format("whether to enable internal libllama performance timings (default: %s)", params.no_perf ? "true" : "false"),
        [](common_params & params, bool value) {
            params.no_perf = !value;
            params.sampling.no_perf = !value;
        }
    ).set_env("LLAMA_ARG_PERF"));
    add_opt(common_arg(
        {"--show-timings"},
        {"--no-show-timings"},
        string_format("whether to show timing information after each response (default: %s)", params.show_timings ? "true" : "false"),
        [](common_params & params, bool value) {
            params.show_timings = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SHOW_TIMINGS"));
    add_opt(common_arg(
        {"-f", "--file"}, "FNAME",
        "a file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.prompt = read_file(value);
            // store the external file name in params
            params.prompt_file = value;
            if (!params.prompt.empty() && params.prompt.back() == '\n') {
                params.prompt.pop_back();
            }
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sysf", "--system-prompt-file"}, "FNAME",
        "a file containing the system prompt (default: none)",
        [](common_params & params, const std::string & value) {
            params.system_prompt = read_file(value);
            if (!params.system_prompt.empty() && params.system_prompt.back() == '\n') {
                params.system_prompt.pop_back();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_DIFFUSION}));
    add_opt(common_arg(
        {"--in-file"}, "FNAME",
        "an input file (repeat to specify multiple files)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.in_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-bf", "--binary-file"}, "FNAME",
        "binary file containing the prompt (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            // store the external file name in params
            params.prompt_file = value;
            std::ostringstream ss;
            ss << file.rdbuf();
            params.prompt = ss.str();
            fprintf(stderr, "Read %zu bytes from binary file %s\n", params.prompt.size(), value.c_str());
        }
    ).set_excludes({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-e", "--escape"},
        {"--no-escape"},
        string_format("whether to process escapes sequences (\\n, \\r, \\t, \\', \\\", \\\\) (default: %s)", params.escape ? "true" : "false"),
        [](common_params & params, bool value) {
            params.escape = value;
        }
    ));
    add_opt(common_arg(
        {"-ptc", "--print-token-count"}, "N",
        string_format("print token count every N tokens (default: %d)", params.n_print),
        [](common_params & params, int value) {
            params.n_print = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache"}, "FNAME",
        "file to cache prompt state for faster startup (default: none)",
        [](common_params & params, const std::string & value) {
            params.path_prompt_cache = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-all"},
        "if specified, saves user input and generations to cache as well\n",
        [](common_params & params) {
            params.prompt_cache_all = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--prompt-cache-ro"},
        "if specified, uses the prompt cache but does not update it",
        [](common_params & params) {
            params.prompt_cache_ro = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-r", "--reverse-prompt"}, "PROMPT",
        "halt generation at PROMPT, return control in interactive mode\n",
        [](common_params & params, const std::string & value) {
            params.antiprompt.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-sp", "--special"},
        string_format("special tokens output enabled (default: %s)", params.special ? "true" : "false"),
        [](common_params & params) {
            params.special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-cnv", "--conversation"},
        {"-no-cnv", "--no-conversation"},
        "whether to run in conversation mode:\n"
        "- does not print special tokens and suffix/prefix\n"
        "- interactive mode is also enabled\n"
        "(default: auto enabled if chat template is available)",
        [](common_params & params, bool value) {
            params.conversation_mode = value ? COMMON_CONVERSATION_MODE_ENABLED : COMMON_CONVERSATION_MODE_DISABLED;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-st", "--single-turn"},
        "run conversation for a single turn only, then exit when done\n"
        "will not be interactive if first turn is predefined with --prompt\n"
        "(default: false)",
        [](common_params & params) {
            params.single_turn = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-i", "--interactive"},
        string_format("run in interactive mode (default: %s)", params.interactive ? "true" : "false"),
        [](common_params & params) {
            params.interactive = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-if", "--interactive-first"},
        string_format("run in interactive mode and wait for input right away (default: %s)", params.interactive_first ? "true" : "false"),
        [](common_params & params) {
            params.interactive_first = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-mli", "--multiline-input"},
        "allows you to write or paste multiple lines without ending each in '\\'",
        [](common_params & params) {
            params.multiline_input = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--in-prefix-bos"},
        "prefix BOS to user inputs, preceding the `--in-prefix` string",
        [](common_params & params) {
            params.input_prefix_bos = true;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-prefix"}, "STRING",
        "string to prefix user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_prefix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--in-suffix"}, "STRING",
        "string to suffix after user inputs with (default: empty)",
        [](common_params & params, const std::string & value) {
            params.input_suffix = value;
            params.enable_chat_template = false;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"--warmup"},
        {"--no-warmup"},
        string_format("whether to perform warmup with an empty run (default: %s)", params.warmup ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.warmup = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--spm-infill"},
        string_format(
            "use Suffix/Prefix/Middle pattern for infill (instead of Prefix/Suffix/Middle) as some models prefer this. (default: %s)",
            params.spm_infill ? "enabled" : "disabled"
        ),
        [](common_params & params) {
            params.spm_infill = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--samplers"}, "SAMPLERS",
        string_format("samplers that will be used for generation in the order, separated by \';\'\n(default: %s)", sampler_type_names.c_str()),
        [](common_params & params, const std::string & value) {
            const auto sampler_names = string_split<std::string>(value, ';');
            params.sampling.samplers = common_sampler_types_from_names(sampler_names, true);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_SAMPLERS;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-s", "--seed"}, "SEED",
        string_format("RNG seed (default: %d, use random seed for %d)", params.sampling.seed, LLAMA_DEFAULT_SEED),
        [](common_params & params, const std::string & value) {
            params.sampling.seed = std::stoul(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--sampling-seq", "--sampler-seq"}, "SEQUENCE",
        string_format("simplified sequence for samplers that will be used (default: %s)", sampler_type_chars.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.samplers = common_sampler_types_from_chars(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--ignore-eos"},
        "ignore end of stream token and continue generating (implies --logit-bias EOS-inf)",
        [](common_params & params) {
            params.sampling.ignore_eos = true;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--temp"}, "N",
        string_format("temperature (default: %.1f)", (double)params.sampling.temp),
        [](common_params & params, const std::string & value) {
            params.sampling.temp = std::stof(value);
            params.sampling.temp = std::max(params.sampling.temp, 0.0f);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TEMP;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--top-k"}, "N",
        string_format("top-k sampling (default: %d, 0 = disabled)", params.sampling.top_k),
        [](common_params & params, int value) {
            params.sampling.top_k = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_K;
        }
    ).set_sparam().set_env("LLAMA_ARG_TOP_K"));
    add_opt(common_arg(
        {"--top-p"}, "N",
        string_format("top-p sampling (default: %.1f, 1.0 = disabled)", (double)params.sampling.top_p),
        [](common_params & params, const std::string & value) {
            params.sampling.top_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_TOP_P;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--min-p"}, "N",
        string_format("min-p sampling (default: %.1f, 0.0 = disabled)", (double)params.sampling.min_p),
        [](common_params & params, const std::string & value) {
            params.sampling.min_p = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIN_P;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--top-nsigma"}, "N",
        string_format("top-n-sigma sampling (default: %.1f, -1.0 = disabled)", params.sampling.top_n_sigma),
        [](common_params & params, const std::string & value) {
            params.sampling.top_n_sigma = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--xtc-probability"}, "N",
        string_format("xtc probability (default: %.1f, 0.0 = disabled)", (double)params.sampling.xtc_probability),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_probability = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_PROBABILITY;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--xtc-threshold"}, "N",
        string_format("xtc threshold (default: %.1f, 1.0 = disabled)", (double)params.sampling.xtc_threshold),
        [](common_params & params, const std::string & value) {
            params.sampling.xtc_threshold = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_XTC_THRESHOLD;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--typical"}, "N",
        string_format("locally typical sampling, parameter p (default: %.1f, 1.0 = disabled)", (double)params.sampling.typ_p),
        [](common_params & params, const std::string & value) {
            params.sampling.typ_p = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--repeat-last-n"}, "N",
        string_format("last n tokens to consider for penalize (default: %d, 0 = disabled, -1 = ctx_size)", params.sampling.penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid repeat-last-n = %d\n", value));
            }
            params.sampling.penalty_last_n = value;
            params.sampling.n_prev = std::max(params.sampling.n_prev, params.sampling.penalty_last_n);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_LAST_N;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--repeat-penalty"}, "N",
        string_format("penalize repeat sequence of tokens (default: %.1f, 1.0 = disabled)", (double)params.sampling.penalty_repeat),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_repeat = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_PENALTY_REPEAT;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--presence-penalty"}, "N",
        string_format("repeat alpha presence penalty (default: %.1f, 0.0 = disabled)", (double)params.sampling.penalty_present),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_present = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--frequency-penalty"}, "N",
        string_format("repeat alpha frequency penalty (default: %.1f, 0.0 = disabled)", (double)params.sampling.penalty_freq),
        [](common_params & params, const std::string & value) {
            params.sampling.penalty_freq = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-multiplier"}, "N",
        string_format("set DRY sampling multiplier (default: %.1f, 0.0 = disabled)", (double)params.sampling.dry_multiplier),
        [](common_params & params, const std::string & value) {
            params.sampling.dry_multiplier = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-base"}, "N",
        string_format("set DRY sampling base value (default: %.2f)", (double)params.sampling.dry_base),
        [](common_params & params, const std::string & value) {
            float potential_base = std::stof(value);
            if (potential_base >= 1.0f)
            {
                params.sampling.dry_base = potential_base;
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-allowed-length"}, "N",
        string_format("set allowed length for DRY sampling (default: %d)", params.sampling.dry_allowed_length),
        [](common_params & params, int value) {
            params.sampling.dry_allowed_length = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-penalty-last-n"}, "N",
        string_format("set DRY penalty for the last n tokens (default: %d, 0 = disable, -1 = context size)", params.sampling.dry_penalty_last_n),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::runtime_error(string_format("error: invalid dry-penalty-last-n = %d\n", value));
            }
            params.sampling.dry_penalty_last_n = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dry-sequence-breaker"}, "STRING",
        string_format("add sequence breaker for DRY sampling, clearing out default breakers (%s) in the process; use \"none\" to not use any sequence breakers\n",
            params.sampling.dry_sequence_breakers.empty() ? "none" :
            std::accumulate(std::next(params.sampling.dry_sequence_breakers.begin()),
                params.sampling.dry_sequence_breakers.end(),
                std::string("'") + (params.sampling.dry_sequence_breakers[0] == "\n" ? "\\n" : params.sampling.dry_sequence_breakers[0]) + "'",
                [](const std::string& a, const std::string& b) {
                    std::string formatted_b = (b == "\n") ? "\\n" : b;
                    return a + ", '" + formatted_b + "'";
                }).c_str()),
        [](common_params & params, const std::string & value) {
            static bool defaults_cleared = false;

            if (!defaults_cleared) {
                params.sampling.dry_sequence_breakers.clear();
                defaults_cleared = true;
            }

            if (value == "none") {
                params.sampling.dry_sequence_breakers.clear();
            } else {
                params.sampling.dry_sequence_breakers.emplace_back(value);
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dynatemp-range"}, "N",
        string_format("dynamic temperature range (default: %.1f, 0.0 = disabled)", (double)params.sampling.dynatemp_range),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_range = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--dynatemp-exp"}, "N",
        string_format("dynamic temperature exponent (default: %.1f)", (double)params.sampling.dynatemp_exponent),
        [](common_params & params, const std::string & value) {
            params.sampling.dynatemp_exponent = std::stof(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat"}, "N",
        string_format("use Mirostat sampling.\nTop K, Nucleus and Locally Typical samplers are ignored if used.\n"
        "(default: %d, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)", params.sampling.mirostat),
        [](common_params & params, int value) {
            params.sampling.mirostat = value;
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat-lr"}, "N",
        string_format("Mirostat learning rate, parameter eta (default: %.1f)", (double)params.sampling.mirostat_eta),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_eta = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_ETA;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--mirostat-ent"}, "N",
        string_format("Mirostat target entropy, parameter tau (default: %.1f)", (double)params.sampling.mirostat_tau),
        [](common_params & params, const std::string & value) {
            params.sampling.mirostat_tau = std::stof(value);
            params.sampling.user_sampling_config |= common_params_sampling_config::COMMON_PARAMS_SAMPLING_CONFIG_MIROSTAT_TAU;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-l", "--logit-bias"}, "TOKEN_ID(+/-)BIAS",
        "modifies the likelihood of token appearing in the completion,\n"
        "i.e. `--logit-bias 15043+1` to increase likelihood of token ' Hello',\n"
        "or `--logit-bias 15043-1` to decrease likelihood of token ' Hello'",
        [](common_params & params, const std::string & value) {
            std::stringstream ss(value);
            llama_token key;
            char sign;
            std::string value_str;
            try {
                if (ss >> key && ss >> sign && std::getline(ss, value_str) && (sign == '+' || sign == '-')) {
                    const float bias = std::stof(value_str) * ((sign == '-') ? -1.0f : 1.0f);
                    params.sampling.logit_bias.push_back({key, bias});
                } else {
                    throw std::invalid_argument("invalid input format");
                }
            } catch (const std::exception&) {
                throw std::invalid_argument("invalid input format");
            }
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--grammar"}, "GRAMMAR",
        string_format("BNF-like grammar to constrain generations (see samples in grammars/ dir) (default: '%s')", params.sampling.grammar.c_str()),
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = value;
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--grammar-file"}, "FNAME",
        "file to read grammar from",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = read_file(value);
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-j", "--json-schema"}, "SCHEMA",
        "JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](common_params & params, const std::string & value) {
            params.sampling.grammar = json_schema_to_grammar(json::parse(value));
        }
    ).set_sparam());
    add_opt(common_arg(
        {"-jf", "--json-schema-file"}, "FILE",
        "File containing a JSON schema to constrain generations (https://json-schema.org/), e.g. `{}` for any JSON object\nFor schemas w/ external $refs, use --grammar + example/json_schema_to_grammar.py instead",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string schema;
            std::copy(
                std::istreambuf_iterator<char>(file),
                std::istreambuf_iterator<char>(),
                std::back_inserter(schema)
            );
            params.sampling.grammar = json_schema_to_grammar(json::parse(schema));
        }
    ).set_sparam());
    add_opt(common_arg(
        {"--pooling"}, "{none,mean,cls,last,rank}",
        "pooling type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.pooling_type = LLAMA_POOLING_TYPE_NONE; }
            else if (value == "mean") { params.pooling_type = LLAMA_POOLING_TYPE_MEAN; }
            else if (value == "cls")  { params.pooling_type = LLAMA_POOLING_TYPE_CLS;  }
            else if (value == "last") { params.pooling_type = LLAMA_POOLING_TYPE_LAST; }
            else if (value == "rank") { params.pooling_type = LLAMA_POOLING_TYPE_RANK; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_RETRIEVAL, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_POOLING"));
    add_opt(common_arg(
        {"--attention"}, "{causal,non-causal}",
        "attention type for embeddings, use model default if unspecified",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "causal") { params.attention_type = LLAMA_ATTENTION_TYPE_CAUSAL; }
            else if (value == "non-causal") { params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--rope-scaling"}, "{none,linear,yarn}",
        "RoPE frequency scaling method, defaults to linear unless specified by the model",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "none") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE; }
            else if (value == "linear") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR; }
            else if (value == "yarn") { params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_ROPE_SCALING_TYPE"));
    add_opt(common_arg(
        {"--rope-scale"}, "N",
        "RoPE context scaling factor, expands context by a factor of N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = 1.0f / std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_SCALE"));
    add_opt(common_arg(
        {"--rope-freq-base"}, "N",
        "RoPE base frequency, used by NTK-aware scaling (default: loaded from model)",
        [](common_params & params, const std::string & value) {
            params.rope_freq_base = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_BASE"));
    add_opt(common_arg(
        {"--rope-freq-scale"}, "N",
        "RoPE frequency scaling factor, expands context by a factor of 1/N",
        [](common_params & params, const std::string & value) {
            params.rope_freq_scale = std::stof(value);
        }
    ).set_env("LLAMA_ARG_ROPE_FREQ_SCALE"));
    add_opt(common_arg(
        {"--yarn-orig-ctx"}, "N",
        string_format("YaRN: original context size of model (default: %d = model training context size)", params.yarn_orig_ctx),
        [](common_params & params, int value) {
            params.yarn_orig_ctx = value;
        }
    ).set_env("LLAMA_ARG_YARN_ORIG_CTX"));
    add_opt(common_arg(
        {"--yarn-ext-factor"}, "N",
        string_format("YaRN: extrapolation mix factor (default: %.1f, 0.0 = full interpolation)", (double)params.yarn_ext_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_ext_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_EXT_FACTOR"));
    add_opt(common_arg(
        {"--yarn-attn-factor"}, "N",
        string_format("YaRN: scale sqrt(t) or attention magnitude (default: %.1f)", (double)params.yarn_attn_factor),
        [](common_params & params, const std::string & value) {
            params.yarn_attn_factor = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_ATTN_FACTOR"));
    add_opt(common_arg(
        {"--yarn-beta-slow"}, "N",
        string_format("YaRN: high correction dim or alpha (default: %.1f)", (double)params.yarn_beta_slow),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_slow = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_SLOW"));
    add_opt(common_arg(
        {"--yarn-beta-fast"}, "N",
        string_format("YaRN: low correction dim or beta (default: %.1f)", (double)params.yarn_beta_fast),
        [](common_params & params, const std::string & value) {
            params.yarn_beta_fast = std::stof(value);
        }
    ).set_env("LLAMA_ARG_YARN_BETA_FAST"));
    add_opt(common_arg(
        {"-gan", "--grp-attn-n"}, "N",
        string_format("group-attention factor (default: %d)", params.grp_attn_n),
        [](common_params & params, int value) {
            params.grp_attn_n = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_N").set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-gaw", "--grp-attn-w"}, "N",
        string_format("group-attention width (default: %d)", params.grp_attn_w),
        [](common_params & params, int value) {
            params.grp_attn_w = value;
        }
    ).set_env("LLAMA_ARG_GRP_ATTN_W").set_examples({LLAMA_EXAMPLE_COMPLETION}));
    add_opt(common_arg(
        {"-kvo", "--kv-offload"},
        {"-nkvo", "--no-kv-offload"},
        string_format("whether to enable KV cache offloading (default: %s)", params.no_kv_offload ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_kv_offload = !value;
        }
    ).set_env("LLAMA_ARG_KV_OFFLOAD"));
    add_opt(common_arg(
        {"--repack"},
        {"-nr", "--no-repack"},
        string_format("whether to enable weight repacking (default: %s)", params.no_extra_bufts ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_extra_bufts = !value;
        }
    ).set_env("LLAMA_ARG_REPACK"));
    add_opt(common_arg(
        {"--no-host"},
        "bypass host buffer allowing extra buffers to be used",
        [](common_params & params) {
            params.no_host = true;
        }
    ).set_env("LLAMA_ARG_NO_HOST"));
    add_opt(common_arg(
        {"-ctk", "--cache-type-k"}, "TYPE",
        string_format(
            "KV cache data type for K\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K"));
    add_opt(common_arg(
        {"-ctv", "--cache-type-v"}, "TYPE",
        string_format(
            "KV cache data type for V\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V"));
    add_opt(common_arg(
        {"--hellaswag"},
        "compute HellaSwag score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.hellaswag = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--hellaswag-tasks"}, "N",
        string_format("number of tasks to use when computing the HellaSwag score (default: %zu)", params.hellaswag_tasks),
        [](common_params & params, int value) {
            params.hellaswag_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande"},
        "compute Winogrande score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.winogrande = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--winogrande-tasks"}, "N",
        string_format("number of tasks to use when computing the Winogrande score (default: %zu)", params.winogrande_tasks),
        [](common_params & params, int value) {
            params.winogrande_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice"},
        "compute multiple choice score over random tasks from datafile supplied with -f",
        [](common_params & params) {
            params.multiple_choice = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--multiple-choice-tasks"}, "N",
        string_format("number of tasks to use when computing the multiple choice score (default: %zu)", params.multiple_choice_tasks),
        [](common_params & params, int value) {
            params.multiple_choice_tasks = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--kl-divergence"},
        "computes KL-divergence to logits provided via --kl-divergence-base",
        [](common_params & params) {
            params.kl_divergence = true;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--save-all-logits", "--kl-divergence-base"}, "FNAME",
        "set logits file",
        [](common_params & params, const std::string & value) {
            params.logits_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-stride"}, "N",
        string_format("stride for perplexity calculation (default: %d)", params.ppl_stride),
        [](common_params & params, int value) {
            params.ppl_stride = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"--ppl-output-type"}, "<0|1>",
        string_format("output type for perplexity calculation (default: %d)", params.ppl_output_type),
        [](common_params & params, int value) {
            params.ppl_output_type = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PERPLEXITY}));
    add_opt(common_arg(
        {"-dt", "--defrag-thold"}, "N",
        string_format("KV cache defragmentation threshold (DEPRECATED)"),
        [](common_params & params, const std::string & value) {
            GGML_UNUSED(params);
            GGML_UNUSED(value);
            LOG_WRN("DEPRECATED: --defrag-thold is deprecated and no longer necessary to specify\n");
        }
    ).set_env("LLAMA_ARG_DEFRAG_THOLD"));
    if (ex == LLAMA_EXAMPLE_SERVER) {
        // this is to make sure this option appears in the server-specific section of the help message
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of server slots (default: %d, -1 = auto)", params.n_parallel),
            [](common_params & params, int value) {
                if (value == 0) {
                    throw std::invalid_argument("error: invalid value for n_parallel\n");
                }
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL").set_examples({LLAMA_EXAMPLE_SERVER}));
    } else {
        add_opt(common_arg(
            {"-np", "--parallel"}, "N",
            string_format("number of parallel sequences to decode (default: %d)", params.n_parallel),
            [](common_params & params, int value) {
                params.n_parallel = value;
            }
        ).set_env("LLAMA_ARG_N_PARALLEL"));
    }
    add_opt(common_arg(
        {"-ns", "--sequences"}, "N",
        string_format("number of sequences to decode (default: %d)", params.n_sequences),
        [](common_params & params, int value) {
            params.n_sequences = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-cb", "--cont-batching"},
        {"-nocb", "--no-cont-batching"},
        string_format("whether to enable continuous batching (a.k.a dynamic batching) (default: %s)", params.cont_batching ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cont_batching = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CONT_BATCHING"));
    add_opt(common_arg(
        {"-mm", "--mmproj"}, "FILE",
        "path to a multimodal projector file. see tools/mtmd/README.md\n"
        "note: if -hf is used, this argument can be omitted",
        [](common_params & params, const std::string & value) {
            params.mmproj.path = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ"));
    add_opt(common_arg(
        {"-mmu", "--mmproj-url"}, "URL",
        "URL to a multimodal projector file. see tools/mtmd/README.md",
        [](common_params & params, const std::string & value) {
            params.mmproj.url = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_URL"));
    add_opt(common_arg(
        {"--mmproj-auto"},
        {"--no-mmproj", "--no-mmproj-auto"},
        string_format("whether to use multimodal projector file (if available), useful when using -hf (default: %s)", params.no_mmproj ? "disabled" : "enabled"),
        [](common_params & params, bool value) {
            params.no_mmproj = !value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_AUTO"));
    add_opt(common_arg(
        {"--mmproj-offload"},
        {"--no-mmproj-offload"},
        string_format("whether to enable GPU offloading for multimodal projector (default: %s)", params.mmproj_use_gpu ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.mmproj_use_gpu = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_MMPROJ_OFFLOAD"));
    add_opt(common_arg(
        {"--image", "--audio"}, "FILE",
        "path to an image or audio file. use with multimodal models, can be repeated if you have multiple files\n",
        [](common_params & params, const std::string & value) {
            params.image.emplace_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_MTMD, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--image-min-tokens"}, "N",
        "minimum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_min_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MIN_TOKENS"));
    add_opt(common_arg(
        {"--image-max-tokens"}, "N",
        "maximum number of tokens each image can take, only used by vision models with dynamic resolution (default: read from model)",
        [](common_params & params, int value) {
            params.image_max_tokens = value;
        }
    ).set_examples(mmproj_examples).set_env("LLAMA_ARG_IMAGE_MAX_TOKENS"));
    if (llama_supports_rpc()) {
        add_opt(common_arg(
            {"--rpc"}, "SERVERS",
            "comma separated list of RPC servers",
            [](common_params & params, const std::string & value) {
                add_rpc_devices(value);
                GGML_UNUSED(params);
            }
        ).set_env("LLAMA_ARG_RPC"));
    }
    add_opt(common_arg(
        {"--mlock"},
        "force system to keep model in RAM rather than swapping or compressing",
        [](common_params & params) {
            params.use_mlock = true;
        }
    ).set_env("LLAMA_ARG_MLOCK"));
    add_opt(common_arg(
        {"--mmap"},
        {"--no-mmap"},
        string_format("whether to memory-map model (if disabled, slower load but may reduce pageouts if not using mlock) (default: %s)", params.use_mmap ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_mmap = value;
        }
    ).set_env("LLAMA_ARG_MMAP"));
    add_opt(common_arg(
        {"--numa"}, "TYPE",
        "attempt optimizations that help on some NUMA systems\n"
        "- distribute: spread execution evenly over all nodes\n"
        "- isolate: only spawn threads on CPUs on the node that execution started on\n"
        "- numactl: use the CPU map provided by numactl\n"
        "if run without this previously, it is recommended to drop the system page cache before using this\n"
        "see https://github.com/ggml-org/llama.cpp/issues/1437",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "distribute" || value == "") { params.numa = GGML_NUMA_STRATEGY_DISTRIBUTE; }
            else if (value == "isolate") { params.numa = GGML_NUMA_STRATEGY_ISOLATE; }
            else if (value == "numactl") { params.numa = GGML_NUMA_STRATEGY_NUMACTL; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_env("LLAMA_ARG_NUMA"));
    add_opt(common_arg(
        {"-dev", "--device"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.devices = parse_device_list(value);
        }
    ).set_env("LLAMA_ARG_DEVICE"));
    add_opt(common_arg(
        {"--list-devices"},
        "print list of available devices and exit",
        [](common_params &) {
            std::vector<ggml_backend_dev_t> devices;
            for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
                auto * dev = ggml_backend_dev_get(i);
                if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                    devices.push_back(dev);
                }
            }
            printf("Available devices:\n");
            for (auto * dev : devices) {
                size_t free, total;
                ggml_backend_dev_memory(dev, &free, &total);
                printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
            }
            exit(0);
        }
    ));
    add_opt(common_arg(
        {"--override-tensor", "-ot"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.tensor_buft_overrides);
        }
    ));
    add_opt(common_arg(
        {"--override-tensor-draft", "-otd"}, "<tensor name pattern>=<buffer type>,...",
        "override tensor buffer type for draft model", [](common_params & params, const std::string & value) {
            parse_tensor_buffer_overrides(value, params.speculative.tensor_buft_overrides);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--cpu-moe", "-cmoe"},
        "keep all Mixture of Experts (MoE) weights in the CPU",
        [](common_params & params) {
            params.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_env("LLAMA_ARG_CPU_MOE"));
    add_opt(common_arg(
        {"--n-cpu-moe", "-ncmoe"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            for (int i = 0; i < value; ++i) {
                // keep strings alive and avoid leaking memory by storing them in a static vector
                static std::list<std::string> buft_overrides;
                buft_overrides.push_back(llm_ffn_exps_block_regex(i));
                params.tensor_buft_overrides.push_back({buft_overrides.back().c_str(), ggml_backend_cpu_buffer_type()});
            }
        }
    ).set_env("LLAMA_ARG_N_CPU_MOE"));
    add_opt(common_arg(
        {"--cpu-moe-draft", "-cmoed"},
        "keep all Mixture of Experts (MoE) weights in the CPU for the draft model",
        [](common_params & params) {
            params.speculative.tensor_buft_overrides.push_back(llm_ffn_exps_cpu_override());
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_CPU_MOE_DRAFT"));
    add_opt(common_arg(
        {"--n-cpu-moe-draft", "-ncmoed"}, "N",
        "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU for the draft model",
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("invalid value");
            }
            for (int i = 0; i < value; ++i) {
                static std::list<std::string> buft_overrides_draft;
                buft_overrides_draft.push_back(llm_ffn_exps_block_regex(i));
                params.speculative.tensor_buft_overrides.push_back({buft_overrides_draft.back().c_str(), ggml_backend_cpu_buffer_type()});
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_N_CPU_MOE_DRAFT"));
    add_opt(common_arg(
        {"-ngl", "--gpu-layers", "--n-gpu-layers"}, "N",
        string_format("max. number of layers to store in VRAM (default: %d)", params.n_gpu_layers),
        [](common_params & params, int value) {
            params.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_env("LLAMA_ARG_N_GPU_LAYERS"));
    add_opt(common_arg(
        {"-sm", "--split-mode"}, "{none,layer,row}",
        "how to split the model across multiple GPUs, one of:\n"
        "- none: use one GPU only\n"
        "- layer (default): split layers and KV across GPUs\n"
        "- row: split rows across GPUs",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;
            if (arg_next == "none") {
                params.split_mode = LLAMA_SPLIT_MODE_NONE;
            } else if (arg_next == "layer") {
                params.split_mode = LLAMA_SPLIT_MODE_LAYER;
            } else if (arg_next == "row") {
                params.split_mode = LLAMA_SPLIT_MODE_ROW;
            } else {
                throw std::invalid_argument("invalid value");
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the split mode has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_SPLIT_MODE"));
    add_opt(common_arg(
        {"-ts", "--tensor-split"}, "N0,N1,N2,...",
        "fraction of the model to offload to each GPU, comma-separated list of proportions, e.g. 3,1",
        [](common_params & params, const std::string & value) {
            std::string arg_next = value;

            // split string by , and /
            const std::regex regex{ R"([,/]+)" };
            std::sregex_token_iterator it{ arg_next.begin(), arg_next.end(), regex, -1 };
            std::vector<std::string> split_arg{ it, {} };
            if (split_arg.size() >= llama_max_devices()) {
                throw std::invalid_argument(
                    string_format("got %d input configs, but system only has %d devices", (int)split_arg.size(), (int)llama_max_devices())
                );
            }
            for (size_t i = 0; i < llama_max_devices(); ++i) {
                if (i < split_arg.size()) {
                    params.tensor_split[i] = std::stof(split_arg[i]);
                } else {
                    params.tensor_split[i] = 0.0f;
                }
            }
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting a tensor split has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_TENSOR_SPLIT"));
    add_opt(common_arg(
        {"-mg", "--main-gpu"}, "INDEX",
        string_format("the GPU to use for the model (with split-mode = none), or for intermediate results and KV (with split-mode = row) (default: %d)", params.main_gpu),
        [](common_params & params, int value) {
            params.main_gpu = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: llama.cpp was compiled without support for GPU offload. Setting the main GPU has no effect.\n");
            }
        }
    ).set_env("LLAMA_ARG_MAIN_GPU"));
    add_opt(common_arg(
        { "-fit", "--fit" }, "[on|off]",
        string_format("whether to adjust unset arguments to fit in device memory ('on' or 'off', default: '%s')", params.fit_params ? "on" : "off"),
        [](common_params & params, const std::string & value) {
            if (is_truthy(value)) {
                params.fit_params = true;
            } else if (is_falsey(value)) {
                params.fit_params = false;
            } else {
                throw std::runtime_error(
                    string_format("error: unkown value for --fit: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_ARG_FIT"));
    add_opt(common_arg(
        { "-fitt", "--fit-target" }, "MiB",
        string_format("target margin per device for --fit option, default: %zu", params.fit_params_target/(1024*1024)),
        [](common_params & params, int value) {
            params.fit_params_target = value * size_t(1024*1024);
        }
    ).set_env("LLAMA_ARG_FIT_TARGET"));
    add_opt(common_arg(
        { "-fitc", "--fit-ctx" }, "N",
        string_format("minimum ctx size that can be set by --fit option, default: %" PRIu32, params.fit_params_min_ctx),
        [](common_params & params, int value) {
            params.fit_params_min_ctx = value;
        }
    ).set_env("LLAMA_ARG_FIT_CTX"));
    add_opt(common_arg(
        {"--check-tensors"},
        string_format("check model tensor data for invalid values (default: %s)", params.check_tensors ? "true" : "false"),
        [](common_params & params) {
            params.check_tensors = true;
        }
    ));
    add_opt(common_arg(
        {"--override-kv"}, "KEY=TYPE:VALUE",
        "advanced option to override model metadata by key. may be specified multiple times.\n"
        "types: int, float, bool, str. example: --override-kv tokenizer.ggml.add_bos_token=bool:false",
        [](common_params & params, const std::string & value) {
            if (!string_parse_kv_override(value.c_str(), params.kv_overrides)) {
                throw std::runtime_error(string_format("error: Invalid type for KV override: %s\n", value.c_str()));
            }
        }
    ));
    add_opt(common_arg(
        {"--op-offload"},
        {"--no-op-offload"},
        string_format("whether to offload host tensor operations to device (default: %s)", params.no_op_offload ? "false" : "true"),
        [](common_params & params, bool value) {
            params.no_op_offload = !value;
        }
    ));
    add_opt(common_arg(
        {"--lora"}, "FNAME",
        "path to LoRA adapter (can be repeated to use multiple adapters)",
        [](common_params & params, const std::string & value) {
            params.lora_adapters.push_back({ std::string(value), 1.0, "", "", nullptr });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--lora-scaled"}, "FNAME", "SCALE",
        "path to LoRA adapter with user defined scaling (can be repeated to use multiple adapters)",
        [](common_params & params, const std::string & fname, const std::string & scale) {
            params.lora_adapters.push_back({ fname, std::stof(scale), "", "", nullptr });
        }
        // we define this arg on both COMMON and EXPORT_LORA, so when showing help message of export-lora, it will be categorized as "example-specific" arg
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}));
    add_opt(common_arg(
        {"--control-vector"}, "FNAME",
        "add a control vector\nnote: this argument can be repeated to add multiple control vectors",
        [](common_params & params, const std::string & value) {
            params.control_vectors.push_back({ 1.0f, value, });
        }
    ));
    add_opt(common_arg(
        {"--control-vector-scaled"}, "FNAME", "SCALE",
        "add a control vector with user defined scaling SCALE\n"
        "note: this argument can be repeated to add multiple scaled control vectors",
        [](common_params & params, const std::string & fname, const std::string & scale) {
            params.control_vectors.push_back({ std::stof(scale), fname });
        }
    ));
    add_opt(common_arg(
        {"--control-vector-layer-range"}, "START", "END",
        "layer range to apply the control vector(s) to, start and end inclusive",
        [](common_params & params, const std::string & start, const std::string & end) {
            params.control_vector_layer_start = std::stoi(start);
            params.control_vector_layer_end = std::stoi(end);
        }
    ));
    add_opt(common_arg(
        {"-a", "--alias"}, "STRING",
        "set alias for model name (to be used by REST API)",
        [](common_params & params, const std::string & value) {
            params.model_alias = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ALIAS"));
    add_opt(common_arg(
        {"-m", "--model"}, "FNAME",
        ex == LLAMA_EXAMPLE_EXPORT_LORA
            ? "model path from which to load base model"
            : "model path to load",
        [](common_params & params, const std::string & value) {
            params.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMMON, LLAMA_EXAMPLE_EXPORT_LORA}).set_env("LLAMA_ARG_MODEL"));
    add_opt(common_arg(
        {"-mu", "--model-url"}, "MODEL_URL",
        "model download url (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.url = value;
        }
    ).set_env("LLAMA_ARG_MODEL_URL"));
    add_opt(common_arg(
        { "-dr", "--docker-repo" }, "[<repo>/]<model>[:quant]",
        "Docker Hub model repository. repo is optional, default to ai/. quant is optional, default to :latest.\n"
        "example: gemma3\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.docker_repo = value;
        }
    ).set_env("LLAMA_ARG_DOCKER_REPO"));
    add_opt(common_arg(
        {"-hf", "-hfr", "--hf-repo"}, "<user>/<model>[:quant]",
        "Hugging Face model repository; quant is optional, case-insensitive, default to Q4_K_M, or falls back to the first file in the repo if Q4_K_M doesn't exist.\n"
        "mmproj is also downloaded automatically if available. to disable, add --no-mmproj\n"
        "example: unsloth/phi-4-GGUF:q4_k_m\n"
        "(default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO"));
    add_opt(common_arg(
        {"-hfd", "-hfrd", "--hf-repo-draft"}, "<user>/<model>[:quant]",
        "Same as --hf-repo, but for the draft model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.model.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HFD_REPO"));
    add_opt(common_arg(
        {"-hff", "--hf-file"}, "FILE",
        "Hugging Face model file. If specified, it will override the quant in --hf-repo (default: unused)",
        [](common_params & params, const std::string & value) {
            params.model.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE"));
    add_opt(common_arg(
        {"-hfv", "-hfrv", "--hf-repo-v"}, "<user>/<model>[:quant]",
        "Hugging Face model repository for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.hf_repo = value;
        }
    ).set_env("LLAMA_ARG_HF_REPO_V"));
    add_opt(common_arg(
        {"-hffv", "--hf-file-v"}, "FILE",
        "Hugging Face model file for the vocoder model (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.hf_file = value;
        }
    ).set_env("LLAMA_ARG_HF_FILE_V"));
    add_opt(common_arg(
        {"-hft", "--hf-token"}, "TOKEN",
        "Hugging Face access token (default: value from HF_TOKEN environment variable)",
        [](common_params & params, const std::string & value) {
            params.hf_token = value;
        }
    ).set_env("HF_TOKEN"));
    add_opt(common_arg(
        {"--context-file"}, "FNAME",
        "file to load context from (repeat to specify multiple files)",
        [](common_params & params, const std::string & value) {
            std::ifstream file(value, std::ios::binary);
            if (!file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            params.context_files.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-size"}, "N",
        string_format("minimum length of embedded text chunks (default: %d)", params.chunk_size),
        [](common_params & params, int value) {
            params.chunk_size = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--chunk-separator"}, "STRING",
        string_format("separator between chunks (default: '%s')", params.chunk_separator.c_str()),
        [](common_params & params, const std::string & value) {
            params.chunk_separator = value;
        }
    ).set_examples({LLAMA_EXAMPLE_RETRIEVAL}));
    add_opt(common_arg(
        {"--junk"}, "N",
        string_format("number of times to repeat the junk text (default: %d)", params.n_junk),
        [](common_params & params, int value) {
            params.n_junk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"--pos"}, "N",
        string_format("position of the passkey in the junk text (default: %d)", params.i_pos),
        [](common_params & params, int value) {
            params.i_pos = value;
        }
    ).set_examples({LLAMA_EXAMPLE_PASSKEY}));
    add_opt(common_arg(
        {"-o", "--output", "--output-file"}, "FNAME",
        string_format("output file (default: '%s')", params.out_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.out_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX, LLAMA_EXAMPLE_CVECTOR_GENERATOR, LLAMA_EXAMPLE_EXPORT_LORA, LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_FINETUNE}));
    add_opt(common_arg(
        {"-ofreq", "--output-frequency"}, "N",
        string_format("output the imatrix every N iterations (default: %d)", params.n_out_freq),
        [](common_params & params, int value) {
            params.n_out_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--output-format"}, "{gguf,dat}",
        string_format("output format for imatrix file (default: %s)", params.imat_dat > 0 ? "dat" : "gguf"),
        [](common_params & params, const std::string & value) {
            /**/ if (value == "gguf") { params.imat_dat = -1; }
            else if (value == "dat")  { params.imat_dat = 1;  }
            else { throw std::invalid_argument("invalid output format"); }
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--save-frequency"}, "N",
        string_format("save an imatrix copy every N iterations (default: %d)", params.n_save_freq),
        [](common_params & params, int value) {
            params.n_save_freq = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--process-output"},
        string_format("collect data for the output tensor (default: %s)", params.process_output ? "true" : "false"),
        [](common_params & params) {
            params.process_output = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--ppl"},
        {"--no-ppl"},
        string_format("whether to compute perplexity (default: %s)", params.compute_ppl ? "true" : "false"),
        [](common_params & params, bool value) {
            params.compute_ppl = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--chunk", "--from-chunk"}, "N",
        string_format("start processing the input from chunk N (default: %d)", params.i_chunk),
        [](common_params & params, int value) {
            params.i_chunk = value;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--show-statistics"},
        string_format("show imatrix statistics and then exit (default: %s)", params.show_statistics ? "true" : "false"),
        [](common_params & params) {
            params.show_statistics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"--parse-special"},
        string_format("parse special tokens (chat, tool, etc) (default: %s)", params.parse_special ? "true" : "false"),
        [](common_params & params) {
            params.parse_special = true;
        }
    ).set_examples({LLAMA_EXAMPLE_IMATRIX}));
    add_opt(common_arg(
        {"-pps"},
        string_format("is the prompt shared across parallel sequences (default: %s)", params.is_pp_shared ? "true" : "false"),
        [](common_params & params) {
            params.is_pp_shared = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-tgs"},
        string_format("is the text generation separated across the different sequences (default: %s)", params.is_tg_separate ? "true" : "false"),
        [](common_params & params) {
            params.is_tg_separate = true;
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH, LLAMA_EXAMPLE_PARALLEL}));
    add_opt(common_arg(
        {"-npp"}, "n0,n1,...",
        "number of prompt tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pp.insert(params.n_pp.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-ntg"}, "n0,n1,...",
        "number of text generation tokens",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_tg.insert(params.n_tg.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"-npl"}, "n0,n1,...",
        "number of parallel prompts",
        [](common_params & params, const std::string & value) {
            auto p = string_split<int>(value, ',');
            params.n_pl.insert(params.n_pl.end(), p.begin(), p.end());
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--embd-normalize"}, "N",
        string_format("normalisation for embeddings (default: %d) (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)", params.embd_normalize),
        [](common_params & params, int value) {
            params.embd_normalize = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-output-format"}, "FORMAT",
        "empty = default, \"array\" = [[],[]...], \"json\" = openai style, \"json+\" = same \"json\" + cosine similarity matrix, \"raw\" = plain whitespace-delimited output (one embedding per line)",
        [](common_params & params, const std::string & value) {
            params.embd_out = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--embd-separator"}, "STRING",
        "separator of embeddings (default \\n) for example \"<#sep#>\"",
        [](common_params & params, const std::string & value) {
            params.embd_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--cls-separator"}, "STRING",
        "separator of classification sequences (default \\t) for example \"<#seq#>\"",
        [](common_params & params, const std::string & value) {
            params.cls_sep = value;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING}));
    add_opt(common_arg(
        {"--host"}, "HOST",
        string_format("ip address to listen, or bind to an UNIX socket if the address ends with .sock (default: %s)", params.hostname.c_str()),
        [](common_params & params, const std::string & value) {
            params.hostname = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_HOST"));
    add_opt(common_arg(
        {"--port"}, "PORT",
        string_format("port to listen (default: %d)", params.port),
        [](common_params & params, int value) {
            params.port = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PORT"));
    add_opt(common_arg(
        {"--path"}, "PATH",
        string_format("path to serve static files from (default: %s)", params.public_path.c_str()),
        [](common_params & params, const std::string & value) {
            params.public_path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_STATIC_PATH"));
    add_opt(common_arg(
        {"--api-prefix"}, "PREFIX",
        string_format("prefix path the server serves from, without the trailing slash (default: %s)", params.api_prefix.c_str()),
        [](common_params & params, const std::string & value) {
            params.api_prefix = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_API_PREFIX"));
    add_opt(common_arg(
        {"--webui"},
        {"--no-webui"},
        string_format("whether to enable the Web UI (default: %s)", params.webui ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.webui = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_WEBUI"));
    add_opt(common_arg(
        {"--embedding", "--embeddings"},
        string_format("restrict to only support embedding use case; use only with dedicated embedding models (default: %s)", params.embedding ? "enabled" : "disabled"),
        [](common_params & params) {
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_EMBEDDINGS"));
    add_opt(common_arg(
        {"--reranking", "--rerank"},
        string_format("enable reranking endpoint on server (default: %s)", "disabled"),
        [](common_params & params) {
            params.embedding = true;
            params.pooling_type = LLAMA_POOLING_TYPE_RANK;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_RERANKING"));
    add_opt(common_arg(
        {"--api-key"}, "KEY",
        "API key to use for authentication (default: none)",
        [](common_params & params, const std::string & value) {
            params.api_keys.push_back(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_API_KEY"));
    add_opt(common_arg(
        {"--api-key-file"}, "FNAME",
        "path to file containing API keys (default: none)",
        [](common_params & params, const std::string & value) {
            std::ifstream key_file(value);
            if (!key_file) {
                throw std::runtime_error(string_format("error: failed to open file '%s'\n", value.c_str()));
            }
            std::string key;
            while (std::getline(key_file, key)) {
                if (!key.empty()) {
                        params.api_keys.push_back(key);
                }
            }
            key_file.close();
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--ssl-key-file"}, "FNAME",
        "path to file a PEM-encoded SSL private key",
        [](common_params & params, const std::string & value) {
            params.ssl_file_key = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_KEY_FILE"));
    add_opt(common_arg(
        {"--ssl-cert-file"}, "FNAME",
        "path to file a PEM-encoded SSL certificate",
        [](common_params & params, const std::string & value) {
            params.ssl_file_cert = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_SSL_CERT_FILE"));
    add_opt(common_arg(
        {"--chat-template-kwargs"}, "STRING",
        string_format("sets additional params for the json template parser"),
        [](common_params & params, const std::string & value) {
            auto parsed = json::parse(value);
            for (const auto & item : parsed.items()) {
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_CHAT_TEMPLATE_KWARGS"));
    add_opt(common_arg(
        {"-to", "--timeout"}, "N",
        string_format("server read/write timeout in seconds (default: %d)", params.timeout_read),
        [](common_params & params, int value) {
            params.timeout_read  = value;
            params.timeout_write = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_TIMEOUT"));
    add_opt(common_arg(
        {"--threads-http"}, "N",
        string_format("number of threads used to process HTTP requests (default: %d)", params.n_threads_http),
        [](common_params & params, int value) {
            params.n_threads_http = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_THREADS_HTTP"));
    add_opt(common_arg(
        {"--cache-reuse"}, "N",
        string_format(
            "min chunk size to attempt reusing from the cache via KV shifting (default: %d)\n"
            "[(card)](https://ggml.ai/f0.png)", params.n_cache_reuse
        ),
        [](common_params & params, int value) {
            params.n_cache_reuse = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CACHE_REUSE"));
    add_opt(common_arg(
        {"--metrics"},
        string_format("enable prometheus compatible metrics endpoint (default: %s)", params.endpoint_metrics ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_metrics = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_METRICS"));
    add_opt(common_arg(
        {"--props"},
        string_format("enable changing global properties via POST /props (default: %s)", params.endpoint_props ? "enabled" : "disabled"),
        [](common_params & params) {
            params.endpoint_props = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_PROPS"));
    add_opt(common_arg(
        {"--slots"},
        {"--no-slots"},
        string_format("expose slots monitoring endpoint (default: %s)", params.endpoint_slots ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.endpoint_slots = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_ENDPOINT_SLOTS"));
    add_opt(common_arg(
        {"--slot-save-path"}, "PATH",
        "path to save slot kv cache (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.slot_save_path = value;
            if (!fs_is_directory(params.slot_save_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.slot_save_path.empty() && params.slot_save_path[params.slot_save_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.slot_save_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--media-path"}, "PATH",
        "directory for loading local media files; files can be accessed via file:// URLs using relative paths (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.media_path = value;
            if (!fs_is_directory(params.media_path)) {
                throw std::invalid_argument("not a directory: " + value);
            }
            // if doesn't end with DIRECTORY_SEPARATOR, add it
            if (!params.media_path.empty() && params.media_path[params.media_path.size() - 1] != DIRECTORY_SEPARATOR) {
                params.media_path += DIRECTORY_SEPARATOR;
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--models-dir"}, "PATH",
        "directory containing models for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_dir = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_DIR"));
    add_opt(common_arg(
        {"--models-preset"}, "PATH",
        "path to INI file containing model presets for the router server (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.models_preset = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_PRESET"));
    add_opt(common_arg(
        {"--models-max"}, "N",
        string_format("for router server, maximum number of models to load simultaneously (default: %d, 0 = unlimited)", params.models_max),
        [](common_params & params, int value) {
            params.models_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_MAX"));
    add_opt(common_arg(
        {"--models-autoload"},
        {"--no-models-autoload"},
        string_format("for router server, whether to automatically load models (default: %s)", params.models_autoload ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.models_autoload = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_MODELS_AUTOLOAD"));
    add_opt(common_arg(
        {"--jinja"},
        {"--no-jinja"},
        string_format("whether to use jinja template engine for chat (default: %s)", params.use_jinja ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.use_jinja = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_JINJA"));
    add_opt(common_arg(
        {"--reasoning-format"}, "FORMAT",
        "controls whether thought tags are allowed and/or extracted from the response, and in which format they're returned; one of:\n"
        "- none: leaves thoughts unparsed in `message.content`\n"
        "- deepseek: puts thoughts in `message.reasoning_content`\n"
        "- deepseek-legacy: keeps `<think>` tags in `message.content` while also populating `message.reasoning_content`\n"
        "(default: auto)",
        [](common_params & params, const std::string & value) {
            params.reasoning_format = common_reasoning_format_from_name(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK"));
    add_opt(common_arg(
        {"--reasoning-budget"}, "N",
        "controls the amount of thinking allowed; currently only one of: -1 for unrestricted thinking budget, or 0 to disable thinking (default: -1)",
        [](common_params & params, int value) {
            if (value != 0 && value != -1) { throw std::invalid_argument("invalid value"); }
            params.reasoning_budget = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_THINK_BUDGET"));
    add_opt(common_arg(
        {"--chat-template"}, "JINJA_TEMPLATE",
        string_format(
            "set custom jinja chat template (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = value;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_MTMD}).set_env("LLAMA_ARG_CHAT_TEMPLATE"));
    add_opt(common_arg(
        {"--chat-template-file"}, "JINJA_TEMPLATE_FILE",
        string_format(
            "set custom jinja chat template file (default: template taken from model's metadata)\n"
            "if suffix/prefix are specified, template will be disabled\n"
            "only commonly used templates are accepted (unless --jinja is set before this flag):\n"
            "list of built-in templates:\n%s", list_builtin_chat_templates().c_str()
        ),
        [](common_params & params, const std::string & value) {
            params.chat_template = read_file(value);
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI, LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_CHAT_TEMPLATE_FILE"));
    add_opt(common_arg(
        {"--prefill-assistant"},
        {"--no-prefill-assistant"},
        string_format(
            "whether to prefill the assistant's response if the last message is an assistant message (default: prefill enabled)\n"
            "when this flag is set, if the last message is an assistant message then it will be treated as a full message and not prefilled\n"
        ),
        [](common_params & params, bool value) {
            params.prefill_assistant = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}).set_env("LLAMA_ARG_PREFILL_ASSISTANT"));
    add_opt(common_arg(
        {"-sps", "--slot-prompt-similarity"}, "SIMILARITY",
        string_format("how much the prompt of a request must match the prompt of a slot in order to use that slot (default: %.2f, 0.0 = disabled)\n", params.slot_prompt_similarity),
        [](common_params & params, const std::string & value) {
            params.slot_prompt_similarity = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--lora-init-without-apply"},
        string_format("load LoRA adapters without applying them (apply later via POST /lora-adapters) (default: %s)", params.lora_init_without_apply ? "enabled" : "disabled"),
        [](common_params & params) {
            params.lora_init_without_apply = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--simple-io"},
        "use basic IO for better compatibility in subprocesses and limited consoles",
        [](common_params & params) {
            params.simple_io = true;
        }
    ).set_examples({LLAMA_EXAMPLE_COMPLETION, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"--positive-file"}, "FNAME",
        string_format("positive prompts file, one prompt per line (default: '%s')", params.cvector_positive_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_positive_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--negative-file"}, "FNAME",
        string_format("negative prompts file, one prompt per line (default: '%s')", params.cvector_negative_file.c_str()),
        [](common_params & params, const std::string & value) {
            params.cvector_negative_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-batch"}, "N",
        string_format("batch size used for PCA. Larger batch runs faster, but uses more memory (default: %d)", params.n_pca_batch),
        [](common_params & params, int value) {
            params.n_pca_batch = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--pca-iter"}, "N",
        string_format("number of iterations used for PCA (default: %d)", params.n_pca_iterations),
        [](common_params & params, int value) {
            params.n_pca_iterations = value;
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--method"}, "{pca, mean}",
        "dimensionality reduction method to be used (default: pca)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "pca") { params.cvector_dimre_method = DIMRE_METHOD_PCA; }
            else if (value == "mean") { params.cvector_dimre_method = DIMRE_METHOD_MEAN; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_CVECTOR_GENERATOR}));
    add_opt(common_arg(
        {"--output-format"}, "{md,jsonl}",
        "output format for batched-bench results (default: md)",
        [](common_params & params, const std::string & value) {
            /**/ if (value == "jsonl") { params.batched_bench_output_jsonl = true; }
            else if (value == "md") { params.batched_bench_output_jsonl = false; }
            else { throw std::invalid_argument("invalid value"); }
        }
    ).set_examples({LLAMA_EXAMPLE_BENCH}));
    add_opt(common_arg(
        {"--log-disable"},
        "Log disable",
        [](common_params &) {
            common_log_pause(common_log_main());
        }
    ));
    add_opt(common_arg(
        {"--log-file"}, "FNAME",
        "Log to file",
        [](common_params &, const std::string & value) {
            common_log_set_file(common_log_main(), value.c_str());
        }
    ).set_env("LLAMA_LOG_FILE"));
    add_opt(common_arg(
        {"--log-colors"}, "[on|off|auto]",
        "Set colored logging ('on', 'off', or 'auto', default: 'auto')\n"
        "'auto' enables colors when output is to a terminal",
        [](common_params &, const std::string & value) {
            if (is_truthy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_ENABLED);
            } else if (is_falsey(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_DISABLED);
            } else if (is_autoy(value)) {
                common_log_set_colors(common_log_main(), LOG_COLORS_AUTO);
            } else {
                throw std::invalid_argument(
                    string_format("error: unknown value for --log-colors: '%s'\n", value.c_str()));
            }
        }
    ).set_env("LLAMA_LOG_COLORS"));
    add_opt(common_arg(
        {"-v", "--verbose", "--log-verbose"},
        "Set verbosity level to infinity (i.e. log all messages, useful for debugging)",
        [](common_params & params) {
            params.verbosity = INT_MAX;
        }
    ));
    add_opt(common_arg(
        {"--offline"},
        "Offline mode: forces use of cache, prevents network access",
        [](common_params & params) {
            params.offline = true;
        }
    ).set_env("LLAMA_OFFLINE"));
    add_opt(common_arg(
        {"-lv", "--verbosity", "--log-verbosity"}, "N",
        string_format("Set the verbosity threshold. Messages with a higher verbosity will be ignored. Values:\n"
            " - 0: generic output\n"
            " - 1: error\n"
            " - 2: warning\n"
            " - 3: info\n"
            " - 4: debug\n"
            "(default: %d)\n", params.verbosity),
        [](common_params & params, int value) {
            params.verbosity = value;
        }
    ).set_env("LLAMA_LOG_VERBOSITY"));
    add_opt(common_arg(
        {"--log-prefix"},
        "Enable prefix in log messages",
        [](common_params &) {
            common_log_set_prefix(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_PREFIX"));
    add_opt(common_arg(
        {"--log-timestamps"},
        "Enable timestamps in log messages",
        [](common_params &) {
            common_log_set_timestamps(common_log_main(), true);
        }
    ).set_env("LLAMA_LOG_TIMESTAMPS"));

    // speculative parameters
    add_opt(common_arg(
        {"-td", "--threads-draft"}, "N",
        "number of threads to use during generation (default: same as --threads)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.n_threads = value;
            if (params.speculative.cpuparams.n_threads <= 0) {
                params.speculative.cpuparams.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-tbd", "--threads-batch-draft"}, "N",
        "number of threads to use during batch and prompt processing (default: same as --threads-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.n_threads = value;
            if (params.speculative.cpuparams_batch.n_threads <= 0) {
                params.speculative.cpuparams_batch.n_threads = std::thread::hardware_concurrency();
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"-Cd", "--cpu-mask-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Crd", "--cpu-range-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams.cpumask)) {
                throw std::invalid_argument("invalid range");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--cpu-strict-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: same as --cpu-strict)",
        [](common_params & params, int value) {
            params.speculative.cpuparams.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--prio-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--poll-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: same as --poll])",
        [](common_params & params, int value) {
            params.speculative.cpuparams.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Cbd", "--cpu-mask-batch-draft"}, "M",
        "Draft model CPU affinity mask. Complements cpu-range-draft (default: same as --cpu-mask)",
        [](common_params & params, const std::string & mask) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_mask(mask, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"-Crbd", "--cpu-range-batch-draft"}, "lo-hi",
        "Ranges of CPUs for affinity. Complements --cpu-mask-draft-batch)",
        [](common_params & params, const std::string & range) {
            params.speculative.cpuparams_batch.mask_valid = true;
            if (!parse_cpu_range(range, params.speculative.cpuparams_batch.cpumask)) {
                throw std::invalid_argument("invalid cpumask");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--cpu-strict-batch-draft"}, "<0|1>",
        "Use strict CPU placement for draft model (default: --cpu-strict-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.strict_cpu = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--prio-batch-draft"}, "N",
        string_format("set draft process/thread priority : 0-normal, 1-medium, 2-high, 3-realtime (default: %d)\n", params.speculative.cpuparams_batch.priority),
        [](common_params & params, int prio) {
            if (prio < 0 || prio > 3) {
                throw std::invalid_argument("invalid value");
            }
            params.speculative.cpuparams_batch.priority = (enum ggml_sched_priority) prio;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--poll-batch-draft"}, "<0|1>",
        "Use polling to wait for draft model work (default: --poll-draft)",
        [](common_params & params, int value) {
            params.speculative.cpuparams_batch.poll = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}));
    add_opt(common_arg(
        {"--draft-max", "--draft", "--draft-n"}, "N",
        string_format("number of tokens to draft for speculative decoding (default: %d)", params.speculative.n_max),
        [](common_params & params, int value) {
            params.speculative.n_max = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MAX"));
    add_opt(common_arg(
        {"--draft-min", "--draft-n-min"}, "N",
        string_format("minimum number of draft tokens to use for speculative decoding (default: %d)", params.speculative.n_min),
        [](common_params & params, int value) {
            params.speculative.n_min = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_LOOKUP, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_MIN"));
    add_opt(common_arg(
        {"--draft-p-split"}, "P",
        string_format("speculative decoding split probability (default: %.1f)", (double)params.speculative.p_split),
        [](common_params & params, const std::string & value) {
            params.speculative.p_split = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE}).set_env("LLAMA_ARG_DRAFT_P_SPLIT"));
    add_opt(common_arg(
        {"--draft-p-min"}, "P",
        string_format("minimum speculative decoding probability (greedy) (default: %.1f)", (double)params.speculative.p_min),
        [](common_params & params, const std::string & value) {
            params.speculative.p_min = std::stof(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_DRAFT_P_MIN"));
    add_opt(common_arg(
        {"-cd", "--ctx-size-draft"}, "N",
        string_format("size of the prompt context for the draft model (default: %d, 0 = loaded from model)", params.speculative.n_ctx),
        [](common_params & params, int value) {
            params.speculative.n_ctx = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_CTX_SIZE_DRAFT"));
    add_opt(common_arg(
        {"-devd", "--device-draft"}, "<dev1,dev2,..>",
        "comma-separated list of devices to use for offloading the draft model (none = don't offload)\n"
        "use --list-devices to see a list of available devices",
        [](common_params & params, const std::string & value) {
            params.speculative.devices = parse_device_list(value);
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-ngld", "--gpu-layers-draft", "--n-gpu-layers-draft"}, "N",
        "number of layers to store in VRAM for the draft model",
        [](common_params & params, int value) {
            params.speculative.n_gpu_layers = value;
            if (!llama_supports_gpu_offload()) {
                fprintf(stderr, "warning: no usable GPU found, --gpu-layers-draft option will be ignored\n");
                fprintf(stderr, "warning: one possible reason is that llama.cpp was compiled without GPU support\n");
                fprintf(stderr, "warning: consult docs/build.md for compilation instructions\n");
            }
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_N_GPU_LAYERS_DRAFT"));
    add_opt(common_arg(
        {"-md", "--model-draft"}, "FNAME",
        "draft model for speculative decoding (default: unused)",
        [](common_params & params, const std::string & value) {
            params.speculative.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_MODEL_DRAFT"));
    add_opt(common_arg(
        {"--spec-replace"}, "TARGET", "DRAFT",
        "translate the string in TARGET into DRAFT if the draft model and main model are not compatible",
        [](common_params & params, const std::string & tgt, const std::string & dft) {
            params.speculative.replacements.push_back({ tgt, dft });
        }
    ).set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
    add_opt(common_arg(
        {"-ctkd", "--cache-type-k-draft"}, "TYPE",
        string_format(
            "KV cache data type for K for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.speculative.cache_type_k)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.cache_type_k = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_K_DRAFT"));
    add_opt(common_arg(
        {"-ctvd", "--cache-type-v-draft"}, "TYPE",
        string_format(
            "KV cache data type for V for the draft model\n"
            "allowed values: %s\n"
            "(default: %s)",
            get_all_kv_cache_types().c_str(),
            ggml_type_name(params.speculative.cache_type_v)
        ),
        [](common_params & params, const std::string & value) {
            params.speculative.cache_type_v = kv_cache_type_from_str(value);
        }
    ).set_env("LLAMA_ARG_CACHE_TYPE_V_DRAFT"));

    add_opt(common_arg(
        {"-mv", "--model-vocoder"}, "FNAME",
        "vocoder model for audio generation (default: unused)",
        [](common_params & params, const std::string & value) {
            params.vocoder.model.path = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_SERVER}));
     add_opt(common_arg(
        {"--tts-use-guide-tokens"},
        "Use guide tokens to improve TTS word recall",
        [](common_params & params) {
            params.vocoder.use_guide_tokens = true;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS, LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--tts-speaker-file"}, "FNAME",
        "speaker file path for audio generation",
        [](common_params & params, const std::string & value) {
            params.vocoder.speaker_file = value;
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    add_opt(common_arg(
        {"--diffusion-steps"}, "N",
        string_format("number of diffusion steps (default: %d)", params.diffusion.steps),
        [](common_params & params, int value) { params.diffusion.steps = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-visual"},
        string_format("enable visual diffusion mode (show progressive generation) (default: %s)", params.diffusion.visual_mode ? "true" : "false"),
        [](common_params & params) { params.diffusion.visual_mode = true; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-eps"}, "F",
        string_format("epsilon for timesteps (default: %.6f)", (double) params.diffusion.eps),
        [](common_params & params, const std::string & value) { params.diffusion.eps = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-algorithm"}, "N",
        string_format("diffusion algorithm: 0=ORIGIN, 1=ENTROPY_BASED, 2=MARGIN_BASED, 3=RANDOM, 4=LOW_CONFIDENCE (default: %d)", params.diffusion.algorithm),
        [](common_params & params, int value) { params.diffusion.algorithm = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-alg-temp"}, "F",
        string_format("dream algorithm temperature (default: %.3f)", (double) params.diffusion.alg_temp),
        [](common_params & params, const std::string & value) { params.diffusion.alg_temp = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-block-length"}, "N",
        string_format("llada block length for generation (default: %d)", params.diffusion.block_length),
        [](common_params & params, int value) { params.diffusion.block_length = value; }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-cfg-scale"}, "F",
        string_format("llada classifier-free guidance scale (default: %.3f)", (double) params.diffusion.cfg_scale),
        [](common_params & params, const std::string & value) { params.diffusion.cfg_scale = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        {"--diffusion-add-gumbel-noise"}, "F",
        string_format("add gumbel noise to the logits if temp > 0.0 (default: %s)", params.diffusion.add_gumbel_noise ? "true" : "false"),
        [](common_params & params, const std::string & value) { params.diffusion.add_gumbel_noise = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_DIFFUSION }));
    add_opt(common_arg(
        { "-lr", "--learning-rate" }, "ALPHA",
        string_format("adamw or sgd optimizer alpha (default: %.2g); note: sgd alpha recommended ~10x (no momentum)", (double) params.lr.lr0),
        [](common_params & params, const std::string & value) { params.lr.lr0 = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg({ "-lr-min", "--learning-rate-min" }, "ALPHA",
        string_format("(if >0) final learning rate after decay (if -decay-epochs is set, default=%.2g)",
            (double) params.lr.lr_min),
        [](common_params & params, const std::string & value) { params.lr.lr_min = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-decay-epochs", "--learning-rate-decay-epochs"}, "ALPHA",
        string_format("(if >0) decay learning rate to -lr-min after this many epochs (exponential decay, default=%.2g)", (double) params.lr.decay_epochs),
        [](common_params & params, const std::string & value) { params.lr.decay_epochs = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-wd", "--weight-decay"}, "WD",
        string_format("adamw or sgd optimizer weight decay (0 is off; recommend very small e.g. 1e-9) (default: %.2g).", (double) params.lr.wd),
        [](common_params & params, const std::string & value) { params.lr.wd = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-val-split", "--val-split"}, "FRACTION",
        string_format("fraction of data to use as validation set for training (default: %.2g).", (double) params.val_split),
        [](common_params & params, const std::string & value) { params.val_split = std::stof(value); }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-epochs", "--epochs"}, "N",
        string_format("optimizer max # of epochs (default: %d)", params.lr.epochs),
        [](common_params & params, int epochs) { params.lr.epochs = epochs; }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));
    add_opt(common_arg(
        {"-opt", "--optimizer"}, "sgd|adamw", "adamw or sgd",
        [](common_params & params, const std::string & name) {
            params.optimizer = common_opt_get_optimizer(name.c_str());
            if (params.optimizer == GGML_OPT_OPTIMIZER_TYPE_COUNT) {
                throw std::invalid_argument("invalid --optimizer, valid options: adamw, sgd");
            }
        }
    ).set_examples({ LLAMA_EXAMPLE_FINETUNE }));

    // presets
    add_opt(common_arg(
        {"--tts-oute-default"},
        string_format("use default OuteTTS models (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "OuteAI/OuteTTS-0.2-500M-GGUF";
            params.model.hf_file = "OuteTTS-0.2-500M-Q8_0.gguf";
            params.vocoder.model.hf_repo = "ggml-org/WavTokenizer";
            params.vocoder.model.hf_file = "WavTokenizer-Large-75-F16.gguf";
        }
    ).set_examples({LLAMA_EXAMPLE_TTS}));

    add_opt(common_arg(
        {"--embd-gemma-default"},
        string_format("use default EmbeddingGemma model (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/embeddinggemma-300M-qat-q4_0-GGUF";
            params.model.hf_file = "embeddinggemma-300M-qat-Q4_0.gguf";
            params.port = 8011;
            params.n_ubatch = 2048;
            params.n_batch = 2048;
            params.n_parallel = 32;
            params.n_ctx = 2048*params.n_parallel;
            params.verbose_prompt = true;
            params.embedding = true;
        }
    ).set_examples({LLAMA_EXAMPLE_EMBEDDING, LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-1.5b-default"},
        string_format("use default Qwen 2.5 Coder 1.5B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-1.5B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-1.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-3b-default"},
        string_format("use default Qwen 2.5 Coder 3B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-3B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-3b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-default"},
        string_format("use default Qwen 2.5 Coder 7B (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-7b-spec"},
        string_format("use Qwen 2.5 Coder 7B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-7b-q8_0.gguf";
            params.speculative.model.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.model.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-14b-spec"},
        string_format("use Qwen 2.5 Coder 14B + 0.5B draft for speculative decoding (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen2.5-Coder-14B-Q8_0-GGUF";
            params.model.hf_file = "qwen2.5-coder-14b-q8_0.gguf";
            params.speculative.model.hf_repo = "ggml-org/Qwen2.5-Coder-0.5B-Q8_0-GGUF";
            params.speculative.model.hf_file = "qwen2.5-coder-0.5b-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--fim-qwen-30b-default"},
        string_format("use default Qwen 3 Coder 30B A3B Instruct (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/Qwen3-Coder-30B-A3B-Instruct-Q8_0-GGUF";
            params.model.hf_file = "qwen3-coder-30b-a3b-instruct-q8_0.gguf";
            params.port = 8012;
            params.n_ubatch = 1024;
            params.n_batch = 1024;
            params.n_ctx = 0;
            params.n_cache_reuse = 256;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER}));

    add_opt(common_arg(
        {"--gpt-oss-20b-default"},
        string_format("use gpt-oss-20b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-20b-GGUF";
            params.model.hf_file = "gpt-oss-20b-mxfp4.gguf";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
            //params.default_template_kwargs["reasoning_effort"] = "\"high\"";
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--gpt-oss-120b-default"},
        string_format("use gpt-oss-120b (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gpt-oss-120b-GGUF";
            params.port = 8013;
            params.n_ubatch = 2048;
            params.n_batch = 32768;
            params.n_parallel = 2;
            params.n_ctx = 131072*params.n_parallel;
            params.sampling.temp = 1.0f;
            params.sampling.top_p = 1.0f;
            params.sampling.top_k = 0;
            params.sampling.min_p = 0.01f;
            params.use_jinja = true;
            //params.default_template_kwargs["reasoning_effort"] = "\"high\"";
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-4b-default"},
        string_format("use Gemma 3 4B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-4b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    add_opt(common_arg(
        {"--vision-gemma-12b-default"},
        string_format("use Gemma 3 12B QAT (note: can download weights from the internet)"),
        [](common_params & params) {
            params.model.hf_repo = "ggml-org/gemma-3-12b-it-qat-GGUF";
            params.port = 8014;
            params.n_ctx = 0;
            params.use_jinja = true;
        }
    ).set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));

    return ctx_arg;
}

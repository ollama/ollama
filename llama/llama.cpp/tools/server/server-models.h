#pragma once

#include "common.h"
#include "preset.h"
#include "server-http.h"

#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>

/**
 * state diagram:
 *
 * UNLOADED ──► LOADING ──► LOADED
 *  ▲            │            │
 *  └───failed───┘            │
 *  ▲                         │
 *  └────────unloaded─────────┘
 */
enum server_model_status {
    // TODO: also add downloading state when the logic is added
    SERVER_MODEL_STATUS_UNLOADED,
    SERVER_MODEL_STATUS_LOADING,
    SERVER_MODEL_STATUS_LOADED
};

static server_model_status server_model_status_from_string(const std::string & status_str) {
    if (status_str == "unloaded") {
        return SERVER_MODEL_STATUS_UNLOADED;
    }
    if (status_str == "loading") {
        return SERVER_MODEL_STATUS_LOADING;
    }
    if (status_str == "loaded") {
        return SERVER_MODEL_STATUS_LOADED;
    }
    throw std::runtime_error("invalid server model status");
}

static std::string server_model_status_to_string(server_model_status status) {
    switch (status) {
        case SERVER_MODEL_STATUS_UNLOADED: return "unloaded";
        case SERVER_MODEL_STATUS_LOADING:  return "loading";
        case SERVER_MODEL_STATUS_LOADED:   return "loaded";
        default:                           return "unknown";
    }
}

struct server_model_meta {
    common_preset preset;
    std::string name;
    std::string path;
    std::string path_mmproj; // only available if in_cache=false
    bool in_cache = false; // if true, use -hf; use -m otherwise
    int port = 0;
    server_model_status status = SERVER_MODEL_STATUS_UNLOADED;
    int64_t last_used = 0; // for LRU unloading
    std::vector<std::string> args; // args passed to the model instance, will be populated by render_args()
    int exit_code = 0; // exit code of the model instance process (only valid if status == FAILED)

    bool is_active() const {
        return status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_LOADING;
    }

    bool is_failed() const {
        return status == SERVER_MODEL_STATUS_UNLOADED && exit_code != 0;
    }
};

// the server_presets struct holds the presets read from presets.ini
// as well as base args from the router server
struct server_presets {
    common_presets presets;
    common_params_context ctx_params;
    std::map<common_arg, std::string> base_args;
    std::map<std::string, common_arg> control_args; // args reserved for server control

    server_presets(int argc, char ** argv, common_params & base_params, const std::string & models_dir);
    common_preset get_preset(const std::string & name);
    void render_args(server_model_meta & meta);
};

struct subprocess_s;

struct server_models {
private:
    struct instance_t {
        std::shared_ptr<subprocess_s> subproc; // shared between main thread and monitoring thread
        std::thread th;
        server_model_meta meta;
        FILE * stdin_file = nullptr;
    };

    std::mutex mutex;
    std::condition_variable cv;
    std::map<std::string, instance_t> mapping;

    common_params base_params;
    std::vector<std::string> base_args;
    std::vector<std::string> base_env;

    server_presets presets;

    void update_meta(const std::string & name, const server_model_meta & meta);

    // unload least recently used models if the limit is reached
    void unload_lru();

    // not thread-safe, caller must hold mutex
    void add_model(server_model_meta && meta);

public:
    server_models(const common_params & params, int argc, char ** argv, char ** envp);

    void load_models();

    // check if a model instance exists
    bool has_model(const std::string & name);

    // return a copy of model metadata
    std::optional<server_model_meta> get_meta(const std::string & name);

    // return a copy of all model metadata
    std::vector<server_model_meta> get_all_meta();

    void load(const std::string & name);
    void unload(const std::string & name);
    void unload_all();

    // update the status of a model instance
    void update_status(const std::string & name, server_model_status status);

    // wait until the model instance is fully loaded
    // return when the model is loaded or failed to load
    void wait_until_loaded(const std::string & name);

    // load the model if not loaded, otherwise do nothing
    // return false if model is already loaded; return true otherwise (meta may need to be refreshed)
    bool ensure_model_loaded(const std::string & name);

    // proxy an HTTP request to the model instance
    server_http_res_ptr proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used);

    // notify the router server that a model instance is ready
    // return the monitoring thread (to be joined by the caller)
    static std::thread setup_child_server(const common_params & base_params, int router_port, const std::string & name, std::function<void(int)> & shutdown_handler);
};

struct server_models_routes {
    common_params params;
    server_models models;
    server_models_routes(const common_params & params, int argc, char ** argv, char ** envp)
            : params(params), models(params, argc, argv, envp) {
        init_routes();
    }

    void init_routes();
    // handlers using lambda function, so that they can capture `this` without `std::bind`
    server_http_context::handler_t get_router_props;
    server_http_context::handler_t proxy_get;
    server_http_context::handler_t proxy_post;
    server_http_context::handler_t get_router_models;
    server_http_context::handler_t post_router_models_load;
    server_http_context::handler_t post_router_models_status;
    server_http_context::handler_t post_router_models_unload;
};

/**
 * A simple HTTP proxy that forwards requests to another server
 * and relays the responses back.
 */
struct server_http_proxy : server_http_res {
    std::function<void()> cleanup = nullptr;
public:
    server_http_proxy(const std::string & method,
                      const std::string & host,
                      int port,
                      const std::string & path,
                      const std::map<std::string, std::string> & headers,
                      const std::string & body,
                      const std::function<bool()> should_stop);
    ~server_http_proxy() {
        if (cleanup) {
            cleanup();
        }
    }
private:
    std::thread thread;
    struct msg_t {
        std::map<std::string, std::string> headers;
        int status = 0;
        std::string data;
        std::string content_type;
    };
};

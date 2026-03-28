#include "server-common.h"
#include "server-models.h"

#include "preset.h"
#include "download.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <sheredom/subprocess.h>

#include <functional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define CMD_EXIT "exit"

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

struct local_model {
    std::string name;
    std::string path;
    std::string path_mmproj;
};

static std::vector<local_model> list_local_models(const std::string & dir) {
    if (!std::filesystem::exists(dir) || !std::filesystem::is_directory(dir)) {
        throw std::runtime_error(string_format("error: '%s' does not exist or is not a directory\n", dir.c_str()));
    }

    std::vector<local_model> models;
    auto scan_subdir = [&models](const std::string & subdir_path, const std::string & name) {
        auto files = fs_list(subdir_path, false);
        common_file_info model_file;
        common_file_info first_shard_file;
        common_file_info mmproj_file;
        for (const auto & file : files) {
            if (string_ends_with(file.name, ".gguf")) {
                if (file.name.find("mmproj") != std::string::npos) {
                    mmproj_file = file;
                } else if (file.name.find("-00001-of-") != std::string::npos) {
                    first_shard_file = file;
                } else {
                    model_file = file;
                }
            }
        }
        // single file model
        local_model model{
            /* name        */ name,
            /* path        */ first_shard_file.path.empty() ? model_file.path : first_shard_file.path,
            /* path_mmproj */ mmproj_file.path // can be empty
        };
        if (!model.path.empty()) {
            models.push_back(model);
        }
    };

    auto files = fs_list(dir, true);
    for (const auto & file : files) {
        if (file.is_dir) {
            scan_subdir(file.path, file.name);
        } else if (string_ends_with(file.name, ".gguf")) {
            // single file model
            std::string name = file.name;
            string_replace_all(name, ".gguf", "");
            local_model model{
                /* name        */ name,
                /* path        */ file.path,
                /* path_mmproj */ ""
            };
            models.push_back(model);
        }
    }
    return models;
}

//
// server_presets
//


server_presets::server_presets(int argc, char ** argv, common_params & base_params, const std::string & presets_path)
        : ctx_params(common_params_parser_init(base_params, LLAMA_EXAMPLE_SERVER)) {
    if (!presets_path.empty()) {
        presets = common_presets_load(presets_path, ctx_params);
        SRV_INF("Loaded %zu presets from %s\n", presets.size(), presets_path.c_str());
    }

    // populate reserved args (will be appended by the router)
    for (auto & opt : ctx_params.options) {
        if (opt.env == nullptr) {
            continue;
        }
        std::string env = opt.env;
        if (env == "LLAMA_ARG_PORT" ||
            env == "LLAMA_ARG_HOST" ||
            env == "LLAMA_ARG_ALIAS" ||
            env == "LLAMA_ARG_API_KEY" ||
            env == "LLAMA_ARG_MODELS_DIR" ||
            env == "LLAMA_ARG_MODELS_MAX" ||
            env == "LLAMA_ARG_MODELS_PRESET" ||
            env == "LLAMA_ARG_MODEL" ||
            env == "LLAMA_ARG_MMPROJ" ||
            env == "LLAMA_ARG_HF_REPO" ||
            env == "LLAMA_ARG_NO_MODELS_AUTOLOAD") {
            control_args[env] = opt;
        }
    }

    // read base args from router's argv
    common_params_to_map(argc, argv, LLAMA_EXAMPLE_SERVER, base_args);

    // remove any router-controlled args from base_args
    for (const auto & cargs : control_args) {
        auto it = base_args.find(cargs.second);
        if (it != base_args.end()) {
            base_args.erase(it);
        }
    }
}

common_preset server_presets::get_preset(const std::string & name) {
    auto it = presets.find(name);
    if (it != presets.end()) {
        return it->second;
    }
    return common_preset();
}

void server_presets::render_args(server_model_meta & meta) {
    common_preset preset = meta.preset; // copy
    // merging 3 kinds of args:
    // 1. model-specific args (from preset)
    // force removing control args if any
    for (auto & cargs : control_args) {
        if (preset.options.find(cargs.second) != preset.options.end()) {
            SRV_WRN("Preset '%s' contains reserved arg '%s', removing it\n", preset.name.c_str(), cargs.second.args[0]);
            preset.options.erase(cargs.second);
        }
    }
    // 2. base args (from router)
    // inherit from base args
    for (const auto & [arg, value] : base_args) {
        preset.options[arg] = value;
    }
    // 3. control args (from router)
    // set control values
    preset.options[control_args["LLAMA_ARG_HOST"]] = CHILD_ADDR;
    preset.options[control_args["LLAMA_ARG_PORT"]] = std::to_string(meta.port);
    preset.options[control_args["LLAMA_ARG_ALIAS"]] = meta.name;
    if (meta.in_cache) {
        preset.options[control_args["LLAMA_ARG_HF_REPO"]] = meta.name;
    } else {
        preset.options[control_args["LLAMA_ARG_MODEL"]] = meta.path;
        if (!meta.path_mmproj.empty()) {
            preset.options[control_args["LLAMA_ARG_MMPROJ"]] = meta.path_mmproj;
        }
    }
    meta.args = preset.to_args();
    // add back the binary path at the front
    meta.args.insert(meta.args.begin(), get_server_exec_path().string());
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv,
        char ** envp) : base_params(params), presets(argc, argv, base_params, params.models_preset) {
    for (int i = 0; i < argc; i++) {
        base_args.push_back(std::string(argv[i]));
    }
    for (char ** env = envp; *env != nullptr; env++) {
        base_env.push_back(std::string(*env));
    }
    GGML_ASSERT(!base_args.empty());
    // set binary path
    try {
        base_args[0] = get_server_exec_path().string();
    } catch (const std::exception & e) {
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", base_args[0].c_str());
    }
    load_models();
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }
    presets.render_args(meta); // populate meta.args
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<subprocess_s>(),
        /* th      */ std::thread(),
        /* meta    */ std::move(meta)
    };
}

static std::vector<local_model> list_custom_path_models(server_presets & presets) {
    // detect any custom-path models in presets
    std::vector<local_model> custom_models;
    for (auto & [model_name, preset] : presets.presets) {
        local_model model;
        model.name = model_name;
        std::vector<common_arg> to_erase;
        for (auto & [arg, value] : preset.options) {
            std::string env(arg.env ? arg.env : "");
            if (env == "LLAMA_ARG_MODEL") {
                model.path = value;
                to_erase.push_back(arg);
            }
            if (env == "LLAMA_ARG_MMPROJ") {
                model.path_mmproj = value;
                to_erase.push_back(arg);
            }
        }
        for (auto & arg : to_erase) {
            preset.options.erase(arg);
        }
        if (!model.name.empty() && !model.path.empty()) {
            custom_models.push_back(model);
        }
    }
    return custom_models;
}

// TODO: allow refreshing cached model list
void server_models::load_models() {
    // loading models from 3 sources:
    // 1. cached models
    auto cached_models = common_list_cached_models();
    for (const auto & model : cached_models) {
        server_model_meta meta{
            /* preset      */ presets.get_preset(model.to_string()),
            /* name        */ model.to_string(),
            /* path        */ model.manifest_path,
            /* path_mmproj */ "", // auto-detected when loading
            /* in_cache    */ true,
            /* port        */ 0,
            /* status      */ SERVER_MODEL_STATUS_UNLOADED,
            /* last_used   */ 0,
            /* args        */ std::vector<std::string>(),
            /* exit_code   */ 0
        };
        add_model(std::move(meta));
    }
    // 2. local models specificed via --models-dir
    if (!base_params.models_dir.empty()) {
        auto local_models = list_local_models(base_params.models_dir);
        for (const auto & model : local_models) {
            if (mapping.find(model.name) != mapping.end()) {
                // already exists in cached models, skip
                continue;
            }
            server_model_meta meta{
                /* preset      */ presets.get_preset(model.name),
                /* name        */ model.name,
                /* path        */ model.path,
                /* path_mmproj */ model.path_mmproj,
                /* in_cache    */ false,
                /* port        */ 0,
                /* status      */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used   */ 0,
                /* args        */ std::vector<std::string>(),
                /* exit_code   */ 0
            };
            add_model(std::move(meta));
        }
    }
    // 3. custom-path models specified in presets
    auto custom_models = list_custom_path_models(presets);
    for (const auto & model : custom_models) {
        server_model_meta meta{
            /* preset      */ presets.get_preset(model.name),
            /* name        */ model.name,
            /* path        */ model.path,
            /* path_mmproj */ model.path_mmproj,
            /* in_cache    */ false,
            /* port        */ 0,
            /* status      */ SERVER_MODEL_STATUS_UNLOADED,
            /* last_used   */ 0,
            /* args        */ std::vector<std::string>(),
            /* exit_code   */ 0
        };
        add_model(std::move(meta));
    }
    // log available models
    SRV_INF("Available models (%zu) (*: custom preset)\n", mapping.size());
    for (const auto & [name, inst] : mapping) {
        SRV_INF("  %c %s\n", inst.meta.preset.name.empty() ? ' ' : '*', name.c_str());
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loaded
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    return mapping.find(name) != mapping.end();
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    return std::nullopt;
}

static int get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (getsockname(sock, (struct sockaddr*)&serv_addr, &namelen) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    int port = ntohs(serv_addr.sin_port);

    CLOSE_SOCKET(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    return port;
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    std::string lru_model_name = "";
    int64_t lru_last_used = ggml_time_ms();
    size_t count_active = 0;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (const auto & m : mapping) {
            if (m.second.meta.is_active()) {
                count_active++;
                if (m.second.meta.last_used < lru_last_used) {
                    lru_model_name = m.first;
                    lru_last_used = m.second.meta.last_used;
                }
            }
        }
    }
    if (!lru_model_name.empty() && count_active >= (size_t)base_params.models_max) {
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        unload(lru_model_name);
    }
}

void server_models::load(const std::string & name) {
    if (!has_model(name)) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    unload_lru();

    std::lock_guard<std::mutex> lk(mutex);

    auto meta = mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    // prepare new instance info
    instance_t inst;
    inst.meta           = meta;
    inst.meta.port      = get_free_port();
    inst.meta.status    = SERVER_MODEL_STATUS_LOADING;
    inst.meta.last_used = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<subprocess_s>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        presets.render_args(inst.meta); // update meta.args

        std::vector<std::string> child_args = inst.meta.args; // copy
        std::vector<std::string> child_env  = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        int result = subprocess_create_ex(argv.data(), options, envp.data(), inst.subproc.get());
        if (result != 0) {
            throw std::runtime_error("failed to spawn server instance");
        }

        inst.stdin_file = subprocess_stdin(inst.subproc.get());
    }

    // start a thread to manage the child process
    // captured variables are guaranteed to be destroyed only after the thread is joined
    inst.th = std::thread([this, name, child_proc = inst.subproc, port = inst.meta.port]() {
        // read stdout/stderr and forward to main server log
        FILE * p_stdout_stderr = subprocess_stdout(child_proc.get());
        if (p_stdout_stderr) {
            char buffer[4096];
            while (fgets(buffer, sizeof(buffer), p_stdout_stderr) != nullptr) {
                LOG("[%5d] %s", port, buffer);
            }
        } else {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
        }
        // we reach here when the child process exits
        int exit_code = 0;
        subprocess_join(child_proc.get(), &exit_code);
        subprocess_destroy(child_proc.get());
        // update PID and status
        {
            std::lock_guard<std::mutex> lk(mutex);
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                auto & meta = it->second.meta;
                meta.exit_code = exit_code;
                meta.status    = SERVER_MODEL_STATUS_UNLOADED;
            }
            cv.notify_all();
        }
        SRV_INF("instance name=%s exited with status %d\n", name.c_str(), exit_code);
    });

    // clean up old process/thread if exists
    {
        auto & old_instance = mapping[name];
        // old process should have exited already, but just in case, we clean it up here
        if (subprocess_alive(old_instance.subproc.get())) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            subprocess_terminate(old_instance.subproc.get()); // force kill
        }
        if (old_instance.th.joinable()) {
            old_instance.th.join();
        }
    }

    mapping[name] = std::move(inst);
    cv.notify_all();
}

static void interrupt_subprocess(FILE * stdin_file) {
    // because subprocess.h does not provide a way to send SIGINT,
    // we will send a command to the child process to exit gracefully
    if (stdin_file) {
        fprintf(stdin_file, "%s\n", CMD_EXIT);
        fflush(stdin_file);
    }
}

void server_models::unload(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.is_active()) {
            SRV_INF("unloading model instance name=%s\n", name.c_str());
            interrupt_subprocess(it->second.stdin_file);
            // status change will be handled by the managing thread
        } else {
            SRV_WRN("model instance name=%s is not loaded\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_active()) {
                SRV_INF("unloading model instance name=%s\n", name.c_str());
                interrupt_subprocess(inst.stdin_file);
                // status change will be handled by the managing thread
            }
            // moving the thread to join list to avoid deadlock
            to_join.push_back(std::move(inst.th));
        }
    }
    for (auto & th : to_join) {
        if (th.joinable()) {
            th.join();
        }
    }
}

void server_models::update_status(const std::string & name, server_model_status status) {
    // for now, we only allow updating to LOADED status
    if (status != SERVER_MODEL_STATUS_LOADED) {
        throw std::runtime_error("invalid status value");
    }
    auto meta = get_meta(name);
    if (meta.has_value()) {
        meta->status = status;
        update_meta(name, meta.value());
    }
}

void server_models::wait_until_loaded(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this, &name]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return it->second.meta.status != SERVER_MODEL_STATUS_LOADING;
        }
        return false;
    });
}

bool server_models::ensure_model_loaded(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->status == SERVER_MODEL_STATUS_LOADED) {
        return false; // already loaded
    }
    if (meta->status == SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model name=%s is not loaded, loading...\n", name.c_str());
        load(name);
    }

    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    wait_until_loaded(name);

    // check final status
    meta = get_meta(name);
    if (!meta.has_value() || meta->is_failed()) {
        throw std::runtime_error("model name=" + name + " failed to load");
    }

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->status != SERVER_MODEL_STATUS_LOADED) {
        throw std::invalid_argument("model name=" + name + " is not loaded");
    }
    if (update_last_used) {
        std::unique_lock<std::mutex> lk(mutex);
        mapping[name].meta.last_used = ggml_time_ms();
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            CHILD_ADDR,
            meta->port,
            req.path,
            req.headers,
            req.body,
            req.should_stop);
    return proxy;
}

std::thread server_models::setup_child_server(const common_params & base_params, int router_port, const std::string & name, std::function<void(int)> & shutdown_handler) {
    // send a notification to the router server that a model instance is ready
    // TODO @ngxson : use HTTP client from libcommon
    httplib::Client cli(base_params.hostname, router_port);
    cli.set_connection_timeout(0, 200000); // 200 milliseconds

    httplib::Request req;
    req.method = "POST";
    req.path   = "/models/status";
    req.set_header("Content-Type", "application/json");
    if (!base_params.api_keys.empty()) {
        req.set_header("Authorization", "Bearer " + base_params.api_keys[0]);
    }

    json body;
    body["model"] = name;
    body["value"] = server_model_status_to_string(SERVER_MODEL_STATUS_LOADED);
    req.body = body.dump();

    SRV_INF("notifying router server (port=%d) that model %s is ready\n", router_port, name.c_str());
    auto result = cli.send(std::move(req));
    if (result.error() != httplib::Error::Success) {
        auto err_str = httplib::to_string(result.error());
        SRV_ERR("failed to notify router server: %s\n", err_str.c_str());
        exit(1); // force exit
    }

    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}



//
// server_models_routes
//

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

static bool router_validate_model(const std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response("model not found", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    if (models_autoload) {
        models.ensure_model_loaded(name);
    } else {
        if (meta->status != SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req) {
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

void server_models_routes::init_routes() {
    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",          "router"},
                {"max_instances", 4}, // dummy value for testing
                // this is a dummy response to make sure webui doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        bool autoload = is_autoload(params, req);
        auto error_res = std::make_unique<server_http_res>();
        if (!router_validate_model(name, models, autoload, error_res)) {
            return error_res;
        }
        return models.proxy_request(req, method, name, true); // update last usage for POST request only
    };

    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        if (model->status == SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is already loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.load(name);
        res_ok(res, {{"success", true}});
        return res;
    };

    // used by child process to notify the router about status change
    // TODO @ngxson : maybe implement authentication for this endpoint in the future
    this->post_router_models_status = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string model = json_value(body, "model", std::string());
        std::string value = json_value(body, "value", std::string());
        models.update_status(model, server_model_status_from_string(value));
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models = [this](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                status["preset"] = meta.preset.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }
            models_json.push_back(json {
                {"id",       meta.name},
                {"object",   "model"},    // for OAI-compat
                {"owned_by", "llamacpp"}, // for OAI-compat
                {"created",  t},          // for OAI-compat
                {"in_cache", meta.in_cache},
                {"path",     meta.path},
                {"status",   status},
                // TODO: add other fields, may require reading GGUF metadata
            });
        }
        res_ok(res, {
            {"data", models_json},
            {"object", "list"},
        });
        return res;
    };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (model->status != SERVER_MODEL_STATUS_LOADED) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(name);
        res_ok(res, {{"success", true}});
        return res;
    };
}



//
// server_http_proxy
//

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true, std::memory_order_relaxed);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::function<bool()> should_stop) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::Client>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    // setup Client
    cli->set_connection_timeout(0, 200000); // 200 milliseconds
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""});
    };

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            req.set_header(key, value);
        }
        req.body = body;
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str, ""}); // body
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}

#include <stdexcept>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-rpc.h"

#include "utils.h"

std::vector<std::string> string_split(const std::string & input, char separator)
{
    std::vector<std::string> parts;
    size_t begin_pos = 0;
    size_t separator_pos = input.find(separator);
    while (separator_pos != std::string::npos) {
        std::string part = input.substr(begin_pos, separator_pos - begin_pos);
        parts.emplace_back(part);
        begin_pos = separator_pos + 1;
        separator_pos = input.find(separator, begin_pos);
    }
    parts.emplace_back(input.substr(begin_pos, separator_pos - begin_pos));
    return parts;
}

void add_rpc_devices(const char* const input_servers) {
    std::string servers = input_servers;
    auto rpc_servers = string_split(servers, ',');
    if (rpc_servers.empty()) {
        throw std::invalid_argument("no RPC servers specified");
    }
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        throw std::invalid_argument("failed to find RPC backend");
    }
    typedef ggml_backend_dev_t (*ggml_backend_rpc_add_device_t)(const char * endpoint);
    ggml_backend_rpc_add_device_t ggml_backend_rpc_add_device_fn = (ggml_backend_rpc_add_device_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_device");
    if (!ggml_backend_rpc_add_device_fn) {
        throw std::invalid_argument("failed to find RPC device add function");
    }
    for (const auto & server : rpc_servers) {
        ggml_backend_dev_t dev = ggml_backend_rpc_add_device_fn(server.c_str());
        if (dev) {
            ggml_backend_device_register(dev);
        } else {
            throw std::invalid_argument("failed to register RPC device");
        }
    }
}

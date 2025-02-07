#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <filesystem>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_KOMPUTE
#include "ggml-kompute.h"
#endif

// disable C++17 deprecation warning for std::codecvt_utf8
#if defined(__clang__)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

static std::wstring utf8_to_utf16(const std::string & str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.from_bytes(str);
}

static std::string utf16_to_utf8(const std::wstring & str) {
    std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
    return converter.to_bytes(str);
}

#if defined(__clang__)
#    pragma clang diagnostic pop
#endif

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

static dl_handle * dl_load_library(const std::wstring & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    HMODULE handle = LoadLibraryW(path.c_str());

    SetErrorMode(old_mode);

    return handle;
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

static void * dl_load_library(const std::wstring & path) {
    dl_handle * handle = dlopen(utf16_to_utf8(path).c_str(), RTLD_NOW | RTLD_LOCAL);

    return handle;
}

static void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<std::pair<ggml_backend_dev_t, int>> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
        register_backend(ggml_backend_vk_reg());
#endif
#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
// #ifdef GGML_USE_BLAS
//         register_backend(ggml_backend_blas_reg());
// #endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_KOMPUTE
        register_backend(ggml_backend_kompute_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        for (auto & entry : backends) {
            if (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, int score = -1, dl_handle_ptr handle = nullptr) {
        if (!reg) {
            return;
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i), score);
        }
    }

    void register_device(ggml_backend_dev_t device, int score = -1) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back({device, score});
    }

    ggml_backend_reg_t load_backend(const std::wstring & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        if (!handle) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(path).c_str());
            }
            return nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        if (score_fn && score_fn() == 0) {
            if (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, utf16_to_utf8(path).c_str());
            }
            return nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        if (!backend_init_fn) {
            if (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, utf16_to_utf8(path).c_str());
            }
            return nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        if (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            if (!silent) {
                if (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n", __func__, utf16_to_utf8(path).c_str());
                } else {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, utf16_to_utf8(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            return nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), utf16_to_utf8(path).c_str());

        register_backend(reg, score_fn ? score_fn() : -1, std::move(handle));

        return reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { return entry.reg == reg; });

        if (it == backends.end()) {
            if (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            return;
        }

        if (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](std::pair<ggml_backend_dev_t, int> dev) { return ggml_backend_dev_backend_reg(dev.first) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    return reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_count() {
    return get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    return get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        if (striequals(ggml_backend_reg_name(reg), name)) {
            return reg;
        }
    }
    return nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    return get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    auto devices = get_reg().devices;
    if (!std::is_heap(devices.begin(), devices.end())) {
        std::make_heap(devices.begin(), devices.end(), [](const auto & a, const auto & b) { return a.second < b.second; });
    }

    return devices[index].first;
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (striequals(ggml_backend_dev_name(dev), name)) {
            return dev;
        }
    }
    return nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == type) {
            return dev;
        }
    }
    return nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (!dev) {
        return nullptr;
    }
    return ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    return get_reg().load_backend(utf8_to_utf16(path), false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static std::wstring get_executable_path() {
#if defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    while (true) {
        size = path.size();
        if (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return utf8_to_utf16(base_path + "/");
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    while (true) {
        // get executable path
#    if defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        if (len == -1) {
            break;
        }
        if (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            if (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    return utf8_to_utf16(base_path + "/");
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    if (len == 0) {
        return {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    if (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    return base_path + L"\\";
#else
    return {};
#endif
}

static std::wstring backend_filename_prefix() {
#ifdef _WIN32
    return L"ggml-";
#else
    return L"libggml-";
#endif
}

static std::wstring backend_filename_suffix() {
#ifdef _WIN32
    return L".dll";
#else
    return L".so";
#endif
}

static std::wstring path_separator() {
#ifdef _WIN32
    return L"\\";
#else
    return L"/";
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
     // TODO: search system paths
    std::wstring file_prefix = backend_filename_prefix() + utf8_to_utf16(name) + L"-";
    std::vector<std::wstring> search_paths;
    if (user_search_path == nullptr) {
        search_paths.push_back(L"." + path_separator());
        search_paths.push_back(get_executable_path());
    } else {
        search_paths.push_back(utf8_to_utf16(user_search_path) + path_separator());
    }

    int best_score = 0;
    std::wstring best_path;

    namespace fs = std::filesystem;
    for (const auto & search_path : search_paths) {
        if (!fs::exists(search_path)) {
            continue;
        }
        fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
        for (const auto & entry : dir_it) {
            if (entry.is_regular_file()) {
                std::wstring filename = entry.path().filename().wstring();
                std::wstring ext = entry.path().extension().wstring();
                if (filename.find(file_prefix) == 0 && ext == backend_filename_suffix()) {
                    dl_handle_ptr handle { dl_load_library(entry.path().wstring()) };
                    if (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str());
                    }
                    if (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        if (score_fn) {
                            int s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str(), s);
#endif
                            if (s > best_score) {
                                best_score = s;
                                best_path = entry.path().wstring();
                            }
                        } else {
                            if (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str());
                            }
                        }
                    }
                }
            }
        }
    }

    if (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            std::wstring path = search_path + backend_filename_prefix() + utf8_to_utf16(name) + backend_filename_suffix();
            if (fs::exists(path)) {
                return get_reg().load_backend(path, silent);
            }
        }
        return nullptr;
    }

    return get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#else
    bool silent = false;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("kompute", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
}

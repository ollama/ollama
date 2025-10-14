// DXGI and PDH Performance Counters Library
// This Windows-only library provides accurate VRAM reporting for Intel GPUs

#include "ggml-impl.h"
#include <filesystem>
#include <mutex>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif

namespace fs = std::filesystem;

static std::mutex ggml_dxgi_pdh_lock;

extern "C" {

    int ggml_dxgi_pdh_init() {
        return -1; // change when implemented
    }

    int ggml_dxgi_pdh_get_device_memory(int adapter_idx, size_t *free, size_t *total) {
        return -1; // change when implemented
    }

    void ggml_dxgi_pdh_release() {
        return; // change when implemented
    }

} // extern "C"
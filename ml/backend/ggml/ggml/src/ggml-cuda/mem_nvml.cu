#if !defined(GGML_USE_HIP)

// NVIDIA Management Library (NVML)
//
// https://developer.nvidia.com/management-library-nvml
//
// This library provides accurate VRAM reporting for NVIDIA GPUs, particularly
// on Windows, where the cuda library provides inaccurate VRAM usage metrics. The
// runtime DLL is installed with every driver on Windows, and most Linux
// systems, and the headers are included in the standard CUDA SDK install.  As
// such, we can include the header here to simplify the code.


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

#include <nvml.h>

struct {
  void *handle;
  nvmlReturn_t (*nvmlInit_v2)(void);
  nvmlReturn_t (*nvmlShutdown)(void);
  nvmlReturn_t (*nvmlDeviceGetHandleByUUID)(const char *, nvmlDevice_t *);
  nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
} nvml;
static std::mutex ggml_nvml_lock;

int ggml_nvml_init() {
    std::lock_guard<std::mutex> lock(ggml_nvml_lock);
    if (nvml.handle != NULL) {
        // Already initialized
        return 0;
    }
#ifdef _WIN32
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    fs::path libPath[2];
    const char * programDir = std::getenv("ProgramW6432");
    if (programDir == NULL) {
        libPath[0] = fs::path("Program Files") / fs::path("NVIDIA Corporation") / fs::path("NVSMI") / fs::path("NVML.dll");
    } else {
        libPath[0] = fs::path(programDir) / fs::path("NVIDIA Corporation") / fs::path("NVSMI") / fs::path("NVML.dll");
    }
    libPath[1] = fs::path("\\Windows") / fs::path("System32") / fs::path("NVML.dll");

    for (int i = 0; i < 2; i++) {
        nvml.handle = (void*)LoadLibraryW(libPath[i].wstring().c_str());
        if (nvml.handle != NULL) {
            break;
        }
    }
    if (nvml.handle == NULL) {
        return NVML_ERROR_NOT_FOUND;
    }

    nvml.nvmlInit_v2 = (nvmlReturn_enum (*)()) GetProcAddress((HMODULE)(nvml.handle), "nvmlInit_v2");
    nvml.nvmlShutdown = (nvmlReturn_enum (*)()) GetProcAddress((HMODULE)(nvml.handle), "nvmlShutdown");
    nvml.nvmlDeviceGetHandleByUUID = (nvmlReturn_t (*)(const char *, nvmlDevice_t *)) GetProcAddress((HMODULE)(nvml.handle), "nvmlDeviceGetHandleByUUID");
    nvml.nvmlDeviceGetMemoryInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *)) GetProcAddress((HMODULE)(nvml.handle), "nvmlDeviceGetMemoryInfo");
    if (nvml.nvmlInit_v2 == NULL || nvml.nvmlShutdown == NULL || nvml.nvmlDeviceGetHandleByUUID == NULL || nvml.nvmlDeviceGetMemoryInfo == NULL) {
        GGML_LOG_INFO("%s unable to locate required symbols in NVML.dll");
        FreeLibrary((HMODULE)(nvml.handle));
        nvml.handle = NULL;
        return NVML_ERROR_NOT_FOUND;
    }

    SetErrorMode(old_mode);

#else
    // Not currently wired up on Linux
    return NVML_ERROR_NOT_SUPPORTED;
#endif
    int status = nvml.nvmlInit_v2();
    return NVML_SUCCESS;
}

void ggml_nvml_release() {
    std::lock_guard<std::mutex> lock(ggml_nvml_lock);
    if (nvml.handle == NULL) {
        // Already free
        return;
    }
    nvmlReturn_enum status = nvml.nvmlShutdown();
    if (status != NVML_SUCCESS) {
        GGML_LOG_INFO("%s failed to shutdown NVML: %d\n", __func__, status);
    }
#ifdef _WIN32
    FreeLibrary((HMODULE)(nvml.handle));
    nvml.handle = NULL;
#else
    // Not currently wired up on Linux
#endif
}

int ggml_nvml_get_device_memory(const char *uuid, size_t *free, size_t *total) {
    std::lock_guard<std::mutex> lock(ggml_nvml_lock);
    if (nvml.handle == NULL) {
        return NVML_ERROR_UNINITIALIZED;
    }
    nvmlDevice_t device;
    auto status = nvml.nvmlDeviceGetHandleByUUID(uuid, &device);
    if (status != NVML_SUCCESS) {
        return status;
    }
    nvmlMemory_t memInfo = {0};
    status = nvml.nvmlDeviceGetMemoryInfo(device, &memInfo);
    if (status == NVML_SUCCESS) {
        *free = memInfo.free;
        *total = memInfo.total;
    }
    return status;
}

#endif
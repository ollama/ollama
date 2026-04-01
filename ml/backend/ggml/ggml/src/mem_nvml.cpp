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
#include <array>
#include <cstring>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#    include <fstream>
#    include <string>
#endif

namespace fs = std::filesystem;

// Minimal definitions to avoid including the nvml.h header
typedef enum nvmlReturn_enum
{
    // cppcheck-suppress *
    NVML_SUCCESS = 0,                          //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,              //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,           //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,              //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,              //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5,        //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,                  //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,          //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,         //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,          //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,                   //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,                 //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,         //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13,        //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,         //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,               //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,            //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,          //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,                    //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_MEMORY = 20,                    //!< Insufficient memory
    NVML_ERROR_NO_DATA = 21,                   //!< No data
    NVML_ERROR_VGPU_ECC_NOT_SUPPORTED = 22,    //!< The requested vgpu operation is not available on target device, becasue ECC is enabled
    NVML_ERROR_INSUFFICIENT_RESOURCES = 23,    //!< Ran out of critical resources, other than memory
    NVML_ERROR_FREQ_NOT_SUPPORTED = 24,        //!< Ran out of critical resources, other than memory
    NVML_ERROR_ARGUMENT_VERSION_MISMATCH = 25, //!< The provided version is invalid/unsupported
    NVML_ERROR_DEPRECATED  = 26,               //!< The requested functionality has been deprecated
    NVML_ERROR_NOT_READY = 27,                 //!< The system is not ready for the request
    NVML_ERROR_GPU_NOT_FOUND = 28,             //!< No GPUs were found
    NVML_ERROR_INVALID_STATE = 29,             //!< Resource not in correct state to perform requested operation
    NVML_ERROR_UNKNOWN = 999                   //!< An internal driver error occurred
} nvmlReturn_t;
typedef struct nvmlDevice_st* nvmlDevice_t;
typedef struct nvmlMemory_st
{
    unsigned long long total;        //!< Total physical device memory (in bytes)
    unsigned long long free;         //!< Unallocated device memory (in bytes)
    unsigned long long used;         //!< Sum of Reserved and Allocated device memory (in bytes).
                                     //!< Note that the driver/GPU always sets aside a small amount of memory for bookkeeping
} nvmlMemory_t;
// end nvml.h definitions

struct {
  void *handle;
  nvmlReturn_t (*nvmlInit_v2)(void);
  nvmlReturn_t (*nvmlShutdown)(void);
  nvmlReturn_t (*nvmlDeviceGetHandleByUUID)(const char *, nvmlDevice_t *);
  nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
  nvmlReturn_t (*nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
  const char * (*nvmlErrorString)(nvmlReturn_t result);
} nvml { NULL, NULL, NULL, NULL, NULL, NULL, NULL };
static std::mutex ggml_nvml_lock;

extern "C" {

#ifndef _WIN32
// Helper function to get available memory from /proc/meminfo on Linux
// Returns MemAvailable as calculated by the kernel
static size_t get_mem_available() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0;
    }

    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemAvailable:") == 0) {
            size_t available_kb;
            sscanf(line.c_str(), "MemAvailable: %zu kB", &available_kb);
            // Convert from kB to bytes
            return available_kb * 1024;
        }
    }

    return 0;
}
#endif

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
    nvml.nvmlDeviceGetName = (nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int)) GetProcAddress((HMODULE)(nvml.handle), "nvmlDeviceGetName");
    nvml.nvmlErrorString = (const char * (*)(nvmlReturn_enum)) GetProcAddress((HMODULE)(nvml.handle), "nvmlErrorString");
    if (nvml.nvmlInit_v2 == NULL || nvml.nvmlShutdown == NULL || nvml.nvmlDeviceGetHandleByUUID == NULL || nvml.nvmlDeviceGetMemoryInfo == NULL || nvml.nvmlDeviceGetName == NULL || nvml.nvmlErrorString == NULL) {
        GGML_LOG_INFO("%s unable to locate required symbols in NVML.dll", __func__);
        FreeLibrary((HMODULE)(nvml.handle));
        nvml.handle = NULL;
        return NVML_ERROR_NOT_FOUND;
    }

    SetErrorMode(old_mode);

    nvmlReturn_t status = nvml.nvmlInit_v2();
    if (status != NVML_SUCCESS) {
        GGML_LOG_INFO("%s unable to initialize NVML: %s\n", __func__, nvml.nvmlErrorString(status));
        FreeLibrary((HMODULE)(nvml.handle));
        nvml.handle = NULL;
        return status;
    }
#else
    constexpr std::array<const char*, 2> libPaths = {
        "/usr/lib/wsl/lib/libnvidia-ml.so.1", // Favor WSL2 path if present
        "libnvidia-ml.so.1" // On a non-WSL2 system, it should be in the path
    };
    for (const char* path : libPaths) {
        nvml.handle = dlopen(path, RTLD_LAZY);
        if (nvml.handle) break;
    }
    if (nvml.handle == NULL) {
        GGML_LOG_INFO("%s unable to load libnvidia-ml: %s\n", __func__, dlerror());
        return NVML_ERROR_NOT_FOUND;
    }
    nvml.nvmlInit_v2 = (nvmlReturn_enum (*)()) dlsym(nvml.handle, "nvmlInit_v2");
    nvml.nvmlShutdown = (nvmlReturn_enum (*)()) dlsym(nvml.handle, "nvmlShutdown");
    nvml.nvmlDeviceGetHandleByUUID = (nvmlReturn_t (*)(const char *, nvmlDevice_t *)) dlsym(nvml.handle, "nvmlDeviceGetHandleByUUID");
    nvml.nvmlDeviceGetMemoryInfo = (nvmlReturn_t (*)(nvmlDevice_t, nvmlMemory_t *)) dlsym(nvml.handle, "nvmlDeviceGetMemoryInfo");
    nvml.nvmlDeviceGetName = (nvmlReturn_t (*)(nvmlDevice_t, char *, unsigned int)) dlsym(nvml.handle, "nvmlDeviceGetName");
    nvml.nvmlErrorString = (const char * (*)(nvmlReturn_enum)) dlsym(nvml.handle, "nvmlErrorString");
    if (nvml.nvmlInit_v2 == NULL || nvml.nvmlShutdown == NULL || nvml.nvmlDeviceGetHandleByUUID == NULL || nvml.nvmlDeviceGetMemoryInfo == NULL || nvml.nvmlDeviceGetName == NULL) {
        GGML_LOG_INFO("%s unable to locate required symbols in libnvidia-ml.so", __func__);
        dlclose(nvml.handle);
        nvml.handle = NULL;
        return NVML_ERROR_NOT_FOUND;
    }
    nvmlReturn_t status = nvml.nvmlInit_v2();
    if (status != NVML_SUCCESS) {
        GGML_LOG_INFO("%s unable to initialize NVML: %s\n", __func__, nvml.nvmlErrorString(status));
        dlclose(nvml.handle);
        nvml.handle = NULL;
        return status;
    }
#endif
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
        GGML_LOG_INFO("%s failed to shutdown NVML: %s\n", __func__, nvml.nvmlErrorString(status));
    }
#ifdef _WIN32
    FreeLibrary((HMODULE)(nvml.handle));
#else
    dlclose(nvml.handle);
#endif
    nvml.handle = NULL;
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
        // NVML working correctly, use its values
        *free = memInfo.free;
        *total = memInfo.total;
        return NVML_SUCCESS;
    }

#ifndef _WIN32
    // Handle NVML_ERROR_NOT_SUPPORTED - this indicates NVML doesn't support
    // reporting framebuffer memory (e.g., unified memory GPUs where FB memory is 0)
    if (status == NVML_ERROR_NOT_SUPPORTED) {
        // Use system memory from /proc/meminfo
        size_t mem_available = get_mem_available();
        size_t mem_total = 0;

        // Read MemTotal
        std::ifstream meminfo("/proc/meminfo");
        if (meminfo.is_open()) {
            std::string line;
            while (std::getline(meminfo, line)) {
                if (line.find("MemTotal:") == 0) {
                    size_t total_kb;
                    sscanf(line.c_str(), "MemTotal: %zu kB", &total_kb);
                    mem_total = total_kb * 1024;
                    break;
                }
            }
        }

        if (mem_total > 0) {
            *total = mem_total;
            *free = mem_available;
            GGML_LOG_INFO("%s NVML not supported for memory query, using system memory (total=%zu, available=%zu)\n",
                          __func__, mem_total, mem_available);
            return NVML_SUCCESS;
        }
    }
#endif

    return status;
}

}
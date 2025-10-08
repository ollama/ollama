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
} nvml { NULL, NULL, NULL, NULL, NULL };
static std::mutex ggml_nvml_lock;

extern "C" {

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
        GGML_LOG_INFO("%s unable to locate required symbols in NVML.dll", __func__);
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

}
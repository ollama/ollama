// AMD HIP Memory Management
//
// This file provides accurate VRAM reporting for AMD GPUs, particularly
// on Windows, where the HIP library provides inaccurate VRAM usage metrics. The
// runtime DLL is installed with every driver on Windows, and most Linux
// systems.

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

#ifdef _WIN32

// Minimal definitions to avoid including the adlx.h header
typedef int ADLX_RESULT;
#define ADLX_SUCCEEDED(res)      (res == 0)
#define ADLX_FAILED(res)         (res != 0)
#define ADLX_OK                  0
#define ADLX_NOT_FOUND           1
#define ADLX_ADL_INIT_ERROR      2
#define ADLX_FULL_VERSION        ((10 << 16) | (0 << 8) | 0)

typedef bool adlx_bool;
typedef int adlx_int;
typedef unsigned int adlx_uint;
typedef unsigned long long adlx_uint64;
typedef double adlx_double;

typedef struct IADLXSystem IADLXSystem;
typedef struct IADLXGPU IADLXGPU;
typedef struct IADLXGPUList IADLXGPUList;
typedef struct IADLXPerformanceMonitoringServices IADLXPerformanceMonitoringServices;
typedef struct IADLXGPUMetricsSupport IADLXGPUMetricsSupport;
typedef struct IADLXGPUMetrics IADLXGPUMetrics;

// IADLXInterface
typedef struct {
    ADLX_RESULT (ADLX_STD_CALL* Release)(/* IADLXInterface* pThis */);
} IADLXInterfaceVtbl;
struct IADLXInterface { const IADLXInterfaceVtbl *pVtbl; };

// IADLXSystem
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(/* IADLXSystem* pThis */);

    ADLX_RESULT (ADLX_STD_CALL* GetGPUs)(IADLXSystem* pThis, IADLXGPUList** ppGPUs); // Used
    ADLX_RESULT (ADLX_STD_CALL* GetPerformanceMonitoringServices)(IADLXSystem* pThis, IADLXPerformanceMonitoringServices** ppPerformanceMonitoringServices); // Used
} IADLXSystemVtbl;
struct IADLXSystem { const IADLXSystemVtbl *pVtbl; };

// IADLXGPUList
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(IADLXGPUList* pThis);

    ADLX_RESULT (ADLX_STD_CALL* At_GPUList)(IADLXGPUList* pThis, adlx_uint location, IADLXGPU** ppGPU); // Used
    adlx_uint (ADLX_STD_CALL* Begin)(IADLXGPUList* pThis); // Used
    adlx_uint (ADLX_STD_CALL* End)(IADLXGPUList* pThis); // Used
    adlx_uint (ADLX_STD_CALL* Size)(/* IADLXGPUList* pThis */);
} IADLXGPUListVtbl;
struct IADLXGPUList { const IADLXGPUListVtbl *pVtbl; };

// IADLXGPU
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(IADLXGPU* pThis);

    ADLX_RESULT (ADLX_STD_CALL* UniqueId)(IADLXGPU* pThis, adlx_int* uniqueId); // Used
    ADLX_RESULT (ADLX_STD_CALL* TotalVRAM)(IADLXGPU* pThis, adlx_uint* vramMB); // Used
} IADLXGPUVtbl;
struct IADLXGPU { const IADLXGPUVtbl *pVtbl; };

// IADLXPerformanceMonitoringServices
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(IADLXPerformanceMonitoringServices* pThis);

    ADLX_RESULT (ADLX_STD_CALL* GetSupportedGPUMetrics)(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetricsSupport** ppMetricsSupport); // Used
    ADLX_RESULT (ADLX_STD_CALL* GetCurrentGPUMetrics)(IADLXPerformanceMonitoringServices* pThis, IADLXGPU* pGPU, IADLXGPUMetrics** ppMetrics); // Used
} IADLXPerformanceMonitoringServicesVtbl;
struct IADLXPerformanceMonitoringServices { const IADLXPerformanceMonitoringServicesVtbl *pVtbl; };

// IADLXGPUMetricsSupport
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(IADLXGPUMetricsSupport* pThis);

    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUUsage)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUClockSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVRAMClockSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUTemperature)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUHotspotTemperature)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUPower)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUTotalBoardPower)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUFanSpeed)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVRAM)(IADLXGPUMetricsSupport* pThis, adlx_bool* supported); // Used
    ADLX_RESULT (ADLX_STD_CALL* IsSupportedGPUVoltage)(/* IADLXGPUMetricsSupport* pThis, adlx_bool* supported */);
} IADLXGPUMetricsSupportVtbl;
struct IADLXGPUMetricsSupport { const IADLXGPUMetricsSupportVtbl *pVtbl; };

// IADLXGPUMetrics
typedef struct {
    // IADLXInterface
    ADLX_RESULT (ADLX_STD_CALL* Release)(IADLXGPUMetrics* pThis);

    //IADLXGPUMetrics
    ADLX_RESULT (ADLX_STD_CALL* TimeStamp)(/* IADLXGPUMetrics* pThis, adlx_int64* ms */);
    ADLX_RESULT (ADLX_STD_CALL* GPUUsage)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUClockSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUVRAMClockSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUTemperature)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUHotspotTemperature)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUPower)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUTotalBoardPower)(/* IADLXGPUMetrics* pThis, adlx_double* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUFanSpeed)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
    ADLX_RESULT (ADLX_STD_CALL* GPUVRAM)(IADLXGPUMetrics* pThis, adlx_int* data); // Used
    ADLX_RESULT (ADLX_STD_CALL* GPUVoltage)(/* IADLXGPUMetrics* pThis, adlx_int* data */);
} IADLXGPUMetricsVtbl;
struct IADLXGPUMetrics { const IADLXGPUMetricsVtbl *pVtbl; };

struct {
  void *handle;
  ADLX_RESULT (*ADLXInitialize)(adlx_uint64 version, IADLXSystem** ppSystem);
  ADLX_RESULT (*ADLXInitializeWithIncompatibleDriver)(adlx_uint64 version, IADLXSystem** ppSystem);
  ADLX_RESULT (*ADLXQueryVersion)(const char** version);
  ADLX_RESULT (*ADLXTerminate)();
  IADLXSystem *sys;
} adlx { NULL, NULL, NULL, NULL, NULL, NULL };
static std::mutex ggml_adlx_lock;

extern "C" {

int ggml_hip_mgmt_init() {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle != NULL) {
        // Already initialized
        return 0;
    }
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
    fs::path libPath = fs::path("\\Windows") / fs::path("System32") / fs::path("amdadlx64.dll");

    adlx.handle = (void*)LoadLibraryW(libPath.wstring().c_str());
    if (adlx.handle == NULL) {
        return ADLX_NOT_FOUND;
    }

    adlx.ADLXInitialize = (ADLX_RESULT (*)(adlx_uint64 version, IADLXSystem **ppSystem)) GetProcAddress((HMODULE)(adlx.handle), "ADLXInitialize");
    adlx.ADLXInitializeWithIncompatibleDriver = (ADLX_RESULT (*)(adlx_uint64 version, IADLXSystem **ppSystem)) GetProcAddress((HMODULE)(adlx.handle), "ADLXInitializeWithIncompatibleDriver");
    adlx.ADLXTerminate = (ADLX_RESULT (*)()) GetProcAddress((HMODULE)(adlx.handle), "ADLXTerminate");
    adlx.ADLXQueryVersion = (ADLX_RESULT (*)(const char **version)) GetProcAddress((HMODULE)(adlx.handle), "ADLXQueryVersion");
    if (adlx.ADLXInitialize == NULL || adlx.ADLXInitializeWithIncompatibleDriver == NULL || adlx.ADLXTerminate == NULL) {
        GGML_LOG_INFO("%s unable to locate required symbols in amdadlx64.dll, falling back to hip free memory reporting", __func__);
        FreeLibrary((HMODULE)(adlx.handle));
        adlx.handle = NULL;
        return ADLX_NOT_FOUND;
    }

    SetErrorMode(old_mode);

    // Aid in troubleshooting...
    if (adlx.ADLXQueryVersion != NULL) {
        const char *version = NULL;
        ADLX_RESULT status = adlx.ADLXQueryVersion(&version);
        if (ADLX_SUCCEEDED(status)) {
            GGML_LOG_DEBUG("%s located ADLX version %s\n", __func__, version);
        }
    }

    ADLX_RESULT status = adlx.ADLXInitialize(ADLX_FULL_VERSION, &adlx.sys);
    if (ADLX_FAILED(status)) {
        // GGML_LOG_DEBUG("%s failed to initialize ADLX error=%d - attempting with incompatible driver...\n", __func__, status);
        // Try with the incompatible driver
        status = adlx.ADLXInitializeWithIncompatibleDriver(ADLX_FULL_VERSION, &adlx.sys);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s failed to initialize ADLX error=%d\n", __func__, status);
            FreeLibrary((HMODULE)(adlx.handle));
            adlx.handle = NULL;
            adlx.sys = NULL;
            return status;
        }
        // GGML_LOG_DEBUG("%s initialized ADLX with incpomatible driver\n", __func__);
    }
    return ADLX_OK;
}

void ggml_hip_mgmt_release() {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle == NULL) {
        // Already free
        return;
    }
    ADLX_RESULT status = adlx.ADLXTerminate();
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s failed to terminate Adlx %d\n", __func__, status);
        // Unload anyway...
    }
    FreeLibrary((HMODULE)(adlx.handle));
    adlx.handle = NULL;
}

#define adlx_gdm_cleanup \
    if (gpuMetricsSupport != NULL) gpuMetricsSupport->pVtbl->Release(gpuMetricsSupport); \
    if (gpuMetrics != NULL) gpuMetrics->pVtbl->Release(gpuMetrics); \
    if (perfMonitoringServices != NULL) perfMonitoringServices->pVtbl->Release(perfMonitoringServices); \
    if (gpus != NULL) gpus->pVtbl->Release(gpus); \
    if (gpu != NULL) gpu->pVtbl->Release(gpu)

int ggml_hip_get_device_memory(int pci_bus_id, int pci_device_id, size_t *free, size_t *total) {
    std::lock_guard<std::mutex> lock(ggml_adlx_lock);
    if (adlx.handle == NULL) {
        GGML_LOG_INFO("%s ADLX was not initialized\n", __func__);
        return ADLX_ADL_INIT_ERROR;
    }
    IADLXGPUMetricsSupport *gpuMetricsSupport = NULL;
    IADLXPerformanceMonitoringServices *perfMonitoringServices = NULL;
    IADLXGPUList* gpus = NULL;
    IADLXGPU* gpu = NULL;
    IADLXGPUMetrics *gpuMetrics = NULL;
    ADLX_RESULT status;
    // The "UniqueID" exposed in ADLX is the PCI Bus and Device IDs
    adlx_int target = (pci_bus_id << 8) | (pci_device_id & 0xff);

    status = adlx.sys->pVtbl->GetPerformanceMonitoringServices(adlx.sys, &perfMonitoringServices);
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s GetPerformanceMonitoringServices failed %d\n", __func__, status);
        return status;
    }

    status = adlx.sys->pVtbl->GetGPUs(adlx.sys, &gpus);
    if (ADLX_FAILED(status)) {
        GGML_LOG_INFO("%s GetGPUs failed %d\n", __func__, status);
        adlx_gdm_cleanup;
        return status;
    }

    // Get GPU list
    for (adlx_uint crt = gpus->pVtbl->Begin(gpus); crt != gpus->pVtbl->End(gpus); ++crt)
    {
        status = gpus->pVtbl->At_GPUList(gpus, crt, &gpu);
        if (ADLX_FAILED(status))
        {
            GGML_LOG_INFO("%s %d] At_GPUList failed %d\n", __func__, crt, status);
            continue;
        }
        adlx_int id;
        status = gpu->pVtbl->UniqueId(gpu, &id);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s %d] UniqueId lookup failed %d\n", __func__, crt, status);
            gpu->pVtbl->Release(gpu);
            gpu = NULL;
            continue;
        }
        if (id != target) {
            GGML_LOG_DEBUG("%s %d] GPU UniqueId: %x does not match target %02x %02x\n", __func__, crt, id, pci_bus_id, pci_device_id);
            gpu->pVtbl->Release(gpu);
            gpu = NULL;
            continue;
        }
        // Any failures at this point should cause a fall-back to other APIs
        status = perfMonitoringServices->pVtbl->GetSupportedGPUMetrics(perfMonitoringServices, gpu, &gpuMetricsSupport);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GetSupportedGPUMetrics failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }
        status = perfMonitoringServices->pVtbl->GetCurrentGPUMetrics(perfMonitoringServices, gpu, &gpuMetrics);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GetCurrentGPUMetrics failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }

        adlx_bool supported = false;
        status = gpuMetricsSupport->pVtbl->IsSupportedGPUVRAM(gpuMetricsSupport, &supported);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s IsSupportedGPUVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }

        adlx_uint totalVRAM = 0;
        status = gpu->pVtbl->TotalVRAM(gpu, &totalVRAM);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s TotalVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }

        adlx_int usedVRAM = 0;
        status = gpuMetrics->pVtbl->GPUVRAM(gpuMetrics, &usedVRAM);
        if (ADLX_FAILED(status)) {
            GGML_LOG_INFO("%s GPUVRAM failed %d\n", __func__, status);
            adlx_gdm_cleanup;
            return status;
        }
        *total = size_t(totalVRAM) * 1024 * 1024;
        *free = size_t(totalVRAM-usedVRAM) * 1024 * 1024;

        adlx_gdm_cleanup;
        return ADLX_OK;
    }
    adlx_gdm_cleanup;
    return ADLX_NOT_FOUND;
}

} // extern "C"

#else // #ifdef _WIN32

extern "C" {

// TODO Linux implementation of accurate VRAM reporting
int ggml_hip_mgmt_init() {
    return -1;
}
void ggml_hip_mgmt_release() {}
int ggml_hip_get_device_memory(int pci_bus_id, int pci_device_id, size_t *free, size_t *total) {
    return -1;
}

} // extern "C"

#endif // #ifdef _WIN32

// Made with Bob

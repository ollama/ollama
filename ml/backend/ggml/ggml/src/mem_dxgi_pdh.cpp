// DXGI and PDH Performance Counters Library
// This Windows-only (10/11) library provides accurate VRAM reporting for Intel GPUs

#include <initguid.h> // Required for GUID definitions
#include "ggml-impl.h"
#include <windows.h>
#include <pdh.h>
#include <dxgi.h>
#include <dxgi1_2.h>
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

static PDH_HQUERY ggml_dxgi_pdh_query = nullptr;
static PDH_HCOUNTER ggml_dxgi_pdh_counter = nullptr;

// TODO tasks
// 1. Enumerate all GPUs (iGPU and dGPU)
// 2. Detect whether IGPU or DGPU using DXCore
// 3. Implement a function to retrieve the memory usage information for a specific GPU
// 4, fetch the corresponding memory info (dedicated memory, shared memory)
// 5. replace all -1 with proper error codes

struct GpuInfo {
    LUID luid;
    std::wstring pdhInstance;
    double dedicatedUsage = 0.0;
    double sharedUsage = 0.0;
    double totalCommitted = 0.0;
    double localUsage = 0.0;
};

/*
Enumerate over the GPU adapters detected using DXGI and return their information
*/
std::vector<GpuInfo> GetDxgiGpuInfos() {
    std::vector<GpuInfo> infos;
    IDXGIFactory1* pFactory = nullptr;

    if (SUCCEEDED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory))) {
        UINT i = 0;
        IDXGIAdapter1* pAdapter = nullptr;
        while (pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 desc;
            pAdapter->GetDesc1(&desc);

            PrintDxgiAdapterDesc1(desc);
            
            std::wstring name(desc.Description);
            
            // Get all the GPU adapter info
            GpuInfo info;
            info.luid = desc.AdapterLuid;
            info.pdhInstance = GeneratePdhInstanceNameFromLuid(desc.AdapterLuid);
            infos.push_back(info);

            pAdapter->Release();
            ++i;
        }
        pFactory->Release();
    }
    return infos;
}

bool GetGpuMemoryUsage(GpuInfo gpu) {
    PDH_HQUERY query;
    if (PdhOpenQuery(NULL, 0, &query) != ERROR_SUCCESS) {
        return false;
    }

    struct GpuCounters {
        PDH_HCOUNTER dedicated;
        PDH_HCOUNTER shared;
        PDH_HCOUNTER committed;
        PDH_HCOUNTER local;
    };

    std::vector<GpuCounters> counters;

    for (auto& info : gpus) {
        std::wstring dedicatedPath = L"\\GPU Adapter Memory(" + info.pdhInstance + L"*)\\Dedicated Usage";
        std::wstring sharedPath = L"\\GPU Adapter Memory(" + info.pdhInstance + L"*)\\Shared Usage";
        std::wstring totalCommittedPath = L"\\GPU Adapter Memory(" + info.pdhInstance + L"*)\\Total Committed";
        std::wstring localUsagePath = L"\\GPU Local Adapter Memory(" + info.pdhInstance + L"*)\\Local Usage";

        GpuCounters gpuCounter{};
        if (PdhAddCounter(query, dedicatedPath.c_str(), 0, &gpuCounter.dedicated) != ERROR_SUCCESS ||
            PdhAddCounter(query, sharedPath.c_str(), 0, &gpuCounter.shared) != ERROR_SUCCESS ||
            PdhAddCounter(query, totalCommittedPath.c_str(), 0, &gpuCounter.committed) != ERROR_SUCCESS ||
            PdhAddCounter(query, localUsagePath.c_str(), 0, &gpuCounter.local) != ERROR_SUCCESS) {
            continue;
        }

        counters.push_back(gpuCounter);
    }

    // Collect data multiple times
    constexpr int sampleCount = 3;
    for (int i = 0; i < sampleCount; ++i) {
        if (PdhCollectQueryData(query) != ERROR_SUCCESS) {
            PdhCloseQuery(query);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Read final values
    for (size_t i = 0; i < gpus.size(); ++i) {
        PDH_FMT_COUNTERVALUE val;

        if (PdhGetFormattedCounterValue(counters[i].dedicated, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
            gpus[i].dedicatedUsage = val.doubleValue;

        if (PdhGetFormattedCounterValue(counters[i].shared, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
            gpus[i].sharedUsage = val.doubleValue;

        if (PdhGetFormattedCounterValue(counters[i].committed, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
            gpus[i].totalCommitted = val.doubleValue;

        if (PdhGetFormattedCounterValue(counters[i].local, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
            gpus[i].localUsage = val.doubleValue;
    }

    PdhCloseQuery(query);
    return true;
}



extern "C" {

    int ggml_dxgi_pdh_init() {

        /*
        fs::path libPath = fs::path("\\Windows") / fs::path("System32") / fs::path("DXCore.dll");
        adlx.handle = (void*)LoadLibraryW(libPath.wstring().c_str());
        if (adlx.handle == NULL) {
            return FAILURE;
        }
        */

        return -1; // change when implemented
    }

    int ggml_dxgi_pdh_get_device_memory(const char* luid, size_t *free, size_t *total, bool is_integrated_gpu) {

        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);

        // Enumerate GPUs using DXGI and find the matching LUID
        std::vector<GpuInfo> gpus = GetDxgiGpuInfos();
        GpuInfo *targetGpu = nullptr;
        for (auto& gpu : gpus) {
            if (memcmp(&gpu.luid, luid, sizeof(LUID)) == 0) {
                targetGpu = &gpu;
                break;
            }
        }
        if (!targetGpu) {
            GGML_LOG_ERROR("GPU with specified LUID not found.\n");
            return -1;
        }

        // Get the memory usage and total memory for the target GPU
        int status = GetGpuMemoryUsage(*targetGpu);
        if (!status) {
            GGML_LOG_ERROR("Failed to get GPU memory usage.\n");
            return -1;
        }

        // Calculate the free memory based on whether it's an integrated or discrete GPU
        if (is_integrated_gpu) {
            // IGPU
        }
        else {
            // DGPU
        }




        return -1; // change when implemented
    }

    void ggml_dxgi_pdh_release() {
        return; // change when implemented
    }

} // extern "C"
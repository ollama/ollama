// DXGI and PDH Performance Counters Library
// This Windows-only (10/11) library provides accurate VRAM reporting for Intel GPUs

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#include <windows.h>
#include <initguid.h> // Required for GUID definitions
#include "ggml-impl.h"
#include <pdh.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include <sstream>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

static std::mutex ggml_dxgi_pdh_lock;

static PDH_HQUERY ggml_dxgi_pdh_query = nullptr;
static PDH_HCOUNTER ggml_dxgi_pdh_counter = nullptr;

// TODO tasks
// 1. replace all -1 with proper error codes

struct GpuInfo {
    std::wstring name; // debug field
    LUID luid;
    std::wstring pdhInstance;
    double dedicatedTotal = 0.0;
    double sharedTotal = 0.0;
    double dedicatedUsage = 0.0;
    double sharedUsage = 0.0;
    double totalCommitted = 0.0;
    double localUsage = 0.0;
};

/*
Maybe not needed
*/
std::wstring GeneratePdhInstanceNameFromLuid(const LUID& luid) {
    std::wstringstream ss;
    ss << L"luid_0x" << std::hex << std::setw(8) << std::setfill(L'0') << std::uppercase << luid.HighPart
        << L"_0x" << std::setw(8) << std::setfill(L'0') << luid.LowPart;
    return ss.str();
}

/*
Conversion from Bytes to GigaBytes
*/
template <typename T>
static inline double BtoGB(T n)
{
    return (double(n) / (1024.0 * 1024 * 1024));
}

/*
Fetch the GPU adapter 'dedicated memory' and 'shared memory' using DXGI
*/
void FetchDxgiAdapterDesc1(const DXGI_ADAPTER_DESC1& desc, GpuInfo* info) {
    auto dedicatedVideoMemory = desc.DedicatedVideoMemory;
    auto sharedSystemMemory = desc.SharedSystemMemory;
    GGML_LOG_DEBUG("Dedicated Video Memory: %.2f GB\n", BtoGB(dedicatedVideoMemory));
    GGML_LOG_DEBUG("Shared System Memory: %.2f GB\n", BtoGB(sharedSystemMemory));

    if (info) {
        info->dedicatedTotal = dedicatedVideoMemory; // values in bytes
        info->sharedTotal = sharedSystemMemory;
    }
}

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
            
            // Get all the GPU adapter info
            GpuInfo info;
            FetchDxgiAdapterDesc1(desc, &info);
            info.name = std::wstring(desc.Description);
            info.luid = desc.AdapterLuid;
            info.pdhInstance = GeneratePdhInstanceNameFromLuid(desc.AdapterLuid); // maybe not needed
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

    GpuCounters gpuCounter;

    std::wstring dedicatedPath = L"\\GPU Adapter Memory(" + gpu.pdhInstance + L"*)\\Dedicated Usage";
    std::wstring sharedPath = L"\\GPU Adapter Memory(" + gpu.pdhInstance + L"*)\\Shared Usage";
    std::wstring totalCommittedPath = L"\\GPU Adapter Memory(" + gpu.pdhInstance + L"*)\\Total Committed";
    std::wstring localUsagePath = L"\\GPU Local Adapter Memory(" + gpu.pdhInstance + L"*)\\Local Usage";

    if (PdhAddCounter(query, dedicatedPath.c_str(), 0, &gpuCounter.dedicated) != ERROR_SUCCESS ||
        PdhAddCounter(query, sharedPath.c_str(), 0, &gpuCounter.shared) != ERROR_SUCCESS ||
        PdhAddCounter(query, totalCommittedPath.c_str(), 0, &gpuCounter.committed) != ERROR_SUCCESS ||
        PdhAddCounter(query, localUsagePath.c_str(), 0, &gpuCounter.local) != ERROR_SUCCESS) {
            GGML_LOG_ERROR("Failed to add PDH counters for GPU %s\n", std::string(gpu.pdhInstance.begin(), gpu.pdhInstance.end()).c_str());
            PdhCloseQuery(query);
            return false;
    }

    // Sample data multiple times
    constexpr int sampleCount = 3;
    for (int i = 0; i < sampleCount; ++i) {
        if (PdhCollectQueryData(query) != ERROR_SUCCESS) {
            PdhCloseQuery(query);
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Read final values
    PDH_FMT_COUNTERVALUE val;

    if (PdhGetFormattedCounterValue(gpuCounter.dedicated, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.dedicatedUsage = val.doubleValue;

    if (PdhGetFormattedCounterValue(gpuCounter.shared, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.sharedUsage = val.doubleValue;

    if (PdhGetFormattedCounterValue(gpuCounter.committed, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.totalCommitted = val.doubleValue;

    if (PdhGetFormattedCounterValue(gpuCounter.local, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.localUsage = val.doubleValue;

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
        GGML_LOG_DEBUG("%s called\n", __func__);
        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);
        return -1; // change when implemented
    }

    int ggml_dxgi_pdh_get_device_memory(const char* luid, size_t *free, size_t *total, bool is_integrated_gpu) {

        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);

        // Enumerate GPUs using DXGI and find the matching LUID
        // This also fetches the total memory info for each of the enumerated GPUs
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

        // Get the current memory usage for the target GPU
        int status = GetGpuMemoryUsage(*targetGpu);
        if (!status) {
            GGML_LOG_ERROR("Failed to get GPU memory usage.\n");
            return -1;
        }

        // Calculate the free memory based on whether it's an integrated or discrete GPU
        if (is_integrated_gpu) {
            // IGPU free = SharedTotal - SharedUsage
            *free = targetGpu->sharedTotal - targetGpu->sharedUsage;
            *total = targetGpu->sharedTotal;
        }
        else {
            // DGPU free = DedicatedTotal - DedicatedUsage
            *free = targetGpu->dedicatedTotal - targetGpu->dedicatedUsage;
            *total = targetGpu->dedicatedTotal;
        }

        return 0; // change when implemented
    }

    void ggml_dxgi_pdh_release() {
        return; // change when implemented
    }

} // extern "C"

#else // #ifdef _WIN32

extern "C" {

    // DXGI + PDH not available for Linux implementation
    int ggml_dxgi_pdh_init() {
        return -1;
    }
    void ggml_dxgi_pdh_release() {}
    int ggml_dxgi_pdh_get_device_memory(const char* luid, size_t *free, size_t *total, bool is_integrated_gpu) {
        return -1;
    }

} // extern "C"

#endif // #ifdef _WIN32
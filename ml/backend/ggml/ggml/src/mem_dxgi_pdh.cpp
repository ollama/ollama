// DXGI and PDH Performance Counters Library
// This Windows-only (10/11) library provides accurate VRAM reporting
#include "ggml.h"
#include "ggml-impl.h"

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#include <windows.h>
#include <pdh.h>
#include <dxgi1_2.h>
#include <sstream>
#include <thread>
#include <filesystem>
#include <mutex>

namespace fs = std::filesystem;

static std::mutex ggml_dxgi_pdh_lock;

/*
Struct to keep track of GPU adapter information at runtime
*/
struct GpuInfo {
    std::wstring description; // debug field
    LUID luid;
    std::wstring pdhInstance;
    double dedicatedTotal = 0.0;
    double sharedTotal = 0.0;
    double dedicatedUsage = 0.0;
    double sharedUsage = 0.0;
};

/*
DLL Function Pointers
*/
struct {
    void *dxgi_dll_handle;
    void *pdh_dll_handle;
    // DXGI Functions
    HRESULT (*CreateDXGIFactory1)(REFIID riid, void **ppFactory);
    // PDH functions  
    PDH_STATUS (*PdhOpenQueryW)(LPCWSTR szDataSource, DWORD_PTR dwUserData, PDH_HQUERY *phQuery);
    PDH_STATUS (*PdhAddCounterW)(PDH_HQUERY hQuery, LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER *phCounter);
    PDH_STATUS (*PdhCollectQueryData)(PDH_HQUERY hQuery);
    PDH_STATUS (*PdhGetFormattedCounterValue)(PDH_HCOUNTER hCounter, DWORD dwFormat, LPDWORD lpdwType, PPDH_FMT_COUNTERVALUE pValue);
    PDH_STATUS (*PdhCloseQuery)(PDH_HQUERY hQuery);
} dll_functions {
    nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr
};

/*
Create a PDH Instance name
*/
static std::wstring generate_pdh_instance_name_from_luid(const LUID& luid) {
    std::wstringstream ss;
    ss << L"luid_0x" << std::hex << std::setw(8) << std::setfill(L'0') << std::uppercase << luid.HighPart
        << L"_0x" << std::setw(8) << std::setfill(L'0') << luid.LowPart;
    return ss.str();
}

/*
Conversion from Bytes to GigaBytes
*/
template <typename T>
static inline double b_to_gb(T n)
{
    return (double(n) / (1024.0 * 1024 * 1024));
}

/*
Fetch the GPU adapter 'dedicated memory' and 'shared memory' using DXGI
*/
static void fetch_dxgi_adapter_desc1(const DXGI_ADAPTER_DESC1& desc, GpuInfo* info) {
    auto dedicatedVideoMemory = desc.DedicatedVideoMemory;
    auto sharedSystemMemory = desc.SharedSystemMemory;
    GGML_LOG_DEBUG("[DXGI] Adapter Description: %ls, LUID: 0x%08X%08X, Dedicated: %.2f GB, Shared: %.2f GB\n", desc.Description, desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart, b_to_gb(dedicatedVideoMemory), b_to_gb(sharedSystemMemory));
    if (info) {
        info->dedicatedTotal = dedicatedVideoMemory; // values in bytes
        info->sharedTotal = sharedSystemMemory;
    }
}

/*
Enumerate over the GPU adapters detected using DXGI and return their information
*/
static std::vector<GpuInfo> get_dxgi_gpu_infos() {
    std::vector<GpuInfo> infos;
    IDXGIFactory1* pFactory = nullptr;

    if (SUCCEEDED(dll_functions.CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&pFactory))) {
        UINT i = 0;
        IDXGIAdapter1* pAdapter = nullptr;
        while (pFactory->EnumAdapters1(i, &pAdapter) != DXGI_ERROR_NOT_FOUND) {
            DXGI_ADAPTER_DESC1 desc;
            pAdapter->GetDesc1(&desc);
            
            // Get all the GPU adapter info
            GpuInfo info;
            fetch_dxgi_adapter_desc1(desc, &info);
            info.description = std::wstring(desc.Description);
            info.luid = desc.AdapterLuid;
            info.pdhInstance = generate_pdh_instance_name_from_luid(desc.AdapterLuid);
            infos.push_back(info);

            pAdapter->Release();
            ++i;
        }
        pFactory->Release();
    }
    return infos;
}

static bool get_gpu_memory_usage(GpuInfo& gpu) {
    PDH_HQUERY query;
    if (dll_functions.PdhOpenQueryW(NULL, 0, &query) != ERROR_SUCCESS) {
        return false;
    }

    struct GpuCounters {
        PDH_HCOUNTER dedicated;
        PDH_HCOUNTER shared;
    };

    GpuCounters gpuCounter{};

    std::wstring dedicatedPath = L"\\GPU Adapter Memory(" + gpu.pdhInstance + L"*)\\Dedicated Usage";
    std::wstring sharedPath = L"\\GPU Adapter Memory(" + gpu.pdhInstance + L"*)\\Shared Usage";

    if (dll_functions.PdhAddCounterW(query, dedicatedPath.c_str(), 0, &gpuCounter.dedicated) != ERROR_SUCCESS ||
        dll_functions.PdhAddCounterW(query, sharedPath.c_str(), 0, &gpuCounter.shared) != ERROR_SUCCESS) {
            GGML_LOG_ERROR("Failed to add PDH counters for GPU %s\n", std::string(gpu.pdhInstance.begin(), gpu.pdhInstance.end()).c_str());
            dll_functions.PdhCloseQuery(query);
            return false;
    }

    // Sample the data
    if (dll_functions.PdhCollectQueryData(query) != ERROR_SUCCESS) {
            dll_functions.PdhCloseQuery(query);
            return false;
    }

    // Read final values
    PDH_FMT_COUNTERVALUE val;

    if (dll_functions.PdhGetFormattedCounterValue(gpuCounter.dedicated, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.dedicatedUsage = val.doubleValue;

    if (dll_functions.PdhGetFormattedCounterValue(gpuCounter.shared, PDH_FMT_DOUBLE, NULL, &val) == ERROR_SUCCESS)
        gpu.sharedUsage = val.doubleValue;

    dll_functions.PdhCloseQuery(query);
    return true;
}


extern "C" {

    int ggml_dxgi_pdh_init() {
        GGML_LOG_DEBUG("%s called\n", __func__);
        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);
        if (dll_functions.dxgi_dll_handle != NULL && dll_functions.pdh_dll_handle != NULL) {
            // Already initialized as we have both DLL handles
            return ERROR_SUCCESS;
        }

        DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
        SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);
        fs::path libPath_dxgi = fs::path("\\Windows") / fs::path("System32") / fs::path("dxgi.dll");
        fs::path libPath_pdh = fs::path("\\Windows") / fs::path("System32") / fs::path("pdh.dll");

        // Call LoadLibraryW on both DLLs to ensure they are loaded
        void *dxgi = (void*)LoadLibraryW(libPath_dxgi.wstring().c_str());
        void *pdh = (void*)LoadLibraryW(libPath_pdh.wstring().c_str());
        if(dxgi == NULL || pdh == NULL) {
            if (dxgi != NULL) {
                FreeLibrary((HMODULE)(dxgi));
            }
            if (pdh != NULL) {
                FreeLibrary((HMODULE)(pdh));
            }
            SetErrorMode(old_mode);
            return ERROR_DLL_NOT_FOUND;
        }
        else {
            // save the dll handles
            dll_functions.dxgi_dll_handle = dxgi;
            dll_functions.pdh_dll_handle = pdh;
        }

        // Get pointers to the library functions loaded by the DLLs
        dll_functions.CreateDXGIFactory1 = (HRESULT (*)(REFIID riid, void **ppFactory)) GetProcAddress((HMODULE)(dll_functions.dxgi_dll_handle), "CreateDXGIFactory1");
        dll_functions.PdhOpenQueryW = (PDH_STATUS (*)(LPCWSTR szDataSource, DWORD_PTR dwUserData, PDH_HQUERY *phQuery)) GetProcAddress((HMODULE)(dll_functions.pdh_dll_handle), "PdhOpenQueryW");
        dll_functions.PdhAddCounterW = (PDH_STATUS (*)(PDH_HQUERY hQuery, LPCWSTR szFullCounterPath, DWORD_PTR dwUserData, PDH_HCOUNTER *phCounter)) GetProcAddress((HMODULE)(dll_functions.pdh_dll_handle), "PdhAddCounterW");
        dll_functions.PdhCollectQueryData = (PDH_STATUS (*)(PDH_HQUERY hQuery)) GetProcAddress((HMODULE)(dll_functions.pdh_dll_handle), "PdhCollectQueryData");
        dll_functions.PdhGetFormattedCounterValue = (PDH_STATUS (*)(PDH_HCOUNTER hCounter, DWORD dwFormat, LPDWORD lpdwType, PPDH_FMT_COUNTERVALUE pValue)) GetProcAddress((HMODULE)(dll_functions.pdh_dll_handle), "PdhGetFormattedCounterValue");
        dll_functions.PdhCloseQuery = (PDH_STATUS (*)(PDH_HQUERY hQuery)) GetProcAddress((HMODULE)(dll_functions.pdh_dll_handle), "PdhCloseQuery");
    
        SetErrorMode(old_mode); // set old mode before any return

        // Check if any function pointers are NULL (not found)
        if (dll_functions.CreateDXGIFactory1 == NULL || dll_functions.PdhOpenQueryW == NULL || dll_functions.PdhAddCounterW == NULL || dll_functions.PdhCollectQueryData == NULL || dll_functions.PdhGetFormattedCounterValue == NULL || dll_functions.PdhCloseQuery == NULL) {
            GGML_LOG_INFO("%s unable to locate required symbols in either dxgi.dll or pdh.dll", __func__);
            FreeLibrary((HMODULE)(dll_functions.dxgi_dll_handle));
            FreeLibrary((HMODULE)(dll_functions.pdh_dll_handle));
            dll_functions.dxgi_dll_handle = NULL;
            dll_functions.pdh_dll_handle = NULL;
            return ERROR_PROC_NOT_FOUND;
        }
    
        // No other initializations needed, successfully loaded the libraries and functions!
        return ERROR_SUCCESS;
    }

    void ggml_dxgi_pdh_release() {
        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);
        if (dll_functions.dxgi_dll_handle == NULL && dll_functions.pdh_dll_handle == NULL) {
            // Already freed the DLLs
            return;
        }

        // Call FreeLibrary on both DLLs
        FreeLibrary((HMODULE)(dll_functions.dxgi_dll_handle));
        FreeLibrary((HMODULE)(dll_functions.pdh_dll_handle));

        dll_functions.dxgi_dll_handle = NULL;
        dll_functions.pdh_dll_handle = NULL;

        return; // successfully released
    }

    int ggml_dxgi_pdh_get_device_memory(const char* luid, size_t *free, size_t *total, bool is_integrated_gpu) {

        std::lock_guard<std::mutex> lock(ggml_dxgi_pdh_lock);

        // Enumerate GPUs using DXGI and find the matching LUID
        // This also fetches the total memory info for each of the enumerated GPUs
        std::vector<GpuInfo> gpus = get_dxgi_gpu_infos();
        GpuInfo *targetGpu = nullptr;
        for (auto& gpu : gpus) {
            char luid_buffer[32]; // "0x" + 16 hex digits + null terminator
            snprintf(luid_buffer, sizeof(luid_buffer), "0x%08x%08x", gpu.luid.HighPart, gpu.luid.LowPart);
            std::string gpu_luid_str(luid_buffer);
            if (gpu_luid_str == std::string(luid)) {
                targetGpu = &gpu;
                break;
            }
        }
        if (!targetGpu) {
            GGML_LOG_ERROR("GPU with LUID %s not found.\n", luid);
            return ERROR_NOT_FOUND;
        }

        // Get the current memory usage for the target GPU
        int status = get_gpu_memory_usage(*targetGpu);
        if (!status) {
            GGML_LOG_ERROR("Failed to get GPU memory usage.\n");
            return ERROR_DEVICE_NOT_AVAILABLE;
        }

        // Calculate the free memory based on whether it's an integrated or discrete GPU
        if (is_integrated_gpu) {
            // IGPU free = SharedTotal - SharedUsage
            GGML_LOG_DEBUG("Integrated GPU (%ls) with LUID %s detected. Shared Total: %.2f bytes (%.2f GB), Shared Usage: %.2f bytes (%.2f GB), Dedicated Total: %.2f bytes (%.2f GB), Dedicated Usage: %.2f bytes (%.2f GB)\n", targetGpu->description.c_str(), luid, targetGpu->sharedTotal, b_to_gb(targetGpu->sharedTotal), targetGpu->sharedUsage, b_to_gb(targetGpu->sharedUsage), targetGpu->dedicatedTotal, b_to_gb(targetGpu->dedicatedTotal), targetGpu->dedicatedUsage, b_to_gb(targetGpu->dedicatedUsage));
            *free = (targetGpu->sharedTotal - targetGpu->sharedUsage) + (targetGpu->dedicatedTotal - targetGpu->dedicatedUsage); // Some IGPUs also have dedicated memory, which can be used along with the IGPU's shared memory
            *total = targetGpu->sharedTotal + targetGpu->dedicatedTotal;
        }
        else {
            // DGPU free = DedicatedTotal - DedicatedUsage
            GGML_LOG_DEBUG("Discrete GPU (%ls) with LUID %s detected. Dedicated Total: %.2f bytes (%.2f GB), Dedicated Usage: %.2f bytes (%.2f GB)\n", targetGpu->description.c_str(), luid, targetGpu->dedicatedTotal, b_to_gb(targetGpu->dedicatedTotal), targetGpu->dedicatedUsage, b_to_gb(targetGpu->dedicatedUsage));
            *free = targetGpu->dedicatedTotal - targetGpu->dedicatedUsage;
            *total = targetGpu->dedicatedTotal;
        }

        return ERROR_SUCCESS;
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
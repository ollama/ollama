// NPU accessibility probe — tries EVERY path to reach the NPU
// Build: cl /EHsc /std:c++17 npu_access_probe.cpp /link d3d12.lib dxgi.lib ole32.lib
// Or:    clang++ -std=c++17 -target aarch64-pc-windows-msvc npu_access_probe.cpp -ld3d12 -ldxgi -lole32 -loleaut32

#include <stdio.h>
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

// ============================================================
// Path 1: DXGI enumeration — can it see the NPU?
// ============================================================
void test_dxgi() {
    printf("[1] DXGI Enumeration\n");
    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void**);
    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    if (!dxgi) { printf("  dxgi.dll not found\n"); return; }
    auto fn = (PFN_CreateDXGIFactory1)GetProcAddress(dxgi, "CreateDXGIFactory1");
    if (!fn) { printf("  CreateDXGIFactory1 not found\n"); return; }

    IDXGIFactory4* factory = nullptr;
    HRESULT hr = fn(__uuidof(IDXGIFactory4), (void**)&factory);
    if (FAILED(hr)) { printf("  CreateDXGIFactory1 failed: 0x%08lx\n", hr); return; }

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = nullptr;
        hr = factory->EnumAdapters1(i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);
        printf("  [%u] \"%ls\" VendorId=0x%04x DeviceId=0x%04x VRAM=%lluMB\n",
            i, desc.Description, desc.VendorId, desc.DeviceId,
            desc.DedicatedVideoMemory / (1024*1024));
        adapter->Release();
    }
    factory->Release();
}

// ============================================================
// Path 2: D3D12 with DXGI adapters — which ones can create devices?
// ============================================================
void test_d3d12_dxgi() {
    printf("\n[2] D3D12 Device Creation (DXGI adapters)\n");
    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void**);
    typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

    HMODULE dxgi = LoadLibraryA("dxgi.dll");
    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    if (!dxgi || !d3d12) return;

    auto createFactory = (PFN_CreateDXGIFactory1)GetProcAddress(dxgi, "CreateDXGIFactory1");
    auto createDev = (PFN_D3D12CreateDevice)GetProcAddress(d3d12, "D3D12CreateDevice");
    if (!createFactory || !createDev) return;

    IDXGIFactory4* factory = nullptr;
    createFactory(__uuidof(IDXGIFactory4), (void**)&factory);
    if (!factory) return;

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        ID3D12Device* dev = nullptr;
        HRESULT hr = createDev(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&dev);
        printf("  [%u] \"%ls\" -> D3D12: %s (0x%08lx)\n",
            i, desc.Description,
            SUCCEEDED(hr) ? "OK" : "FAILED", hr);
        if (dev) dev->Release();
        adapter->Release();
    }
    factory->Release();
}

// ============================================================
// Path 3: DirectML device creation on each D3D12 device
// ============================================================
void test_directml() {
    printf("\n[3] DirectML Device Creation\n");

    // DML types
    enum DML_FEATURE_LEVEL_enum {
        DML_FEATURE_LEVEL_1_0 = 0x1000,
        DML_FEATURE_LEVEL_2_0 = 0x2000,
        DML_FEATURE_LEVEL_3_0 = 0x3000,
        DML_FEATURE_LEVEL_4_0 = 0x4000,
        DML_FEATURE_LEVEL_5_0 = 0x5000,
        DML_FEATURE_LEVEL_6_0 = 0x6000,
        DML_FEATURE_LEVEL_6_4 = 0x6400,
    };

    typedef HRESULT (WINAPI *PFN_DMLCreateDevice1)(ID3D12Device*, UINT flags,
        DML_FEATURE_LEVEL_enum, REFIID, void**);
    typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    typedef HRESULT (WINAPI *PFN_CreateDXGIFactory1)(REFIID, void**);

    HMODULE dml = LoadLibraryA("DirectML.dll");
    if (!dml) dml = LoadLibraryA("C:\\Windows\\System32\\DirectML.dll");
    if (!dml) { printf("  DirectML.dll not found\n"); return; }

    auto dmlCreate = (PFN_DMLCreateDevice1)GetProcAddress(dml, "DMLCreateDevice1");
    if (!dmlCreate) { printf("  DMLCreateDevice1 not found\n"); return; }

    auto createDev = (PFN_D3D12CreateDevice)GetProcAddress(LoadLibraryA("d3d12.dll"), "D3D12CreateDevice");
    auto createFactory = (PFN_CreateDXGIFactory1)GetProcAddress(LoadLibraryA("dxgi.dll"), "CreateDXGIFactory1");

    IDXGIFactory4* factory = nullptr;
    createFactory(__uuidof(IDXGIFactory4), (void**)&factory);
    if (!factory) return;

    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc;
        adapter->GetDesc1(&desc);

        ID3D12Device* d3dDev = nullptr;
        HRESULT hr = createDev(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void**)&d3dDev);
        if (FAILED(hr)) {
            printf("  [%u] \"%ls\" -> no D3D12 device, skipping DML\n", i, desc.Description);
            adapter->Release();
            continue;
        }

        // Try DML at feature level 1.0 (minimum)
        IUnknown* dmlDev = nullptr;
        // IID for IDMLDevice -- {6dbd6437-96fd-423f-a98c-ae5e7c2a573f}
        static const GUID IID_IDMLDevice =
            {0x6dbd6437, 0x96fd, 0x423f, {0xa9, 0x8c, 0xae, 0x5e, 0x7c, 0x2a, 0x57, 0x3f}};

        hr = dmlCreate(d3dDev, 0, DML_FEATURE_LEVEL_1_0, IID_IDMLDevice, (void**)&dmlDev);
        printf("  [%u] \"%ls\" -> DML: %s (0x%08lx)\n",
            i, desc.Description,
            SUCCEEDED(hr) ? "OK" : "FAILED", hr);
        if (dmlDev) dmlDev->Release();
        d3dDev->Release();
        adapter->Release();
    }
    factory->Release();
}

// ============================================================
// Path 4: Try ORT directly with DML EP + device_filter=npu
// Load our onnxruntime.dll and check what device it selects
// ============================================================
void test_ort_dml_npu() {
    printf("\n[4] ONNX Runtime DML EP (direct)\n");

    // Try loading from our lib directory first
    SetDllDirectoryA("lib\\ollama\\ortgenai");
    HMODULE ort = LoadLibraryA("lib\\ollama\\ortgenai\\onnxruntime.dll");
    SetDllDirectoryA(NULL);

    if (!ort) {
        printf("  onnxruntime.dll not found in lib/ollama/ortgenai/\n");
        printf("  Trying system onnxruntime.dll...\n");
        ort = LoadLibraryA("onnxruntime.dll");
    }
    if (!ort) {
        printf("  onnxruntime.dll not found anywhere\n");
        return;
    }

    // Just check if the DML EP provider factory is available
    auto getApi = (void* (*)(uint32_t))GetProcAddress(ort, "OrtGetApiBase");
    if (getApi) {
        printf("  OrtGetApiBase found — ORT loaded OK\n");
    } else {
        printf("  OrtGetApiBase not found\n");
    }

    // Check for DML-specific API
    auto sessionOptionsAppend = GetProcAddress(ort, "OrtSessionOptionsAppendExecutionProvider_DML");
    printf("  OrtSessionOptionsAppendExecutionProvider_DML: %s\n",
        sessionOptionsAppend ? "FOUND" : "not found (use generic provider API)");
}

// ============================================================
// Path 5: Check Windows ML / WinML
// ============================================================
void test_winml() {
    printf("\n[5] Windows ML Check\n");

    HMODULE winml = LoadLibraryA("winml.dll");
    printf("  winml.dll: %s\n", winml ? "loaded" : "NOT found");
    if (winml) FreeLibrary(winml);

    HMODULE winai = LoadLibraryA("Windows.AI.MachineLearning.dll");
    printf("  Windows.AI.MachineLearning.dll: %s\n", winai ? "loaded" : "NOT found");
    if (winai) FreeLibrary(winai);

    printf("  (WinML WinRT activation requires C++/WinRT, skipping)\n");
}

// ============================================================
// Path 6: Check if NPU driver is a "real" D3D12 driver
// ============================================================
void test_npu_driver() {
    printf("\n[6] NPU Driver Information\n");

    // Query registry for display adapter info
    HKEY key;
    char subkey[256];
    for (int i = 0; i < 10; i++) {
        snprintf(subkey, sizeof(subkey),
            "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\%04d", i);
        if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, subkey, 0, KEY_READ, &key) != ERROR_SUCCESS)
            continue;

        char desc[256] = {};
        DWORD descSize = sizeof(desc);
        RegQueryValueExA(key, "DriverDesc", NULL, NULL, (BYTE*)desc, &descSize);

        char provider[256] = {};
        DWORD provSize = sizeof(provider);
        RegQueryValueExA(key, "ProviderName", NULL, NULL, (BYTE*)provider, &provSize);

        char infPath[256] = {};
        DWORD infSize = sizeof(infPath);
        RegQueryValueExA(key, "InfPath", NULL, NULL, (BYTE*)infPath, &infSize);

        char umDriver[256] = {};
        DWORD umSize = sizeof(umDriver);
        // UserModeDriverName is the D3D user-mode driver DLL
        RegQueryValueExA(key, "UserModeDriverName", NULL, NULL, (BYTE*)umDriver, &umSize);

        char umDriverGfx[256] = {};
        DWORD umGfxSize = sizeof(umDriverGfx);
        RegQueryValueExA(key, "UserModeDriverNameInKernelMode", NULL, NULL, (BYTE*)umDriverGfx, &umGfxSize);

        if (desc[0]) {
            printf("  [%d] %s\n", i, desc);
            printf("      Provider: %s\n", provider);
            printf("      Inf: %s\n", infPath);
            printf("      UserModeDriver: %s\n", umDriver[0] ? umDriver : "(none)");
            printf("      UserModeDriverKM: %s\n", umDriverGfx[0] ? umDriverGfx : "(none)");
        }

        RegCloseKey(key);
    }
}

int main() {
    printf("============================================\n");
    printf("  NPU Accessibility Probe\n");
    printf("  Testing ALL paths to reach the NPU\n");
    printf("============================================\n\n");

    test_dxgi();
    test_d3d12_dxgi();
    test_directml();
    test_ort_dml_npu();
    test_winml();
    test_npu_driver();

    printf("\n============================================\n");
    printf("  Probe complete\n");
    printf("============================================\n");
    return 0;
}

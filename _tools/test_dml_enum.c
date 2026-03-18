#include <windows.h>
#include <initguid.h>
#include <dxgi1_6.h>
#include <d3d12.h>
#include <stdio.h>

// Minimal DirectML types
enum DML_FEATURE_LEVEL {
    DML_FEATURE_LEVEL_1_0 = 0x1000,
};
enum DML_CREATE_DEVICE_FLAGS {
    DML_CREATE_DEVICE_FLAG_NONE = 0,
};

typedef HRESULT (WINAPI *PFN_DMLCreateDevice1)(
    ID3D12Device *d3d12Device, enum DML_CREATE_DEVICE_FLAGS flags,
    enum DML_FEATURE_LEVEL minFeatureLevel, REFIID riid, void **ppv);

int main() {
    HRESULT hr;
    printf("=== DirectML Device Enumeration Test ===\n\n");

    // Load DirectML
    HMODULE dml_mod = LoadLibraryA("DirectML.dll");
    if (!dml_mod) {
        printf("ERROR: DirectML.dll not found (error %lu)\n", GetLastError());
        return 1;
    }
    printf("OK: DirectML.dll loaded\n");

    PFN_DMLCreateDevice1 DMLCreateDevice1 = (PFN_DMLCreateDevice1)GetProcAddress(dml_mod, "DMLCreateDevice1");
    if (!DMLCreateDevice1) {
        printf("ERROR: DMLCreateDevice1 not found\n");
        return 1;
    }
    printf("OK: DMLCreateDevice1 found\n\n");

    // Create DXGI Factory
    IDXGIFactory4* factory = NULL;
    hr = CreateDXGIFactory1(&IID_IDXGIFactory4, (void**)&factory);
    if (FAILED(hr)) {
        printf("ERROR: CreateDXGIFactory1 failed: 0x%08x\n", (unsigned)hr);
        return 1;
    }
    printf("OK: DXGI Factory created\n\n");

    // Enumerate adapters
    for (UINT i = 0; ; i++) {
        IDXGIAdapter1* adapter = NULL;
        hr = factory->lpVtbl->EnumAdapters1(factory, i, &adapter);
        if (hr == DXGI_ERROR_NOT_FOUND) break;

        DXGI_ADAPTER_DESC1 desc;
        adapter->lpVtbl->GetDesc1(adapter, &desc);

        char name[256] = {};
        WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, name, sizeof(name)-1, NULL, NULL);

        printf("Adapter %u: %s\n", i, name);
        printf("  VendorId: 0x%04X, DeviceId: 0x%04X\n", desc.VendorId, desc.DeviceId);
        printf("  Flags: 0x%X%s\n", desc.Flags, (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ? " (SOFTWARE)" : "");
        printf("  DedicatedVideoMemory: %llu MB\n", (unsigned long long)(desc.DedicatedVideoMemory / (1024*1024)));
        printf("  DedicatedSystemMemory: %llu MB\n", (unsigned long long)(desc.DedicatedSystemMemory / (1024*1024)));
        printf("  SharedSystemMemory: %llu MB\n", (unsigned long long)(desc.SharedSystemMemory / (1024*1024)));

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) {
            printf("  -> Skipping software adapter\n\n");
            adapter->lpVtbl->Release(adapter);
            continue;
        }

        // Try D3D12 device creation
        ID3D12Device* d3d_device = NULL;
        hr = D3D12CreateDevice((IUnknown*)adapter, D3D_FEATURE_LEVEL_11_0, &IID_ID3D12Device, (void**)&d3d_device);
        if (FAILED(hr)) {
            printf("  -> D3D12CreateDevice FAILED: 0x%08x\n\n", (unsigned)hr);
            adapter->lpVtbl->Release(adapter);
            continue;
        }
        printf("  -> D3D12 device created OK\n");

        // Try DirectML device creation
        IUnknown* dml_device = NULL;
        // Use IID_IUnknown as a generic REFIID since we don't have the DML IID
        hr = DMLCreateDevice1(d3d_device, DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_1_0, &IID_IUnknown, (void**)&dml_device);
        if (FAILED(hr)) {
            printf("  -> DMLCreateDevice1 FAILED: 0x%08x\n\n", (unsigned)hr);
        } else {
            printf("  -> DirectML device created OK!\n\n");
            dml_device->lpVtbl->Release(dml_device);
        }

        d3d_device->lpVtbl->Release(d3d_device);
        adapter->lpVtbl->Release(adapter);
    }

    factory->lpVtbl->Release(factory);
    FreeLibrary(dml_mod);

    printf("=== Done ===\n");
    return 0;
}

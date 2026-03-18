// Minimal probe to check if we can create a WinML device targeting NPU
// Build: clang++ -std=c++17 -o npu_winml_probe.exe npu_winml_probe.cpp -lole32 -loleaut32 -lruntimeobject
//   or:  cl /EHsc /std:c++17 npu_winml_probe.cpp ole32.lib oleaut32.lib runtimeobject.lib

#include <stdio.h>
#include <windows.h>
#include <dxgi1_4.h>
#include <d3d12.h>

// We'll also try DXCore path
typedef HRESULT (WINAPI *PFN_DXCoreCreateAdapterFactory)(REFIID riid, void** ppFactory);

#include <initguid.h>

// DXCore GUIDs
DEFINE_GUID(CLSID_DXCoreAdapterFactory, 0x7c6fdb0a, 0x81b0, 0x4d7a, 0xb2, 0x1c, 0xad, 0x7a, 0xb0, 0x53, 0xac, 0x42);
DEFINE_GUID(IID_IDXCoreAdapterFactory, 0x78ee5945, 0xc36e, 0x4b13, 0xa6, 0x69, 0x00, 0x5d, 0xd1, 0x1c, 0x0f, 0x06);
DEFINE_GUID(IID_IDXCoreAdapterList, 0x526c7776, 0x40e9, 0x459b, 0xb7, 0x11, 0xf3, 0x2a, 0xd7, 0x6d, 0xfc, 0x28);
DEFINE_GUID(IID_IDXCoreAdapter, 0xf0db4c7f, 0xfe5a, 0x42a2, 0xbd, 0x62, 0xf2, 0xa6, 0xcf, 0x6f, 0xc8, 0x3e);
DEFINE_GUID(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML, 0xb71b0d41, 0x1088, 0x422f, 0xa2, 0x7c, 0x2, 0x50, 0xb7, 0xd3, 0xa9, 0x88);

// DXCore hardware type attribute
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU, 0xd46140c4, 0xadd7, 0x451b, 0x9e, 0x56, 0xd0, 0x42, 0x59, 0x8f, 0x76, 0x08);
DEFINE_GUID(DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU, 0x52612d67, 0x3aae, 0x4e63, 0xa1, 0xf8, 0x9e, 0x79, 0x62, 0xc4, 0xe7, 0x91);

// IDXCoreAdapter interface (partial - just what we need)
struct IDXCoreAdapter : public IUnknown {
    virtual bool STDMETHODCALLTYPE IsValid() = 0;
    virtual bool STDMETHODCALLTYPE IsAttributeSupported(REFGUID attributeGUID) = 0;
    virtual bool STDMETHODCALLTYPE IsPropertySupported(uint32_t property) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetProperty(uint32_t property, size_t bufferSize, void* propertyData) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetProperty(uint32_t property, size_t bufferSize, const void* propertyData) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetPropertySize(uint32_t property, size_t* bufferSize) = 0;
    virtual bool STDMETHODCALLTYPE IsQueryStateSupported(uint32_t property) = 0;
    virtual HRESULT STDMETHODCALLTYPE QueryState(uint32_t state, size_t inputStateDetailsSize, const void* inputStateDetails, size_t outputBufferSize, void* outputBuffer) = 0;
    virtual bool STDMETHODCALLTYPE IsSetStateSupported(uint32_t property) = 0;
    virtual HRESULT STDMETHODCALLTYPE SetState(uint32_t state, size_t inputStateDetailsSize, const void* inputStateDetails, size_t inputDataSize, const void* inputData) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetFactory(REFIID riid, void** ppFactory) = 0;
};

struct IDXCoreAdapterList : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE GetAdapter(uint32_t index, REFIID riid, void** ppAdapter) = 0;
    virtual uint32_t STDMETHODCALLTYPE GetAdapterCount() = 0;
    virtual bool STDMETHODCALLTYPE IsStale() = 0;
    virtual HRESULT STDMETHODCALLTYPE GetFactory(REFIID riid, void** ppFactory) = 0;
    virtual HRESULT STDMETHODCALLTYPE Sort(uint32_t numPreferences, const uint32_t* preferences) = 0;
    virtual bool STDMETHODCALLTYPE IsAdapterPreferenceSupported(uint32_t preference) = 0;
};

struct IDXCoreAdapterFactory : public IUnknown {
    virtual HRESULT STDMETHODCALLTYPE CreateAdapterList(uint32_t numAttributes, const GUID* filterAttributes, REFIID riid, void** ppAdapterList) = 0;
    virtual HRESULT STDMETHODCALLTYPE GetAdapterByLuid(LUID adapterLuid, REFIID riid, void** ppAdapter) = 0;
    virtual bool STDMETHODCALLTYPE IsNotificationTypeSupported(uint32_t notificationType) = 0;
    virtual HRESULT STDMETHODCALLTYPE RegisterEventNotification(IUnknown* dxCoreObject, uint32_t notificationType, void* callback, void* callbackContext, uint32_t* eventCookie) = 0;
    virtual HRESULT STDMETHODCALLTYPE UnregisterEventNotification(uint32_t eventCookie) = 0;
};

// DXCore property enums
#define DXCORE_ADAPTER_PROPERTY_DRIVER_DESCRIPTION 1
#define DXCORE_ADAPTER_PROPERTY_HARDWARE_ID 2
#define DXCORE_ADAPTER_PROPERTY_IS_HARDWARE 4

int main() {
    printf("=== NPU Accessibility Probe ===\n\n");

    // 1. DXCore enumeration
    printf("[1] DXCore Enumeration\n");
    HMODULE dxcore = LoadLibraryA("dxcore.dll");
    if (!dxcore) {
        printf("  ERROR: Cannot load dxcore.dll\n");
    } else {
        auto createFactory = (PFN_DXCoreCreateAdapterFactory)GetProcAddress(dxcore, "DXCoreCreateAdapterFactory");
        if (!createFactory) {
            printf("  ERROR: DXCoreCreateAdapterFactory not found\n");
        } else {
            IDXCoreAdapterFactory* factory = nullptr;
            HRESULT hr = createFactory(IID_IDXCoreAdapterFactory, (void**)&factory);
            if (FAILED(hr)) {
                printf("  ERROR: DXCoreCreateAdapterFactory failed: 0x%08lx\n", hr);
            } else {
                GUID attrs[] = { DXCORE_ADAPTER_ATTRIBUTE_D3D12_GENERIC_ML };
                IDXCoreAdapterList* list = nullptr;
                hr = factory->CreateAdapterList(1, attrs, IID_IDXCoreAdapterList, (void**)&list);
                if (FAILED(hr)) {
                    printf("  ERROR: CreateAdapterList failed: 0x%08lx\n", hr);
                } else {
                    uint32_t count = list->GetAdapterCount();
                    printf("  Found %u GENERIC_ML adapters\n", count);

                    for (uint32_t i = 0; i < count; i++) {
                        IDXCoreAdapter* adapter = nullptr;
                        hr = list->GetAdapter(i, IID_IDXCoreAdapter, (void**)&adapter);
                        if (FAILED(hr)) continue;

                        // Get name
                        size_t nameSize = 0;
                        adapter->GetPropertySize(DXCORE_ADAPTER_PROPERTY_DRIVER_DESCRIPTION, &nameSize);
                        char name[256] = {};
                        if (nameSize > 0 && nameSize < sizeof(name)) {
                            adapter->GetProperty(DXCORE_ADAPTER_PROPERTY_DRIVER_DESCRIPTION, nameSize, name);
                        }

                        bool isNPU = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_NPU);
                        bool isGPU = adapter->IsAttributeSupported(DXCORE_HARDWARE_TYPE_ATTRIBUTE_GPU);
                        bool isHW = false;
                        adapter->GetProperty(DXCORE_ADAPTER_PROPERTY_IS_HARDWARE, sizeof(isHW), &isHW);

                        printf("  [%u] \"%s\" npu=%d gpu=%d hw=%d\n", i, name, isNPU, isGPU, isHW);

                        if (isNPU) {
                            printf("      -> NPU FOUND! Trying D3D12 device creation...\n");

                            // Get LUID for D3D12 device creation
                            // Try all feature levels
                            typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
                            HMODULE d3d12 = LoadLibraryA("d3d12.dll");
                            if (d3d12) {
                                auto createDev = (PFN_D3D12CreateDevice)GetProcAddress(d3d12, "D3D12CreateDevice");
                                if (createDev) {
                                    // Try with the DXCore adapter as IUnknown
                                    D3D_FEATURE_LEVEL levels[] = {
                                        (D3D_FEATURE_LEVEL)0x100,  // D3D_FEATURE_LEVEL_1_0_GENERIC
                                        (D3D_FEATURE_LEVEL)0x1000, // D3D_FEATURE_LEVEL_1_0_CORE
                                        D3D_FEATURE_LEVEL_11_0,
                                        D3D_FEATURE_LEVEL_12_0,
                                    };
                                    const char* levelNames[] = {"1_0_GENERIC", "1_0_CORE", "11_0", "12_0"};

                                    for (int l = 0; l < 4; l++) {
                                        ID3D12Device* dev = nullptr;
                                        hr = createDev((IUnknown*)adapter, levels[l], __uuidof(ID3D12Device), (void**)&dev);
                                        if (SUCCEEDED(hr) && dev) {
                                            printf("      -> D3D12CreateDevice(%s) SUCCEEDED!\n", levelNames[l]);
                                            dev->Release();
                                        } else {
                                            printf("      -> D3D12CreateDevice(%s) FAILED: 0x%08lx\n", levelNames[l], hr);
                                        }
                                    }
                                }
                            }
                        }

                        adapter->Release();
                    }
                    list->Release();
                }
                factory->Release();
            }
        }
    }

    // 2. Try WinML via LoadLibrary
    printf("\n[2] Windows ML (winml.dll) Check\n");
    HMODULE winml = LoadLibraryA("winml.dll");
    if (winml) {
        printf("  winml.dll loaded OK\n");
        // WinML is a WinRT API, can't easily call from plain C
        // But its presence confirms the platform support
        FreeLibrary(winml);
    } else {
        printf("  winml.dll NOT found\n");
    }

    HMODULE winai = LoadLibraryA("Windows.AI.MachineLearning.dll");
    if (winai) {
        printf("  Windows.AI.MachineLearning.dll loaded OK (size check)\n");
        FreeLibrary(winai);
    }

    // 3. Check DirectML.dll version
    printf("\n[3] DirectML Check\n");
    HMODULE dml = LoadLibraryA("DirectML.dll");
    if (!dml) {
        // Try system path
        dml = LoadLibraryA("C:\\Windows\\System32\\DirectML.dll");
    }
    if (dml) {
        printf("  DirectML.dll loaded OK\n");

        // Check for DMLCreateDevice1
        auto dmlCreate1 = GetProcAddress(dml, "DMLCreateDevice1");
        printf("  DMLCreateDevice1: %s\n", dmlCreate1 ? "FOUND" : "not found");

        FreeLibrary(dml);
    } else {
        printf("  DirectML.dll NOT found\n");
    }

    printf("\n=== Probe complete ===\n");
    return 0;
}

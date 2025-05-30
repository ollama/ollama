#include <iostream>
#include <dlfcn.h>
#include <vector>
#include <string>
#include <cstring>

// Function pointer types matching the SYCL wrapper functions
typedef int (*GetPlatformCountFunc)();
typedef int (*GetDeviceCountFunc)();
typedef void (*GetDeviceIdsFunc)(int*, int);
typedef void (*GetDeviceNameFunc)(int, char*, size_t);
typedef void (*GetDeviceVendorFunc)(int, char*, size_t);
typedef void (*GetDeviceMemoryFunc)(int, size_t*, size_t*);
typedef int (*IsGpuFunc)(int);

int main() {
    // Load the SYCL wrapper library
    void* handle = dlopen("./libsycl_wrapper.so", RTLD_LAZY);
    if (!handle) {
        std::cerr << "Error loading library: " << dlerror() << std::endl;
        return 1;
    }

    // Load all the functions
    GetPlatformCountFunc getPlatformCount = (GetPlatformCountFunc)dlsym(handle, "sycl_get_platform_count");
    GetDeviceCountFunc getDeviceCount = (GetDeviceCountFunc)dlsym(handle, "sycl_get_device_count");
    GetDeviceIdsFunc getDeviceIds = (GetDeviceIdsFunc)dlsym(handle, "sycl_get_device_ids");
    GetDeviceNameFunc getDeviceName = (GetDeviceNameFunc)dlsym(handle, "sycl_get_device_name");
    GetDeviceVendorFunc getDeviceVendor = (GetDeviceVendorFunc)dlsym(handle, "sycl_get_device_vendor");
    GetDeviceMemoryFunc getDeviceMemory = (GetDeviceMemoryFunc)dlsym(handle, "sycl_get_device_memory");
    IsGpuFunc isGpu = (IsGpuFunc)dlsym(handle, "sycl_is_gpu");

    // Check for errors in function loading
    const char* error = dlerror();
    if (error) {
        std::cerr << "Error loading functions: " << error << std::endl;
        dlclose(handle);
        return 1;
    }

    // Test platform count
    std::cout << "=== Testing SYCL Wrapper Functions ===" << std::endl;
    std::cout << "\nPlatforms:" << std::endl;
    int platformCount = getPlatformCount();
    std::cout << "Platform count: " << platformCount << std::endl;

    // Test device count
    std::cout << "\nDevices:" << std::endl;
    int deviceCount = getDeviceCount();
    std::cout << "Device count: " << deviceCount << std::endl;

    if (deviceCount > 0) {
        // Get device IDs
        std::vector<int> deviceIds(deviceCount);
        getDeviceIds(deviceIds.data(), deviceCount);

        // Test each device
        for (int i = 0; i < deviceCount; i++) {
            std::cout << "\nDevice " << i << " (ID: " << deviceIds[i] << "):" << std::endl;
            
            // Test device name
            char name[256] = {0};
            getDeviceName(i, name, sizeof(name));
            std::cout << "  Name: " << name << std::endl;
            
            // Test device vendor
            char vendor[256] = {0};
            getDeviceVendor(i, vendor, sizeof(vendor));
            std::cout << "  Vendor: " << vendor << std::endl;
            
            // Test device memory
            size_t free = 0, total = 0;
            getDeviceMemory(i, &free, &total);
            std::cout << "  Memory: " << (free / (1024*1024)) << " MB free / " 
                      << (total / (1024*1024)) << " MB total" << std::endl;
            
            // Test if GPU
            bool gpu = isGpu(i) != 0;
            std::cout << "  Is GPU: " << (gpu ? "Yes" : "No") << std::endl;
        }
    }

    // Close the library
    dlclose(handle);
    return 0;
} 
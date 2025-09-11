#include "device.cuh"
#include <mutex>

static cudaDeviceProp  deviceProp;
static std::once_flag  devicePropInitFlag;

static void InitDeviceProp() {
    int dev = 0;
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        // fallback: assume device 0
        dev = 0;
        cudaGetDevice(&dev);
    }
    // query and stash all of its properties
    cudaGetDeviceProperties(&deviceProp, dev);
}

const cudaDeviceProp & getCachedDeviceProperties() {
    std::call_once(devicePropInitFlag, InitDeviceProp);
    return deviceProp;
}

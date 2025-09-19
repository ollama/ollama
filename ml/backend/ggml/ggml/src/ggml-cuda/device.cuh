#pragma once

#include "common.cuh"
#include <cuda_runtime.h>

/// Get a reference to the cached cudaDeviceProp for the current CUDA device.
/// The first call will perform cudaGetDevice() + cudaGetDeviceProperties(),
/// and all subsequent calls are just a fast reference lookup.
const cudaDeviceProp & getCachedDeviceProperties();

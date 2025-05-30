# SYCL Wrapper Library for Ollama

This library provides a C wrapper around the SYCL C++ API, allowing Ollama to discover and use SYCL-compatible GPUs.

## Requirements

- A C++17 compiler
- A SYCL implementation (Intel oneAPI DPC++ or ComputeCpp)

## Building

### For Intel oneAPI DPC++

1. Install Intel oneAPI Base Toolkit which includes DPC++
2. Set up the oneAPI environment:
   ```
   source /opt/intel/oneapi/setvars.sh
   ```
3. Build the library:
   ```
   make
   ```

### For ComputeCpp

1. Install ComputeCpp
2. Edit the Makefile to uncomment the ComputeCpp flags and set the COMPUTECPP_DIR variable
3. Comment out the Intel extensions: `#INTEL_EXTENSIONS = -DDPCT_COMPATIBILITY_TEMP`
4. Build the library:
   ```
   make COMPUTECPP_DIR=/path/to/computecpp
   ```

## Installation

```
sudo make install
```

This will install the library to `/usr/local/lib/libsycl.so`.

## Usage

Once installed, Ollama will be able to discover SYCL-compatible GPUs using this library.

## Functions

The library exports the following C functions:

- `int sycl_get_platform_count()`: Get the number of SYCL platforms available
- `int sycl_get_device_count()`: Get the total number of devices across all platforms
- `void sycl_get_device_ids(int* id_list, int max_len)`: Get a list of all device IDs
- `void sycl_get_device_name(int device_index, char* name, size_t name_size)`: Get the name of a device
- `void sycl_get_device_vendor(int device_index, char* vendor, size_t vendor_size)`: Get the vendor of a device
- `void sycl_get_device_memory(int device_index, size_t* free, size_t* total)`: Get memory information for a device
- `int sycl_is_gpu(int device_index)`: Check if a device is a GPU

## Memory Reporting

Standard SYCL does not provide an API to query available (free) memory on a device. This library handles memory reporting in the following ways:

1. **Intel DPC++ Extensions**: If using Intel oneAPI DPC++ with extensions enabled, the library will use `dpct::get_memory_info()` to get accurate free memory information.

2. **Estimation Fallback**: If extensions are not available or fail, the library will estimate free memory as 80% of total memory.

For more accurate memory reporting with specific vendors, you may need to modify the code to use vendor-specific interoperability:
- NVIDIA GPUs: Use CUDA interop to call `cudaMemGetInfo()`
- Intel GPUs: Use Level Zero interop
- AMD GPUs: Use ROCm interop

## Notes

- To enable Intel DPC++ extensions, the `DPCT_COMPATIBILITY_TEMP` macro is defined in the Makefile
- If not using Intel DPC++, comment out the `INTEL_EXTENSIONS` line in the Makefile 
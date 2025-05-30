#include <CL/sycl.hpp>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

// Debug logging control
#define SYCL_WRAPPER_DEBUG 1  // Set to 0 to disable debug output

// Debug logging function
static void debug_log(const std::string& message) {
#if SYCL_WRAPPER_DEBUG
    static std::ofstream log_file;
    if (!log_file.is_open()) {
        log_file.open("sycl_wrapper_debug.log", std::ios::out | std::ios::app);
    }
    
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    log_file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S")
             << '.' << std::setfill('0') << std::setw(3) << now_ms.count()
             << " [SYCL] " << message << std::endl;
    
    // Also print to stderr for immediate visibility
    std::cerr << "[SYCL] " << message << std::endl;
#endif
}

// Helper functions for device type and backend information
inline std::string get_device_type_name(const cl::sycl::device &Device) {
    auto DeviceType = Device.get_info<cl::sycl::info::device::device_type>();
    switch (DeviceType) {
    case cl::sycl::info::device_type::cpu:
        return "cpu";
    case cl::sycl::info::device_type::gpu:
        return "gpu";
    case cl::sycl::info::device_type::host:
        return "host";
    case cl::sycl::info::device_type::accelerator:
        return "acc";
    default:
        return "unknown";
    }
}

inline std::string get_device_backend_and_type(const cl::sycl::device &device) {
    std::stringstream device_type;
    cl::sycl::backend backend = device.get_backend();
    device_type << backend << ":" << get_device_type_name(device);
    return device_type.str();
}

// Helper function to log backend information for a device
static void log_backend_info(const cl::sycl::device& device) {
#if SYCL_WRAPPER_DEBUG
    try {
        std::stringstream ss;
        ss << "Backend info for device: " << device.get_info<cl::sycl::info::device::name>() << std::endl;
        
        // Use the provided helper functions
        ss << "  Device Type: " << get_device_type_name(device) << std::endl;
        ss << "  Backend and Type: " << get_device_backend_and_type(device) << std::endl;
        
        // Standard device info that works across all SYCL implementations
        ss << "  Device Vendor: " << device.get_info<cl::sycl::info::device::vendor>() << std::endl;
        ss << "  Device Version: " << device.get_info<cl::sycl::info::device::version>() << std::endl;
        ss << "  Driver Version: " << device.get_info<cl::sycl::info::device::driver_version>() << std::endl;
        ss << "  Global Mem Size: " << device.get_info<cl::sycl::info::device::global_mem_size>() << " bytes" << std::endl;
        ss << "  Local Mem Size: " << device.get_info<cl::sycl::info::device::local_mem_size>() << " bytes" << std::endl;
        ss << "  Max Compute Units: " << device.get_info<cl::sycl::info::device::max_compute_units>() << std::endl;
        ss << "  Max Work Group Size: " << device.get_info<cl::sycl::info::device::max_work_group_size>() << std::endl;
        
        // Platform info
        auto platform = device.get_platform();
        ss << "  Platform Name: " << platform.get_info<cl::sycl::info::platform::name>() << std::endl;
        ss << "  Platform Vendor: " << platform.get_info<cl::sycl::info::platform::vendor>() << std::endl;
        ss << "  Platform Version: " << platform.get_info<cl::sycl::info::platform::version>() << std::endl;
        
#ifdef DPCT_COMPATIBILITY_TEMP
        // Try Intel-specific extensions if available
        try {
            // This is a safer approach that doesn't rely on specific backend types
            ss << "  Intel DPC++ compatibility is enabled" << std::endl;
        } catch (...) {
            ss << "  Failed to use Intel extensions" << std::endl;
        }
#endif
        
        debug_log(ss.str());
    } catch (const std::exception& e) {
        debug_log("Error in log_backend_info: " + std::string(e.what()));
    } catch (...) {
        debug_log("Unknown error in log_backend_info");
    }
#endif
}

// Include Intel DPC++ compatibility headers if extensions are enabled
#ifdef DPCT_COMPATIBILITY_TEMP
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <dpct/dpct.hpp>
#endif

// This file implements a C wrapper around the SYCL C++ API
// It exports C functions that can be loaded dynamically by Ollama

extern "C" {

// Get the number of SYCL platforms available
int sycl_get_platform_count() {
    debug_log("Calling sycl_get_platform_count()");
    try {
        auto platforms = cl::sycl::platform::get_platforms();
        debug_log("Found " + std::to_string(platforms.size()) + " SYCL platforms");
        return platforms.size();
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_platform_count: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_platform_count: " + std::string(e.what()));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_platform_count: " << e.what() << std::endl;
        debug_log("Error in sycl_get_platform_count: " + std::string(e.what()));
        return 0;
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_platform_count" << std::endl;
        debug_log("Unknown error in sycl_get_platform_count");
        return 0;
    }
}

// Get the total number of devices across all platforms
int sycl_get_device_count() {
    debug_log("Calling sycl_get_device_count()");
    try {
        int count = 0;
        auto platforms = cl::sycl::platform::get_platforms();
        
        for (const auto& platform : platforms) {
            auto devices = platform.get_devices();
            count += devices.size();
            debug_log("Platform: " + platform.get_info<cl::sycl::info::platform::name>() + 
                      " has " + std::to_string(devices.size()) + " devices");
        }
        
        debug_log("Total device count: " + std::to_string(count));
        return count;
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_device_count: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_device_count: " + std::string(e.what()));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_device_count: " << e.what() << std::endl;
        debug_log("Error in sycl_get_device_count: " + std::string(e.what()));
        return 0;
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_device_count" << std::endl;
        debug_log("Unknown error in sycl_get_device_count");
        return 0;
    }
}

// Get a list of all device IDs
void sycl_get_device_ids(int* id_list, int max_len) {
    debug_log("Calling sycl_get_device_ids(max_len=" + std::to_string(max_len) + ")");
    try {
        int index = 0;
        auto platforms = cl::sycl::platform::get_platforms();
        
        for (const auto& platform : platforms) {
            auto devices = platform.get_devices();
            
            for (size_t i = 0; i < devices.size() && index < max_len; i++) {
                id_list[index++] = index;  // Just use sequential IDs
                debug_log("Added device ID " + std::to_string(index-1) + " to list");
            }
        }
        debug_log("Populated " + std::to_string(index) + " device IDs");
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_device_ids: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_device_ids: " + std::string(e.what()));
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_device_ids: " << e.what() << std::endl;
        debug_log("Error in sycl_get_device_ids: " + std::string(e.what()));
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_device_ids" << std::endl;
        debug_log("Unknown error in sycl_get_device_ids");
    }
}

// Helper function to get a device by index
cl::sycl::device get_device_by_index(int device_index) {
    debug_log("Getting device by index: " + std::to_string(device_index));
    auto platforms = cl::sycl::platform::get_platforms();
    int current_index = 0;
    
    for (const auto& platform : platforms) {
        auto devices = platform.get_devices();
        
        for (const auto& device : devices) {
            if (current_index == device_index) {
                debug_log("Found device at index " + std::to_string(device_index) + 
                          ": " + device.get_info<cl::sycl::info::device::name>());
                log_backend_info(device);
                return device;
            }
            current_index++;
        }
    }
    
    // If we get here, the index is out of range
    debug_log("Device index out of range: " + std::to_string(device_index));
    throw std::out_of_range("Device index out of range");
}

// Get the name of a device
void sycl_get_device_name(int device_index, char* name, size_t name_size) {
    debug_log("Calling sycl_get_device_name(index=" + std::to_string(device_index) + 
              ", name_size=" + std::to_string(name_size) + ")");
    try {
        auto device = get_device_by_index(device_index);
        std::string device_name = device.get_info<cl::sycl::info::device::name>();
        
        strncpy(name, device_name.c_str(), name_size - 1);
        name[name_size - 1] = '\0';  // Ensure null termination
        debug_log("Device name: " + device_name);
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_device_name: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_device_name: " + std::string(e.what()));
        strncpy(name, "Unknown SYCL Device", name_size - 1);
        name[name_size - 1] = '\0';
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_device_name: " << e.what() << std::endl;
        debug_log("Error in sycl_get_device_name: " + std::string(e.what()));
        strncpy(name, "Error", name_size - 1);
        name[name_size - 1] = '\0';
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_device_name" << std::endl;
        debug_log("Unknown error in sycl_get_device_name");
        strncpy(name, "Error", name_size - 1);
        name[name_size - 1] = '\0';
    }
}

// Get the vendor of a device
void sycl_get_device_vendor(int device_index, char* vendor, size_t vendor_size) {
    debug_log("Calling sycl_get_device_vendor(index=" + std::to_string(device_index) + 
              ", vendor_size=" + std::to_string(vendor_size) + ")");
    try {
        auto device = get_device_by_index(device_index);
        std::string device_vendor = device.get_info<cl::sycl::info::device::vendor>();
        
        strncpy(vendor, device_vendor.c_str(), vendor_size - 1);
        vendor[vendor_size - 1] = '\0';  // Ensure null termination
        debug_log("Device vendor: " + device_vendor);
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_device_vendor: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_device_vendor: " + std::string(e.what()));
        strncpy(vendor, "Unknown", vendor_size - 1);
        vendor[vendor_size - 1] = '\0';
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_device_vendor: " << e.what() << std::endl;
        debug_log("Error in sycl_get_device_vendor: " + std::string(e.what()));
        strncpy(vendor, "Error", vendor_size - 1);
        vendor[vendor_size - 1] = '\0';
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_device_vendor" << std::endl;
        debug_log("Unknown error in sycl_get_device_vendor");
        strncpy(vendor, "Error", vendor_size - 1);
        vendor[vendor_size - 1] = '\0';
    }
}

// Get memory information for a device
void sycl_get_device_memory(int device_index, size_t* free, size_t* total) {
    debug_log("Calling sycl_get_device_memory(index=" + std::to_string(device_index) + ")");
    try {
        auto device = get_device_by_index(device_index);
        
        // Get total global memory
        *total = device.get_info<cl::sycl::info::device::global_mem_size>();
        debug_log("Total memory: " + std::to_string(*total) + " bytes");
        
        // Try to get free memory using Intel's DPC++ extension if available
#ifdef DPCT_COMPATIBILITY_TEMP
        // Intel DPC++ extension approach
        try {
            dpct::dev_mgr::instance().get_device(device_index).get_memory_info(*free, *total);
            debug_log("Free memory (from extension): " + std::to_string(*free) + " bytes");
        } catch (...) {
            // If the extension fails, fall back to estimation
            *free = *total * 0.8;  // Assume 80% is free
            debug_log("Free memory (estimated from catch): " + std::to_string(*free) + " bytes");
        }
#else
        // SYCL doesn't provide a direct way to get free memory
        // We'll estimate it as a percentage of total for now
        *free = *total * 0.8;  // Assume 80% is free
        debug_log("Free memory (estimated): " + std::to_string(*free) + " bytes");
        
        // Note: For vendor-specific implementations, you might be able to use:
        // - For NVIDIA: Use CUDA interop to call cudaMemGetInfo
        // - For Intel: Use Level Zero interop
        // - For AMD: Use ROCm interop
#endif
        
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_get_device_memory: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_get_device_memory: " + std::string(e.what()));
        *free = 0;
        *total = 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_get_device_memory: " << e.what() << std::endl;
        debug_log("Error in sycl_get_device_memory: " + std::string(e.what()));
        *free = 0;
        *total = 0;
    } catch (...) {
        std::cerr << "Unknown error in sycl_get_device_memory" << std::endl;
        debug_log("Unknown error in sycl_get_device_memory");
        *free = 0;
        *total = 0;
    }
}

// Check if a device is a GPU
int sycl_is_gpu(int device_index) {
    debug_log("Calling sycl_is_gpu(index=" + std::to_string(device_index) + ")");
    try {
        auto device = get_device_by_index(device_index);
        bool is_gpu = device.is_gpu();
        debug_log("Device is GPU: " + std::string(is_gpu ? "true" : "false"));
        return is_gpu ? 1 : 0;
    } catch (const cl::sycl::exception& e) {
        std::cerr << "SYCL error in sycl_is_gpu: " << e.what() << std::endl;
        debug_log("SYCL error in sycl_is_gpu: " + std::string(e.what()));
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error in sycl_is_gpu: " << e.what() << std::endl;
        debug_log("Error in sycl_is_gpu: " + std::string(e.what()));
        return 0;
    } catch (...) {
        std::cerr << "Unknown error in sycl_is_gpu" << std::endl;
        debug_log("Unknown error in sycl_is_gpu");
        return 0;
    }
}

} // extern "C" 
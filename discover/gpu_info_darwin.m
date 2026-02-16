#import <Foundation/Foundation.h>
#import <mach/mach.h>
#include "gpu_info_darwin.h"

uint64_t getRecommendedMaxVRAM() {
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  uint64_t result = device.recommendedMaxWorkingSetSize;
  CFRelease(device);
  return result;
}

// getPhysicalMemory returns the total physical memory in bytes
uint64_t getPhysicalMemory() {
  return [NSProcessInfo processInfo].physicalMemory;
}

// getFreeMemory returns the total free memory in bytes, including inactive
// memory that can be reclaimed by the system.
uint64_t getFreeMemory() {
  mach_port_t host_port = mach_host_self();
  mach_msg_type_number_t host_size = sizeof(vm_statistics64_data_t) / sizeof(integer_t);
  vm_size_t pagesize;
  vm_statistics64_data_t vm_stat;

  host_page_size(host_port, &pagesize);
  if (host_statistics64(host_port, HOST_VM_INFO64, (host_info64_t)&vm_stat, &host_size) != KERN_SUCCESS) {
    return 0;
  }

  uint64_t free_memory = (uint64_t)vm_stat.free_count * pagesize;
  free_memory += (uint64_t)vm_stat.speculative_count * pagesize;
  free_memory += (uint64_t)vm_stat.inactive_count * pagesize;

  return free_memory;
}

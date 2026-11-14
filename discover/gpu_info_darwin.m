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

static kern_return_t readMemoryStatistics(vm_statistics64_data_t *stats, vm_size_t *pagesize) {
  mach_port_t host = mach_host_self();
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  kern_return_t status = host_page_size(host, pagesize);
  if (status == KERN_SUCCESS) {
    status = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)stats, &count);
  }
  mach_port_deallocate(mach_task_self(), host);
  return status;
}

// getFreeMemory returns the total free memory in bytes, including inactive
// memory that can be reclaimed by the system.
uint64_t getFreeMemory() {
  vm_size_t pagesize;
  vm_statistics64_data_t vm_stat;
  if (readMemoryStatistics(&vm_stat, &pagesize) != KERN_SUCCESS) {
    return 0;
  }

  uint64_t used = (uint64_t)vm_stat.active_count * pagesize
    + (uint64_t)vm_stat.inactive_count * pagesize
    + (uint64_t)vm_stat.speculative_count * pagesize
    + (uint64_t)vm_stat.wire_count * pagesize
    + (uint64_t)vm_stat.compressor_page_count * pagesize
    - (uint64_t)vm_stat.purgeable_count * pagesize
    - (uint64_t)vm_stat.external_page_count * pagesize;

  uint64_t total_memory = [NSProcessInfo processInfo].physicalMemory;
  if (used >= total_memory) {
    return 0;
  }
  return total_memory - used;
}

// Observe reclaimable capacity separately from immediately available pages and
// uncompressed content stored by the system compressor.
int getMemoryHeadroom(uint64_t *available, uint64_t *immediate, uint64_t *compressed) {
  vm_size_t pagesize;
  vm_statistics64_data_t stats;
  if (readMemoryStatistics(&stats, &pagesize) != KERN_SUCCESS) {
    return 0;
  }
  *immediate = ((uint64_t)stats.free_count + stats.purgeable_count
    - stats.speculative_count) * pagesize;
  *available = *immediate + (uint64_t)stats.external_page_count * pagesize;
  *compressed = stats.total_uncompressed_pages_in_compressor * pagesize;
  return 1;
}

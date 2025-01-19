#ifndef __APPLE__
#ifndef __GPU_INFO_VULKAN_H__
#define __GPU_INFO_VULKAN_H__

#include "gpu_info.h"

#ifdef __linux__
#include <sys/capability.h>
#endif

#include <vulkan/vulkan.h>

typedef struct {
  void* vk_handle;
  void* cap_handle;
  uint16_t verbose;

  VkInstance vk;
  int num_devices;

#ifdef __linux__
  cap_t (*cap_get_proc)(void);

  int (*cap_get_bound)(cap_value_t);
  int (*cap_set_flag)(cap_t, cap_flag_t, int, const cap_value_t *, cap_flag_value_t);
  int (*cap_set_proc)(cap_t);
  int (*cap_free)(cap_t);
#endif

  void (*vkGetPhysicalDeviceProperties)(
    VkPhysicalDevice                            physicalDevice,
    VkPhysicalDeviceProperties*                 pProperties);
  VkResult (*vkEnumerateDeviceExtensionProperties)(
      VkPhysicalDevice                            physicalDevice,
      const char*                                 pLayerName,
      uint32_t*                                   pPropertyCount,
      VkExtensionProperties*                      pProperties);
  VkResult (*vkCreateInstance)(
      const VkInstanceCreateInfo*                 pCreateInfo,
      const VkAllocationCallbacks*                pAllocator,
      VkInstance*                                 pInstance);
  VkResult (*vkEnumeratePhysicalDevices)(
      VkInstance                                  instance,
      uint32_t*                                   pPhysicalDeviceCount,
      VkPhysicalDevice*                           pPhysicalDevices);
  void (*vkGetPhysicalDeviceMemoryProperties2)(
      VkPhysicalDevice                            physicalDevice,
      VkPhysicalDeviceMemoryProperties2*          pMemoryProperties);
  void (*vkDestroyInstance)(
      VkInstance                                  instance,
      const VkAllocationCallbacks*                pAllocator);
} vk_handle_t;

typedef struct vk_init_resp
{
  char *err; // If err is non-null handle is invalid
  int num_devices;
  vk_handle_t ch;
} vk_init_resp_t;

void vk_init(char* vk_lib_path, char* cap_lib_path, vk_init_resp_t *resp);
void vk_check_vram(vk_handle_t rh, int i, mem_info_t *resp);
void vk_release(vk_handle_t rh);

#endif
#endif
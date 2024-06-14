#include "gpu_vulkan.h"

#include <string.h>

int check_perfmon() {
#ifdef __linux__
  cap_t caps;
  const cap_value_t cap_list[2] = {CAP_PERFMON};

  if (!CAP_IS_SUPPORTED(CAP_SETFCAP))
    return -1;

  caps = cap_get_proc();
  if (caps == NULL)
    return -1;

  if (cap_set_flag(caps, CAP_EFFECTIVE, 2, cap_list, CAP_SET) == -1)
    return -1;

  if (cap_set_proc(caps) == -1)
    return -1;

  if (cap_free(caps) == -1)
    return -1;

  return 0;
#else
  return 0;
#endif
}

void vk_init(vk_init_resp_t *resp) {
  if (check_perfmon() != 0) {
    resp->err = "Performance monitoring is not allowed. Please enable CAP_PERFMON or run as root to use Vulkan.";
    return;
  }

  VkInstance instance;
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = NULL;
  appInfo.pApplicationName = "Ollama";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "No Engine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_2;
  VkInstanceCreateInfo createInfo = {};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  createInfo.pNext = NULL;
  createInfo.flags = 0;
  createInfo.enabledExtensionCount = 1;
  const char* extensions[] = { VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME };
  createInfo.ppEnabledExtensionNames = extensions;
  createInfo.pApplicationInfo = &appInfo;
  VkResult result = vkCreateInstance(&createInfo, NULL, &instance);
  if (result != VK_SUCCESS) {
    resp.err = sprintf("Failed to create instance: %d", result);
    return;
  }

  uint32_t deviceCount;
  result = vkEnumeratePhysicalDevices(instance, &deviceCount, NULL);
  if (result != VK_SUCCESS) {
    resp.err = sprintf("Failed to enumerate physical devices: %d", result);
    return;
  }

  resp.err = NULL;
  resp.oh = instance;
  resp.num_devices = deviceCount;
}

void vk_check_vram(vk_handle_t rh, int i, mem_info_t *resp) {
  uint32_t deviceCount = rh->num_devices;
  VkInstance instance = rh->oh;

  VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
  result = vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
  if (result != VK_SUCCESS) {
    resp.err = sprintf("Failed to enumerate physical devices: %d", result);
    return;
  }

  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(devices[i], &properties);
  LOG(h.verbose, "Vulkan device %d: %s\n", i, properties.deviceName);
  int supports_budget = support_memory_budget(devices[i]);
  if (!supports_budget) {
    resp.err = sprintf("Device %d does not support memory budget\n", i);
    return;
  }
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    resp.err = sprintf("Device %d is a CPU, skipped\n", i);
    return;
  }

  VkPhysicalDeviceMemoryBudgetPropertiesEXT physical_device_memory_budget_properties;
  physical_device_memory_budget_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  physical_device_memory_budget_properties.pNext = NULL;

  VkPhysicalDeviceMemoryProperties2 device_memory_properties;
  device_memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  device_memory_properties.pNext = &physical_device_memory_budget_properties;

  vkGetPhysicalDeviceMemoryProperties2(devices[i], &device_memory_properties);

  VkDeviceSize device_memory_total_usage = 0;
  VkDeviceSize device_memory_heap_budget = 0;

  for (uint32_t j = 0; j < device_memory_properties.memoryProperties.memoryHeapCount; j++) {
    VkMemoryHeap heap = device_memory_properties.memoryProperties.memoryHeaps[j];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_memory_total_usage += physical_device_memory_budget_properties.heapUsage[j];
      device_memory_heap_budget += physical_device_memory_budget_properties.heapBudget[j];
    }
  }

  resp->err = NULL;
  snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
  snprintf(&resp->gpu_name[0], GPU_NAME_LEN, "%s", properties.deviceName);
  resp->total = (uint64_t) device_memory_total_usage;
  resp->free = (uint64_t) device_memory_total_usage;
  resp->major = VK_API_VERSION_MAJOR(properties.apiVersion);
  resp->minor = VK_API_VERSION_MINOR(properties.apiVersion);
  resp->patch = VK_API_VERSION_PATCH(properties.apiVersion);
}

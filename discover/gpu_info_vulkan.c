#ifndef __APPLE__
#include "gpu_info_vulkan.h"

#include <string.h>

int is_extension_supported(vk_handle_t* rh, VkPhysicalDevice device, char* extension) {
  uint32_t extensionCount;
  (*rh->vkEnumerateDeviceExtensionProperties)(device, NULL, &extensionCount, NULL);

  if (extensionCount == 0) {
    return 0;
  }

  VkExtensionProperties* extensions = malloc(extensionCount * sizeof(VkExtensionProperties));
  if (extensions == NULL) {
    return 0;
  }

  (*rh->vkEnumerateDeviceExtensionProperties)(device, NULL, &extensionCount, extensions);

  for (int j = 0; j < extensionCount; j++) {
    if (strcmp(extensions[j].extensionName, extension) == 0) {
      free(extensions);
      return 1;
    }
  }

  free(extensions);
  return 0;
}

void vk_init(char* vk_lib_path, vk_init_resp_t *resp) {
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"vkGetPhysicalDeviceProperties", (void *)&resp->ch.vkGetPhysicalDeviceProperties},
      {"vkGetPhysicalDeviceProperties2", (void *)&resp->ch.vkGetPhysicalDeviceProperties2},
      {"vkEnumerateDeviceExtensionProperties", (void *)&resp->ch.vkEnumerateDeviceExtensionProperties},
      {"vkCreateInstance", (void *)&resp->ch.vkCreateInstance},
      {"vkEnumeratePhysicalDevices", (void *)&resp->ch.vkEnumeratePhysicalDevices},
      {"vkGetPhysicalDeviceMemoryProperties2", (void *)&resp->ch.vkGetPhysicalDeviceMemoryProperties2},
      {"vkDestroyInstance", (void *)&resp->ch.vkDestroyInstance},
      {NULL, NULL},
  };

  resp->ch.vk_handle = LOAD_LIBRARY(vk_lib_path, RTLD_LAZY);
  if (!resp->ch.vk_handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", vk_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for Vulkan GPUs: %s",
            vk_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; l[i].s != NULL; i++) {
    *l[i].p = LOAD_SYMBOL(resp->ch.vk_handle, l[i].s);
    if (!*l[i].p) {
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ch.vk_handle);
      resp->ch.vk_handle = NULL;
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
              msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
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

  VkResult result = (*resp->ch.vkCreateInstance)(&createInfo, NULL, &instance);
  if (result != VK_SUCCESS) {
    resp->err = strdup("failed to create instance");
    return;
  }

  uint32_t deviceCount;
  result = (*resp->ch.vkEnumeratePhysicalDevices)(instance, &deviceCount, NULL);
  if (result != VK_SUCCESS) {
    resp->err = strdup("failed to enumerate physical devices");
    return;
  }

  resp->err = NULL;
  resp->ch.vk = instance;
  resp->ch.num_devices = deviceCount;
  resp->num_devices = deviceCount;
}

int vk_check_flash_attention(vk_handle_t rh, int i) {
  VkInstance instance = rh.vk;
  uint32_t deviceCount = rh.num_devices;

  VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
  if (devices == NULL) {
    return 0;
  }

  VkResult result = (*rh.vkEnumeratePhysicalDevices)(instance, &deviceCount, devices);
  if (result != VK_SUCCESS) {
    free(devices);
    return 0;
  }

  VkPhysicalDeviceProperties properties;
  (*rh.vkGetPhysicalDeviceProperties)(devices[i], &properties);

  int supports_nv_coopmat2 = is_extension_supported(&rh, devices[i], VK_NV_COOPERATIVE_MATRIX_2_EXTENSION_NAME);
  if (!supports_nv_coopmat2) {
    free(devices);
    return 1;
  }

  free(devices);
  return 0;
}

void vk_check_vram(vk_handle_t rh, int i, mem_info_t *resp) {
  VkInstance instance = rh.vk;
  uint32_t deviceCount = rh.num_devices;

  VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
  if (devices == NULL) {
    resp->err = strdup("memory allocation failed for devices array");
    return;
  }

  VkResult result = (*rh.vkEnumeratePhysicalDevices)(instance, &deviceCount, devices);
  if (result != VK_SUCCESS) {
    free(devices);
    resp->err = strdup("failed to enumerate physical devices");
    return;
  }

  VkPhysicalDeviceProperties properties;
  (*rh.vkGetPhysicalDeviceProperties)(devices[i], &properties);

  int supports_budget = is_extension_supported(&rh, devices[i], VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
  if (!supports_budget) {
    free(devices);
    resp->err = strdup("device does not support memory budget");
    return;
  }

  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    free(devices);
    resp->err = strdup("device is a CPU");
    return;
  }

  VkPhysicalDeviceProperties2 device_props2 = {};
  device_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

  VkPhysicalDeviceIDProperties id_props = {};
  id_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

  device_props2.pNext = &id_props;
  (*rh.vkGetPhysicalDeviceProperties2)(devices[i], &device_props2);

  VkPhysicalDeviceMemoryBudgetPropertiesEXT physical_device_memory_budget_properties;
  physical_device_memory_budget_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  physical_device_memory_budget_properties.pNext = NULL;

  VkPhysicalDeviceMemoryProperties2 device_memory_properties;
  device_memory_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
  device_memory_properties.pNext = &physical_device_memory_budget_properties;

  (*rh.vkGetPhysicalDeviceMemoryProperties2)(devices[i], &device_memory_properties);

  VkDeviceSize device_memory_total_size  = 0;
  VkDeviceSize device_memory_heap_budget = 0;

  for (uint32_t j = 0; j < device_memory_properties.memoryProperties.memoryHeapCount; j++) {
    VkMemoryHeap heap = device_memory_properties.memoryProperties.memoryHeaps[j];
    if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      device_memory_total_size  += heap.size;
      device_memory_heap_budget += physical_device_memory_budget_properties.heapBudget[j];
    }
  }

  free(devices);

  resp->err = NULL;
  snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
  resp->gpu_name[GPU_NAME_LEN - 1] = '\0';
  strncpy(&resp->gpu_name[0], properties.deviceName, GPU_NAME_LEN - 1);
  resp->gpu_name[GPU_NAME_LEN - 1] = '\0';
  const uint8_t *uuid = id_props.deviceUUID;
  snprintf(&resp->gpu_id[0], GPU_ID_LEN,
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      uuid[0], uuid[1], uuid[2], uuid[3],
      uuid[4], uuid[5],
      uuid[6], uuid[7],
      uuid[8], uuid[9],
      uuid[10], uuid[11], uuid[12], uuid[13], uuid[14], uuid[15]
    );
  resp->total = (uint64_t) device_memory_total_size;
  resp->free = (uint64_t) device_memory_heap_budget;
  resp->major = VK_API_VERSION_MAJOR(properties.apiVersion);
  resp->minor = VK_API_VERSION_MINOR(properties.apiVersion);
  resp->patch = VK_API_VERSION_PATCH(properties.apiVersion);
}

void vk_release(vk_handle_t rh) {
  LOG(rh.verbose, "releasing vulkan library\n");
  (*rh.vkDestroyInstance)(rh.vk, NULL);
  UNLOAD_LIBRARY(rh.vk_handle);
  rh.vk_handle = NULL;
}

#endif  // __APPLE__

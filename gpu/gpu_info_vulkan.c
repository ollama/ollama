#include "gpu_info_vulkan.h"

#include <string.h>

int check_perfmon(vk_handle_t* rh) {
#ifdef __linux__
  cap_t caps;
  const cap_value_t cap_list[1] = {CAP_PERFMON};

  caps = (*rh->cap_get_proc)();
  if (caps == NULL)
    return -1;

  if ((*rh->cap_set_flag)(caps, CAP_EFFECTIVE, 1, cap_list, CAP_SET) == -1)
    return -1;

  if ((*rh->cap_set_proc)(caps) == -1)
    return -1;

  if ((*rh->cap_free)(caps) == -1)
    return -1;
#endif

  return 0;
}

int support_memory_budget(vk_handle_t* rh, VkPhysicalDevice device) {
  VkPhysicalDeviceProperties properties;
  (*rh->vkGetPhysicalDeviceProperties)(device, &properties);
  uint32_t extensionCount;
  (*rh->vkEnumerateDeviceExtensionProperties)(device, NULL, &extensionCount, NULL);
  VkExtensionProperties* extensions = malloc(extensionCount * sizeof(VkExtensionProperties));
  (*rh->vkEnumerateDeviceExtensionProperties)(device, NULL, &extensionCount, extensions);
  for (int j = 0; j < extensionCount; j++) {
    if (strcmp(extensions[j].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
      return 1;
    }
  }
  return 0;
}

void vk_init(char* vk_lib_path, char* cap_lib_path, vk_init_resp_t *resp) {
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    int is_cap;
    char *s;
    void **p;
  } l[] = {
#ifdef __linux__
      {1, "cap_get_proc", (void *)&resp->ch.cap_get_proc},
      {1, "cap_get_bound", (void *)&resp->ch.cap_get_bound},
      {1, "cap_set_flag", (void *)&resp->ch.cap_set_flag},
      {1, "cap_set_proc", (void *)&resp->ch.cap_set_proc},
      {1, "cap_free", (void *)&resp->ch.cap_free},
#endif
      {0, "vkGetPhysicalDeviceProperties", (void *)&resp->ch.vkGetPhysicalDeviceProperties},
      {0, "vkEnumerateDeviceExtensionProperties", (void *)&resp->ch.vkEnumerateDeviceExtensionProperties},
      {0, "vkCreateInstance", (void *)&resp->ch.vkCreateInstance},
      {0, "vkEnumeratePhysicalDevices", (void *)&resp->ch.vkEnumeratePhysicalDevices},
      {0, "vkGetPhysicalDeviceMemoryProperties2", (void *)&resp->ch.vkGetPhysicalDeviceMemoryProperties2},
      {0, "vkDestroyInstance", (void *)&resp->ch.vkDestroyInstance},
      {0, NULL, NULL},
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

#ifdef __linux__
  resp->ch.cap_handle = LOAD_LIBRARY(cap_lib_path, RTLD_LAZY);
  if (!resp->ch.cap_handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", cap_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for Vulkan GPUs: %s",
            cap_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }
#endif

  for (i = 0; l[i].s != NULL; i++) {
    if (l[i].is_cap)
#ifdef __linux__
      *l[i].p = LOAD_SYMBOL(resp->ch.cap_handle, l[i].s);
#else
      continue;
#endif
    else
      *l[i].p = LOAD_SYMBOL(resp->ch.vk_handle, l[i].s);
    if (!*l[i].p) {
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      if (l[i].is_cap) {
        UNLOAD_LIBRARY(resp->ch.cap_handle);
        resp->ch.cap_handle = NULL;
      } else {
        UNLOAD_LIBRARY(resp->ch.vk_handle);
        resp->ch.vk_handle = NULL;
      }
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
              msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  if (check_perfmon(&resp->ch) != 0) {
    resp->err = strdup("performance monitoring is not allowed. Please enable CAP_PERFMON or run as root to use Vulkan.");
    LOG(resp->ch.verbose, "vulkan: %s", resp->err);
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

void vk_check_vram(vk_handle_t rh, int i, mem_info_t *resp) {
  VkInstance instance = rh.vk;
  uint32_t deviceCount = rh.num_devices;

  VkPhysicalDevice* devices = malloc(deviceCount * sizeof(VkPhysicalDevice));
  VkResult result = (*rh.vkEnumeratePhysicalDevices)(instance, &deviceCount, devices);
  if (result != VK_SUCCESS) {
    resp->err = strdup("failed to enumerate physical devices");
    return;
  }

  VkPhysicalDeviceProperties properties;
  (*rh.vkGetPhysicalDeviceProperties)(devices[i], &properties);
  int supports_budget = support_memory_budget(&rh, devices[i]);
  if (!supports_budget) {
    resp->err = strdup("device does not support memory budget");
    return;
  }
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
    resp->err = strdup("device is a CPU");
    return;
  }

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

  resp->err = NULL;
  snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
  resp->gpu_name[GPU_NAME_LEN - 1] = '\0';
  strncpy(&resp->gpu_name[0], properties.deviceName, GPU_NAME_LEN - 1);
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
#ifdef __linux__
  LOG(rh.verbose, "releasing libcap library\n");
  UNLOAD_LIBRARY(rh.cap_handle);
  rh.cap_handle = NULL;
#endif
}

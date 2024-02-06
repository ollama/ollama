#ifndef __APPLE__  // TODO - maybe consider nvidia support on intel macs?

#include <string.h>
#include "gpu_info_cuda.h"

static deviceMap_t *deviceMap = 0;

void cuda_init(char *cuda_lib_path, cuda_init_resp_t *resp) {
  nvmlReturn_t ret;
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"nvmlInit_v2", (void *)&resp->ch.nvmlInit_v2},
      {"nvmlShutdown", (void *)&resp->ch.nvmlShutdown},
      {"nvmlDeviceGetHandleByIndex", (void *)&resp->ch.nvmlDeviceGetHandleByIndex},
      {"nvmlDeviceGetMemoryInfo", (void *)&resp->ch.nvmlDeviceGetMemoryInfo},
      {"nvmlDeviceGetCount_v2", (void *)&resp->ch.nvmlDeviceGetCount_v2},
      {"nvmlDeviceGetCudaComputeCapability", (void *)&resp->ch.nvmlDeviceGetCudaComputeCapability},
      {"nvmlSystemGetDriverVersion", (void *)&resp->ch.nvmlSystemGetDriverVersion},
      {"nvmlDeviceGetName", (void *)&resp->ch.nvmlDeviceGetName},
      {"nvmlDeviceGetSerial", (void *)&resp->ch.nvmlDeviceGetSerial},
      {"nvmlDeviceGetVbiosVersion", (void *)&resp->ch.nvmlDeviceGetVbiosVersion},
      {"nvmlDeviceGetBoardPartNumber", (void *)&resp->ch.nvmlDeviceGetBoardPartNumber},
      {"nvmlDeviceGetBrand", (void *)&resp->ch.nvmlDeviceGetBrand},
      {"nvmlDeviceGetMigMode", (void *)&resp->ch.nvmlDeviceGetMigMode},
      {"nvmlDeviceGetMigDeviceHandleByIndex", (void *)&resp->ch.nvmlDeviceGetMigDeviceHandleByIndex},
      {"nvmlDeviceGetDeviceHandleFromMigDeviceHandle", (void *)&resp->ch.nvmlDeviceGetDeviceHandleFromMigDeviceHandle},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(cuda_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", cuda_lib_path, msg);
    snprintf(buf, buflen,
             "Unable to load %s library to query for Nvidia GPUs: %s",
             cuda_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->ch.verbose, "wiring nvidia management library functions in %s\n", cuda_lib_path);
  
  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->ch.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!l[i].p) {
      resp->ch.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ch.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
               msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->ch.nvmlInit_v2)();
  if (ret != NVML_SUCCESS) {
    LOG(resp->ch.verbose, "nvmlInit_v2 err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "nvml vram init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->ch.nvmlSystemGetDriverVersion)(buf, buflen);
  if (ret != NVML_SUCCESS) {
    LOG(resp->ch.verbose, "nvmlSystemGetDriverVersion failed: %d\n", ret);
  } else {
    LOG(resp->ch.verbose, "CUDA driver version: %s\n", buf);
  }
}

/**
 * Allocates and initializes a new device map for tracking NVML devices and
 * their MIG instances. This map is structured to hold references to both
 * primary CUDA devices and up to 8 MIG device handles per primary device,
 * reflecting the maximum possible configuration.
 * 
 * @param size The number of primary CUDA devices to allocate space for in the
 *             device map. This does not directly correlate to the number of
 *             MIG instances, which are dynamically discovered.
 * 
 * @return A pointer to the initialized deviceMap_t structure, or NULL if the
 *         memory allocation fails. Each primary device slot is pre-allocated
 *         space for up to 8 MIG device handles plus a terminator value.
 */
static deviceMap_t *new_device_map(unsigned size) {
  deviceMap_t *deviceMap = (deviceMap_t*)malloc(sizeof(deviceMap_t));

  deviceMap->layout = (nvmlDevice_t**)malloc(size*sizeof(nvmlDevice_t*));
  for(int i = 0;i < size;i++) {
    deviceMap->layout[i] = (nvmlDevice_t*)malloc(9*sizeof(nvmlDevice_t));
  }
  deviceMap->numDevices = size;
  return deviceMap;
}

/**
 * Frees the memory allocated for a deviceMap_t structure. It iterates through 
 * each device layout array within the device map, freeing each array of 
 * nvmlDevice_t handles before finally freeing the layout pointer itself and 
 * the deviceMap structure.
 * 
 * @param deviceMap Pointer to the deviceMap_t structure to be freed. If the 
 *        pointer is NULL, the function does nothing.
 */
static void free_device_map(deviceMap_t *deviceMap) {
  if(!deviceMap) return;
  for(int i = 0;i < deviceMap->numDevices;i++) {
    if(deviceMap->layout[i]) free(deviceMap->layout[i]);
  }
  if(deviceMap->layout) free(deviceMap->layout);
}

/**
 * Retrieves a map of devices available to the CUDA handle, including MIG 
 * device handles if MIG mode is enabled. It iterates over all devices 
 * reported by NVML, checks for MIG mode, and attempts to populate a 
 * device map with both primary and MIG device handles.
 * 
 * @param h The CUDA handle containing NVML function pointers.
 * @param err Pointer to a char* for returning an error message on failure.
 * 
 * @return A pointer to a deviceMap_t structure with the device mapping, 
 *         or NULL on failure with an error message set in err.
 */
static deviceMap_t *get_device_map(cuda_handle_t h,char **err) {
  unsigned numDevices = 0;
  deviceMap_t *deviceMap = 0;
  nvmlDevice_t device;
  nvmlReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];

  ret = (*h.nvmlDeviceGetCount_v2)(&numDevices);
  if (ret != NVML_SUCCESS) {
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    *err = strdup(buf);
    return 0;
  }

  deviceMap = new_device_map(numDevices);

  for(int i = 0;i < numDevices;i++) {
    unsigned int currentMode = 0,pendingMode;

    ret = (*h.nvmlDeviceGetHandleByIndex)(i, &device);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "unable to get device handle %d: %d", i, ret);
      *err = strdup(buf);
      free_device_map(deviceMap);
      return 0;
    }

    ret = (*h.nvmlDeviceGetMigMode)(device,&currentMode,&pendingMode);
    if (ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_SUPPORTED) {
      snprintf(buf, buflen, "unable to get MIG mode for device %d: %d", i, ret);
      *err = strdup(buf);
      free_device_map(deviceMap);
      return 0;
    }
    else if(h.verbose && ret != NVML_ERROR_NOT_SUPPORTED) LOG(h.verbose,"MIG Mode is %d\n", currentMode);

    deviceMap->layout[i][0] = device;
    deviceMap->layout[i][1] = 0;

    if (currentMode == 0x1) {
      // Get the MIG device handle for a specific MIG device index
      for(int j = 0;j < 8;j++) {
        nvmlDevice_t migDevice;
        ret = (*h.nvmlDeviceGetMigDeviceHandleByIndex)(device, j, &migDevice);
        if (ret != NVML_SUCCESS && ret != NVML_ERROR_NOT_FOUND) {
          snprintf(buf, buflen, "failed to get MIG device %d handle for device %d: %d", j, i, ret);
          *err = strdup(buf);
          free_device_map(deviceMap);
          return 0;
        }
        if(ret == NVML_SUCCESS) {
          if(h.verbose) LOG(h.verbose,"MIG Device Intance %d:%d found\n", i,j);
          deviceMap->layout[i][j + 1] = migDevice;
          deviceMap->layout[i][j + 2] = 0;
        }
        else break;
      }
    }
  }
  return deviceMap;
}

/**
 * Retrieves the VRAM information for a specified CUDA device, optionally 
 * querying a MIG device if provided. It fills in the provided mem_info_t 
 * struct with the total and free memory. In verbose mode, it also logs 
 * additional device information such as device name, part number, serial 
 * number, VBIOS version, and brand type without affecting the function's 
 * primary operation.
 * 
 * @param h CUDA handle containing NVML function pointers.
 * @param device The NVML device handle of the primary CUDA device.
 * @param migp Optional pointer to a MIG device handle to query instead of 
 *        the primary device.
 * @param deviceId The index of the CUDA device being queried.
 * @param resp Pointer to mem_info_t struct where memory information is stored.
 * 
 * @return 1 on success, 0 on failure with an error message set in resp->err.
 */
int get_device_vram(cuda_handle_t h, nvmlDevice_t device,nvmlDevice_t *migp,unsigned deviceId,mem_info_t *resp) {
  nvmlDevice_t queryDevice = device;
  nvmlMemory_t memInfo = {0};
  const int buflen = 256;
  char buf[buflen + 1];
  nvmlReturn_t ret;

  if(migp) queryDevice = *migp;

  ret = (*h.nvmlDeviceGetMemoryInfo)(queryDevice, &memInfo);
  if (ret != NVML_SUCCESS) {
    snprintf(buf, buflen, "device memory info lookup failure %d: %d", deviceId, ret);
    resp->err = strdup(buf);
    return 0;
  }
    
  if (h.verbose) {
    nvmlBrandType_t brand = 0;
    // When in verbose mode, report more information about
    // the card we discover, but don't fail on error
    ret = (*h.nvmlDeviceGetName)(queryDevice, buf, buflen);
    if (ret != RSMI_STATUS_SUCCESS) {
      LOG(h.verbose, "nvmlDeviceGetName failed: %d\n", ret);
    } else {
      LOG(h.verbose, "[%d] CUDA device name: %s\n", deviceId, buf);
    }
    ret = (*h.nvmlDeviceGetBoardPartNumber)(device, buf, buflen);
    if (ret != RSMI_STATUS_SUCCESS) {
      LOG(h.verbose, "nvmlDeviceGetBoardPartNumber failed: %d\n", ret);
    } else {
      LOG(h.verbose, "[%d] CUDA part number: %s\n", deviceId, buf);
    }
    ret = (*h.nvmlDeviceGetSerial)(device, buf, buflen);
    if (ret != RSMI_STATUS_SUCCESS) {
      LOG(h.verbose, "nvmlDeviceGetSerial failed: %d\n", ret);
    } else {
      LOG(h.verbose, "[%d] CUDA S/N: %s\n", deviceId, buf);
    }
    ret = (*h.nvmlDeviceGetVbiosVersion)(device, buf, buflen);
    if (ret != RSMI_STATUS_SUCCESS) {
      LOG(h.verbose, "nvmlDeviceGetVbiosVersion failed: %d\n", ret);
    } else {
      LOG(h.verbose, "[%d] CUDA vbios version: %s\n", deviceId, buf);
    }
    ret = (*h.nvmlDeviceGetBrand)(device, &brand);
    if (ret != RSMI_STATUS_SUCCESS) {
      LOG(h.verbose, "nvmlDeviceGetBrand failed: %d\n", ret);
    } else {
      LOG(h.verbose, "[%d] CUDA brand: %d\n", deviceId, brand);
    }
  }

  LOG(h.verbose, "[%d] CUDA totalMem %ld\n", deviceId, memInfo.total);
  LOG(h.verbose, "[%d] CUDA freeMem %ld\n", deviceId, memInfo.free);

  resp->total += memInfo.total;
  resp->free += memInfo.free;
  return 1;
}

/**
 * Checks and aggregates VRAM information for all CUDA devices and their MIG 
 * instances, if available. It initializes the device map if not already done,
 * iterates over all devices, querying VRAM for both primary and MIG devices,
 * and aggregates total and free VRAM. If the NVML handle is not initialized or
 * any VRAM query fails, it sets an error message in the response structure.
 * 
 * @param h CUDA handle with NVML function pointers and state.
 * @param resp Pointer to mem_info_t structure to populate with VRAM info and 
 *        potential errors.
 */
void cuda_check_vram(cuda_handle_t h, mem_info_t *resp) {
  resp->err = NULL;
  unsigned deviceId = 0;

  resp->total = 0;
  resp->free = 0;
  resp->count = 0;
  if (h.handle == NULL) {
    resp->err = strdup("nvml handle isn't initialized");
    return;
  }

  // If deviceMap is not initialized, do so
  if(!deviceMap) {
    if(!(deviceMap = get_device_map(h,&resp->err))) return;
  }

  for (int i = 0; i < deviceMap->numDevices; i++) {
    if(!deviceMap->layout[i][1]) {
      if(!get_device_vram(h,deviceMap->layout[i][0],0,deviceId,resp)) {
        return;
      }
      deviceId++;
    }
    else {
      int migIndex = 1;

      while(deviceMap->layout[i][migIndex]) {
        if(!get_device_vram(h,deviceMap->layout[i][0],deviceMap->layout[i] + migIndex++, deviceId,resp)) {
          return;
        }
        deviceId++;
      }
    }
  }
  resp->count = deviceId;
}

/**
 * Queries and reports the lowest CUDA compute capability among all CUDA devices
 * managed by the given NVML handle. This is important for determining the 
 * compatibility of CUDA applications with the available hardware. The function
 * initializes the device map if necessary and iterates through each device,
 * querying its CUDA compute capability. If the NVML handle is not initialized
 * or any device query fails, it sets an error message in the response structure.
 * 
 * @param h CUDA handle with NVML function pointers and state.
 * @param resp Pointer to cuda_compute_capability_t structure to populate with 
 *        the lowest compute capability and potential errors.
 */
void cuda_compute_capability(cuda_handle_t h, cuda_compute_capability_t *resp) {
  int major = 0;
  int minor = 0;
  nvmlReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];

  resp->err = NULL;
  resp->major = 0;
  resp->minor = 0;
  if (h.handle == NULL) {
    resp->err = strdup("nvml handle not initialized");
    return;
  }

  // If deviceMap is not initialized, do so
  if(!deviceMap) {
    if(!(deviceMap = get_device_map(h,&resp->err))) return;
  }

  for (int i = 0; i < deviceMap->numDevices; i++) {
    ret = (*h.nvmlDeviceGetCudaComputeCapability)(deviceMap->layout[i][0], &major, &minor);
    if (ret != NVML_SUCCESS) {
      snprintf(buf, buflen, "device compute capability lookup failure %d: %d", i, ret);
      resp->err = strdup(buf);
      return;
    }
    // Report the lowest major.minor we detect as that limits our compatibility
    if (resp->major == 0 || resp->major > major ) {
      resp->major = major;
      resp->minor = minor;
    } else if (resp->major == major && resp->minor > minor) {
      resp->minor = minor;
    }
  }
}
#endif  // __APPLE__
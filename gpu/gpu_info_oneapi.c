#ifndef __APPLE__

#include "gpu_info_oneapi.h"

#include <string.h>

void oneapi_init(char *oneapi_lib_path, oneapi_init_resp_t *resp) {
  ze_result_t ret;
  resp->err = NULL;
  resp->oh.devices = NULL;
  resp->oh.num_devices = NULL;
  resp->oh.drivers = NULL;
  resp->oh.num_drivers = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i, d;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"zesInit", (void *)&resp->oh.zesInit},
      {"zesDriverGet", (void *)&resp->oh.zesDriverGet},
      {"zesDeviceGet", (void *)&resp->oh.zesDeviceGet},
      {"zesDeviceGetProperties", (void *)&resp->oh.zesDeviceGetProperties},
      {"zesDeviceEnumMemoryModules",
       (void *)&resp->oh.zesDeviceEnumMemoryModules},
      {"zesMemoryGetProperties", (void *)&resp->oh.zesMemoryGetProperties},
      {"zesMemoryGetState", (void *)&resp->oh.zesMemoryGetState},
      {NULL, NULL},
  };

  resp->oh.handle = LOAD_LIBRARY(oneapi_lib_path, RTLD_LAZY);
  if (!resp->oh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Intel GPUs: %s\n",
             oneapi_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  // TODO once we've squashed the remaining corner cases remove this log
  LOG(resp->oh.verbose,
      "wiring Level-Zero management library functions in %s\n",
      oneapi_lib_path);

  for (i = 0; l[i].s != NULL; i++) {
    // TODO once we've squashed the remaining corner cases remove this log
    LOG(resp->oh.verbose, "dlsym: %s\n", l[i].s);

    *l[i].p = LOAD_SYMBOL(resp->oh.handle, l[i].s);
    if (!*(l[i].p)) {
      resp->oh.handle = NULL;
      char *msg = LOAD_ERR();
      LOG(resp->oh.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->oh.handle);
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s, msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  LOG(resp->oh.verbose, "calling zesInit\n");

  ret = (*resp->oh.zesInit)(0);
  if (ret != ZE_RESULT_SUCCESS) {
    LOG(resp->oh.verbose, "zesInit err: %x\n", ret);
    snprintf(buf, buflen, "oneapi vram init failure: %x", ret);
    resp->err = strdup(buf);
    oneapi_release(resp->oh);
    return;
  }

  LOG(resp->oh.verbose, "calling zesDriverGet\n");
  ret = (*resp->oh.zesDriverGet)(&resp->oh.num_drivers, NULL);
  if (ret != ZE_RESULT_SUCCESS) {
    LOG(resp->oh.verbose, "zesDriverGet err: %x\n", ret);
    snprintf(buf, buflen, "unable to get driver count: %x", ret);
    resp->err = strdup(buf);
    oneapi_release(resp->oh);
    return;
  }
  LOG(resp->oh.verbose, "oneapi driver count: %d\n", resp->oh.num_drivers);
  resp->oh.drivers = malloc(resp->oh.num_drivers * sizeof(zes_driver_handle_t));
  resp->oh.num_devices = malloc(resp->oh.num_drivers * sizeof(uint32_t));
  memset(&resp->oh.num_devices[0], 0, resp->oh.num_drivers * sizeof(uint32_t));
  resp->oh.devices =
      malloc(resp->oh.num_drivers * sizeof(zes_device_handle_t *));
  ret = (*resp->oh.zesDriverGet)(&resp->oh.num_drivers, &resp->oh.drivers[0]);
  if (ret != ZE_RESULT_SUCCESS) {
    LOG(resp->oh.verbose, "zesDriverGet err: %x\n", ret);
    snprintf(buf, buflen, "unable to get driver count: %x", ret);
    resp->err = strdup(buf);
    oneapi_release(resp->oh);
    return;
  }

  for (d = 0; d < resp->oh.num_drivers; d++) {
    LOG(resp->oh.verbose, "calling zesDeviceGet count %d: %p\n", d, resp->oh.drivers[d]);
    ret = (*resp->oh.zesDeviceGet)(resp->oh.drivers[d],
                                   &resp->oh.num_devices[d], NULL);
    if (ret != ZE_RESULT_SUCCESS) {
      LOG(resp->oh.verbose, "zesDeviceGet err: %x\n", ret);
      snprintf(buf, buflen, "unable to get device count: %x", ret);
      resp->err = strdup(buf);
      oneapi_release(resp->oh);
      return;
    }
    resp->oh.devices[d] =
        malloc(resp->oh.num_devices[d] * sizeof(zes_device_handle_t));
    ret = (*resp->oh.zesDeviceGet)(
        resp->oh.drivers[d], &resp->oh.num_devices[d], resp->oh.devices[d]);
    if (ret != ZE_RESULT_SUCCESS) {
      LOG(resp->oh.verbose, "zesDeviceGet err: %x\n", ret);
      snprintf(buf, buflen, "unable to get device count: %x", ret);
      resp->err = strdup(buf);
      oneapi_release(resp->oh);
      return;
    }
  }

  return;
}

void oneapi_check_vram(oneapi_handle_t h, int driver, int device,
                       mem_info_t *resp) {
  ze_result_t ret;
  resp->err = NULL;
  uint64_t totalMem = 0;
  uint64_t usedMem = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i, d, m;

  if (h.handle == NULL) {
    resp->err = strdup("Level-Zero handle not initialized");
    return;
  }

  if (driver > h.num_drivers || device > h.num_devices[driver]) {
    resp->err = strdup("driver of device index out of bounds");
    return;
  }

  resp->total = 0;
  resp->free = 0;

  zes_device_ext_properties_t ext_props;
  ext_props.stype = ZES_STRUCTURE_TYPE_DEVICE_EXT_PROPERTIES;
  ext_props.pNext = NULL;

  zes_device_properties_t props;
  props.stype = ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  props.pNext = &ext_props;

  ret = (*h.zesDeviceGetProperties)(h.devices[driver][device], &props);
  if (ret != ZE_RESULT_SUCCESS) {
    snprintf(buf, buflen, "unable to get device properties: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  snprintf(&resp->gpu_name[0], GPU_NAME_LEN, "%s", props.modelName);

  // TODO this needs to map to ONEAPI_DEVICE_SELECTOR syntax
  // (this is probably wrong...)
  // TODO - the driver isn't included - what if there are multiple drivers?
  snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", device);

  if (h.verbose) {
    // When in verbose mode, report more information about
    // the card we discover.
    LOG(h.verbose, "[%d:%d] oneAPI device name: %s\n", driver, device,
        props.modelName);
    LOG(h.verbose, "[%d:%d] oneAPI brand: %s\n", driver, device,
        props.brandName);
    LOG(h.verbose, "[%d:%d] oneAPI vendor: %s\n", driver, device,
        props.vendorName);
    LOG(h.verbose, "[%d:%d] oneAPI S/N: %s\n", driver, device,
        props.serialNumber);
    LOG(h.verbose, "[%d:%d] oneAPI board number: %s\n", driver, device,
        props.boardNumber);
  }

  // TODO
  // Compute Capability equivalent in resp->major, resp->minor, resp->patch

  uint32_t memCount = 0;
  ret = (*h.zesDeviceEnumMemoryModules)(h.devices[driver][device], &memCount,
                                        NULL);
  if (ret != ZE_RESULT_SUCCESS) {
    snprintf(buf, buflen, "unable to enumerate Level-Zero memory modules: %x",
             ret);
    resp->err = strdup(buf);
    return;
  }

  LOG(h.verbose, "discovered %d Level-Zero memory modules\n", memCount);

  zes_mem_handle_t *mems = malloc(memCount * sizeof(zes_mem_handle_t));
  (*h.zesDeviceEnumMemoryModules)(h.devices[driver][device], &memCount, mems);

  for (m = 0; m < memCount; m++) {
    zes_mem_state_t state;
    state.stype = ZES_STRUCTURE_TYPE_MEM_STATE;
    state.pNext = NULL;
    ret = (*h.zesMemoryGetState)(mems[m], &state);
    if (ret != ZE_RESULT_SUCCESS) {
      snprintf(buf, buflen, "unable to get memory state: %x", ret);
      resp->err = strdup(buf);
      free(mems);
      return;
    }

    resp->total += state.size;
    resp->free += state.free;
  }

  free(mems);
}

void oneapi_release(oneapi_handle_t h) {
  int d;
  LOG(h.verbose, "releasing oneapi library\n");
  for (d = 0; d < h.num_drivers; d++) {
    if (h.devices != NULL && h.devices[d] != NULL) {
      free(h.devices[d]);
    }
  }
  if (h.devices != NULL) {
    free(h.devices);
    h.devices = NULL;
  }
  if (h.num_devices != NULL) {
    free(h.num_devices);
    h.num_devices = NULL;
  }
  if (h.drivers != NULL) {
    free(h.drivers);
    h.drivers = NULL;
  }
  h.num_drivers = 0;
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

int oneapi_get_device_count(oneapi_handle_t h, int driver) {
  if (h.handle == NULL || h.num_devices == NULL) {
    return 0;
  }
  if (driver > h.num_drivers) {
    return 0;
  }
  return (int)h.num_devices[driver];
}

#endif // __APPLE__

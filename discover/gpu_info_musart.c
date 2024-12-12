#ifndef __APPLE__

#include <string.h>
#include "gpu_info_musart.h"

void musart_init(char *musart_lib_path, musart_init_resp_t *resp) {
  musartReturn_t ret;
  resp->err = NULL;
  resp->num_devices = 0;
  const int buflen = 256;
  char buf[buflen + 1];
  int i;

  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"musaSetDevice", (void *)&resp->ch.musaSetDevice},
      {"musaDeviceSynchronize", (void *)&resp->ch.musaDeviceSynchronize},
      {"musaDeviceReset", (void *)&resp->ch.musaDeviceReset},
      {"musaMemGetInfo", (void *)&resp->ch.musaMemGetInfo},
      {"musaGetDeviceCount", (void *)&resp->ch.musaGetDeviceCount},
      {"musaDeviceGetAttribute", (void *)&resp->ch.musaDeviceGetAttribute},
      {"musaDriverGetVersion", (void *)&resp->ch.musaDriverGetVersion},
      {"musaGetDeviceProperties", (void *)&resp->ch.musaGetDeviceProperties},
      {NULL, NULL},
  };

  resp->ch.handle = LOAD_LIBRARY(musart_lib_path, RTLD_LAZY);
  if (!resp->ch.handle) {
    char *msg = LOAD_ERR();
    LOG(resp->ch.verbose, "library %s load err: %s\n", musart_lib_path, msg);
    snprintf(buf, buflen,
            "Unable to load %s library to query for Moore Threads GPUs: %s",
            musart_lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }

  for (i = 0; l[i].s != NULL; i++) {
    *l[i].p = LOAD_SYMBOL(resp->ch.handle, l[i].s);
    if (!*(l[i].p)) {
      char *msg = LOAD_ERR();
      LOG(resp->ch.verbose, "dlerr: %s\n", msg);
      UNLOAD_LIBRARY(resp->ch.handle);
      resp->ch.handle = NULL;
      snprintf(buf, buflen, "symbol lookup for %s failed: %s", l[i].s,
              msg);
      free(msg);
      resp->err = strdup(buf);
      return;
    }
  }

  ret = (*resp->ch.musaSetDevice)(0);
  if (ret != MUSART_SUCCESS) {
    LOG(resp->ch.verbose, "musaSetDevice err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    if (ret == MUSART_ERROR_INSUFFICIENT_DRIVER) {
      resp->err = strdup("your MUSA driver is too old or missing.  If you have a Moore Threads GPU please upgrade to run ollama");
      return;
    }
    snprintf(buf, buflen, "musart init failure: %d", ret);
    resp->err = strdup(buf);
    return;
  }

  int version = 0;
  musartDriverVersion_t driverVersion;
  driverVersion.major = 0;
  driverVersion.minor = 0;

  // Report driver version if we're in verbose mode, ignore errors
  ret = (*resp->ch.musaDriverGetVersion)(&version);
  if (ret != MUSART_SUCCESS) {
    LOG(resp->ch.verbose, "musaDriverGetVersion failed: %d\n", ret);
  } else {
    driverVersion.major = version / 1000;
    driverVersion.minor = (version - (driverVersion.major * 1000)) / 10;
    LOG(resp->ch.verbose, "MUSA driver version: %d-%d\n", driverVersion.major, driverVersion.minor);
  }

  ret = (*resp->ch.musaGetDeviceCount)(&resp->num_devices);
  if (ret != MUSART_SUCCESS) {
    LOG(resp->ch.verbose, "musaGetDeviceCount err: %d\n", ret);
    UNLOAD_LIBRARY(resp->ch.handle);
    resp->ch.handle = NULL;
    snprintf(buf, buflen, "unable to get device count: %d", ret);
    resp->err = strdup(buf);
    return;
  }
}


void musart_bootstrap(musart_handle_t h, int i, mem_info_t *resp) {
  resp->err = NULL;
  musartMemory_t memInfo = {0,0,0};
  musartReturn_t ret;
  const int buflen = 256;
  char buf[buflen + 1];

  if (h.handle == NULL) {
    resp->err = strdup("musart handle isn't initialized");
    return;
  }

  ret = (*h.musaSetDevice)(i);
  if (ret != MUSART_SUCCESS) {
    snprintf(buf, buflen, "musart device failed to initialize");
    resp->err = strdup(buf);
    return;
  }

  musaDeviceProp_t props;
  ret = (*h.musaGetDeviceProperties)(&props, i);
  if (ret != MUSART_SUCCESS) {
    LOG(h.verbose, "[%d] device properties lookup failure: %d\n", i, ret);
    snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
    resp->major = 0;
    resp->minor = 0;
  } else {
    int allNull = 1;
    for (int j = 0; j < 16; j++) {
      if (props.uuid.bytes[j] != 0) {
        allNull = 0;
        break;
      }
    }
    if (allNull != 0) {
      snprintf(&resp->gpu_id[0], GPU_ID_LEN, "%d", i);
    } else {
      // GPU-d110a105-ac29-1d54-7b49-9c90440f215b
      snprintf(&resp->gpu_id[0], GPU_ID_LEN,
          "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
          props.uuid.bytes[0],
          props.uuid.bytes[1],
          props.uuid.bytes[2],
          props.uuid.bytes[3],
          props.uuid.bytes[4],
          props.uuid.bytes[5],
          props.uuid.bytes[6],
          props.uuid.bytes[7],
          props.uuid.bytes[8],
          props.uuid.bytes[9],
          props.uuid.bytes[10],
          props.uuid.bytes[11],
          props.uuid.bytes[12],
          props.uuid.bytes[13],
          props.uuid.bytes[14],
          props.uuid.bytes[15]
        );
    }
    strncpy(&resp->gpu_name[0], props.name, GPU_NAME_LEN - 1);
    resp->gpu_name[GPU_NAME_LEN - 1] = '\0';
    resp->major = props.major;
    resp->minor = props.minor;

    // TODO add other useful properties from props
  }
  ret = (*h.musaMemGetInfo)(&memInfo.free, &memInfo.total);
  if (ret != MUSART_SUCCESS) {
    snprintf(buf, buflen, "musart device memory info lookup failure %d", ret);
    resp->err = strdup(buf);
    return;
  }

  resp->total = memInfo.total;
  resp->free = memInfo.free;
  resp->used = memInfo.used;

  LOG(h.verbose, "[%s] MUSA totalMem %lu\n", resp->gpu_id, resp->total);
  LOG(h.verbose, "[%s] MUSA freeMem %lu\n", resp->gpu_id, resp->free);
  LOG(h.verbose, "[%s] MUSA usedMem %lu\n", resp->gpu_id, resp->used);
  LOG(h.verbose, "[%s] Compute Capability %d.%d\n", resp->gpu_id, resp->major, resp->minor);
}

void musart_release(musart_handle_t h) {
  LOG(h.verbose, "releasing musart library\n");
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

#endif  // __APPLE__
#ifndef __APPLE__

#include "gpu_info_sycl.h"
#include "ggml-sycl.h"

#include <string.h>

typedef int (*GetDeviceCount)();

void sycl_init(char *lib_path, sycl_init_resp_t *resp) {
  resp->err = NULL;
  const int buflen = 256;
  char buf[buflen + 1];
  int i, d;
  struct lookup {
    char *s;
    void **p;
  } l[] = {
      {"ggml_backend_sycl_get_gpu_list", (void *)&resp->oh.ggml_backend_sycl_get_gpu_list},
      {"ggml_backend_sycl_print_sycl_devices", (void *)&resp->oh.ggml_backend_sycl_print_sycl_devices},
      {"ggml_backend_sycl_get_device_count", (void *)&resp->oh.ggml_backend_sycl_get_device_count},
      {"ggml_backend_sycl_get_device_memory", (void *)&resp->oh.ggml_backend_sycl_get_device_memory},
      {"ggml_backend_sycl_get_device_description", (void *)&resp->oh.ggml_backend_sycl_get_device_description},
      {NULL, NULL},
  };

  resp->oh.handle = (void *)LOAD_LIBRARY(lib_path, RTLD_LAZY);
  if (!resp->oh.handle) {
    char *msg = LOAD_ERR();
    snprintf(buf, buflen,
             "Unable to load %s library to query for Intel GPUs: %s\n", lib_path, msg);
    free(msg);
    resp->err = strdup(buf);
    return;
  }
  LOG(resp->oh.verbose, "wiring sycl management library functions in %s\n", lib_path);

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
  return;
}

void sycl_release(sycl_handle_t h) {
  LOG(h.verbose, "releasing sycl library\n");
  if (h.num_devices != NULL) {
    free(h.num_devices);
    h.num_devices = NULL;
  }
  UNLOAD_LIBRARY(h.handle);
  h.handle = NULL;
}

void sycl_get_gpu_list(sycl_handle_t *oh, int *id_list, int max_len) {
    (oh->ggml_backend_sycl_get_gpu_list)(id_list, max_len);
}

void sycl_print_sycl_devices(sycl_handle_t *oh) {
    (oh->ggml_backend_sycl_print_sycl_devices)();
}

int sycl_get_device_count(sycl_handle_t *oh) {
  int ret = (oh->ggml_backend_sycl_get_device_count)();
  return ret;
}

void sycl_check_vram(sycl_handle_t h, int device, mem_info_t *resp) {
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
  resp->total = 0;
  resp->free = 0;

  (h.ggml_backend_sycl_get_device_memory)(device, &resp->free, &resp->total);
  (h.ggml_backend_sycl_get_device_description)(device, buf, buflen);

  snprintf(&resp->gpu_name[0], GPU_NAME_LEN, "%s", buf);
}

#endif // __APPLE__

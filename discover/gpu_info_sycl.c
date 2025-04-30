#ifndef __APPLE__

#include "gpu_info_sycl.h"
#include "ggml-sycl.h"

#include <string.h>

typedef int (*GetDeviceCount)();

void sycl_init(char *lib_path, sycl_init_resp_t *resp) {
  resp->err = NULL;
//  resp->oh.devices = NULL;
//  resp->oh.num_devices = NULL;
//  resp->oh.drivers = NULL;
//  resp->oh.num_drivers = 0;
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
  LOG(resp->oh.verbose, "Sycl lib Init OK\n");
  return;
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

void sycl_get_device_memory(sycl_handle_t *oh, int device, size_t *free, size_t *total) {
  (oh->ggml_backend_sycl_get_device_memory)(device, free, total);
}

#endif // __APPLE__

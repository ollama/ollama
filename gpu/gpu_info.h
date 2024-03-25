#ifndef __APPLE__
#ifndef __GPU_INFO_H__
#define __GPU_INFO_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef _WIN32
#include <dlfcn.h>
#define LOAD_LIBRARY(lib, flags) dlopen(lib, flags)
#define LOAD_SYMBOL(handle, sym) dlsym(handle, sym)
#define LOAD_ERR() strdup(dlerror())
#define UNLOAD_LIBRARY(handle) dlclose(handle)
#else
#include <windows.h>
#define LOAD_LIBRARY(lib, flags) LoadLibrary(lib)
#define LOAD_SYMBOL(handle, sym) GetProcAddress(handle, sym)
#define UNLOAD_LIBRARY(handle) FreeLibrary(handle)
#define LOAD_ERR() ({\
  LPSTR messageBuffer = NULL; \
  size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, \
                                 NULL, GetLastError(), MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL); \
  char *resp = strdup(messageBuffer); \
  LocalFree(messageBuffer); \
  resp; \
})

#endif

#define LOG(verbose, ...) \
  do { \
    if (verbose) { \
      fprintf(stderr, __VA_ARGS__); \
    } \
  } while (0)

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mem_info {
  uint64_t total;
  uint64_t free;
  unsigned int count;
  int igpu_index; // If >= 0, we detected an integrated GPU to ignore
  char *err;  // If non-nill, caller responsible for freeing
} mem_info_t;

void cpu_check_ram(mem_info_t *resp);

#ifdef __cplusplus
}
#endif

#include "gpu_info_cuda.h"

#endif  // __GPU_INFO_H__
#endif  // __APPLE__
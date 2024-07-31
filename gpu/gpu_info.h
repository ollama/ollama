<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
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

#define GPU_ID_LEN 64
#define GPU_NAME_LEN 96

typedef struct mem_info {
  char *err;  // If non-nill, caller responsible for freeing
  char gpu_id[GPU_ID_LEN];
  char gpu_name[GPU_NAME_LEN];
  uint64_t total;
  uint64_t free;
<<<<<<< HEAD
  uint64_t used;
=======
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289

  // Compute Capability
  int major; 
  int minor;
  int patch;
} mem_info_t;

void cpu_check_ram(mem_info_t *resp);

#ifdef __cplusplus
}
#endif

#include "gpu_info_cudart.h"
#include "gpu_info_nvcuda.h"
<<<<<<< HEAD
#include "gpu_info_nvml.h"
#include "gpu_info_oneapi.h"

#endif  // __GPU_INFO_H__
=======
#include "gpu_info_oneapi.h"

#endif  // __GPU_INFO_H__
=======
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

#define GPU_ID_LEN 64
#define GPU_NAME_LEN 96

typedef struct mem_info {
  char *err;  // If non-nill, caller responsible for freeing
  char gpu_id[GPU_ID_LEN];
  char gpu_name[GPU_NAME_LEN];
  uint64_t total;
  uint64_t free;
  uint64_t used;

  // Compute Capability
  int major; 
  int minor;
  int patch;
} mem_info_t;

void cpu_check_ram(mem_info_t *resp);

#ifdef __cplusplus
}
#endif

#include "gpu_info_cudart.h"
#include "gpu_info_nvcuda.h"
#include "gpu_info_nvml.h"
#include "gpu_info_oneapi.h"

#endif  // __GPU_INFO_H__
>>>>>>> 0b01490f7a487eae06890c5eabcd2270e32605a5
>>>>>>> 73e3b128c2a287e5cf76ba2f9fcda3086476a289
#endif  // __APPLE__
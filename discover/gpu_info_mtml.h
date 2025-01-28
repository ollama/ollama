#ifndef __APPLE__
#ifndef __GPU_INFO_MTML_H__
#define __GPU_INFO_MTML_H__
#include "gpu_info.h"

// Just enough typedef's to dlopen/dlsym for memory information
typedef enum
{
  MTML_SUCCESS = 0,
  // Other values omitted for now...
} MtmlReturn;

typedef struct mtmlMemory_st {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
} mtmlMemory_t;

typedef struct MtmlLibrary MtmlLibrary;
typedef struct MtmlSystem MtmlSystem;
typedef struct MtmlDevice MtmlDevice;
typedef struct MtmlMemory MtmlMemory;

typedef struct mtml_handle {
  void *handle;
  uint16_t verbose;
  MtmlLibrary *lib;
  MtmlSystem *sys;
  MtmlReturn (*mtmlLibraryInit)(MtmlLibrary **lib);
  MtmlReturn (*mtmlLibraryInitSystem)(const MtmlLibrary *lib, MtmlSystem **sys);
  MtmlReturn (*mtmlLibraryShutDown)(MtmlLibrary *lib);
  MtmlReturn (*mtmlLibraryInitDeviceByIndex)(const MtmlLibrary *lib, unsigned int index, MtmlDevice **dev);
  MtmlReturn (*mtmlDeviceInitMemory)(const MtmlDevice *dev, MtmlMemory **mem);
  MtmlReturn (*mtmlDeviceFreeMemory)(const MtmlMemory *mem);
  MtmlReturn (*mtmlMemoryGetTotal)(const MtmlMemory *mem, unsigned long long *total);
  MtmlReturn (*mtmlMemoryGetUsed)(const MtmlMemory *mem, unsigned long long *used);
} mtml_handle_t;

typedef struct mtml_init_resp {
  char *err;  // If err is non-null handle is invalid
  mtml_handle_t ch;
} mtml_init_resp_t;

void mtml_init(char *mtml_lib_path, mtml_init_resp_t *resp);
void mtml_get_free(mtml_handle_t ch, int device_id, uint64_t *free, uint64_t *total, uint64_t *used);
void mtml_release(mtml_handle_t ch);

#endif  // __GPU_INFO_MTML_H__
#endif  // __APPLE__
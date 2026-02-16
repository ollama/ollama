#include "dynamic.h"

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
#define DLOPEN(path) LoadLibraryA(path)
#define DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <libgen.h>
#endif
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define DLCLOSE(handle) dlclose(handle)
#endif

static int mlx_dynamic_open(mlx_dynamic_handle* handle, const char* path) {
    handle->ctx = (void*) DLOPEN(path);
    CHECK(handle->ctx != NULL);
    return 0;
}

int mlx_dynamic_load(mlx_dynamic_handle* handle, const char *path) {
    return mlx_dynamic_open(handle, path);
}

void mlx_dynamic_unload(mlx_dynamic_handle* handle) {
    if (handle->ctx) {
        DLCLOSE(handle->ctx);
        handle->ctx = NULL;
    }
}

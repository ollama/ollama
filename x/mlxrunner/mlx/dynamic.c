#include "dynamic.h"

#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
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

#ifdef _WIN32
// On Windows, add the library's directory to the DLL search path so that
// companion DLLs in the same directory (e.g. nvrtc-builtins) are found
// when loaded at runtime by transitive dependencies like NVRTC.
static int mlx_dynamic_open(mlx_dynamic_handle* handle, const char* path) {
    if (path) {
        char dir_path[MAX_PATH];
        strncpy(dir_path, path, MAX_PATH - 1);
        dir_path[MAX_PATH - 1] = '\0';
        char* last_slash = strrchr(dir_path, '\\');
        if (!last_slash) last_slash = strrchr(dir_path, '/');
        if (last_slash) {
            *last_slash = '\0';
            SetDllDirectoryA(dir_path);
        }
    }
    handle->ctx = (void*) LoadLibraryA(path);
    if (handle->ctx == NULL) {
        return 1;
    }
    return 0;
}
#else
static int mlx_dynamic_open(mlx_dynamic_handle* handle, const char* path) {
    handle->ctx = (void*) DLOPEN(path);
    if (handle->ctx == NULL) {
        return 1;
    }
    return 0;
}
#endif

int mlx_dynamic_load(mlx_dynamic_handle* handle, const char *path) {
    return mlx_dynamic_open(handle, path);
}

void mlx_dynamic_unload(mlx_dynamic_handle* handle) {
    if (handle->ctx) {
        DLCLOSE(handle->ctx);
        handle->ctx = NULL;
    }
}

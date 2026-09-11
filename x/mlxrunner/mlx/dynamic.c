#include "dynamic.h"

#include <stdio.h>
#include <string.h>

// Holds the name of the first symbol that failed to resolve during
// mlx_dynamic_load_symbols, for the Go side to report lazily.
static char mlx_load_error_symbol[128];

void mlx_dynamic_record_load_error(const char* symbol) {
    strncpy(mlx_load_error_symbol, symbol, sizeof(mlx_load_error_symbol) - 1);
    mlx_load_error_symbol[sizeof(mlx_load_error_symbol) - 1] = '\0';
}

const char* mlx_dynamic_load_error(void) {
    return mlx_load_error_symbol;
}

#ifdef _WIN32
#include <windows.h>
#include <string.h>

static void* mlx_dlopen(const char* path) {
    // Windows doesn't search the DLL's own directory for dependencies.
    char dir[MAX_PATH] = {0};
    strncpy(dir, path, MAX_PATH - 1);
    char* last_sep = NULL;
    for (char* p = dir; *p; p++) {
        if (*p == '\\' || *p == '/') last_sep = p;
    }
    if (last_sep) *last_sep = '\0';
    else dir[0] = '\0';

    if (dir[0]) SetDllDirectoryA(dir);
    void* h = (void*) LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
    SetDllDirectoryA(NULL);
    return h;
}

#define DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <libgen.h>
#endif
#include <dlfcn.h>
static void* mlx_dlopen(const char* path) {
    return dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
}
#define DLCLOSE(handle) dlclose(handle)
#endif

static int mlx_dynamic_open(mlx_dynamic_handle* handle, const char* path) {
    handle->ctx = mlx_dlopen(path);
    if (handle->ctx == NULL) {
        return 1;
    }
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

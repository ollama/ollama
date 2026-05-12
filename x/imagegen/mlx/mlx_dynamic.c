// mlx_dynamic.c - Dynamic loading wrapper for MLX-C library
// This file provides runtime dynamic loading of libmlxc instead of link-time binding

#include "mlx_dynamic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle_t;
static char win_error_buffer[256] = {0};
static const char* get_win_error(void) {
    DWORD err = GetLastError();
    snprintf(win_error_buffer, sizeof(win_error_buffer), "error code %lu", err);
    return win_error_buffer;
}
#define LIB_ERROR() get_win_error()
#else
#include <dlfcn.h>
typedef void* lib_handle_t;
#define LIB_ERROR() dlerror()
#endif

static lib_handle_t mlx_handle = NULL;
static int mlx_initialized = 0;
static char mlx_error_buffer[512] = {0};

#ifdef _WIN32
// Windows: Load library from a path with dependency resolution.
// Temporarily adds the library's directory to the DLL search path
// so that dependencies (like mlx.dll) in the same directory are found.
static int try_load_win(const char* path) {
    if (!path) return 0;

    // Extract directory and add to DLL search path for dependency resolution
    char dir_path[MAX_PATH];
    strncpy(dir_path, path, MAX_PATH - 1);
    dir_path[MAX_PATH - 1] = '\0';
    char* last_slash = strrchr(dir_path, '\\');
    if (!last_slash) last_slash = strrchr(dir_path, '/');
    if (last_slash) {
        *last_slash = '\0';
        SetDllDirectoryA(dir_path);
    }

    mlx_handle = LoadLibraryA(path);
    SetDllDirectoryA(NULL);
    return mlx_handle != NULL;
}
#endif

// Try to load library from a specific path
static int try_load_lib(const char* path) {
    if (!path) return 0;
#ifdef _WIN32
    return try_load_win(path);
#else
    mlx_handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    return mlx_handle != NULL;
#endif
}

// Initialize the MLX dynamic library from a specific path.
// Returns 0 on success, -1 on failure.
int mlx_dynamic_init_path(const char* path) {
    if (mlx_initialized) {
        return 0;
    }

    if (try_load_lib(path)) {
        mlx_initialized = 1;
        snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
                 "MLX: Successfully loaded %s", path ? path : "library");
        return 0;
    }

    const char* err = LIB_ERROR();
    snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
             "MLX: Failed to load %s: %s", path ? path : "(null)", err ? err : "unknown error");
    return -1;
}

// Get the last error message
const char* mlx_dynamic_error(void) {
    return mlx_error_buffer;
}

// Get the library handle (for use by generated wrappers)
void* mlx_get_handle(void) {
    return mlx_handle;
}


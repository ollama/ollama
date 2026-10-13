// mlx_dynamic.c - Dynamic loading wrapper for MLX-C library
// This file provides runtime dynamic loading of libmlxc instead of link-time binding

#include "mlx_dynamic.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE lib_handle_t;
#define LOAD_LIB(path) LoadLibraryA(path)
#define GET_SYMBOL(handle, name) GetProcAddress(handle, name)
#define CLOSE_LIB(handle) FreeLibrary(handle)
#define LIB_ERROR() "LoadLibrary failed"
static const char* LIB_NAMES[] = {"libmlxc.dll", NULL};
#else
#include <dlfcn.h>
typedef void* lib_handle_t;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define GET_SYMBOL(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#define LIB_ERROR() dlerror()
#ifdef __APPLE__
static const char* LIB_NAMES[] = {
    "libmlxc.dylib",
    "@loader_path/../build/lib/ollama/libmlxc.dylib",
    "@executable_path/../build/lib/ollama/libmlxc.dylib",
    "build/lib/ollama/libmlxc.dylib",
    "../build/lib/ollama/libmlxc.dylib",
    NULL
};
#else
static const char* LIB_NAMES[] = {
    "libmlxc.so",
    "$ORIGIN/../build/lib/ollama/libmlxc.so",
    "build/lib/ollama/libmlxc.so",
    "../build/lib/ollama/libmlxc.so",
    NULL
};
#endif
#endif

static lib_handle_t mlx_handle = NULL;
static int mlx_initialized = 0;
static char mlx_error_buffer[512] = {0};

// Initialize MLX dynamic library
// Returns 0 on success, -1 on failure
// On failure, call mlx_dynamic_error() to get error message
int mlx_dynamic_init(void) {
    if (mlx_initialized) {
        return 0;  // Already initialized
    }

    // Try each possible library path
    for (int i = 0; LIB_NAMES[i] != NULL; i++) {
        mlx_handle = LOAD_LIB(LIB_NAMES[i]);
        if (mlx_handle != NULL) {
            mlx_initialized = 1;
            snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
                     "MLX: Successfully loaded %s", LIB_NAMES[i]);
            return 0;
        }
    }

    // Failed to load library
    const char* err = LIB_ERROR();
    snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
             "MLX: Failed to load libmlxc library. %s",
             err ? err : "Unknown error");
    return -1;
}

// Get the last error message
const char* mlx_dynamic_error(void) {
    return mlx_error_buffer;
}

// Check if MLX is initialized
int mlx_dynamic_is_initialized(void) {
    return mlx_initialized;
}

// Cleanup (optional, called at program exit)
void mlx_dynamic_cleanup(void) {
    if (mlx_handle != NULL) {
        CLOSE_LIB(mlx_handle);
        mlx_handle = NULL;
        mlx_initialized = 0;
    }
}

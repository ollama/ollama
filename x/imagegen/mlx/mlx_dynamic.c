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
#else
#include <dlfcn.h>
typedef void* lib_handle_t;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define GET_SYMBOL(handle, name) dlsym(handle, name)
#define CLOSE_LIB(handle) dlclose(handle)
#define LIB_ERROR() dlerror()
#ifdef __APPLE__
#include <mach-o/dyld.h>
#include <libgen.h>
#endif
#endif

static lib_handle_t mlx_handle = NULL;
static int mlx_initialized = 0;
static char mlx_error_buffer[512] = {0};

#ifdef __APPLE__
// Get path to library in same directory as executable
static char* get_exe_relative_path(const char* libname) {
    static char path[1024];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) {
        return NULL;
    }
    // Get directory of executable
    char* dir = dirname(path);
    static char fullpath[1024];
    snprintf(fullpath, sizeof(fullpath), "%s/%s", dir, libname);
    return fullpath;
}
#endif

// Try to load library from a specific path
static int try_load_lib(const char* path) {
    if (!path) return 0;
    mlx_handle = LOAD_LIB(path);
    return mlx_handle != NULL;
}

// Initialize MLX dynamic library
// Returns 0 on success, -1 on failure
// On failure, call mlx_dynamic_error() to get error message
int mlx_dynamic_init(void) {
    if (mlx_initialized) {
        return 0;  // Already initialized
    }

    const char* lib_path = NULL;
    const char* tried_paths[8] = {0};
    int num_tried = 0;

#ifdef _WIN32
    // Windows: try same directory as executable
    lib_path = "libmlxc.dll";
    tried_paths[num_tried++] = lib_path;
    if (try_load_lib(lib_path)) goto success;
#elif defined(__APPLE__)
    // macOS: try executable directory first
    lib_path = get_exe_relative_path("libmlxc.dylib");
    if (lib_path) {
        tried_paths[num_tried++] = lib_path;
        if (try_load_lib(lib_path)) goto success;
    }
    // Try build directory (for tests run from repo root)
    lib_path = "./build/lib/ollama/libmlxc.dylib";
    tried_paths[num_tried++] = lib_path;
    if (try_load_lib(lib_path)) goto success;
    // Fallback to system paths
    lib_path = "libmlxc.dylib";
    tried_paths[num_tried++] = lib_path;
    if (try_load_lib(lib_path)) goto success;
#else
    // Linux: try build directory first (for tests)
    lib_path = "./build/lib/ollama/libmlxc.so";
    tried_paths[num_tried++] = lib_path;
    if (try_load_lib(lib_path)) goto success;
    // Fallback to system paths
    lib_path = "libmlxc.so";
    tried_paths[num_tried++] = lib_path;
    if (try_load_lib(lib_path)) goto success;
#endif

    // Failed to load library - build error message with all tried paths
    {
        const char* err = LIB_ERROR();
        int offset = snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
                     "MLX: Failed to load libmlxc library. Tried: ");
        for (int i = 0; i < num_tried && offset < (int)sizeof(mlx_error_buffer) - 50; i++) {
            offset += snprintf(mlx_error_buffer + offset, sizeof(mlx_error_buffer) - offset,
                             "%s%s", i > 0 ? ", " : "", tried_paths[i]);
        }
        if (err) {
            snprintf(mlx_error_buffer + offset, sizeof(mlx_error_buffer) - offset,
                    ". Last error: %s", err);
        }
    }
    return -1;

success:
    mlx_initialized = 1;
    snprintf(mlx_error_buffer, sizeof(mlx_error_buffer),
             "MLX: Successfully loaded %s", lib_path ? lib_path : "library");
    return 0;
}

// Get the last error message
const char* mlx_dynamic_error(void) {
    return mlx_error_buffer;
}

// Check if MLX is initialized
int mlx_dynamic_is_initialized(void) {
    return mlx_initialized;
}

// Get the library handle (for use by generated wrappers)
void* mlx_get_handle(void) {
    return mlx_handle;
}

// Cleanup (optional, called at program exit)
void mlx_dynamic_cleanup(void) {
    if (mlx_handle != NULL) {
        CLOSE_LIB(mlx_handle);
        mlx_handle = NULL;
        mlx_initialized = 0;
    }
}

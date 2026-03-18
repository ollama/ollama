#include "dynamic.h"

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define DLCLOSE(handle) FreeLibrary((HMODULE)(handle))
#else
#include <dlfcn.h>
#define DLOPEN(path) dlopen(path, RTLD_LAZY | RTLD_GLOBAL)
#define DLCLOSE(handle) dlclose(handle)
#endif

static int oga_dynamic_open(oga_dynamic_handle* handle, const char* path) {
#ifdef _WIN32
    // On Windows, set the DLL directory to the library's parent directory
    // so that dependent DLLs (onnxruntime.dll, DirectML.dll) are found
    // from the same directory rather than from an older system copy.
    char dir[MAX_PATH];
    size_t len = strlen(path);
    if (len >= MAX_PATH) len = MAX_PATH - 1;
    memcpy(dir, path, len);
    dir[len] = '\0';
    // Find the last path separator and truncate to get the directory
    char* sep = strrchr(dir, '\\');
    if (!sep) sep = strrchr(dir, '/');
    if (sep) {
        *sep = '\0';
        SetDllDirectoryA(dir);
    }
    handle->ctx = (void*) LoadLibraryA(path);
    // Reset DLL directory to default search order
    SetDllDirectoryA(NULL);
#else
    handle->ctx = (void*) DLOPEN(path);
#endif
    if (handle->ctx == NULL) {
        return 1;
    }
    return 0;
}

int oga_dynamic_load(oga_dynamic_handle* handle, const char *path) {
    return oga_dynamic_open(handle, path);
}

void oga_dynamic_unload(oga_dynamic_handle* handle) {
    if (handle->ctx) {
        DLCLOSE(handle->ctx);
        handle->ctx = NULL;
    }
}

#include "dynamic.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif

static void* open_library(const char* path) {
    int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, NULL, 0);
    if (length == 0) {
        return NULL;
    }
    wchar_t* wide_path = malloc((size_t)length * sizeof(wchar_t));
    if (wide_path == NULL) {
        return NULL;
    }
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1, wide_path, length) == 0) {
        free(wide_path);
        return NULL;
    }
    HMODULE module = LoadLibraryExW(
        wide_path,
        NULL,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    free(wide_path);
    return (void*)module;
}

#define OPEN(path) open_library(path)
#define CLOSE(handle) FreeLibrary((HMODULE)(handle))
#define SYMBOL(handle, name) ((void*)GetProcAddress((HMODULE)(handle), name))
#else
#include <dlfcn.h>
#define OPEN(path) dlopen(path, RTLD_NOW | RTLD_LOCAL)
#define CLOSE(handle) dlclose(handle)
#define SYMBOL(handle, name) dlsym(handle, name)
#endif

static const char* (*version_fn)(void);
static const char* (*last_error_fn)(void);
static int (*compiler_new_fn)(const char*, size_t, const uint64_t*, size_t, int32_t, const int32_t*, size_t, int32_t, int64_t, ollama_xgrammar_compiler**);
static void (*compiler_free_fn)(ollama_xgrammar_compiler*);
static int (*matcher_new_fn)(ollama_xgrammar_compiler*, ollama_xgrammar_kind, const char*, size_t, ollama_xgrammar_matcher**);
static void (*matcher_free_fn)(ollama_xgrammar_matcher*);
static int (*matcher_fill_fn)(ollama_xgrammar_matcher*, int32_t*, size_t, int*);
static int (*matcher_accept_fn)(ollama_xgrammar_matcher*, int32_t, int*);
static const char* load_error;

static void clear_symbols(void) {
    version_fn = NULL;
    last_error_fn = NULL;
    compiler_new_fn = NULL;
    compiler_free_fn = NULL;
    matcher_new_fn = NULL;
    matcher_free_fn = NULL;
    matcher_fill_fn = NULL;
    matcher_accept_fn = NULL;
}

#define LOAD(handle, field, name) do { \
    *(void**)(&field) = SYMBOL((handle)->ctx, name); \
    if ((field) == NULL) { load_error = "xgrammar library is missing symbol " name; goto fail; } \
} while (0)

int ollama_xgrammar_dynamic_load(ollama_xgrammar_dynamic_handle* handle, const char* path) {
    clear_symbols();
    if (handle == NULL || path == NULL) {
        load_error = "invalid xgrammar library path";
        return 1;
    }
    handle->ctx = OPEN(path);
    if (handle->ctx == NULL) {
        load_error = "unable to open xgrammar library";
        return 1;
    }
    LOAD(handle, version_fn, "ollama_xgrammar_version");
    LOAD(handle, last_error_fn, "ollama_xgrammar_last_error");
    LOAD(handle, compiler_new_fn, "ollama_xgrammar_compiler_new");
    LOAD(handle, compiler_free_fn, "ollama_xgrammar_compiler_free");
    LOAD(handle, matcher_new_fn, "ollama_xgrammar_matcher_new");
    LOAD(handle, matcher_free_fn, "ollama_xgrammar_matcher_free");
    LOAD(handle, matcher_fill_fn, "ollama_xgrammar_matcher_fill");
    LOAD(handle, matcher_accept_fn, "ollama_xgrammar_matcher_accept");
    load_error = NULL;
    return 0;

fail:
    CLOSE(handle->ctx);
    handle->ctx = NULL;
    clear_symbols();
    return 1;
}

const char* ollama_xgrammar_dynamic_version(void) {
    return version_fn == NULL ? "" : version_fn();
}

const char* ollama_xgrammar_dynamic_error(void) {
    if (last_error_fn != NULL) {
        const char* message = last_error_fn();
        if (message != NULL && message[0] != '\0') {
            return message;
        }
    }
    return load_error == NULL ? "xgrammar library error" : load_error;
}

static int capture_error(int result, char** error) {
    if (error != NULL) {
        *error = NULL;
    }
    if (result == 0 || error == NULL) {
        return result;
    }
    const char* message = ollama_xgrammar_dynamic_error();
    size_t size = strlen(message) + 1;
    *error = malloc(size);
    if (*error != NULL) {
        memcpy(*error, message, size);
    }
    return result;
}

int ollama_xgrammar_dynamic_compiler_new(const char* d, size_t ds, const uint64_t* o, size_t n, int32_t v, const int32_t* s, size_t ns, int32_t mt, int64_t cb, ollama_xgrammar_compiler** m, char** error) {
    return capture_error(compiler_new_fn(d, ds, o, n, v, s, ns, mt, cb, m), error);
}
void ollama_xgrammar_dynamic_compiler_free(ollama_xgrammar_compiler* m) { compiler_free_fn(m); }
int ollama_xgrammar_dynamic_matcher_new(ollama_xgrammar_compiler* m, ollama_xgrammar_kind k, const char* s, size_t n, ollama_xgrammar_matcher** out, char** error) {
    return capture_error(matcher_new_fn(m, k, s, n, out), error);
}
void ollama_xgrammar_dynamic_matcher_free(ollama_xgrammar_matcher* m) { matcher_free_fn(m); }
int ollama_xgrammar_dynamic_matcher_fill(ollama_xgrammar_matcher* m, int32_t* b, size_t n, int* a, char** error) {
    return capture_error(matcher_fill_fn(m, b, n, a), error);
}
int ollama_xgrammar_dynamic_matcher_accept(ollama_xgrammar_matcher* m, int32_t t, int* a, char** error) {
    return capture_error(matcher_accept_fn(m, t, a), error);
}

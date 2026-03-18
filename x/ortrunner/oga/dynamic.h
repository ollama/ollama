#ifndef OGA_DYNAMIC_H
#define OGA_DYNAMIC_H

#ifdef _WIN32
#include <windows.h>
#define DLSYM(handle, symbol) (void*)GetProcAddress((HMODULE)(handle.ctx), symbol)
#else
#include <dlfcn.h>
#define DLSYM(handle, symbol) dlsym(handle.ctx, symbol)
#endif

#include <stdio.h>

#ifdef ERROR
#undef ERROR
#endif
#define OGA_ERROR(fmt, ...) fprintf(stderr, "%s %s - ERROR - %s:%d - " fmt "\n", __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__); return 1
#define CHECK(x) if (!(x)) { OGA_ERROR("CHECK failed: " #x); }
#define CHECK_LOAD(handle, x) *(void**)(&x##_) = DLSYM(handle, #x); CHECK(x##_)

typedef struct {
    void* ctx;
} oga_dynamic_handle;

int oga_dynamic_load(
    oga_dynamic_handle* handle,
    const char *path);

void oga_dynamic_unload(
    oga_dynamic_handle* handle);

#endif // OGA_DYNAMIC_H

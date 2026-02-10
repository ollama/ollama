#ifndef MLX_DYNAMIC_H
#define MLX_DYNAMIC_H

#ifdef _WIN32
#include <windows.h>
#define DLSYM(handle, symbol) GetProcAddress((HMODULE)(handle), symbol)
#else
#include <dlfcn.h>
#define DLSYM(handle, symbol) dlsym(handle.ctx, symbol)
#endif

#define ERROR(fmt, ...) fprintf(stderr, "%s %s - ERROR - %s:%d - " fmt "\n", __DATE__, __TIME__, __FILE__, __LINE__, ##__VA_ARGS__); return 1
#define CHECK(x) if (!(x)) { ERROR("CHECK failed: " #x); }
#define CHECK_LOAD(handle, x) x##_ = DLSYM(handle, #x); CHECK(x##_)

typedef struct {
    void* ctx;
} mlx_dynamic_handle;

int mlx_dynamic_load(
    mlx_dynamic_handle* handle,
    const char *path);

void mlx_dynamic_unload(
    mlx_dynamic_handle* handle);

#endif // MLX_DYNAMIC_H

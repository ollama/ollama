// mlx_dynamic.h - Dynamic loading interface for MLX-C library
#ifndef MLX_DYNAMIC_H
#define MLX_DYNAMIC_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the MLX dynamic library from a specific path
// Returns 0 on success, -1 on failure
int mlx_dynamic_init_path(const char* path);

// Get the last error message from dynamic loading
const char* mlx_dynamic_error(void);

// Get the library handle (for use by generated wrappers)
void* mlx_get_handle(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_DYNAMIC_H

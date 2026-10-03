// mlx_dynamic.h - Dynamic loading interface for MLX-C library
#ifndef MLX_DYNAMIC_H
#define MLX_DYNAMIC_H

#ifdef __cplusplus
extern "C" {
#endif

// Initialize the MLX dynamic library
// Returns 0 on success, -1 on failure
int mlx_dynamic_init(void);

// Get the last error message from dynamic loading
const char* mlx_dynamic_error(void);

// Check if MLX is initialized
int mlx_dynamic_is_initialized(void);

// Cleanup resources (optional, for clean shutdown)
void mlx_dynamic_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif // MLX_DYNAMIC_H

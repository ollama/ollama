// mlx_error_handler.h - Safe error handling for MLX initialization
// This replaces the default exit(-1) MLX error handler during init()
// so that GPU failures don't kill the process.

#ifndef MLX_ERROR_HANDLER_H
#define MLX_ERROR_HANDLER_H

// Enter safe mode before any MLX compute calls during init().
// Replaces the default exit(-1) handler with one that silently stores errors.
void mlx_set_safe_init_mode(void);

// Restore the default MLX error handler (exit on error).
// Call from runner entry points after confirming MLX is available.
void mlx_set_default_error_mode(void);

// Check whether an error occurred while in safe init mode.
int mlx_had_init_error(void);

// Get the error message from the last init error, or NULL if none.
const char* mlx_get_init_error(void);

#endif // MLX_ERROR_HANDLER_H

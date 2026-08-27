// Package mlx wraps the MLX C API.
//
// # Threading
//
// MLX default streams are thread-local. Keep construction and evaluation of a
// lazy graph on one locked OS thread. Materialize arrays with Eval before
// handing them to another MLX thread.
//
// Inference uses x/internal/mlxthread, creation uses x/create's pinned thread,
// and tests use x/internal/mlxtest.
package mlx

// Package mlx wraps the MLX C API.
//
// # Threading contract
//
// MLX default streams are thread-local: an array records the default stream of
// the thread that created it and can only be evaluated on that thread.
// Cross-thread evaluation can panic with "There is no Stream(gpu, N) in current
// thread".
//
// Callers must therefore do MLX work from a single pinned goroutine per
// component. Production uses x/internal/mlxthread for inference and tests use
// x/internal/mlxtest. DefaultStream resolves per thread. To pass arrays across
// threads, evaluate them first.
package mlx

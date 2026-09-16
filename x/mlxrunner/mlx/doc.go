// Package mlx wraps the MLX C API.
//
// # Threading contract (read before calling from a new goroutine)
//
// MLX default streams are thread-local (since MLX 0.31.2): an array
// records the default stream of the thread that created it and can only
// be evaluated on that thread. Cross-thread evaluation panics with
// "There is no Stream(gpu, N) in current thread". There are no defensive
// checks — the panic is the enforcement.
//
// Callers must therefore do all MLX work from a single pinned goroutine
// per component: production uses x/internal/mlxthread (inference) and
// x/create's pinned thread (quantization); tests use x/internal/mlxtest.
// DefaultStream resolves per thread. To pass arrays across threads, Eval
// them first — an already-evaluated array is plain data and safe to read
// from any thread (upstream behavior, ml-explore/mlx#3529).
package mlx

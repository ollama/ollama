// Package mlxtest provides shared scaffolding for tests that exercise MLX
// through the cgo wrapper in x/mlxrunner/mlx.
package mlxtest

import (
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SkipIfUnavailable skips the test when the MLX dynamic library cannot be
// loaded (e.g. no MLX backend built for this platform).
func SkipIfUnavailable(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// Setup prepares a test that calls into MLX natively: it skips when MLX is
// unavailable and pins the test goroutine to its OS thread for the duration
// of the test.
//
// The thread pin is load-bearing, not defensive: MLX's default stream cache
// is thread-local, and anything that migrates the goroutine mid-test (the
// race detector's scheduler in particular) otherwise panics with
// "There is no Stream(gpu, 0) in current thread".
//
// Setup leaves device and allocator state unchanged because some tests share
// materialized arrays with subtests running on other threads.
func Setup(t *testing.T) {
	t.Helper()

	SkipIfUnavailable(t)

	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
}

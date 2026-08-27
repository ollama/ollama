// Package mlxtest provides shared scaffolding for tests that exercise MLX
// through the cgo wrapper in x/mlxrunner/mlx.
package mlxtest

import (
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/internal/mlxthreadtest"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var testThread = sync.OnceValues(func() (*mlxthread.Thread, error) {
	return mlxthread.Start("mlx-test", func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
})

// T is the test state available to callbacks running on the MLX thread.
type T = mlxthreadtest.T

// SkipIfUnavailable skips the test when the MLX dynamic library cannot be
// loaded (e.g. no MLX backend built for this platform).
func SkipIfUnavailable(t *testing.T) {
	t.Helper()
	if _, err := testThread(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// Run executes fn on the MLX thread shared by the package's test binary.
func Run(t *testing.T, fn func(*T)) {
	t.Helper()

	thread, err := testThread()
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	mlxthreadtest.Run(t, thread, fn)
}

// RunSubtest runs a named subtest on the shared MLX test thread.
func RunSubtest(t *testing.T, name string, fn func(*T)) {
	t.Helper()
	t.Run(name, func(t *testing.T) { Run(t, fn) })
}

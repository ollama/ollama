// Package mlxtest provides shared scaffolding for tests that exercise MLX
// through the cgo wrapper in x/mlxrunner/mlx.
package mlxtest

import (
	"context"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var (
	testThreadOnce sync.Once
	testThread     *mlxthread.Thread
	testThreadErr  error
)

// SkipIfUnavailable skips the test when the MLX dynamic library cannot be
// loaded (e.g. no MLX backend built for this platform).
func SkipIfUnavailable(t *testing.T) {
	t.Helper()
	if err := mlx.CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

// Run executes fn on the MLX thread shared by the package's test binary.
func Run(t *testing.T, fn func(*testing.T)) {
	t.Helper()

	testThreadOnce.Do(func() {
		testThread, testThreadErr = mlxthread.Start("mlx-test", func() error {
			if err := mlx.CheckInit(); err != nil {
				return err
			}
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
			}
			return nil
		})
	})
	if testThreadErr != nil {
		t.Skipf("MLX not available: %v", testThreadErr)
	}

	if err := testThread.Do(context.Background(), func() error {
		fn(t)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

// RunSubtest runs a named subtest on the shared MLX test thread.
func RunSubtest(t *testing.T, name string, fn func(*testing.T)) {
	t.Helper()
	t.Run(name, func(t *testing.T) { Run(t, fn) })
}

// Cleanup registers fn to run on the shared MLX test thread.
func Cleanup(t *testing.T, fn func()) {
	t.Helper()
	t.Cleanup(func() {
		Run(t, func(*testing.T) { fn() })
	})
}

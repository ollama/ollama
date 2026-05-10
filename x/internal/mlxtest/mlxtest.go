package mlxtest

import (
	"context"
	"runtime"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// SkipIfUnavailable binds the current test goroutine to an MLX-capable OS
// thread and skips the test when MLX is unavailable.
func SkipIfUnavailable(tb testing.TB) {
	tb.Helper()

	runtime.LockOSThread()
	if err := mlx.CheckInit(); err != nil {
		runtime.UnlockOSThread()
		tb.Skipf("MLX not available: %v", err)
	}

	mlx.BindCurrentThread()
	if mlx.GPUIsAvailable() {
		mlx.SetDefaultDeviceGPU()
	}

	tb.Cleanup(func() {
		cleanupCurrentThread()
		runtime.UnlockOSThread()
	})
}

// StartThread creates a long-lived MLX worker thread for tests that need to
// run all MLX work on a dedicated locked OS thread.
func StartThread(tb testing.TB, name string) *mlxthread.Thread {
	tb.Helper()

	thread, err := mlxthread.Start(name, func() error {
		if err := mlx.CheckInit(); err != nil {
			return err
		}
		mlx.BindCurrentThread()
		if mlx.GPUIsAvailable() {
			mlx.SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		tb.Skipf("MLX not available: %v", err)
	}

	return thread
}

// StopThread stops a thread created by StartThread and scrubs the MLX state
// that belongs to that OS thread.
func StopThread(tb testing.TB, thread *mlxthread.Thread) {
	tb.Helper()

	if err := thread.Stop(context.Background(), cleanupCurrentThread); err != nil {
		tb.Fatal(err)
	}
}

// WithThread runs fn on a dedicated MLX thread, skipping the test if MLX is
// unavailable.
func WithThread(tb testing.TB, fn func()) {
	tb.Helper()

	thread := StartThread(tb, "mlx-test")
	defer StopThread(tb, thread)

	if err := thread.Do(context.Background(), func() error {
		fn()
		return nil
	}); err != nil {
		tb.Fatal(err)
	}
}

func cleanupCurrentThread() {
	mlx.Sweep()
	mlx.ClearCache()
	mlx.ResetPeakMemory()
	mlx.UnbindCurrentThread()
}

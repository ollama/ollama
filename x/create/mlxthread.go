package create

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var (
	mlxThreadOnce    sync.Once
	mlxThreadStarted atomic.Bool
	mlxWork          chan func()
	mlxInitErr       error
)

// runOnMLXThread runs f on the MLX thread and returns its error. The thread is
// started (and MLX initialized) on first use. A panic in f is recovered and
// returned as an error so a kernel failure cannot kill the pinned thread.
//
// TODO(pdevine): This method should be revisited when the `ollama create` is
// instead run on the ollama server process instead of the client.
func runOnMLXThread(f func() error) error {
	mlxThreadOnce.Do(func() {
		mlxWork = make(chan func())
		ready := make(chan error)
		go func() {
			runtime.LockOSThread() // pinned for the process lifetime; never unlocked
			err := mlx.CheckInit()
			if err == nil && mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
			}
			ready <- err
			if err != nil {
				return
			}
			for work := range mlxWork {
				work()
			}
		}()
		mlxInitErr = <-ready
		mlxThreadStarted.Store(mlxInitErr == nil)
	})
	if mlxInitErr != nil {
		return fmt.Errorf("MLX init failed: %w", mlxInitErr)
	}

	done := make(chan error, 1)
	mlxWork <- func() {
		defer func() {
			if r := recover(); r != nil {
				done <- fmt.Errorf("mlx: %v", r)
			}
		}()
		done <- f()
	}
	return <-done
}

// sweepMLX releases the MLX buffer cache. It is a no-op if no MLX work has run.
func sweepMLX() {
	if !mlxThreadStarted.Load() {
		return
	}
	_ = runOnMLXThread(func() error {
		mlx.ClearCache()
		mlx.Sweep()
		return nil
	})
}

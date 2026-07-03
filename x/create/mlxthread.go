package create

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

var (
	createMLXThread = sync.OnceValues(func() (*mlxthread.Thread, error) {
		return mlxthread.Start("mlx-create", func() error {
			if err := mlx.CheckInit(); err != nil {
				return err
			}
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
			}
			return nil
		})
	})
	mlxThreadStarted atomic.Bool
)

// runOnMLXThread runs f on the MLX thread and returns its error. The thread is
// started (and MLX initialized) on first use.
func runOnMLXThread(ctx context.Context, f func() error) error {
	if err := checkContext(ctx); err != nil {
		return err
	}

	thread, err := createMLXThread()
	if err != nil {
		return fmt.Errorf("MLX init failed: %w", err)
	}
	mlxThreadStarted.Store(true)

	return thread.Do(ctx, f)
}

// sweepMLX releases the MLX buffer cache. It is a no-op if no MLX work has run.
func sweepMLX() {
	if !mlxThreadStarted.Load() {
		return
	}
	_ = runOnMLXThread(context.Background(), func() error {
		mlx.ClearCache()
		mlx.Sweep()
		return nil
	})
}

package client

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

	"github.com/ollama/ollama/x/internal/mlxthread"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// MLX requires its CGO calls to run on a thread that has been initialized
// with a default device (see x/mlxrunner/server.go for the inference-side
// pattern). Without this, MLX panics with "There is no Stream(gpu, 1) in
// current thread" the first time it tries to schedule work on the GPU.
//
// Create-time quantization originally called mlx.* directly from arbitrary
// goroutines, which violated that invariant once the per-thread state
// requirement was tightened. This worker exists so all create-time MLX work
// runs on a single, properly-initialized OS thread, mirroring the runner.
var (
	workerOnce sync.Once
	workerInst *mlxthread.Thread
	workerErr  error
)

// mlxWorker returns the package-level MLX worker thread, starting it lazily
// on first use. Callers must wrap any mlx.* call in worker.Do(...).
func mlxWorker() (*mlxthread.Thread, error) {
	workerOnce.Do(func() {
		workerInst, workerErr = mlxthread.Start("create", func() error {
			if err := mlx.CheckInit(); err != nil {
				return fmt.Errorf("MLX not available: %w", err)
			}
			if mlx.GPUIsAvailable() {
				mlx.SetDefaultDeviceGPU()
				slog.Debug("create: MLX worker initialized", "device", "gpu", "version", mlx.Version())
			} else {
				slog.Debug("create: MLX worker initialized", "device", "cpu", "version", mlx.Version())
			}
			return nil
		})
	})
	return workerInst, workerErr
}

// runOnMLXWorker executes fn on the package-level MLX worker thread.
// Any function that calls mlx.* must use this (or be called transitively
// from a function that does).
func runOnMLXWorker(fn func() error) error {
	worker, err := mlxWorker()
	if err != nil {
		return fmt.Errorf("failed to start MLX worker: %w", err)
	}
	return worker.Do(context.Background(), fn)
}

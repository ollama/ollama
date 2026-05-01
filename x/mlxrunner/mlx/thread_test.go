package mlx

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func startMLXThread(t *testing.T) *mlxthread.Thread {
	t.Helper()

	thread, err := mlxthread.Start("mlx-test", func() error {
		if err := CheckInit(); err != nil {
			return err
		}
		if GPUIsAvailable() {
			SetDefaultDeviceGPU()
		}
		return nil
	})
	if err != nil {
		t.Skipf("MLX not available: %v", err)
	}

	return thread
}

func stopMLXThread(t *testing.T, thread *mlxthread.Thread) {
	t.Helper()

	if err := thread.Stop(context.Background(), func() {
		Sweep()
		ClearCache()
		resetDefaultStreamCache()
	}); err != nil {
		t.Fatal(err)
	}
}

func withMLXThread(t *testing.T, fn func()) {
	t.Helper()

	thread := startMLXThread(t)
	defer stopMLXThread(t, thread)

	if err := thread.Do(context.Background(), func() error {
		fn()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestThreadedMLXOperations(t *testing.T) {
	thread := startMLXThread(t)
	defer stopMLXThread(t, thread)

	oldProcs := runtime.GOMAXPROCS(8)
	defer runtime.GOMAXPROCS(oldProcs)

	const goroutines = 8
	const iterations = 8

	var wg sync.WaitGroup
	errCh := make(chan error, goroutines)
	for range goroutines {
		wg.Add(1)
		go func() {
			defer wg.Done()

			for range iterations {
				if err := thread.Do(context.Background(), func() error {
					a := FromValues([]float32{1, 2, 3, 4}, 2, 2)
					b := Matmul(a, a)
					AsyncEval(b)
					Eval(b)
					Sweep()
					ClearCache()
					return nil
				}); err != nil {
					errCh <- err
					return
				}
			}
		}()
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Fatal(err)
	}
}

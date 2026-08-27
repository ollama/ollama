package mlx

import (
	"context"
	"runtime"
	"sync"
	"testing"

	"github.com/ollama/ollama/x/internal/mlxthread"
)

var (
	testThreadOnce sync.Once
	testThread     *mlxthread.Thread
	testThreadErr  error
)

func skipIfNoMLX(t *testing.T) {
	t.Helper()
	if err := CheckInit(); err != nil {
		t.Skipf("MLX not available: %v", err)
	}
}

func mlxTestThread(t *testing.T) *mlxthread.Thread {
	t.Helper()

	testThreadOnce.Do(func() {
		testThread, testThreadErr = mlxthread.Start("mlx-test", func() error {
			if err := CheckInit(); err != nil {
				return err
			}
			if GPUIsAvailable() {
				SetDefaultDeviceGPU()
			}
			return nil
		})
	})
	if testThreadErr != nil {
		t.Skipf("MLX not available: %v", testThreadErr)
	}

	return testThread
}

func withMLXThread(t *testing.T, fn func()) {
	t.Helper()

	if err := mlxTestThread(t).Do(context.Background(), func() error {
		fn()
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}

func TestThreadedMLXOperations(t *testing.T) {
	thread := mlxTestThread(t)

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

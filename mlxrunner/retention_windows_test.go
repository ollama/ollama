package mlxrunner

import (
	"math"
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestWindowsRetentionObserver(t *testing.T) {
	mlxtest.Run(t, func(t *mlxtest.T) {
		if !mlx.CUDAIsAvailable() || mlx.GPUCount() != 1 {
			t.Skip("requires one CUDA device")
		}
		integrated, err := mlx.IntegratedGPU()
		if err != nil || !integrated {
			t.Skipf("requires an integrated CUDA device: %v", err)
		}
		budget, err := mlx.DeviceMemoryBudget()
		if err != nil {
			t.Skipf("requires native WDDM budget reporting: %v", err)
		}
		if budget == nil {
			t.Fatal("Windows CUDA budget missing without error")
		}
		observer, err := newCUDARetention()
		if err != nil {
			t.Fatal(err)
		}
		if observer == nil {
			t.Fatal("qualified iGPU did not start pressure monitoring")
		}
		defer observer.Close()
		memory, err := observer.memory.Read()
		if err != nil {
			t.Fatal(err)
		}
		if memory.sampledAt.IsZero() || memory.limit < 0 || memory.available.commit == math.MaxInt64 || memory.available.device == math.MaxInt64 {
			t.Fatalf("invalid observation: %+v", memory)
		}
		t.Logf("Windows iGPU observation: %+v", memory)
	})
}

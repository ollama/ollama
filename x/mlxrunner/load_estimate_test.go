package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestEstimateLoad(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "1048576")

	t.Run("no gpu leaves fit unknown to the backend", func(t *testing.T) {
		estimate := EstimateLoad(1<<30, nil)
		if estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad rejects a model without a GPU memory limit")
		}
	})

	t.Run("applies runner memory reserves", func(t *testing.T) {
		gpu := ml.DeviceInfo{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			FreeMemory: 2 << 30,
		}
		estimate := EstimateLoad(1<<30, []ml.DeviceInfo{gpu})
		wantAvailable := gpu.FreeMemory - gpu.MinimumMemory() - 1048576
		if estimate.available != wantAvailable {
			t.Fatalf("available memory = %d, want %d", estimate.available, wantAvailable)
		}
		if estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad rejects a model that fits")
		}
	})

	t.Run("rejects model larger than available memory", func(t *testing.T) {
		gpu := ml.DeviceInfo{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			FreeMemory: 1 << 30,
		}
		estimate := EstimateLoad(768<<20, []ml.DeviceInfo{gpu})
		if !estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad accepts a model larger than available memory")
		}
	})
}

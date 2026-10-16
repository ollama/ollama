package mlxrunner

import (
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestEstimateLoad(t *testing.T) {
	t.Setenv("OLLAMA_GPU_OVERHEAD", "1048576")

	t.Run("no gpu leaves fit unknown to the backend", func(t *testing.T) {
		estimate := EstimateLoad(1<<30, ml.SystemInfo{}, nil, true)
		if estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad rejects a model without a GPU memory limit")
		}
	})

	t.Run("applies runner memory reserves", func(t *testing.T) {
		gpu := ml.DeviceInfo{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			FreeMemory: 2 << 30,
		}
		estimate := EstimateLoad(1<<30, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true)
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
		estimate := EstimateLoad(768<<20, ml.SystemInfo{}, []ml.DeviceInfo{gpu}, true)
		if !estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad accepts a model larger than available memory")
		}
	})

	t.Run("caps integrated GPU memory by host memory for full loads", func(t *testing.T) {
		gpu := ml.DeviceInfo{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			Integrated: true,
			FreeMemory: 4 << 30,
		}
		systemInfo := ml.SystemInfo{FreeMemory: 1 << 30}
		estimate := EstimateLoad(768<<20, systemInfo, []ml.DeviceInfo{gpu}, true)
		if !estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad ignores the host memory limit for an integrated GPU")
		}
	})

	t.Run("does not cap a partial load by host memory", func(t *testing.T) {
		gpu := ml.DeviceInfo{
			DeviceID:   ml.DeviceID{Library: "Metal"},
			Integrated: true,
			FreeMemory: 4 << 30,
		}
		systemInfo := ml.SystemInfo{FreeMemory: 1 << 30}
		estimate := EstimateLoad(768<<20, systemInfo, []ml.DeviceInfo{gpu}, false)
		if estimate.ExceedsAvailableMemory() {
			t.Fatal("EstimateLoad applies the host memory limit to a partial load")
		}
	})
}

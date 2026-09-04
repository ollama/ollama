package mlxrunner

import (
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/ml"
)

// LoadEstimate is the memory decision MLX uses before starting a runner.
type LoadEstimate struct {
	modelSize uint64
	available uint64
	overhead  uint64
	hasGPU    bool
}

// ExceedsAvailableMemory reports whether the model weights are larger than
// the first GPU's free memory after MLX reserves. It returns false when no GPU
// is supplied; false means the estimate did not prove a non-fit, not that the
// complete model is guaranteed to load.
func (e LoadEstimate) ExceedsAvailableMemory() bool {
	return e.hasGPU && e.modelSize > e.available
}

// EstimateLoad returns MLX's current pre-load memory decision for modelSize.
// Keeping this calculation outside Client lets manifest-first pull admission
// and scheduler loads use the same backend-owned rule.
func EstimateLoad(modelSize uint64, gpus []ml.DeviceInfo) LoadEstimate {
	estimate := LoadEstimate{modelSize: modelSize}
	if len(gpus) == 0 {
		return estimate
	}

	// MLX currently runs on only the first GPU reported by discovery.
	estimate.hasGPU = true
	gpuFree := gpus[0].FreeMemory
	estimate.overhead = gpus[0].MinimumMemory() + envconfig.GpuOverhead()
	if gpuFree > estimate.overhead {
		estimate.available = gpuFree - estimate.overhead
	}
	return estimate
}

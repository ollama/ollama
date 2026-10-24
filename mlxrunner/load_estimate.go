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
// the first GPU's available memory after MLX reserves. It returns false when
// no GPU is supplied; false means the estimate did not prove a non-fit, not
// that the complete model is guaranteed to load.
func (e LoadEstimate) ExceedsAvailableMemory() bool {
	return e.hasGPU && e.modelSize > e.available
}

// EstimateLoad returns MLX's current pre-load memory decision for modelSize.
// Keeping this calculation outside Client lets manifest-first pull admission
// and scheduler loads use the same backend-owned rule.
func EstimateLoad(modelSize uint64, systemInfo ml.SystemInfo, gpus []ml.DeviceInfo, requireFull bool) LoadEstimate {
	estimate := LoadEstimate{modelSize: modelSize}
	if len(gpus) == 0 {
		return estimate
	}

	// MLX currently runs on only the first GPU reported by discovery.
	estimate.hasGPU = true
	estimate.available = gpus[0].FreeMemory
	if requireFull && gpus[0].Integrated && systemInfo.FreeMemory > 0 && systemInfo.FreeMemory < estimate.available {
		estimate.available = systemInfo.FreeMemory
	}

	estimate.overhead = gpus[0].MinimumMemory() + envconfig.GpuOverhead()
	if estimate.available > estimate.overhead {
		estimate.available -= estimate.overhead
	} else {
		estimate.available = 0
	}
	return estimate
}

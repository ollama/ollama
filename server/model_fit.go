package server

import (
	"context"
	"fmt"
	"math"
	"slices"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/x/mlxrunner"
)

type modelFitStatus uint8

const (
	modelFitUnknown modelFitStatus = iota
	modelDoesNotFit
)

type modelFitBackend string

const (
	modelFitBackendUnknown modelFitBackend = ""
	modelFitBackendMLX     modelFitBackend = "mlx"
	modelFitBackendLlama   modelFitBackend = "llama-server"
)

type modelFitAssessment struct {
	Status  modelFitStatus
	Backend modelFitBackend
}

// assessManifestFit applies the same backend load admission rule the scheduler
// uses, but against idle hardware capacity. Pull must not reject a model merely
// because another Ollama runner currently occupies memory the scheduler could
// reclaim before loading it.
func (s *Scheduler) assessManifestFit(ctx context.Context, mf *manifest.Manifest) modelFitAssessment {
	backend := fitBackendForManifest(mf)
	if backend != modelFitBackendMLX || s == nil || s.getGpuFn == nil {
		return modelFitAssessment{Status: modelFitUnknown, Backend: backend}
	}

	modelSize, ok := mlxManifestTensorSize(mf)
	if !ok {
		return modelFitAssessment{Status: modelFitUnknown, Backend: backend}
	}

	gpus := slices.Clone(s.getGpuFn(ctx, nil))
	if len(gpus) == 0 || gpus[0].TotalMemory == 0 {
		return modelFitAssessment{Status: modelFitUnknown, Backend: backend}
	}
	for i := range gpus {
		gpus[i].FreeMemory = gpus[i].TotalMemory
	}

	if mlxrunner.EstimateLoad(modelSize, gpus).ExceedsAvailableMemory() {
		return modelFitAssessment{Status: modelDoesNotFit, Backend: backend}
	}
	return modelFitAssessment{Status: modelFitUnknown, Backend: backend}
}

func (s *Scheduler) checkManifestFit(ctx context.Context, model string, mf *manifest.Manifest) error {
	assessment := s.assessManifestFit(ctx, mf)
	if assessment.Status != modelDoesNotFit {
		return nil
	}
	return fmt.Errorf("%s is too large to run on this system - Try a smaller local model, consider a cloud model, or --force to pull anyway", model)
}

func fitBackendForManifest(mf *manifest.Manifest) modelFitBackend {
	if mf == nil {
		return modelFitBackendUnknown
	}
	backend := modelFitBackendUnknown
	for _, layer := range mf.Layers {
		switch layer.MediaType {
		case manifest.MediaTypeImageTensor:
			return modelFitBackendMLX
		case "application/vnd.ollama.image.model":
			// GGUF fit needs metadata that is not currently present in the
			// registry manifest. Keep the backend visible while returning unknown.
			backend = modelFitBackendLlama
		}
	}
	return backend
}

func mlxManifestTensorSize(mf *manifest.Manifest) (uint64, bool) {
	if mf == nil {
		return 0, false
	}

	var size uint64
	found := false
	for _, layer := range mf.Layers {
		if layer.MediaType != manifest.MediaTypeImageTensor {
			continue
		}
		if layer.Size <= 0 || uint64(layer.Size) > math.MaxUint64-size {
			return 0, false
		}
		size += uint64(layer.Size)
		found = true
	}
	return size, found
}

package server

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/ml"
	"github.com/ollama/ollama/mlxrunner"
	"github.com/ollama/ollama/types/model"
)

type modelFitStatus uint8

const (
	modelFitUnknown modelFitStatus = iota
	modelDoesNotFit
)

// maxFitSuggestions caps the local models named in a non-fit error.
const maxFitSuggestions = 3

type modelFitAssessment struct {
	Status modelFitStatus

	// Idle capacity the assessment was made against, used to check
	// alternative models with the same rule.
	systemInfo ml.SystemInfo
	gpus       []ml.DeviceInfo
}

// fits reports whether a model needing size bytes passes the same admission
// rule. It is false when capacity is unknown.
func (a modelFitAssessment) fits(size uint64) bool {
	return len(a.gpus) > 0 && !mlxrunner.EstimateLoad(size, a.systemInfo, a.gpus, true).ExceedsAvailableMemory()
}

// assessManifestFit applies the same backend load admission rule the scheduler
// uses, but against idle hardware capacity. Pull must not reject a model merely
// because another Ollama runner currently occupies memory the scheduler could
// reclaim before loading it.
func (s *Scheduler) assessManifestFit(ctx context.Context, mf *manifest.Manifest) modelFitAssessment {
	// Only MLX fit can be judged from the manifest; GGUF fit needs metadata
	// the registry manifest does not carry.
	if !requiresMLX(mf) || s.getGpuFn == nil {
		return modelFitAssessment{Status: modelFitUnknown}
	}

	modelSize, ok := mlxManifestTensorSize(mf)
	if !ok {
		return modelFitAssessment{Status: modelFitUnknown}
	}

	gpus := slices.Clone(s.getGpuFn(ctx, nil))
	if len(gpus) == 0 || gpus[0].TotalMemory == 0 {
		return modelFitAssessment{Status: modelFitUnknown}
	}
	for i := range gpus {
		gpus[i].FreeMemory = gpus[i].TotalMemory
	}

	var systemInfo ml.SystemInfo
	if s.getSystemInfoFn != nil {
		systemInfo = s.getSystemInfoFn()
		systemInfo.FreeMemory = systemInfo.TotalMemory
	}

	assessment := modelFitAssessment{Status: modelFitUnknown, systemInfo: systemInfo, gpus: gpus}
	if !assessment.fits(modelSize) {
		assessment.Status = modelDoesNotFit
	}
	return assessment
}

// fitRecommendations returns current model recommendations for suggesting
// alternatives to a model that does not fit.
func (s *Server) fitRecommendations(ctx context.Context) []api.ModelRecommendation {
	recs := defaultModelRecommendations
	if s.modelCaches != nil && s.modelCaches.recommendations != nil {
		recs = s.modelCaches.recommendations.GetFresh(ctx)
	}
	return recs
}

// modelDoesNotFitError explains a rejected pull. With cloud enabled it names
// recommended local models that fit this system and a cloud alternative;
// otherwise, or without usable recommendations, it gives generic advice.
func modelDoesNotFitError(name string, assessment modelFitAssessment, recs []api.ModelRecommendation) error {
	var local []string
	var cloud string
	if !envconfig.NoCloud() {
		local, cloud = fitSuggestions(name, assessment, recs)
	}
	var msg string
	switch {
	case len(local) > 0 && cloud != "":
		msg = fmt.Sprintf("You can try %s locally, or consider %s. Use --force to pull %s anyway.", joinOr(local), cloud, name)
	case len(local) > 0:
		msg = fmt.Sprintf("You can try %s, or use --force to pull %s anyway.", joinOr(local), name)
	case cloud != "":
		msg = fmt.Sprintf("Try a smaller model, or consider %s. Use --force to pull %s anyway.", cloud, name)
	default:
		msg = fmt.Sprintf("Try a smaller model, or use --force to pull %s anyway.", name)
	}
	return fmt.Errorf("%s may not fit on your system. %s", name, msg)
}

// fitSuggestions picks the local models with the largest minimum VRAM that
// still fit, assuming a larger requirement means a more capable model, plus
// the first cloud recommendation.
func fitSuggestions(name string, assessment modelFitAssessment, recs []api.ModelRecommendation) (local []string, cloud string) {
	type candidate struct {
		model string
		vram  int64
	}
	var candidates []candidate
	requested := model.ParseName(name)
	for _, rec := range recs {
		if isCloudRecommendation(rec.Model) {
			if cloud == "" {
				cloud = rec.Model
			}
			continue
		}
		if model.ParseName(rec.Model).EqualFold(requested) {
			continue
		}
		if rec.VRAMBytes > 0 && assessment.fits(uint64(rec.VRAMBytes)) {
			candidates = append(candidates, candidate{rec.Model, rec.VRAMBytes})
		}
	}
	// Stable so equal requirements keep recommendation order.
	slices.SortStableFunc(candidates, func(a, b candidate) int { return cmp.Compare(b.vram, a.vram) })
	for _, c := range candidates[:min(len(candidates), maxFitSuggestions)] {
		local = append(local, c.model)
	}
	return local, cloud
}

// joinOr formats items as "a", "a or b", or "a, b, or c".
func joinOr(items []string) string {
	switch len(items) {
	case 0:
		return ""
	case 1:
		return items[0]
	case 2:
		return items[0] + " or " + items[1]
	}
	return strings.Join(items[:len(items)-1], ", ") + ", or " + items[len(items)-1]
}

func mlxManifestTensorSize(mf *manifest.Manifest) (uint64, bool) {
	layers := mf.TensorLayers()
	var size uint64
	for _, layer := range layers {
		if layer.Size <= 0 || uint64(layer.Size) > math.MaxUint64-size {
			return 0, false
		}
		size += uint64(layer.Size)
	}
	return size, len(layers) > 0
}

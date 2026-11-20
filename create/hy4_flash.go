package create

import (
	"encoding/json"
	"strings"
)

type hy4FlashImportTransform struct{}

func newHy4FlashImportTransform(json.RawMessage) (quantizePolicy, error) {
	return hy4FlashImportTransform{}, nil
}

func (hy4FlashImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// Tune the routed expert banks first. Engram stays BF16 for the mapped
	// lookup and its trained E4M3 read path. The other projections are small
	// compared to the experts and remain in publisher precision.
	if len(shape) == 3 && strings.Contains(name, ".mlp.experts.") {
		return GetTensorQuantization(name, shape, quantize)
	}
	return ""
}

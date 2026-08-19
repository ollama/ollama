package create

import (
	"encoding/json"
	"strings"
)

type dflash2ImportTransform struct{ defaultQuantPolicy }

func newDFlash2ImportTransform(json.RawMessage) (quantizePolicy, error) {
	return dflash2ImportTransform{}, nil
}

func (dflash2ImportTransform) quantizationType(name string, shape []int32, requested string) string {
	if strings.HasSuffix(name, "candidate_selector.predecessor_codebook") ||
		strings.HasSuffix(name, "candidate_selector.successor_codebook") {
		requested = normalizeQuantType(requested)
		if len(shape) == 2 && isAligned(shape, requested) {
			return requested
		}
		return ""
	}
	return defaultQuantPolicy{}.quantizationType(name, shape, requested)
}

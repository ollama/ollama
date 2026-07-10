package create

import (
	"encoding/json"
	"strings"
)

type lagunaImportTransform struct{}

func newLagunaImportTransform(json.RawMessage) (quantizePolicy, error) {
	return lagunaImportTransform{}, nil
}

func (lagunaImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if !lagunaIsHFRoutedExpertWeight(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func lagunaIsHFRoutedExpertWeight(name string) bool {
	return strings.HasSuffix(name, ".weight") && strings.Contains(name, ".mlp.experts.")
}

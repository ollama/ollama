package create

import (
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type lagunaImportTransform struct{}

func newLagunaImportTransform(string, sourceModelConfig) (tensorImportTransform, error) {
	return lagunaImportTransform{}, nil
}

func (lagunaImportTransform) skipTensor(string) bool { return false }

func (lagunaImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}
	return []*safetensors.TensorData{td}, nil
}

func (lagunaImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if !lagunaIsHFRoutedExpertWeight(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func (lagunaImportTransform) sourceFP8TensorQuantization(name string, shape []int32, requested string, fallback string) string {
	if !lagunaIsHFRoutedExpertWeight(name) {
		return ""
	}

	switch normalizeQuantType(requested) {
	case "nvfp4", "mxfp4":
		if lagunaKeepSourceFP8TensorAtMXFP8(name, shape) {
			return "mxfp8"
		}
	}
	return fallback
}

func (lagunaImportTransform) sourceFP8BF16Quantization(string, []int32, string) string {
	return ""
}

func lagunaKeepSourceFP8TensorAtMXFP8(name string, shape []int32) bool {
	if len(shape) != 2 || !isAligned(shape, "mxfp8") {
		return false
	}

	return strings.Contains(name, "down_proj")
}

func lagunaIsHFRoutedExpertWeight(name string) bool {
	return strings.HasSuffix(name, ".weight") && strings.Contains(name, ".mlp.experts.")
}

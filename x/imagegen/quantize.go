package imagegen

import (
	"io"
	"strings"
)

// QuantizingTensorLayerCreator creates tensor layers with optional quantization.
// When quantize is true, returns multiple layers (weight + scales + biases).
type QuantizingTensorLayerCreator func(r io.Reader, name, dtype string, shape []int32, quantize bool) ([]LayerInfo, error)

// ShouldQuantize returns true if a tensor should be quantized.
// Quantizes linear weights only, skipping VAE, embeddings, norms, and biases.
func ShouldQuantize(name, component string) bool {
	if component == "vae" {
		return false
	}
	if strings.Contains(name, "embed") || strings.Contains(name, "norm") {
		return false
	}
	return strings.HasSuffix(name, ".weight")
}

package create

import (
	"encoding/json"
	"strings"
)

// olalaImportTransform quantizes only Olala's large matrices. The router,
// the small per-head attention projections, the Mamba-3 per-head parameters,
// norms and geodesic scalars keep source precision: OVHai reports NaN on
// realistic inputs with generic recipes that quantize them.
//
// With a 4-bit request only the routed experts take 4 bits and the other
// matrices take the 8-bit type of the same family (OVHai's q4e8 recipe):
// 4-bit dense projections lose too much quality.
type olalaImportTransform struct{}

func newOlalaImportTransform(json.RawMessage) (quantizePolicy, error) {
	return olalaImportTransform{}, nil
}

var olalaQuantizedSuffixes = []string{
	".mixer.in_proj.weight",
	".mixer_proj.weight",
	".mlp.in_cat.weight",
	".mlp.out_cat.weight",
}

var olalaExpertSuffixes = []string{
	".mlp.experts.experts.weight",
	".mlp.experts.output_experts.weight",
}

func (olalaImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if quantize == "" {
		return ""
	}
	quantNorm := normalizeQuantType(quantize)
	if name == "model.embedding.weight" || name == "lm_head.weight" {
		return promoteEmbedding(shape, quantNorm)
	}
	for _, suffix := range olalaExpertSuffixes {
		if strings.HasSuffix(name, suffix) {
			if !isAligned(shape, quantNorm) {
				return ""
			}
			return quantNorm
		}
	}
	for _, suffix := range olalaQuantizedSuffixes {
		if strings.HasSuffix(name, suffix) {
			q := eightBit(quantNorm)
			if !isAligned(shape, q) {
				return ""
			}
			return q
		}
	}
	return ""
}

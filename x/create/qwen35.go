package create

import (
	"encoding/json"
	"strings"
)

type qwen35ImportTransform struct{}

func newQwen35ImportTransform(json.RawMessage) (quantizePolicy, error) {
	return qwen35ImportTransform{}, nil
}

func (qwen35ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// The vision tower is not yet supported and the low-rank linear-attention
	// projections are sensitive; keep both at source precision. Everything else
	// follows the generic policy, which already keeps embeddings, norms, biases,
	// and routing gates unquantized.
	if isVisionTower(name) || qwen35IsLowRankProjection(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func qwen35IsLowRankProjection(name string) bool {
	return strings.HasSuffix(name, ".linear_attn.in_proj_a.weight") ||
		strings.HasSuffix(name, ".linear_attn.in_proj_b.weight") ||
		strings.HasSuffix(name, ".linear_attn.in_proj_ba.weight")
}

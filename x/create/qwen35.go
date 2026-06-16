package create

import "strings"

type qwen35ImportTransform struct{}

func newQwen35ImportTransform(_ string, _ sourceModelConfig) (quantizePolicy, error) {
	return qwen35ImportTransform{}, nil
}

func (qwen35ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// The vision tower is not yet supported and the low-rank linear-attention
	// projections are sensitive; keep both at source precision. Everything else
	// follows the generic policy, which already keeps embeddings, norms, biases,
	// and routing gates unquantized.
	if qwen35IsVisionTower(name) || qwen35IsLowRankProjection(name) {
		return ""
	}
	return GetTensorQuantization(name, shape, quantize)
}

func qwen35IsVisionTower(name string) bool {
	return strings.Contains(name, "vision_tower") || strings.Contains(name, ".visual.")
}

func qwen35IsLowRankProjection(name string) bool {
	return strings.HasSuffix(name, ".linear_attn.in_proj_a.weight") ||
		strings.HasSuffix(name, ".linear_attn.in_proj_b.weight") ||
		strings.HasSuffix(name, ".linear_attn.in_proj_ba.weight")
}

package create

import (
	"regexp"
	"strconv"
	"strings"
)

// defaultQuantPolicy is the quantize policy for any architecture without a
// registered override: the shared GetTensorQuantization decision with no
// architecture-specific adjustments.
type defaultQuantPolicy struct{}

func (defaultQuantPolicy) quantizationType(name string, shape []int32, quantize string) string {
	return GetTensorQuantization(name, shape, quantize)
}

// layerIndexRe extracts the layer index from tensor names like
// "model.language_model.layers.5.self_attn.v_proj.weight" or
// "model.language_model.layers.5.moe.experts.42.down_proj.weight"
var layerIndexRe = regexp.MustCompile(`\.layers\.(\d+)\.`)

// layerIndex returns the transformer layer index encoded in name, or -1.
func layerIndex(name string) int {
	m := layerIndexRe.FindStringSubmatch(name)
	if m == nil {
		return -1
	}
	idx, err := strconv.Atoi(m[1])
	if err != nil {
		return -1
	}
	return idx
}

// useMoreBits returns true for layers where quantization-sensitive tensors
// should use higher precision: the first and last 1/8 of layers (which handle
// input grounding and final output refinement), plus every 3rd layer in between
// to limit error accumulation through the residual stream.
func useMoreBits(layerIdx, numLayers int) bool {
	return layerIdx < numLayers/8 ||
		layerIdx >= 7*numLayers/8 ||
		(layerIdx-numLayers/8)%3 == 2
}

// eightBit returns the 8-bit quantization type in base's family: int8 for the
// affine family, mxfp8 for the fp4 family.
func eightBit(base string) string {
	if base == "int4" || base == "int8" {
		return "int8"
	}
	return "mxfp8"
}

// promoteEmbedding returns the 8-bit type in base's family when the embedding
// shape fits it, or "" when it does not. Token embeddings often double as the
// lm_head projection, where an 8-bit type keeps quality close to bf16 while
// saving decode bandwidth; the caller decides the fallback when 8-bit does not
// fit (the base type, or source precision).
func promoteEmbedding(shape []int32, base string) string {
	if e := eightBit(base); isAligned(shape, e) {
		return e
	}
	return ""
}

// sensitiveType resolves a quantization-sensitive projection (v/k/down): the
// 8-bit type in base's family when promote is set and fits the shape,
// otherwise the base type when it fits, otherwise source precision.
func sensitiveType(promote bool, shape []int32, base string) string {
	if promote {
		if e := eightBit(base); isAligned(shape, e) {
			return e
		}
	}
	if isAligned(shape, base) {
		return base
	}
	return ""
}

// isEmbedTokensWeight returns true for the main token embedding weight.
func isEmbedTokensWeight(name string) bool {
	return strings.HasSuffix(name, "embed_tokens.weight") &&
		!strings.Contains(name, "per_layer")
}

// isVisionTower reports tensors under a model's vision tower.
func isVisionTower(name string) bool {
	return strings.Contains(name, "vision_tower") || strings.Contains(name, ".visual.")
}

// isAudioTower reports tensors under a model's audio tower or audio embedding.
func isAudioTower(name string) bool {
	return strings.Contains(name, "audio_tower") || strings.Contains(name, "embed_audio")
}

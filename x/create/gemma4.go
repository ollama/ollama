package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

type gemma4ImportTransform struct {
	numLayers  int
	numExperts int
}

// gemma4Config is a minimal subset of the Gemma 4 config.json used for quant decisions.
type gemma4Config struct {
	NumHiddenLayers int `json:"num_hidden_layers"`
	NumExperts      int `json:"num_experts"`
	TextConfig      struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
		NumExperts      int `json:"num_experts"`
	} `json:"text_config"`
}

func newGemma4ImportTransform(modelDir string, _ sourceModelConfig) (quantizePolicy, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return gemma4ImportTransform{}, nil //nolint:nilerr // fallback to no heuristic
	}
	var cfg gemma4Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return gemma4ImportTransform{}, nil //nolint:nilerr // fallback to no heuristic
	}

	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.TextConfig.NumHiddenLayers
	}
	numExperts := cfg.NumExperts
	if numExperts == 0 {
		numExperts = cfg.TextConfig.NumExperts
	}

	return gemma4ImportTransform{numLayers: numLayers, numExperts: numExperts}, nil
}

// layerIndexRe extracts the layer index from tensor names like
// "model.language_model.layers.5.self_attn.v_proj.weight" or
// "model.language_model.layers.5.moe.experts.42.down_proj.weight"
var layerIndexRe = regexp.MustCompile(`\.layers\.(\d+)\.`)

// useMoreBits returns true for layers where quantization-sensitive tensors
// should use higher precision: the first and last 1/8 of layers (which handle
// input grounding and final output refinement), plus every 3rd layer in between
// to limit error accumulation through the residual stream.
func useMoreBits(layerIdx, numLayers int) bool {
	return layerIdx < numLayers/8 ||
		layerIdx >= 7*numLayers/8 ||
		(layerIdx-numLayers/8)%3 == 2
}

func (t gemma4ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	switch {
	case isEmbedTokensWeight(name):
		// The embedding doubles as the lm_head projection; an 8-bit type keeps
		// quality close to bf16 (matching GGUF Q6_K) while saving bandwidth.
		return gemma4EmbeddingType(shape, base)
	case t.isSensitiveProjection(name) && eightBit(base) != base:
		return t.sensitiveProjectionType(name, shape, base)
	default:
		// Routing gates, norms, embeddings, etc. are handled by the generic
		// policy; everything else quantizes at the requested type.
		return GetTensorQuantization(name, shape, quantize)
	}
}

// eightBit returns the 8-bit quantization type in base's family: int8 for the
// affine family, mxfp8 for the fp4 family.
func eightBit(base string) string {
	if base == "int4" || base == "int8" {
		return "int8"
	}
	return "mxfp8"
}

// gemma4EmbeddingType quantizes the token embedding to an 8-bit type, or keeps
// it at source precision when no aligned type fits.
func gemma4EmbeddingType(shape []int32, base string) string {
	if e := eightBit(base); isAligned(shape, e) {
		return e
	}
	if isAligned(shape, base) {
		return base
	}
	return ""
}

// isSensitiveProjection reports the value/key/down projections whose precision
// most affects quality — attention output (v/k) and the residual stream
// (down). Audio and vision tower tensors are excluded and follow the generic
// policy.
func (t gemma4ImportTransform) isSensitiveProjection(name string) bool {
	if strings.Contains(name, "audio_tower") || strings.Contains(name, "vision_tower") {
		return false
	}
	return strings.Contains(name, ".v_proj") ||
		strings.Contains(name, ".k_proj") ||
		strings.Contains(name, "down_proj")
}

// sensitiveProjectionType uses 8-bit precision at sensitive layers, otherwise
// the requested type, or source precision when neither fits the tensor shape.
func (t gemma4ImportTransform) sensitiveProjectionType(name string, shape []int32, base string) string {
	if t.promoteSensitive(name) {
		if e := eightBit(base); isAligned(shape, e) {
			return e
		}
	}
	if isAligned(shape, base) {
		return base
	}
	return ""
}

// promoteSensitive decides whether a sensitive projection uses 8-bit precision.
// 8-expert models share very few KV heads, so their k/v projections are always
// promoted; otherwise v/down projections are promoted at the input and output
// layers and periodically between (useMoreBits), where residual-stream error
// accumulates most.
func (t gemma4ImportTransform) promoteSensitive(name string) bool {
	if t.numLayers == 0 {
		return false
	}
	if t.numExperts == 8 && (strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj")) {
		return true
	}
	if strings.Contains(name, ".k_proj") {
		return false // k_proj is promoted only via the 8-expert path
	}
	layer := gemma4LayerIndex(name)
	return layer >= 0 && useMoreBits(layer, t.numLayers)
}

// gemma4LayerIndex returns the transformer layer index encoded in name, or -1.
func gemma4LayerIndex(name string) int {
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

// isEmbedTokensWeight returns true for the main token embedding weight.
func isEmbedTokensWeight(name string) bool {
	return strings.HasSuffix(name, "embed_tokens.weight") &&
		!strings.Contains(name, "per_layer")
}

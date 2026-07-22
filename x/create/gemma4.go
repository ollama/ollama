package create

import (
	"encoding/json"
	"fmt"
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

func newGemma4ImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg gemma4Config
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("gemma4: parse config.json: %w", err)
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

func (t gemma4ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	switch {
	case isEmbedTokensWeight(name):
		// The embedding doubles as the lm_head projection; an 8-bit type keeps
		// quality close to bf16 (matching GGUF Q6_K) while saving bandwidth.
		// Fall back to the base type when 8-bit does not fit the vocab shape.
		if e := promoteEmbedding(shape, base); e != "" {
			return e
		}
		if isAligned(shape, base) {
			return base
		}
		return ""
	case t.isSensitiveProjection(name) && eightBit(base) != base:
		return sensitiveType(t.promoteSensitive(name), shape, base)
	default:
		// Routing gates, norms, embeddings, etc. are handled by the generic
		// policy; everything else quantizes at the requested type.
		return GetTensorQuantization(name, shape, quantize)
	}
}

// isSensitiveProjection reports the value/key/down projections whose precision
// most affects quality — attention output (v/k) and the residual stream
// (down). Audio and vision tower tensors are excluded and follow the generic
// policy.
func (t gemma4ImportTransform) isSensitiveProjection(name string) bool {
	if isVisionTower(name) || isAudioTower(name) {
		return false
	}
	return strings.Contains(name, ".v_proj") ||
		strings.Contains(name, ".k_proj") ||
		strings.Contains(name, "down_proj")
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
	layer := layerIndex(name)
	return layer >= 0 && useMoreBits(layer, t.numLayers)
}

package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type glimmerImportTransform struct {
	numLayers int
}

type glimmerConfig struct {
	NumHiddenLayers int `json:"num_hidden_layers"`
	TextConfig      struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
	} `json:"text_config"`
}

func newGlimmerImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg glimmerConfig
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("glimmer: parse config.json: %w", err)
	}
	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.TextConfig.NumHiddenLayers
	}
	return glimmerImportTransform{numLayers: numLayers}, nil
}

func (t glimmerImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	// Preserve vision tensors at source precision so image text and fine detail
	// are not degraded by the language model's quantization policy.
	if isGlimmerVisionTensor(name) {
		return ""
	}

	base := normalizeQuantType(quantize)
	if isEmbedTokensWeight(name) {
		if e := promoteEmbedding(shape, base); e != "" {
			return e
		}
		if isAligned(shape, base) {
			return base
		}
		return ""
	}

	if isGlimmerSensitiveProjection(name) && eightBit(base) != base {
		return sensitiveType(t.promoteSensitive(name), shape, base)
	}

	return GetTensorQuantization(name, shape, quantize)
}

func isGlimmerVisionTensor(name string) bool {
	return isVision(name)
}

func isGlimmerSensitiveProjection(name string) bool {
	return strings.Contains(name, ".self_attn.q_proj") ||
		strings.Contains(name, ".self_attn.o_proj") ||
		strings.Contains(name, ".self_attn.k_proj") ||
		strings.Contains(name, ".self_attn.v_proj") ||
		strings.Contains(name, ".self_attn.gate_proj") ||
		strings.Contains(name, ".self_attn.output_gate_proj") ||
		strings.Contains(name, ".mlp.down_proj")
}

func (t glimmerImportTransform) promoteSensitive(name string) bool {
	if strings.Contains(name, ".self_attn.q_proj") ||
		strings.Contains(name, ".self_attn.o_proj") ||
		strings.Contains(name, ".self_attn.k_proj") ||
		strings.Contains(name, ".self_attn.v_proj") {
		return true
	}
	layer := layerIndex(name)
	return t.numLayers > 0 && layer >= 0 && useMoreBits(layer, t.numLayers)
}

package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type nemotronHImportTransform struct {
	numLayers int
}

func newNemotronHImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
		LLMConfig       struct {
			NumHiddenLayers int `json:"num_hidden_layers"`
		} `json:"llm_config"`
	}
	if err := json.Unmarshal(rawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("nemotron_h: parse config.json: %w", err)
	}
	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = cfg.LLMConfig.NumHiddenLayers
	}
	return nemotronHImportTransform{numLayers: numLayers}, nil
}

// Nemotron's modality tower names do not match the shared predicates.
func nemotronHIsVisionTower(name string) bool {
	return strings.HasPrefix(name, "vision_model.") ||
		strings.HasPrefix(name, "mlp1.")
}

func nemotronHIsAudioTower(name string) bool {
	return strings.HasPrefix(name, "sound_encoder.") ||
		strings.HasPrefix(name, "sound_projection.")
}

func nemotronHShouldKeepBF16ForDirectNonAffine(name string) bool {
	switch {
	case strings.HasSuffix(name, ".mixer.gate.weight"):
		return true
	case strings.HasSuffix(name, ".mixer.conv1d.weight"):
		return true
	default:
		return false
	}
}

func nemotronHIsAttentionProjection(name string) bool {
	return strings.HasSuffix(name, ".mixer.q_proj.weight") ||
		strings.HasSuffix(name, ".mixer.k_proj.weight") ||
		strings.HasSuffix(name, ".mixer.v_proj.weight") ||
		strings.HasSuffix(name, ".mixer.o_proj.weight")
}

// promoteSensitive reports whether a sensitive tensor takes the 8-bit type.
// Attention always does: few layers carry it, 4-bit attention degrades
// structured output, and promoting all of it is free. Experts keep the
// schedule, where the decode bandwidth saving is real.
func (t nemotronHImportTransform) promoteSensitive(name string) bool {
	if nemotronHIsAttentionProjection(name) {
		return true
	}
	layerIdx := layerIndex(name)
	return layerIdx < 0 || useMoreBits(layerIdx, t.numLayers)
}

func (t nemotronHImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if nemotronHIsVisionTower(name) || nemotronHIsAudioTower(name) || nemotronHShouldKeepBF16ForDirectNonAffine(name) {
		return ""
	}

	quantNorm := normalizeQuantType(quantize)

	// lm_head and token embeddings are sensitive but high-bandwidth;
	// promote them to 8-bit in the requested quant family when the
	// shape fits, otherwise keep them at source precision.
	if strings.HasSuffix(name, "embeddings.weight") || strings.HasSuffix(name, "lm_head.weight") {
		return promoteEmbedding(shape, quantNorm)
	}

	if quantNorm == "nvfp4" || quantNorm == "mxfp4" {
		isSensitive := nemotronHIsAttentionProjection(name) ||
			strings.HasSuffix(name, ".mixer.out_proj.weight") ||
			strings.HasSuffix(name, ".mixer.down_proj.weight") ||
			strings.Contains(name, ".mixer.experts.") && strings.HasSuffix(name, ".down_proj.weight") ||
			strings.HasSuffix(name, ".mixer.shared_experts.down_proj.weight")
		if isSensitive {
			if isAligned(shape, "mxfp8") && t.promoteSensitive(name) {
				return "mxfp8"
			}
			if isAligned(shape, quantNorm) {
				return quantNorm
			}
			return ""
		}
	}

	return GetTensorQuantization(name, shape, quantize)
}

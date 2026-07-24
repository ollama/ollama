package create

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
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

func nemotronHIsUnsupportedModalityTensor(name string) bool {
	return strings.HasPrefix(name, "vision_model.") ||
		strings.HasPrefix(name, "mlp1.") ||
		strings.HasPrefix(name, "sound_encoder.") ||
		strings.HasPrefix(name, "sound_projection.")
}

func nemotronHShouldKeepBF16ForDirectNonAffine(name string) bool {
	switch {
	case strings.HasSuffix(name, "embeddings.weight"):
		return true
	case strings.HasSuffix(name, "lm_head.weight"):
		return true
	case strings.HasSuffix(name, ".mixer.gate.weight"):
		return true
	case strings.HasSuffix(name, ".mixer.conv1d.weight"):
		return true
	default:
		return false
	}
}

var nemotronHLayerIndexRe = regexp.MustCompile(`\.layers\.(\d+)\.`)

func nemotronHUseMoreBits(layerIdx, numLayers int) bool {
	if numLayers <= 0 {
		return true
	}
	return layerIdx < numLayers/8 ||
		layerIdx >= 7*numLayers/8 ||
		(layerIdx-numLayers/8)%3 == 2
}

func (t nemotronHImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if nemotronHIsUnsupportedModalityTensor(name) || nemotronHShouldKeepBF16ForDirectNonAffine(name) {
		return ""
	}

	quantNorm := normalizeQuantType(quantize)
	if quantNorm == "nvfp4" || quantNorm == "mxfp4" {
		layerIdx := -1
		if m := nemotronHLayerIndexRe.FindStringSubmatch(name); m != nil {
			if idx, err := strconv.Atoi(m[1]); err == nil {
				layerIdx = idx
			}
		}

		isSensitive := strings.HasSuffix(name, ".mixer.out_proj.weight") ||
			strings.HasSuffix(name, ".mixer.o_proj.weight") ||
			strings.HasSuffix(name, ".mixer.v_proj.weight") ||
			strings.HasSuffix(name, ".mixer.down_proj.weight") ||
			strings.Contains(name, ".mixer.experts.") && strings.HasSuffix(name, ".down_proj.weight") ||
			strings.HasSuffix(name, ".mixer.shared_experts.down_proj.weight")
		if isSensitive {
			if isAligned(shape, "mxfp8") && (layerIdx < 0 || nemotronHUseMoreBits(layerIdx, t.numLayers)) {
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

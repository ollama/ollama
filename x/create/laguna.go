package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type lagunaImportTransform struct {
	denseMLPLayers map[int]bool
	numLayers      int
}

type lagunaConfig struct {
	NumHiddenLayers int      `json:"num_hidden_layers"`
	MLPOnlyLayers   []int    `json:"mlp_only_layers"`
	MLPLayerTypes   []string `json:"mlp_layer_types"`
}

func newLagunaImportTransform(rawConfig json.RawMessage) (quantizePolicy, error) {
	var cfg lagunaConfig
	if len(rawConfig) > 0 {
		if err := json.Unmarshal(rawConfig, &cfg); err != nil {
			return nil, fmt.Errorf("laguna: parse config.json: %w", err)
		}
	}

	denseLayers := make(map[int]bool)
	for i, typ := range cfg.MLPLayerTypes {
		if typ == "dense" {
			denseLayers[i] = true
		}
	}
	for _, layer := range cfg.MLPOnlyLayers {
		denseLayers[layer] = true
	}
	if len(denseLayers) == 0 {
		denseLayers[0] = true
	}

	numLayers := cfg.NumHiddenLayers
	if numLayers == 0 {
		numLayers = len(cfg.MLPLayerTypes)
	}
	if numLayers == 0 {
		numLayers = 40
	}

	return lagunaImportTransform{
		denseMLPLayers: denseLayers,
		numLayers:      numLayers,
	}, nil
}

func (t lagunaImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	if !lagunaFPQuant(base) {
		return GetTensorQuantization(name, shape, quantize)
	}

	// Laguna keeps the tied output head and router at source precision.
	switch {
	case strings.HasSuffix(name, "lm_head.weight") || strings.HasSuffix(name, ".mlp.gate.weight"):
		return ""
	case base == "mxfp8" && lagunaAttentionProjection(name):
		return ""
	case lagunaAttentionProjection(name):
		return lagunaQuantizationType(name, shape, base)
	case lagunaDenseMLPProjection(name) && t.denseMLPLayers[layerIndex(name)]:
		return lagunaQuantizationType(name, shape, base)
	case base == "mxfp8" && lagunaRoutedExpertDownProjection(name):
		if lagunaPromoteExpertDown(layerIndex(name), t.numLayers) {
			return ""
		}
		return lagunaQuantizationType(name, shape, base)
	case lagunaSharedExpertDownProjection(name):
		return lagunaSensitiveType(lagunaPromoteExpertDown(layerIndex(name), t.numLayers), name, shape, base)
	case lagunaSharedExpertProjection(name):
		return lagunaQuantizationType(name, shape, base)
	case lagunaRoutedExpertProjection(name):
		return lagunaQuantizationType(name, shape, base)
	default:
		return ""
	}
}

func lagunaFPQuant(quantize string) bool {
	return quantize == "nvfp4" || quantize == "mxfp4" || quantize == "mxfp8"
}

func lagunaAttentionProjection(name string) bool {
	return strings.Contains(name, ".self_attn.q_proj.weight") ||
		strings.Contains(name, ".self_attn.k_proj.weight") ||
		strings.Contains(name, ".self_attn.v_proj.weight") ||
		strings.Contains(name, ".self_attn.o_proj.weight") ||
		strings.Contains(name, ".self_attn.g_proj.weight")
}

func lagunaDenseMLPProjection(name string) bool {
	return strings.Contains(name, ".mlp.gate_proj.weight") ||
		strings.Contains(name, ".mlp.up_proj.weight") ||
		strings.Contains(name, ".mlp.down_proj.weight")
}

func lagunaRoutedExpertProjection(name string) bool {
	if !lagunaMLPProjectionWeight(name) {
		return false
	}
	return strings.Contains(name, ".mlp.experts.")
}

func lagunaRoutedExpertDownProjection(name string) bool {
	return strings.Contains(name, ".mlp.experts.") && strings.HasSuffix(name, ".down_proj.weight")
}

func lagunaSharedExpertProjection(name string) bool {
	if !lagunaMLPProjectionWeight(name) {
		return false
	}
	return strings.Contains(name, ".mlp.shared_expert.")
}

func lagunaSharedExpertDownProjection(name string) bool {
	return strings.Contains(name, ".mlp.shared_expert.down_proj.weight")
}

// Laguna XS 2 and 2.1 are sensitive to fully quantizing expert down
// projections. Keep the same cadence for both: use higher precision on the
// input-side layers, final layers, and a sparse cadence early in the residual
// stream. For 4-bit fp quants that higher precision is mxfp8. For mxfp8, the
// shared expert down projections stay at source precision because the tensor
// class is small; routed expert down projections use the cadence to avoid
// pushing the model too close to bf16 size.
func lagunaPromoteExpertDown(layerIdx, numLayers int) bool {
	return useMoreBitsWithMiddleEnd(layerIdx, numLayers, numLayers/2-4)
}

func lagunaMLPProjectionWeight(name string) bool {
	return strings.HasSuffix(name, ".gate_proj.weight") ||
		strings.HasSuffix(name, ".up_proj.weight") ||
		strings.HasSuffix(name, ".down_proj.weight")
}

func lagunaQuantizationType(name string, shape []int32, quantize string) string {
	q := GetTensorQuantization(name, shape, quantize)
	// Laguna's architecture policy decides which sensitive tensors to
	// promote. Undo the generic policy's blanket 4-to-8-bit promotion here.
	if q != quantize && q == eightBit(quantize) {
		return quantize
	}
	return q
}

func lagunaSensitiveType(promote bool, name string, shape []int32, quantize string) string {
	if quantize == "mxfp8" {
		return ""
	}
	if promote {
		return GetTensorQuantization(name, shape, quantize)
	}
	return lagunaQuantizationType(name, shape, quantize)
}

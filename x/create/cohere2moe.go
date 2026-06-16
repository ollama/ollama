package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// cohere2MoeImportTransform adjusts quantization for Cohere2 MoE imports
// (Command A family / North models).
type cohere2MoeImportTransform struct {
	numLayers int
}

func newCohere2MoeImportTransform(modelDir string, _ sourceModelConfig) (quantizePolicy, error) {
	data, err := os.ReadFile(filepath.Join(modelDir, "config.json"))
	if err != nil {
		return cohere2MoeImportTransform{}, nil //nolint:nilerr // fallback to no heuristic
	}
	var cfg struct {
		NumHiddenLayers int `json:"num_hidden_layers"`
	}
	if err := json.Unmarshal(data, &cfg); err != nil {
		return cohere2MoeImportTransform{}, nil //nolint:nilerr // fallback to no heuristic
	}
	return cohere2MoeImportTransform{numLayers: cfg.NumHiddenLayers}, nil
}

var cohere2MoeLayerIndexRe = regexp.MustCompile(`\.layers\.(\d+)\.`)

func (t cohere2MoeImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	quantNorm := normalizeQuantType(quantize)

	// The embedding serves double duty: lookup (via QuantizedEmbedding) and the
	// tied lm_head projection (via AsLinear). With a 262k vocab the bf16
	// embedding dominates decode bandwidth through the lm_head matmul, so
	// quantize it to the 8-bit variant of the requested mode.
	if strings.HasSuffix(name, "embed_tokens.weight") && len(shape) == 2 {
		switch quantNorm {
		case "int4", "int8":
			if isAligned(shape, "int8") {
				return "int8"
			}
		case "mxfp4", "nvfp4", "mxfp8":
			if isAligned(shape, "mxfp8") {
				return "mxfp8"
			}
		}
		return ""
	}

	// The MoE router picks the top-k expert set; quantization noise there can
	// flip expert selection and compound downstream. It is tiny, so keep it in
	// source precision. (GetTensorQuantization already skips "mlp.gate.weight";
	// kept explicit here so renames in the default policy cannot regress this.)
	if strings.HasSuffix(name, ".mlp.gate.weight") {
		return ""
	}

	// Sensitive tensors (v_proj, k_proj, down_proj) get higher precision only
	// at quantization-sensitive layer positions (gemma4's useMoreBits
	// heuristic) instead of the default policy's blanket promotion. The
	// blanket int8 down_proj costs ~25% of decode bandwidth on a top-8 MoE;
	// the layer-position heuristic keeps the early/late layers (and every
	// third in between) at 8 bits where residual-stream error matters most.
	promote := ""
	switch quantNorm {
	case "int4":
		promote = "int8"
	case "mxfp4", "nvfp4":
		promote = "mxfp8"
	}
	isSensitive := strings.Contains(name, ".v_proj") || strings.Contains(name, ".k_proj") || strings.Contains(name, "down_proj")
	if promote != "" && isSensitive && t.numLayers > 0 {
		layerIdx := -1
		if m := cohere2MoeLayerIndexRe.FindStringSubmatch(name); m != nil {
			if idx, err := strconv.Atoi(m[1]); err == nil {
				layerIdx = idx
			}
		}
		if layerIdx >= 0 {
			if useMoreBits(layerIdx, t.numLayers) && isAligned(shape, promote) {
				return promote
			}
			if !isAligned(shape, quantNorm) {
				return ""
			}
			// Bypass GetTensorQuantization's blanket promotion — the
			// layer-position heuristic is authoritative here.
			return quantNorm
		}
	}

	return GetTensorQuantization(name, shape, quantize)
}

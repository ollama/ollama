package create

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
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

func newGemma4ImportTransform(modelDir string, _ sourceModelConfig) (tensorImportTransform, error) {
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

func (t gemma4ImportTransform) skipTensor(name string) bool {
	return false
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
	quantNorm := normalizeQuantType(quantize)

	// Embedding: quantize to 8-bit variant for bandwidth efficiency.
	// The embedding serves double duty: lookup (via QuantizedEmbedding) and
	// lm_head projection (via AsLinear). Using 8-bit matches GGUF Q6_K quality
	// (strictly higher at 8 bpw vs 6.5 bpw) while saving ~2.8 GB on 31B vs bf16.
	if isEmbedTokensWeight(name) {
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
		if isAligned(shape, quantNorm) {
			return quantNorm
		}
		return ""
	}

	// MoE router logits choose the top-k expert set. Quantization noise here
	// can flip expert selection, after which downstream activations diverge
	// sharply. The tensor is small, so leave it in source precision.
	if isGemma4RouterProjection(name) {
		return ""
	}

	// Mixed-precision quantization: sensitive tensors get higher precision.
	//
	// Value projections (v_proj) directly determine attention output quality.
	// Down projections (down_proj) are the final MLP output and errors there
	// propagate directly to the residual stream. Both benefit from higher
	// precision at early layers, late layers, and periodically in between
	// (the "useMoreBits" heuristic).
	//
	// For int4:        promote → int8 (same affine family, GatherQMM compatible).
	// For mxfp4/nvfp4: promote → mxfp8. MLX quantized_matmul handles mixed
	//   nvfp4+mxfp8 modes within the same model — each tensor carries its own
	//   quant metadata and the kernel dispatches per-tensor.
	if t.numLayers > 0 {
		layerIdx := -1
		if m := layerIndexRe.FindStringSubmatch(name); m != nil {
			if idx, err := strconv.Atoi(m[1]); err == nil {
				layerIdx = idx
			}
		}

		// Determine promotion target for sensitive tensors.
		// "int8"  = int4 base → int8 (affine family)
		// "mxfp8" = mxfp4/nvfp4 base → mxfp8
		// ""      = no promotion (int8/mxfp8, already 8-bit)
		promote := ""
		switch quantNorm {
		case "int4":
			promote = "int8"
		case "mxfp4", "nvfp4":
			promote = "mxfp8"
		}

		// Only apply to language model tensors — audio/vision tower tensors
		// should pass through to GetTensorQuantization which skips them.
		isModelTensor := !strings.Contains(name, "audio_tower") &&
			!strings.Contains(name, "vision_tower")
		isSensitive := isModelTensor &&
			(strings.Contains(name, ".v_proj") || strings.Contains(name, "down_proj"))
		isSensitiveK := isModelTensor && strings.Contains(name, "k_proj")

		if promote != "" && (isSensitive || isSensitiveK) {
			shouldPromote := false

			// 8-expert models: v_proj and k_proj share very few KV heads,
			// so quantization errors are amplified. Always promote.
			if t.numExperts == 8 && (strings.Contains(name, ".v_proj") || isSensitiveK) {
				shouldPromote = true
			}

			// Layer-position heuristic for v_proj and down_proj.
			if isSensitive && layerIdx >= 0 && useMoreBits(layerIdx, t.numLayers) {
				shouldPromote = true
			}

			if shouldPromote && isAligned(shape, promote) {
				return promote
			}

			// Sensitive tensor at a non-promoted layer: use base quant type.
			// Return directly to bypass GetTensorQuantization's uniform
			// promotion — the layer-position heuristic is authoritative here.
			if !isAligned(shape, quantNorm) {
				return ""
			}
			return quantNorm
		}
	}

	return GetTensorQuantization(name, shape, quantize)
}

// isEmbedTokensWeight returns true for the main token embedding weight.
func isEmbedTokensWeight(name string) bool {
	return strings.HasSuffix(name, "embed_tokens.weight") &&
		!strings.Contains(name, "per_layer")
}

func isGemma4RouterProjection(name string) bool {
	return strings.HasSuffix(name, ".router.proj.weight") &&
		!strings.Contains(name, "audio_tower") &&
		!strings.Contains(name, "vision_tower")
}

func (t gemma4ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	if newName, ok := canonicalGemma4MoEName(td.Name, td.Shape); ok {
		return []*safetensors.TensorData{td.WithName(newName)}, nil
	}

	return []*safetensors.TensorData{td}, nil
}

// canonicalGemma4MoEName rewrites pre-stacked Gemma 4 MoE expert tensors to the
// on-disk name expected by the runtime, preserving the [experts, ..., ...]
// shape:
//
//	<layerPrefix>.moe.switch_mlp.<proj>.weight
//
// Source layouts handled:
//   - "<layerPrefix>.moe.{gate,up,down}_proj[.weight]"     (older split form)
//   - "<layerPrefix>.experts.{gate_up,down}_proj[.weight]" (newer fused form)
//
// gate_up_proj stays fused; the runtime splits axis 1 at load time.
//
// The ".moe." segment is kept for backwards compatibility with published gemma4
// models using incorrect names. Do not copy this pattern for new models; use
// ".mlp.switch_mlp.<proj>.weight"
func canonicalGemma4MoEName(name string, shape []int32) (string, bool) {
	if len(shape) != 3 {
		return "", false
	}
	parts := strings.Split(strings.TrimSuffix(name, ".weight"), ".")
	if len(parts) < 3 {
		return "", false
	}

	proj := parts[len(parts)-1]
	if !strings.HasSuffix(proj, "_proj") {
		return "", false
	}
	switch parts[len(parts)-2] {
	case "moe", "experts":
	default:
		return "", false
	}
	prefix := strings.Join(parts[:len(parts)-2], ".")
	return prefix + ".moe.switch_mlp." + proj + ".weight", true
}

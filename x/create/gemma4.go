package create

import (
	"encoding/json"
	"fmt"
	"io"
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

func (t gemma4ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	// Split pre-stacked MoE expert tensors [N, out, in] into per-expert
	// [out, in] tensors so they go through the standard expert packing and
	// quantization flow (ExpertGroupPrefix matching, per-expert quantize).
	if isGemma4StackedMoETensor(td.Name, td.Shape) {
		return splitStackedMoETensor(td)
	}

	return []*safetensors.TensorData{td}, nil
}

// isGemma4StackedMoETensor checks if this is a pre-stacked MoE expert weight.
// Gemma 4 HF weights come in two layouts depending on the model version:
//   - Older: model.language_model.layers.N.moe.{gate,up,down}_proj [experts, dim1, dim2]
//   - Newer: model.language_model.layers.N.experts.{gate_up,down}_proj [experts, dim1, dim2]
//
// The newer layout has gate+up already fused. We keep it fused (no splitting)
// so the tensors flow through the standard expert packing and quantization path.
func isGemma4StackedMoETensor(name string, shape []int32) bool {
	if len(shape) != 3 {
		return false
	}
	if strings.Contains(name, ".moe.") || strings.Contains(name, ".experts.") {
		return strings.HasSuffix(name, "_proj") || strings.HasSuffix(name, "_proj.weight")
	}
	return false
}

// splitStackedMoETensor splits a [N, out, in] stacked expert tensor into
// N individual [out, in] tensors named with the per-expert convention that
// ExpertGroupPrefix expects: prefix.moe.experts.{E}.{proj}.weight
func splitStackedMoETensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor %s: %w", td.Name, err)
	}

	numExperts := int(td.Shape[0])
	rows := int(td.Shape[1]) // out_features in HF layout
	cols := int(td.Shape[2]) // in_features in HF layout

	elemSize, err := DTypeSize(td.Dtype)
	if err != nil {
		return nil, fmt.Errorf("failed to get dtype size for %s: %w", td.Dtype, err)
	}

	perExpertBytes := rows * cols * elemSize
	if len(raw) != numExperts*perExpertBytes {
		return nil, fmt.Errorf("tensor %s: raw byte length %d does not match shape %v and dtype %s",
			td.Name, len(raw), td.Shape, td.Dtype)
	}

	// Determine the per-expert name pattern.
	// Two source layouts:
	//   Old: model.language_model.layers.N.moe.gate_proj
	//     -> model.language_model.layers.N.moe.experts.E.gate_proj.weight
	//   New: model.language_model.layers.N.experts.gate_up_proj
	//     -> model.language_model.layers.N.moe.experts.E.gate_up_proj.weight
	baseName := td.Name
	baseName = strings.TrimSuffix(baseName, ".weight")
	lastDot := strings.LastIndex(baseName, ".")
	if lastDot < 0 {
		return nil, fmt.Errorf("tensor %s: unexpected name format", td.Name)
	}
	parentPrefix := baseName[:lastDot] // "...layers.N.moe" or "...layers.N.experts"
	projName := baseName[lastDot+1:]   // "gate_proj" or "gate_up_proj"

	// Normalize: if parent already ends with ".experts", use the grandparent + ".moe"
	// so we get a consistent "layers.N.moe.experts.E" pattern.
	var moePrefix string
	if cut, ok := strings.CutSuffix(parentPrefix, ".experts"); ok {
		moePrefix = cut + ".moe"
	} else {
		moePrefix = parentPrefix
	}

	transposedShape := []int32{td.Shape[1], td.Shape[2]}

	results := make([]*safetensors.TensorData, numExperts)
	for e := range numExperts {
		expertName := fmt.Sprintf("%s.experts.%d.%s.weight", moePrefix, e, projName)
		start := e * perExpertBytes
		end := start + perExpertBytes
		results[e] = safetensors.NewTensorDataFromBytes(expertName, td.Dtype, transposedShape, raw[start:end])
	}

	return results, nil
}

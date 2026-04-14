package create

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/x/safetensors"
)

type qwen35ImportTransform struct {
	shouldShiftNormWeights bool
	rewriteLanguageModel   bool
}

type qwen35SourceInfo struct {
	hasPrequantizedWeights bool
	shouldShiftNormWeights bool
}

func newQwen35ImportTransform(modelDir string, cfg sourceModelConfig) (tensorImportTransform, error) {
	sourceInfo, err := qwen35InspectSource(modelDir)
	if err != nil {
		return qwen35ImportTransform{}, err
	}
	if sourceInfo.hasPrequantizedWeights {
		return noopImportTransform{}, nil
	}

	return qwen35ImportTransform{
		shouldShiftNormWeights: sourceInfo.shouldShiftNormWeights,
		rewriteLanguageModel:   strings.Contains(cfg.Architecture(), "ConditionalGeneration"),
	}, nil
}

func qwen35InspectSource(modelDir string) (qwen35SourceInfo, error) {
	entries, err := os.ReadDir(modelDir)
	if err != nil {
		return qwen35SourceInfo{}, err
	}

	var info qwen35SourceInfo
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".safetensors") {
			continue
		}

		extractor, err := safetensors.OpenForExtraction(filepath.Join(modelDir, entry.Name()))
		if err != nil {
			return qwen35SourceInfo{}, err
		}

		for _, name := range extractor.ListTensors() {
			if strings.HasSuffix(name, ".scales") {
				extractor.Close()
				info.hasPrequantizedWeights = true
				return info, nil
			}
			// This should change when MTP is supported
			if strings.Contains(name, "mtp.") {
				info.shouldShiftNormWeights = true
				continue
			}
			if info.shouldShiftNormWeights || !strings.Contains(name, "conv1d.weight") {
				continue
			}

			td, err := extractor.GetTensor(name)
			if err != nil {
				extractor.Close()
				return qwen35SourceInfo{}, err
			}
			if len(td.Shape) == 3 && td.Shape[2] != 1 {
				info.shouldShiftNormWeights = true
			}
		}

		extractor.Close()
	}

	return info, nil
}

func (t qwen35ImportTransform) skipTensor(name string) bool {
	return strings.Contains(name, "mtp.")
}

func qwen35ShouldKeepBF16ForDirectNonAffine(name string) bool {
	switch {
	case strings.HasSuffix(name, "embed_tokens.weight"):
		return true
	case strings.HasSuffix(name, "lm_head.weight"):
		return true
	case strings.HasSuffix(name, ".linear_attn.in_proj_a.weight"):
		return true
	case strings.HasSuffix(name, ".linear_attn.in_proj_b.weight"):
		return true
	case strings.HasSuffix(name, ".linear_attn.in_proj_ba.weight"):
		return true
	case strings.HasSuffix(name, ".mlp.gate.weight") && !strings.Contains(name, "_proj"):
		return true
	case strings.HasSuffix(name, ".mlp.shared_expert_gate.weight"):
		return true
	default:
		return false
	}
}

func (t qwen35ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	if strings.HasPrefix(name, "vision_tower.") {
		return ""
	}

	stackedExpert := isStackedExpertWeight(name)
	if strings.HasSuffix(name, ".bias") || strings.HasSuffix(name, ".scale") || strings.HasSuffix(name, ".qbias") ||
		strings.HasSuffix(name, ".biases") || strings.HasSuffix(name, ".scales") {
		return ""
	}
	if !stackedExpert && !strings.HasSuffix(name, ".weight") {
		return ""
	}
	if strings.Contains(name, "norm") || strings.Contains(name, "ln_") || strings.Contains(name, "layernorm") {
		return ""
	}
	if len(shape) != 2 && !(len(shape) == 3 && stackedExpert) {
		return ""
	}

	var elems int64 = 1
	for _, d := range shape {
		elems *= int64(d)
	}
	if elems < 1024 {
		return ""
	}

	quantNorm := normalizeQuantType(quantize)
	groupSize := int32(32)
	switch quantNorm {
	case "nvfp4":
		groupSize = 16
	case "int4", "int8":
		groupSize = 64
	}
	if shape[len(shape)-1]%groupSize != 0 {
		return ""
	}

	// Match the working HF-FP8 import policy for direct NVFP4/MXFP4/MXFP8 imports:
	// keep embeddings, LM head, low-rank linear_attn projections, and routing
	// gates in BF16 rather than forcing them into a non-affine quantized format.
	if (quantNorm == "nvfp4" || quantNorm == "mxfp4" || quantNorm == "mxfp8") && qwen35ShouldKeepBF16ForDirectNonAffine(name) {
		return ""
	}

	return quantNorm
}

func (t qwen35ImportTransform) rewriteTensorData(td *safetensors.TensorData) (*safetensors.TensorData, error) {
	if td == nil {
		return td, nil
	}

	shiftNorm := t.shouldShiftNormWeights && qwen35ShouldShiftNormKey(td.Name)
	transposeConv := strings.Contains(td.Name, "conv1d.weight") && len(td.Shape) == 3 && td.Shape[2] != 1
	castToBF16 := qwen35NeedsCastToBF16(td.Name, td.Dtype)
	if !shiftNorm && !transposeConv && !castToBF16 {
		return td, nil
	}

	raw, err := io.ReadAll(td.Reader())
	if err != nil {
		return nil, fmt.Errorf("failed to read tensor %s: %w", td.Name, err)
	}

	values, err := DecodeFloatTensor(td.Dtype, raw)
	if err != nil {
		return nil, fmt.Errorf("failed to decode tensor %s: %w", td.Name, err)
	}

	shape := append([]int32(nil), td.Shape...)
	if transposeConv {
		values, shape = qwen35TransposeConv1D(values, shape)
	}
	if shiftNorm {
		for i := range values {
			values[i] += 1.0
		}
	}

	targetDtype := td.Dtype
	if castToBF16 {
		targetDtype = "BF16"
	}

	out, err := EncodeFloatTensor(targetDtype, values)
	if err != nil {
		return nil, fmt.Errorf("failed to encode tensor %s: %w", td.Name, err)
	}

	return safetensors.NewTensorDataFromBytes(td.Name, targetDtype, shape, out), nil
}

func (t qwen35ImportTransform) transformTensor(td *safetensors.TensorData) ([]*safetensors.TensorData, error) {
	if td == nil {
		return nil, nil
	}

	name := t.canonicalTensorName(td.Name)

	// Phase 1: rename/split into intermediate tensors
	var intermediates []*safetensors.TensorData
	stripped := strings.TrimSuffix(name, ".weight")
	switch {
	case strings.HasSuffix(stripped, ".mlp.experts.gate_up_proj"):
		prefix := strings.TrimSuffix(stripped, ".mlp.experts.gate_up_proj")
		raw, err := io.ReadAll(td.Reader())
		if err != nil {
			return nil, fmt.Errorf("failed to read tensor %s: %w", td.Name, err)
		}
		gateRaw, upRaw, splitShape, err := qwen35SplitAxis1Raw(raw, td.Dtype, td.Shape)
		if err != nil {
			return nil, fmt.Errorf("failed to split tensor %s: %w", td.Name, err)
		}
		intermediates = []*safetensors.TensorData{
			safetensors.NewTensorDataFromBytes(prefix+".mlp.switch_mlp.gate_proj.weight", td.Dtype, splitShape, gateRaw),
			safetensors.NewTensorDataFromBytes(prefix+".mlp.switch_mlp.up_proj.weight", td.Dtype, splitShape, upRaw),
		}
	case strings.HasSuffix(stripped, ".mlp.experts.down_proj"):
		newName := strings.TrimSuffix(stripped, ".mlp.experts.down_proj") + ".mlp.switch_mlp.down_proj.weight"
		intermediates = []*safetensors.TensorData{td.WithName(newName)}
	default:
		intermediates = []*safetensors.TensorData{td.WithName(name)}
	}

	// Phase 2: rewrite all intermediates
	results := make([]*safetensors.TensorData, 0, len(intermediates))
	for _, inter := range intermediates {
		rewritten, err := t.rewriteTensorData(inter)
		if err != nil {
			return nil, err
		}
		results = append(results, rewritten)
	}
	return results, nil
}

func (t qwen35ImportTransform) canonicalTensorName(name string) string {
	// Vision tensors: normalize to vision_tower.* prefix
	switch {
	case strings.HasPrefix(name, "model.visual."):
		return "vision_tower." + strings.TrimPrefix(name, "model.visual.")
	case strings.HasPrefix(name, "vision_tower."):
		return name
	}

	// Language model tensors: normalize to language_model.model.* prefix
	if !t.rewriteLanguageModel {
		return name
	}
	switch {
	case strings.HasPrefix(name, "model.language_model"):
		return "language_model.model" + strings.TrimPrefix(name, "model.language_model")
	case strings.HasPrefix(name, "language_model."):
		return name
	default:
		return "language_model." + name
	}
}

func qwen35ShouldShiftNormKey(key string) bool {
	for _, suffix := range []string{
		".input_layernorm.weight",
		".post_attention_layernorm.weight",
		"model.norm.weight",
		".q_norm.weight",
		".k_norm.weight",
	} {
		if strings.HasSuffix(key, suffix) {
			return true
		}
	}
	return false
}

func qwen35NeedsCastToBF16(name, dtype string) bool {
	if strings.HasSuffix(name, "A_log") {
		return false
	}
	switch strings.ToUpper(dtype) {
	case "F16", "F32", "F64":
		return true
	default:
		return false
	}
}

func qwen35TransposeConv1D(values []float32, shape []int32) ([]float32, []int32) {
	if len(shape) != 3 {
		return values, shape
	}

	d0, d1, d2 := int(shape[0]), int(shape[1]), int(shape[2])
	out := make([]float32, len(values))
	for i := range d0 {
		for j := range d1 {
			for k := range d2 {
				inIdx := (i*d1+j)*d2 + k
				outIdx := (i*d2+k)*d1 + j
				out[outIdx] = values[inIdx]
			}
		}
	}

	return out, []int32{shape[0], shape[2], shape[1]}
}

func qwen35SplitAxis1Raw(raw []byte, dtype string, shape []int32) ([]byte, []byte, []int32, error) {
	if len(shape) != 3 {
		return nil, nil, nil, fmt.Errorf("expected 3D tensor, got shape %v", shape)
	}
	if shape[1]%2 != 0 {
		return nil, nil, nil, fmt.Errorf("axis 1 dim %d is not even", shape[1])
	}

	elemSize, err := DTypeSize(dtype)
	if err != nil {
		return nil, nil, nil, err
	}

	d0, d1, d2 := int(shape[0]), int(shape[1]), int(shape[2])
	perExpertBytes := d1 * d2 * elemSize
	if len(raw) != d0*perExpertBytes {
		return nil, nil, nil, fmt.Errorf("raw byte length %d does not match shape %v and dtype %s", len(raw), shape, dtype)
	}

	halfD1 := d1 / 2
	halfExpertBytes := halfD1 * d2 * elemSize
	gateRaw := make([]byte, d0*halfExpertBytes)
	upRaw := make([]byte, d0*halfExpertBytes)
	for e := range d0 {
		src := e * perExpertBytes
		dst := e * halfExpertBytes
		copy(gateRaw[dst:dst+halfExpertBytes], raw[src:src+halfExpertBytes])
		copy(upRaw[dst:dst+halfExpertBytes], raw[src+halfExpertBytes:src+perExpertBytes])
	}

	return gateRaw, upRaw, []int32{shape[0], int32(halfD1), shape[2]}, nil
}

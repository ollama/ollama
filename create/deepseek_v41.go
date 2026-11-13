package create

import (
	"encoding/json"
	"fmt"
	"slices"
	"strconv"
	"strings"
)

// planDeepSeekV41 imports the publisher's mixed FP4/FP8 checkpoint without
// decoding and rounding the weights a second time. The source's E8M0 scales
// are already in the byte format used by MLX's MX kernels.
func planDeepSeekV41(inv Inventory) ([]BlobSpec, error) {
	var cfg struct {
		Quantization struct {
			Method      string  `json:"quant_method"`
			BlockSize   []int32 `json:"weight_block_size"`
			ScaleFormat string  `json:"scale_fmt"`
			ExpertDtype string  `json:"expert_dtype"`
		} `json:"quantization_config"`
		Text struct {
			Layers        int `json:"num_hidden_layers"`
			Experts       int `json:"n_routed_experts"`
			MTPLayers     int `json:"num_nextn_predict_layers"`
			DSparkExperts int `json:"dspark_n_routed_experts"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(inv.RawConfig, &cfg); err != nil {
		return nil, fmt.Errorf("DeepSeek V4.1 config: %w", err)
	}
	if cfg.Quantization.Method != "fp8" || !slices.Equal(cfg.Quantization.BlockSize, []int32{32, 32}) ||
		cfg.Quantization.ScaleFormat != "ue8m0" || cfg.Quantization.ExpertDtype != "fp4" ||
		cfg.Text.Layers < 1 || cfg.Text.Experts < 1 || cfg.Text.MTPLayers < 1 || cfg.Text.DSparkExperts < 1 {
		return nil, fmt.Errorf("DeepSeek V4.1 has unsupported quantization or expert config")
	}

	groups := make(map[string]map[string]map[int]SourceTensor)
	consumed := make(map[string]bool)
	var specs []BlobSpec
	for _, name := range sortedTensorNames(inv) {
		if consumed[name] {
			continue
		}
		w := inv.Tensors[name]
		if w.Dtype == "F8_E8M0" && strings.HasSuffix(name, ".scale") {
			continue
		}
		switch w.Dtype {
		case "F8_E4M3":
			if !strings.HasSuffix(name, ".weight") {
				return nil, fmt.Errorf("DeepSeek V4.1 FP8 tensor %s is not a weight", name)
			}
			base := strings.TrimSuffix(name, ".weight")
			scale, ok := inv.Tensors[base+".scale"]
			if !ok {
				return nil, fmt.Errorf("DeepSeek V4.1 FP8 weight %s has no E8M0 scale", name)
			}
			if err := deepSeekV41ScaleLayout(w, scale, 8); err != nil {
				return nil, err
			}
			if strings.HasSuffix(name, ".engram.embed.weight") && (w.Shape[1] != 256 || scale.Shape[0] != w.Shape[0]) {
				return nil, fmt.Errorf("DeepSeek V4.1 engram table %s has unsupported dimensions", name)
			}
			outScale := base + ".weight.scale"
			st := TensorSpec{Name: outScale, Sources: []SourceTensor{scale}, Transform: TransformRelabelU8, OutDtype: "U8", OutShape: scale.Shape}
			if scale.Shape[0] != w.Shape[0] {
				st.Transform = TransformExpandE8M0
				st.OutShape = []int32{w.Shape[0], scale.Shape[1]}
			}
			specs = append(specs, BlobSpec{
				Name: name, Metadata: map[string]string{"quant_type": "mxfp8", "group_size": "32"},
				Tensors: []TensorSpec{
					{Name: name, Sources: []SourceTensor{w}, Transform: TransformRepackFP8, OutDtype: "U32", OutShape: []int32{w.Shape[0], w.Shape[1] / 4}},
					st,
				},
			})
			consumed[scale.Name] = true
		case "I8":
			group, idx, proj, err := deepSeekV41ExpertName(name)
			if err != nil {
				return nil, err
			}
			scale, ok := inv.Tensors[strings.TrimSuffix(name, ".weight")+".scale"]
			if !ok {
				return nil, fmt.Errorf("DeepSeek V4.1 FP4 expert %s has no E8M0 scale", name)
			}
			if err := deepSeekV41ScaleLayout(w, scale, 4); err != nil {
				return nil, err
			}
			if groups[group] == nil {
				groups[group] = make(map[string]map[int]SourceTensor)
			}
			if groups[group][proj] == nil {
				groups[group][proj] = make(map[int]SourceTensor)
			}
			groups[group][proj][idx] = w
			consumed[scale.Name] = true
		case "BF16", "F32", "F16":
			specs = append(specs, BlobSpec{Name: name, Tensors: []TensorSpec{{Name: name, Sources: []SourceTensor{w}}}})
		default:
			return nil, fmt.Errorf("DeepSeek V4.1 unexpected tensor %s dtype %s", name, w.Dtype)
		}
	}
	for name, tensor := range inv.Tensors {
		if tensor.Dtype == "F8_E8M0" && !consumed[name] {
			return nil, fmt.Errorf("DeepSeek V4.1 unpaired E8M0 scale %s", name)
		}
	}

	for _, group := range sortedKeys(groups) {
		count, err := deepSeekV41ExpectedExperts(group, cfg.Text.Layers, cfg.Text.Experts, cfg.Text.MTPLayers, cfg.Text.DSparkExperts)
		if err != nil {
			return nil, err
		}
		projections := groups[group]
		if len(projections) != 3 {
			return nil, fmt.Errorf("DeepSeek V4.1 group %s has %d projections, want 3", group, len(projections))
		}
		bank := BlobSpec{Name: group, Metadata: map[string]string{"quant_type": "mxfp4", "group_size": "32"}}
		for _, proj := range []string{"w1", "w2", "w3"} {
			weights := projections[proj]
			if len(weights) != count {
				return nil, fmt.Errorf("DeepSeek V4.1 group %s projection %s has %d experts, want %d", group, proj, len(weights), count)
			}
			weightSources := make([]SourceTensor, count)
			scaleSources := make([]SourceTensor, count)
			for idx := range count {
				w, ok := weights[idx]
				if !ok {
					return nil, fmt.Errorf("DeepSeek V4.1 group %s projection %s lacks expert %d", group, proj, idx)
				}
				s := inv.Tensors[strings.TrimSuffix(w.Name, ".weight")+".scale"]
				if idx > 0 && (!slices.Equal(w.Shape, weightSources[0].Shape) || !slices.Equal(s.Shape, scaleSources[0].Shape)) {
					return nil, fmt.Errorf("DeepSeek V4.1 group %s projection %s has mismatched expert shape", group, proj)
				}
				weightSources[idx], scaleSources[idx] = w, s
			}
			base := group + "." + proj + ".weight"
			bank.Tensors = append(bank.Tensors,
				TensorSpec{Name: base, Sources: weightSources, Transform: TransformStackExperts, OutDtype: "U32", OutShape: []int32{int32(count), weightSources[0].Shape[0], weightSources[0].Shape[1] / 4}},
				TensorSpec{Name: base + ".scale", Sources: scaleSources, Transform: TransformStackExperts, OutDtype: "U8", OutShape: []int32{int32(count), scaleSources[0].Shape[0], scaleSources[0].Shape[1]}},
			)
		}
		specs = append(specs, bank)
	}
	return specs, nil
}

func deepSeekV41ScaleLayout(weight, scale SourceTensor, bits int) error {
	if scale.Dtype != "F8_E8M0" || len(weight.Shape) != 2 || len(scale.Shape) != 2 ||
		weight.Shape[0] < 1 || weight.Shape[1] < 1 || weight.Shape[1]%32 != 0 ||
		weight.Shape[1]%4 != 0 || scale.Shape[1] != weight.Shape[1]/int32(16*bits/4) {
		return fmt.Errorf("DeepSeek V4.1 weight %s and scale %s have incompatible layouts: %s %v, %s %v", weight.Name, scale.Name, weight.Dtype, weight.Shape, scale.Dtype, scale.Shape)
	}
	fp4RowsMismatch := bits == 4 && scale.Shape[0] != weight.Shape[0]
	fp8RowsMismatch := bits == 8 && scale.Shape[0] != weight.Shape[0] &&
		(weight.Shape[0]%32 != 0 || scale.Shape[0] != weight.Shape[0]/32)
	if fp4RowsMismatch || fp8RowsMismatch {
		return fmt.Errorf("DeepSeek V4.1 weight %s and scale %s have incompatible row blocks", weight.Name, scale.Name)
	}
	return nil
}

func deepSeekV41ExpertName(name string) (string, int, string, error) {
	for _, root := range []string{"layers.", "mtp."} {
		if !strings.HasPrefix(name, root) {
			continue
		}
		rest := strings.TrimPrefix(name, root)
		layer, suffix, ok := strings.Cut(rest, ".ffn.experts.")
		if !ok {
			break
		}
		if idx, err := strconv.Atoi(layer); err == nil && idx >= 0 && strconv.Itoa(idx) == layer {
			group := root + layer + ".ffn.experts"
			expert, proj, err := parseExpertTensor(group, name)
			if err == nil && (proj == "w1" || proj == "w2" || proj == "w3") && suffix != "" {
				return group, expert, proj, nil
			}
		}
		break
	}
	return "", 0, "", fmt.Errorf("DeepSeek V4.1 unexpected FP4 expert name %s", name)
}

func deepSeekV41ExpectedExperts(group string, layers, experts, mtpLayers, dsparkExperts int) (int, error) {
	root, rest, ok := strings.Cut(group, ".")
	if !ok {
		return 0, fmt.Errorf("DeepSeek V4.1 invalid expert group %s", group)
	}
	index, err := strconv.Atoi(strings.TrimSuffix(rest, ".ffn.experts"))
	if err != nil || index < 0 {
		return 0, fmt.Errorf("DeepSeek V4.1 invalid expert group %s", group)
	}
	switch root {
	case "layers":
		if index < layers {
			return experts, nil
		}
	case "mtp":
		if index < mtpLayers {
			return dsparkExperts, nil
		}
	}
	return 0, fmt.Errorf("DeepSeek V4.1 expert group %s exceeds configured layers", group)
}

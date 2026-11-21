package create

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/mlx/quant"
)

const inkling2Architecture = "Inkling2LoopForConditionalGeneration"

// Expose Inkling's model_max_length as max_position_embeddings for ollama show.
func inkling2ContextConfig(raw []byte) ([]byte, error) {
	var config map[string]any
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&config); err != nil {
		return nil, err
	}
	text, _ := config["text_config"].(map[string]any)
	limit, _ := text["model_max_length"].(json.Number)
	if n, err := limit.Int64(); err != nil || n <= 0 {
		return nil, fmt.Errorf("Inkling 2 requires a positive text_config.model_max_length")
	}
	if existing, ok := text["max_position_embeddings"]; ok {
		if existing != limit {
			return nil, fmt.Errorf("Inkling 2 text_config.max_position_embeddings disagrees with model_max_length")
		}
		return raw, nil
	}
	text["max_position_embeddings"] = limit
	return json.Marshal(config)
}

func inkling2BankQuantization(name string, inv Inventory) (string, error) {
	if !strings.HasSuffix(name, ".experts.w13_weight") && !strings.HasSuffix(name, ".experts.w2_weight") {
		return "", nil
	}
	weight, scale := inv.Tensors[name], inv.Tensors[name+"_scale"]
	var mode string
	switch {
	case strings.EqualFold(weight.Dtype, "U8") && isE4M3Dtype(scale.Dtype):
		mode = "nvfp4"
	case isE4M3Dtype(weight.Dtype) && strings.EqualFold(scale.Dtype, "U8"):
		mode = "mxfp8"
	default:
		return "", nil
	}
	// Packing the last dimension into U32 words must preserve every byte.
	if len(weight.Shape) != 3 || weight.Shape[2] <= 0 || weight.Shape[2]%4 != 0 {
		return "", fmt.Errorf("packed Inkling2 expert bank %s has invalid shape %v", name, weight.Shape)
	}
	return mode, nil
}

func classifyInkling2Banks(inv Inventory, kind SourceKind) (SourceKind, error) {
	for name := range inv.Tensors {
		mode, err := inkling2BankQuantization(name, inv)
		if err != nil {
			return kind, err
		}
		if mode != "" {
			kind = SourcePrequantized
		}
	}
	return kind, nil
}

// Inkling ships stacked gate/up and down projections. The runtime looks up
// each bank and its scales by their publisher names. Only their headers change.
func inkling2ExpertBank(name string, inv Inventory) (BlobSpec, bool) {
	mode, err := inkling2BankQuantization(name, inv)
	if err != nil || mode == "" {
		return BlobSpec{}, false
	}
	weight, scale := inv.Tensors[name], inv.Tensors[name+"_scale"]
	w := TensorSpec{Name: name, Sources: []SourceTensor{weight}, OutDtype: "U32", OutShape: []int32{weight.Shape[0], weight.Shape[1], weight.Shape[2] / 4}}
	s := TensorSpec{Name: scale.Name, Sources: []SourceTensor{scale}}
	w.Transform = TransformRepackFP8
	if mode == "nvfp4" {
		w.Transform = TransformRepackFP4
		s.Transform, s.OutDtype = TransformRelabelU8, "U8"
	}
	groupSize, _, _ := quant.Params(mode)
	return BlobSpec{
		Name:     name,
		Tensors:  []TensorSpec{w, s},
		Metadata: prequantMetadata(inv, mode, strconv.Itoa(groupSize)),
	}, true
}

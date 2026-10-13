package create

import (
	"encoding/json"
	"fmt"
	"strings"
)

type kevImportTransform struct{}

func kevFileType(inv Inventory) (string, error) {
	// The pointer head is FP32; the text embedding records the backbone dtype.
	var embedding string
	for _, name := range sortedTensorNames(inv) {
		if name != "embed_tokens.weight" && !strings.HasSuffix(name, ".embed_tokens.weight") {
			continue
		}
		if embedding != "" {
			return "", fmt.Errorf("Kev source has multiple text embedding tensors: %q and %q", embedding, name)
		}
		embedding = name
	}
	if embedding == "" {
		return "", fmt.Errorf("Kev source has no text embedding tensor")
	}
	dtype := strings.ToUpper(inv.Tensors[embedding].Dtype)
	switch dtype {
	case "BF16", "F16", "F32":
		return dtype, nil
	default:
		return "", fmt.Errorf("unsupported Kev text embedding dtype %q in %s", dtype, embedding)
	}
}

func newKevImportTransform(_ json.RawMessage) (quantizePolicy, error) {
	return kevImportTransform{}, nil
}

func (kevImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Current adapter checkpoints use model.*; the full-weight 27B text
	// checkpoint has top-level layers.*. Keep the FP32 pointer head intact.
	if !strings.HasPrefix(name, "model.") && !strings.HasPrefix(name, "layers.") {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}

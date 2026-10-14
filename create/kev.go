package create

import (
	"fmt"
	"strings"
)

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

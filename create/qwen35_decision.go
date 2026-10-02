package create

import (
	"encoding/json"
	"strings"
)

type qwen35DecisionImportTransform struct{}

func newQwen35DecisionImportTransform(json.RawMessage) (quantizePolicy, error) {
	return qwen35DecisionImportTransform{}, nil
}

func (qwen35DecisionImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Only backbone tensors are quantized. Decision heads and any
	// lexical output embeddings retain source precision.
	if !strings.HasPrefix(name, "model.") {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}

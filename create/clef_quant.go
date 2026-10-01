package create

import (
	"encoding/json"
	"strings"
)

type clefImportTransform struct{}

func newClefImportTransform(json.RawMessage) (quantizePolicy, error) {
	return clefImportTransform{}, nil
}

func (clefImportTransform) quantizationType(name string, shape []int32, requested string) string {
	// Only backbone tensors are quantized. The joint head and its lexical
	// output embeddings retain source precision.
	if !strings.HasPrefix(name, "model.") {
		return ""
	}
	return qwen35ImportTransform{}.quantizationType(name, shape, requested)
}

package create

import (
	"encoding/json"
	"strings"
)

// Keep the router and sandwich norms in source precision. Expert matrices
// dominate this 78B/3.5B-active model and remain in the requested format.
// The much smaller shared experts, attention outputs and vocabulary matrices
// use eight bits to limit error without doubling routed-expert bandwidth.
type kolibri1ImportTransform struct{}

func newKolibri1ImportTransform(json.RawMessage) (quantizePolicy, error) {
	return kolibri1ImportTransform{}, nil
}

func (kolibri1ImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)
	if base == "" || !strings.HasSuffix(name, ".weight") || len(shape) < 2 || strings.HasSuffix(name, ".mlp.gate.weight") {
		return ""
	}
	if strings.Contains(name, ".mlp.experts.") {
		return sensitiveType(false, shape, base)
	}
	return sensitiveType(true, shape, base)
}

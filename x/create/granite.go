package create

import (
	"encoding/json"
	"strings"
)

// graniteImportTransform adjusts quantization for dense Granite imports.
type graniteImportTransform struct{}

func newGraniteImportTransform(_ json.RawMessage) (quantizePolicy, error) {
	return graniteImportTransform{}, nil
}

func (graniteImportTransform) quantizationType(name string, shape []int32, quantize string) string {
	base := normalizeQuantType(quantize)

	// o_proj's block absmax runs small enough on Granite's dense checkpoints
	// that fp4's absmax/6 scale can underflow the e4m3 scale encoding (min
	// subnormal ~0.00195), flushing the scale to zero and corrupting
	// dequantization for that block. The generic policy already promotes
	// v_proj/k_proj/down_proj for the same reason; o_proj only needs it here.
	if strings.Contains(name, ".self_attn.o_proj.weight") && (base == "nvfp4" || base == "mxfp4") {
		if e := eightBit(base); isAligned(shape, e) {
			return e
		}
	}

	return GetTensorQuantization(name, shape, quantize)
}

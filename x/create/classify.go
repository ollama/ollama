package create

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/x/quant"
)

// SourceKind is the overarching dtype for a given safetensors model
type SourceKind int

const (
	SourceFloat        SourceKind = iota // bf16/fp16/fp32 — quantizable on request
	SourceBlockFP8                       // HF block-FP8 — auto-converted to mxfp8
	SourcePrequantized                   // already quantized — copied through
)

func (k SourceKind) String() string {
	switch k {
	case SourceFloat:
		return "float"
	case SourceBlockFP8:
		return "block-fp8"
	case SourcePrequantized:
		return "prequantized"
	default:
		return "unknown"
	}
}

// Classification is the decision about a source model: its kind and the
// quantization that will actually be applied. An empty Quantize means the
// tensors are stored at source precision (no quantization).
type Classification struct {
	Kind     SourceKind
	Quantize string
}

// Classify decides a source model's kind and resolves the effective
// quantization from the user's requested type, rejecting requests that are not
// allowed for the kind.
func Classify(inv Inventory, requested string) (Classification, error) {
	requested, err := normalizeRequested(requested)
	if err != nil {
		return Classification{}, err
	}

	if name, ok := firstUnsupportedFP8(inv); ok {
		return Classification{}, fmt.Errorf("unsupported fp8 source: tensor %s is F8_E5M2; only F8_E4M3 block-FP8 sources are supported", name)
	}

	switch detectKind(inv) {
	case SourceFloat:
		return Classification{Kind: SourceFloat, Quantize: requested}, nil

	case SourcePrequantized:
		if requested != "" {
			return Classification{}, fmt.Errorf("cannot requantize an already-quantized source model (requested %q): only bf16/fp16/fp32 sources can be quantized", requested)
		}
		return Classification{Kind: SourcePrequantized}, nil

	case SourceBlockFP8:
		rows, cols, ok := inv.Config.HFFP8WeightBlockSize()
		if !ok {
			return Classification{}, fmt.Errorf("fp8 source model is missing weight_block_size metadata")
		}
		if rows != 128 || cols != 128 {
			return Classification{}, fmt.Errorf("unsupported fp8 source block size %dx%d (only 128x128 is supported)", rows, cols)
		}
		if requested != "" {
			return Classification{}, fmt.Errorf("cannot quantize an fp8 source model (requested %q): fp8 sources are converted to mxfp8 automatically; only bf16/fp16/fp32 sources can be quantized", requested)
		}
		return Classification{Kind: SourceBlockFP8, Quantize: "mxfp8"}, nil
	}

	return Classification{}, fmt.Errorf("could not classify source model in %s", inv.Dir)
}

// normalizeRequested validates the user's quantize value and returns its
// canonical form ("" for no quantization).
func normalizeRequested(requested string) (string, error) {
	if strings.TrimSpace(requested) == "" {
		return "", nil
	}
	c := quant.Canonical(requested)
	if c == "" {
		return "", fmt.Errorf("unsupported quantize type %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", requested)
	}
	return c, nil
}

// detectKind sorts a source into Float, BlockFP8, or Prequantized using only
// the inventory's tensor names, dtypes, and config. Prequantized is detected
// from the tensors themselves, so a model whose quantization config sidecar is
// missing (e.g. a ModelOpt checkpoint without hf_quant_config.json) is still
// recognized as already-quantized and not mistaken for a float model.
func detectKind(inv Inventory) SourceKind {
	var hasMLXScales, hasPacked, hasNVFP4Scale, hasFP8Weight bool
	for name, t := range inv.Tensors {
		switch {
		case strings.HasSuffix(name, ".scales"):
			hasMLXScales = true
		case strings.HasSuffix(name, ".weight_packed"):
			hasPacked = true
		case strings.HasSuffix(name, ".weight_scale"):
			// An NVFP4 per-block scale sits on a packed (U8) weight. A
			// block-FP8 source also has a scale companion, but its weight is
			// F8_E4M3 — so the base weight's dtype disambiguates the two.
			if bt, ok := inv.Tensors[strings.TrimSuffix(name, "_scale")]; ok && isPackedDtype(bt.Dtype) {
				hasNVFP4Scale = true
			}
		}
		if strings.HasSuffix(name, ".weight") && isE4M3Dtype(t.Dtype) {
			hasFP8Weight = true
		}
	}

	switch {
	case hasMLXScales || hasPacked || hasNVFP4Scale:
		return SourcePrequantized
	case hasFP8Weight:
		return SourceBlockFP8
	default:
		return SourceFloat
	}
}

// firstUnsupportedFP8 returns the name of the first F8_E5M2 weight in the
// source, if any. We decode only E4M3, so an E5M2 source must be rejected
// explicitly rather than silently mishandled.
func firstUnsupportedFP8(inv Inventory) (string, bool) {
	for name, t := range inv.Tensors {
		if strings.HasSuffix(name, ".weight") && isE5M2Dtype(t.Dtype) {
			return name, true
		}
	}
	return "", false
}

func isPackedDtype(dtype string) bool {
	switch strings.ToUpper(dtype) {
	case "U8", "U32": // current .weight_scale producers ship U8; U32 covers a future word-packed source
		return true
	default:
		return false
	}
}

func isE4M3Dtype(dtype string) bool {
	switch strings.ToUpper(dtype) {
	case "F8_E4M3", "F8_E4M3FN":
		return true
	default:
		return false
	}
}

func isE5M2Dtype(dtype string) bool {
	switch strings.ToUpper(dtype) {
	case "F8_E5M2", "F8_E5M2FNUZ":
		return true
	default:
		return false
	}
}

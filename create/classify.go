package create

import (
	"fmt"
	"strings"

	"github.com/ollama/ollama/mlx/quant"
)

// SourceKind is the overarching dtype for a given safetensors model
type SourceKind int

const (
	SourceFloat        SourceKind = iota // bf16/fp16/fp32 — quantizable on request
	SourceFP8                            // FP8 weights with float scales — auto-converted to mxfp8
	SourcePrequantized                   // already quantized — copied through
)

func (k SourceKind) String() string {
	switch k {
	case SourceFloat:
		return "float"
	case SourceFP8:
		return "fp8"
	case SourcePrequantized:
		return "prequantized"
	default:
		return "unknown"
	}
}

// Classification is the decision about a source model: its kind and the
// effective quantization of the imported weights. Quantize may describe a
// requested conversion or a quantization already present in the source.
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
		return Classification{}, fmt.Errorf("unsupported fp8 source: tensor %s is F8_E5M2; only F8_E4M3 sources are supported", name)
	}

	kind, err := detectKind(inv)
	if err != nil {
		return Classification{}, err
	}
	switch kind {
	case SourceFloat:
		return Classification{Kind: SourceFloat, Quantize: requested}, nil

	case SourcePrequantized:
		if requested != "" {
			return Classification{}, fmt.Errorf("cannot requantize an already-quantized source model (requested %q): only bf16/fp16/fp32 sources can be quantized", requested)
		}
		return Classification{
			Kind:     SourcePrequantized,
			Quantize: detectPrequantizedQuantization(inv),
		}, nil

	case SourceFP8:
		if requested != "" {
			return Classification{}, fmt.Errorf("cannot quantize an fp8 source model (requested %q): fp8 weights are stored with their published values; only bf16/fp16/fp32 sources can be quantized", requested)
		}
		return Classification{Kind: SourceFP8, Quantize: fp8DecodeTarget}, nil
	}

	return Classification{}, fmt.Errorf("could not classify source model in %s", inv.Dir)
}

// detectPrequantizedQuantization reports a single quantization shared by the
// source's recognized prequantized weights. A mixed or unrecognized source has
// no single file type to record in the model manifest.
func detectPrequantizedQuantization(inv Inventory) string {
	var detected string
	for _, name := range sortedTensorNames(inv) {
		spec, _, ok := matchPrequant(name, inv)
		if !ok {
			continue
		}
		q := quant.Canonical(spec.Metadata["quant_type"])
		if q == "" {
			continue
		}
		if detected != "" && detected != q {
			return ""
		}
		detected = q
	}
	return detected
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

// detectKind recognizes each weight's encoding and sorts the source by the
// most specific one present: Prequantized if any weight is already quantized
// in a form the runtime executes, else FP8 if any weight is FP8 with a float
// scale, else Float. A source can mix encodings (ModelOpt mixed precision
// ships NVFP4 experts beside FP8 projections and BF16 attention); the planner
// handles each weight by its own encoding. An FP8 weight with no usable scale
// is an error: it cannot be decoded and the runtime has no FP8 type.
func detectKind(inv Inventory) (SourceKind, error) {
	var prequant, fp8 bool
	for _, name := range sortedTensorNames(inv) {
		if _, _, ok := matchPrequant(name, inv); ok {
			prequant = true
			continue
		}
		_, ok, err := matchFP8(name, inv)
		if err != nil {
			return SourceFloat, err
		}
		if ok {
			fp8 = true
			continue
		}
		if t := inv.Tensors[name]; strings.HasSuffix(name, ".weight") && isE4M3Dtype(t.Dtype) {
			return SourceFloat, fmt.Errorf("fp8 weight %s has no float scale companion (%s_scale or %s_scale_inv)", name, name, name)
		}
	}
	switch {
	case prequant:
		return SourcePrequantized, nil
	case fp8:
		return SourceFP8, nil
	default:
		return SourceFloat, nil
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

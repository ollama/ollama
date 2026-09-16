// Package quant holds the quantization format facts shared by the model
// importer, the runtime loader, and `ollama show`. It deliberately has no
// dependency on the MLX C library, so any package can use it without pulling
// in cgo — which is what keeps these facts from drifting between separate
// hand-maintained copies.
package quant

import "strings"

type params struct {
	groupSize int
	bits      int
	mode      string
}

// byType maps each canonical quantization type to its parameters. Aliases are
// resolved to a canonical name by Canonical before lookup.
var byType = map[string]params{
	"nvfp4": {groupSize: 16, bits: 4, mode: "nvfp4"},
	"mxfp4": {groupSize: 32, bits: 4, mode: "mxfp4"},
	"int4":  {groupSize: 64, bits: 4, mode: "affine"},
	"mxfp8": {groupSize: 32, bits: 8, mode: "mxfp8"},
	"int8":  {groupSize: 64, bits: 8, mode: "affine"},
}

// Canonical returns the canonical name for a quantization type, resolving
// aliases (for example "FP8" and "Q8" both map to "int8"). It returns "" for
// the empty string and for any type it does not recognize.
func Canonical(quantType string) string {
	switch strings.ToUpper(strings.TrimSpace(quantType)) {
	case "NVFP4":
		return "nvfp4"
	case "MXFP4":
		return "mxfp4"
	case "MXFP8":
		return "mxfp8"
	case "INT4", "FP4", "Q4":
		return "int4"
	case "INT8", "FP8", "Q8":
		return "int8"
	default:
		return ""
	}
}

// Params returns the default group size, bit width, and mode for a
// quantization type. The empty string returns zeros. An unrecognized
// non-empty type falls back to 8-bit affine, matching the runtime loader's
// historical leniency toward unexpected metadata.
func Params(quantType string) (groupSize, bits int, mode string) {
	if strings.TrimSpace(quantType) == "" {
		return 0, 0, ""
	}
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.groupSize, p.bits, p.mode
	}
	return 32, 8, "affine"
}

// Bits returns the bit width of a recognized quantization type, or 0 if the
// type is empty or unrecognized. Unlike Params it applies no fallback, so
// callers that size or display tensors never act on an unknown type.
func Bits(quantType string) int {
	if p, ok := byType[Canonical(quantType)]; ok {
		return p.bits
	}
	return 0
}

// PackFactor returns how many quantized values are packed into one 32-bit
// word, or 0 for an empty or unrecognized type. MLX stores quantized weights
// packed into U32 words, so a tensor's logical last dimension is its stored
// last dimension times this factor.
func PackFactor(quantType string) int {
	if b := Bits(quantType); b > 0 {
		return 32 / b
	}
	return 0
}

package quant

import "testing"

func TestParams(t *testing.T) {
	tests := []struct {
		in        string
		groupSize int
		bits      int
		mode      string
	}{
		{"nvfp4", 16, 4, "nvfp4"},
		{"NVFP4", 16, 4, "nvfp4"},
		{"mxfp4", 32, 4, "mxfp4"},
		{"mxfp8", 32, 8, "mxfp8"},
		{"int4", 64, 4, "affine"},
		{"int8", 64, 8, "affine"},
		{"fp4", 64, 4, "affine"},
		{"q4", 64, 4, "affine"},
		{"fp8", 64, 8, "affine"},
		{"q8", 64, 8, "affine"},
		{"", 0, 0, ""},
		// Unknown non-empty types fall back to 8-bit affine, matching the
		// runtime loader's historical behavior.
		{"something-else", 32, 8, "affine"},
	}
	for _, tt := range tests {
		gs, bits, mode := Params(tt.in)
		if gs != tt.groupSize || bits != tt.bits || mode != tt.mode {
			t.Errorf("Params(%q) = (%d, %d, %q), want (%d, %d, %q)", tt.in, gs, bits, mode, tt.groupSize, tt.bits, tt.mode)
		}
	}
}

func TestBitsAndPackFactor(t *testing.T) {
	tests := []struct {
		in         string
		bits       int
		packFactor int
	}{
		{"int4", 4, 8},
		{"nvfp4", 4, 8},
		{"mxfp4", 4, 8}, // regression: mxfp4 was missing from the old show.go switch
		{"int8", 8, 4},
		{"mxfp8", 8, 4},
		{"FP8", 8, 4},
		{"", 0, 0},
		{"unknown", 0, 0}, // strict: no fallback for sizing/display
	}
	for _, tt := range tests {
		if got := Bits(tt.in); got != tt.bits {
			t.Errorf("Bits(%q) = %d, want %d", tt.in, got, tt.bits)
		}
		if got := PackFactor(tt.in); got != tt.packFactor {
			t.Errorf("PackFactor(%q) = %d, want %d", tt.in, got, tt.packFactor)
		}
	}
}

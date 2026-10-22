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
		{"int3", 64, 3, "affine"},
		{"Q3", 64, 3, "affine"},
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

func TestBits(t *testing.T) {
	for _, tt := range []struct {
		in   string
		bits int
	}{
		{"int3", 3},
		{"int4", 4},
		{"nvfp4", 4},
		{"mxfp4", 4},
		{"int8", 8},
		{"mxfp8", 8},
		{"FP8", 8},
		{"", 0},
		{"unknown", 0},
	} {
		if got := Bits(tt.in); got != tt.bits {
			t.Errorf("Bits(%q) = %d, want %d", tt.in, got, tt.bits)
		}
	}
}

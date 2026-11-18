package model

import "testing"

func TestSupportsGatherQMM(t *testing.T) {
	for _, tt := range []struct {
		mode string
		bits int
		want bool
	}{
		{mode: "affine", bits: 4, want: true},
		{mode: "affine", bits: 8, want: true},
		{mode: "affine", bits: 3, want: false},
		{mode: "mxfp8", bits: 8, want: true},
		{mode: "mxfp8", bits: 4, want: false},
		{mode: "nvfp4", bits: 4, want: true},
		{mode: "mxfp4", bits: 4, want: true},
		{mode: "unknown", bits: 4, want: false},
	} {
		if got := SupportsGatherQMM(tt.mode, tt.bits); got != tt.want {
			t.Fatalf("SupportsGatherQMM(%q, %d) = %v, want %v", tt.mode, tt.bits, got, tt.want)
		}
	}
}

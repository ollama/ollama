package model

import (
	"testing"

	"github.com/ollama/ollama/mlx"
)

func TestFreeTensorKeys(t *testing.T) {
	w, s := new(mlx.Array), new(mlx.Array)
	tensors := map[string]*mlx.Array{"w": w, "w_scale": s, "empty": nil, "": w}
	FreeTensorKeys(tensors, "w", "", "empty", "missing")

	if _, ok := tensors["w"]; ok {
		t.Fatal("w was not freed")
	}
	if tensors["w_scale"] != s {
		t.Fatal("w_scale was freed without being listed")
	}
	// Keys that hold no tensor, including the empty key, stay in the map.
	if _, ok := tensors["empty"]; !ok {
		t.Fatal("nil entry was removed")
	}
	if tensors[""] != w {
		t.Fatal("empty key was removed")
	}
}

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

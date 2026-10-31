package model

import (
	"testing"

	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxtest"
)

func TestResolveLinearQuantParams(t *testing.T) {
	const name = "layers.0.mlp.gate_proj.weight"
	labeled := func(quantType string, groupSize int) map[string]*TensorQuantInfo {
		return map[string]*TensorQuantInfo{name: {QuantType: quantType, GroupSize: groupSize}}
	}

	tests := []struct {
		name          string
		bits          int
		groupSize     int
		tensorQuant   map[string]*TensorQuantInfo
		wantGroupSize int
		wantBits      int
	}{
		// mlx-lm per-module overrides: create stamps every tensor with the
		// checkpoint-wide type, so an 8-bit tensor arrives labeled int4.
		{name: "8-bit tensor labeled int4", bits: 8, groupSize: 64, tensorQuant: labeled("INT4", 64), wantGroupSize: 64, wantBits: 8},
		{name: "4-bit tensor labeled int8", bits: 4, groupSize: 64, tensorQuant: labeled("INT8", 64), wantGroupSize: 64, wantBits: 4},
		{name: "4-bit tensor labeled int4", bits: 4, groupSize: 64, tensorQuant: labeled("INT4", 64), wantGroupSize: 64, wantBits: 4},
		{name: "8-bit tensor labeled int8", bits: 8, groupSize: 64, tensorQuant: labeled("INT8", 64), wantGroupSize: 64, wantBits: 8},
		// 4-bit/64 and 8-bit/32 pack alike, so the label's group size decides.
		{name: "packs like 4-bit, labeled group 32", bits: 8, groupSize: 32, tensorQuant: labeled("INT8", 32), wantGroupSize: 32, wantBits: 8},
		{name: "packs like 8-bit, labeled group 64", bits: 4, groupSize: 64, tensorQuant: labeled("INT4", 64), wantGroupSize: 64, wantBits: 4},
		{name: "wrong group size falls back to shapes", bits: 8, groupSize: 64, tensorQuant: labeled("INT4", 32), wantGroupSize: 64, wantBits: 8},
		{name: "no metadata", bits: 8, groupSize: 64, wantGroupSize: 64, wantBits: 8},
	}

	for _, tt := range tests {
		mlxtest.RunSubtest(t, tt.name, func(t *mlxtest.T) {
			dense := mlx.Zeros(mlx.DTypeBFloat16, 2, 256)
			weight, scales, biases := mlx.Quantize(dense, tt.groupSize, tt.bits, "affine")
			mlx.Eval(weight, scales, biases)

			groupSize, bits, mode := ResolveLinearQuantParams(64, 4, "affine", tt.tensorQuant, name, weight, scales)
			if groupSize != tt.wantGroupSize || bits != tt.wantBits || mode != "affine" {
				t.Fatalf("got (%d, %d, %q), want (%d, %d, %q)", groupSize, bits, mode, tt.wantGroupSize, tt.wantBits, "affine")
			}
		})
	}
}

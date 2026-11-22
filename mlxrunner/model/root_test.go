package model

import "testing"

func TestRootQuantization(t *testing.T) {
	for _, tt := range []struct {
		name      string
		root      *Root
		groupSize int
		bits      int
		mode      string
	}{
		// Unquantized models leave the linear defaults zero, as constructors
		// that only set them for a non-empty quant type did.
		{name: "unquantized", root: &Root{}},
		{name: "default group size", root: &Root{quantType: "NVFP4", groupSize: 16}, groupSize: 16, bits: 4, mode: "nvfp4"},
		{name: "blob group size", root: &Root{quantType: "INT4", groupSize: 32}, groupSize: 32, bits: 4, mode: "affine"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			groupSize, bits, mode, tensorQuant := tt.root.Quantization()
			if groupSize != tt.groupSize || bits != tt.bits || mode != tt.mode {
				t.Fatalf("Quantization() = %d, %d, %q, want %d, %d, %q", groupSize, bits, mode, tt.groupSize, tt.bits, tt.mode)
			}
			if tensorQuant == nil || len(tensorQuant) != 0 {
				t.Fatalf("tensorQuant = %v, want an empty map", tensorQuant)
			}
		})
	}
}

func TestRootQuantizationCopiesTensorQuant(t *testing.T) {
	root := &Root{
		quantType:   "INT8",
		groupSize:   64,
		tensorQuant: map[string]*TensorQuantInfo{"w": {QuantType: "INT4", GroupSize: 32}},
	}
	_, _, _, tensorQuant := root.Quantization()
	if got := tensorQuant["w"]; got == nil || got.QuantType != "INT4" || got.GroupSize != 32 {
		t.Fatalf("tensorQuant[w] = %+v, want INT4 group 32", got)
	}
	tensorQuant["w"].GroupSize = 128
	if root.tensorQuant["w"].GroupSize != 32 {
		t.Fatal("Quantization returned the root's own tensor metadata instead of a copy")
	}
}

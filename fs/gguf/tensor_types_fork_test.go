package gguf

import "testing"

// Sizes mirror the block structs in ggml/src/ggml-common.h of the vendored
// llama.cpp (unslothai/llama.cpp b11724-mix-c208e5f).
func TestForkTensorTypeSizes(t *testing.T) {
	cases := []struct {
		tt        TensorType
		value     uint32
		name      string
		typeSize  int64
		blockSize int64
	}{
		{TensorTypeQ2_0, 42, "q2_0", 18, 64},
		{TensorTypeMXFP8, 43, "mxfp8", 33, 32},
		{TensorTypeIQ1_XS, 64, "iq1_xs", 46, 256},
		{TensorTypeIQ1_XXS, 65, "iq1_xxs", 42, 256},
		{TensorTypeIQ1_XXXS, 66, "iq1_xxxs", 38, 256},
	}
	for _, c := range cases {
		if uint32(c.tt) != c.value {
			t.Errorf("%s: value = %d, want %d", c.name, uint32(c.tt), c.value)
		}
		if got := c.tt.String(); got != c.name {
			t.Errorf("value %d: String() = %q, want %q", c.value, got, c.name)
		}
		if got := c.tt.typeSize(); got != c.typeSize {
			t.Errorf("%s: typeSize = %d, want %d", c.name, got, c.typeSize)
		}
		if got := c.tt.blockSize(); got != c.blockSize {
			t.Errorf("%s: blockSize = %d, want %d", c.name, got, c.blockSize)
		}
	}

	// DeepSeek-V4.1-Flash-1017 Engram table: per_layer_token_embd.weight [256, 768022850] MXFP8.
	ti := TensorInfo{Name: "per_layer_token_embd.weight", Shape: []uint64{256, 768022850}, Type: TensorTypeMXFP8}
	if got, want := ti.NumBytes(), int64(256*768022850/32*33); got != want {
		t.Errorf("engram NumBytes = %d, want %d", got, want)
	}
}

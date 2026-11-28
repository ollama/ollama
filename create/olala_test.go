package create

import "testing"

func TestOlalaQuantizationType(t *testing.T) {
	policy, err := newOlalaImportTransform(nil)
	if err != nil {
		t.Fatal(err)
	}
	big := []int32{1536, 1536}
	experts := []int32{256, 768, 384}
	cases := []struct {
		name     string
		shape    []int32
		quantize string
		want     string
	}{
		{"model.layers.0.mixer.in_proj.weight", big, "int8", "int8"},
		{"model.layers.0.mixer_proj.weight", big, "int8", "int8"},
		{"model.layers.0.mlp.in_cat.weight", big, "int8", "int8"},
		{"model.layers.0.mlp.out_cat.weight", big, "int8", "int8"},
		{"model.layers.0.mlp.experts.experts.weight", experts, "int8", "int8"},
		{"model.embedding.weight", big, "int8", "int8"},
		{"lm_head.weight", big, "int8", "int8"},

		// 4-bit requests quantize only the experts to 4 bits
		{"model.layers.0.mlp.experts.output_experts.weight", experts, "int4", "int4"},
		{"model.layers.0.mixer.in_proj.weight", big, "int4", "int8"},
		{"lm_head.weight", big, "int4", "int8"},

		// small or sensitive tensors keep source precision
		{"model.layers.4.mixer.in_proj_small.weight", big, "int8", ""},
		{"model.layers.0.mlp.moe_gate.weight", []int32{256, 1536}, "int8", ""},
		{"model.layers.0.mixer.in_proj_mimo_x", []int32{48, 4, 64}, "int8", ""},
		{"model.layers.0.mixer.B_bias", []int32{48, 4, 128}, "int8", ""},
		{"model.layers.0.mixer.in_proj.weight", big, "", ""},
	}
	for _, tc := range cases {
		if got := policy.quantizationType(tc.name, tc.shape, tc.quantize); got != tc.want {
			t.Errorf("%s (%s): got %q, want %q", tc.name, tc.quantize, got, tc.want)
		}
	}
}

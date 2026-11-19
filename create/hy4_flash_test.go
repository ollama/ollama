package create

import "testing"

func TestHy4FlashQuantizationPreservesMappedEngram(t *testing.T) {
	policy := hy4FlashImportTransform{}
	for _, test := range []struct {
		name  string
		shape []int32
		want  string
	}{
		{"model.layers.1.mlp.experts.gate_up_proj", []int32{512, 3072, 1536}, "nvfp4"},
		{"model.layers.1.mlp.experts.down_proj", []int32{512, 1536, 1536}, "nvfp4"},
		{"model.layers.1.engram_layer.multi_head_embedding.embedding.weight", []int32{164994380, 96}, ""},
		{"model.layers.1.self_attn.kv_b_proj.weight", []int32{14336, 512}, ""},
		{"model.layers.1.mlp.gate.weight", []int32{512, 3072}, ""},
		{"vit.perceive.proj.0.weight", []int32{4608, 1152, 2, 2}, ""},
	} {
		if got := policy.quantizationType(test.name, test.shape, "nvfp4"); got != test.want {
			t.Errorf("%s: precision %q, want %q", test.name, got, test.want)
		}
	}
}

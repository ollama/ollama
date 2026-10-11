package create

import "testing"

func TestQwen35DecisionQuantizationPolicy(t *testing.T) {
	for _, arch := range []string{"ClefForDecision", "StrandsDeciderForDecision"} {
		inv := Inventory{Config: sourceModelConfig{Architectures: []string{arch}}}
		policy, err := newTensorImportTransform(inv)
		if err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name string
			want string
		}{
			{"memory_proj.weight", ""},
			{"q.weight", ""},
			{"k.weight", ""},
			{"norm.weight", ""},
			{"lm_head.weight", ""},
			{"model.language_model.layers.0.linear_attn.in_proj_a.weight", ""},
			{"model.language_model.layers.0.mlp.gate_proj.weight", "mxfp8"},
		} {
			t.Run(arch+"/"+tc.name, func(t *testing.T) {
				if got := policy.quantizationType(tc.name, []int32{1024, 1024}, "mxfp8"); got != tc.want {
					t.Errorf("quantization = %q, want %q", got, tc.want)
				}
			})
		}
	}
	ordinary := qwen35ImportTransform{}
	if got := ordinary.quantizationType("lm_head.weight", []int32{1024, 1024}, "mxfp8"); got != "mxfp8" {
		t.Errorf("ordinary Qwen output quantization = %q, want mxfp8", got)
	}
}

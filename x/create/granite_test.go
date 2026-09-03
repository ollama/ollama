package create

import "testing"

func TestGraniteQuantizationType(t *testing.T) {
	policy := graniteImportTransform{}

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{
			name:     "o_proj promoted for nvfp4",
			tensor:   "model.layers.0.self_attn.o_proj.weight",
			shape:    []int32{2560, 2560},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
		{
			name:     "o_proj promoted for mxfp4",
			tensor:   "model.layers.0.self_attn.o_proj.weight",
			shape:    []int32{2560, 2560},
			quantize: "mxfp4",
			want:     "mxfp8",
		},
		{
			name:     "o_proj left alone for int4",
			tensor:   "model.layers.0.self_attn.o_proj.weight",
			shape:    []int32{2560, 2560},
			quantize: "int4",
			want:     "int4",
		},
		{
			name:     "q_proj unaffected",
			tensor:   "model.layers.0.self_attn.q_proj.weight",
			shape:    []int32{2560, 2560},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "lm_head still promoted by the generic policy",
			tensor:   "lm_head.weight",
			shape:    []int32{100352, 2560},
			quantize: "nvfp4",
			want:     "mxfp8",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := policy.quantizationType(tt.tensor, tt.shape, tt.quantize); got != tt.want {
				t.Fatalf("quantizationType(%q, %v, %q) = %q, want %q", tt.tensor, tt.shape, tt.quantize, got, tt.want)
			}
		})
	}
}

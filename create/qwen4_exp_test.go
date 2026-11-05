package create

import "testing"

func TestQwen4ExpQuantizationType(t *testing.T) {
	policy := qwen4ExpImportTransform{}

	tests := []struct {
		name     string
		tensor   string
		shape    []int32
		quantize string
		want     string
	}{
		{
			name:     "PLE shard",
			tensor:   "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
			shape:    []int32{156250, 2560},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "token embedding",
			tensor:   "model.language_model.embed_tokens.weight",
			shape:    []int32{248320, 2560},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "low rank projection",
			tensor:   "model.language_model.layers.0.linear_attn.in_proj_a.weight",
			shape:    []int32{24, 5120},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "ordinary projection",
			tensor:   "model.language_model.layers.0.linear_attn.out_proj.weight",
			shape:    []int32{5120, 3072},
			quantize: "nvfp4",
			want:     "nvfp4",
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

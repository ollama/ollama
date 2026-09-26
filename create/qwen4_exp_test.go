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
			name:     "PLE shard uses requested NVFP4",
			tensor:   "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
			shape:    []int32{156250, 2560},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "PLE shard preserves explicit MXFP8 request",
			tensor:   "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
			shape:    []int32{156250, 2560},
			quantize: "mxfp8",
			want:     "mxfp8",
		},
		{
			name:     "token embedding",
			tensor:   "model.language_model.embed_tokens.weight",
			shape:    []int32{248320, 2560},
			quantize: "nvfp4",
			want:     "mxfp8",
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
			want:     "mxfp8",
		},
		{
			name:     "QSA query projection",
			tensor:   "model.language_model.layers.3.self_attn.q_proj.weight",
			shape:    []int32{12288, 2560},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "MTP routed expert",
			tensor:   "mtp.layers.0.mlp.experts.0.down_proj.weight",
			shape:    []int32{2560, 1024},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "MTP control path",
			tensor:   "mtp.layers.0.self_attn.q_proj.weight",
			shape:    []int32{12288, 2560},
			quantize: "nvfp4",
			want:     "",
		},
		{
			name:     "main routed expert",
			tensor:   "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
			shape:    []int32{2560, 1024},
			quantize: "nvfp4",
			want:     "nvfp4",
		},
		{
			name:     "QSA query projection mxfp8",
			tensor:   "model.language_model.layers.3.self_attn.q_proj.weight",
			shape:    []int32{12288, 2560},
			quantize: "mxfp8",
			want:     "mxfp8",
		},
		{
			name:     "hyperconnection projection",
			tensor:   "model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight",
			shape:    []int32{2560, 2560},
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

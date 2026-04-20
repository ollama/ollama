package server

import (
	"testing"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

func TestLagunaGGUFQuantization(t *testing.T) {
	cases := []struct {
		name          string
		tensor        string
		originalType  fsggml.TensorType
		requestedType fsggml.TensorType
		fileType      fsggml.FileType
		blockCount    int
		wantType      fsggml.TensorType
		wantQuantize  bool
	}{
		{
			name:          "non_routed_weights_preserved",
			tensor:        "blk.1.attn_q.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ8_0,
			fileType:      fsggml.FileTypeQ8_0,
			blockCount:    2,
			wantType:      fsggml.TensorTypeBF16,
			wantQuantize:  false,
		},
		{
			name:          "shared_expert_weights_preserved",
			tensor:        "blk.1.ffn_gate_shexp.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ4_K,
			fileType:      fsggml.FileTypeQ4_K_M,
			blockCount:    2,
			wantType:      fsggml.TensorTypeBF16,
			wantQuantize:  false,
		},
		{
			name:          "routed_gate_q8",
			tensor:        "blk.1.ffn_gate_exps.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ8_0,
			fileType:      fsggml.FileTypeQ8_0,
			blockCount:    2,
			wantType:      fsggml.TensorTypeQ8_0,
			wantQuantize:  true,
		},
		{
			name:          "routed_down_q4_promoted",
			tensor:        "blk.1.ffn_down_exps.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ4_K,
			fileType:      fsggml.FileTypeQ4_K_M,
			blockCount:    2,
			wantType:      fsggml.TensorTypeQ6_K,
			wantQuantize:  true,
		},
		{
			name:          "routed_down_q4_not_promoted_when_q8_requested",
			tensor:        "blk.1.ffn_down_exps.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ8_0,
			fileType:      fsggml.FileTypeQ4_K_M,
			blockCount:    2,
			wantType:      fsggml.TensorTypeQ8_0,
			wantQuantize:  true,
		},
		{
			name:          "routed_down_q4_k_s_promoted",
			tensor:        "blk.0.ffn_down_exps.weight",
			originalType:  fsggml.TensorTypeBF16,
			requestedType: fsggml.TensorTypeQ4_K,
			fileType:      fsggml.FileTypeQ4_K_S,
			blockCount:    8,
			wantType:      fsggml.TensorTypeQ5_K,
			wantQuantize:  true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			gotType, gotQuantize := lagunaGGUFQuantization(tt.tensor, tt.originalType, tt.requestedType, tt.fileType, tt.blockCount)
			if gotType != tt.wantType || gotQuantize != tt.wantQuantize {
				t.Fatalf("lagunaGGUFQuantization(%q) = (%s, %v), want (%s, %v)", tt.tensor, gotType, gotQuantize, tt.wantType, tt.wantQuantize)
			}
		})
	}
}

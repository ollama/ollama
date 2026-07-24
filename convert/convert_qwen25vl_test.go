package convert

import (
	"slices"
	"testing"
)

func TestQwen25VLFullAttentionBlockDefaults(t *testing.T) {
	tokenizer := &Tokenizer{Vocabulary: &Vocabulary{}}

	for _, tt := range []struct {
		name   string
		blocks []int32
		want   []int32
	}{
		{
			name: "nil",
			want: []int32{7, 15, 23, 31},
		},
		{
			name:   "empty",
			blocks: []int32{},
			want:   []int32{7, 15, 23, 31},
		},
		{
			name:   "custom",
			blocks: []int32{5, 17},
			want:   []int32{5, 17},
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model := &qwen25VLModel{}
			model.VisionModel.FullAttentionBlocks = tt.blocks

			got, ok := model.KV(tokenizer)["qwen25vl.vision.fullatt_block_indexes"].([]int32)
			if !ok {
				t.Fatalf("fullatt_block_indexes has unexpected type %T", got)
			}
			if !slices.Equal(got, tt.want) {
				t.Fatalf("fullatt_block_indexes = %v, want %v", got, tt.want)
			}
		})
	}
}

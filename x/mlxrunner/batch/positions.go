package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// SequentialPositions builds per-token sequential positions for all sequences
// in the batch. Each sequence's positions start at its corresponding offset.
//
// offsets must have one entry per sequence (matching batch.SeqIDs), representing
// the starting position for that sequence's new tokens (typically the cache offset).
func SequentialPositions(b *ForwardBatch, offsets []int32) *mlx.Array {
	total := b.TotalLen()
	pos := make([]int32, 0, total)
	for i, seqLen := range b.SeqLens {
		offset := offsets[i]
		for j := range seqLen {
			pos = append(pos, offset+int32(j))
		}
	}
	return mlx.NewArrayInt32(pos, []int32{int32(total)})
}

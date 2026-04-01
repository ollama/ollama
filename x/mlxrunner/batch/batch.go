package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// ForwardBatch carries per-step metadata through the model forward pass.
// InputIDs shape is [1, N] where N = sum(SeqLens). SeqLens indicates how
// many tokens belong to each sequence.
type ForwardBatch struct {
	// InputIDs holds token IDs across all sequences. Shape: [1, N].
	InputIDs *mlx.Array

	// SeqIDs uniquely identifies each sequence in the batch.
	SeqIDs []int

	// SeqLens is the number of new tokens per sequence in this step.
	// For decode batching every entry is 1. For prefill it may vary.
	SeqLens []int
}

// TotalLen returns the total number of tokens across all sequences.
func (b *ForwardBatch) TotalLen() int {
	n := 0
	for _, l := range b.SeqLens {
		n += l
	}
	return n
}

// NumSeqs returns the number of sequences in the batch.
func (b *ForwardBatch) NumSeqs() int {
	return len(b.SeqIDs)
}

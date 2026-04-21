package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Batch is the per-forward-pass input handed to a model.
type Batch struct {
	// InputIDs is the input token IDs for this forward pass, shape (B, L).
	InputIDs *mlx.Array

	// SeqOffsets gives each row's current position within its sequence —
	// where the chunk in InputIDs starts.  Length equals the batch dimension
	// of InputIDs.
	SeqOffsets []int32
}

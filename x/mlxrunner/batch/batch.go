package batch

import "github.com/ollama/ollama/x/mlxrunner/mlx"

// Batch is the per-forward-pass input handed to a model.
type Batch struct {
	// InputIDs is the input token IDs for this forward pass, shape (B, L).
	InputIDs *mlx.Array
}

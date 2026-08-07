package base

import (
	"github.com/ollama/ollama/x/mlxrunner/batch"
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// MediaInput is one media attachment referenced by a tagged prompt.
type MediaInput struct {
	ID   int
	Kind string
	Data []byte
}

// PreparedMultimodalPrompt owns request-scoped model inputs used during
// multimodal prefill. Decode continues through Model.Forward after prefill.
type PreparedMultimodalPrompt interface {
	Tokens() []int32
	Forward(b *batch.Batch, caches []cache.Cache) *mlx.Array
	Close()
}

// MultimodalModel is optionally implemented by models that prepare media and
// inject its embeddings during prompt prefill.
//
// headroomBytes is how much memory the runner has left over beyond the resident
// model. The runner owns the budget; the model owns the cost curve, so it is
// the model's job to turn a byte count into a fidelity choice. Zero means the
// runner could not determine a budget and the model should use its cheapest
// setting rather than fail.
type MultimodalModel interface {
	PrepareMultimodalPrompt(prompt string, media []MediaInput, headroomBytes int) (PreparedMultimodalPrompt, error)
}

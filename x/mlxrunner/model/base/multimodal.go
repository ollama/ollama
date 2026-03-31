package base

import (
	"github.com/ollama/ollama/x/mlxrunner/cache"
	"github.com/ollama/ollama/x/mlxrunner/mlx"
)

// ImageInput is a single image attached to a prompt.
type ImageInput struct {
	ID   int
	Data []byte
}

// PromptTokenization contains tokenized prompt IDs plus optional request-scoped
// model metadata needed during forward.
type PromptTokenization struct {
	Tokens []int32
	State  any
}

// MultimodalPromptTokenizer is an optional model interface used by mlxrunner
// to expand tagged multimodal prompts into token IDs.
type MultimodalPromptTokenizer interface {
	TokenizePromptWithImages(prompt string, images []ImageInput) ([]int32, error)
}

// MultimodalPromptTokenizerWithState is a richer tokenizer variant that can
// return request-scoped state to be attached to the forward pass.
type MultimodalPromptTokenizerWithState interface {
	TokenizePromptWithImagesState(prompt string, images []ImageInput) (*PromptTokenization, error)
}

// ForwardWithStateModel is an optional model interface for request-scoped
// forward metadata that should not be stored in shared caches.
type ForwardWithStateModel interface {
	ForwardWithState(inputs *mlx.Array, cache []cache.Cache, state any) *mlx.Array
}
